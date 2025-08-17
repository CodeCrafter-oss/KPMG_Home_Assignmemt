from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langdetect import detect
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

def _load_env_once() -> None:
    load_dotenv()

def _get_env_or_raise(name: str, alt: str | None = None) -> str:
    val = os.getenv(name)
    if not val and alt:
        val = os.getenv(alt)
    if not val:
        if alt:
            raise RuntimeError(f"Missing environment variable: {name} (or {alt}). Set it in your .env")
        raise RuntimeError(f"Missing environment variable: {name}. Set it in your .env")
    return val

def _build_di_client() -> DocumentAnalysisClient:
    _load_env_once()
    endpoint = os.getenv("AZURE_DOCUMENTINTELLIGENCE_ENDPOINT") or os.getenv("AZURE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENTINTELLIGENCE_KEY") or os.getenv("AZURE_KEY")
    if not endpoint or not key:
        raise RuntimeError(
            "Missing Document Intelligence credentials. Set AZURE_DOCUMENTINTELLIGENCE_ENDPOINT/AZURE_DOCUMENTINTELLIGENCE_KEY "
            "(or AZURE_ENDPOINT/AZURE_KEY) in your .env"
        )
    return DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def _build_aoai_client() -> AzureOpenAI:
    _load_env_once()
    endpoint = _get_env_or_raise("AZURE_OPENAI_ENDPOINT")
    api_key = _get_env_or_raise("AZURE_OPENAI_KEY")
    api_version = _get_env_or_raise("AZURE_OPENAI_API_VERSION")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

def _get_deployment_name() -> str:
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

def _get_layout_result(file_path: str):
    client = _build_di_client()
    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document(model_id="prebuilt-layout", document=f)
    return poller.result()

def _extract_text_from_layout(result, max_chars: Optional[int] = None) -> str:
    lines = []
    for page in getattr(result, "pages", []):
        for line in getattr(page, "lines", []):
            if line and getattr(line, "content", None):
                lines.append(line.content)
    text = "\n".join(lines)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars]
    return text

def preview_ocr(file_path: str) -> Optional[str]:
    try:
        result = _get_layout_result(file_path)
        return _extract_text_from_layout(result, max_chars=1500)
    except Exception:
        return None

def _detect_language(text: str) -> str:
    try:
        lang = detect(text or "")
        return "he" if lang == "he" else "en"
    except Exception:
        return "en"

def _read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path.as_posix()}")
    return path.read_text(encoding="utf-8")

def _load_extraction_prompt(language: str, full_text: str) -> str:
    prompts_dir = Path("prompts")
    prompt_name = "prompt_he.txt" if language == "he" else "prompt_en.txt"
    prompt_path = prompts_dir / prompt_name
    base = _read_text_file(prompt_path)
    return f"{base}\n\n\"\"\"\n{full_text}\n\"\"\""

def _load_translate_prompt() -> str:
    prompts_dir = Path("prompts")
    primary = prompts_dir / "translate_json_fields_prompt.txt"
    alias = prompts_dir / "translate_json_fields.txt"
    if primary.exists():
        return _read_text_file(primary)
    if alias.exists():
        return _read_text_file(alias)
    raise FileNotFoundError(
        "Prompt file not found: prompts/translate_json_fields_prompt.txt (or prompts/translate_json_fields.txt)"
    )

def _strip_code_fences(s: str) -> str:
    txt = s.strip()
    if txt.startswith("```"):
        txt = txt.lstrip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:].lstrip()
        if txt.endswith("```"):
            txt = txt[:-3].rstrip()
    return txt

def _chat_complete(prompt: str) -> str:
    client = _build_aoai_client()
    deployment = _get_deployment_name()
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a precise information extraction assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    return content or ""

def run_extraction_pipeline(file_path: str) -> str:
    try:
        layout = _get_layout_result(file_path)
    except Exception as e:
        raise RuntimeError(f"OCR failed: {e}")
    full_text = _extract_text_from_layout(layout)
    language = _detect_language(full_text)
    extraction_prompt = _load_extraction_prompt(language, full_text)
    extracted = _chat_complete(extraction_prompt)
    extracted = _strip_code_fences(extracted)
    translate_prompt = _load_translate_prompt()
    normalized = _chat_complete(f"{translate_prompt}\n\n{extracted}")
    normalized = _strip_code_fences(normalized)
    try:
        parsed = json.loads(normalized)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        return normalized
