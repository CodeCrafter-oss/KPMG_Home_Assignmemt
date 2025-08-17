"""
Pipeline Module
---------------
- Runs OCR (extract text + language detection).
- Loads the appropriate prompt (EN/HE).
- Sends OCR text + prompt to GPT-4o to extract fields.
- Maps fields to English schema using translate prompt.
"""

import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from modules.ocr_module import extract_text

load_dotenv()

# ---- Azure OpenAI Config ----
OA_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OA_KEY = os.getenv("AZURE_OPENAI_KEY")
OA_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4O")

client = AzureOpenAI(
    api_key=OA_KEY,
    azure_endpoint=OA_ENDPOINT,
    api_version=OA_VERSION
)


def load_prompt(language: str, ocr_text: str) -> str:
    """Load base prompt (EN/HE) and append OCR text."""
    prompt_file = "prompts/prompt_he.txt" if language == "he" else "prompts/prompt_en.txt"
    with open(prompt_file, encoding="utf-8") as f:
        base_prompt = f.read()
    return base_prompt + f'\n\n"""\n{ocr_text}\n"""'


def call_gpt(prompt: str) -> str:
    """Call Azure OpenAI GPT-4o with given prompt."""
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful form-extraction assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content


def clean_json_output(output: str) -> dict:
    """Remove Markdown fences and parse JSON."""
    if output.startswith("```json"):
        output = output.removeprefix("```json").removesuffix("```").strip()
    elif output.startswith("```"):
        output = output.removeprefix("```").removesuffix("```").strip()
    return json.loads(output)


def run_pipeline(file_path: str) -> dict:
    """Full pipeline: OCR -> GPT extraction -> Translate fields -> Clean JSON."""
    full_text, language = extract_text(file_path)

    extraction_prompt = load_prompt(language, full_text)
    extracted_json = call_gpt(extraction_prompt)

    with open("prompts/translate_json_fields_prompt.txt", encoding="utf-8") as f:
        translate_prompt = f.read()
    translate_full_prompt = translate_prompt + f"\n\n{extracted_json}"
    translated_json = call_gpt(translate_full_prompt)

    return clean_json_output(translated_json)
