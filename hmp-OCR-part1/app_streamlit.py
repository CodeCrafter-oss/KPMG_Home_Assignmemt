"""
app_streamlit.py
Streamlit UI for OCR → field extraction → validation.
Run: streamlit run app_streamlit.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, List

import streamlit as st
from dotenv import load_dotenv

from extract_fields import preview_ocr, run_extraction_pipeline  # type: ignore


# ---------- Prompt alias (supports your alternate filename) ----------
def ensure_prompt_aliases() -> Path:
    prompts_dir = Path("prompts")
    prompts_dir.mkdir(exist_ok=True)

    expected = prompts_dir / "translate_json_fields_prompt.txt"
    if expected.exists():
        return expected

    alt = prompts_dir / "translate_json_fields.txt"
    if alt.exists():
        shutil.copyfile(alt, expected)
    return expected


# ---------- Env helpers ----------
def env_present(keys: List[str]) -> Tuple[bool, Dict[str, bool]]:
    status = {k: bool(os.getenv(k)) for k in keys}
    return all(status.values()), status


def show_env_panel() -> None:
    st.sidebar.header("Environment")
    st.sidebar.caption("Detected environment variables and prompts.")

    di_keys = ["AZURE_DOCUMENTINTELLIGENCE_ENDPOINT", "AZURE_DOCUMENTINTELLIGENCE_KEY"]
    legacy_di_keys = ["AZURE_ENDPOINT", "AZURE_KEY"]
    oai_keys = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_API_VERSION"]

    di_ok, di_map = env_present(di_keys)
    legacy_ok, legacy_map = env_present(legacy_di_keys)
    oai_ok, oai_map = env_present(oai_keys)

    st.sidebar.subheader("Azure Document Intelligence (OCR)")
    if di_ok:
        st.sidebar.success("Using AZURE_DOCUMENTINTELLIGENCE_*")
    elif legacy_ok:
        st.sidebar.info("Using legacy AZURE_ENDPOINT/AZURE_KEY")
    else:
        st.sidebar.error(
            "Missing OCR credentials.\n"
            "Set either AZURE_DOCUMENTINTELLIGENCE_ENDPOINT & AZURE_DOCUMENTINTELLIGENCE_KEY\n"
            "or AZURE_ENDPOINT & AZURE_KEY"
        )
    with st.sidebar.expander("OCR variables"):
        st.code(json.dumps({**di_map, **legacy_map}, indent=2))

    st.sidebar.subheader("Azure OpenAI")
    if oai_ok:
        st.sidebar.success("OpenAI variables found")
    else:
        st.sidebar.error(
            "Missing: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_API_VERSION"
        )
    with st.sidebar.expander("OpenAI variables"):
        st.code(json.dumps(oai_map, indent=2))

    st.sidebar.subheader("Prompts")
    prompts_dir = Path("prompts")
    expected_translate = ensure_prompt_aliases()
    files = {
        "prompt_en.txt": (prompts_dir / "prompt_en.txt").exists(),
        "prompt_he.txt": (prompts_dir / "prompt_he.txt").exists(),
        expected_translate.name: expected_translate.exists(),
    }
    if all(files.values()):
        st.sidebar.success("All prompt files found")
    else:
        st.sidebar.warning("One or more prompt files are missing")
    with st.sidebar.expander("Prompt files present"):
        st.code(json.dumps(files, indent=2))


# ---------- Validation & language ----------
REQUIRED_SCHEMA = {
    "lastName": "",
    "firstName": "",
    "idNumber": "",
    "gender": "",
    "dateOfBirth": {"day": "", "month": "", "year": ""},
    "address": {
        "street": "",
        "houseNumber": "",
        "entrance": "",
        "apartment": "",
        "city": "",
        "postalCode": "",
        "poBox": "",
    },
    "landlinePhone": "",
    "mobilePhone": "",
    "jobType": "",
    "dateOfInjury": {"day": "", "month": "", "year": ""},
    "timeOfInjury": "",
    "accidentLocation": "",
    "accidentAddress": "",
    "accidentDescription": "",
    "injuredBodyPart": "",
    "signature": "",
    "formFillingDate": {"day": "", "month": "", "year": ""},
    "formReceiptDateAtClinic": {"day": "", "month": "", "year": ""},
    "medicalInstitutionFields": {
        "healthFundMember": "",
        "natureOfAccident": "",
        "medicalDiagnoses": "",
    },
}


def deep_keys(d: Dict[str, Any], prefix: str = "") -> List[str]:
    out: List[str] = []
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(deep_keys(v, path))
        else:
            out.append(path)
    return out


def validate_schema(payload: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    required_paths = set(deep_keys(REQUIRED_SCHEMA))
    got_paths = set(deep_keys(payload)) if isinstance(payload, dict) else set()
    missing = sorted(list(required_paths - got_paths))
    extras = sorted(list(got_paths - required_paths))
    if missing:
        issues.append(f"Missing keys/paths: {missing}")
    if extras:
        issues.append(f"Unexpected extra keys/paths: {extras}")

    if isinstance(payload, dict):
        idn = str(payload.get("idNumber", "") or "")
        if idn and (not idn.isdigit() or len(idn) != 9):
            issues.append("idNumber must be exactly 9 digits (or empty).")

        def check_date(node: Dict[str, Any], label: str) -> None:
            d = str(node.get("day", "") or "")
            m = str(node.get("month", "") or "")
            y = str(node.get("year", "") or "")
            if any([d, m, y]):
                if not (len(d) == 2 and d.isdigit()):
                    issues.append(f"{label}.day must be 2 digits (or empty).")
                if not (len(m) == 2 and m.isdigit()):
                    issues.append(f"{label}.month must be 2 digits (or empty).")
                if not (len(y) == 4 and y.isdigit()):
                    issues.append(f"{label}.year must be 4 digits (or empty).")

        for date_key in ["dateOfBirth", "dateOfInjury", "formFillingDate", "formReceiptDateAtClinic"]:
            node = payload.get(date_key, {}) if isinstance(payload.get(date_key), dict) else {}
            check_date(node, date_key)

    return issues


def detect_language_from_text(text: str) -> str:
    if not text:
        return "unknown"
    if re.search(r"[א-ת]", text):
        return "he"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "unknown"


# ---------- App ----------
def main() -> None:
    load_dotenv()
    ensure_prompt_aliases()

    st.set_page_config(page_title="Field Extraction (Part 1)", page_icon="🧾", layout="wide")
    show_env_panel()

    st.title("Part 1 — Field Extraction UI")
    st.caption("Upload a PDF/JPG/PNG. Preview OCR, extract fields, and validate the JSON.")

    uploaded = st.file_uploader(
        "Upload a PDF or image", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=False
    )

    status_box = st.empty()
    tab_ocr, tab_json, tab_validation = st.tabs(["🔎 OCR Preview", "🧷 JSON Output", "✅ Validation"])

    # State computed from upload
    file_path: str | None = None
    ocr_preview_text: str = ""
    detected_lang: str = "unknown"

    if uploaded:
        tmp_dir = Path("uploads")
        tmp_dir.mkdir(exist_ok=True)
        file_path = str(tmp_dir / uploaded.name)
        with open(file_path, "wb") as f:
            f.write(uploaded.read())

        try:
            ocr_preview_text = preview_ocr(file_path) or ""
        except Exception as e:
            ocr_preview_text = ""
            with tab_ocr:
                st.error(f"OCR preview failed: {e}")
                st.code(traceback.format_exc())

        detected_lang = detect_language_from_text(ocr_preview_text)

    # Render OCR area ONCE per run (fixed key -> no duplicates)
    with tab_ocr:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("OCR Text (preview)")
        with col2:
            st.metric(
                label="Detected language",
                value={"he": "Hebrew (he)", "en": "English (en)"}.get(detected_lang, "Unknown"),
            )
        st.text_area(
            "First lines from OCR:",
            ocr_preview_text,
            height=520,
            key="ocr_preview_area",  # unique and stable key
        )

    run_btn = st.button("Run Extraction", type="primary", disabled=not file_path)

    # Results placeholders
    with tab_json:
        json_holder = st.empty()
    with tab_validation:
        val_holder = st.empty()

    if run_btn and file_path:
        try:
            with st.spinner("Running pipeline (OCR → GPT extraction → translation/mapping)…"):
                raw = run_extraction_pipeline(file_path)

            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.startswith("json"):
                    cleaned = cleaned[len("json"):].lstrip()

            result_json: Dict[str, Any] | None = None
            try:
                result_json = json.loads(cleaned)
            except Exception:
                result_json = None

            with tab_json:
                st.subheader("JSON Output")
                if result_json is not None:
                    json_holder.json(result_json, expanded=True)
                else:
                    st.warning("Could not parse JSON. Showing raw model output below.")
                    json_holder.code(raw)

            with tab_validation:
                st.subheader("Validation")
                if result_json is None:
                    val_holder.error("Validation skipped: output was not valid JSON.")
                else:
                    issues = validate_schema(result_json)
                    if not issues:
                        val_holder.success("Output matches the required schema. ✅")
                    else:
                        val_holder.warning("Found issues with the output:")
                        for i in issues:
                            st.write(f"- {i}")

            status_box.success("Pipeline completed.")

        except Exception as e:
            status_box.error(f"Pipeline failed: {e}")
            with tab_json:
                st.subheader("Error details")
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
