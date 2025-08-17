"""
OCR Module
----------
- Extracts text from PDF/JPG using Azure Document Intelligence (prebuilt-layout).
- Detects language (Hebrew/English) using langdetect.
"""

import os
import logging
from dotenv import load_dotenv
from langdetect import detect
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load credentials
DI_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
DI_KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")

client = DocumentIntelligenceClient(DI_ENDPOINT, AzureKeyCredential(DI_KEY))


def detect_language(text: str) -> str:
    """Detect Hebrew ('he') or English ('en') based on text snippet."""
    try:
        lang = detect(text)
        logger.info(f"Detected language: {lang}")
        return lang
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "en"


def extract_text(file_path: str, max_preview_words: int = 100):
    """Run OCR on file and detect language."""
    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-layout", f)
        result = poller.result()

    # Concatenate lines
    all_lines = [line.content for page in result.pages for line in page.lines]
    full_text = "\n".join(all_lines)

    # Detect language from preview
    preview = " ".join(full_text.split()[:max_preview_words])
    language = detect_language(preview)

    return full_text, language
