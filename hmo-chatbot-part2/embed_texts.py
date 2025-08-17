# embed_texts.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI
from html_to_text import html_to_text

def _client() -> AzureOpenAI:
    load_dotenv()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    if not endpoint or not key:
        raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_KEY")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=api_version)

def _embed(texts: List[str]) -> List[List[float]]:
    dep = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-3-large")
    cli = _client()
    resp = cli.embeddings.create(model=dep, input=texts)
    return [d.embedding for d in resp.data]

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end == len(text): break
        start = max(0, end - overlap)
    return chunks

def build_index_from_dir(data_dir: str, index_path: str) -> Dict[str, Any]:
    pdir = Path(data_dir)
    files = sorted(pdir.glob("*.html"))
    if not files:
        raise RuntimeError(f"No .html files found in {data_dir}")

    all_chunks: List[str] = []
    metas: List[Dict[str, Any]] = []

    for f in files:
        html = f.read_text(encoding="utf-8", errors="ignore")
        txt = html_to_text(html)
        for i, ch in enumerate(chunk_text(txt)):
            all_chunks.append(ch)
            metas.append({"source": f.name, "chunk_id": i})

    vecs = _embed(all_chunks)
    arr = np.array(vecs, dtype="float32")
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10)

    payload = {"vectors": arr.tolist(), "chunks": all_chunks, "metas": metas}
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(index_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload

if __name__ == "__main__":
    load_dotenv()
    data_dir = os.getenv("DATA_DIR", "phase2_data")
    index_path = os.getenv("INDEX_PATH", "index/phase2_index.json")
    build_index_from_dir(data_dir, index_path)
    print(f"Index written to {index_path}")
