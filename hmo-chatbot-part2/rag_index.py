# rag_utils.py
from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from bs4 import BeautifulSoup
from openai import AzureOpenAI

# ---------- Azure OpenAI ----------
def _oai_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    if not (endpoint and key):
        raise RuntimeError("Missing Azure OpenAI credentials in env (.env)")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=api_version)

def embed_texts(texts: List[str]) -> List[List[float]]:
    dep = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-3-large")
    client = _oai_client()
    resp = client.embeddings.create(model=dep, input=texts)
    return [d.embedding for d in resp.data]

# ---------- HTML → text ----------
def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n".join(rows) + "\n")

    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)

# ---------- Chunking ----------
def chunk_text(text: str, max_chars: int = 1100, overlap: int = 200) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks

# ---------- Build index ----------
def build_index_from_dir(data_dir: str, index_path: str) -> Dict[str, Any]:
    pdir = Path(data_dir)
    files = sorted(list(pdir.glob("*.html")))
    if not files:
        raise RuntimeError(f"No .html files found under {data_dir}")

    all_chunks: List[str] = []
    metas: List[Dict[str, Any]] = []

    for f in files:
        html = f.read_text(encoding="utf-8", errors="ignore")
        txt = html_to_text(html)
        chunks = chunk_text(txt)
        for i, ch in enumerate(chunks):
            metas.append({"source": str(f.name), "chunk_id": i})
            all_chunks.append(ch)

    vectors = embed_texts(all_chunks)
    arr = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
    arr = arr / norms

    payload = {"vectors": arr.tolist(), "metas": metas, "chunks": all_chunks}
    p = Path(index_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload

# ---------- Load index ----------
def load_index(index_path: str) -> Dict[str, Any]:
    p = Path(index_path)
    if not p.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    return json.loads(p.read_text(encoding="utf-8"))

# ---------- Search (cosine on normalized vectors) ----------
def search(index: Dict[str, Any], query: str, k: int = 5) -> List[Dict[str, Any]]:
    vecs = np.array(index["vectors"], dtype="float32")
    q = np.array(embed_texts([query])[0], dtype="float32")
    q = q / (np.linalg.norm(q) + 1e-10)
    sims = (vecs @ q).tolist()
    top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]

    out = []
    for i in top_idx:
        out.append({
            "score": float(sims[i]),
            "text": index["chunks"][i],
            "source": index["metas"][i]["source"],
            "chunk_id": index["metas"][i]["chunk_id"],
        })
    return out
