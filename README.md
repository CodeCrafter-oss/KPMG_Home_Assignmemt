# README

This repository contains **two components** that can be used together or separately:

* **Part 1 — OCR → Field Extraction (Streamlit):** Upload a scanned form (PDF/JPG/PNG), run **Azure Document Intelligence** OCR, then use **Azure OpenAI (GPT-4o)** to extract a **strict JSON** (Hebrew/English aware) via prompts and light validation.
* **Part 2 — HMO RAG Chatbot (FastAPI + Gradio):** Build a local knowledge base from your `.html` files, create embeddings with **Azure OpenAI**, rank by **cosine similarity**, and expose a small **chat UI** for bilingual Q\&A. The Gradio UI is configured to **hide the raw sources** and show only the final answer.

---

## Prerequisites

* Python **3.10+**
* Azure subscriptions for:

  * **Azure Document Intelligence** (OCR).
  * **Azure OpenAI** (Chat Completions + Embeddings).
* Deployed models in Azure OpenAI:

  * A **chat** model deployment (e.g., `gpt-4o`).
  * An **embeddings** model deployment (e.g., `text-embedding-3-large`).

> ⚠️ The code uses **deployment names** for Azure OpenAI calls (sent as `model=...`). Make sure your env vars match your *deployment names* (not just base model names).

---

## Folder layout

```
hmp-OCR-part1/
  phase1_data/       # put your PDF/JPG/PNG here
  app_streamlit.py
  extract_fields.py
  modules/
    ocr_module.py
    pipeline.py
  prompts/
    prompt_he.txt
    prompt_en.txt
    translate_json_fields_prompt.txt  (or translate_json_fields.txt)
  uploads/           # created at runtime
  .env               # DI + AOAI creds for Part 1

hmo-chatbot-part2/
  embed_texts.py
  html_to_text.py
  rag_index.py       # optional helper (alternative to embed_texts.py)
  server.py
  gradio_app.py      # configured to hide sources in chat
  prompts/
    collect_user_bilingual.txt
    extract_user_json_bilingual.txt
    qa_bilingual.txt
  phase2_data/       # put your .html files here
  index/             # built index JSON (created by embed_texts.py)
  .env               # AOAI chat/embeddings + API_KEY for Part 2

requirements.txt     # unified deps for both parts
```

---

# Quick Setup (both parts)

You already have a **requirements.txt**, so there’s **no need to install packages one-by-one**:

```bash
python -m venv .venv
# Windows PowerShell:
#   .\.venv\Scripts\Activate.ps1
# macOS/Linux:
#   source .venv/bin/activate

pip install -r requirements.txt
```

This installs all dependencies for **both** Part 1 and Part 2.

---

# Part 1 — OCR → Field Extraction (Streamlit)

### 1) Environment variables (`hmp-OCR-part1/.env`)

```
# --- Azure Document Intelligence (OCR) ---
AZURE_DOCUMENTINTELLIGENCE_ENDPOINT=https://<your-di>.cognitiveservices.azure.com/
AZURE_DOCUMENTINTELLIGENCE_KEY=<your-di-key>
# (legacy fallback also supported: AZURE_ENDPOINT / AZURE_KEY)

# --- Azure OpenAI (Chat Completions) ---
AZURE_OPENAI_ENDPOINT=https://<your-aoai>.openai.azure.com/
AZURE_OPENAI_KEY=<your-aoai-key>
AZURE_OPENAI_API_VERSION=2024-06-01

# Chat deployment name
AZURE_OPENAI_DEPLOYMENT=gpt-4o
# (modular pipeline also accepts AZURE_OPENAI_DEPLOYMENT_GPT4O=gpt-4o)
```

### 2) Run the UI (Part 1)

```bash
cd hmp-OCR-part1
streamlit run app_streamlit.py
```

**Flow:**

1. Upload a **PDF/JPG/PNG** → 2) **OCR Preview** (first lines + language hint) →
2. **Run Extraction** → JSON output → 4) **Validation** of schema (IDs/dates/phones).

### 3) How it works (Part 1)

* **OCR**: DI *prebuilt-layout* extracts lines → concatenated text. Two SDK flavors provided (`azure-ai-formrecognizer` and `azure-ai-documentintelligence`).
* **Extraction prompts**: `prompt_he.txt` / `prompt_en.txt` force **JSON-only** (dates split to day/month/year, 9-digit ID, digits-only phones, etc.).
* **Normalization**: a second prompt (`translate_json_fields_prompt.txt` or `translate_json_fields.txt`) maps/cleans to a **fixed English schema**.
* **UI**: shows raw JSON + validation issues when parsing fails.

---

# Part 2 — HMO RAG Chatbot (FastAPI API + Gradio UI)

### 1) Environment variables (`hmo-chatbot-part2/.env`)

```
# --- Azure OpenAI ---
AZURE_OPENAI_ENDPOINT=https://<your-aoai>.openai.azure.com/
AZURE_OPENAI_KEY=<your-aoai-key>
AZURE_OPENAI_API_VERSION=2024-06-01

# Exact deployment names:
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-large

# RAG data & index
DATA_DIR=phase2_data
INDEX_PATH=index/phase2_index.json

# App auth & URLs
API_KEY=dev-key
API_URL=http://127.0.0.1:8000

# Gradio
GRADIO_SERVER=127.0.0.1
GRADIO_PORT=7860
GRADIO_SHARE=false
```

> If your server defaults still show `text-embedding-ada-002`, set `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` explicitly to your active embeddings deployment.

### 2) Build the index (one-time or when HTML changes)

Put your `.html` files under `phase2_data/`, then:

```bash
cd hmo-chatbot-part2
python embed_texts.py
# => writes index/phase2_index.json
```

What happens:

* HTML → **clean text** (scripts/styles removed, tables flattened).
* Text **chunked** (\~1100–1200 chars, **200 overlap**).
* Embeddings via your Azure OpenAI embeddings deployment.
* Vectors **L2-normalized** and saved with chunk metadata into a JSON index.

### 3) Start the API (FastAPI)

```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Endpoints:

* `GET /health` — status + number of chunks loaded.
* `POST /collect` — bilingual intake; returns `reply` and optional `extracted_json`.
* `POST /qa` — RAG answer: returns `{ reply, sources[] }` (UI hides sources).

### 4) Start the UI (Gradio)

```bash
python gradio_app.py
# Opens http://127.0.0.1:7860
```

* **“איסוף פרטים”**: guides to a validated profile and asks for confirmation.
* **“שאלות ותשובות”**: sends `/qa` requests; **only the answer** is shown.

---

## How RAG works in Part 2 (based on `rag_utils.py`)

1. **HTML → Text**

   * Strips `<script>/<style>/<noscript>`.
   * Flattens tables to pipe-separated rows.
   * Normalizes whitespace and joins non-empty lines.

2. **Chunking**

   * Splits long text into windows of \~**1100 chars** with **200-char overlap**, preserving context across chunks.

3. **Embeddings**

   * Calls Azure OpenAI **embeddings** (deployment from `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT`) on every chunk.
   * Creates a matrix `vectors` and **L2-normalizes** each row.

4. **Index build & save**

   * Saves `{ vectors, metas, chunks }` as JSON to `INDEX_PATH`.
   * `metas` keep `{ source: <filename>, chunk_id: <ordinal> }`.

5. **Search (online)**

   * Embeds the **query**, **L2-normalizes** it, and computes `sims = vectors @ query` (dot-product).
   * Because vectors are unit-length, this equals **cosine similarity**.
   * Returns the **Top-K** chunks with `{ score, text, source, chunk_id }`.

6. **Answering**

   * The API formats Top-K chunks into a numbered CONTEXT block, adds the user profile (if any), and asks GPT-4o to answer **only from the provided context**.

This is a **semantic information-retrieval** pipeline: instead of matching keywords, it retrieves by **meaning** using embeddings and cosine similarity. It excels when users phrase questions differently from the documents (synonyms, paraphrases, typos, bilingual queries). For exact keyword scenarios you can add a **lexical IR** layer (e.g., BM25) and fuse results (RRF/MMR) for a **hybrid IR** setup.

---

## Testing & examples

**API health**

```bash
curl http://127.0.0.1:8000/health
```

**/collect (remember `x-api-key`)**

```bash
curl -X POST "http://127.0.0.1:8000/collect" \
  -H "Content-Type: application/json" -H "x-api-key: dev-key" \
  -d '{"history":[{"role":"user","content":"שלום"}],"language":"he"}'
```

**/qa**

```bash
curl -X POST "http://127.0.0.1:8000/qa" \
  -H "Content-Type: application/json" -H "x-api-key: dev-key" \
  -d '{"history":[{"role":"user","content":"מה הזכאות לאופטיקה?"}],"language":"he","user":{},"top_k":5}'
```

---

## Troubleshooting

* **404 `DeploymentNotFound`** — `AZURE_OPENAI_*_DEPLOYMENT` must equal your **deployment names**; also verify endpoint region and `AZURE_OPENAI_API_VERSION`.
* **400 `messages[x].role` invalid** — Roles must be one of: `system | user | assistant | function | tool | developer`.
* **“Index not loaded” (503)** — Run `python embed_texts.py` to create `index/phase2_index.json`. Check `INDEX_PATH` in `.env`.
* **Missing OCR creds (Part 1)** — Set `AZURE_DOCUMENTINTELLIGENCE_ENDPOINT` and `AZURE_DOCUMENTINTELLIGENCE_KEY` (or legacy `AZURE_ENDPOINT` / `AZURE_KEY`).
* **Output not valid JSON (Part 1)** — The UI shows raw model output and validation issues; re-run or tune prompts.

---



