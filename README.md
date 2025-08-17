# README 

This repository contains **two independent components** that can be used together or separately:

- **Part 1 — OCR → Field Extraction (Streamlit):** Upload a scanned form (PDF/JPG/PNG), run **Azure Document Intelligence** OCR, then use **Azure OpenAI (GPT-4o)** to extract a **strict JSON** (Hebrew/English aware) via prompts and light validation.
- **Part 2 — HMO RAG Chatbot (FastAPI + Gradio):** Build a local knowledge base from your `.html` files, create embeddings with **Azure OpenAI**, rank by **cosine similarity**, and expose a small **chat UI** for bilingual Q&A. The Gradio UI is configured to **hide the raw sources** and show only the final answer.

---

## Prerequisites

- Python **3.10+**
- Azure subscriptions for:
  - **Azure Document Intelligence** (for OCR) — a.k.a. Form Recognizer / Document Intelligence.
  - **Azure OpenAI** (for Chat Completions + Embeddings).
- Deployed models in Azure OpenAI:
  - A **chat** model deployment (e.g., `gpt-4o` or `gpt-4o-mini`).
  - An **embeddings** model deployment (e.g., `text-embedding-3-large`).

> ⚠️ The code uses **deployment names** for Azure OpenAI calls (sent as `model=...`). Make sure your env vars match your *deployment names* (not just base model names).

---

## Folder layout (suggested)

```
part1/
  phase1_data        # put your pdf files here
  app_streamlit.py
  extract_fields.py
  ocr_module.py
  pipeline.py
  prompts/
    prompt_he.txt
    prompt_en.txt
    translate_json_fields_prompt.txt  (or translate_json_fields.txt)
  uploads/             # created at runtime
 

part2/
  embed_texts.py
  html_to_text.py
  rag_index.py         # optional helper (alternative to embed_texts.py)
  server.py
  gradio_app.py
  prompts/
    collect_user_bilingual.txt
    extract_user_json_bilingual.txt
    qa_bilingual.txt
  phase2_data/         # put your .html files here
  index/               # built index JSON (created by embed_texts.py)

requirements.txt       # unified deps for both parts
 .env                 # your secrets (git-ignored)
```

---

# Quick Setup (both parts)

> You already have a **requirements.txt**, so there’s **no need to manually install individual packages**. Create/activate a virtualenv (recommended), then run:

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
# (also supported as legacy fallback: AZURE_ENDPOINT / AZURE_KEY)

# --- Azure OpenAI (Chat Completions) ---
AZURE_OPENAI_ENDPOINT=https://<your-aoai>.openai.azure.com/
AZURE_OPENAI_KEY=<your-aoai-key>
AZURE_OPENAI_API_VERSION=2024-06-01

# Deployment name of your chat model
AZURE_OPENAI_DEPLOYMENT=gpt-4o
# (modular pipeline variant also supports AZURE_OPENAI_DEPLOYMENT_GPT4O=gpt-4o)
```

### 2) Run the UI (Part 1)

```bash
cd hmp-OCR-part1
streamlit run app_streamlit.py
```

Flow:
1. Upload a **PDF/JPG/PNG**.
2. See **OCR Preview** (first lines + language hint).
3. Click **Run Extraction** → get **JSON output**.
4. **Validation** tab checks JSON structure and constraints (e.g., 9-digit ID, date parts).

### 3) How it works (Part 1)

- **OCR**: uses *prebuilt-layout* to extract lines, then concatenates to text. Two SDK flavors are included (`azure-ai-formrecognizer` and `azure-ai-documentintelligence`); either works.
- **Extraction prompts**: `prompt_he.txt` / `prompt_en.txt` tell GPT-4o to return **JSON only**, following a strict schema (dates split into day/month/year, 9-digit ID, digits-only phones, etc.).
- **Normalization step**: a second prompt (`translate_json_fields_prompt.txt` or `translate_json_fields.txt`) maps/normalizes to the **target English schema** and again enforces **JSON only**.
- **UI**: shows the raw JSON, then validates against the required schema and displays issues (if any).

---

# Part 2 — HMO RAG Chatbot (FastAPI API + Gradio UI)

### 1) Environment variables (`hmo-chatbot-part2/.env`)

```
# --- Azure OpenAI ---
AZURE_OPENAI_ENDPOINT=https://<your-aoai>.openai.azure.com/
AZURE_OPENAI_KEY=<your-aoai-key>
AZURE_OPENAI_API_VERSION=2024-06-01

# Use your exact Azure OpenAI deployment names:
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

> If your server defaults still show `text-embedding-ada-002`, set `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` explicitly to your current embeddings deployment (e.g., `text-embedding-3-large`).

### 2) Build the index (one-time or on HTML changes)

Put your `.html` files under `phase2_data/` and run:

```bash
cd hmo-chatbot-part2
python embed_texts.py
# => writes index/phase2_index.json
```

What happens:
- HTML is cleaned to **text** (scripts/styles removed, tables flattened).
- Text is **chunked** (~1100–1200 chars with ~200 overlap).
- Embeddings are created via your **Azure OpenAI** embeddings deployment.
- Vectors are **L2-normalized** and saved along with chunk metadata into a JSON index.

### 3) Start the API (FastAPI)

```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

- `GET /health` — returns status and number of chunks loaded.
- `POST /collect` — bilingual *intake* flow that gathers & validates a user profile (returns a `reply` and optional `extracted_json` when confirmed).
- `POST /qa` — performs RAG: embeds the latest user question, retrieves Top-K context with cosine similarity, and asks GPT-4o to answer **only from the context**. The response includes `{ reply, sources[] }` (the Gradio app hides these sources).

### 4) Start the UI (Gradio)

```bash
python gradio_app.py
# Opens http://127.0.0.1:7860
```

- **Mode “איסוף פרטים”** (Collect): guides the user to a validated profile and asks for confirmation.
- **Mode “שאלות ותשובות”** (Q&A): sends `/qa` requests. The UI is configured to **not display the sources**—only the final answer is shown.

### 5) How RAG ranking works

- The index builder normalizes all vectors; query vectors are normalized too.
- Ranking uses **cosine similarity** via dot-product between normalized vectors.
- Top-K chunks are formatted as a numbered context block and injected into the **QA** system prompt together with the (optional) user profile JSON.

---

## Testing & examples

**API health:**
```bash
curl http://127.0.0.1:8000/health
```

**/collect (remember the header):**
```bash
curl -X POST "http://127.0.0.1:8000/collect" \
  -H "Content-Type: application/json" -H "x-api-key: dev-key" \
  -d '{"history":[{"role":"user","content":"שלום"}],"language":"he"}'
```

**/qa:**
```bash
curl -X POST "http://127.0.0.1:8000/qa" \
  -H "Content-Type: application/json" -H "x-api-key: dev-key" \
  -d '{"history":[{"role":"user","content":"מה הזכאות לאופטיקה?"}],"language":"he","user":{},"top_k":5}'
```

---

## Troubleshooting

- **404 `DeploymentNotFound`**  
  Your `AZURE_OPENAI_*_DEPLOYMENT` must match the **deployment name** in your Azure OpenAI resource. Also verify the **endpoint region** and `AZURE_OPENAI_API_VERSION`.

- **400 `messages[x].role` invalid**  
  Each message must use a valid role: `system | user | assistant | function | tool | developer`.

- **“Index not loaded” (503)**  
  Run `python embed_texts.py` to create `index/phase2_index.json`. Check `INDEX_PATH` in your `.env`.

- **Missing OCR creds (Part 1)**  
  Set `AZURE_DOCUMENTINTELLIGENCE_ENDPOINT` and `AZURE_DOCUMENTINTELLIGENCE_KEY` (or legacy `AZURE_ENDPOINT` / `AZURE_KEY`).

- **Output not valid JSON (Part 1)**  
  The UI shows raw model output when parsing fails and lists validation issues. Re-run, or tune prompts if needed.

---

## Notes & conventions

- The UI and API are **stateless** on the server: the client sends the **entire history** and **user profile** on each call.
- In the Gradio UI we intentionally **hide `sources`** returned from `/qa`; only the `reply` text is shown.
- Prompts are **bilingual** and keep the user’s language (Hebrew/English) in answers.
- For reproducibility in production, consider pinning exact package versions (already captured in `requirements.txt`) and using a lockfile.

---


