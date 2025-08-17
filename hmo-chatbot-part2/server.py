# server.py
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AzureOpenAI

# imports בראש הקובץ (אם חסר):
from typing import Optional
from fastapi import Header, HTTPException

API_KEY = os.getenv("API_KEY", "dev-key")
ALLOW_EMPTY_API_KEY = os.getenv("ALLOW_EMPTY_API_KEY", "false").lower() == "true"

def require_key(x_api_key: Optional[str] = Header(None)):
    """
    Dev-friendly auth:
    - אם לא נשלח header וב-ENV מוגדר ALLOW_EMPTY_API_KEY=true => נשתמש ב-API_KEY מה-ENV.
    - אם API_KEY לא מוגדר בכלל (מחרוזת ריקה) => לא נבצע אימות (פיתוח בלבד).
    - אחרת, חייב להיות התאמה מלאה בין ההדר לערך שב-ENV.
    """
    # בלי API_KEY ב-ENV? מאשרים (פיתוח בלבד)
    if not API_KEY:
        return

    # חסר header? אם מותר – נשתמש בברירת המחדל
    effective = x_api_key or (API_KEY if ALLOW_EMPTY_API_KEY else None)
    if effective != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

# -------------------- environment & logging --------------------
load_dotenv()

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "chatbot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("chatbot.server")

API_KEY = os.getenv("API_KEY", "dev-key")
INDEX_PATH = os.getenv("INDEX_PATH", "index/phase2_index.json")
CHAT_DEP = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
EMB_DEP = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002")

PROMPTS_DIR = Path("prompts")


# -------------------- helpers --------------------
def require_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def _client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_KEY")
    ver = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    if not endpoint or not key:
        raise RuntimeError("Missing Azure OpenAI environment variables")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=ver)


def _read_prompt(path: Path, default_text: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        logger.warning("Prompt missing at %s — using built-in default.", path)
        return default_text


# -------------------- prompts (with safe fallbacks) --------------------
COLLECT_PROMPT = _read_prompt(
    PROMPTS_DIR / "collect_user_bilingual.txt",
    """You are a helpful assistant collecting user details in Hebrew or English (match the user's language).
Collect (one or two questions per turn) and validate:
- first & last name
- 9-digit ID number
- gender
- age (0–120)
- HMO (מכבי | מאוחדת | כללית)
- 9-digit HMO card number
- tier (זהב | כסף | ארד)

After all fields are gathered, show a compact summary and ask for confirmation.
When the user confirms, output:

<<<JSON>>>
{"firstName":"","lastName":"","idNumber":"","gender":"","age":0,"hmo":"","hmoCard":"","tier":""}
<<<END>>>
""",
)

# Not strictly required here, but kept for completeness if you later add an /extract route.
EXTRACT_PROMPT = _read_prompt(
    PROMPTS_DIR / "extract_user_json_bilingual.txt",
    """Extract a structured JSON of the user profile from the conversation so far.

Return only:
{"firstName":"","lastName":"","idNumber":"","gender":"","age":0,"hmo":"","hmoCard":"","tier":""}
""",
)

QA_PROMPT = _read_prompt(
    PROMPTS_DIR / "qa_bilingual.txt",
    """You answer questions about Israeli HMOs (מכבי, מאוחדת, כללית) using the provided context.
- Answer in the user's language.
- Tailor the answer to the user's HMO and tier.
- If the answer is not in CONTEXT, say you don't know.

USER PROFILE:
{{USER_PROFILE}}

CONTEXT:
{{CONTEXT}}

Answer clearly and concisely. Cite with [1], [2]... where relevant based on the numbered context blocks.
""",
)


# -------------------- index & RAG search --------------------
def _load_index(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Index file not found: {path}. Run embed_texts.py first.")
    data = json.loads(p.read_text(encoding="utf-8"))
    vecs = np.array(data["vectors"], dtype="float32")
    # Ensure normalized (cosine)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    data["_vecs"] = vecs / norms
    return data


try:
    INDEX = _load_index(INDEX_PATH)
    logger.info("Loaded index with %d chunks from %s", len(INDEX["chunks"]), INDEX_PATH)
except Exception as e:
    logger.error("Failed to load index: %s", e)
    INDEX = None


def _embed_query(q: str) -> np.ndarray:
    cli = _client()
    r = cli.embeddings.create(model=EMB_DEP, input=[q])
    v = np.array(r.data[0].embedding, dtype="float32")
    v /= (np.linalg.norm(v) + 1e-10)
    return v


def rag_search(q: str, k: int = 5) -> List[Dict[str, Any]]:
    if INDEX is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    vecs = INDEX["_vecs"]
    qv = _embed_query(q)
    sims = (vecs @ qv).tolist()
    top = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    out = []
    for i in top:
        out.append(
            {
                "score": float(sims[i]),
                "text": INDEX["chunks"][i],
                "source": INDEX["metas"][i]["source"],
                "chunk_id": INDEX["metas"][i]["chunk_id"],
            }
        )
    return out


def _chat(messages: List[Dict[str, str]], max_tokens: int = 800, temperature: float = 0.2) -> str:
    cli = _client()
    r = cli.chat.completions.create(
        model=CHAT_DEP,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content


# -------------------- schemas --------------------
class Message(BaseModel):
    role: str
    content: str


class UserProfile(BaseModel):
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    idNumber: Optional[str] = None  # 9 digits
    gender: Optional[str] = None
    age: Optional[int] = None
    hmo: Optional[str] = None  # מכבי | מאוחדת | כללית
    hmoCard: Optional[str] = None  # 9-digit
    tier: Optional[str] = None  # זהב | כסף | ארד


class CollectRequest(BaseModel):
    history: List[Message]
    language: str = Field("he", description="he|en")


class CollectResponse(BaseModel):
    reply: str
    extracted_json: Optional[Dict[str, Any]] = None


class QARequest(BaseModel):
    history: List[Message]
    user: UserProfile
    language: str = Field("he", description="he|en")
    top_k: int = 5


class QAResponse(BaseModel):
    reply: str
    sources: List[Dict[str, Any]]


# -------------------- app --------------------
app = FastAPI(title="Medical RAG Chatbot", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    if INDEX is None:
        return {"status": "degraded", "index": 0}
    return {"status": "ok", "index": len(INDEX["chunks"])}


@app.post("/collect", dependencies=[Depends(require_key)], response_model=CollectResponse)
def collect(req: CollectRequest):
    system = COLLECT_PROMPT
    msgs = [{"role": "system", "content": system}] + [m.model_dump() for m in req.history]
    reply = _chat(msgs, max_tokens=700)

    extracted_json = None
    if "<<<JSON>>>" in reply and "<<<END>>>" in reply:
        try:
            blob = reply.split("<<<JSON>>>", 1)[1].split("<<<END>>>", 1)[0].strip()
            extracted_json = json.loads(blob)
        except Exception as e:
            logger.warning("Failed to parse collected JSON: %s", e)

    return CollectResponse(reply=reply, extracted_json=extracted_json)


@app.post("/qa", dependencies=[Depends(require_key)], response_model=QAResponse)
def qa(req: QARequest):
    if INDEX is None:
        raise HTTPException(status_code=503, detail="Index not loaded")

    user_txt = json.dumps(req.user.model_dump(exclude_none=True), ensure_ascii=False)
    last_user_msg = next((m.content for m in reversed(req.history) if m.role == "user"), "")
    retrieved = rag_search(last_user_msg, k=req.top_k)

    context = "\n\n".join(
        [f"[{i+1}] ({r['source']} • score={r['score']:.3f})\n{r['text']}" for i, r in enumerate(retrieved)]
    )
    sys_prompt = QA_PROMPT.replace("{{USER_PROFILE}}", user_txt).replace("{{CONTEXT}}", context)
    msgs = [{"role": "system", "content": sys_prompt}] + [m.model_dump() for m in req.history]

    answer = _chat(msgs, max_tokens=800)
    return QAResponse(reply=answer, sources=retrieved)
