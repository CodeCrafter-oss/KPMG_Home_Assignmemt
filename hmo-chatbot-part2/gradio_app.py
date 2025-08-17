from __future__ import annotations
import os, json, requests, re, io
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "dev-key")

# ---------- Helpers ----------
def _read_text(path: str) -> str:
    # קורא UTF-8 (גם עם BOM), נוח לשמירה מוורד/נוטפד++
    with io.open(path, "r", encoding="utf-8-sig") as f:
        return f.read()

def _resolve_prompt(env_key: str, default_file: str, fallback_text: str) -> str:
    """
    - אם יש ENV עם נתיב – נשתמש בו
    - אחרת ננסה קובץ ליד הסקריפט
    - אחרת ננסה ./prompts/<file>
    - אחרת נשתמש בטקסט ברירת מחדל
    """
    p = os.getenv(env_key)
    if p and os.path.exists(p):
        return _read_text(p)

    here = os.path.dirname(os.path.abspath(__file__))
    cand1 = os.path.join(here, default_file)
    cand2 = os.path.join(here, "prompts", default_file)
    for c in (cand1, cand2):
        if os.path.exists(c):
            return _read_text(c)
    return fallback_text

def _post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}",
                      headers={"x-api-key": API_KEY},
                      json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def _append(history, role, content):
    history = history or []
    history.append({"role": role, "content": content})
    return history

def _loads_or(default_obj, s: str | None):
    if not s:
        return default_obj
    try:
        return json.loads(s)
    except Exception:
        return default_obj

def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)

def parse_json_from_reply(text: str):
    if not text:
        return None
    m = re.search(r"<<<JSON>>>\s*({.*?})\s*<<<END>>>", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    stripped = text.strip().strip("`").strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except Exception:
            pass
    return None

# ---------- Load prompts from files (with sane fallbacks) ----------
COLLECT_PROMPT_DEFAULT = "Collect user profile step by step."
EXTRACT_PROMPT_DEFAULT = "Return ONLY JSON or {'status':'incomplete'}."

COLLECT_PROMPT = _resolve_prompt(
    "PROMPT_COLLECT_PATH",
    "collect_user_bilingual.txt",
    COLLECT_PROMPT_DEFAULT,
)
EXTRACT_PROMPT = _resolve_prompt(
    "PROMPT_EXTRACT_PATH",
    "extract_user_json_bilingual.txt",  # שם מתוקן
    EXTRACT_PROMPT_DEFAULT,
)
QA_PROMPT = _resolve_prompt(
    "PROMPT_QA_PATH",
    "qa_bilingual.txt",
    "Answer from provided snippets. If not covered, say you cannot find it.",
)

with gr.Blocks(theme=gr.themes.Soft(), title="צ'ט רפואי – קופות חולים") as demo:
    gr.Markdown(
        "## 🇮🇱 צ'טבוט רפואי (שלב 2)\n"
        "איסוף פרטים מאומת ואישור סופי; לאחר מכן שאלות ותשובות על בסיס RAG."
    )

    with gr.Row():
        phase = gr.Radio(["איסוף פרטים", "שאלות ותשובות"], value="איסוף פרטים", label="מצב")
        lang = gr.Radio(["he", "en"], value="he", label="שפה")

    chat = gr.Chatbot(height=420, type="messages", label="שיחה", rtl=True)

    # נשמור state כמחרוזות JSON כדי לעקוף באגי schema
    user_state_json = gr.State(value="{}")   # פרופיל המשתמש הסופי (כפי שנחצב)
    hist_state_json = gr.State(value="[]")   # היסטוריית הודעות

    # כפתור הצגת הפרופיל
    btn_show_profile = gr.Button("הצג פרופיל (JSON)", variant="secondary")

    def kickoff(phase, lang, chat_value, hist_json, user_json):
        hist = _loads_or([], hist_json)
        user = _loads_or({}, user_json)

        if phase == "איסוף פרטים":
            sys = "התחל באיסוף פרטים. אמור 'שלום' ושאל שאלה ראשונה."
        else:
            sys = "מצב שאלות ותשובות. שאל אותי שאלה לגבי השירותים."

        hist = _append(hist, "assistant", sys)
        chat_value = (chat_value or [])
        chat_value.append({"role": "assistant", "content": sys})
        return chat_value, _dumps(hist), _dumps(user)

    def user_send(message, phase, lang, chat_value, hist_json, user_json):
        chat_value = chat_value or []
        hist = _loads_or([], hist_json)
        user = _loads_or({}, user_json)

        hist = _append(hist, "user", message)
        chat_value.append({"role": "user", "content": message})

        if phase == "איסוף פרטים":
            payload = {
                "history": hist,
                "language": lang,
                "system_prompt": COLLECT_PROMPT,
                "extract_prompt": EXTRACT_PROMPT,
            }
            resp = _post("/collect", payload)
            reply = resp.get("reply", "")

            chat_value.append({"role": "assistant", "content": reply})
            hist = _append(hist, "assistant", reply)

            # נעדכן את הפרופיל מתוך השרת או מתוך תשובת המודל
            maybe = resp.get("extracted_json") or parse_json_from_reply(reply)
            if isinstance(maybe, dict):
                user = maybe

            return chat_value, _dumps(hist), _dumps(user), gr.update(value="")

        else:
            payload = {
                "history": hist,
                "language": lang,
                "user": user,
                "qa_prompt": QA_PROMPT,   # אופציונלי: השרת יכול להשתמש בזה
            }
            resp = _post("/qa", payload)
            reply = resp.get("reply", "")

            # *** שינוי מרכזי: לא מציגים מקורות כלל ***
            full = reply

            chat_value.append({"role": "assistant", "content": full})
            # בהיסטוריה נשמור רק את התשובה (בלי מקורות)
            hist = _append(hist, "assistant", reply)

            return chat_value, _dumps(hist), _dumps(user), gr.update(value="")

    def show_profile(user_json: str, chat_value):
        user = _loads_or({}, user_json)
        pretty = json.dumps(user, ensure_ascii=False, indent=2)
        chat_value = (chat_value or [])
        chat_value.append({"role": "assistant", "content": f"```json\n{pretty}\n```"})
        return chat_value

    with gr.Row():
        btn_start = gr.Button("התחל", variant="primary")
        btn_clear = gr.Button("נקה", variant="secondary")

    msg = gr.Textbox(
        label="הודעה",
        placeholder="כתבי/כתוב הודעה ושלחי/שלח…",
        scale=6,
        rtl=True,
    )
    btn = gr.Button("שלח", variant="primary", scale=1)

    # חיבורים
    btn_start.click(kickoff,
                    [phase, lang, chat, hist_state_json, user_state_json],
                    [chat, hist_state_json, user_state_json])

    btn_clear.click(lambda: ([], "[]", "{}"),
                    None, [chat, hist_state_json, user_state_json],
                    queue=False)

    btn.click(user_send,
              [msg, phase, lang, chat, hist_state_json, user_state_json],
              [chat, hist_state_json, user_state_json, msg])

    msg.submit(user_send,
               [msg, phase, lang, chat, hist_state_json, user_state_json],
               [chat, hist_state_json, user_state_json, msg])

    btn_show_profile.click(show_profile, [user_state_json, chat], [chat])

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_PORT", "7860"))
    share_env = os.getenv("GRADIO_SHARE", "").lower() in {"1", "true", "yes"}
    demo.launch(server_name=server_name, server_port=server_port, share=share_env, show_error=True)
