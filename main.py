# main.py - JonBet IA v2 (FastAPI + OpenAI + SQLite)
import os
import json
import time
import logging
import sqlite3
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
import httpx
from dotenv import load_dotenv

# OpenAI client (newer SDK)
try:
    from openai import OpenAI
except Exception:
    # fallback to openai package if installed differently
    import openai as _old_openai
    class OpenAI:
        def __init__(self, api_key=None): 
            _old_openai.api_key = api_key
            self._old = _old_openai
        def chat(self): 
            return self._old.ChatCompletion

load_dotenv()

# --- CONFIG ---
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")
WEBHOOK_TOKEN: str = os.getenv("WEBHOOK_TOKEN", "Jonbet")
CANAL_ORIGEM_IDS_STR: str = os.getenv("CANAL_ORIGEM_IDS", "")
CANAL_DESTINO_ID: str = os.getenv("CANAL_DESTINO_ID", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "30"))
HISTORY_DB_PATH = os.getenv("HISTORY_DB_PATH", "./signals.db")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MESSAGE_URL = f"{TELEGRAM_API_URL}/sendMessage"

CANAL_ORIGEM_IDS: List[str] = [i.strip() for i in CANAL_ORIGEM_IDS_STR.split(",") if i.strip()]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# OpenAI client init
if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY não definido. Análises IA estarão desativadas.")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# --- Estado ---
last_signal_time = 0

app = FastAPI(title="JonBet IA v2")

# --- DB helpers ---
def init_db(path: str = HISTORY_DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            iso TEXT NOT NULL,
            chat_id TEXT,
            text TEXT,
            kind TEXT,         -- 'signal' or 'result'
            outcome TEXT       -- 'green', 'loss', or NULL
        )
    """)
    conn.commit()
    return conn

db = init_db()

def insert_event(kind: str, chat_id: str, text: str, outcome: Optional[str] = None):
    ts = int(time.time())
    iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    cur = db.cursor()
    cur.execute("INSERT INTO events (ts, iso, chat_id, text, kind, outcome) VALUES (?, ?, ?, ?, ?, ?)",
                (ts, iso, chat_id, text, kind, outcome))
    db.commit()
    logging.debug("Evento inserido no DB: %s %s", kind, outcome)

def fetch_recent_events(limit: int = 200) -> List[Dict[str, Any]]:
    cur = db.cursor()
    cur.execute("SELECT ts, iso, chat_id, text, kind, outcome FROM events ORDER BY ts DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({"ts": r[0], "iso": r[1], "chat_id": r[2], "text": r[3], "kind": r[4], "outcome": r[5]})
    return out[::-1]  # return chronological

# --- Mensagens/template ---
def build_final_message(analysis_text: str = "") -> str:
    base = (
        "🚨 **ENTRADA IMEDIATA NO BRANCO!** ⚪️\n\n"
        "🎯 JOGO: Double JonBet\n"
        "🔥 FOCO: BRANCO (FORÇADO)\n"
        "📊 Confiança: `Automática via IA`\n"
        "🧠 Observações IA: \n"
    )
    if analysis_text:
        base += f"{analysis_text}\n\n"
    base += "⚠️ **ESTRATÉGIA: G0 (SEM GALES)**\n💻 Site: Acessar Double"
    return base[:4096]

def build_simple_placar(text_lower: str) -> Optional[str]:
    # Detect result messages
    if any(p in text_lower for p in ["✅", "vitória", "green", "win"]):
        return f"✅ **GREEN!** 🤑\n\nÚltimo resultado no Double JonBet."
    if any(p in text_lower for p in ["loss", "perda", "perdeu", "❌"]):
        return f"❌ **LOSS!** 😥\n\nPronto para o próximo sinal de entrada."
    return None

# --- OpenAI analysis ---
def prepare_analysis_prompt(recent_events: List[Dict[str, Any]], lookback: int = 100) -> str:
    # keep last "lookback" items
    trimmed = recent_events[-lookback:]
    # create succinct dataset: timestamp hour, kind, outcome
    records = []
    for e in trimmed:
        ts = datetime.fromtimestamp(e["ts"], tz=timezone.utc)
        hour = ts.strftime("%H:%M")
        records.append({"iso": e["iso"], "hour": hour, "kind": e["kind"], "outcome": e["outcome"]})
    prompt = (
        "Você é um analista que recebe histórico de sinais e resultados do jogo Double JonBet.\n"
        "Analise os registros abaixo e responda em JSON com estas chaves:\n"
        "  - confianca_percent: int (taxa de acerto estimada para o BRANCO nos últimos registros)\n"
        "  - horas_quentes: lista de strings com intervalos 'HH:MM-HH:MM' onde ocorreram mais GREENS\n"
        "  - tendencia: 'alta'|'media'|'baixa'\n"
        "Use apenas os dados fornecidos e seja sucinto.\n\n"
        "Dados:\n"
        f"{json.dumps(records, ensure_ascii=False)}\n\n"
        "Retorne somente um JSON válido."
    )
    return prompt

def analyze_with_openai(recent_events: List[Dict[str, Any]]) -> str:
    if not openai_client:
        return "IA desativada (OPENAI_API_KEY ausente)."
    prompt = prepare_analysis_prompt(recent_events, lookback=200)
    try:
        # Use chat completion style; adapt call to your SDK
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300
        )
        # Extract text (SDK variations exist)
        text = resp.choices[0].message.content if hasattr(resp.choices[0].message, "content") else resp.choices[0].text
        # Try to parse JSON safely
        try:
            parsed = json.loads(text.strip())
            pretty = json.dumps(parsed, ensure_ascii=False)
            return pretty
        except Exception:
            # If model returned plain text, just return it raw
            return text.strip()
    except Exception as e:
        logging.error("Erro OpenAI: %s", e)
        return "Erro na análise IA."

# --- Telegram sender ---
async def send_telegram_message(chat_id: str, text: str):
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(SEND_MESSAGE_URL, json=payload, timeout=10)
            r.raise_for_status()
            logging.info("Mensagem enviada para %s", chat_id)
        except Exception as e:
            logging.error("Erro ao enviar Telegram: %s", e)

# --- FastAPI endpoints ---
@app.get("/")
def root():
    return {"status": "ok", "service": "Jonbet IA v2"}

@app.post("/webhook/{webhook_token}")
async def webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token inválido.")
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Payload inválido.")

    message = data.get("message", {}) or {}
    chat = message.get("chat", {}) or {}
    chat_id = str(chat.get("id", ""))
    text = message.get("text", "")
    if not chat_id or not text:
        return {"ok": True, "action": "ignored_no_text_or_chat"}

    # only process messages from configured source channels
    if CANAL_ORIGEM_IDS and chat_id not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored_wrong_source"}

    global last_signal_time
    text_lower = text.lower()

    # 1) Detectar se é placar/resultado (GREEN/LOSS)
    placar_message = build_simple_placar(text_lower)
    if placar_message:
        # é resultado: registrar no DB como 'result' e enviar placar formatado
        # Determina outcome
        outcome = "green" if any(p in text_lower for p in ["✅", "vitória", "green", "win"]) else "loss"
        insert_event(kind="result", chat_id=chat_id, text=text, outcome=outcome)
        await send_telegram_message(CANAL_DESTINO_ID, placar_message)
        logging.info("Placar processado e registrado: %s", outcome)
        return {"ok": True, "action": "result_logged_sent"}

    # 2) Se não for resultado, considerar sinal de entrada
    # Evitar spam via cooldown
    now = time.time()
    if now - last_signal_time < COOLDOWN_SECONDS:
        logging.info("Ignorado por cooldown.")
        # ainda gravamos o sinal no histórico como 'signal' (opcional)
        insert_event(kind="signal", chat_id=chat_id, text=text, outcome=None)
        return {"ok": True, "action": "ignored_cooldown_logged"}

    # Registramos o sinal
    insert_event(kind="signal", chat_id=chat_id, text=text, outcome=None)

    # Buscar histórico e pedir análise para a IA
    recent = fetch_recent_events(limit=500)
    ia_analysis = analyze_with_openai(recent)

    # Formatamos a mensagem final (forçar BRANCO)
    final = build_final_message(analysis_text=ia_analysis)

    await send_telegram_message(CANAL_DESTINO_ID, final)

    last_signal_time = now
    logging.info("Sinal (entrada forçada no BRANCO) enviado com análise IA.")
    return {"ok": True, "action": "signal_sent_with_analysis"}

# --- endpoint opcional para inspecionar DB (segurança: usar só internamente) ---
@app.get("/debug/recent")
def debug_recent(n: int = 50):
    rows = fetch_recent_events(limit=n)
    return {"count": len(rows), "rows": rows}