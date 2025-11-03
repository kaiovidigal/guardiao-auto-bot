# -*- coding: utf-8 -*-
# ✅ JonBet Auto Bot - Modo Aprendizado Ativo (Render)
# Reenvia todos os sinais no branco, armazena aprendizado, sem travas ou estratégias.

import os
import json
import time
import logging
import re
import unicodedata
from statistics import median
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
import httpx

# ===================== CONFIG =====================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "Jonbet")
CANAL_ORIGEM_IDS = [s.strip() for s in os.getenv("CANAL_ORIGEM_IDS", "-1003156785631").split(",")]
CANAL_DESTINO_ID = os.getenv("CANAL_DESTINO_ID", "-1002796105884")

COOLDOWN_SECONDS = 0
RESULT_WINDOW_SECONDS = 600

SMART_TIMING = True
AUTO_EXECUTE = True
CONF_THRESHOLD = 0.65
MIN_SAMPLES = 20
ALWAYS_SEND_ON_ENTRY = True

DATA_DIR = "/var/data"
os.makedirs(DATA_DIR, exist_ok=True)
HISTORICO_PATH = os.path.join(DATA_DIR, "historico.json")
LEARN_PATH = os.path.join(DATA_DIR, "learn.json")

app = FastAPI()
last_signal_time = 0
last_signal_msg_id: Optional[int] = None
app.state.processed_entries = set()

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MESSAGE_URL = f"{TELEGRAM_API_URL}/sendMessage"
EDIT_MESSAGE_URL = f"{TELEGRAM_API_URL}/editMessageText"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===================== APRENDIZADO =====================
learn_state = {
    "deltas": [],
    "last_entry_ts": None,
    "last_white_ts": None,
    "white_gaps": [],
    "stones_since_last_white": 0,
    "stones_gaps": [],
}

def _save_learn():
    try:
        with open(LEARN_PATH, "w") as f:
            json.dump(learn_state, f)
    except Exception as e:
        logging.error(f"Erro ao salvar aprendizado: {e}")

def _load_learn():
    global learn_state
    try:
        if os.path.exists(LEARN_PATH):
            with open(LEARN_PATH, "r") as f:
                learn_state.update(json.load(f))
    except Exception:
        pass

_load_learn()

# ===================== FUNÇÕES =====================
def _strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def _now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _append_bounded(lst, val, maxlen=200):
    lst.append(val)
    if len(lst) > maxlen:
        del lst[:len(lst)-maxlen]

async def send_telegram_message(chat_id, text):
    async with httpx.AsyncClient() as client:
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        try:
            r = await client.post(SEND_MESSAGE_URL, json=payload, timeout=15)
            r.raise_for_status()
            return r.json().get("result", {}).get("message_id")
        except Exception as e:
            logging.error(f"Erro ao enviar mensagem: {e}")
            return None

def extract_message(data):
    msg = data.get("message") or data.get("channel_post") or {}
    return {
        "chat": msg.get("chat", {}),
        "text": msg.get("text") or "",
        "message_id": msg.get("message_id")
    }

def is_entrada_branco(raw):
    t = _strip_accents(raw.lower())
    return "branco" in t or "⚪" in raw or "⬜" in raw

def classificar_resultado(txt):
    t = _strip_accents(txt.lower())
    if any(w in t for w in ["vitoria", "acertamos", "acerto"]) and "branco" in t:
        return "GREEN_VALIDO"
    if any(w in t for w in ["loss", "derrota", "nao deu", "não deu", "falhou"]):
        return "LOSS"
    return None

# ===================== MENSAGENS =====================
def build_final_message():
    return (
        "🚨 **ENTRADA IMEDIATA NO BRANCO!** ⚪️\n\n"
        "🎯 JOGO: Double JonBet\n"
        "🔥 FOCO: BRANCO\n"
        "📊 Confiança: `Filtro ON (TEXTUAL)`\n"
        "🧠 Análise: _Aprendizado ativo._\n\n"
        "⚠️ **ESTRATÉGIA: G0 (ZERO GALES)**\n"
        "💻 Site: Acessar Double"
    )

def build_result_message(resultado_txt):
    since = time.time() - float(learn_state["last_white_ts"]) if learn_state.get("last_white_ts") else 0
    stones = learn_state.get("stones_since_last_white", 0)
    med_gap = int(median(learn_state["white_gaps"])) if learn_state["white_gaps"] else 0
    med_stones = int(median(learn_state["stones_gaps"])) if learn_state["stones_gaps"] else 0

    return (
        f"{resultado_txt}\n\n"
        f"⏱️ *Tempo desde o último branco:* {int(since//60)}m {int(since%60)}s (mediana: {int(med_gap//60)}m {int(med_gap%60)}s)\n"
        f"🪙 *Distância entre brancos:* {stones} pedras (mediana: {med_stones} pedras)\n"
        f"🕒 *Último branco:* {datetime.fromtimestamp(learn_state['last_white_ts']).strftime('%H:%M:%S') if learn_state.get('last_white_ts') else '-'}"
    )

# ===================== ROTAS =====================
@app.get("/")
def root():
    return {"status": "ok", "service": "JonBet - Modo Aprendizado Ativo"}

@app.post(f"/webhook/{{webhook_token}}")
async def webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token incorreto")

    global last_signal_time, last_signal_msg_id

    data = await request.json()
    msg = extract_message(data)
    chat_id = str(msg["chat"].get("id"))
    text = msg["text"].strip()

    # só processa canal origem
    if chat_id not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored_source"}

    # aprendizado: conta pedra
    learn_state["stones_since_last_white"] = learn_state.get("stones_since_last_white", 0) + 1

    # detectar resultado
    resultado = classificar_resultado(text)
    if resultado == "GREEN_VALIDO":
        now = time.time()
        if learn_state.get("last_white_ts"):
            gap = now - float(learn_state["last_white_ts"])
            _append_bounded(learn_state["white_gaps"], gap, 200)
            _append_bounded(learn_state["stones_gaps"], learn_state["stones_since_last_white"], 200)
        learn_state["last_white_ts"] = now
        learn_state["stones_since_last_white"] = 0
        _save_learn()

        await send_telegram_message(CANAL_DESTINO_ID, build_result_message("✅ **GREEN no BRANCO!** ⚪️"))
        return {"ok": True, "action": "green_logged"}

    elif resultado == "LOSS":
        await send_telegram_message(CANAL_DESTINO_ID, build_result_message("❌ **LOSS** 😥"))
        return {"ok": True, "action": "loss_logged"}

    # detectar entrada branco
    if is_entrada_branco(text):
        last_signal_msg_id = await send_telegram_message(CANAL_DESTINO_ID, build_final_message())
        last_signal_time = time.time()
        with open(HISTORICO_PATH, "a") as f:
            f.write(json.dumps({"hora": _now_iso(), "tipo": "entrada"}) + "\n")
        return {"ok": True, "action": "entry_forwarded"}

    _save_learn()
    return {"ok": True, "action": "ignored"}