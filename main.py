# -*- coding: utf-8 -*-
# ✅ JonBet Auto Bot — Modo Aprendizado Ativo (Render)
# - Reenvia todo sinal como "Entrada no BRANCO"
# - Substitui o sinal anterior com resultado (GREEN/LOSS)
# - Ignora G1, G2, Gale, VW, variação etc
# - Mantém aprendizado: distância entre brancos
# - Proteção anti-duplicação de resultados

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

DATA_DIR = "/var/data"
os.makedirs(DATA_DIR, exist_ok=True)
LEARN_PATH = os.path.join(DATA_DIR, "learn.json")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MESSAGE_URL = f"{TELEGRAM_API_URL}/sendMessage"
EDIT_MESSAGE_URL = f"{TELEGRAM_API_URL}/editMessageText"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()
last_signal_time = 0
last_signal_msg_id: Optional[int] = None
last_result_text = ""
last_result_time = 0
app.state.processed_entries = set()

# ===================== APRENDIZADO =====================
learn_state = {
    "last_white_ts": None,
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
                data = json.load(f)
            for k, v in learn_state.items():
                data.setdefault(k, v)
            learn_state = data
    except Exception:
        pass

_load_learn()

# ===================== FUNÇÕES =====================
def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def _append_bounded(lst, val, maxlen=200):
    lst.append(val)
    if len(lst) > maxlen:
        del lst[:len(lst)-maxlen]

def extract_message(data: dict) -> dict:
    msg = data.get("message") or data.get("channel_post") or {}
    return {
        "chat": msg.get("chat", {}),
        "text": msg.get("text") or "",
        "message_id": msg.get("message_id")
    }

# ===================== CLASSIFICAÇÕES =====================
def is_entry_signal_any_color(raw: str) -> bool:
    t = _strip_accents(raw.lower())
    has_call = any(w in t for w in ["entrada confirmada", "entrar", "entrada", "apostar", "aposta"])
    has_color = any(w in t for w in ["verde", "preto", "⚫", "⬛", "🟢", "protecao", "proteção", "branco"])
    # ignora gale, vw, etc
    if re.search(r"\bg[ ]?1\b|\bg[ ]?2\b|gale|vw|v[ ]?w|variacao win|variação win", t):
        return False
    return has_call and has_color

def classificar_resultado(txt: str):
    t = _strip_accents(txt.lower())
    menciona_branco = ("branco" in t) or ("⚪" in txt) or ("⬜" in txt)
    if any(p in t for p in ["vitoria", "vitória", "ganho", "ganhamos", "bateu", "acertou", "win"]) and menciona_branco:
        return "GREEN"
    if any(p in t for p in ["loss", "derrota", "nao bateu", "não bateu", "nao deu", "falhou", "perdeu"]):
        return "LOSS"
    if any(p in t for p in ["vitoria", "vitória", "bateu", "acertou", "win"]) and not menciona_branco:
        return "LOSS"
    return None

# ===================== MENSAGENS =====================
def build_entry_message() -> str:
    return (
        "🚨 **ENTRADA IMEDIATA NO BRANCO!** ⚪️\n\n"
        "🎯 JOGO: Double JonBet\n"
        "🔥 FOCO: BRANCO\n"
        "📊 Aprendizado ativo (sem travas)\n\n"
        "⚠️ **ESTRATÉGIA: G0 (ZERO GALES)**\n"
        "💻 Site: Acessar Double"
    )

def build_result_message(resultado_txt: str) -> str:
    stones = learn_state.get("stones_since_last_white", 0)
    med_stones = int(median(learn_state["stones_gaps"])) if learn_state["stones_gaps"] else 0
    return (
        f"{resultado_txt}\n\n"
        f"🪙 *Distância entre BRANCOS:* {stones} pedras "
        f"(mediana: {med_stones} pedras)"
    )

# ===================== TELEGRAM =====================
async def send_telegram_message(chat_id: str, text: str):
    async with httpx.AsyncClient() as client:
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        try:
            r = await client.post(SEND_MESSAGE_URL, json=payload, timeout=15)
            r.raise_for_status()
            return r.json().get("result", {}).get("message_id")
        except Exception as e:
            logging.error(f"Erro ao enviar mensagem: {e}")
            return None

async def edit_telegram_message(chat_id: str, msg_id: int, text: str):
    async with httpx.AsyncClient() as client:
        payload = {
            "chat_id": chat_id,
            "message_id": msg_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            r = await client.post(EDIT_MESSAGE_URL, json=payload, timeout=15)
            r.raise_for_status()
        except Exception as e:
            logging.error(f"Erro ao editar mensagem: {e}")

# ===================== ROTAS =====================
@app.get("/")
def root():
    return {"status": "ok", "service": "JonBet — Substitui resultado no branco"}

@app.post(f"/webhook/{{webhook_token}}")
async def webhook(webhook_token: str, request: Request):
    global last_signal_msg_id, last_signal_time, last_result_text, last_result_time

    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token incorreto")

    data = await request.json()
    msg = extract_message(data)
    chat_id = str(msg["chat"].get("id"))
    text = msg["text"].strip()

    if chat_id not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored"}

    # conta pedras
    learn_state["stones_since_last_white"] = learn_state.get("stones_since_last_white", 0) + 1

    # classifica resultado
    resultado = classificar_resultado(text)
    if resultado in ("GREEN", "LOSS"):
        # anti duplicação: evita repetição do mesmo texto
        if text == last_result_text and time.time() - last_result_time < 20:
            return {"ok": True, "action": "ignored_duplicate_result"}
        last_result_text = text
        last_result_time = time.time()

        if resultado == "GREEN":
            _append_bounded(learn_state["stones_gaps"], int(learn_state["stones_since_last_white"]), 200)
            learn_state["stones_since_last_white"] = 0
            learn_state["last_white_ts"] = time.time()
            _save_learn()
            msg_text = build_result_message("✅ **GREEN no BRANCO!** ⚪️")
        else:
            msg_text = build_result_message("❌ **LOSS** 😥")

        if last_signal_msg_id:
            await edit_telegram_message(CANAL_DESTINO_ID, last_signal_msg_id, msg_text)
            last_signal_msg_id = None
        else:
            await send_telegram_message(CANAL_DESTINO_ID, msg_text)

        return {"ok": True, "action": f"{resultado.lower()}_logged"}

    # detectar entrada
    if is_entry_signal_any_color(text):
        msg_id = await send_telegram_message(CANAL_DESTINO_ID, build_entry_message())
        if msg_id:
            last_signal_msg_id = msg_id
            last_signal_time = time.time()
        _save_learn()
        return {"ok": True, "action": "entry_forwarded"}

    _save_learn()
    return {"ok": True, "action": "ignored"}