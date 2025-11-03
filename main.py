# -*- coding: utf-8 -*-
# ✅ JonBet Auto Bot - Reenvio no Branco + Aprendizado Ativo
# Ignora G1, G2, VW e envia sinais instantâneos no branco

import os
import json
import time
import logging
import re
import unicodedata
from datetime import datetime
from statistics import median
from fastapi import FastAPI, HTTPException, Request
import httpx

# ===================== CONFIG =====================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "Jonbet")
CANAL_ORIGEM_IDS = [s.strip() for s in os.getenv("CANAL_ORIGEM_IDS", "-1003156785631").split(",")]
CANAL_DESTINO_ID = os.getenv("CANAL_DESTINO_ID", "-1002796105884")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MESSAGE_URL = f"{TELEGRAM_API_URL}/sendMessage"
EDIT_MESSAGE_URL = f"{TELEGRAM_API_URL}/editMessageText"

DATA_DIR = "/var/data"
os.makedirs(DATA_DIR, exist_ok=True)
LEARN_PATH = os.path.join(DATA_DIR, "learn.json")
HISTORICO_PATH = os.path.join(DATA_DIR, "historico.json")

app = FastAPI()
last_signal_time = 0
last_signal_msg_id = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===================== APRENDIZADO =====================
learn_state = {
    "last_white_ts": None,
    "stones_since_last_white": 0,
    "stones_gaps": []
}

def _save_learn():
    try:
        with open(LEARN_PATH, "w") as f:
            json.dump(learn_state, f)
    except Exception as e:
        logging.error(f"Erro ao salvar aprendizado: {e}")

def _load_learn():
    global learn_state
    if os.path.exists(LEARN_PATH):
        try:
            with open(LEARN_PATH, "r") as f:
                learn_state.update(json.load(f))
        except Exception:
            pass

_load_learn()

# ===================== FUNÇÕES =====================
def _strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def _append_bounded(lst, val, maxlen=200):
    lst.append(val)
    if len(lst) > maxlen:
        del lst[:len(lst) - maxlen]

def _now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

async def edit_telegram_message(chat_id, message_id, new_text):
    async with httpx.AsyncClient() as client:
        payload = {"chat_id": chat_id, "message_id": message_id, "text": new_text, "parse_mode": "Markdown"}
        try:
            r = await client.post(EDIT_MESSAGE_URL, json=payload, timeout=15)
            r.raise_for_status()
            return True
        except Exception as e:
            logging.error(f"Erro ao editar mensagem: {e}")
            return False

def extract_message(data):
    msg = data.get("message") or data.get("channel_post") or {}
    return {"chat": msg.get("chat", {}), "text": msg.get("text") or "", "message_id": msg.get("message_id")}

def classificar_resultado(txt):
    t = _strip_accents(txt.lower())
    if any(w in t for w in ["vitoria", "vitória", "green", "acertamos", "acerto"]) and "branco" in t:
        return "GREEN_VALIDO"
    if any(w in t for w in ["loss", "derrota", "nao deu", "não deu", "falhou", "perdeu"]):
        return "LOSS"
    return None

def is_entrada_confirmada(t):
    tnorm = _strip_accents(t.lower())
    if not ("entrada confirmada" in tnorm or "apostar" in tnorm or "entrada" in tnorm):
        return False
    # ignora gales e variações
    if any(w in tnorm for w in ["g1", "g 1", "g2", "g 2", "vw", "variacao", "variação win", "win variação"]):
        return False
    # garante que tem preto/verde
    if any(c in tnorm for c in ["verde", "preto"]):
        return True
    return False

# ===================== MENSAGENS =====================
def build_entry_message():
    return (
        "🚨 **ENTRADA IMEDIATA NO BRANCO!** ⚪️\n\n"
        "🎯 JOGO: Double JonBet\n"
        "🔥 FOCO: BRANCO\n"
        "📊 Aprendizado ativo (sem travas)\n\n"
        "⚠️ **ESTRATÉGIA: G0 (ZERO GALES)**\n"
        "💻 Site: Acessar Double"
    )

def build_result_message(resultado_txt):
    stones = learn_state.get("stones_since_last_white", 0)
    stones_med = int(median(learn_state["stones_gaps"])) if learn_state["stones_gaps"] else 0
    return (
        f"{resultado_txt}\n\n"
        f"🪙 *Distância entre brancos:* {stones} pedras • mediana global {stones_med} pedras"
    )

# ===================== ROTAS =====================
@app.get("/")
def root():
    return {"status": "ok", "service": "JonBet Branco - Aprendizado + Reenvio"}

@app.post(f"/webhook/{{webhook_token}}")
async def webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token incorreto.")

    global last_signal_time, last_signal_msg_id

    data = await request.json()
    msg = extract_message(data)
    chat_id = str(msg["chat"].get("id"))
    text = msg["text"].strip()

    if chat_id not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored_source"}

    # incremento de pedras (toda msg conta)
    learn_state["stones_since_last_white"] = learn_state.get("stones_since_last_white", 0) + 1

    # resultado
    resultado = classificar_resultado(text)
    if resultado:
        if resultado == "GREEN_VALIDO":
            now = time.time()
            if learn_state.get("last_white_ts"):
                gap = now - float(learn_state["last_white_ts"])
                _append_bounded(learn_state["stones_gaps"], learn_state["stones_since_last_white"], 200)
            learn_state["last_white_ts"] = now
            learn_state["stones_since_last_white"] = 0
            _save_learn()
            await edit_telegram_message(CANAL_DESTINO_ID, last_signal_msg_id, build_result_message("✅ **GREEN no BRANCO!** ⚪️"))
        elif resultado == "LOSS":
            await edit_telegram_message(CANAL_DESTINO_ID, last_signal_msg_id, build_result_message("❌ **LOSS** 😥"))
        return {"ok": True, "action": "resultado_processado"}

    # sinal de entrada
    if is_entrada_confirmada(text):
        last_signal_msg_id = await send_telegram_message(CANAL_DESTINO_ID, build_entry_message())
        last_signal_time = time.time()
        with open(HISTORICO_PATH, "a") as f:
            f.write(json.dumps({"hora": _now_iso(), "tipo": "entrada"}) + "\n")
        _save_learn()
        return {"ok": True, "action": "sinal_enviado"}

    _save_learn()
    return {"ok": True, "action": "ignored"}