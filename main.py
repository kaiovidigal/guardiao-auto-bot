# -*- coding: utf-8 -*-
# ✅ JonBet Auto Bot - Conversor de sinais (Versão Final + Diagnóstico Completo)
# Atualização 04/11/2025
# - Diagnóstico detalhado de envio (retorna status_code e texto do Telegram)
# - Endpoint /debug/sendtest para validar o envio
# - Correção de erro de parse_mode (Markdown inválido)
# - Logs reforçados para rastrear travas, unlock e bloqueios

import os
import json
import time
import logging
import re
import unicodedata
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
import httpx
from statistics import median

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

app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===================== ESTADO DE APRENDIZADO =====================
learn_state = {
    "last_white_ts": None,
    "white_gaps": [],
    "stones_since_last_white": 0,
    "stones_gaps": [],
    "entry_active": False,
    "processed_ids": []
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
                loaded_state = json.load(f)
                learn_state.update(loaded_state)
                learn_state["processed_ids"] = learn_state.get("processed_ids", [])[-100:]
    except Exception:
        pass

_load_learn()

# ===================== UTILIDADES =====================
def _strip_accents(s: str) -> str:
    nfkd_form = unicodedata.normalize('NFKD', s)
    only_ascii = nfkd_form.encode('ascii', 'ignore').decode('utf-8')
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', only_ascii)
    return re.sub(r'\s+', ' ', cleaned).strip()

def _append_bounded(lst, val, maxlen=200):
    lst.append(val)
    if len(lst) > maxlen:
        del lst[:len(lst)-maxlen]

def extract_message(data: dict):
    msg = data.get("message") or data.get("channel_post") or {}
    return {
        "chat": msg.get("chat", {}),
        "text": msg.get("text") or "",
        "message_id": msg.get("message_id")
    }

# ===================== ENVIO TELEGRAM =====================
async def send_telegram_message(chat_id: str, text: str):
    """Envia mensagem ao Telegram com diagnóstico e proteção de formatação"""
    async with httpx.AsyncClient() as client:
        payload = {"chat_id": chat_id, "text": text}

        # tenta Markdown e recua para texto puro se falhar
        for mode in ["Markdown", None]:
            if mode:
                payload["parse_mode"] = mode
            else:
                payload.pop("parse_mode", None)

            try:
                r = await client.post(SEND_MESSAGE_URL, json=payload, timeout=15)
                logging.info(f"🔧 Envio Telegram: {r.status_code} | {r.text[:120]}")
                r.raise_for_status()
                logging.info(f"✅ Mensagem enviada com sucesso para {chat_id}")
                return
            except Exception as e:
                logging.error(f"Erro ao enviar ({mode}): {e}")
                # tenta novamente com texto puro
                continue

# ===================== LÓGICA =====================
def is_entrada_confirmada(text: str) -> bool:
    t_cleaned = _strip_accents(text).lower()
    is_double_blaze = "double blaze" in t_cleaned
    is_entry_format = "entrada sera para" in t_cleaned
    mentions_gale = "gale" in t_cleaned
    is_not_result = not any(w in t_cleaned for w in ["win", "loss", "derrota"])
    return is_double_blaze and is_entry_format and mentions_gale and is_not_result

def build_entry_message(text_original: str) -> str:
    return (
        "🚨 **CONVERSÃO: ENTRADA IMEDIATA NO BRANCO!** ⚪️\n\n"
        f"Apostar no **Branco** ⚪️\n"
        f"Entrar após: ⚪️ ?\n\n"
        "🎰 Jogo: Double - JonBet\n"
        "💻 Site: Acessar Double"
    )

def classificar_resultado(txt: str) -> Optional[str]:
    t_cleaned = _strip_accents(txt).lower()
    if ("win" in t_cleaned or "vitoria" in t_cleaned) and ("branco" in t_cleaned or "⚪" in txt):
        return "GREEN_VALIDO"
    if "loss" in t_cleaned or "derrota" in t_cleaned or "❌" in txt or \
       (("win" in t_cleaned or "vitoria" in t_cleaned) and any(c in txt for c in ["⚫", "🔴", "🟢"])):
        return "LOSS"
    return None

def build_result_message(resultado_status: str) -> str:
    stones = learn_state.get("stones_since_last_white", 0)
    try:
        med_stones = int(median(learn_state["stones_gaps"])) if learn_state["stones_gaps"] else 0
    except Exception:
        med_stones = 0

    status_msg = "✅ **GREEN!**" if resultado_status == "GREEN_VALIDO" else "❌ **LOSS** 😥"
    return (
        f"Resultado: {status_msg}\n\n"
        f"🪙 *Distância entre brancos:* {stones} pedras (mediana: {med_stones})"
    )

# ===================== WEBHOOK =====================
@app.get("/")
def root():
    return {"status": "ok", "service": "JonBet - Branco Automático (Diagnóstico Final)"}

@app.get("/debug/sendtest")
async def send_test():
    """Teste manual para verificar envio ao canal destino"""
    await send_telegram_message(CANAL_DESTINO_ID, "🚀 Teste direto do JonBet Auto Bot (Render OK)")
    return {"ok": True, "sent_to": CANAL_DESTINO_ID}

@app.post(f"/webhook/{{webhook_token}}")
async def webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token incorreto")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON inválido")

    msg = extract_message(data)
    chat_id = str(msg.get("chat", {}).get("id"))
    text = (msg.get("text") or "").strip()
    message_id = msg.get("message_id")

    if message_id and message_id in learn_state["processed_ids"]:
        logging.info(f"Ignorando ID duplicado: {message_id}")
        return {"ok": True, "action": "ignored_duplicate_id"}

    if chat_id == CANAL_DESTINO_ID or chat_id not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored_channel"}

    if message_id:
        learn_state["processed_ids"].append(message_id)
        learn_state["processed_ids"] = learn_state["processed_ids"][-100:]
        _save_learn()

    resultado = classificar_resultado(text)

    # === RESULTADO (UNLOCK) ===
    if resultado in ["GREEN_VALIDO", "LOSS"]:
        if learn_state.get("entry_active"):
            learn_state["entry_active"] = False

        if resultado == "GREEN_VALIDO":
            now = time.time()
            if learn_state.get("last_white_ts"):
                gap = now - float(learn_state["last_white_ts"])
                _append_bounded(learn_state["white_gaps"], gap, 200)
                _append_bounded(learn_state["stones_gaps"], learn_state["stones_since_last_white"], 200)

            learn_state["last_white_ts"] = now
            learn_state["stones_since_last_white"] = 0

        msg_text = build_result_message(resultado)
        logging.info(f"📩 Enviando resultado: {resultado}")
        await send_telegram_message(CANAL_DESTINO_ID, msg_text)
        _save_learn()
        return {"ok": True, "action": f"result_logged_and_unlocked ({resultado})"}

    # === ENTRADA (LOCK) ===
    if is_entrada_confirmada(text):
        if learn_state.get("entry_active"):
            return {"ok": True, "action": "ignored_entry_active_lock"}

        learn_state["entry_active"] = True
        learn_state["stones_since_last_white"] = learn_state.get("stones_since_last_white", 0) + 1
        msg_text = build_entry_message(text)
        logging.info("📩 Enviando entrada convertida para BRANCO ⚪️")
        await send_telegram_message(CANAL_DESTINO_ID, msg_text)
        _save_learn()
        return {"ok": True, "action": "entry_converted_and_locked"}

    _save_learn()
    return {"ok": True, "action": "ignored_non_entry_non_result"}