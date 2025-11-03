# -*- coding: utf-8 -*-
# ✅ JonBet Auto Bot - Conversor de sinais (somente Branco ⚪️)
# Modo: Aprendizado ativo + fluxo contínuo (sem gale, sem travas)

import os
import json
import time
import logging
import re
import unicodedata
from datetime import datetime
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
app.state.processed_entries = set()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===================== APRENDIZADO =====================
learn_state = {
    "last_white_ts": None,
    "white_gaps": [],
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
    try:
        if os.path.exists(LEARN_PATH):
            with open(LEARN_PATH, "r") as f:
                learn_state.update(json.load(f))
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

def extract_message(data: dict):
    msg = data.get("message") or data.get("channel_post") or {}
    return {
        "chat": msg.get("chat", {}),
        "text": msg.get("text") or "",
        "message_id": msg.get("message_id")
    }

async def send_telegram_message(chat_id: str, text: str):
    async with httpx.AsyncClient() as client:
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        try:
            r = await client.post(SEND_MESSAGE_URL, json=payload, timeout=15)
            r.raise_for_status()
        except Exception as e:
            logging.error(f"Erro ao enviar mensagem: {e}")

def is_entrada_confirmada(text: str) -> bool:
    t = _strip_accents(text.lower())
    return (
        "entrada confirmada" in t
        or "apostar no" in t
        or "entrar apos" in t
        or "🎰 jogo" in t
    )

def ignorar_gale(text: str) -> bool:
    t = _strip_accents(text.lower())
    return any(x in t for x in ["g1", "g2", "vw", "protecao", "proteção", "⚠️"])

def classificar_resultado(txt: str) -> Optional[str]:
    t = _strip_accents(txt.lower())
    if any(w in t for w in ["vitoria", "vitória", "acertamos", "acerto"]) and "branco" in t:
        return "GREEN_VALIDO"
    if any(w in t for w in ["loss", "derrota", "nao deu", "não deu", "falhou"]):
        return "LOSS"
    return None

def build_entry_message(num_alvo: str) -> str:
    return (
        "✅ Entrada confirmada!\n"
        "Apostar no branco ⚪️\n"
        f"Entrar após: ⚪️ {num_alvo}\n"
        "🎰 Jogo: Double - JonBet\n"
        "💻 Site: Acessar Double"
    )

def build_result_message(resultado_txt: str) -> str:
    stones = learn_state.get("stones_since_last_white", 0)
    med_stones = int(median(learn_state["stones_gaps"])) if learn_state["stones_gaps"] else 0
    return f"{resultado_txt}\n\n🪙 *Distância entre brancos:* {stones} pedras (mediana: {med_stones})"

# ===================== WEBHOOK =====================
@app.get("/")
def root():
    return {"status": "ok", "service": "JonBet - Branco Automático (Fluxo Livre)"}

@app.post(f"/webhook/{{webhook_token}}")
async def webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token incorreto")

    data = await request.json()
    msg = extract_message(data)
    chat_id = str(msg["chat"].get("id"))
    text = msg["text"].strip()

    if chat_id not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored_source"}

    # Conta pedras
    learn_state["stones_since_last_white"] = learn_state.get("stones_since_last_white", 0) + 1

    # Classifica resultados
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

    # Detecta entrada confirmada
    if is_entrada_confirmada(text) and not ignorar_gale(text):
        # Pega número após "Entrar após:"
        m = re.search(r"(\d{1,2})", text)
        num_alvo = m.group(1) if m else "?"
        await send_telegram_message(CANAL_DESTINO_ID, build_entry_message(num_alvo))
        return {"ok": True, "action": "entry_forwarded"}

    _save_learn()
    return {"ok": True, "action": "ignored"}