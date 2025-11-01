import os
import json
import time
import logging
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request
import httpx

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "Jonbet")
CANAL_ORIGEM_IDS = [s.strip() for s in os.getenv("CANAL_ORIGEM_IDS", "-1003156785631").split(",")]
CANAL_DESTINO_ID = os.getenv("CANAL_DESTINO_ID", "-1002796105884")
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "30"))

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MESSAGE_URL = f"{TELEGRAM_API_URL}/sendMessage"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()
last_signal_time = 0
historico_path = "historico.json"


def salvar_evento(tipo, resultado=None):
    registro = {
        "hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tipo": tipo,
        "resultado": resultado
    }
    with open(historico_path, "a") as f:
        f.write(json.dumps(registro) + "\n")


def build_final_message() -> str:
    return (
        "🚨 **ENTRADA IMEDIATA NO BRANCO!** ⚪️\n\n"
        "🎯 JOGO: Double JonBet\n"
        "🔥 FOCO: BRANCO\n"
        "📊 Confiança: `Filtro ON (TEXTUAL)`\n"
        "🧠 Análise: _Filtro de Texto Agressivo Ativado._\n\n"
        "⚠️ **ESTRATÉGIA: G0 (ZERO GALES)**\n"
        "💻 Site: Acessar Double"
    )[:4096]


async def send_telegram_message(chat_id: str, text: str):
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    async with httpx.AsyncClient() as client:
        try:
            await client.post(SEND_MESSAGE_URL, json=payload, timeout=15)
        except Exception as e:
            logging.error(f"Erro ao enviar mensagem: {e}")


def extract_message(data: dict) -> dict:
    msg = (
        data.get("message")
        or data.get("channel_post")
        or data.get("edited_message")
        or data.get("edited_channel_post")
        or {}
    )
    return {
        "chat": msg.get("chat", {}),
        "text": msg.get("text") or msg.get("caption") or "",
        "message_id": msg.get("message_id"),
    }


@app.post(f"/webhook/{{webhook_token}}")
async def webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token incorreto.")

    data = await request.json()
    msg = extract_message(data)
    chat_id = str(msg["chat"].get("id"))
    text = msg["text"].lower().strip()

    if chat_id not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored_wrong_source"}

    global last_signal_time
    now = time.time()

    if "branco" in text or "⚪" in text:
        if now - last_signal_time < COOLDOWN_SECONDS:
            return {"ok": True, "action": "cooldown"}
        salvar_evento("entrada")
        await send_telegram_message(CANAL_DESTINO_ID, build_final_message())
        last_signal_time = now
        return {"ok": True, "action": "entrada_enviada"}

    if any(w in text for w in ["green", "vitória", "✅"]):
        salvar_evento("resultado", "GREEN")
        await send_telegram_message(CANAL_DESTINO_ID, "✅ GREEN registrado.")
        return {"ok": True, "action": "green"}

    if any(w in text for w in ["loss", "❌", "perda"]):
        salvar_evento("resultado", "LOSS")
        await send_telegram_message(CANAL_DESTINO_ID, "❌ LOSS registrado.")
        return {"ok": True, "action": "loss"}

    return {"ok": True, "action": "ignored"}