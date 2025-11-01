import os
import json
import time
import logging
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request
import httpx

# ========== CONFIG ==========

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "Jonbet")
CANAL_ORIGEM_IDS = [s.strip() for s in os.getenv("CANAL_ORIGEM_IDS", "-1003156785631").split(",")]
CANAL_DESTINO_ID = os.getenv("CANAL_DESTINO_ID", "-1002796105884")
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "30"))

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MESSAGE_URL = f"{TELEGRAM_API_URL}/sendMessage"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========== VARIÁVEIS GLOBAIS ==========
app = FastAPI()
last_signal_time = 0
stats = {"green": 0, "loss": 0, "date": datetime.now().strftime("%Y-%m-%d")}


# ========== FUNÇÕES BASE ==========

def reset_daily_stats():
    """Zera contadores diariamente."""
    global stats
    today = datetime.now().strftime("%Y-%m-%d")
    if stats["date"] != today:
        stats = {"green": 0, "loss": 0, "date": today}
        logging.info("✅ Contadores zerados para o novo dia.")


def build_final_message() -> str:
    """Mensagem limpa de ENTRADA BRANCO"""
    return (
        "🚨 **ENTRADA IMEDIATA NO BRANCO!** ⚪️\n\n"
        "🎯 JOGO: Double JonBet\n"
        "🔥 FOCO: BRANCO\n"
        "📊 Confiança: `Filtro ON (TEXTUAL)`\n"
        "🧠 Análise: _Filtro de Texto Agressivo Ativado._\n\n"
        "⚠️ **ESTRATÉGIA: G0 (ZERO GALES)**\n"
        "💻 Site: Acessar Double"
    )[:4096]


def build_simple_placar(text_lower: str) -> Optional[str]:
    """Placar simples com contador"""
    global stats
    reset_daily_stats()

    if any(w in text_lower for w in ["green", "vitória", "✅", "🟢", "⚪"]):
        stats["green"] += 1
        return (
            f"✅ **GREEN!** 🤑\n\n"
            f"Último resultado no Double JonBet.\n\n"
            f"📈 Contagem diária: {stats['green']} GREEN | {stats['loss']} LOSS"
        )
    if any(w in text_lower for w in ["loss", "perda", "❌", "🔴"]):
        stats["loss"] += 1
        return (
            f"❌ **LOSS!** 😥\n\n"
            f"Pronto para o próximo sinal.\n\n"
            f"📉 Contagem diária: {stats['green']} GREEN | {stats['loss']} LOSS"
        )
    return None


async def send_telegram_message(chat_id: str, text: str, reply_to: Optional[int] = None):
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    if reply_to:
        payload["reply_to_message_id"] = reply_to

    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(SEND_MESSAGE_URL, json=payload, timeout=15)
            r.raise_for_status()
            logging.info(f"Mensagem enviada: {text[:40]}")
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
    if not msg:
        return {}

    return {
        "chat": msg.get("chat", {}),
        "text": msg.get("text") or msg.get("caption") or "",
        "message_id": msg.get("message_id"),
    }


# ========== ENDPOINTS ==========

@app.get("/")
def root():
    return {"status": "ok", "service": "Jonbet Bot ativo e rodando."}


@app.post(f"/webhook/{{webhook_token}}")
async def webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token incorreto.")

    try:
        data = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Payload inválido.")

    msg = extract_message(data)
    if not msg:
        return {"ok": True, "action": "ignored_no_message"}

    chat_id = str(msg["chat"].get("id"))
    text = msg["text"].strip().lower()
    message_id = msg["message_id"]

    if chat_id not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored_wrong_source"}

    logging.info(f"Mensagem recebida: {text[:60]}")

    global last_signal_time
    now = time.time()
    reset_daily_stats()

    # ===== ENTRADA =====
    if "branco" in text or "⚪" in text or "⬜" in text:
        contains_sujeira = any(w in text for w in ["gale", "preto", "vermelho", "verde", "loss", "✅", "vitória"])
        if contains_sujeira:
            return {"ok": True, "action": "ignored_mixed_signal"}

        if now - last_signal_time < COOLDOWN_SECONDS:
            return {"ok": True, "action": "ignored_cooldown"}

        await send_telegram_message(CANAL_DESTINO_ID, build_final_message())
        last_signal_time = now
        return {"ok": True, "action": "signal_sent"}

    # ===== PLACAR =====
    if any(w in text for w in ["green", "loss", "vitória", "perda", "✅", "❌"]):
        placar = build_simple_placar(text)
        if placar:
            # envia como resposta (logo abaixo)
            await send_telegram_message(CANAL_DESTINO_ID, placar)
            return {"ok": True, "action": "placar_sent"}

    return {"ok": True, "action": "ignored_no_match"}