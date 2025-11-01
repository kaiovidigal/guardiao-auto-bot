import os
import json
import time
import logging
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request
import httpx

# ---------- CONFIG ----------
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")
WEBHOOK_TOKEN: str = os.getenv("WEBHOOK_TOKEN", "Jonbet")

CANAL_ORIGEM_IDS_STR: str = os.getenv("CANAL_ORIGEM_IDS", "-1003156785631")
CANAL_DESTINO_ID: str = os.getenv("CANAL_DESTINO_ID", "-1002796105884")

COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "30"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

CANAL_ORIGEM_IDS: List[str] = [s.strip() for s in CANAL_ORIGEM_IDS_STR.split(",") if s.strip()]

TELEGRAM_API_URL: str = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MESSAGE_URL: str = f"{TELEGRAM_API_URL}/sendMessage"

# ---------- STATE ----------
last_signal_time = 0

# ---------- APP ----------
app = FastAPI()


# ---------- HELPERS ----------
def build_final_message() -> str:
    """Mensagem LIMPA de entrada branca (G0)."""
    return (
        "🚨 **ENTRADA IMEDIATA NO BRANCO!** ⚪️\n\n"
        "🎯 JOGO: Double JonBet\n"
        "🔥 FOCO: BRANCO\n"
        "📊 Confiança: `Filtro ON` (TEXTUAL)\n"
        "🧠 Análise: _Filtro de Texto Agressivo Ativado._\n\n"
        "⚠️ **ESTRATÉGIA: G0 (ZERO GALES).**\n"
        "💻 Site: Acessar Double"
    )[:4096]


def build_simple_placar(text_lower: str) -> Optional[str]:
    """Placar simples GREEN/LOSS."""
    contem_branco = ("branco" in text_lower) or ("⚪" in text_lower) or ("⬜" in text_lower)
    contem_cores = any(w in text_lower for w in ["preto", "vermelho", "verde", "⚫", "🔴", "🟢"])

    if contem_branco or "green" in text_lower or "vitória" in text_lower or "✅" in text_lower:
        return "✅ **GREEN!** 🤑\n\nÚltimo resultado no Double JonBet."
    if contem_cores or "loss" in text_lower or "perda" in text_lower:
        return "❌ **LOSS!** 😥\n\nPronto para o próximo sinal de entrada."
    return None


async def send_telegram_message(chat_id: str, text: str):
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(SEND_MESSAGE_URL, json=payload, timeout=15)
            r.raise_for_status()
            logging.info(f"Mensagem enviada para {chat_id}.")
        except Exception as e:
            logging.error(f"Erro ao enviar mensagem: {e}")


def extract_message(payload: dict) -> dict:
    """
    Aceita mensagens normais, posts de canal e edições.
    Também cobre textos que vêm como 'caption' (mídia).
    """
    msg = (
        payload.get("message")
        or payload.get("channel_post")
        or payload.get("edited_message")
        or payload.get("edited_channel_post")
        or {}
    )
    if not msg:
        return {}

    # Normaliza campos principais
    msg_norm = {
        "chat": msg.get("chat") or {},
        "text": msg.get("text") or msg.get("caption") or "",
        "date": msg.get("date"),
        "message_id": msg.get("message_id"),
    }
    return msg_norm


# ---------- ROUTES ----------
@app.get("/")
def root():
    return {"status": "ok", "service": "Jonbet Telegram Bot running (Text Filter Only)."}


@app.get("/debug/ping")
async def debug_ping():
    """Envia um ping para o canal destino (prova de vida)."""
    await send_telegram_message(CANAL_DESTINO_ID, "✅ Bot online (ping).")
    return {"ok": True}


@app.post(f"/webhook/{{webhook_token}}")
async def telegram_webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token de segurança inválido.")

    try:
        data = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Payload inválido.")

    msg = extract_message(data)
    if not msg:
        logging.info("Ignorado: payload sem message/channel_post.")
        return {"ok": True, "action": "ignored_no_message"}

    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    text = (msg.get("text") or "").strip()
    if not chat_id or not text:
        logging.info("Ignorado: sem chat_id ou sem texto/caption.")
        return {"ok": True, "action": "ignored_no_text_or_chat"}

    chat_id_str = str(chat_id)
    if chat_id_str not in CANAL_ORIGEM_IDS:
        logging.info(f"Ignorado: origem {chat_id_str} não autorizada.")
        return {"ok": True, "action": "ignored_wrong_source"}

    logging.info("Mensagem válida recebida para processamento.")

    # --- Decide se é PLACAR ---
    text_lower = text.lower()
    is_placar = any(w in text_lower for w in ["loss", "perda", "vitória", "✅", "🟢"])
    contains_entrada_palavras_aposta = any(w in text_lower for w in ["aposta", "entrar", "duplo", "double"])

    if is_placar:
        if contains_entrada_palavras_aposta or "gale" in text_lower:
            logging.info("Placar ignorado: mistura com entrada/gale.")
            return {"ok": True, "action": "ignored_mixed_placar"}
        placar = build_simple_placar(text_lower)
        if placar:
            await send_telegram_message(CANAL_DESTINO_ID, placar)
            logging.info("Placar simples enviado.")
            return {"ok": True, "action": "placar_sent"}

    # --- Entrada BRANCO (Filtro agressivo G0) ---
    contains_branco = ("branco" in text_lower) or ("⚪" in text) or ("⬜" in text)
    contains_sujeira_entrada = any(
        w in text_lower
        for w in [
            "gale", "gales",
            "preto", "vermelho", "verde",
            "⚫", "🔴", "🟢",
            "vitória", "loss", "✅", "perda"
        ]
    )

    if not contains_branco:
        logging.info("Sinal ignorado: não contém BRANCO.")
        return {"ok": True, "action": "ignored_not_branco"}

    if contains_sujeira_entrada:
        logging.info(f"Sinal ignorado (sujeira): '{text[:80]}'")
        return {"ok": True, "action": "ignored_mixed_signal_not_g0"}

    global last_signal_time
    now = time.time()
    if now - last_signal_time < COOLDOWN_SECONDS:
        logging.info("Sinal ignorado por cooldown.")
        return {"ok": True, "action": "ignored_cooldown"}

    await send_telegram_message(CANAL_DESTINO_ID, build_final_message())
    last_signal_time = now
    logging.info("Sinal BRANCO enviado!")
    return {"ok": True, "action": "signal_sent_text_filter"}