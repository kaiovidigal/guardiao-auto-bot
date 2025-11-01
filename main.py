import os
import json
import time
import logging
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Request
import httpx

# --- CONFIGURAÇÕES DE AMBIENTE ---
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "SEU_BOT_TOKEN_AQUI")
WEBHOOK_TOKEN: str = os.getenv("WEBHOOK_TOKEN", "Jonbet")
CANAL_ORIGEM_IDS_STR: str = os.getenv("CANAL_ORIGEM_IDS", "")
CANAL_DESTINO_ID: str = os.getenv("CANAL_DESTINO_ID", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")  # Mantido apenas por compatibilidade

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CANAL_ORIGEM_IDS: List[str] = [id.strip() for id in CANAL_ORIGEM_IDS_STR.split(',') if id.strip()]
TELEGRAM_API_URL: str = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MESSAGE_URL: str = f"{TELEGRAM_API_URL}/sendMessage"

# --- CONFIGURAÇÕES DE ESTADO ---
last_signal_time = 0
COOLDOWN_SECONDS = 30  # tempo mínimo entre sinais

app = FastAPI()

# --- FUNÇÕES AUXILIARES ---


def build_final_message() -> str:
    """Formata a mensagem padrão de ENTRADA no BRANCO."""
    return (
        f"🚨 **ENTRADA IMEDIATA NO BRANCO!** ⚪️\n\n"
        f"🎯 JOGO: Double JonBet\n"
        f"🔥 FOCO: BRANCO (FORÇADO)\n"
        f"📊 Confiança: `Filtro ON`\n"
        f"🧠 Estratégia: Entrada Automática Forçada em BRANCO\n\n"
        f"⚠️ **ESTRATÉGIA: G0 (SEM GALES)**\n"
        f"💻 Site: Acessar Double"
    )[:4096]


def build_simple_placar(text_lower: str) -> Optional[str]:
    """Detecta mensagens de resultado (GREEN/LOSS)."""
    contem_branco = "branco" in text_lower or "⚪" in text_lower or "⬜" in text_lower
    contem_cores = any(cor in text_lower for cor in ["preto", "vermelho", "verde", "⚫", "🔴", "🟢"])

    if contem_branco or "green" in text_lower or "vitória" in text_lower or "✅" in text_lower:
        return f"✅ **GREEN!** 🤑\n\nÚltimo resultado no Double JonBet."
    if contem_cores or "loss" in text_lower or "perda" in text_lower:
        return f"❌ **LOSS!** 😥\n\nPronto para o próximo sinal de entrada."
    return None


async def send_telegram_message(chat_id: str, text: str):
    """Envia a mensagem para o canal destino."""
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(SEND_MESSAGE_URL, json=payload, timeout=10)
            response.raise_for_status()
            logging.info(f"Mensagem enviada com sucesso para {chat_id}.")
        except httpx.HTTPStatusError as e:
            logging.error(f"Erro HTTP {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            logging.error(f"Erro de requisição: {e}")


# --- ENDPOINTS PRINCIPAIS ---


@app.get("/")
def read_root():
    return {"status": "ok", "service": "Jonbet Telegram Bot ativo e rodando."}


@app.post(f"/webhook/{{webhook_token}}")
async def telegram_webhook(webhook_token: str, request: Request):
    """Webhook do Telegram para processar mensagens do canal de sinais."""
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token de segurança inválido.")

    try:
        data = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Payload inválido.")

    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text")
    if not chat_id or not text:
        return {"ok": True, "action": "ignored_no_text_or_chat"}

    global last_signal_time
    text_lower = text.lower()

    chat_id_str = str(chat_id)
    if chat_id_str not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored_wrong_source"}

    logging.info("Mensagem recebida e roteada para processamento.")

    # --- BLOCO DE DETECÇÃO DE RESULTADO (IGNORAR GREEN/LOSS) ---
    is_resultado = any(p in text_lower for p in ["vitória", "✅", "loss", "perda", "green", "win", "gale"])
    if is_resultado:
        logging.info("Mensagem ignorada: detectado resultado (não é sinal de entrada).")
        return {"ok": True, "action": "ignored_result_message"}

    # --- BLOCO DE SINAL DE ENTRADA ---
    logging.info("Sinal identificado: convertendo para ENTRADA NO BRANCO.")

    # Respeita cooldown
    current_time = time.time()
    if current_time - last_signal_time < COOLDOWN_SECONDS:
        logging.info("Sinal ignorado devido ao COOLDOWN.")
        return {"ok": True, "action": "ignored_cooldown"}

    final_message = build_final_message()
    await send_telegram_message(CANAL_DESTINO_ID, final_message)

    last_signal_time = current_time
    logging.info("Sinal de ENTRADA (forçado BRANCO) enviado com sucesso!")
    return {"ok": True, "action": "signal_sent_force_branco"}