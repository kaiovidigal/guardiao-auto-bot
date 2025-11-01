import os
import json
import time
import logging
from typing import Optional, Dict, Any, List

# As bibliotecas da IA são importadas, mas a funcionalidade é desativada.
from fastapi import FastAPI, HTTPException, Request
import httpx
from google import genai 
from google.genai import types

# --- CONFIGURAÇÕES DE AMBIENTE ---
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "SEU_BOT_TOKEN_AQUI")
WEBHOOK_TOKEN: str = os.getenv("WEBHOOK_TOKEN", "Jonbet")
CANAL_ORIGEM_IDS_STR: str = os.getenv("CANAL_ORIGEM_IDS", "")
CANAL_DESTINO_ID: str = os.getenv("CANAL_DESTINO_ID", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "") # Não será usada, mas mantida.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CANAL_ORIGEM_IDS: List[str] = [id.strip() for id in CANAL_ORIGEM_IDS_STR.split(',') if id.strip()]
TELEGRAM_API_URL: str = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MESSAGE_URL: str = f"{TELEGRAM_API_URL}/sendMessage"

# --- CONFIGURAÇÕES DE ESTADO ---
last_signal_time = 0
COOLDOWN_SECONDS = 30 

# --- CONFIGURAÇÕES DE IA (DESATIVADA) ---
PERCENTUAL_MINIMO_CONFIANCA: float = float(os.getenv("MIN_CONFIDENCE", "1.0"))
genai_client = None # FORÇANDO A DESATIVAÇÃO DA IA

app = FastAPI()

# --- FUNÇÕES ---

def build_final_message() -> str:
    """Formata a mensagem de ENTRADA BRANCO PADRÃO e LIMPA, sem depender da IA."""
    
    # Usamos valores fixos para a confiança/justificativa, pois a IA está desativada.
    return (
        f"🚨 **ENTRADA IMEDIATA NO BRANCO!** ⚪️\n\n"
        f"🎯 JOGO: Double JonBet\n"
        f"🔥 FOCO: BRANCO\n"
        f"📊 Confiança: `Filtro ON` (FILTRADO)\n"
        f"🧠 Análise Gemini: _Filtro de Texto Agressivo Ativado._\n\n"
        f"⚠️ **ESTRATÉGIA: G0 (ZERO GALES).**\n"
        f"💻 Site: Acessar Double"
    )[:4096]

def build_simple_placar(text_lower: str) -> Optional[str]:
    """Cria a mensagem de Placar Simples (GREEN/LOSS) Limpa."""
    
    # Regras de inferência (Branco = GREEN, Cores = LOSS)
    contem_branco = "branco" in text_lower or "⚪" in text_lower or "⬜" in text_lower
    contem_cores = "preto" in text_lower or "vermelho" in text_lower or "verde" in text_lower or "⚫" in text_lower or "🔴" in text_lower or "🟢" in text_lower
    
    # 1. Identificar GREEN
    if contem_branco or "green" in text_lower or "vitória" in text_lower or "✅" in text_lower:
        return f"✅ **GREEN!** 🤑\n\nÚltimo resultado no Double JonBet."
    
    # 2. Identificar LOSS
    if contem_cores or "loss" in text_lower or "perda" in text_lower:
        return f"❌ **LOSS!** 😥\n\nPronto para o próximo sinal de entrada."
        
    return None


async def send_telegram_message(chat_id: str, text: str):
    """Envia a mensagem formatada para o canal de destino."""
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(SEND_MESSAGE_URL, json=payload, timeout=10)
            response.raise_for_status()
            logging.info(f"Mensagem enviada com sucesso para {chat_id}.")
        except httpx.HTTPStatusError as e:
            logging.error(f"Erro ao enviar mensagem HTTP {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            logging.error(f"Erro de requisição ao enviar mensagem: {e}")


# --- ENDPOINTS DA APLICAÇÃO ---

@app.get("/")
def read_root(): return {"status": "ok", "service": "Jonbet Telegram Bot is running."}

@app.post(f"/webhook/{{webhook_token}}")
async def telegram_webhook(webhook_token: str, request: Request):
    
    if webhook_token != WEBHOOK_TOKEN: raise HTTPException(status_code=403, detail="Token de segurança inválido.")

    try: data = await request.json()
    except json.JSONDecodeError: raise HTTPException(status_code=400, detail="Payload inválido.")
    
    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text")
    if not chat_id or not text: return {"ok": True, "action": "ignored_no_text_or_chat"}

    global last_signal_time
    text_lower = text.lower()
    
    chat_id_str = str(chat_id)
    if chat_id_str not in CANAL_ORIGEM_IDS: return {"ok": True, "action": "ignored_wrong_source"}
    
    logging.info("Mensagem roteada para PROCESSAMENTO DE SINAL.")

    # --- BLOCO DE FILTRAGEM: PLACAR OU ENTRADA? ---

    # Usamos is_placar para identificar o que deve ser enviado como GREEN/LOSS
    is_placar = "loss" in text_lower or "perda" in text_lower or "vitória" in text_lower or "✅" in text_lower or "🟢" in text_lower
    contains_entrada_palavras_aposta = "aposta" in text_lower or "entrar" in text_lower or "duplo" in text_lower

    
    # === AVALIAÇÃO DE PLACAR/RESULTADO ===
    if is_placar:
        # Rejeita placar se for misturado com GALE ou palavras de entrada
        if contains_entrada_palavras_aposta or "gale" in text_lower:
            logging.info("Placar ignorado: Está misturado com sinais de entrada ou Gale.")
            return {"ok": True, "action": "ignored_mixed_placar"}
            
        final_placar_message = build_simple_placar(text_lower)
        if final_placar_message:
            await send_telegram_message(CANAL_DESTINO_ID, final_placar_message)
            logging.info("Placar simples enviado.")
            return {"ok": True, "action": "placar_sent"}


    # === AVALIAÇÃO DE ENTRADA BRANCO / G0 (FILTRO AGRESSIVO) ===
    contains_branco = "branco" in text_lower or "⚪" in text or "⬜" in text
    
    # FILTRO AGRESSIVO: Rejeita tudo que for placar/gale/cores (O MAIS IMPORTANTE!)
    contains_sujeira_entrada = (
        "gale" in text_lower or "gales" in text_lower or 
        "preto" in text_lower or "vermelho" in text_lower or "verde" in text_lower or 
        "⚫" in text_lower or "🔴" in text_lower or "🟢" in text_lower or 
        "vitória" in text_lower or "loss" in text_lower or "✅" in text_lower or 
        "perda" in text_lower
    )

    if not contains_branco:
        logging.info("Sinal ignorado: Não contém a palavra/emoji 'BRANCO'.")
        return {"ok": True, "action": "ignored_not_branco"}

    # Aplica o filtro agressivo
    if contains_sujeira_entrada:
        logging.info("Sinal ignorado: Contém BRANCO, mas também SUJEIRA (GALE, CORES, RESULTADO).")
        return {"ok": True, "action": "ignored_mixed_signal_not_g0"}
    
    # 4. COOLDOWN (APENAS PARA SINAIS DE ENTRADA LIMPA)
    # Se chegou até aqui, é um sinal de entrada limpo (passou no filtro agressivo).

    current_time = time.time()
    if current_time - last_signal_time < COOLDOWN_SECONDS:
        logging.info(f"Sinal ignorado devido ao COOLDOWN.")
        return {"ok": True, "action": "ignored_cooldown"}
        
    # Envia o modelo de entrada limpa
    final_message = build_final_message() 
    await send_telegram_message(CANAL_DESTINO_ID, final_message)
    
    last_signal_time = current_time 
    
    logging.info("Sinal (Filtrado por Texto) enviado!")
    return {"ok": True, "action": "signal_sent_text_filter"}
