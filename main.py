import os
import json
import time
import logging
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Request
import httpx
from google import genai
from google.genai import types

# --- CONFIGURAÇÕES DE AMBIENTE ---
# Puxe variáveis de ambiente ou use valores de fallback
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "SEU_BOT_TOKEN_AQUI")
WEBHOOK_TOKEN: str = os.getenv("WEBHOOK_TOKEN", "Jonbet") # Seu token de seguranca
CANAL_ORIGEM_IDS_STR: str = os.getenv("CANAL_ORIGEM_IDS", "")
CANAL_DESTINO_ID: str = os.getenv("CANAL_DESTINO_ID", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Conversão de IDs de Canal (lista de strings)
CANAL_ORIGEM_IDS: List[str] = [id.strip() for id in CANAL_ORIGEM_IDS_STR.split(',') if id.strip()]

# URLs da API do Telegram
TELEGRAM_API_URL: str = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MESSAGE_URL: str = f"{TELEGRAM_API_URL}/sendMessage"

# --- CONFIGURAÇÕES DE ESTADO (Para evitar duplicação - COOLDOWN) ---
# O servidor manterá esta variável na memória enquanto estiver ativo.
last_signal_time = 0
COOLDOWN_SECONDS = 30 # Tempo mínimo entre o envio de sinais (em segundos)

# --- CONFIGURAÇÕES DE IA E CONFIANÇA ---
# Modo Destravado (APRENDIZADOBRUTO)
PERCENTUAL_MINIMO_CONFIANCA: float = float(os.getenv("MIN_CONFIDENCE", "0.0"))

# Inicialização da API do Gemini
try:
    if GEMINI_API_KEY:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        logging.info("Cliente Gemini inicializado com sucesso.")
    else:
        genai_client = None
        logging.warning("GEMINI_API_KEY ausente. A funcionalidade de IA será desativada.")
except Exception as e:
    genai_client = None
    logging.error(f"Erro ao inicializar o cliente Gemini: {e}")

app = FastAPI()

# --- MODELO DE IA ---
SYSTEM_INSTRUCTION = (
    "Você é um especialista em análise de sinais para jogos de Double/Roleta, focado exclusivamente na cor BRANCO. "
    "Sua única tarefa é analisar o texto da mensagem de entrada. Você deve seguir estritamente as seguintes regras: "
    "1. Seu FOCO É EXCLUSIVO NO BRANCO. Você deve analisar a mensagem e determinar se ela está sugerindo uma aposta ou padrão que favorece o BRANCO (⚪). "
    "2. Se a mensagem for sobre BRANCO, você deve gerar uma 'Confiança' (score de 0.0 a 100.0) e uma 'Justificativa'. "
    "3. Se a mensagem for sobre outras cores (PRETO, VERMELHO, VERDE etc.) ou for MISTURADA e você não conseguir extrair uma indicação clara de BRANCO, sua Confiança DEVE SER SEMPRE 0.0. "
    "4. SUA RESPOSTA DEVE SER APENAS O JSON, NADA MAIS. O JSON DEVE SEGUIR ESTA ESTRUTURA RIGOROSA: "
    '{"confianca": <float entre 0.0 e 100.0>, "justificativa": "<string explicando a confiança em até 50 palavras>"}'
)

def analyze_message_with_gemini(message_text: str) -> Optional[Dict[str, Any]]:
    """Envia a mensagem para o Gemini para análise e pontuação."""
    if not genai_client:
        return None

    try:
        response = genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=message_text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
            )
        )
        
        json_output = json.loads(response.text)
        
        if "confianca" in json_output and "justificativa" in json_output:
            json_output["confianca"] = max(0.0, min(100.0, float(json_output["confianca"])))
            return json_output
        else:
            logging.error("Resposta da IA com formato JSON inválido.")
            return None

    except Exception as e:
        logging.error(f"Erro na comunicação ou processamento da IA: {e}")
        return None

def build_final_message(original_text: str, ai_analysis: Dict[str, Any]) -> str:
    """Formata a mensagem final com análise da IA."""
    confianca: float = ai_analysis.get("confianca", 0.0)
    justificativa: str = ai_analysis.get("justificativa", "Análise indisponível.")

    modo = "APRENDIZADOBRUTO" if PERCENTUAL_MINIMO_CONFIANCA == 0.0 else "PRODUÇÃO"

    header = (
        f"⚠️ SINAL EXCLUSIVO BRANCO (MODO: {modo}) ⚠️\n"
        f"🎯 JOGO: Double JonBet\n"
        f"🔥 FOCO TOTAL NO **BRANCO** 🔥\n\n"
        f"📊 Confiança: `{confianca:.2f}%`\n"
        f"🧠 Análise Gemini: _{justificativa}_\n\n"
    )

    footer = (
        f"\n---\n"
        f"🔔 Sinal Original: {original_text}"
    )

    final_message = header + footer
    return final_message[:4096]

async def send_telegram_message(chat_id: str, text: str):
    """Envia a mensagem formatada para o canal de destino."""
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
            logging.error(f"Erro ao enviar mensagem HTTP {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            logging.error(f"Erro de requisição ao enviar mensagem: {e}")


# --- ENDPOINTS DA APLICAÇÃO ---

@app.get("/")
def read_root():
    """Endpoint de saúde para verificar se o serviço está vivo."""
    return {"status": "ok", "service": "Jonbet Telegram Bot is running."}

@app.post(f"/webhook/{{webhook_token}}")
async def telegram_webhook(webhook_token: str, request: Request):
    """Manipula as requisições Webhook do Telegram."""
    
    # 1. VERIFICAÇÃO DO TOKEN DE SEGURANÇA
    if webhook_token != WEBHOOK_TOKEN:
        logging.error(f"Tentativa de acesso com token inválido: {webhook_token}")
        raise HTTPException(status_code=403, detail="Token de segurança inválido.")

    try:
        data = await request.json()
    except json.JSONDecodeError:
        logging.error("Payload não é um JSON válido.")
        raise HTTPException(status_code=400, detail="Payload inválido.")

    # 2. EXTRAÇÃO E VERIFICAÇÃO BÁSICA
    if "message" not in data:
        return {"ok": True, "action": "ignored_no_message"}

    message: Dict[str, Any] = data["message"]
    chat_id: Optional[int] = message.get("chat", {}).get("id")
    text: Optional[str] = message.get("text")

    if not chat_id or not text:
        return {"ok": True, "action": "ignored_no_text_or_chat"}

    global last_signal_time
    logging.info(f"Mensagem recebida de Chat ID: {chat_id}")

    # 3. FILTRAGEM DE CANAL DE ORIGEM
    chat_id_str = str(chat_id)
    if chat_id_str not in CANAL_ORIGEM_IDS:
        logging.info(f"Mensagem ignorada: ID de chat {chat_id} não é um canal de origem configurado.")
        return {"ok": True, "action": "ignored_wrong_source"}

    logging.info("Mensagem roteada para PROCESSAMENTO DE SINAL.")


    # --- BLOCO DE FILTRAGEM DE CONTEÚDO: FOCO TOTAL NO BRANCO ---
    text_lower = text.lower()

    # 1. Deve conter BRANCO para ser considerado.
    contains_branco = "branco" in text_lower or "⚪" in text or "⬜" in text

    # 2. NÃO DEVE conter outras CORES, GALE, WIN, ou MARTINGALE.
    contains_outras_cores_ou_gale = (
        "preto" in text_lower or "⚫" in text_lower or 
        "vermelho" in text_lower or "🔴" in text_lower or
        "verde" in text_lower or "🟢" in text_lower or
        "gale" in text_lower or "gales" in text_lower or # Exclui GALE
        "✅" in text_lower or "win" in text_lower or # Exclui WIN/Check Mark
        "loss" in text_lower # Exclui LOSS (garante que só é sinal de entrada)
    )

    if not contains_branco:
        logging.info("Sinal ignorado: Não contém a palavra/emoji 'BRANCO'.")
        return {"ok": True, "action": "ignored_not_branco"}

    if contains_outras_cores_ou_gale:
        logging.info("Sinal ignorado: Contém BRANCO, mas também contém GALE, WIN, LOSS ou outras CORES (sinal misto/resultado).")
        return {"ok": True, "action": "ignored_mixed_signal"}
    # --- FIM DO BLOCO DE FILTRAGEM ---


    # 4. ANÁLISE DE IA
    if not genai_client:
        logging.warning("IA desativada. Sinal não processado por IA.")
        final_message = f"🚨 IA DESATIVADA. Sinal Original:\n\n{text}"
        await send_telegram_message(CANAL_DESTINO_ID, final_message)
        last_signal_time = time.time() # Atualiza o lock
        return {"ok": True, "action": "sent_original_message_ai_off"}


    ai_analysis = analyze_message_with_gemini(text)
    
    if not ai_analysis:
        logging.error("Análise da IA falhou ou retornou um JSON inválido.")
        return {"ok": True, "action": "ai_analysis_failed"}
    
    confianca_ia = ai_analysis.get("confianca", 0.0)

    # 5. FILTRO DE CONFIANÇA (Modo Destravado: 0.0)
    if confianca_ia >= PERCENTUAL_MINIMO_CONFIANCA:
        
        # 6. FILTRO ANTI-DUPLICAÇÃO (Timestamp Lock)
        current_time = time.time()
        
        if current_time - last_signal_time < COOLDOWN_SECONDS:
            cooldown_remaining = COOLDOWN_SECONDS - (current_time - last_signal_time)
            logging.info(f"Sinal ignorado devido ao COOLDOWN. {cooldown_remaining:.2f}s restantes.")
            return {"ok": True, "action": "ignored_cooldown"}
            
        # Se passou no cooldown, envia e atualiza o timestamp
        final_message = build_final_message(text, ai_analysis)
        await send_telegram_message(CANAL_DESTINO_ID, final_message)
        
        last_signal_time = current_time # Atualiza o lock
        
        logging.info(f"Sinal enviado! Confiança: {confianca_ia:.2f}%")
        return {"ok": True, "action": "signal_sent", "confidence": confianca_ia}
    else:
        logging.info(f"Sinal ignorado pelo filtro de confiança da IA: {confianca_ia:.2f}% (Mínimo: {PERCENTUAL_MINIMO_CONFIANCA:.2f}%)")
        return {"ok": True, "action": "ignored_low_confidence", "confidence": confianca_ia}
