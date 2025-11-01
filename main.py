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
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "SEU_BOT_TOKEN_AQUI")
WEBHOOK_TOKEN: str = os.getenv("WEBHOOK_TOKEN", "Jonbet")
CANAL_ORIGEM_IDS_STR: str = os.getenv("CANAL_ORIGEM_IDS", "")
CANAL_DESTINO_ID: str = os.getenv("CANAL_DESTINO_ID", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CANAL_ORIGEM_IDS: List[str] = [id.strip() for id in CANAL_ORIGEM_IDS_STR.split(',') if id.strip()]
TELEGRAM_API_URL: str = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MESSAGE_URL: str = f"{TELEGRAM_API_URL}/sendMessage"

# --- CONFIGURAÇÕES DE ESTADO ---
last_signal_time = 0
COOLDOWN_SECONDS = 30 

# --- CONFIGURAÇÕES DE IA E CONFIANÇA ---
# Recomendado: Mantenha em 1.0. Se for 0.0, o modo APRENDIZADOBRUTO aceitará sinais misturados.
PERCENTUAL_MINIMO_CONFIANCA: float = float(os.getenv("MIN_CONFIDENCE", "1.0"))

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
    global genai_client
    if not genai_client: return None

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

def build_final_message(ai_analysis: Dict[str, Any]) -> str:
    """Formata a mensagem de ENTRADA BRANCO PADRÃO e LIMPA (MODELO 1)."""
    confianca: float = ai_analysis.get("confianca", 0.0)
    justificativa: str = ai_analysis.get("justificativa", "Análise indisponível.")

    modo = "APRENDIZADOBRUTO" if PERCENTUAL_MINIMO_CONFIANCA == 0.0 else "PRODUÇÃO"

    return (
        f"🚨 **ENTRADA IMEDIATA NO BRANCO!** ⚪️\n\n"
        f"🎯 JOGO: Double JonBet\n"
        f"🔥 FOCO: BRANCO\n"
        f"📊 Confiança: `{confianca:.2f}%` ({modo})\n"
        f"🧠 Análise Gemini: _{justificativa}_\n\n"
        f"⚠️ **ESTRATÉGIA: G0 (ZERO GALES).**\n"
        f"💻 Site: Acessar Double"
    )[:4096]

def build_simple_placar(text_lower: str) -> Optional[str]:
    """Cria a mensagem de Placar Simples (GREEN/LOSS) (MODELOS 2 E 3)."""
    
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

    is_placar = "loss" in text_lower or "perda" in text_lower or "vitória" in text_lower or "✅" in text_lower or "🟢" in text_lower
    contains_entrada_palavras_aposta = "aposta" in text_lower or "entrar" in text_lower or "duplo" in text_lower

    
    # === AVALIAÇÃO DE PLACAR/RESULTADO ===
    if is_placar:
        # Rejeita placar se for misturado com GALE ou palavras de entrada (tipo: "VITÓRIA! Aposta de novo!")
        if contains_entrada_palavras_aposta or "gale" in text_lower:
            logging.info("Placar ignorado: Está misturado com sinais de entrada ou Gale.")
            return {"ok": True, "action": "ignored_mixed_placar"}
            
        final_placar_message = build_simple_placar(text_lower)
        if final_placar_message:
            await send_telegram_message(CANAL_DESTINO_ID, final_placar_message)
            logging.info("Placar simples enviado.")
            return {"ok": True, "action": "placar_sent"}


    # === AVALIAÇÃO DE ENTRADA BRANCO / G0 ===
    contains_branco = "branco" in text_lower or "⚪" in text or "⬜" in text
    contains_gale_ou_cores_mistas = "gale" in text_lower or "gales" in text_lower or "preto" in text_lower or "vermelho" in text_lower or "verde" in text_lower or "⚫" in text_lower or "🔴" in text_lower or "🟢" in text_lower

    if not contains_branco:
        logging.info("Sinal ignorado: Não contém a palavra/emoji 'BRANCO'.")
        return {"ok": True, "action": "ignored_not_branco"}

    if contains_gale_ou_cores_mistas:
        logging.info("Sinal ignorado: Contém BRANCO, mas também contém GALE ou CORES (não é G0 puro).")
        return {"ok": True, "action": "ignored_mixed_signal_not_g0"}
    
    # 4. ANÁLISE DE IA E COOLDOWN (APENAS PARA SINAIS DE ENTRADA)
    if not genai_client: return {"ok": True, "action": "ai_disabled_no_action"}
    
    ai_analysis = analyze_message_with_gemini(text)
    if not ai_analysis: return {"ok": True, "action": "ai_analysis_failed"}
    confianca_ia = ai_analysis.get("confianca", 0.0)

    if confianca_ia >= PERCENTUAL_MINIMO_CONFIANCA:
        current_time = time.time()
        if current_time - last_signal_time < COOLDOWN_SECONDS:
            logging.info(f"Sinal ignorado devido ao COOLDOWN.")
            return {"ok": True, "action": "ignored_cooldown"}
            
        final_message = build_final_message(ai_analysis) 
        await send_telegram_message(CANAL_DESTINO_ID, final_message)
        
        last_signal_time = current_time 
        
        logging.info(f"Sinal enviado! Confiança: {confianca_ia:.2f}%")
        return {"ok": True, "action": "signal_sent", "confidence": confianca_ia}
    else:
        logging.info(f"Sinal ignorado pelo filtro de confiança da IA: {confianca_ia:.2f}%")
        return {"ok": True, "action": "ignored_low_confidence", "confidence": confianca_ia}
