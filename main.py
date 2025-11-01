# main.py - Versão FINAL SEM GEMINI (Modo Destravado: Coleta de Dados)

import os
import sqlite3
import re
import logging
import time
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
import httpx 

# Importações do Gemini Removidas

# ====================================================================
# CONFIGURAÇÃO GERAL E LOGGING
# ====================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# VARIÁVEIS DE AMBIENTE (Lidas do Render)
BOT_TOKEN   = os.environ.get("BOT_TOKEN", "").strip()
WEBHOOK_TOKEN = os.environ.get("WEBHOOK_TOKEN", "default_secret_token").strip() 
# GEMINI_API_KEY Removida

# IDs dos Canais (Lidos de ENVs)
CANAL_ORIGEM_IDS_STR = os.environ.get("CANAL_ORIGEM_IDS", "-1003156785631").strip()
CANAL_DESTINO_ID_STR = os.environ.get("CANAL_DESTINO_ID", "-1002796105884").strip()
CANAL_FEEDBACK_ID_STR = os.environ.get("CANAL_FEEDBACK_ID", "-1002796105884").strip()

# Conversão dos IDs de canal
try:
    CANAL_ORIGEM_IDS = [int(id_.strip()) for id_ in CANAL_ORIGEM_IDS_STR.split(',') if id_.strip()]
    CANAL_DESTINO_ID = int(CANAL_DESTINO_ID_STR)
    CANAL_FEEDBACK_ID = int(CANAL_FEEDBACK_ID_STR)
except ValueError as e:
    logging.critical(f"ERRO: IDs de canais inválidos nas ENVs. Verifique: {e}")
    CANAL_ORIGEM_IDS, CANAL_DESTINO_ID, CANAL_FEEDBACK_ID = [], 0, 0 

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# ====================================================================
# CONFIGURAÇÃO DE APRENDIZADO E FILTRAGEM
# ====================================================================
MIN_JOGADAS_APRENDIZADO = 10
# MODO DESTRAVADO: Confiança mínima para enviar. 0.0 garante que envie para COLETAR DADOS.
PERCENTUAL_MINIMO_CONFIANCA = 0.0 

# Variável de estado global para armazenar o ÚLTIMO SINAL ENVIADO (Chave com Tempo)
LAST_SENT_SIGNAL = {"text": None, "timestamp": 0} 

# ====================================================================
# BANCO DE DADOS (SQLite) E PERSISTÊNCIA DE DISCO NO RENDER
# ====================================================================
DB_MOUNT_PATH = os.environ.get("DB_MOUNT_PATH", "/var/data") 
DB_NAME = os.path.join(DB_MOUNT_PATH, 'double_jonbet_data.db') 

def setup_db():
    os.makedirs(DB_MOUNT_PATH, exist_ok=True)
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sinais_performance (
        sinal_original TEXT PRIMARY KEY,
        jogadas_analisadas INTEGER DEFAULT 0,
        acertos_branco INTEGER DEFAULT 0
    )
    ''')
    conn.commit()
    logging.info(f"DB configurado em: {DB_NAME}")
    return conn, cursor

conn, cursor = setup_db()

def get_performance(sinal):
    """Retorna a performance, confiança e histórico (jogadas e acertos) de um sinal."""
    cursor.execute("SELECT jogadas_analisadas, acertos_branco FROM sinais_performance WHERE sinal_original = ?", (sinal,))
    data = cursor.fetchone()
    
    if data:
        analisadas, acertos = data
        if analisadas > 0:
            confianca = (acertos / analisadas) * 100
            # Retorna o histórico de jogadas/acertos 
            historico_string = f"JOGADAS: {analisadas}, ACERTOS BRANCO: {acertos}"
            return analisadas, confianca, historico_string
        return 0, 0.0, "JOGADAS: 0, ACERTOS BRANCO: 0"
    
    cursor.execute("INSERT OR IGNORE INTO sinais_performance (sinal_original) VALUES (?)", (sinal,))
    conn.commit()
    return 0, 0.0, "JOGADAS: 0, ACERTOS BRANCO: 0"

def atualizar_performance(sinal, is_win):
    """Atualiza o histórico de jogadas e acertos no banco de dados."""
    cursor.execute("SELECT jogadas_analisadas, acertos_branco FROM sinais_performance WHERE sinal_original = ?", (sinal,))
    data = cursor.fetchone()
    
    if not data:
        # Insere se o sinal não existe
        cursor.execute("INSERT INTO sinais_performance (sinal_original, jogadas_analisadas, acertos_branco) VALUES (?, 1, ?)", 
                       (sinal, 1 if is_win else 0))
    else:
        jogadas, acertos = data
        jogadas += 1
        acertos += 1 if is_win else 0
        cursor.execute("UPDATE sinais_performance SET jogadas_analisadas = ?, acertos_branco = ? WHERE sinal_original = ?", 
                       (jogadas, acertos, sinal))
                       
    conn.commit()
    logging.info(f"Performance atualizada para '{sinal}'. Win: {is_win}.")

# ----------------------------------------------------
# FUNÇÃO DE DECISÃO DE ENVIO (SIMPLIFICADA)
# ----------------------------------------------------

async def deve_enviar_sinal(sinal: str):
    """Lógica original de aprendizado estatístico."""
    
    analisadas, confianca_estatistica, historico_db = get_performance(sinal)
    
    # 1. Modo APRENDIZADO (Envio garantido para coletar dados)
    if analisadas < MIN_JOGADAS_APRENDIZADO: 
        return True, "APRENDIZADO_BRUTO", confianca_estatistica
    
    # 2. Envio pelo Fallback Estatístico
    # Como o PERCENTUAL_MINIMO_CONFIANCA está em 0.0, ele enviará sempre
    if confianca_estatistica >= PERCENTUAL_MINIMO_CONFIANCA:
        return True, "ESTATISTICA_FALLBACK", confianca_estatistica
        
    return False, "BLOQUEIO_ZERO_CONFIANÇA", confianca_estatistica


async def tg_send_message(chat_id: int, text: str):
    """Envia a mensagem para o Telegram usando httpx."""
    
    if not BOT_TOKEN:
        logging.critical("BOT_TOKEN não configurado. Não é possível enviar mensagem.")
        return False
        
    url = f"{TELEGRAM_API}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    
    try:
        # Usa httpx para requisições assíncronas (async)
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status() 
            return response.json().get('ok', False)
    except httpx.HTTPStatusError as e:
        logging.error(f"Erro HTTP ao enviar Telegram: {e.response.text}")
    except Exception as e:
        logging.error(f"Erro ao enviar Telegram: {e}")
    return False


# ====================================================================
# APLICAÇÃO FASTAPI (Webhook)
# ====================================================================

app = FastAPI(title="Double JonBet Webhook", version="1.0")

@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    global LAST_SENT_SIGNAL
    
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid Webhook Token")

    try:
        data = await request.json()
    except Exception:
        return {"ok": True, "skipped": "invalid_json"}

    message = data.get("message") or data.get("channel_post") or data.get("edited_channel_post") or {}
    chat = message.get("chat") or {}
    
    if not message or not chat:
        return {"ok": True, "skipped": "no_message"}

    chat_id = chat.get("id", 0)
    text = (message.get("text") or message.get("caption") or "").strip()
    
    if not text or not chat_id:
        return {"ok": True, "skipped": "no_text_or_chat"}

    logging.info(f"Mensagem recebida de Chat ID: {chat_id}")
    
    # --- 2. LÓGICA DE ROTEAMENTO E PROCESSAMENTO ---
    
    # A) PROCESSAR SINAL DE ORIGEM
    if chat_id in CANAL_ORIGEM_IDS:
        logging.info("Mensagem roteada para PROCESSAMENTO DE SINAL.")
        
        # Filtra SÓ se a mensagem for um sinal com foco em BRANCO (⚪)
        if "branco" not in text.lower() and "⚪" not in text:
            logging.info("Sinal ignorado: Não foca em BRANCO.")
            return {"ok": True, "action": "ignored_not_branco"}
        
        # OBTÉM A HORA E MINUTO ATUAIS para a chave de aprendizado
        hora_minuto_atual = datetime.now().strftime("%H:%M") 
        
        sinal_limpo_texto = re.sub(r'#[0-9]+', '', text).strip()
        
        # CHAVE DE APRENDIZADO COMPLETA (texto + tempo)
        sinal_com_tempo = f"{sinal_limpo_texto} | {hora_minuto_atual}"
        
        deve_enviar, modo, confianca = await deve_enviar_sinal(sinal_com_tempo) # Agora é assíncrona
        
        if deve_enviar:
            sinal_convertido = (
                f"⚠️ **SINAL EXCLUSIVO BRANCO ({modo})** ⚠️\n\n"
                f"🎯 JOGO: **Double JonBet**\n"
                f"🔥 FOCO TOTAL NO **BRANCO** 🔥\n\n"
                f"⏰ Entrada às: **{hora_minuto_atual}**\n"
                f"📊 Confiança: `{confianca:.2f}%` (Modo: {modo})\n"
                f"🔔 Sinal Original: {sinal_limpo_texto}"
            )
            
            await tg_send_message(CANAL_DESTINO_ID, sinal_convertido)
            
            # Registra o último sinal ENVIADO com a CHAVE COMPLETA (texto + tempo)
            LAST_SENT_SIGNAL["text"] = sinal_com_tempo 
            LAST_SENT_SIGNAL["timestamp"] = time.time()
            logging.warning(f"Sinal ENVIADO: '{sinal_com_tempo}'. Esperando feedback em {CANAL_FEEDBACK_ID}.")
        
        else:
            logging.info(f"Sinal IGNORADO: '{sinal_com_tempo}'. Confiança: {confianca:.2f}% (Motivo: {modo})")
            
        return {"ok": True, "action": "processed_sinal"}

    # B) PROCESSAR FEEDBACK (WIN/LOSS)
    elif chat_id == CANAL_FEEDBACK_ID:
        logging.info("Mensagem roteada para PROCESSAMENTO DE FEEDBACK.")
        
        if LAST_SENT_SIGNAL["text"] is None or (time.time() - LAST_SENT_SIGNAL["timestamp"]) > 3600: # Ignora feedback antigo (> 1h)
            logging.warning("Feedback recebido, mas nenhum sinal recente pendente ou sinal expirado.")
            return {"ok": True, "action": "feedback_ignored_no_signal"}

        feedback_text = text.strip().upper()
        
        # Lógica de Feedback:
        is_feedback_received = any(k in feedback_text for k in ["VITÓRIA", "GREEN", "LOSS", "RED", "PERDEU", "GANHOU", "GANHO"])
        is_win_branco = "BRANCO" in feedback_text or "⚪" in text
        
        is_win = False
        
        if is_win_branco:
            is_win = True
        elif is_feedback_received:
            # Qualquer outro resultado de rodada é LOSS para o sistema do BRANCO
            is_win = False
            
        if is_win_branco or is_feedback_received:
            sinal_para_atualizar = LAST_SENT_SIGNAL["text"]
            
            atualizar_performance(sinal_para_atualizar, is_win)
            
            sinal_original_apenas_texto = sinal_para_atualizar.split('|')[0].strip()
            
            if is_win:
                resultado_msg = f"✅ **WIN BRANCO!**\nSinal: `{sinal_original_apenas_texto}`. Feedback: `{feedback_text}`"
            else:
                resultado_msg = f"❌ **LOSS BRANCO.**\nSinal: `{sinal_original_apenas_texto}`. Feedback: `{feedback_text}`"
                
            await tg_send_message(CANAL_DESTINO_ID, resultado_msg)
            
            LAST_SENT_SIGNAL["text"] = None 
            logging.info("Estado de feedback limpo. Pronto para o próximo sinal.")
            
            return {"ok": True, "action": "feedback_processed", "outcome": "WIN" if is_win else "LOSS"}
        
        else:
            logging.info("Feedback recebido, mas não é um WIN/LOSS reconhecido.")
            return {"ok": True, "action": "feedback_unrecognized"}
            
    # C) CHAT NÃO CONFIGURADO
    else:
        logging.info(f"Mensagem de chat {chat_id} ignorada (Não é origem ou feedback).")
        return {"ok": True, "action": "chat_ignored"}

# ----------------------------------------
# ROTA DE STATUS (Health Check)
# ----------------------------------------
@app.get("/")
async def health_check():
    return {"status": "running", "service": "Double JonBet Webhook", "db_path": DB_NAME}
