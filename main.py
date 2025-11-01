# main.py - Versão FINAL COM APRENDIZADO e Persistência de Disco (WEBHOOK/FASTAPI)

import os
import sqlite3
import re
import logging
import time
from fastapi import FastAPI, Request, HTTPException
import httpx # Necessário para enviar mensagens assíncronas ao Telegram

# ====================================================================
# CONFIGURAÇÃO GERAL E LOGGING
# ====================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# VARIÁVEIS DE AMBIENTE (Lidas do Render)
# ATENÇÃO: API_ID/API_HASH NÃO SÃO MAIS NECESSÁRIOS.
BOT_TOKEN   = os.environ.get("BOT_TOKEN", "").strip()
WEBHOOK_TOKEN = os.environ.get("WEBHOOK_TOKEN", "default_secret_token").strip() # Novo token de segurança

# IDs dos Canais (Lidos de ENVs para maior flexibilidade no Webhook)
CANAL_ORIGEM_IDS_STR = os.environ.get("CANAL_ORIGEM_IDS", "-1003156785631,-1009998887776").strip()
CANAL_DESTINO_ID_STR = os.environ.get("CANAL_DESTINO_ID", "-1002796105884").strip()
CANAL_FEEDBACK_ID_STR = os.environ.get("CANAL_FEEDBACK_ID", "-1009990001112").strip()

# Converte IDs de string para lista de inteiros
try:
    CANAL_ORIGEM_IDS = [int(id_.strip()) for id_ in CANAL_ORIGEM_IDS_STR.split(',') if id_.strip()]
    CANAL_DESTINO_ID = int(CANAL_DESTINO_ID_STR)
    CANAL_FEEDBACK_ID = int(CANAL_FEEDBACK_ID_STR)
except ValueError as e:
    logging.critical(f"ERRO: IDs de canais inválidos nas ENVs. Verifique: {e}")
    CANAL_ORIGEM_IDS, CANAL_DESTINO_ID, CANAL_FEEDBACK_ID = [], 0, 0 # Valores de falha

# URL base da API do Telegram para envio de mensagens
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# ====================================================================
# CONFIGURAÇÃO DE APRENDIZADO E FILTRAGEM
# ====================================================================
MIN_JOGADAS_APRENDIZADO = 10
PERCENTUAL_MINIMO_CONFIANCA = 79.0 

# Variável de estado global para armazenar o ÚLTIMO SINAL ENVIADO (MANTIDA FORA DO DB PARA SIMPLICIDADE)
LAST_SENT_SIGNAL = {"text": None, "timestamp": 0} 

# ====================================================================
# BANCO DE DADOS (SQLite) E PERSISTÊNCIA DE DISCO NO RENDER
# ====================================================================
DB_MOUNT_PATH = os.environ.get("DB_MOUNT_PATH", "/var/data") 
DB_NAME = os.path.join(DB_MOUNT_PATH, 'double_jonbet_data.db') 

def setup_db():
    # ... (A lógica de setup_db() permanece a mesma)
    os.makedirs(DB_MOUNT_PATH, exist_ok=True)
    # ATENÇÃO: check_same_thread=False é crucial para FastAPI/Async
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
    # ... (A lógica de get_performance() permanece a mesma)
    cursor.execute("SELECT jogadas_analisadas, acertos_branco FROM sinais_performance WHERE sinal_original = ?", (sinal,))
    data = cursor.fetchone()
    if data:
        analisadas, acertos = data
        if analisadas > 0:
            confianca = (acertos / analisadas) * 100
            return analisadas, confianca
        return analisadas, 0.0
    
    cursor.execute("INSERT OR IGNORE INTO sinais_performance (sinal_original) VALUES (?)", (sinal,))
    conn.commit()
    return 0, 0.0

def deve_enviar_sinal(sinal):
    # ... (A lógica de deve_enviar_sinal() permanece a mesma)
    analisadas, confianca = get_performance(sinal)
    
    if analisadas < MIN_JOGADAS_APRENDIZADO: 
        return True, "APRENDIZADO"

    if confianca > PERCENTUAL_MINIMO_CONFIANCA:
        return True, "CONFIANÇA"
        
    return False, "BLOQUEIO"

def atualizar_performance(sinal, is_win):
    # ... (A lógica de atualizar_performance() permanece a mesma)
    novo_acerto = 1 if is_win else 0
    
    cursor.execute("""
    UPDATE sinais_performance SET 
        jogadas_analisadas = jogadas_analisadas + 1, 
        acertos_branco = acertos_branco + ?
    WHERE sinal_original = ?
    """, (novo_acerto, sinal))
    
    conn.commit()
    logging.info(f"DB Atualizado: {sinal} - WIN BRANCO: {is_win}")

# ====================================================================
# FUNÇÃO DE ENVIO DE MENSAGENS TELEGRAM (AGORA HTTPX)
# ====================================================================

async def tg_send_message(chat_id: int, text: str):
    """Envia uma mensagem assíncrona usando a API HTTP do Telegram."""
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

app = FastAPI(title="Double JonBet IA Webhook", version="1.0")

# ----------------------------------------
# ROTA PRINCIPAL DO WEBHOOK
# ----------------------------------------
@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    global LAST_SENT_SIGNAL
    
    # 1. VALIDAÇÃO DE SEGURANÇA
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid Webhook Token")

    try:
        data = await request.json()
    except Exception:
        logging.warning("Received invalid JSON data.")
        return {"ok": True}

    # Extrai a mensagem (pode vir como 'message' ou 'channel_post')
    message = data.get("message") or data.get("channel_post") or data.get("edited_channel_post") or {}
    chat = message.get("chat") or {}
    
    # Validação mínima
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
        
        sinal_limpo = re.sub(r'#[0-9]+', '', text).strip() # Limpa o sinal
        deve_enviar, modo = deve_enviar_sinal(sinal_limpo)
        analisadas, confianca = get_performance(sinal_limpo)
        
        if deve_enviar:
            sinal_convertido = (
                f"⚠️ **SINAL EXCLUSIVO BRANCO ({modo})** ⚠️\n\n"
                f"🎯 JOGO: **Double JonBet**\n"
                f"🔥 FOCO TOTAL NO **BRANCO** 🔥\n\n"
                f"📊 Confiança: `{confianca:.2f}%` (Base: {analisadas} análises)\n"
                f"🔔 Sinal Original: {sinal_limpo}"
            )
            
            # Envia para o canal de destino
            await tg_send_message(CANAL_DESTINO_ID, sinal_convertido)
            
            # Registra o último sinal enviado para o sistema de feedback
            LAST_SENT_SIGNAL["text"] = sinal_limpo
            LAST_SENT_SIGNAL["timestamp"] = time.time()
            logging.warning(f"Sinal ENVIADO: '{sinal_limpo}'. Esperando feedback em {CANAL_FEEDBACK_ID}.")
        
        else:
            logging.info(f"Sinal IGNORADO: '{sinal_limpo}'. Confiança: {confianca:.2f}%")
            
        return {"ok": True, "action": "processed_sinal"}

    # B) PROCESSAR FEEDBACK (WIN/LOSS)
    elif chat_id == CANAL_FEEDBACK_ID:
        logging.info("Mensagem roteada para PROCESSAMENTO DE FEEDBACK.")
        
        feedback_text = text.strip().upper()
        
        if LAST_SENT_SIGNAL["text"] is None:
            logging.warning("Feedback recebido, mas nenhum sinal recente pendente.")
            return {"ok": True, "action": "feedback_ignored_no_signal"}

        # Lógica para detectar WIN ou LOSS
        is_win = "WIN" in feedback_text or "GREEN" in feedback_text
        is_loss = "LOSS" in feedback_text or "RED" in feedback_text or "NO WIN" in feedback_text
        
        if is_win or is_loss:
            sinal_para_atualizar = LAST_SENT_SIGNAL["text"]
            
            # 1. Atualiza o DB
            atualizar_performance(sinal_para_atualizar, is_win)
            
            # 2. Envia a confirmação para o canal de destino
            if is_win:
                resultado_msg = f"✅ **WIN BRANCO!** Feedback: `{feedback_text}`"
            else:
                resultado_msg = f"❌ **LOSS BRANCO.** Feedback: `{feedback_text}`"
                
            await tg_send_message(CANAL_DESTINO_ID, resultado_msg)
            
            # 3. Limpa o estado
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
    return {"status": "running", "service": "Double JonBet Webhook IA", "db_path": DB_NAME}
