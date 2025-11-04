# -*- coding: utf-8 -*-
# ✅ JonBet Auto Bot - Conversor de sinais (Versão Final e Máxima Robustez - Com Trava de ID)
# Esta versão é a solução lógica final:
# 1. Filtro agressivo para novo formato (sem entrada imediata e com emojis).
# 2. Conversão forçada para o BRANCO (⚪️).
# 3. Trava dupla (entry_active + processed_ids) para minimizar duplicação durante instabilidade do servidor.

import os
import json
import time
import logging
import re
import unicodedata
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
import httpx
from statistics import median

# ===================== CONFIG =====================
# Variáveis de Ambiente. Certifique-se de que estão definidas no Render!
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===================== APRENDIZADO =====================
learn_state = {
    "last_white_ts": None,
    "white_gaps": [],
    "stones_since_last_white": 0,
    "stones_gaps": [],
    "entry_active": False, # Trava de fluxo 1:1
    "processed_ids": []     # Armazena IDs de mensagens processadas (Trava por ID)
}

def _save_learn():
    """Salva o estado atual do aprendizado (gaps/pedras/lock) no arquivo."""
    try:
        with open(LEARN_PATH, "w") as f:
            json.dump(learn_state, f)
    except Exception as e:
        logging.error(f"Erro ao salvar aprendizado: {e}")

def _load_learn():
    """Carrega o estado do aprendizado ao iniciar o bot."""
    global learn_state
    try:
        if os.path.exists(LEARN_PATH):
            with open(LEARN_PATH, "r") as f:
                loaded_state = json.load(f)
                learn_state.update(loaded_state) 
                # Garante que a lista de IDs seja limitada a 100
                learn_state["processed_ids"] = learn_state.get("processed_ids", [])[-100:]
    except Exception:
        pass

_load_learn()

# ===================== FUNÇÕES DE UTILIDADE =====================
def _strip_accents(s: str) -> str:
    """
    Remove acentos, emojis e caracteres especiais de uma string, deixando apenas
    letras, números e espaços, para garantir o reconhecimento das palavras-chave.
    """
    nfkd_form = unicodedata.normalize('NFKD', s)
    only_ascii = nfkd_form.encode('ascii', 'ignore').decode('utf-8')
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', only_ascii)
    return re.sub(r'\s+', ' ', cleaned).strip()

def _append_bounded(lst, val, maxlen=200):
    """Adiciona valor à lista, mantendo o tamanho máximo."""
    lst.append(val)
    if len(lst) > maxlen:
        del lst[:len(lst)-maxlen]

def extract_message(data: dict):
    """Extrai informações relevantes da requisição do Telegram."""
    msg = data.get("message") or data.get("channel_post") or {}
    return {
        "chat": msg.get("chat", {}),
        "text": msg.get("text") or "", 
        "message_id": msg.get("message_id")
    }

async def send_telegram_message(chat_id: str, text: str):
    """Envia uma mensagem formatada via API do Telegram."""
    async with httpx.AsyncClient() as client:
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        try:
            r = await client.post(SEND_MESSAGE_URL, json=payload, timeout=15)
            r.raise_for_status()
        except Exception as e:
            logging.error(f"Erro ao enviar mensagem: {e}")

# ===================== FUNÇÕES DE LÓGICA =====================

def is_entrada_confirmada(text: str) -> bool:
    """
    <<< FILTRO FLEXÍVEL - RECONHECE O NOVO FORMATO >>>
    Só retorna True se a mensagem for uma entrada, ignorando resultados.
    """
    t_cleaned = _strip_accents(text).lower()
    
    # Verifica as palavras-chave essenciais do novo formato (sem entrada imediata)
    is_double_blaze = "double blaze" in t_cleaned
    is_entry_format = "entrada sera para" in t_cleaned 
    mentions_gale = "gale" in t_cleaned
    
    # Critério MAIS IMPORTANTE: Deve IGNORAR resultados.
    is_not_result = not any(w in t_cleaned for w in ["win", "loss", "derrota"])

    return is_double_blaze and is_entry_format and mentions_gale and is_not_result

def build_entry_message(text_original: str) -> str:
    """
    Constrói a mensagem de entrada, forçando o sinal para o BRANCO (⚪️).
    A 'Entrar após' é colocada como '?' para representar a conversão de entrada imediata.
    """
    
    return (
        "🚨 **CONVERSÃO: ENTRADA IMEDIATA NO BRANCO!** ⚪️\n\n"
        f"Apostar no **Branco** ⚪️\n"
        f"Entrar após: ⚪️ ?\n\n"
        "🎰 Jogo: Double - JonBet\n"
        "💻 Site: Acessar Double"
    )

def classificar_resultado(txt: str) -> Optional[str]:
    """
    Classifica a mensagem como GREEN, LOSS ou None (ignorável) com MÁXIMA RIGIDEZ.
    """
    t_cleaned = _strip_accents(txt).lower()
    
    # MÁXIMA RIGIDEZ PARA GREEN: WIN/Vitória E BRANCO/⚪
    if ("win" in t_cleaned or "vitoria" in t_cleaned) and ("branco" in t_cleaned or "⚪" in txt):
        return "GREEN_VALIDO"
    
    # MÁXIMA RIGIDEZ PARA LOSS (Cobre Derrota e Wins de outras cores)
    if "loss" in t_cleaned or "derrota" in t_cleaned or "❌" in txt or \
       (("win" in t_cleaned or "vitoria" in t_cleaned) and any(c in txt for c in ["⚫", "🔴", "🟢"])):
        return "LOSS"
        
    return None

def build_result_message(resultado_status: str) -> str:
    """
    Gera a mensagem de resultado formatada com dados de aprendizado e STATUS SIMPLIFICADO.
    """
    stones = learn_state.get("stones_since_last_white", 0)
    try:
        med_stones = int(median(learn_state["stones_gaps"])) if learn_state["stones_gaps"] else 0
    except Exception:
        med_stones = 0
        
    if resultado_status == "GREEN_VALIDO":
        status_msg = "✅ **GREEN!**"
    else: 
        status_msg = "❌ **LOSS** 😥"
        
    return (
        f"Resultado: {status_msg}\n\n"
        f"🪙 *Distância entre brancos:* {stones} pedras (mediana: {med_stones})"
    )


# ===================== WEBHOOK =====================
@app.get("/")
def root():
    return {"status": "ok", "service": "JonBet - Branco Automático (FINAL ID LOCK)"}

@app.post(f"/webhook/{{webhook_token}}")
async def webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token incorreto")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON inválido")

    msg = extract_message(data)
    chat_id = str(msg.get("chat", {}).get("id"))
    text = (msg.get("text") or "").strip()
    message_id = msg.get("message_id")

    # IGNORA: Mensagens já processadas (Trava por ID - Defesa contra duplicação)
    if message_id and message_id in learn_state["processed_ids"]:
        logging.info(f"Ignorando ID duplicado: {message_id}")
        return {"ok": True, "action": "ignored_duplicate_id"}

    # Ignora mensagens do próprio canal de destino e de fontes não autorizadas
    if chat_id == CANAL_DESTINO_ID or chat_id not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored_channel"}
    
    # Adiciona o ID à lista de processados
    if message_id:
        learn_state["processed_ids"].append(message_id)
        learn_state["processed_ids"] = learn_state["processed_ids"][-100:]
        _save_learn() # Salva a trava de ID imediatamente

    # TENTA CLASSIFICAR RESULTADO PRIMEIRO
    resultado = classificar_resultado(text)
    
    # ========================== BLOCO DE RESULTADO (UNLOCK) ==========================
    if resultado in ["GREEN_VALIDO", "LOSS"]:
        
        # DESTRAVA, independentemente do estado anterior
        if learn_state.get("entry_active"):
            learn_state["entry_active"] = False 
            
        if resultado == "GREEN_VALIDO":
            now = time.time()
            if learn_state.get("last_white_ts"):
                gap = now - float(learn_state["last_white_ts"])
                _append_bounded(learn_state["white_gaps"], gap, 200)
                _append_bounded(learn_state["stones_gaps"], learn_state["stones_since_last_white"], 200)
                
            learn_state["last_white_ts"] = now
            learn_state["stones_since_last_white"] = 0 

        msg_text = build_result_message(resultado) 
        await send_telegram_message(CANAL_DESTINO_ID, msg_text)
        _save_learn() # Salva o estado DESTRAVADO
        return {"ok": True, "action": f"result_logged_and_unlocked ({resultado})"}
        
    # ========================== BLOCO DE ENTRADA (LOCK) ==========================
    if is_entrada_confirmada(text):
        
        # Trava: IGNORA se já houver um sinal ativo
        if learn_state.get("entry_active"):
            return {"ok": True, "action": "ignored_entry_active_lock"}

        # LOCK: TRAVA o fluxo
        learn_state["entry_active"] = True 
        
        learn_state["stones_since_last_white"] = learn_state.get("stones_since_last_white", 0) + 1
        msg_text = build_entry_message(text)
        
        await send_telegram_message(CANAL_DESTINO_ID, msg_text)
        _save_learn() # Salva o estado TRAVADO
        return {"ok": True, "action": "entry_converted_and_locked"}

    # ========================== BLOCO DE IGNORAR (TUDO MAIS) ==========================
    _save_learn() 
    return {"ok": True, "action": "ignored_non_entry_non_result"}
