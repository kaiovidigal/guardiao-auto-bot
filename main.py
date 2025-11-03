# -*- coding: utf-8 -*-
# ✅ JonBet Auto Bot - Conversor de sinais (Conversão Total para o BRANCO)
# REGRAS:
# 1. FILTRO ESTRITO: Só aceita sinais no padrão 'Apostar no [cor] e ⚪ branco como proteção! [x] Gales'.
# 2. CONVERSÃO: Converte o sinal filtrado para uma entrada simples no BRANCO.
# 3. RESULTADO: Só aceita "GREEN no BRANCO" como GREEN. Outros resultados são LOSS.
# 4. CONTROLE DE FLUXO: Trava (Lock) 1:1 ativada para evitar duplicação.

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
app.state.processed_entries = set()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===================== APRENDIZADO =====================
learn_state = {
    "last_white_ts": None,
    "white_gaps": [],
    "stones_since_last_white": 0,
    "stones_gaps": [],
    "entry_active": False # Trava de fluxo 1:1
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
    except Exception:
        pass

_load_learn()

# ===================== FUNÇÕES DE UTILIDADE =====================
def _strip_accents(s: str) -> str:
    """Remove acentos de uma string para facilitar a comparação."""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

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
    <<< FILTRO ESTRITO >>>
    Só retorna True se a mensagem contiver as palavras-chave do padrão de aposta mista
    (entrada, branco como proteção, e gales). Ignora todo o resto.
    """
    t = _strip_accents(text.lower())
    
    # Critério 1: Deve ser um sinal de aposta (entrada/apostar)
    is_entry_format = any(x in t for x in ["entrada confirmada", "apostar no"])

    # Critério 2: Deve ser uma aposta mista/proteção focada no BRANCO
    is_mixed_bet = ("branco como protecao" in t or "branco como proteção" in t)
    
    # Critério 3: Deve mencionar Gales (como nos exemplos do usuário)
    mentions_gale = ("gale" in t or "gales" in t)

    # Só aceita se atender a todos os critérios.
    return is_entry_format and is_mixed_bet and mentions_gale

def build_entry_message(text_original: str) -> str:
    """
    Constrói a mensagem de entrada, forçando o sinal para o BRANCO (⚪️).
    """
    m = re.search(r"(\d{1,2})", text_original)
    num_alvo = m.group(1) if m else "?"
    
    return (
        "🚨 **CONVERSÃO: ENTRADA IMEDIATA NO BRANCO!** ⚪️\n\n"
        f"Apostar no **Branco** ⚪️\n"
        f"Entrar após: ⚪️ {num_alvo}\n\n"
        "🎰 Jogo: Double - JonBet\n"
        "💻 Site: Acessar Double"
    )

def classificar_resultado(txt: str) -> Optional[str]:
    """
    Classifica a mensagem como GREEN, LOSS ou None (ignorável).
    APENAS VITÓRIAS NO BRANCO são GREEN.
    """
    t = _strip_accents(txt.lower())
    
    # BLOCO 1: DETECTA VITÓRIA (GREEN)
    if any(w in t for w in ["vitoria", "vitória", "acertamos", "acerto", "green"]):
        # ✅ VERIFICA RIGOROSA: SÓ ACEITA SE HOUVER A PALAVRA 'BRANCO' OU O SÍMBOLO '⚪'
        if "branco" in t or "⚪" in txt:
            return "GREEN_VALIDO"
        
        # Se for vitória (preto/verde), classifica como LOSS para a nossa estratégia
        return "LOSS" 
    
    # BLOCO 2: DETECTA LOSS EXPLÍCITO
    if any(w in t for w in ["loss", "derrota", "nao deu", "não deu", "falhou"]):
        return "LOSS"
        
    return None

def build_result_message(resultado_txt: str) -> str:
    """Gera a mensagem de resultado formatada com dados de aprendizado."""
    stones = learn_state.get("stones_since_last_white", 0)
    try:
        med_stones = int(median(learn_state["stones_gaps"])) if learn_state["stones_gaps"] else 0
    except Exception:
        med_stones = 0
        
    return f"Resultado: {resultado_txt}\n\n🪙 *Distância entre brancos:* {stones} pedras (mediana: {med_stones})"


# ===================== WEBHOOK =====================
@app.get("/")
def root():
    return {"status": "ok", "service": "JonBet - Branco Automático (Filtro Estrito e Trava 1:1 Ativa)"}

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

    # Ignora mensagens do próprio canal de destino e de fontes não autorizadas
    if chat_id == CANAL_DESTINO_ID or chat_id not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored_channel"}

    # TENTA CLASSIFICAR RESULTADO PRIMEIRO
    resultado = classificar_resultado(text)
    
    # ========================== BLOCO DE RESULTADO (UNLOCK) ==========================
    if resultado in ["GREEN_VALIDO", "LOSS"]:
        # Se um resultado chegou, DESTRAVA o fluxo de entrada.
        if learn_state.get("entry_active"):
            learn_state["entry_active"] = False # <--- DESTRAVA A ENTRADA
            
        if resultado == "GREEN_VALIDO":
            now = time.time()
            if learn_state.get("last_white_ts"):
                gap = now - float(learn_state["last_white_ts"])
                _append_bounded(learn_state["white_gaps"], gap, 200)
                _append_bounded(learn_state["stones_gaps"], learn_state["stones_since_last_white"], 200)
                
            learn_state["last_white_ts"] = now
            learn_state["stones_since_last_white"] = 0 # Zera a contagem de pedras (saiu branco)

            msg_text = build_result_message("✅ **GREEN no BRANCO!** ⚪️")
        else: # Resultado é LOSS (inclui vitórias preto/verde e derrotas explícitas)
            msg_text = build_result_message("❌ **LOSS** 😥")

        await send_telegram_message(CANAL_DESTINO_ID, msg_text)
        _save_learn()
        return {"ok": True, "action": f"result_logged_and_unlocked ({resultado})"}
        
    # ========================== BLOCO DE ENTRADA (LOCK) ==========================
    if is_entrada_confirmada(text):
        
        # Trava: IGNORA se já houver um sinal ativo
        if learn_state.get("entry_active"):
            return {"ok": True, "action": "ignored_entry_active_lock"}

        # LOCK: Se não houver sinal ativo, TRAVA o fluxo para esperar o resultado
        learn_state["entry_active"] = True # <--- TRAVA A ENTRADA
        
        # Executa o envio e aumenta o contador
        learn_state["stones_since_last_white"] = learn_state.get("stones_since_last_white", 0) + 1
        msg_text = build_entry_message(text)
        
        await send_telegram_message(CANAL_DESTINO_ID, msg_text)
        _save_learn()
        return {"ok": True, "action": "entry_converted_and_locked"}

    # ========================== BLOCO DE IGNORAR (TUDO MAIS) ==========================
    _save_learn() 
    return {"ok": True, "action": "ignored"}
