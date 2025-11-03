# -*- coding: utf-8 -*-
# ✅ JonBet Auto Bot - Conversor de sinais (modo teste/debug)
# Modo: Aprendizado ativo + fluxo contínuo (sem gale, sem travas)

import os
import json
import time
import logging
import re
import unicodedata
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
import httpx
from statistics import median

# ===================== CONFIG =====================
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
    "entry_active": False # <--- NOVA VARIÁVEL DE ESTADO
}

def _save_learn():
    try:
        with open(LEARN_PATH, "w") as f:
            json.dump(learn_state, f)
    except Exception as e:
        logging.error(f"Erro ao salvar aprendizado: {e}")

def _load_learn():
    global learn_state
    try:
        if os.path.exists(LEARN_PATH):
            with open(LEARN_PATH, "r") as f:
                loaded_state = json.load(f)
                # Mantém a nova variável de estado se não existir no arquivo (primeira execução)
                learn_state.update(loaded_state) 
    except Exception:
        pass

_load_learn()

# ===================== FUNÇÕES =====================
def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def _append_bounded(lst, val, maxlen=200):
    lst.append(val)
    if len(lst) > maxlen:
        del lst[:len(lst)-maxlen]

def extract_message(data: dict):
    msg = data.get("message") or data.get("channel_post") or {}
    return {
        "chat": msg.get("chat", {}),
        "text": msg.get("text") or "",
        "message_id": msg.get("message_id")
    }

async def send_telegram_message(chat_id: str, text: str):
    async with httpx.AsyncClient() as client:
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        try:
            r = await client.post(SEND_MESSAGE_URL, json=payload, timeout=15)
            r.raise_for_status()
        except Exception as e:
            logging.error(f"Erro ao enviar mensagem: {e}")

def is_entrada_confirmada(text: str) -> bool:
    t = _strip_accents(text.lower())
    return (
        "entrada confirmada" in t
        or "apostar no" in t
        or "entrar apos" in t
        or "🎰 jogo" in t
    )

def ignorar_gale(text: str) -> bool:
    t = _strip_accents(text.lower())
    return any(x in t for x in ["g1", "g2", "vw", "protecao", "proteção", "⚠️"])

def classificar_resultado(txt: str) -> Optional[str]:
    t = _strip_accents(txt.lower())
    
    # 1. Padrão exato (visto no print) para GREEN no BRANCO (sem ser proteção)
    if "green no branco" in t and not ignorar_gale(txt):
        return "GREEN_VALIDO"
    
    # 2. Padrão para LOSS (sem ser proteção)
    if "loss" in t and not ignorar_gale(txt):
        return "LOSS"
        
    # Padrões originais de segurança:
    if any(w in t for w in ["vitoria", "vitória", "acertamos", "acerto"]) and "branco" in t:
        return "GREEN_VALIDO"
    if any(w in t for w in ["derrota", "nao deu", "não deu", "falhou"]):
        return "LOSS"
        
    return None

def build_entry_message(num_alvo: str) -> str:
    return (
        "✅ Entrada confirmada!\n"
        "Apostar no branco ⚪️\n"
        f"Entrar após: ⚪️ {num_alvo}\n"
        "🎰 Jogo: Double - JonBet\n"
        "💻 Site: Acessar Double"
    )

def build_result_message(resultado_txt: str) -> str:
    stones = learn_state.get("stones_since_last_white", 0)
    try:
        med_stones = int(median(learn_state["stones_gaps"])) if learn_state["stones_gaps"] else 0
    except Exception:
        med_stones = 0
        
    return f"{resultado_txt}\n\n🪙 *Distância entre brancos:* {stones} pedras (mediana: {med_stones})"


# ===================== WEBHOOK (COM DEBUG) =====================
@app.get("/")
def root():
    return {"status": "ok", "service": "JonBet - Branco Automático (Modo Teste)"}

@app.post(f"/webhook/{{webhook_token}}")
async def webhook(webhook_token: str, request: Request):
    print("🚀 [DEBUG] Webhook acionado >>>")

    if webhook_token != WEBHOOK_TOKEN:
        print("❌ [DEBUG] Token incorreto recebido:", webhook_token)
        raise HTTPException(status_code=403, detail="Token incorreto")

    try:
        data = await request.json()
        print("📦 [DEBUG] JSON recebido bruto:", json.dumps(data, ensure_ascii=False))
    except Exception as e:
        print("❌ [DEBUG] Erro ao ler JSON:", e)
        raise HTTPException(status_code=400, detail="JSON inválido")

    msg = extract_message(data)
    chat_id = str(msg.get("chat", {}).get("id"))
    text = (msg.get("text") or "").strip()

    print("💬 [DEBUG] chat_id:", chat_id)
    print("📝 [DEBUG] Texto recebido:", text)

    if chat_id not in CANAL_ORIGEM_IDS:
        print("⚠️ [DEBUG] Ignorado: Canal não autorizado ->", chat_id)
        return {"ok": True, "action": "ignored_source"}

    # TENTA CLASSIFICAR RESULTADO PRIMEIRO
    resultado = classificar_resultado(text)
    print("🔍 [DEBUG] Resultado classificado:", resultado)
    
    entry_is_active = learn_state.get("entry_active", False) # Verifica se há um sinal ativo
    
    # ========================== BLOCO DE RESULTADO ==========================
    if resultado == "GREEN_VALIDO" or resultado == "LOSS":
        
        if not entry_is_active:
            # IGNORA o resultado se NENHUM SINAL DE ENTRADA foi postado.
            print(f"⚠️ [DEBUG] {resultado} ignorado: Nenhuma entrada ativa registrada para este resultado.")
            _save_learn() 
            return {"ok": True, "action": "result_ignored_no_entry"}
            
        # O SINAL ESTÁ ATIVO, ENTÃO PROCESSAMOS O RESULTADO:
        learn_state["entry_active"] = False # RESETA O ESTADO: um resultado finalizou a entrada
        
        if resultado == "GREEN_VALIDO":
            now = time.time()
            if learn_state.get("last_white_ts"):
                gap = now - float(learn_state["last_white_ts"])
                _append_bounded(learn_state["white_gaps"], gap, 200)
                _append_bounded(learn_state["stones_gaps"], learn_state["stones_since_last_white"], 200)
                
            learn_state["last_white_ts"] = now
            learn_state["stones_since_last_white"] = 0 # Zera a contagem de pedras (saiu branco)

            msg_text = build_result_message("✅ **GREEN no BRANCO!** ⚪️")
            print("✅ [DEBUG] Enviando mensagem de GREEN (sinal ativo):", msg_text)
            await send_telegram_message(CANAL_DESTINO_ID, msg_text)
            _save_learn()
            return {"ok": True, "action": "green_logged"}

        elif resultado == "LOSS":
            # Não zera a contagem de pedras (porque o branco não saiu)
            msg_text = build_result_message("❌ **LOSS** 😥")
            print("❌ [DEBUG] Enviando mensagem de LOSS (sinal ativo):", msg_text)
            await send_telegram_message(CANAL_DESTINO_ID, msg_text)
            _save_learn()
            return {"ok": True, "action": "loss_logged"}
        
    # ========================== BLOCO DE ENTRADA ==========================
    if is_entrada_confirmada(text) and not ignorar_gale(text):
        
        if entry_is_active:
             # IGNORA o novo sinal se o sinal anterior ainda não foi finalizado.
             print("⚠️ [DEBUG] Entrada ignorada: Já existe um sinal ativo aguardando resultado.")
             return {"ok": True, "action": "entry_ignored_active_trade"}
             
        # Aumenta a contagem de pedras (pois este é o novo sinal de entrada)
        learn_state["stones_since_last_white"] = learn_state.get("stones_since_last_white", 0) + 1
        learn_state["entry_active"] = True # <--- ATIVA O ESTADO: AGORA ESTAMOS AGUARDANDO UM RESULTADO

        m = re.search(r"(\d{1,2})", text)
        num_alvo = m.group(1) if m else "?"
        msg_text = build_entry_message(num_alvo)
        print("🎯 [DEBUG] Entrada detectada! Enviando:", msg_text)
        await send_telegram_message(CANAL_DESTINO_ID, msg_text)
        _save_learn()
        return {"ok": True, "action": "entry_forwarded"}

    # ========================== BLOCO DE IGNORAR ==========================
    # Se a mensagem não é nem resultado (e não estava ativo) nem entrada, é ignorada.
    print("⚪ [DEBUG] Nenhum evento identificado. Texto:", text)
    _save_learn() # Salvando caso o estado entry_active seja a única alteração
    return {"ok": True, "action": "ignored"}
