import os
import json
import time
import logging
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
import httpx

# ========== CONFIG ==========
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "Jonbet")
CANAL_ORIGEM_IDS = [s.strip() for s in os.getenv("CANAL_ORIGEM_IDS", "-1003156785631").split(",")]
CANAL_DESTINO_ID = os.getenv("CANAL_DESTINO_ID", "-1002796105884")
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "5"))  # sugerido 5s

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MESSAGE_URL = f"{TELEGRAM_API_URL}/sendMessage"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========== STORAGE ==========
DATA_DIR = "/var/data"
os.makedirs(DATA_DIR, exist_ok=True)
HISTORICO_PATH = os.path.join(DATA_DIR, "historico.json")
COUNTERS_PATH = os.path.join(DATA_DIR, "counters.json")
logging.info(f"📁 DATA_DIR: {DATA_DIR}")
logging.info(f"🗂️ histórico: {HISTORICO_PATH}")
logging.info(f"🗂️ counters:  {COUNTERS_PATH}")

# ========== APP ==========
app = FastAPI()
last_signal_time = 0

# ========== HELPERS ==========
async def send_telegram_message(chat_id: str, text: str):
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True}
    async with httpx.AsyncClient() as client:
        try:
            await client.post(SEND_MESSAGE_URL, json=payload, timeout=15)
        except Exception as e:
            logging.error(f"Erro ao enviar mensagem: {e}")

def build_final_message() -> str:
    return (
        "🚨 **ENTRADA IMEDIATA NO BRANCO!** ⚪️\n\n"
        "🎯 JOGO: Double JonBet\n"
        "🔥 FOCO: BRANCO\n"
        "📊 Confiança: `Filtro ON (TEXTUAL)`\n"
        "🧠 Análise: _Filtro de Texto Agressivo Ativado._\n\n"
        "⚠️ **ESTRATÉGIA: G0 (ZERO GALES)**\n"
        "💻 Site: Acessar Double"
    )

def salvar_evento(tipo: str, resultado: Optional[str] = None):
    registro = {"hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "tipo": tipo, "resultado": resultado}
    with open(HISTORICO_PATH, "a") as f:
        f.write(json.dumps(registro) + "\n")
    logging.info(f"💾 Evento salvo: {registro}")

def extract_message(data: dict) -> dict:
    msg = (
        data.get("message")
        or data.get("channel_post")
        or data.get("edited_message")
        or data.get("edited_channel_post")
        or {}
    )
    return {"chat": msg.get("chat", {}), "text": msg.get("text") or msg.get("caption") or "", "message_id": msg.get("message_id")}

# ===== CONTADORES DIÁRIOS =====
def _load_counters():
    try:
        with open(COUNTERS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"date": datetime.now().strftime("%Y-%m-%d"), "green": 0, "loss": 0}

def _save_counters(data):
    with open(COUNTERS_PATH, "w") as f:
        json.dump(data, f)

def _reset_if_new_day():
    c = _load_counters()
    today = datetime.now().strftime("%Y-%m-%d")
    if c.get("date") != today:
        c = {"date": today, "green": 0, "loss": 0}
        _save_counters(c)
        logging.info("✅ Contadores zerados (00:00).")
    return c

def contabilizar(resultado: str):
    c = _reset_if_new_day()
    if resultado == "GREEN":
        c["green"] += 1
    elif resultado == "LOSS":
        c["loss"] += 1
    _save_counters(c)
    return c

def get_status_msg():
    c = _reset_if_new_day()
    return f"📊 *Parcial do dia* ({c['date']}):\n✅ GREEN: {c['green']}\n❌ LOSS: {c['loss']}"

# ===== CLASSIFICAÇÃO DE RESULTADO =====
def classificar_resultado(texto: str) -> Optional[str]:
    t = texto.lower()

    menciona_vitoria = any(p in t for p in ["vitória", "vitoria", "acertamos", "acerto"])
    menciona_branco  = ("branco" in t) or ("⚪" in t) or ("⬜" in t)
    if menciona_vitoria and menciona_branco:
        if "como proteção" not in t and "protecao" not in t and "proteção" not in t:
            return "GREEN_VALIDO"

    if any(p in t for p in ["derrota", "loss", "❌", "perdeu", "perda", "não bateu", "nao bateu", "não deu", "nao deu", "falhou"]):
        return "LOSS"

    if any(p in t for p in ["vitória de primeira", "vitoria de primeira", "vitória com", "vitoria com", "gale", "g1", "g 1", "g2", "g 2"]):
        return "LOSS"

    if "green" in t and not menciona_branco:
        return "LOSS"

    return None

# ========== ROUTES ==========
@app.get("/")
def root():
    return {"status": "ok", "service": "Jonbet - Branco (destravado + contador diário)"}

@app.post(f"/webhook/{{webhook_token}}")
async def webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token incorreto.")

    data = await request.json()
    msg = extract_message(data)
    chat_id = str(msg["chat"].get("id"))
    text = (msg["text"] or "").strip()
    text_lower = text.lower()

    if chat_id not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored_wrong_source"}

    # ===== COMANDOS =====
    if text_lower.startswith("/status"):
        await send_telegram_message(CANAL_DESTINO_ID, get_status_msg())
        return {"ok": True, "action": "status"}

    if text_lower.startswith("/zerar"):
        _save_counters({"date": datetime.now().strftime("%Y-%m-%d"), "green": 0, "loss": 0})
        await send_telegram_message(CANAL_DESTINO_ID, "♻️ Contadores zerados manualmente.")
        return {"ok": True, "action": "reset_manual"}

    # ===== PLACAR (só é GREEN se for vitória no BRANCO) =====
    resultado = classificar_resultado(text)
    if resultado == "GREEN_VALIDO":
        salvar_evento("resultado", "GREEN")
        contabilizar("GREEN")
        await send_telegram_message(CANAL_DESTINO_ID, f"✅ **GREEN no BRANCO!** ⚪️\n\n{get_status_msg()}")
        return {"ok": True, "action": "green_branco"}

    if resultado == "LOSS":
        salvar_evento("resultado", "LOSS")
        contabilizar("LOSS")
        await send_telegram_message(CANAL_DESTINO_ID, f"❌ **LOSS** 😥\n\n{get_status_msg()}")
        return {"ok": True, "action": "loss"}

    # ===== ENTRADA BRANCO (aceita 'Entrada confirmada' com ✅/proteção) =====
    contains_branco = ("branco" in text_lower) or ("⚪" in text) or ("⬜" in text)
    has_entrada_words = any(w in text_lower for w in ["entrada", "entrar", "entrada confirmada"])

    # Resultado verdadeiro: tem palavras de resultado E não é uma mensagem de entrada
    has_result_words = any(w in text_lower for w in ["vitória", "vitoria", "win", "loss", "derrota", "perda", "não bateu", "nao bateu", "não deu", "nao deu", "falhou"])
    is_resultado_msg = has_result_words and not has_entrada_words
    # ⚠️ NÃO trate ✅/🟢 como resultado por si só — aparecem muito nas entradas

    if not contains_branco:
        logging.info("Ignorado: não contém BRANCO.")
        return {"ok": True, "action": "ignored_not_branco"}

    if is_resultado_msg:
        logging.info("Ignorado: mensagem de resultado, não de entrada.")
        return {"ok": True, "action": "ignored_result_message"}

    # Ignora pré-sinais
    if any(w in text_lower for w in ["possível entrada", "possivel entrada", "analisando"]):
        logging.info("Ignorado: possível entrada/análise.")
        return {"ok": True, "action": "ignored_possible_entry"}

    # Aceita “branco com proteção” e similares
    global last_signal_time
    now = time.time()
    if now - last_signal_time < COOLDOWN_SECONDS:
        logging.info("Sinal ignorado por cooldown curto.")
        return {"ok": True, "action": "ignored_cooldown"}

    salvar_evento("entrada")
    await send_telegram_message(CANAL_DESTINO_ID, build_final_message())
    last_signal_time = now
    logging.info("Sinal BRANCO enviado (entrada confirmada).")
    return {"ok": True, "action": "signal_sent_white"}