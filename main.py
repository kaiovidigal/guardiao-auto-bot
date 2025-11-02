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
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "0"))  # 0 no modo livre
STRICT_MODE_ENV = os.getenv("STRICT_MODE", "false").lower() == "true"  # default livre

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

# ========== APP/STATE ==========
app = FastAPI()
last_signal_time = 0
processed_entries = set()            # evita duplicar entradas
STRICT_MODE_FLAG = STRICT_MODE_ENV   # pode trocar via /modo

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
    modo = "STRICT" if STRICT_MODE_FLAG else "LIVRE"
    return f"📊 *Parcial do dia* ({c['date']}):\n✅ GREEN: {c['green']}\n❌ LOSS: {c['loss']}\n\n⚙️ Modo: *{modo}*  |  ⏱ Cooldown: {COOLDOWN_SECONDS}s"

# ===== RESULTADOS =====
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

def is_result_message(text_lower: str, has_entrada_words: bool) -> bool:
    # Resultado real = palavras de resultado E NÃO ser mensagem de entrada
    has_result_words = any(w in text_lower for w in [
        "vitória","vitoria","win","loss","derrota","perda","não bateu","nao bateu","não deu","nao deu","falhou"
    ])
    return has_result_words and not has_entrada_words

# ========== ROUTES ==========
@app.get("/")
def root():
    return {"status": "ok", "service": "Jonbet - Branco (modo livre/strict + contador diário)"}

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

    if text_lower.startswith("/modo"):
        global STRICT_MODE_FLAG
        if "strict" in text_lower:
            STRICT_MODE_FLAG = True
        elif "livre" in text_lower:
            STRICT_MODE_FLAG = False
        await send_telegram_message(CANAL_DESTINO_ID, f"⚙️ Modo atualizado: *{'STRICT' if STRICT_MODE_FLAG else 'LIVRE'}*")
        return {"ok": True, "action": "mode_set"}

    # ===== PLACAR (só GREEN se for vitória NO BRANCO) =====
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

    # ===== ENTRADA =====
    contains_branco   = ("branco" in text_lower) or ("⚪" in text) or ("⬜" in text)
    has_entrada_words = any(w in text_lower for w in ["entrada", "entrar", "entrada confirmada"])

    if contains_branco and has_entrada_words:
        # Ignorar pré-sinais
        if any(w in text_lower for w in ["possível entrada","possivel entrada","analisando"]):
            logging.info("Ignorado: possível entrada/análise.")
            return {"ok": True, "action": "ignored_possible_entry"}

        # Não confundir com placar
        if is_result_message(text_lower, has_entrada_words):
            logging.info("Ignorado: texto parece resultado, não entrada.")
            return {"ok": True, "action": "ignored_result_like"}

        # Antiduplicação
        mid = msg.get("message_id")
        if mid and mid in processed_entries:
            logging.info("Entrada duplicada ignorada.")
            return {"ok": True, "action": "ignored_duplicate_entry"}
        if mid:
            processed_entries.add(mid)

        # Cooldown (só quando STRICT_MODE_FLAG=True)
        global last_signal_time
        now = time.time()
        if STRICT_MODE_FLAG and now - last_signal_time < COOLDOWN_SECONDS:
            logging.info("Cooldown (STRICT_MODE).")
            return {"ok": True, "action": "ignored_cooldown"}

        salvar_evento("entrada")
        await send_telegram_message(CANAL_DESTINO_ID, build_final_message())
        last_signal_time = now
        logging.info("Sinal BRANCO enviado (modo %s).", "STRICT" if STRICT_MODE_FLAG else "LIVRE")
        return {"ok": True, "action": "signal_sent_white"}

    return {"ok": True, "action": "ignored"}