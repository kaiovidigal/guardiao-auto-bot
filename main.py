import os
import json
import time
import logging
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request
import httpx

# ========== CONFIGURAÇÕES ==========
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "Jonbet")
CANAL_ORIGEM_IDS = [s.strip() for s in os.getenv("CANAL_ORIGEM_IDS", "-1003156785631").split(",")]
CANAL_DESTINO_ID = os.getenv("CANAL_DESTINO_ID", "-1002796105884")
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "30"))

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MESSAGE_URL = f"{TELEGRAM_API_URL}/sendMessage"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========== STORAGE ==========
DATA_DIR = "/var/data"
os.makedirs(DATA_DIR, exist_ok=True)
historico_path = os.path.join(DATA_DIR, "historico.json")
COUNTERS_PATH = os.path.join(DATA_DIR, "counters.json")

logging.info(f"📁 Diretório de dados: {DATA_DIR}")
logging.info(f"🗂️ Histórico: {historico_path}")

# ========== FASTAPI ==========
app = FastAPI()
last_signal_time = 0


# ========== FUNÇÕES AUXILIARES ==========
def salvar_evento(tipo, resultado=None):
    registro = {
        "hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tipo": tipo,
        "resultado": resultado
    }
    with open(historico_path, "a") as f:
        f.write(json.dumps(registro) + "\n")
    logging.info(f"💾 Evento salvo: {registro}")


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


async def send_telegram_message(chat_id: str, text: str):
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    async with httpx.AsyncClient() as client:
        try:
            await client.post(SEND_MESSAGE_URL, json=payload, timeout=15)
        except Exception as e:
            logging.error(f"Erro ao enviar mensagem: {e}")


def extract_message(data: dict) -> dict:
    msg = (
        data.get("message")
        or data.get("channel_post")
        or data.get("edited_message")
        or data.get("edited_channel_post")
        or {}
    )
    return {
        "chat": msg.get("chat", {}),
        "text": msg.get("text") or msg.get("caption") or "",
        "message_id": msg.get("message_id"),
    }


# ===== CONTADORES DIÁRIOS =====
def _load_counters():
    try:
        with open(COUNTERS_PATH, "r") as f:
            data = json.load(f)
    except Exception:
        data = {"date": datetime.now().strftime("%Y-%m-%d"), "green": 0, "loss": 0}
    return data

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


# --- CLASSIFICAÇÃO DE RESULTADO ---
def classificar_resultado(texto: str) -> Optional[str]:
    t = texto.lower()

    menciona_vitoria = any(p in t for p in ["vitória", "vitoria", "acertamos", "acerto"])
    menciona_branco  = ("branco" in t) or ("⚪" in t) or ("⬜" in t)
    if menciona_vitoria and menciona_branco:
        if "como proteção" not in t and "protecao" not in t and "proteção" not in t:
            return "GREEN_VALIDO"

    if any(p in t for p in [
        "derrota", "loss", "❌", "perdeu", "perda",
        "não bateu", "nao bateu", "não deu", "nao deu", "falhou"
    ]):
        return "LOSS"

    if any(p in t for p in [
        "vitória de primeira", "vitoria de primeira",
        "vitória com", "vitoria com", "gale", "g1", "g 1", "g2", "g 2"
    ]):
        return "LOSS"

    if "green" in t and not menciona_branco:
        return "LOSS"

    return None


# ========== ENDPOINTS ==========
@app.get("/")
def root():
    return {"status": "ok", "service": "Jonbet Bot com contador diário"}


@app.post(f"/webhook/{{webhook_token}}")
async def webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token incorreto.")

    data = await request.json()
    msg = extract_message(data)
    chat_id = str(msg["chat"].get("id"))
    text = msg["text"].lower().strip()

    if chat_id not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored_wrong_source"}

    global last_signal_time
    now = time.time()

    # ===== COMANDOS =====
    if text.startswith("/status"):
        await send_telegram_message(CANAL_DESTINO_ID, get_status_msg())
        return {"ok": True, "action": "status"}

    if text.startswith("/zerar"):
        _save_counters({"date": datetime.now().strftime("%Y-%m-%d"), "green": 0, "loss": 0})
        await send_telegram_message(CANAL_DESTINO_ID, "♻️ Contadores zerados manualmente.")
        return {"ok": True, "action": "reset_manual"}

    # ===== PLACAR =====
    resultado = classificar_resultado(text)
    if resultado == "GREEN_VALIDO":
        salvar_evento("resultado", "GREEN")
        c = contabilizar("GREEN")
        await send_telegram_message(
            CANAL_DESTINO_ID,
            f"✅ **GREEN no BRANCO!** ⚪️\n\n{get_status_msg()}"
        )
        return {"ok": True, "action": "green_branco"}

    if resultado == "LOSS":
        salvar_evento("resultado", "LOSS")
        c = contabilizar("LOSS")
        await send_telegram_message(
            CANAL_DESTINO_ID,
            f"❌ **LOSS** 😥\n\n{get_status_msg()}"
        )
        return {"ok": True, "action": "loss"}

    # ===== ENTRADA =====
    is_branco_puro = (
        ("entrada" in text or "entrar" in text or "entrada confirmada" in text)
        and (("branco" in text) or ("⚪" in text) or ("⬜" in text))
        and not any(p in text for p in ["preto", "vermelho", "verde", "gale", "proteção", "protecao", "como proteção"])
    )

    if is_branco_puro:
        if now - last_signal_time < COOLDOWN_SECONDS:
            return {"ok": True, "action": "cooldown"}
        salvar_evento("entrada")
        await send_telegram_message(CANAL_DESTINO_ID, build_final_message())
        last_signal_time = now
        return {"ok": True, "action": "entrada_enviada"}

    return {"ok": True, "action": "ignored"}