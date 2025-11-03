# -*- coding: utf-8 -*-
# ✅ JonBet Auto Bot — Modo Aprendizado Ativo (Render)
# - Reenvia TODO sinal como "Entrada no BRANCO" pro canal destino
# - Resultado: só GREEN se a mensagem afirmar vitória no branco
# - Aprende tempo entre brancos e "pedras" (contador incremental)
# - Sem travas, sem estratégia, só forward + aprendizado

import os
import json
import time
import logging
import re
import unicodedata
from statistics import median
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
import httpx

# ===================== CONFIG =====================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "Jonbet")
CANAL_ORIGEM_IDS = [s.strip() for s in os.getenv("CANAL_ORIGEM_IDS", "-1003156785631").split(",")]
CANAL_DESTINO_ID = os.getenv("CANAL_DESTINO_ID", "-1002796105884")

# Sem travas
COOLDOWN_SECONDS = 0
RESULT_WINDOW_SECONDS = 600  # janela para parear resultado com última entrada (p/ editar no futuro, se quiser)

# Aprendizado ligado, mas SEM usar para travar fluxo
SMART_TIMING = True

DATA_DIR = "/var/data"
os.makedirs(DATA_DIR, exist_ok=True)
HISTORICO_PATH = os.path.join(DATA_DIR, "historico.json")
LEARN_PATH = os.path.join(DATA_DIR, "learn.json")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MESSAGE_URL = f"{TELEGRAM_API_URL}/sendMessage"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===================== APP/STATE =====================
app = FastAPI()
last_signal_time = 0
last_signal_msg_id: Optional[int] = None
app.state.processed_entries = set()

# ===================== APRENDIZADO =====================
learn_state = {
    "deltas": [],                 # segundos sinal->resultado (se quiser usar depois)
    "last_entry_ts": None,
    "last_white_ts": None,        # timestamp do último GREEN no branco
    "white_gaps": [],             # segundos entre brancos
    "stones_since_last_white": 0, # "pedras" desde o último branco (contador incremental)
    "stones_gaps": [],            # histórico de pedras entre brancos
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
                data = json.load(f)
            # garante chaves
            for k, v in learn_state.items():
                data.setdefault(k, v)
            learn_state = data
    except Exception:
        pass

_load_learn()

# ===================== HELPERS =====================
def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def _now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _append_bounded(lst, val, maxlen=200):
    lst.append(val)
    if len(lst) > maxlen:
        del lst[:len(lst)-maxlen]

def extract_message(data: dict) -> dict:
    msg = data.get("message") or data.get("channel_post") or data.get("edited_message") or data.get("edited_channel_post") or {}
    return {
        "chat": msg.get("chat", {}),
        "text": msg.get("text") or msg.get("caption") or "",
        "message_id": msg.get("message_id")
    }

# ===== entradas =====
def is_pre_signal(t: str) -> bool:
    t = _strip_accents(t)
    return any(w in t for w in ["possivel entrada","possível entrada","analisando","analise","analise","aguarde","ainda nao","ainda não","esperar","esperem"])

def is_entry_signal_any_color(raw: str) -> bool:
    """
    Detecta "sinal de entrada" independente da cor (verde/preto),
    para sempre reenviar como ENTRADA NO BRANCO no canal destino.
    """
    t = _strip_accents(raw.lower())
    # palavras fortes de chamada
    has_call = any(w in t for w in [
        "entrada confirmada", "entrar", "entrada", "apostar", "aposta", "aposte"
    ])
    # indícios de cor/estratégia do canal fonte (vamos converter tudo pra branco)
    has_color_or_prot = any(w in t for w in ["verde", "preto", "⚫", "⬛", "🟢", "protecao", "proteção", "branco"])
    return has_call and not is_pre_signal(t) and has_color_or_prot

# ===== resultados =====
def classificar_resultado(txt: str) -> Optional[str]:
    """
    Regras do destino:
      - GREEN apenas se a mensagem indicar vitória no BRANCO (⚪ / "branco").
      - Qualquer outro win (verde/preto) => LOSS no nosso canal.
    """
    t = _strip_accents(txt.lower())
    menciona_branco = ("branco" in t) or ("⚪" in txt) or ("⬜" in txt)

    # termos de vitória
    has_vitoria = any(p in t for p in [
        "vitoria", "vitória", "ganho", "ganhamos", "bateu", "acertou", "deu certo", "win"
    ])
    # termos de derrota
    has_derrota = any(p in t for p in [
        "derrota", "loss", "perdeu", "nao bateu", "não bateu", "nao deu", "não deu", "falhou"
    ])

    if has_vitoria and menciona_branco:
        return "GREEN_VALIDO"
    if has_derrota:
        return "LOSS"

    # Se disser "bateu verde/preto" sem citar branco => LOSS
    if has_vitoria and not menciona_branco:
        return "LOSS"

    return None

# ===================== MENSAGENS =====================
def build_entry_message() -> str:
    return (
        "🚨 **ENTRADA IMEDIATA NO BRANCO!** ⚪️\n\n"
        "🎯 JOGO: Double JonBet\n"
        "🔥 FOCO: BRANCO\n"
        "📊 Confiança: `Filtro ON (TEXTUAL)`\n"
        "🧠 Análise: _Aprendizado ativo._\n\n"
        "⚠️ **ESTRATÉGIA: G0 (ZERO GALES)**\n"
        "💻 Site: Acessar Double"
    )

def build_result_message(resultado_txt: str) -> str:
    since_sec = time.time() - float(learn_state["last_white_ts"]) if learn_state.get("last_white_ts") else 0
    stones = learn_state.get("stones_since_last_white", 0)
    med_gap = int(median(learn_state["white_gaps"])) if learn_state["white_gaps"] else 0
    med_stones = int(median(learn_state["stones_gaps"])) if learn_state["stones_gaps"] else 0
    last_white_clock = "-"
    if learn_state.get("last_white_ts"):
        last_white_clock = datetime.fromtimestamp(learn_state["last_white_ts"]).strftime("%Y-%m-%d %H:%M:%S")

    return (
        f"{resultado_txt}\n\n"
        f"⏱️ *Tempo desde o último branco:* {int(since_sec//60)}m {int(since_sec%60)}s "
        f"(mediana: {int(med_gap//60)}m {int(med_gap%60)}s)\n"
        f"🪙 *Distância entre brancos:* {stones} pedras "
        f"(mediana: {med_stones} pedras)\n"
        f"🕒 *Último branco:* {last_white_clock}"
    )

# ===================== TELEGRAM =====================
async def send_telegram_message(chat_id: str, text: str) -> Optional[int]:
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True}
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(SEND_MESSAGE_URL, json=payload, timeout=15)
            r.raise_for_status()
            return r.json().get("result", {}).get("message_id")
        except Exception as e:
            logging.error(f"Erro ao enviar mensagem: {e}")
            return None

# ===================== ROTAS =====================
@app.get("/")
def root():
    return {"status": "ok", "service": "JonBet — Aprendizado + Forward para BRANCO"}

@app.post(f"/webhook/{{webhook_token}}")
async def webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token incorreto.")

    global last_signal_time, last_signal_msg_id

    data = await request.json()
    msg = extract_message(data)
    chat_id = str(msg["chat"].get("id"))
    text = (msg["text"] or "").strip()

    # Só processa canal(es) fonte configurado(s)
    if chat_id not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored_wrong_source"}

    # ===== Aprendizado: incrementa "pedras" a cada mensagem do canal-fonte
    try:
        learn_state["stones_since_last_white"] = int(learn_state.get("stones_since_last_white", 0)) + 1
    except Exception:
        learn_state["stones_since_last_white"] = 1
    _save_learn()

    # ===== Resultado (GREEN/LOSS) — regra do destino
    res = classificar_resultado(text)
    if res == "GREEN_VALIDO":
        now = time.time()
        # aprender tempo e pedras
        if learn_state.get("last_white_ts"):
            gap = now - float(learn_state["last_white_ts"])
            if gap > 0:
                _append_bounded(learn_state["white_gaps"], gap, 200)
        _append_bounded(learn_state["stones_gaps"], int(learn_state.get("stones_since_last_white", 0)), 200)
        learn_state["last_white_ts"] = now
        learn_state["stones_since_last_white"] = 0
        _save_learn()

        await send_telegram_message(CANAL_DESTINO_ID, build_result_message("✅ **GREEN no BRANCO!** ⚪️"))
        return {"ok": True, "action": "green_logged"}

    if res == "LOSS":
        # Não zera pedras aqui (não bateu branco)
        await send_telegram_message(CANAL_DESTINO_ID, build_result_message("❌ **LOSS** 😥"))
        return {"ok": True, "action": "loss_logged"}

    # ===== Entrada — qualquer cor do fonte vira ENTRADA NO BRANCO no destino
    if is_entry_signal_any_color(text):
        # sem cooldown e sem bloqueios
        last_signal_msg_id = await send_telegram_message(CANAL_DESTINO_ID, build_entry_message())
        last_signal_time = time.time()
        # log simples (opcional)
        try:
            with open(HISTORICO_PATH, "a") as f:
                f.write(json.dumps({"hora": _now_iso(), "tipo": "entrada", "fonte": chat_id}) + "\n")
        except Exception:
            pass
        # marca ts p/ futura medição (se quiser)
        if SMART_TIMING:
            learn_state["last_entry_ts"] = time.time()
            _save_learn()
        return {"ok": True, "action": "entry_forwarded"}

    # Se não for entrada nem resultado, só aprendemos as pedras e seguimos
    _save_learn()
    return {"ok": True, "action": "ignored"}