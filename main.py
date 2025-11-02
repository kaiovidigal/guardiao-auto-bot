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

# ===== IA analítica (ChatGPT) =====
from openai import OpenAI

# ========== CONFIG ==========
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "Jonbet")
CANAL_ORIGEM_IDS = [s.strip() for s in os.getenv("CANAL_ORIGEM_IDS", "-1003156785631").split(",")]
CANAL_DESTINO_ID = os.getenv("CANAL_DESTINO_ID", "-1002796105884")

COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "0"))                # 0 = sem travar
RESULT_WINDOW_SECONDS = int(os.getenv("RESULT_WINDOW_SECONDS", "600"))    # 10min p/ parear placar

# Aprendizado & critérios
SMART_TIMING = os.getenv("SMART_TIMING", "true").lower() == "true"        # aprende tempo sinal→placar
AUTO_EXECUTE = os.getenv("AUTO_EXECUTE", "true").lower() == "true"        # só solta com confiança
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.65"))               # corte 0..1
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "20"))                         # amostras mín p/ hora

# IA (relatório /analise)
ANALYTICS_MODE = os.getenv("ANALYTICS_MODE", "true").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MESSAGE_URL = f"{TELEGRAM_API_URL}/sendMessage"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========== STORAGE ==========
DATA_DIR = "/var/data"
os.makedirs(DATA_DIR, exist_ok=True)
HISTORICO_PATH = os.path.join(DATA_DIR, "historico.json")
COUNTERS_PATH = os.path.join(DATA_DIR, "counters.json")
LEARN_PATH = os.path.join(DATA_DIR, "learn.json")
logging.info(f"📁 DATA_DIR: {DATA_DIR}")
logging.info(f"🗂️ histórico: {HISTORICO_PATH}")
logging.info(f"🗂️ counters:  {COUNTERS_PATH}")
logging.info(f"🗂️ learn:     {LEARN_PATH}")

# ========== APP/STATE ==========
app = FastAPI()
last_signal_time = 0
last_signal_msg_id: Optional[int] = None
app.state.processed_entries = set()  # antiduplicação por message_id

# ===== learn_state (tempo em SEGUNDOS) =====
learn_state = {
    "deltas": [],            # últimos deltas sinal→placar (s)
    "last_entry_ts": None,   # ts do último sinal enviado
    "by_hour": {},           # estatística de resultados por hora: {"13":{"g":X,"l":Y}}
    "last_white_ts": None,   # ts do último GREEN no BRANCO
    "white_gaps": [],        # últimos gaps (s) entre BRANCOS
    "white_gaps_by_hour": {} # por hora do ÚLTIMO branco: {"13":[...]}
}

def _load_learn():
    global learn_state
    try:
        with open(LEARN_PATH, "r") as f:
            data = json.load(f)
        # merge com defaults
        for k, v in learn_state.items():
            if k not in data:
                data[k] = v
        # sane defaults
        if not isinstance(data.get("deltas", []), list):
            data["deltas"] = []
        if not isinstance(data.get("by_hour", {}), dict):
            data["by_hour"] = {}
        if not isinstance(data.get("white_gaps", []), list):
            data["white_gaps"] = []
        if not isinstance(data.get("white_gaps_by_hour", {}), dict):
            data["white_gaps_by_hour"] = {}
        learn_state = data
    except Exception:
        pass

def _save_learn():
    try:
        with open(LEARN_PATH, "w") as f:
            json.dump(learn_state, f)
    except Exception as e:
        logging.error(f"Erro ao salvar learn.json: {e}")

_load_learn()

# ========== TELEGRAM SENDER ==========
async def send_telegram_message(chat_id: str, text: str, reply_to_message_id: Optional[int] = None) -> Optional[int]:
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    if reply_to_message_id:
        payload["reply_to_message_id"] = reply_to_message_id
        payload["allow_sending_without_reply"] = True

    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(SEND_MESSAGE_URL, json=payload, timeout=15)
            r.raise_for_status()
            data = r.json()
            return data.get("result", {}).get("message_id")
        except Exception as e:
            logging.error(f"Erro ao enviar mensagem: {e}")
            return None

# ========== HELPERS ==========
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

def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def is_pre_signal(text_lower: str) -> bool:
    return any(w in text_lower for w in [
        "possível entrada", "possivel entrada", "analisando", "análise", "analise",
        "ainda não", "ainda nao", "aguarde", "esperar", "esperem"
    ])

def is_result_message(text_lower: str, has_entrada_words: bool) -> bool:
    has_result_words = any(w in text_lower for w in [
        "vitória","vitoria","win","loss","derrota","perda",
        "não bateu","nao bateu","não deu","nao deu","falhou"
    ])
    return has_result_words and not has_entrada_words  # ✅ emojis não contam como placar

def is_entrada_branco(text_raw: str) -> bool:
    """
    Aceita variações: 'entrada confirmada ... branco/⚪/⬜', 'apostar no branco ...',
    'entrar após ... branco', emojis/ordem trocada/acentos.
    """
    t = _strip_accents(text_raw.strip().lower())
    has_branco = ("branco" in t) or ("⚪" in text_raw) or ("⬜" in text_raw)
    if not has_branco:
        return False

    has_entrada_words = any(w in t for w in [
        "entrada", "entrar", "entrada confirmada", "confirmada", "confirmado",
        "aposta", "apostar", "aposte", "aposta confirmada"
    ])
    if not has_entrada_words:
        patterns = [
            r'\bentrada\b.{0,40}\b(branco)\b',
            r'\b(apostar|aposta|aposte)\b.{0,60}\b(branco)\b',
            r'\bconfirmad[oa]\b.{0,40}\b(entrada)\b.{0,40}\b(branco)\b',
            r'\b(branco)\b.{0,40}\b(entrada|aposta[rdo]?|confirmad[oa])\b',
            r'\bentrar(?:\s+apos|\s*:\s*)?.{0,10}(branco)\b',
        ]
        for p in patterns:
            if re.search(p, t):
                has_entrada_words = True
                break

    return has_entrada_words and not is_pre_signal(t)

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

# ===== RESULTADOS (só GREEN se for vitória no BRANCO) =====
def classificar_resultado(texto: str) -> Optional[str]:
    t = _strip_accents(texto.lower())
    menciona_vitoria = any(p in t for p in ["vitoria", "acertamos", "acerto"])
    menciona_branco  = ("branco" in t) or ("⚪" in texto) or ("⬜" in texto)
    if menciona_vitoria and menciona_branco and ("protecao" not in t and "como protecao" not in t):
        return "GREEN_VALIDO"

    if any(p in t for p in ["derrota","loss","perdeu","perda","nao bateu","nao deu","falhou"]):
        return "LOSS"

    if any(p in t for p in ["vitoria de primeira","vitoria com","gale","g1","g 1","g2","g 2"]):
        return "LOSS"

    if "green" in t and not menciona_branco:
        return "LOSS"

    return None

# ===== IA ANALÍTICA (/analise) =====
def _get_openai_client():
    if not ANALYTICS_MODE or not OPENAI_API_KEY:
        return None
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logging.error(f"OpenAI client error: {e}")
        return None

def _carregar_eventos():
    eventos = []
    try:
        with open(HISTORICO_PATH, "r") as f:
            for line in f:
                try:
                    eventos.append(json.loads(line))
                except:
                    pass
    except FileNotFoundError:
        pass
    return eventos

def _horario(hora_str):
    return hora_str[11:13] if hora_str and len(hora_str) >= 13 else "??"

def _sinal_para_placar_deltas(eventos):
    deltas = []
    last_entry_ts = None
    for e in eventos:
        if e.get("tipo") == "entrada":
            last_entry_ts = e.get("hora")
        elif e.get("tipo") == "resultado" and last_entry_ts:
            try:
                t1 = datetime.strptime(last_entry_ts, "%Y-%m-%d %H:%M:%S").timestamp()
                t2 = datetime.strptime(e["hora"], "%Y-%m-%d %H:%M:%S").timestamp()
                if t2 >= t1:
                    deltas.append(t2 - t1)
                    last_entry_ts = None
            except:
                pass
    return deltas

async def gerar_analise_ia_e_postar():
    client = _get_openai_client()
    if not client:
        await send_telegram_message(CANAL_DESTINO_ID, "⚠️ IA desativada ou sem OPENAI_API_KEY.")
        return

    eventos = _carregar_eventos()
    if not eventos:
        await send_telegram_message(CANAL_DESTINO_ID, "⚠️ Sem dados ainda para analisar.")
        return

    por_hora_green = {}
    greens = losses = 0
    for e in eventos:
        if e.get("tipo") == "resultado":
            if e.get("resultado") == "GREEN":
                greens += 1
                h = _horario(e.get("hora"))
                por_hora_green[h] = por_hora_green.get(h, 0) + 1
            elif e.get("resultado") == "LOSS":
                losses += 1

    deltas = _sinal_para_placar_deltas(eventos)
    med_delta = int(sorted(deltas)[len(deltas)//2]) if deltas else 0

    resumo_json = json.dumps({
        "greens_por_hora": por_hora_green,
        "greens_total": greens,
        "losses_total": losses,
        "mediana_segundos_sinal_para_placar": med_delta
    }, ensure_ascii=False, indent=2)

    prompt = (
        "Você é um analista de desempenho para sinais de 'BRANCO' no double.\n"
        "Com base nos dados abaixo, gere um relatório curto e objetivo com:\n"
        "• Horários com maior frequência de GREEN (top 3)\n"
        "• Janela recomendada de pareamento (use a mediana + margem)\n"
        "• Taxa geral de acerto (aprox.)\n"
        "• Observações úteis\n\n"
        f"DADOS:\n{resumo_json}\n"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Seja conciso, prático e direto."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        analise = resp.choices[0].message.content.strip()
        await send_telegram_message(CANAL_DESTINO_ID, f"🧠 *Análise IA (BRANCO)*\n\n{analise}")
    except Exception as e:
        logging.error(f"OpenAI request error: {e}")
        await send_telegram_message(CANAL_DESTINO_ID, f"⚠️ Erro ao gerar análise IA: {e}")

# ===== Utilidades de tempo/gaps =====
def _med_or_none(vals):
    if not vals: return None
    try: return median(vals)
    except: return None

def _append_bounded(lst, val, maxlen=200):
    lst.append(val)
    if len(lst) > maxlen:
        del lst[:len(lst)-maxlen]

def _predict_next_white_eta():
    """
    Retorna (eta_ts, gap_med_seg, fonte: 'hora'|'global'|None)
    """
    now = time.time()
    last_ts = learn_state.get("last_white_ts")
    if not last_ts:
        return None, None, None
    try:
        hh_last = datetime.fromtimestamp(float(last_ts)).strftime("%H")
    except:
        hh_last = None

    med_hora = None
    if hh_last:
        gaps_h = learn_state.get("white_gaps_by_hour", {}).get(hh_last, [])
        med_hora = _med_or_none(gaps_h)

    med_global = _med_or_none(learn_state.get("white_gaps", []))

    if med_hora:
        return float(last_ts) + med_hora, med_hora, "hora"
    if med_global:
        return float(last_ts) + med_global, med_global, "global"
    return None, None, None

def _since_last_white():
    lw = learn_state.get("last_white_ts")
    return (time.time() - float(lw)) if lw else None

def _hour_stats(hh: str):
    slot = learn_state.get("by_hour", {}).get(hh, {"g":0,"l":0})
    g, l = slot.get("g",0), slot.get("l",0)
    tot = g + l
    rate = g / tot if tot > 0 else 0.0
    return rate, tot

def _recent_perf(n=12):
    evts = _carregar_eventos()
    res = [e for e in evts if e.get("tipo") == "resultado"]
    res = res[-n:] if len(res) > n else res
    if not res: return 0.0
    g = sum(1 for e in res if e.get("resultado") == "GREEN")
    return g / len(res)

def _text_signal_score(t: str):
    t0 = _strip_accents(t.lower())
    score = 0.0
    if "entrada" in t0 or "entrar" in t0 or "confirmada" in t0:
        score += 0.35
    if "branco" in t0 or "⚪" in t or "⬜" in t:
        score += 0.35
    if any(x in t0 for x in ["agora", "imediata", "confirmada", "gatilho"]):
        score += 0.15
    if any(x in t0 for x in ["possivel", "possível", "analisando", "teste", "simulacao", "simulação"]):
        score -= 0.25
    return max(0.0, min(score, 1.0))

def _confidence_for_entry(text: str, now_ts: float):
    hh = datetime.now().strftime("%H")
    hour_rate, hour_samples = _hour_stats(hh)   # 0..1, qtd
    recent = _recent_perf(12)                   # 0..1
    txt = _text_signal_score(text)              # 0..1

    # Proximidade do tempo típico desde o último BRANCO (em segundos)
    prox = 0.0
    since = _since_last_white()
    eta_ts, gap_med, fonte = _predict_next_white_eta()
    if since and gap_med:
        ratio = since / gap_med
        if 0.5 <= ratio <= 1.8:
            if ratio <= 1.0:
                prox = (ratio - 0.5) / 0.5  # 0..1
            else:
                prox = max(0.0, 1.0 - (ratio - 1.0) / 0.8)  # 1..0 até 1.8
            prox = max(0.0, min(prox, 1.0))

    # Pesos
    w_hour, w_rec, w_text, w_time = 0.35, 0.25, 0.20, 0.20
    if hour_samples < MIN_SAMPLES:
        w_hour, w_rec, w_text, w_time = 0.20, 0.25, 0.25, 0.30

    conf = w_hour*hour_rate + w_rec*recent + w_text*txt + w_time*prox

    # penalidade se 3 LOSS seguidos
    evts = _carregar_eventos()
    streak = [e.get("resultado") for e in evts if e.get("tipo")=="resultado"][-3:]
    if len(streak)==3 and all(s=="LOSS" for s in streak):
        conf *= 0.9

    conf = max(0.0, min(conf, 1.0))
    return conf, {"hour_rate":hour_rate, "hour_samples":hour_samples, "recent":recent, "txt":txt, "prox":prox, "gap_med":gap_med}

# ========== ROUTES ==========
@app.get("/")
def root():
    return {"status": "ok", "service": "Jonbet - Branco (sinal + reply + aprendizado em segundos + IA)"}

@app.post(f"/webhook/{{webhook_token}}")
async def webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token incorreto.")

    data = await request.json()
    msg = extract_message(data)
    chat_id = str(msg["chat"].get("id"))
    text = (msg["text"] or "").strip()
    tnorm = _strip_accents(text.lower())

    if chat_id not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored_wrong_source"}

    # ===== COMANDOS =====
    if tnorm.startswith("/status"):
        await send_telegram_message(CANAL_DESTINO_ID, get_status_msg())
        return {"ok": True, "action": "status"}

    if tnorm.startswith("/zerar"):
        _save_counters({"date": datetime.now().strftime("%Y-%m-%d"), "green": 0, "loss": 0})
        await send_telegram_message(CANAL_DESTINO_ID, "♻️ Contadores zerados manualmente.")
        return {"ok": True, "action": "reset_manual"}

    if tnorm.startswith("/analise") or tnorm.startswith("/análise"):
        await gerar_analise_ia_e_postar()
        return {"ok": True, "action": "analysis_posted"}

    if tnorm.startswith("/aprendizado"):
        deltas = learn_state.get("deltas", [])
        med = int(median(deltas)) if deltas else 0
        byh = learn_state.get("by_hour", {})
        top = sorted(((h, v.get("g",0), v.get("l",0)) for h,v in byh.items()),
                     key=lambda x: (x[1]-x[2]), reverse=True)[:3]
        linhas = [f"{h}h → G:{g} / L:{l}" for h,g,l in top] or ["(sem dados)"]
        msg_txt = (
            "🧠 *Aprendizado*\n"
            f"• Mediana Δ sinal→placar: {med}s\n"
            f"• RESULT_WINDOW_SECONDS: {RESULT_WINDOW_SECONDS}s\n"
            "• Top horários (G-L):\n- " + "\n- ".join(linhas)
        )
        await send_telegram_message(CANAL_DESTINO_ID, msg_txt)
        return {"ok": True, "action": "learn_status"}

    if tnorm.startswith("/confstatus"):
        conf, parts = _confidence_for_entry("entrada branco ⚪", time.time())
        await send_telegram_message(
            CANAL_DESTINO_ID,
            "🔎 *Confiança agora*\n"
            f"• conf: {conf:.2f} (cutoff {CONF_THRESHOLD})\n"
            f"• hora_rate: {parts['hour_rate']:.2f} (samples {parts['hour_samples']})\n"
            f"• recent: {parts['recent']:.2f}\n"
            f"• texto: {parts['txt']:.2f}\n"
            f"• prox(tempo): {parts['prox']:.2f} • gap_med: {int(parts['gap_med'] or 0)}s"
        )
        return {"ok": True, "action": "conf_status"}

    if tnorm.startswith("/setconf"):
        try:
            val = float(tnorm.split(maxsplit=1)[1])
            global CONF_THRESHOLD
            CONF_THRESHOLD = max(0.0, min(val, 1.0))
            await send_telegram_message(CANAL_DESTINO_ID, f"⚙️ CONF_THRESHOLD atualizado para {CONF_THRESHOLD:.2f}")
        except Exception:
            await send_telegram_message(CANAL_DESTINO_ID, "Uso: /setconf 0.70")
        return {"ok": True, "action": "conf_set"}

    if tnorm.startswith("/forcar"):
        # Força uma entrada independente do corte de confiança
        salvar_evento("entrada")
        sent_id = await send_telegram_message(CANAL_DESTINO_ID, build_final_message())
        if sent_id:
            global last_signal_msg_id, last_signal_time
            last_signal_msg_id = sent_id
            last_signal_time = time.time()
            if SMART_TIMING:
                learn_state["last_entry_ts"] = time.time()
                _save_learn()
            return {"ok": True, "action": "forced_entry"}
        return {"ok": False, "action": "force_failed"}

    if tnorm.startswith("/branco_tempo"):
        eta_ts, gap_med, fonte = _predict_next_white_eta()
        gaps = learn_state.get("white_gaps", [])
        med_g = _med_or_none(gaps)
        byh = learn_state.get("white_gaps_by_hour", {})
        rows = []
        for h, lst in byh.items():
            if len(lst) >= 3:
                rows.append((h, _med_or_none(lst), len(lst)))
        rows.sort(key=lambda x: (x[1] if x[1] is not None else 1e9))
        top = rows[:5]
        linhas = [f"{h}h → med {int(m/60)}m {int(m%60)}s ({n} amostras)" for h,m,n in top if m]
        if not linhas:
            linhas = ["(sem dados por hora)"]
        eta_txt = "indefinido"
        if eta_ts and gap_med:
            mins = int(gap_med // 60)
            secs = int(gap_med % 60)
            when = datetime.fromtimestamp(eta_ts).strftime("%H:%M:%S")
            eta_txt = f"~{mins}m {secs}s após o último • ETA ≈ {when} ({'por ' + fonte})"
        msg_txt = (
            "⏱️ *Tempo entre BRANCOS*\n"
            f"• Mediana global: {int((med_g or 0)//60)}m {int((med_g or 0)%60)}s\n"
            f"• Próximo branco (estimativa): {eta_txt}\n"
            "• Top horas (menor intervalo entre brancos):\n- " + "\n- ".join(linhas)
        )
        await send_telegram_message(CANAL_DESTINO_ID, msg_txt)
        return {"ok": True, "action": "white_gap_status"}

    global last_signal_time, last_signal_msg_id

    # ===== PLACAR (reply ao último sinal recente) =====
    resultado = classificar_resultado(text)
    if resultado in ("GREEN_VALIDO", "LOSS"):
        now = time.time()
        if last_signal_msg_id and (now - last_signal_time <= RESULT_WINDOW_SECONDS):
            salvar_evento("resultado", "GREEN" if resultado == "GREEN_VALIDO" else "LOSS")
            contabilizar("GREEN" if resultado == "GREEN_VALIDO" else "LOSS")

            # aprender delta sinal→placar e ajustar janela
            if SMART_TIMING and learn_state.get("last_entry_ts"):
                delta = time.time() - float(learn_state["last_entry_ts"])
                learn_state["deltas"] = (learn_state.get("deltas", []) + [delta])[-60:]
                learn_state["last_entry_ts"] = None
                try:
                    med = median(learn_state["deltas"])
                    new_win = int(max(180, min(med + 60, 1800)))
                    global RESULT_WINDOW_SECONDS
                    if RESULT_WINDOW_SECONDS == 0 or abs(new_win - RESULT_WINDOW_SECONDS)/max(RESULT_WINDOW_SECONDS,1) > 0.2:
                        RESULT_WINDOW_SECONDS = new_win
                        logging.info(f"🧠 SMART_TIMING: RESULT_WINDOW_SECONDS={RESULT_WINDOW_SECONDS}s (med={int(med)}s)")
                except Exception as e:
                    logging.warning(f"SMART_TIMING mediana erro: {e}")

            # aprender gap entre BRANCOS quando GREEN
            if resultado == "GREEN_VALIDO":
                now_ts = time.time()
                last_w = learn_state.get("last_white_ts")
                if last_w:
                    gap = now_ts - float(last_w)
                    if gap > 0:
                        _append_bounded(learn_state.setdefault("white_gaps", []), gap, maxlen=200)
                        hh_last = datetime.fromtimestamp(float(last_w)).strftime("%H")
                        wb = learn_state.setdefault("white_gaps_by_hour", {})
                        wb_list = wb.get(hh_last, [])
                        _append_bounded(wb_list, gap, maxlen=120)
                        wb[hh_last] = wb_list
                learn_state["last_white_ts"] = now_ts
                _save_learn()

            texto_msg = "✅ **GREEN no BRANCO!** ⚪️" if resultado == "GREEN_VALIDO" else "❌ **LOSS** 😥"
            await send_telegram_message(
                CANAL_DESTINO_ID,
                f"{texto_msg}\n\n{get_status_msg()}",
                reply_to_message_id=last_signal_msg_id,
            )
            logging.info(f"Placar postado como reply ao sinal. ref_msg_id={last_signal_msg_id}")
            # libera para o próximo ciclo
            last_signal_msg_id = None
            last_signal_time = 0
            return {"ok": True, "action": "result_replied"}
        else:
            logging.info("Resultado recebido sem sinal recente -> ignorado.")
            return {"ok": True, "action": "result_ignored_no_recent_signal"}

    # ===== ENTRADA (detecção ampla) =====
    if is_entrada_branco(text):
        # Evitar confundir com resultado
        has_entrada_words = any(w in tnorm for w in ["entrada", "entrar", "entrada confirmada", "confirmada", "aposta", "apostar", "aposte", "aposta confirmada"])
        if is_result_message(tnorm, has_entrada_words):
            logging.info("Ignorado: parece resultado, não entrada.")
            return {"ok": True, "action": "ignored_result_like"}

        if is_pre_signal(tnorm):
            logging.info("Ignorado: possível entrada/análise.")
            return {"ok": True, "action": "ignored_possible_entry"}

        # GATE por confiança (opcional)
        if AUTO_EXECUTE:
            conf, parts = _confidence_for_entry(text, time.time())
            if conf < CONF_THRESHOLD:
                logging.info(f"🔒 AUTO_EXECUTE bloqueou: conf={conf:.2f} cutoff={CONF_THRESHOLD} parts={parts}")
                return {"ok": True, "action": "blocked_low_confidence"}

        # Antiduplicação
        mid = msg.get("message_id")
        if mid and mid in app.state.processed_entries:
            logging.info("Entrada duplicada ignorada.")
            return {"ok": True, "action": "ignored_duplicate_entry"}
        if mid:
            app.state.processed_entries.add(mid)

        now = time.time()
        if COOLDOWN_SECONDS and now - last_signal_time < COOLDOWN_SECONDS:
            logging.info("Cooldown ativo: sinal ignorado.")
            return {"ok": True, "action": "ignored_cooldown"}

        salvar_evento("entrada")
        sent_id = await send_telegram_message(CANAL_DESTINO_ID, build_final_message())
        if sent_id:
            last_signal_msg_id = sent_id
            last_signal_time = now
            if SMART_TIMING:
                learn_state["last_entry_ts"] = time.time()
                _save_learn()
            logging.info(f"Sinal BRANCO enviado. message_id={sent_id}")
        else:
            logging.warning("Sinal enviado, mas sem message_id retornado.")
        return {"ok": True, "action": "signal_sent_white"}

    return {"ok": True, "action": "ignored"}