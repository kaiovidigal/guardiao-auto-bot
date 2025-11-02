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
# (import do OpenAI fica lazy dentro da função)

# ========== CONFIG ==========
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "Jonbet")
CANAL_ORIGEM_IDS = [s.strip() for s in os.getenv("CANAL_ORIGEM_IDS", "-1003156785631").split(",")]
CANAL_DESTINO_ID = os.getenv("CANAL_DESTINO_ID", "-1002796105884")

COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "0"))               # 0 = sem travar
RESULT_WINDOW_SECONDS = int(os.getenv("RESULT_WINDOW_SECONDS", "600"))   # pareamento sinal→placar

# Aprendizado & critérios
SMART_TIMING = os.getenv("SMART_TIMING", "true").lower() == "true"
AUTO_EXECUTE = os.getenv("AUTO_EXECUTE", "true").lower() == "true"
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.65"))              # default (pode ser sobrescrito via /setconf)
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "20"))

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

# ========== APP/STATE ==========
app = FastAPI()
last_signal_time = 0
last_signal_msg_id: Optional[int] = None
app.state.processed_entries = set()  # anti-duplicação

# ===== estado de aprendizado =====
learn_state = {
    "deltas": [],              # segundos sinal→placar
    "last_entry_ts": None,     # ts do último sinal
    "by_hour": {},             # {"13":{"g":X,"l":Y}}
    "last_white_ts": None,     # ts do último GREEN BRANCO
    "white_gaps": [],          # segundos entre brancos
    "white_gaps_by_hour": {},  # {"13":[...]} hora do branco anterior
    "conf_threshold": None     # corte persistido via /setconf
}

def _load_learn():
    global learn_state
    try:
        with open(LEARN_PATH, "r") as f:
            data = json.load(f)
        # garante chaves e tipos
        for k, v in learn_state.items():
            data.setdefault(k, v)
        if not isinstance(data.get("deltas"), list): data["deltas"] = []
        if not isinstance(data.get("by_hour"), dict): data["by_hour"] = {}
        if not isinstance(data.get("white_gaps"), list): data["white_gaps"] = []
        if not isinstance(data.get("white_gaps_by_hour"), dict): data["white_gaps_by_hour"] = {}
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

# ---------- helper seguro para cutoff ----------
def _get_cutoff() -> float:
    """
    Retorna um float sempre. Cai para CONF_THRESHOLD se:
      - learn_state['conf_threshold'] não existir
      - for None
      - não for conversível para float
    """
    v = learn_state.get("conf_threshold", None)
    try:
        return float(v) if v is not None else CONF_THRESHOLD
    except Exception:
        return CONF_THRESHOLD

# ========== TELEGRAM ==========
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
            return r.json().get("result", {}).get("message_id")
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

def extract_message(data: dict) -> dict:
    msg = (
        data.get("message") or data.get("channel_post") or
        data.get("edited_message") or data.get("edited_channel_post") or {}
    )
    return {
        "chat": msg.get("chat", {}),
        "text": msg.get("text") or msg.get("caption") or "",
        "message_id": msg.get("message_id")
    }

def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def is_pre_signal(t: str) -> bool:
    return any(w in t for w in ["possivel entrada","possível entrada","analisando","analise","análise","aguarde","ainda nao","ainda não","esperar","esperem"])

def is_result_message(t: str, has_entrada_words: bool) -> bool:
    has_result = any(w in t for w in ["vitoria","vitória","win","loss","derrota","perda","nao bateu","não bateu","nao deu","não deu","falhou"])
    return has_result and not has_entrada_words

def is_entrada_branco(raw: str) -> bool:
    t = _strip_accents(raw.strip().lower())
    has_branco = ("branco" in t) or ("⚪" in raw) or ("⬜" in raw)
    if not has_branco:
        return False
    has_entrada = any(w in t for w in ["entrada","entrar","entrada confirmada","confirmada","confirmado","aposta","apostar","aposte","aposta confirmada"])
    if not has_entrada:
        for p in [
            r'\bentrada\b.{0,40}\b(branco)\b',
            r'\b(apostar|aposta|aposte)\b.{0,60}\b(branco)\b',
            r'\bconfirmad[oa]\b.{0,40}\b(entrada)\b.{0,40}\b(branco)\b',
            r'\b(branco)\b.{0,40}\b(entrada|aposta[rdo]?|confirmad[oa])\b',
            r'\bentrar(?:\s+apos|\s*:\s*)?.{0,10}(branco)\b',
        ]:
            if re.search(p, t):
                has_entrada = True
                break
    return has_entrada and not is_pre_signal(t)

# contadores
def _load_counters():
    try:
        with open(COUNTERS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"date": datetime.now().strftime("%Y-%m-%d"), "green": 0, "loss": 0}

def _save_counters(d):
    open(COUNTERS_PATH, "w").write(json.dumps(d))

def _reset_if_new_day():
    c = _load_counters()
    today = datetime.now().strftime("%Y-%m-%d")
    if c.get("date") != today:
        c = {"date": today, "green": 0, "loss": 0}
        _save_counters(c)
    return c

def contabilizar(r):
    c = _reset_if_new_day()
    c["green" if r == "GREEN" else "loss"] += 1
    _save_counters(c)
    return c

def get_status_msg():
    c = _reset_if_new_day()
    return f"📊 *Parcial do dia* ({c['date']}):\n✅ GREEN: {c['green']}\n❌ LOSS: {c['loss']}"

# classificar resultado
def classificar_resultado(txt: str) -> Optional[str]:
    t = _strip_accents(txt.lower())
    vitoria = any(p in t for p in ["vitoria", "acertamos", "acerto"])
    branco = ("branco" in t) or ("⚪" in txt) or ("⬜" in txt)
    if vitoria and branco and ("protecao" not in t and "como protecao" not in t):
        return "GREEN_VALIDO"
    if any(p in t for p in ["derrota", "loss", "perdeu", "perda", "nao bateu", "nao deu", "falhou"]):
        return "LOSS"
    if any(p in t for p in ["vitoria de primeira", "vitoria com", "gale", "g1", "g 1", "g2", "g 2"]):
        return "LOSS"
    if "green" in t and not branco:
        return "LOSS"
    return None

# IA analítica (import lazy)
def _get_openai_client():
    if not ANALYTICS_MODE or not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI  # import aqui evita ModuleNotFoundError no deploy
        return OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        logging.error("Pacote 'openai' não instalado; IA desativada.")
        return None
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

def _horario(h):
    return h[11:13] if h and len(h) >= 13 else "??"

def _sinal_para_placar_deltas(eventos):
    deltas = []
    last = None
    for e in eventos:
        if e.get("tipo") == "entrada":
            last = e.get("hora")
        elif e.get("tipo") == "resultado" and last:
            try:
                t1 = datetime.strptime(last, "%Y-%m-%d %H:%M:%S").timestamp()
                t2 = datetime.strptime(e["hora"], "%Y-%m-%d %H:%M:%S").timestamp()
                if t2 >= t1:
                    deltas.append(t2 - t1)
                    last = None
            except:
                pass
    return deltas

async def gerar_analise_ia_e_postar():
    client = _get_openai_client()
    if not client:
        await send_telegram_message(CANAL_DESTINO_ID, "⚠️ IA desativada ou sem OPENAI_API_KEY.")
        return
    ev = _carregar_eventos()
    if not ev:
        await send_telegram_message(CANAL_DESTINO_ID, "⚠️ Sem dados ainda para analisar.")
        return
    por_hora_green = {}
    g = l = 0
    for e in ev:
        if e.get("tipo") == "resultado":
            if e.get("resultado") == "GREEN":
                g += 1
                por_hora_green[_horario(e.get("hora"))] = por_hora_green.get(_horario(e.get("hora")), 0) + 1
            elif e.get("resultado") == "LOSS":
                l += 1
    deltas = _sinal_para_placar_deltas(ev)
    med = int(sorted(deltas)[len(deltas)//2]) if deltas else 0
    resumo = json.dumps({
        "greens_por_hora": por_hora_green,
        "greens_total": g,
        "losses_total": l,
        "mediana_segundos_sinal_para_placar": med
    }, ensure_ascii=False, indent=2)
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Seja conciso e direto."},
                {"role": "user", "content": f"Analise estes dados e traga top horários, janela de pareamento e taxa aproximada:\n{resumo}"}
            ],
            temperature=0.2
        )
        await send_telegram_message(CANAL_DESTINO_ID, f"🧠 *Análise IA (BRANCO)*\n\n{r.choices[0].message.content.strip()}")
    except Exception as e:
        await send_telegram_message(CANAL_DESTINO_ID, f"⚠️ Erro ao gerar análise IA: {e}")

# Utilidades tempo/gaps
def _med_or_none(v):
    if not v:
        return None
    try:
        return median(v)
    except:
        return None

def _append_bounded(lst, val, maxlen=200):
    lst.append(val)
    if len(lst) > maxlen:
        del lst[:len(lst) - maxlen]

def _predict_next_white_eta():
    last = learn_state.get("last_white_ts")
    if not last:
        return None, None, None
    try:
        hh_last = datetime.fromtimestamp(float(last)).strftime("%H")
    except:
        hh_last = None
    med_h = None
    if hh_last:
        med_h = _med_or_none(learn_state.get("white_gaps_by_hour", {}).get(hh_last, []))
    med_g = _med_or_none(learn_state.get("white_gaps", []))
    if med_h:
        return float(last) + med_h, med_h, "hora"
    if med_g:
        return float(last) + med_g, med_g, "global"
    return None, None, None

def _since_last_white():
    lw = learn_state.get("last_white_ts")
    return (time.time() - float(lw)) if lw else None

def _hour_stats(hh):
    slot = learn_state.get("by_hour", {}).get(hh, {"g": 0, "l": 0})
    g, l = slot.get("g", 0), slot.get("l", 0)
    tot = g + l
    return (g / tot if tot > 0 else 0.0), tot

def _recent_perf(n=12):
    ev = _carregar_eventos()
    res = [e for e in ev if e.get("tipo") == "resultado"]
    res = res[-n:] if len(res) > n else res
    if not res:
        return 0.0
    return sum(1 for e in res if e.get("resultado") == "GREEN") / len(res)

def _text_signal_score(t):
    t0 = _strip_accents(t.lower())
    s = 0.0
    if any(w in t0 for w in ["entrada", "entrar", "confirmada"]):
        s += 0.35
    if ("branco" in t0) or ("⚪" in t) or ("⬜" in t):
        s += 0.35
    if any(w in t0 for w in ["agora", "imediata", "confirmada", "gatilho"]):
        s += 0.15
    if any(w in t0 for w in ["possivel", "possível", "analisando", "teste", "simulacao", "simulação"]):
        s -= 0.25
    return max(0.0, min(s, 1.0))

def _confidence_for_entry(text, now_ts):
    hh = datetime.now().strftime("%H")
    hour_rate, samples = _hour_stats(hh)
    recent = _recent_perf(12)
    txt = _text_signal_score(text)
    prox = 0.0
    since = _since_last_white()
    _, gap_med, _ = _predict_next_white_eta()
    if since and gap_med:
        ratio = since / gap_med
        if 0.5 <= ratio <= 1.8:
            prox = (ratio - 0.5) / 0.5 if ratio <= 1.0 else max(0.0, 1.0 - (ratio - 1.0) / 0.8)
            prox = max(0.0, min(prox, 1.0))
    w_hour, w_rec, w_txt, w_time = (0.35, 0.25, 0.20, 0.20) if samples >= MIN_SAMPLES else (0.20, 0.25, 0.25, 0.30)
    conf = w_hour * hour_rate + w_rec * recent + w_txt * txt + w_time * prox
    # penalidade por 3 losses seguidos
    ev = _carregar_eventos()
    streak = [e.get("resultado") for e in ev if e.get("tipo") == "resultado"][-3:]
    if len(streak) == 3 and all(s == "LOSS" for s in streak):
        conf *= 0.9
    return max(0.0, min(conf, 1.0)), {"hour_rate": hour_rate, "hour_samples": samples, "recent": recent, "txt": txt, "prox": prox, "gap_med": gap_med}

# ========== ROUTES ==========
@app.get("/")
def root():
    return {"status": "ok", "service": "Jonbet - Branco (aprendizado em segundos + confiança persistida + reply pareado)"}

@app.post(f"/webhook/{{webhook_token}}")
async def webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token incorreto.")

    # globais no topo
    global last_signal_time, last_signal_msg_id, RESULT_WINDOW_SECONDS

    data = await request.json()
    msg = extract_message(data)
    chat_id = str(msg["chat"].get("id"))
    text = (msg["text"] or "").strip()
    tnorm = _strip_accents(text.lower())
    if chat_id not in CANAL_ORIGEM_IDS:
        return {"ok": True, "action": "ignored_wrong_source"}

    # -------- Comandos --------
    if tnorm.startswith("/status"):
        await send_telegram_message(CANAL_DESTINO_ID, get_status_msg()); return {"ok": True}
    if tnorm.startswith("/zerar"):
        _save_counters({"date": datetime.now().strftime("%Y-%m-%d"), "green": 0, "loss": 0})
        await send_telegram_message(CANAL_DESTINO_ID, "♻️ Contadores zerados."); return {"ok": True}
    if tnorm.startswith("/analise") or tnorm.startswith("/análise"):
        await gerar_analise_ia_e_postar(); return {"ok": True}
    if tnorm.startswith("/aprendizado"):
        deltas = learn_state.get("deltas", [])
        med = int(median(deltas)) if deltas else 0
        byh = learn_state.get("by_hour", {})
        top = sorted(((h, v.get("g", 0), v.get("l", 0)) for h, v in byh.items()), key=lambda x: (x[1] - x[2]), reverse=True)[:3]
        linhas = [f"{h}h → G:{g} / L:{l}" for h, g, l in top] or ["(sem dados)"]
        await send_telegram_message(
            CANAL_DESTINO_ID,
            f"🧠 *Aprendizado*\n• Mediana Δ sinal→placar: {med}s\n• RESULT_WINDOW_SECONDS: {RESULT_WINDOW_SECONDS}s\n• Top horários (G-L):\n- " + "\n- ".join(linhas)
        )
        return {"ok": True}
    if tnorm.startswith("/confstatus"):
        cutoff = _get_cutoff()
        conf, parts = _confidence_for_entry("entrada branco ⚪", time.time())
        await send_telegram_message(
            CANAL_DESTINO_ID,
            "🔎 *Confiança agora*\n"
            f"• conf: {conf:.2f} (cutoff {cutoff:.2f})\n"
            f"• hora_rate: {parts['hour_rate']:.2f} (samples {parts['hour_samples']})\n"
            f"• recent: {parts['recent']:.2f}\n"
            f"• texto: {parts['txt']:.2f}\n"
            f"• prox(tempo): {parts['prox']:.2f} • gap_med: {int(parts['gap_med'] or 0)}s"
        )
        return {"ok": True}
    if tnorm.startswith("/setconf"):
        try:
            val = float(tnorm.split(maxsplit=1)[1])
            val = max(0.0, min(val, 1.0))
            learn_state["conf_threshold"] = val
            _save_learn()
            await send_telegram_message(CANAL_DESTINO_ID, f"⚙️ CONF_THRESHOLD atualizado para {val:.2f}")
        except Exception:
            await send_telegram_message(CANAL_DESTINO_ID, "Uso: /setconf 0.70")
        return {"ok": True}
    if tnorm.startswith("/forcar"):
        salvar_evento("entrada")
        sent_id = await send_telegram_message(CANAL_DESTINO_ID, build_final_message())
        if sent_id:
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
        rows = [(h, _med_or_none(lst), len(lst)) for h, lst in byh.items() if len(lst) >= 3]
        rows.sort(key=lambda x: (x[1] if x[1] is not None else 1e9))
        top = rows[:5]
        linhas = [f"{h}h → med {int(m//60)}m {int(m%60)}s ({n} amostras)" for h, m, n in top if m] or ["(sem dados por hora)"]
        eta_txt = "indefinido"
        if eta_ts and gap_med:
            eta_txt = f"~{int(gap_med//60)}m {int(gap_med%60)}s após o último • ETA ≈ {datetime.fromtimestamp(eta_ts).strftime('%H:%M:%S')} ({'por ' + fonte})"
        await send_telegram_message(
            CANAL_DESTINO_ID,
            "⏱️ *Tempo entre BRANCOS*\n"
            f"• Mediana global: {int((med_g or 0)//60)}m {int((med_g or 0)%60)}s\n"
            f"• Próximo branco (estimativa): {eta_txt}\n"
            "• Top horas (menor intervalo):\n- " + "\n- ".join(linhas)
        )
        return {"ok": True}

    # -------- PLACAR (reply no último sinal) --------
    resultado = classificar_resultado(text)
    if resultado in ("GREEN_VALIDO", "LOSS"):
        now = time.time()
        if last_signal_msg_id and (now - last_signal_time <= RESULT_WINDOW_SECONDS):
            salvar_evento("resultado", "GREEN" if resultado == "GREEN_VALIDO" else "LOSS")
            contabilizar("GREEN" if resultado == "GREEN_VALIDO" else "LOSS")

            # aprender delta sinal→placar
            if SMART_TIMING and learn_state.get("last_entry_ts"):
                delta = time.time() - float(learn_state["last_entry_ts"])
                learn_state["deltas"] = (learn_state.get("deltas", []) + [delta])[-60:]
                learn_state["last_entry_ts"] = None
                try:
                    med = median(learn_state["deltas"])
                    new = int(max(180, min(med + 60, 1800)))
                    if RESULT_WINDOW_SECONDS == 0 or abs(new - RESULT_WINDOW_SECONDS) / max(RESULT_WINDOW_SECONDS, 1) > 0.2:
                        RESULT_WINDOW_SECONDS = new
                        logging.info(f"🧠 SMART_TIMING: RESULT_WINDOW_SECONDS={RESULT_WINDOW_SECONDS}s (med={int(med)}s)")
                except Exception as e:
                    logging.warning(f"SMART_TIMING erro: {e}")

            # aprender gap entre brancos quando GREEN
            if resultado == "GREEN_VALIDO":
                now_ts = time.time()
                last_w = learn_state.get("last_white_ts")
                if last_w:
                    gap = now_ts - float(last_w)
                    if gap > 0:
                        lst = learn_state.setdefault("white_gaps", [])
                        _append_bounded(lst, gap, 200)
                        learn_state["white_gaps"] = lst
                        hh_last = datetime.fromtimestamp(float(last_w)).strftime("%H")
                        wb = learn_state.setdefault("white_gaps_by_hour", {})
                        wb_list = wb.get(hh_last, [])
                        _append_bounded(wb_list, gap, 120)
                        wb[hh_last] = wb_list
                learn_state["last_white_ts"] = now_ts
                _save_learn()

            txt = "✅ **GREEN no BRANCO!** ⚪️" if resultado == "GREEN_VALIDO" else "❌ **LOSS** 😥"
            await send_telegram_message(CANAL_DESTINO_ID, f"{txt}\n\n{get_status_msg()}", reply_to_message_id=last_signal_msg_id)
            last_signal_msg_id = None
            last_signal_time = 0
            return {"ok": True, "action": "result_replied"}
        else:
            return {"ok": True, "action": "result_ignored_no_recent_signal"}

    # -------- ENTRADA (detecção ampla) --------
    if is_entrada_branco(text):
        has_ent = any(w in tnorm for w in ["entrada", "entrar", "entrada confirmada", "confirmada", "aposta", "apostar", "aposte", "aposta confirmada"])
        if is_result_message(tnorm, has_ent):
            return {"ok": True, "action": "ignored_result_like"}
        if is_pre_signal(tnorm):
            return {"ok": True, "action": "ignored_possible_entry"}

        if AUTO_EXECUTE:
            cutoff = _get_cutoff()  # <-- SEMPRE float
            conf, parts = _confidence_for_entry(text, time.time())
            if conf < cutoff:
                logging.info(f"🔒 Bloqueado por confiança: {conf:.2f} < {cutoff:.2f} parts={parts}")
                return {"ok": True, "action": "blocked_low_confidence"}

        mid = msg.get("message_id")
        if mid and mid in app.state.processed_entries:
            return {"ok": True, "action": "ignored_duplicate_entry"}
        if mid:
            app.state.processed_entries.add(mid)

        now = time.time()
        if COOLDOWN_SECONDS and now - last_signal_time < COOLDOWN_SECONDS:
            return {"ok": True, "action": "ignored_cooldown"}

        salvar_evento("entrada")
        sent_id = await send_telegram_message(CANAL_DESTINO_ID, build_final_message())
        if sent_id:
            last_signal_msg_id = sent_id
            last_signal_time = now
            if SMART_TIMING:
                learn_state["last_entry_ts"] = time.time()
                _save_learn()
        return {"ok": True, "action": "signal_sent_white"}

    return {"ok": True, "action": "ignored"}