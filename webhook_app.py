# -*- coding: utf-8 -*-
# IA Worker — Guardião Fan Tan (FIRE-only) + Relatório do Guardião
# - Publica apenas sinal de IA (sem replicar texto do canal de origem)
# - Usa o mesmo DB do serviço "canal" para aprender / consultar estatísticas
# - Envia Relatório do Guardião (manual via /debug/flush e automático em intervalo)
#
# ENV necessários (Render):
#   TG_BOT_TOKEN             -> token do bot que VAI POSTAR no canal réplica
#   REPL_CHANNEL             -> ID do canal réplica (ex: -1002796105884)
#   DB_PATH                  -> MESMO caminho do serviço canal (ex: /var/data/data.db/main.sqlite)
#   INTEL_ANALYZE_INTERVAL   -> (opcional) intervalo de análise da IA em segundos (default: 1)
#   FLUSH_KEY                -> (opcional) segredo simples para debug/flush (default: meusegredo123)
#   HEALTH_INTERVAL          -> (opcional) intervalo (s) do relatório automático (default: 1800 -> 30min)
#   SELF_LABEL_IA            -> (opcional) rótulo do sinal IA (default: "Tiro seco por IA")
#
# Procfile:
#   web: uvicorn ia_worker_guardiao:app --host 0.0.0.0 --port $PORT
#
# requirements.txt (mínimo):
#   fastapi
#   uvicorn
#   httpx
#
import os, time, json, sqlite3, asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple
from collections import Counter

import httpx
from fastapi import FastAPI, Query

# ========= ENV =========
DB_PATH = os.getenv("DB_PATH", "/var/data/data.db/main.sqlite").strip()
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
REPL_CHANNEL = os.getenv("REPL_CHANNEL", "").strip()
SELF_LABEL_IA = os.getenv("SELF_LABEL_IA", "Tiro seco por IA").strip()

INTEL_ANALYZE_INTERVAL = float(os.getenv("INTEL_ANALYZE_INTERVAL", "1"))
FLUSH_KEY = os.getenv("FLUSH_KEY", "meusegredo123").strip()
HEALTH_INTERVAL = float(os.getenv("HEALTH_INTERVAL", "1800"))  # 30 min

if not TG_BOT_TOKEN:
    print("⚠️ Defina TG_BOT_TOKEN.")
if not REPL_CHANNEL:
    print("⚠️ Defina REPL_CHANNEL (ID do canal réplica).")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# ========= Hiperparâmetros de IA =========
WINDOW = 400
DECAY  = 0.985
W4, W3, W2, W1 = 0.38, 0.30, 0.20, 0.12
ALPHA, BETA, GAMMA = 1.05, 0.70, 0.40

# Destravamento por confiança (pedido: mínimo 30%)
MIN_CONF_FIRE = 0.30
MIN_SAMPLES = 1000
GAP_MIN = 0.04
CONF_CAP = 0.999

# Antispam baixo delay para envio imediato
MIN_SECONDS_BETWEEN_FIRE = 5
MAX_PER_HOUR = 40

# ========= Estado IA/telemetry =========
def now_ts() -> int: 
    return int(time.time())

_ia_last_reason: str = "—"
_ia_last_reason_ts: int = 0

def _mark_reason(txt: str):
    global _ia_last_reason, _ia_last_reason_ts
    _ia_last_reason = txt
    _ia_last_reason_ts = now_ts()

_last_fire_ts = 0
_sent_this_hour = 0
_hour_bucket = None

def _hour_key() -> int:
    return int(datetime.now(timezone.utc).strftime("%Y%m%d%H"))

def _reset_hour():
    global _sent_this_hour, _hour_bucket
    hb = _hour_key()
    if _hour_bucket != hb:
        _sent_this_hour = 0

# ========= DB helpers (somente leitura/consulta rápida) =========
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def _get(sql: str, params: tuple=()) -> Optional[sqlite3.Row]:
    con=_connect()
    try:
        return con.execute(sql, params).fetchone()
    finally:
        con.close()

def _all(sql: str, params: tuple=()) -> list:
    con=_connect()
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()

# ========= Telegram =========
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN or not chat_id: 
        return
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": parse, "disable_web_page_preview": True},
        )

async def tg_broadcast(text: str, parse: str="HTML"):
    if REPL_CHANNEL:
        await tg_send_text(REPL_CHANNEL, text, parse)

# ========= Modelo (n-gramas) =========
def get_recent_tail(window: int = WINDOW) -> List[int]:
    rows = _all("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (window,))
    return [r["number"] for r in rows][::-1]

def prob_from_ngrams(ctx: List[int], candidate: int) -> float:
    n = len(ctx) + 1
    if n < 2 or n > 5: return 0.0
    ctx_key = ",".join(str(x) for x in ctx)
    row = _get("SELECT SUM(weight) AS w FROM ngram_stats WHERE n=? AND ctx=?", (n, ctx_key))
    tot = (row["w"] or 0.0) if row else 0.0
    if tot <= 0: return 0.0
    row2 = _get("SELECT weight FROM ngram_stats WHERE n=? AND ctx=? AND next=?", (n, ctx_key, candidate))
    w = (row2["weight"] or 0.0) if row2 else 0.0
    return w / tot

def ngram_backoff_score(tail: List[int], candidate: int) -> float:
    if not tail: return 0.0
    ctx4 = tail[-4:] if len(tail)>=4 else []
    ctx3 = tail[-3:] if len(tail)>=3 else []
    ctx2 = tail[-2:] if len(tail)>=2 else []
    ctx1 = tail[-1:] if len(tail)>=1 else []
    parts=[] 
    if len(ctx4)==4: parts.append((W4, prob_from_ngrams(ctx4[:-1], candidate)))
    if len(ctx3)==3: parts.append((W3, prob_from_ngrams(ctx3[:-1], candidate)))
    if len(ctx2)==2: parts.append((W2, prob_from_ngrams(ctx2[:-1], candidate)))
    if len(ctx1)==1: parts.append((W1, prob_from_ngrams(ctx1[:-1], candidate)))
    return sum(w*p for w,p in parts)

def tail_top2_boost(tail: List[int], k:int=40) -> Dict[int, float]:
    boosts={1:1.00, 2:1.00, 3:1.00, 4:1.00}
    if not tail:
        return boosts
    c = Counter(tail[-k:] if len(tail)>=k else tail[:])
    freq = c.most_common()
    if len(freq)>=1: boosts[freq[0][0]]=1.04
    if len(freq)>=2: boosts[freq[1][0]]=1.02
    return boosts

def _after_hint_from_tail(tail: List[int], best:int) -> int:
    """Heurística simples: usa o número mais frequente na cauda (40) diferente de 'best'.
       Se não houver, usa o último número observado."""
    if not tail:
        return 1
    last40 = tail[-40:] if len(tail)>=40 else tail[:]
    c = Counter(last40)
    for num,_ in c.most_common():
        if num != best:
            return num
    return tail[-1]

def suggest_number() -> Tuple[Optional[int], float, int, Dict[int,float], Optional[int]]:
    base=[1,2,3,4]
    tail = get_recent_tail(WINDOW)
    boosts = tail_top2_boost(tail, k=40)
    scores={}
    for c in base:
        ng = ngram_backoff_score(tail, c)
        prior = 1.0/len(base)
        score = prior * ((ng or 1e-6) ** ALPHA) * boosts.get(c,1.0)
        scores[c]=score
    total=sum(scores.values()) or 1e-9
    post={k:v/total for k,v in scores.items()}
    a=sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    if not a:
        _mark_reason("sem_posterior")
        return None,0.0,len(tail),post,None
    gap = (a[0][1] - (a[1][1] if len(a)>1 else 0.0))
    number = a[0][0] if gap >= GAP_MIN else None
    conf = post.get(number,0.0) if number is not None else 0.0

    row = _get("SELECT SUM(weight) AS s FROM ngram_stats")
    samples = int((row["s"] or 0) if row else 0)

    if samples < MIN_SAMPLES:
        _mark_reason(f"amostra_insuficiente({samples}<{MIN_SAMPLES})")
        return None,0.0,samples,post,None

    if number is None or conf < MIN_CONF_FIRE:
        _mark_reason(f"reprovado(conf={conf:.3f}, gap={gap:.3f})")
        return None,conf,samples,post,None

    after_hint = _after_hint_from_tail(tail, number)
    _mark_reason(f"FIRE(best={number}, conf={conf:.3f}, gap={gap:.3f}, tail={len(tail)})")
    return number, conf, samples, post, after_hint

# ========= Relatório do Guardião =========
def _get_scalar(sql:str, params:tuple=(), default:int|float=0):
    try:
        row = _get(sql, params)
        if not row: return default
        try:
            return row[0] if row[0] is not None else default
        except Exception:
            keys = row.keys()
            return row[keys[0]] if keys and row[keys[0]] is not None else default
    except Exception:
        return default

def today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")

def _daily_score_snapshot():
    y = today_key()
    try:
        row = _get("SELECT g0,g1,g2,loss,streak FROM daily_score WHERE yyyymmdd=?", (y,))
        if not row: return 0,0,0,0,0.0
        g0 = row["g0"] or 0
        loss = row["loss"] or 0
        streak = row["streak"] or 0
        total = g0 + loss
        acc = (g0/total*100.0) if total else 0.0
        return g0, loss, streak, total, acc
    except Exception:
        return 0,0,0,0,0.0

def _daily_score_snapshot_ia():
    y = today_key()
    try:
        row = _get("SELECT g0,loss,streak FROM daily_score_ia WHERE yyyymmdd=?", (y,))
        g0 = row["g0"] if row else 0
        loss = row["loss"] if row else 0
        total = g0 + loss
        acc = (g0/total*100.0) if total else 0.0
        return g0, loss, total, acc
    except Exception:
        return 0,0,0,0.0

def _ia_acc_days(days:int=7) -> Tuple[int,int,float]:
    days = max(1, min(int(days), 30))
    try:
        rows = _all("SELECT g0, loss FROM daily_score_ia ORDER BY yyyymmdd DESC LIMIT ?", (days,))
        g = sum((r["g0"] or 0) for r in rows)
        l = sum((r["loss"] or 0) for r in rows)
        acc = (g/(g+l)) if (g+l)>0 else 0.0
        return int(g), int(l), float(acc)
    except Exception:
        return 0,0,0.0

def _health_text() -> str:
    timeline_cnt   = _get_scalar("SELECT COUNT(*) FROM timeline")
    ngram_rows     = _get_scalar("SELECT COUNT(*) FROM ngram_stats")
    ngram_samples  = _get_scalar("SELECT SUM(weight) FROM ngram_stats")
    pat_rows       = _get_scalar("SELECT COUNT(*) FROM stats_pattern")
    pat_events     = _get_scalar("SELECT SUM(wins+losses) FROM stats_pattern")
    strat_rows     = _get_scalar("SELECT COUNT(*) FROM stats_strategy")
    strat_events   = _get_scalar("SELECT SUM(wins+losses) FROM stats_strategy")
    pend_open      = _get_scalar("SELECT COUNT(*) FROM pending_outcome WHERE open=1")

    g0, loss, streak, total, acc = _daily_score_snapshot()
    ia_g0, ia_loss, ia_total, ia_acc = _daily_score_snapshot_ia()

    age = max(0, now_ts() - (_ia_last_reason_ts or now_ts()))
    last_reason_line = f"🤖 Motivo último NÃO-FIRE/FIRE: {_ia_last_reason} (há {age}s)"

    g7, l7, acc7 = _ia_acc_days(7)

    return (
        "🩺 <b>Saúde do Guardião</b>\n"
        f"⏱️ UTC: <code>{datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()}</code>\n"
        "—\n"
        f"🗄️ timeline: <b>{timeline_cnt}</b>\n"
        f"📚 ngram_stats: <b>{ngram_rows}</b> | amostras≈<b>{int(ngram_samples or 0)}</b>\n"
        f"🧩 stats_pattern: chaves=<b>{pat_rows}</b> | eventos=<b>{int(pat_events or 0)}</b>\n"
        f"🧠 stats_strategy: chaves=<b>{strat_rows}</b> | eventos=<b>{int(strat_events or 0)}</b>\n"
        f"⏳ pendências abertas: <b>{pend_open}</b>\n"
        "—\n"
        f"📊 Placar (hoje - G0 only): G0=<b>{g0}</b> | Loss=<b>{loss}</b> | Total=<b>{total}</b>\n"
        f"✅ Acerto: {acc:.2f}% | 🔥 Streak: <b>{streak}</b>\n"
        "—\n"
        f"🤖 IA G0=<b>{ia_g0}</b> | Loss=<b>{ia_loss}</b> | Total=<b>{ia_total}</b>\n"
        f"✅ IA Acerto (dia): {ia_acc:.2f}%\n"
        f"{last_reason_line}\n"
        f"🧮 IA 7d: {(g7+l7)} ops • {acc7*100:.2f}%\n"
    )

# ========= Loop IA =========
async def ia_loop_once():
    global _last_fire_ts, _sent_this_hour
    number, conf, samples, post, after_hint = suggest_number()
    if not number:
        return

    # antispam leve (prioridade: rapidez)
    _reset_hour()
    if _sent_this_hour >= MAX_PER_HOUR:
        _mark_reason("limite_hora")
        return
    if (now_ts() - _last_fire_ts) < MIN_SECONDS_BETWEEN_FIRE:
        _mark_reason("espacamento_minimo")
        return

    conf_capped = max(0.0, min(float(conf), CONF_CAP))
    # Mensagem no formato pedido
    txt = (
        f"🤖 <b>{SELF_LABEL_IA} [FIRE]</b>\n"
        f"🎯 <b>Entrar no {number}</b>\n"
        f"➡️ <i>Depois do {after_hint}</i>\n"
        f"📈 Conf: <b>{conf_capped*100:.2f}%</b> | Amostra≈<b>{samples}</b>"
    )
    await tg_broadcast(txt)
    _last_fire_ts = now_ts()
    _sent_this_hour += 1

# ========= FastAPI =========
app = FastAPI(title="IA Worker — Guardião", version="1.2.0")

@app.get("/")
async def root():
    row = _get("SELECT SUM(weight) AS s FROM ngram_stats")
    samples = int((row["s"] or 0) if row else 0)
    return {"ok": True, "samples": samples, "enough_samples": samples >= MIN_SAMPLES}

@app.on_event("startup")
async def _boot():
    # Loop IA
    async def _loop():
        while True:
            try:
                await ia_loop_once()
            except Exception as e:
                print("[IA] erro:", e)
            await asyncio.sleep(max(0.2, INTEL_ANALYZE_INTERVAL))
    asyncio.create_task(_loop())

    # Relatório automático
    async def _health_loop():
        while True:
            try:
                await tg_broadcast(_health_text())
            except Exception as e:
                print("[HEALTH] erro ao enviar relatório:", e)
            await asyncio.sleep(max(60.0, HEALTH_INTERVAL))
    asyncio.create_task(_health_loop())

# --- Debug endpoints ---
@app.get("/debug/ping")
async def debug_ping(key: str = Query(default="")):
    if not key or key != FLUSH_KEY:
        return {"ok": False, "error": "unauthorized"}
    try:
        await tg_broadcast("🔔 Ping de teste: o bot está conseguindo postar no canal.")
        return {"ok": True, "sent": True, "channel": REPL_CHANNEL}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/debug/say")
async def debug_say(text: str, key: str = Query(default="")):
    if not key or key != FLUSH_KEY:
        return {"ok": False, "error": "unauthorized"}
    try:
        await tg_broadcast(text)
        return {"ok": True, "sent": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/debug/flush")
async def debug_flush(key: str = Query(default="")):
    if not key or key != FLUSH_KEY:
        return {"ok": False, "error": "unauthorized"}
    try:
        await tg_broadcast(_health_text())
        return {"ok": True, "flushed": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
