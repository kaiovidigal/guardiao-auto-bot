# -*- coding: utf-8 -*-
# IA Worker — Guardião Fan Tan (FIRE-only)
# Extrai APENAS a IA (sem parser de canal). Usa o mesmo banco do serviço "canal".
# Requisitos de ENV (mesmos do seu serviço):
#   TG_BOT_TOKEN (obrigatório)
#   REPL_CHANNEL  (canal/ID onde a IA vai publicar os sinais)
#   DB_PATH       (use o MESMO caminho do serviço canal, ex: /var/data/data.db/main.sqlite)
#   INTEL_ANALYZE_INTERVAL (opcional; padrão 2s)
#
# Como usar (Render):
# - Procfile:  web: uvicorn ia_worker:app --host 0.0.0.0 --port $PORT
# - requirements.txt: fastapi uvicorn httpx
# - Deploy apontando para este repo/pasta.
#
# O worker roda em loop e, quando encontra oportunidade, envia:
#   "🤖 Tiro seco por IA [FIRE] ..."
#
import os, time, json, sqlite3, asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple
from collections import Counter

import httpx
from fastapi import FastAPI

# ========= ENV =========
DB_PATH = os.getenv("DB_PATH", "/var/data/data.db/main.sqlite").strip()
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
REPL_CHANNEL = os.getenv("REPL_CHANNEL", "").strip()   # -100...
SELF_LABEL_IA = os.getenv("SELF_LABEL_IA", "Tiro seco por IA").strip()
INTEL_ANALYZE_INTERVAL = float(os.getenv("INTEL_ANALYZE_INTERVAL", "2"))

if not TG_BOT_TOKEN:
    print("⚠️ Defina TG_BOT_TOKEN.")
if not REPL_CHANNEL:
    print("⚠️ Defina REPL_CHANNEL (ID do canal réplica).")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# ========= Hiperparâmetros (iguais ao serviço completo) =========
WINDOW = 400
DECAY  = 0.985
W4, W3, W2, W1 = 0.38, 0.30, 0.20, 0.12
ALPHA, BETA, GAMMA = 1.05, 0.70, 0.40
MIN_SAMPLES = 1000
CONF_CAP = 0.999
GAP_MIN = 0.08

def now_ts() -> int: return int(time.time())

def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def _get(sql: str, params: tuple=()) -> Optional[sqlite3.Row]:
    con=_connect(); r=con.execute(sql, params).fetchone(); con.close(); return r

def _all(sql: str, params: tuple=()) -> list:
    con=_connect(); r=con.execute(sql, params).fetchall(); con.close(); return r

async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN or not chat_id: return
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse, "disable_web_page_preview": True})

async def tg_broadcast(text: str, parse: str="HTML"):
    if REPL_CHANNEL: await tg_send_text(REPL_CHANNEL, text, parse)

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
    parts=[]; 
    if len(ctx4)==4: parts.append((W4, prob_from_ngrams(ctx4[:-1], candidate)))
    if len(ctx3)==3: parts.append((W3, prob_from_ngrams(ctx3[:-1], candidate)))
    if len(ctx2)==2: parts.append((W2, prob_from_ngrams(ctx2[:-1], candidate)))
    if len(ctx1)==1: parts.append((W1, prob_from_ngrams(ctx1[:-1], candidate)))
    return sum(w*p for w,p in parts)

from collections import Counter
def tail_top2_boost(tail: List[int], k:int=40) -> Dict[int, float]:
    boosts={1:1.00,2:1.00,3:1.00,4:1.00}
    if not tail: return boosts
    c = Counter(tail[-k:] if len(tail)>=k else tail[:])
    freq = c.most_common()
    if len(freq)>=1: boosts[freq[0][0]]=1.04
    if len(freq)>=2: boosts[freq[1][0]]=1.02
    return boosts

def suggest_number() -> Tuple[Optional[int], float, int, Dict[int,float]]:
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
    if not a: return None,0.0,len(tail),post
    gap = (a[0][1] - (a[1][1] if len(a)>1 else 0.0))
    number = a[0][0] if gap >= GAP_MIN else None
    conf = post.get(number,0.0) if number is not None else 0.0
    row = _get("SELECT SUM(weight) AS s FROM ngram_stats")
    samples = int((row["s"] or 0) if row else 0)
    if samples < MIN_SAMPLES: return None,0.0,samples,post
    return number, conf, samples, post

_last_fire_ts = 0
MIN_SECONDS_BETWEEN_FIRE = 10
MAX_PER_HOUR = 30
_sent_this_hour=0
_hour_bucket=None
def _reset_hour():
    global _sent_this_hour, _hour_bucket
    hb=int(datetime.now(timezone.utc).strftime("%Y%m%d%H"))
    if _hour_bucket!=hb: _hour_bucket=hb; _sent_this_hour=0

async def ia_loop_once():
    global _last_fire_ts, _sent_this_hour
    number, conf, samples, post = suggest_number()
    if number is None: return
    _reset_hour()
    if _sent_this_hour >= MAX_PER_HOUR: return
    if (now_ts() - _last_fire_ts) < MIN_SECONDS_BETWEEN_FIRE: return
    conf_capped = max(0.0, min(float(conf), CONF_CAP))
    txt = (f"🤖 <b>{SELF_LABEL_IA} [FIRE]</b>\n"
           f"🎯 Número seco (G0): <b>{number}</b>\n"
           f"📈 Conf: <b>{conf_capped*100:.2f}%</b> | Amostra≈<b>{samples}</b>")
    await tg_broadcast(txt)
    _last_fire_ts = now_ts()
    _sent_this_hour += 1

from fastapi import FastAPI
app = FastAPI(title="IA Worker — Guardião", version="1.0.0")

@app.get("/")
async def root():
    row = _get("SELECT SUM(weight) AS s FROM ngram_stats")
    samples = int((row["s"] or 0) if row else 0)
    return {"ok": True, "samples": samples, "enough_samples": samples >= MIN_SAMPLES}

@app.on_event("startup")
async def _boot():
    async def _loop():
        while True:
            try:
                await ia_loop_once()
            except Exception as e:
                print("[IA] erro:", e)
            await asyncio.sleep(max(0.2, INTEL_ANALYZE_INTERVAL))
    asyncio.create_task(_loop())