# -*- coding: utf-8 -*-
# Guardião — Webhook Híbrido (Canal + IA) com travas MÍNIMAS
# - Escuta o canal de origem via webhook do Telegram
# - Atualiza o banco (timeline / ngram_stats)
# - Gera tiro seco por IA e publica no canal réplica
#
# Endpoints úteis:
#   GET  /                         -> status
#   GET  /debug/samples            -> snapshot de amostras e thresholds
#   GET  /debug/ping?key=...       -> envia ping de teste no canal réplica
#   GET  /debug/force_fire?key=... -> força um FIRE (para teste)
#
# ENV obrigatórias:
#   TG_BOT_TOKEN    -> token do bot que POSTA no canal réplica
#   REPL_CHANNEL    -> chat_id do canal réplica (ex: -1002796105884)
#   WEBHOOK_TOKEN   -> segredo do caminho do webhook (ex: meusegredo123)
#
# ENV recomendadas:
#   DB_PATH         -> caminho do SQLite (default: /var/data/data.db/main.sqlite)
#   SELF_LABEL_IA   -> rótulo do sinal (default: "Tiro seco por IA")
#   FLUSH_KEY       -> chave para /debug (default: meusegredo123)
#
import os, re, time, json, sqlite3, asyncio
from typing import List, Optional, Dict, Tuple
from collections import Counter
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, HTTPException, Query
from pydantic import BaseModel

# ========= ENV =========
DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db/main.sqlite").strip()
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
REPL_CHANNEL   = os.getenv("REPL_CHANNEL", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "meusegredo123").strip()
SELF_LABEL_IA  = os.getenv("SELF_LABEL_IA", "Tiro seco por IA").strip()
FLUSH_KEY      = os.getenv("FLUSH_KEY", "meusegredo123").strip()

if not TG_BOT_TOKEN:
    print("⚠️ Defina TG_BOT_TOKEN.")
if not REPL_CHANNEL:
    print("⚠️ Defina REPL_CHANNEL.")
if not WEBHOOK_TOKEN:
    print("⚠️ Defina WEBHOOK_TOKEN.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# ========= IA — destravada (mínimos) =========
WINDOW   = 100   # analisar cauda curta para reagir rápido
DECAY    = 0.990
W4, W3, W2, W1 = 0.38, 0.30, 0.20, 0.12
ALPHA, BETA, GAMMA = 1.05, 0.70, 0.40

MIN_SAMPLES = 80     # amostra mínima (bem baixa)
GAP_MIN     = 0.01   # gap quase zero
CONF_MIN    = 0.30   # confiança mínima para liberar FIRE

# ========= DB helpers =========
def _ensure_dir(p: str):
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    except Exception:
        pass

def _connect() -> sqlite3.Connection:
    _ensure_dir(DB_PATH)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def exec_write(sql: str, params: tuple = ()):
    con = _connect()
    try:
        con.execute(sql, params)
        con.commit()
    finally:
        con.close()

def query_all(sql: str, params: tuple = ()) -> list:
    con = _connect()
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()

def query_one(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    con = _connect()
    try:
        return con.execute(sql, params).fetchone()
    finally:
        con.close()

def init_db():
    con = _connect()
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS ngram_stats (
        n INTEGER NOT NULL, ctx TEXT NOT NULL, next INTEGER NOT NULL, weight REAL NOT NULL,
        PRIMARY KEY (n, ctx, next)
    )""")
    con.commit()
    con.close()
init_db()

# ========= Telegram =========
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN or not chat_id:
        return
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": parse, "disable_web_page_preview": True})

async def tg_broadcast(text: str, parse: str="HTML"):
    if REPL_CHANNEL:
        await tg_send_text(REPL_CHANNEL, text, parse)

# ========= N-gram timeline =========
def append_timeline(n: int):
    exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)",
               (int(time.time()), int(n)))
    update_ngrams()

def get_recent_tail(window: int = WINDOW) -> List[int]:
    rows = query_all("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (window,))
    return [r["number"] for r in rows][::-1]

def update_ngrams(decay: float = DECAY, max_n: int = 5, window: int = WINDOW):
    tail = get_recent_tail(window)
    if len(tail) < 2:
        return
    for t in range(1, len(tail)):
        nxt = tail[t]
        dist = (len(tail)-1) - t
        w = (decay ** dist)
        for n in range(2, max_n+1):
            if t-(n-1) < 0: break
            ctx = tail[t-(n-1):t]
            ctx_key = ",".join(str(x) for x in ctx)
            exec_write("""
                INSERT INTO ngram_stats (n, ctx, next, weight)
                VALUES (?,?,?,?)
                ON CONFLICT(n, ctx, next) DO UPDATE SET weight = weight + excluded.weight
            """, (n, ctx_key, int(nxt), float(w)))

def prob_from_ngrams(ctx: List[int], candidate: int) -> float:
    n = len(ctx) + 1
    if n < 2 or n > 5: return 0.0
    ctx_key = ",".join(str(x) for x in ctx)
    row = query_one("SELECT SUM(weight) AS w FROM ngram_stats WHERE n=? AND ctx=?", (n, ctx_key))
    tot = (row["w"] or 0.0) if row else 0.0
    if tot <= 0: return 0.0
    row2 = query_one("SELECT weight FROM ngram_stats WHERE n=? AND ctx=? AND next=?", (n, ctx_key, candidate))
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
    if not a:
        return None,0.0,len(tail),post

    # gap quase livre
    gap = (a[0][1] - (a[1][1] if len(a)>1 else 0.0))
    number = a[0][0] if gap >= GAP_MIN else None
    conf = post.get(number,0.0) if number is not None else 0.0

    # amostras totais (peso acumulado)
    row = query_one("SELECT SUM(weight) AS s FROM ngram_stats")
    samples = int((row["s"] or 0) if row else 0)

    # destrava por amostra e confiança
    if samples < MIN_SAMPLES:
        return None, 0.0, samples, post
    if number is None or conf < CONF_MIN:
        return None, 0.0, samples, post

    return number, conf, samples, post

# ========= FastAPI =========
app = FastAPI(title="Guardião Híbrido (Destravado)", version="1.1.0")

class Update(BaseModel):
    update_id: int
    channel_post: Optional[dict] = None
    message: Optional[dict] = None
    edited_channel_post: Optional[dict] = None
    edited_message: Optional[dict] = None

@app.get("/")
async def root():
    row = query_one("SELECT SUM(weight) AS s FROM ngram_stats")
    samples = int((row["s"] or 0) if row else 0)
    return {"ok": True, "samples": samples, "min_samples": MIN_SAMPLES, "enough_samples": samples >= MIN_SAMPLES}

@app.get("/debug/samples")
async def debug_samples():
    row = query_one("SELECT SUM(weight) AS s FROM ngram_stats")
    samples = int((row["s"] or 0) if row else 0)
    return {
        "window": WINDOW,
        "min_samples": MIN_SAMPLES,
        "gap_min": GAP_MIN,
        "conf_min": CONF_MIN,
        "samples": samples,
        "enough_samples": samples >= MIN_SAMPLES
    }

@app.get("/debug/ping")
async def debug_ping(key: str = Query(default="")):
    if not key or key != FLUSH_KEY:
        return {"ok": False, "error": "unauthorized"}
    await tg_broadcast("🔔 Ping de teste: bot online e com permissão de post.")
    return {"ok": True, "sent": True, "channel": REPL_CHANNEL}

@app.get("/debug/force_fire")
async def debug_force_fire(key: str = Query(default="")):
    if not key or key != FLUSH_KEY:
        return {"ok": False, "error": "unauthorized"}
    num, conf, samples, post = suggest_number()
    if not num:
        return {"ok": False, "error": "sem_numero_confiavel", "samples": samples}
    txt = (f"🤖 <b>{SELF_LABEL_IA} [FIRE]</b>\n"
           f"🎯 Número seco (G0): <b>{num}</b>\n"
           f"📈 Conf: <b>{conf*100:.2f}%</b> | Amostra≈<b>{samples}</b>")
    await tg_broadcast(txt)
    return {"ok": True, "fire": num, "conf": conf, "samples": samples}

# ========== Webhook principal (Telegram) ==========
@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()
    # Telegram Update básico
    try:
        msg = data.get("channel_post") or data.get("message") or data.get("edited_channel_post") or data.get("edited_message") or {}
    except Exception:
        msg = {}
    text = (msg.get("text") or msg.get("caption") or "").strip()
    if not text:
        return {"ok": True}

    t = re.sub(r"\s+", " ", text)

    # 1) Se GREEN/RED, registra o número (o ÚLTIMO entre parênteses ou após 'Número:')
    if re.search(r"\bGREEN\b", t, flags=re.I) or re.search(r"\bRED\b", t, flags=re.I):
        # captura número mais à direita
        nums = re.findall(r"[1-4]", t)
        if nums:
            append_timeline(int(nums[-1]))
        return {"ok": True, "logged": True}

    # 2) Se "ANALISANDO" com sequência, alimenta timeline
    if re.search(r"\bANALISANDO\b", t, flags=re.I):
        parts = re.findall(r"[1-4]", t)
        if parts:
            # mantém ordem natural (antigo->novo)
            for n in [int(x) for x in parts]:
                append_timeline(n)
        return {"ok": True, "analise": True}

    # 3) Em "ENTRADA CONFIRMADA", IA dispara (se passar nos mínimos)
    if re.search(r"ENTRADA\s+CONFIRMADA", t, flags=re.I):
        num, conf, samples, post = suggest_number()
        if num:
            txt = (f"🤖 <b>{SELF_LABEL_IA} [FIRE]</b>\n"
                   f"🎯 Número seco (G0): <b>{num}</b>\n"
                   f"📈 Conf: <b>{conf*100:.2f}%</b> | Amostra≈<b>{samples}</b>")
            await tg_broadcast(txt)
            return {"ok": True, "fire": num, "conf": conf, "samples": samples}
        return {"ok": True, "skipped_low_conf_or_samples": True}

    # Outros textos são ignorados (mas não dão erro)
    return {"ok": True, "ignored": True}