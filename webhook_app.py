# -*- coding: utf-8 -*-
# Fan Tan — Guardião Híbrido (Canal + IA + Réplica)
# Escuta o canal de origem, atualiza o banco, roda IA e publica no canal réplica.

import os, re, json, time, sqlite3, asyncio
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone, timedelta
from collections import Counter

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# ========= ENV =========
DB_PATH = os.getenv("DB_PATH", "/var/data/main.sqlite").strip()
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
PUBLIC_CHANNEL = os.getenv("PUBLIC_CHANNEL", "").strip()
REPL_CHANNEL   = os.getenv("REPL_CHANNEL", "").strip()
SELF_LABEL_IA  = os.getenv("SELF_LABEL_IA", "Tiro seco por IA").strip()

if not TG_BOT_TOKEN:
    raise RuntimeError("⚠️ Defina TG_BOT_TOKEN.")
if not REPL_CHANNEL:
    print("⚠️ Defina REPL_CHANNEL.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# ========= DB =========
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def exec_write(sql: str, params: tuple = ()):
    con = _connect()
    con.execute(sql, params)
    con.commit()
    con.close()

def query_all(sql: str, params: tuple = ()) -> list:
    con = _connect()
    rows = con.execute(sql, params).fetchall()
    con.close()
    return rows

def query_one(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    con = _connect()
    row = con.execute(sql, params).fetchone()
    con.close()
    return row

def append_timeline(n: int):
    exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)", (int(time.time()), int(n)))

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

# ========= IA =========
WINDOW = 400
DECAY  = 0.985
W4, W3, W2, W1 = 0.38, 0.30, 0.20, 0.12
ALPHA, BETA, GAMMA = 1.05, 0.70, 0.40
MIN_SAMPLES = 1000
GAP_MIN = 0.08

def get_recent_tail(window: int = WINDOW) -> List[int]:
    rows = query_all("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (window,))
    return [r["number"] for r in rows][::-1]

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
    gap = (a[0][1] - (a[1][1] if len(a)>1 else 0.0))
    number = a[0][0] if gap >= GAP_MIN else None
    conf = post.get(number,0.0) if number is not None else 0.0
    row = query_one("SELECT SUM(weight) AS s FROM ngram_stats")
    samples = int((row["s"] or 0) if row else 0)
    if samples < MIN_SAMPLES:
        return None,0.0,samples,post
    return number, conf, samples, post

# ========= FASTAPI =========
app = FastAPI(title="Guardião Híbrido", version="1.0.0")

class Update(BaseModel):
    update_id: int
    channel_post: Optional[dict] = None
    message: Optional[dict] = None
    edited_channel_post: Optional[dict] = None
    edited_message: Optional[dict] = None

@app.get("/")
async def root():
    return {"ok": True, "detail": "Webhook ativo Guardião Híbrido"}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    # Agora o token esperado é o TG_BOT_TOKEN
    if token != TG_BOT_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()
    upd = Update(**data)
    msg = upd.channel_post or upd.message or upd.edited_channel_post or upd.edited_message
    if not msg:
        return {"ok": True}

    text = (msg.get("text") or msg.get("caption") or "").strip()
    t = re.sub(r"\s+", " ", text)

    # Registrar GREEN/RED
    if "GREEN" in t.upper():
        n = re.findall(r"[1-4]", t)
        if n: append_timeline(int(n[0]))
        return {"ok": True, "event": "GREEN"}
    if "RED" in t.upper():
        n = re.findall(r"[1-4]", t)
        if n: append_timeline(int(n[0]))
        return {"ok": True, "event": "RED"}

    # Quando vem entrada confirmada -> IA sugere
    if "ENTRADA CONFIRMADA" in t.upper():
        num, conf, samples, post = suggest_number()
        if num:
            txt = (f"🤖 <b>{SELF_LABEL_IA} [FIRE]</b>\n"
                   f"🎯 Número seco (G0): <b>{num}</b>\n"
                   f"📈 Conf: <b>{conf*100:.2f}%</b> | Amostra≈<b>{samples}</b>")
            await tg_broadcast(txt)
            return {"ok": True, "fire": num, "conf": conf}
        return {"ok": True, "skipped": True}

    return {"ok": True, "ignored": True}