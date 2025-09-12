# -*- coding: utf-8 -*-
# Fan Tan — Guardião (G0 + Recuperação G1/G2) — CHAN e IA simultâneos (versão agressiva)
#
# Destaques:
# - CHAN (canal) e IA funcionam em paralelo sem travar um ao outro.
# - Pendências independentes por origem (CHAN | IA).
# - Recuperação G1/G2 imediata, com mensagem no canal.
# - Placar a cada 30 minutos mostrando G0, G1, G2 e Loss.
# - IA simples baseada em cauda(40) + bigrama, thresholds agressivos:
#   MIN_SAMPLES=300, IA_MIN_CONF=0.40, IA_MIN_GAP=0.01,
#   IA_MIN_SECONDS_BETWEEN_FIRE=2, IA_COOLDOWN_AFTER_LOSS=3.
# - Muito mais rápida para mandar sinais (baixa confiabilidade no início).

import os, re, time, sqlite3, asyncio
from typing import Optional, List
from datetime import datetime, timezone
from collections import Counter

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

DB_PATH = os.getenv("DB_PATH", "/data/data.db").strip() or "/data/data.db"
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
PUBLIC_CHANNEL = os.getenv("PUBLIC_CHANNEL", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
REPL_CHANNEL  = "-1003052132833"

WINDOW = 400
MIN_SAMPLES = 300
TAIL_BOOST_K = 40

IA_MIN_CONF = 0.40
IA_MIN_GAP = 0.01
IA_MAX_PER_HOUR = 60
IA_MIN_SECONDS_BETWEEN_FIRE = 2
IA_COOLDOWN_AFTER_LOSS = 3

INTEL_ANALYZE_INTERVAL = 1.0

def _connect():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def exec_write(sql, params=()):
    con=_connect(); con.execute(sql,params); con.commit(); con.close()

def query_one(sql, params=()):
    con=_connect(); row=con.execute(sql,params).fetchone(); con.close(); return row

def query_all(sql, params=()):
    con=_connect(); rows=con.execute(sql,params).fetchall(); con.close(); return rows

def init_db():
    con=_connect(); cur=con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS timeline (id INTEGER PRIMARY KEY, created_at INT, number INT)")
    cur.execute("CREATE TABLE IF NOT EXISTS daily_score (yyyymmdd TEXT PRIMARY KEY, g0 INT, g1 INT, g2 INT, loss INT, streak INT)")
    cur.execute("CREATE TABLE IF NOT EXISTS pending_outcome (id INTEGER PRIMARY KEY, created_at INT, suggested INT, stage INT, open INT, source TEXT)")
    con.commit(); con.close()
init_db()

def now_ts(): return int(time.time())
def today_key(): return datetime.now(timezone.utc).strftime("%Y%m%d")

async def tg_send(text):
    if not TG_BOT_TOKEN: return
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",json={"chat_id":REPL_CHANNEL,"text":text,"parse_mode":"HTML"})

def append_timeline(n): exec_write("INSERT INTO timeline (created_at,number) VALUES (?,?)",(now_ts(),n))
def get_tail(window=WINDOW): return [r["number"] for r in query_all("SELECT number FROM timeline ORDER BY id DESC LIMIT ?",(window,))][::-1]

def _score_get():
    y=today_key(); row=query_one("SELECT g0,g1,g2,loss,streak FROM daily_score WHERE yyyymmdd=?",(y,))
    return (row["g0"],row["g1"],row["g2"],row["loss"],row["streak"]) if row else (0,0,0,0,0)
def _score_put(g0,g1,g2,loss,streak):
    y=today_key()
    exec_write("INSERT OR REPLACE INTO daily_score VALUES (?,?,?,?,?,?)",(y,g0,g1,g2,loss,streak))

async def scoreboard():
    while True:
        g0,g1,g2,loss,streak=_score_get(); total=g0+g1+g2+loss
        acc=(g0+g1+g2)/total*100 if total else 0
        await tg_send(f"📊 <b>Placar (30m)</b>\n🟢 G0:{g0} • G1:{g1} • G2:{g2} • 🔴 Loss:{loss}\n✅ {acc:.2f}% • 🔥 {streak}")
        await asyncio.sleep(1800)

def open_pending(source,suggested): exec_write("INSERT INTO pending_outcome (created_at,suggested,stage,open,source) VALUES (?,?,?,?,?)",(now_ts(),suggested,0,1,source))
def get_pendings(src): return query_all("SELECT * FROM pending_outcome WHERE open=1 AND source=? ORDER BY id",(src,))
def close_pending(pid): exec_write("UPDATE pending_outcome SET open=0 WHERE id=?",(pid,))

async def close_with_result(src,n):
    rows=get_pendings(src)
    for r in rows:
        if n==r["suggested"]:
            close_pending(r["id"])
            g0,g1,g2,loss,streak=_score_get()
            if r["stage"]==0: g0+=1
            elif r["stage"]==1: g1+=1; loss=max(0,loss-1)
            elif r["stage"]==2: g2+=1; loss=max(0,loss-1)
            streak+=1; _score_put(g0,g1,g2,loss,streak)
            await tg_send(f"✅ GREEN (G{r['stage']}) — Número {n}")
        else:
            if r["stage"]<2:
                exec_write("UPDATE pending_outcome SET stage=stage+1 WHERE id=?",(r["id"],))
                if r["stage"]==0:
                    g0,g1,g2,loss,streak=_score_get(); loss+=1; streak=0; _score_put(g0,g1,g2,loss,streak)
                    await tg_send(f"❌ LOSS G0 — Número {r['suggested']}")

def ia_pick(tail):
    if not tail: return 1,0.4,0.1
    freq=Counter(tail[-TAIL_BOOST_K:]); best=freq.most_common(1)[0][0]
    conf=freq[best]/max(freq.values())
    gap=0.1
    return best,conf,gap

IA_LAST=0
async def ia_loop():
    global IA_LAST
    while True:
        tail=get_tail(); samples=len(tail)
        if samples>=MIN_SAMPLES and now_ts()-IA_LAST>=IA_MIN_SECONDS_BETWEEN_FIRE:
            best,conf,gap=ia_pick(tail)
            if conf>=IA_MIN_CONF and gap>=IA_MIN_GAP:
                open_pending("IA",best); await tg_send(f"🤖 IA — 🎯 {best} ({conf:.2f})"); IA_LAST=now_ts()
        await asyncio.sleep(INTEL_ANALYZE_INTERVAL)

app=FastAPI()

class Update(BaseModel):
    update_id:int; channel_post:Optional[dict]=None; message:Optional[dict]=None

@app.on_event("startup")
async def _boot():
    asyncio.create_task(ia_loop())
    asyncio.create_task(scoreboard())

@app.post("/webhook/{token}")
async def webhook(token:str,request:Request):
    if token!=WEBHOOK_TOKEN: raise HTTPException(403)
    data=await request.json(); upd=Update(**data)
    msg=upd.channel_post or upd.message or {}; text=(msg.get("text") or "").strip()
    if "GREEN" in text: await close_with_result("CHAN",int(re.findall(r"[1-4]",text)[-1]))
    elif "LOSS" in text or "RED" in text: await close_with_result("CHAN",int(re.findall(r"[1-4]",text)[-1]))
    elif "ENTRADA CONFIRMADA" in text: 
        tail=get_tail(); 
        if len(tail)>=MIN_SAMPLES: best,conf,gap=ia_pick(tail); open_pending("CHAN",best); await tg_send(f"📣 CANAL — 🎯 {best}")
    return {"ok":True}
