#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GuardiAo Auto Bot — webhook_app.py
v7.2 (parser com fechamento entre parênteses, watchdog anti-trava,
      endpoints admin, dedupe curto)

Start command:
  uvicorn webhook_app:app --host 0.0.0.0 --port $PORT
"""

import os, re, json, time, sqlite3, datetime, hashlib, asyncio
from typing import List, Dict, Optional, Tuple
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "").strip()

SHOW_DEBUG       = os.getenv("SHOW_DEBUG", "False").strip().lower() == "true"
MAX_GALE         = int(os.getenv("MAX_GALE", "1"))
OBS_TIMEOUT_SEC  = int(os.getenv("OBS_TIMEOUT_SEC", "420"))
DEDUP_WINDOW_SEC = int(os.getenv("DEDUP_WINDOW_SEC", "10"))

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN or not TARGET_CHANNEL:
    raise RuntimeError("Faltam ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN, TARGET_CHANNEL.")
TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
DB_PATH = os.getenv("DB_PATH", "/opt/render/project/src/main.sqlite")

app = FastAPI(title="GuardiAo Auto Bot", version="7.2")

# --- DB ----------------------------------------------------------
def _con():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL;")
    return c

def db_init():
    c=_con(); x=c.cursor()
    x.execute("""CREATE TABLE IF NOT EXISTS pending(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER,
        opened_at  INTEGER,
        suggested  INTEGER,
        seen TEXT DEFAULT '',
        open INTEGER DEFAULT 1)""")
    x.execute("""CREATE TABLE IF NOT EXISTS timeline(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER, number INTEGER)""")
    x.execute("""CREATE TABLE IF NOT EXISTS score(
        id INTEGER PRIMARY KEY CHECK(id=1), green INTEGER DEFAULT 0, loss INTEGER DEFAULT 0)""")
    x.execute("""CREATE TABLE IF NOT EXISTS dedupe(kind TEXT, dkey TEXT, ts INTEGER, PRIMARY KEY(kind,dkey))""")
    if not x.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        x.execute("INSERT INTO score(id,green,loss) VALUES(1,0,0)")
    c.commit(); c.close()
db_init()

# --- helpers -----------------------------------------------------
def _pending_get():
    c=_con(); r=c.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone(); c.close(); return r
def _append_seq(seq): 
    if not seq: return
    c=_con(); now=int(time.time()); c.executemany("INSERT INTO timeline(created_at,number) VALUES(?,?)",[(now,int(x)) for x in seq]); c.commit(); c.close()
def _score_add(out):
    c=_con(); g,l=c.execute("SELECT green,loss FROM score WHERE id=1").fetchone(); 
    if out=="GREEN": g+=1
    else: l+=1
    c.execute("UPDATE score SET green=?,loss=? WHERE id=1",(g,l)); c.commit(); c.close()
def _score_text():
    c=_con(); g,l=c.execute("SELECT green,loss FROM score WHERE id=1").fetchone(); c.close(); t=g+l; p=(g/t*100) if t else 0; return f"{g} GREEN × {l} LOSS — {p:.1f}%"
def _dedupe_key(txt): return hashlib.sha1(re.sub(r"\s+"," ",txt.lower()).encode()).hexdigest()
def _seen_recent(k,d):
    now=int(time.time()); c=_con(); r=c.execute("SELECT ts FROM dedupe WHERE kind=? AND dkey=?",(k,d)).fetchone()
    if r and now-int(r["ts"])<=DEDUP_WINDOW_SEC: c.close(); return True
    c.execute("INSERT OR REPLACE INTO dedupe(kind,dkey,ts) VALUES(?,?,?)",(k,d,now)); c.commit(); c.close(); return False

# --- telegram ----------------------------------------------------
async def tg_send(cid,txt):
    try:
        async with httpx.AsyncClient(timeout=15) as cl:
            await cl.post(f"{TELEGRAM_API}/sendMessage",json={"chat_id":cid,"text":txt,"parse_mode":"HTML"})
    except: pass

# --- parse regex -------------------------------------------------
RX_ENTRADA = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
RX_ANALISE = re.compile(r"ANALISANDO", re.I)
RX_FECHA   = re.compile(r"(APOSTA\s+ENCERRADA|FECHAD[OA]|FINALIZAD[OA])", re.I)
RX_NUMS    = re.compile(r"[1-4]")
RX_PAREN   = re.compile(r"\(([^\)]*)\)")
def _parse_paren_pair(t,need=2):
    m=RX_PAREN.findall(t or ""); out=[]
    for chunk in m:
        out += [int(x) for x in RX_NUMS.findall(chunk)]
    return out[:need]

# --- IA simplificada ---------------------------------------------
def _timeline_tail(n=300):
    c=_con(); r=c.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?",(n,)).fetchall(); c.close()
    return [int(x["number"]) for x in r][::-1]
def _freq(tail,k):
    if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
    w=tail[-k:] if len(tail)>=k else tail; tot=len(w)
    return {c:w.count(c)/tot for c in (1,2,3,4)}
def _choose():
    t=_timeline_tail(); m=_freq(t,55)
    best=max(m,key=m.get); conf=m[best]; return best,conf

# --- pending ops -------------------------------------------------
def _pending_open(sug):
    if _pending_get(): return False
    c=_con(); now=int(time.time()); c.execute("INSERT INTO pending(created_at,opened_at,suggested) VALUES(?,?,?)",(now,now,int(sug))); c.commit(); c.close(); return True
def _pending_seen_append(nums,need=2):
    row=_pending_get(); 
    if not row: return
    arr=[s for s in (row["seen"] or "").split("-") if s]
    for n in nums:
        if len(arr)>=need: break
        arr.append(str(int(n)))
    txt="-".join(arr[:need]); c=_con(); c.execute("UPDATE pending SET seen=? WHERE id=?",(txt,row["id"])); c.commit(); c.close()
def _pending_close(final_seen,outcome,sug):
    row=_pending_get(); 
    if not row: return ""
    c=_con(); c.execute("UPDATE pending SET open=0,seen=? WHERE id=?",(final_seen,row["id"])); c.commit(); c.close()
    _score_add(outcome)
    _append_seq([int(x) for x in final_seen.split("-") if x.isdigit()])
    our=sug if outcome=="GREEN" else "X"
    return f"{'🟢' if outcome=='GREEN' else '🔴'} {outcome} — nosso={our}, seq={final_seen}\n📊 {_score_text()}"

# --- watchdog ----------------------------------------------------
def _pending_age():
    r=_pending_get(); 
    if not r: return None
    return int(time.time())-int(r["opened_at"])
async def _wd():
    while True:
        try:
            r=_pending_get()
            if r and _pending_age()>=OBS_TIMEOUT_SEC:
                sug=int(r["suggested"]); seen=[s for s in (r["seen"] or '').split('-') if s]
                while len(seen)<2: seen.append('X')
                msg=_pending_close("-".join(seen[:2]),"LOSS",sug)
                if msg: await tg_send(TARGET_CHANNEL,"⚠️ Timeout — fechando automático.\n"+msg)
        except: pass
        await asyncio.sleep(5)
@app.on_event("startup")
async def _boot(): asyncio.create_task(_wd())

# --- rotas -------------------------------------------------------
@app.get("/health")
async def h(): return {"ok":True,"db":os.path.exists(DB_PATH)}

@app.post("/admin/force-close/{token}")
async def force_close(token:str):
    if token!=WEBHOOK_TOKEN: raise HTTPException(status_code=403)
    r=_pending_get()
    if not r: return {"ok":True,"msg":"no pending"}
    sug=int(r["suggested"]); seen=[s for s in (r["seen"] or '').split('-') if s]
    while len(seen)<2: seen.append('X')
    msg=_pending_close("-".join(seen[:2]),"LOSS",sug)
    if msg: await tg_send(TARGET_CHANNEL,"🛠 Fechamento forçado.\n"+msg)
    return {"ok":True,"msg":"closed"}

# --- webhook principal -------------------------------------------
@app.post("/webhook/{token}")
async def hook(token:str, req:Request):
    if token!=WEBHOOK_TOKEN: raise HTTPException(status_code=403)
    d=await req.json(); txt=(d.get("message",{}).get("text") or d.get("channel_post",{}).get("text") or "").strip()
    if not txt: return {"ok":True}

    # ENTRADA
    if RX_ENTRADA.search(txt):
        best,conf=_choose()
        _pending_open(best)
        await tg_send(TARGET_CHANNEL,f"🤖 <b>IA SUGERE:</b> {best}\nConf {conf*100:.1f}%")
        return {"ok":True,"sug":best}

    # FECHAMENTO
    if RX_FECHA.search(txt) or "GREEN" in txt or "RED" in txt:
        pair=_parse_paren_pair(txt,2)     # PRIORIDADE: números dentro de ()
        if not pair: pair=[int(x) for x in RX_NUMS.findall(txt)][:2]
        if pair: _pending_seen_append(pair,2)
        r=_pending_get()
        if not r: return {"ok":True,"no_pending":True}
        sug=int(r["suggested"]); seen=[s for s in (r["seen"] or '').split('-') if s]
        out="LOSS"
        if seen and seen[0].isdigit() and int(seen[0])==sug: out="GREEN"
        msg=_pending_close("-".join(seen[:2]) or "X",out,sug)
        if msg: await tg_send(TARGET_CHANNEL,msg)
        return {"ok":True,"closed":out}

    # ANALISANDO: alimenta memória
    if RX_ANALISE.search(txt):
        pair=_parse_paren_pair(txt,2)
        if pair: _append_seq(pair)
        return {"ok":True,"analise":len(pair)}

    if SHOW_DEBUG: await tg_send(TARGET_CHANNEL,"DEBUG: msg ignorada")
    return {"ok":True}