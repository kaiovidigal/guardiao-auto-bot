#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time, json, sqlite3, traceback, asyncio
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone
import httpx
from fastapi import FastAPI, Request, HTTPException

# ========= ENV =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1002796105884").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "-1002810508717").strip()
DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"

if not TG_BOT_TOKEN: raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")
if not WEBHOOK_TOKEN: raise RuntimeError("Defina WEBHOOK_TOKEN no ambiente.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
app = FastAPI(title="guardiao-auto-bot (GEN híbrido + estratégia)", version="3.0.0")

# ========= CONFIG =========
SHORT_WINDOW    = 1000
LONG_WINDOW     = 5000
CONF_SHORT_MIN  = 0.55
CONF_LONG_MIN   = 0.58
GAP_MIN         = 0.040
FINAL_TIMEOUT   = 45  # segundos (quando houver 2 observados)

# ========= Utils =========
def now_ts() -> int: return int(time.time())
def ts_str(ts=None) -> str:
    if ts is None: ts = now_ts()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# ========= DB =========
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    return con

def migrate_db():
    con = _connect(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pending (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER,
        suggested INTEGER,
        open INTEGER DEFAULT 1,
        seen TEXT,
        d_final INTEGER,
        pattern_key TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS score (
        id INTEGER PRIMARY KEY CHECK (id=1),
        g0 INTEGER DEFAULT 0,
        g1 INTEGER DEFAULT 0,
        g2 INTEGER DEFAULT 0,
        loss INTEGER DEFAULT 0
    )""")
    if not cur.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        cur.execute("INSERT INTO score (id,g0,g1,g2,loss) VALUES (1,0,0,0,0)")
    con.commit(); con.close()
migrate_db()

# ========= Telegram =========
async def tg_send(chat_id: str, text: str):
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode":"HTML"})

# ========= Score =========
def bump_score(stage: str):
    con=_connect();cur=con.cursor()
    row=cur.execute("SELECT g0,g1,g2,loss FROM score WHERE id=1").fetchone()
    g0,g1,g2,l=(row["g0"],row["g1"],row["g2"],row["loss"])
    if stage=="G0": g0+=1
    elif stage=="G1": g1+=1
    elif stage=="G2": g2+=1
    elif stage=="LOSS": l+=1
    cur.execute("UPDATE score SET g0=?,g1=?,g2=?,loss=? WHERE id=1",(g0,g1,g2,l))
    con.commit();con.close()

def score_text()->str:
    con=_connect();row=con.execute("SELECT * FROM score WHERE id=1").fetchone();con.close()
    g0,g1,g2,l=row["g0"],row["g1"],row["g2"],row["loss"]
    total=g0+g1+g2+l
    perc=(g0+g1+g2)/total*100 if total>0 else 0
    return f"📊 Hoje: {g0}×G0 | {g1}×G1 | {g2}×G2 | {l}×LOSS — {perc:.1f}%"

# ========= Payload seguro =========
async def read_payload_any(request)->dict:
    try: return await request.json()
    except: pass
    try:
        form=await request.form()
        for k in ("payload","update","data"):
            if k in form:
                try: return json.loads(form[k])
                except: pass
    except: pass
    try:
        raw=await request.body()
        if raw: return json.loads(raw.decode("utf-8","ignore"))
    except: pass
    return {}

# ========= Webhook =========
@app.post("/webhook/{token}")
async def webhook(token:str,request:Request):
    if token!=WEBHOOK_TOKEN: raise HTTPException(status_code=403)
    try:
        data=await read_payload_any(request)
        msg=(data.get("channel_post") or data.get("message") or {})
        text=(msg.get("text") or "").strip()
        chat_id=str((msg.get("chat") or {}).get("id") or "")
        if chat_id!=SOURCE_CHANNEL: return {"ok":True,"skip":"chat"}
        if not text: return {"ok":True,"skip":"no_text"}

        # Aqui você coloca as regras de parse/fechamento (GREEN/LOSS)…

        return {"ok":True,"received":text[:50]}
    except Exception as e:
        print("[WEBHOOK][ERROR]",e,traceback.format_exc(limit=5))
        return {"ok":True,"error":str(e)}

# ========= Reporter (5 em 5 min) =========
async def reporter_loop():
    while True:
        try:
            txt=score_text()
            if "—" in txt: await tg_send(TARGET_CHANNEL,"⏱ Relatório 5min:\n"+txt)
        except Exception as e: print("[REPORT][ERROR]",e)
        await asyncio.sleep(300)  # 5 minutos

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(reporter_loop())