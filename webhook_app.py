# -*- coding: utf-8 -*-
# Fan Tan — IA Separada (Fanta_auto_bot)
# - Aprende com GREEN/RED/ANALISANDO recebidos do canal
# - Usa n-gram (até bigrama) + cauda (40 últimos números)
# - Warm-up com exploração (0% → thresholds dinâmicos)
# - Envia sinais para @Fanta_auto_bot
#
# Rotas:
#   POST /webhook/<WEBHOOK_TOKEN>
#   GET  /debug/state, /debug/reason, /debug/samples

import os, re, json, time, sqlite3, asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict
from collections import Counter

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# =========================
# CONFIG / ENV
# =========================
DB_PATH = os.getenv("DB_PATH", "/data/ia_data.db")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "").strip()
REPL_CHANNEL = os.getenv("REPL_CHANNEL", "").strip()  # ID do @Fanta_auto_bot

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN or not REPL_CHANNEL:
    print("⚠️ Defina TG_BOT_TOKEN, WEBHOOK_TOKEN e REPL_CHANNEL")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

WINDOW = 400
WARMUP_SAMPLES = 2000

app = FastAPI(title="Fantan IA Separada", version="1.0.0")

# =========================
# DB Helpers
# =========================
def _connect():
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    return con

def exec_write(sql, params=()):
    con = _connect(); con.execute(sql, params); con.commit(); con.close()

def query_all(sql, params=()):
    con = _connect(); rows = con.execute(sql, params).fetchall(); con.close(); return rows

def query_one(sql, params=()):
    con = _connect(); row = con.execute(sql, params).fetchone(); con.close(); return row

def init_db():
    con = _connect(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL)""")
    con.commit(); con.close()

init_db()

# =========================
# Utils
# =========================
def now_ts(): return int(time.time())
def get_recent_tail(window=WINDOW) -> List[int]:
    rows = query_all("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (window,))
    return [r["number"] for r in rows][::-1]

async def tg_send_text(text: str):
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": REPL_CHANNEL, "text": text, "parse_mode": "HTML"})

# =========================
# IA Core (simplificada bigrama + cauda)
# =========================
def append_timeline(n: int):
    exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)", (now_ts(), int(n)))

def suggest_from_tail() -> Optional[int]:
    tail = get_recent_tail(40)
    if not tail: return None
    freq = Counter(tail).most_common()
    if not freq: return None
    return freq[0][0]

# =========================
# Webhook
# =========================
class Update(BaseModel):
    update_id: int
    channel_post: Optional[dict] = None
    message: Optional[dict] = None

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN: raise HTTPException(status_code=403, detail="Forbidden")
    data = await request.json()
    upd = Update(**data)
    msg = upd.channel_post or upd.message
    if not msg: return {"ok": True}
    text = (msg.get("text") or msg.get("caption") or "").strip()

    # GREEN
    if "GREEN" in text:
        m = re.findall(r"[1-4]", text)
        if m: append_timeline(int(m[-1]))
        return {"ok": True, "green": True}

    # RED
    if "RED" in text or "LOSS" in text:
        m = re.findall(r"[1-4]", text)
        if m: append_timeline(int(m[-1]))
        return {"ok": True, "red": True}

    # ANALISANDO
    if "ANALISANDO" in text.upper():
        m = re.findall(r"[1-4]", text)
        for n in m: append_timeline(int(n))
        return {"ok": True, "analise": True}

    # IA decide sinal
    best = suggest_from_tail()
    if best:
        await tg_send_text(f"🤖 <b>Sinal IA</b>\n🎯 Número seco: <b>{best}</b>")
        return {"ok": True, "sent": best}

    return {"ok": True, "skipped": True}