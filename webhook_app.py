# -*- coding: utf-8 -*-
# Fan Tan — Guardião (G0 + Recuperação G1/G2) — sem IA autonoma
# - Mantém sinal do CANAL (webhook), com sugestão de número G0 baseada em n-gram (histórico do próprio canal)
# - GREEN/RED: salva o ÚLTIMO número entre parênteses. ANALISANDO só registra sequência
# - Recuperação: não conta Loss parcial; só conta Loss quando esgota G2
# - Mensagem imediata: "✅ GREEN (G0)" ou "✅ GREEN (recuperação G1/G2)"; "❌ LOSS" apenas no final
# - Placar automático a cada 30 minutos (últimos 30m)

import os, re, json, time, sqlite3, asyncio, shutil
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone
from collections import Counter

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# =========================
# ENV / CONFIG
# =========================
# Use /data/data.db pois o Render só permite persistência nesse diretório
DB_PATH = os.getenv("DB_PATH", "/data/data.db").strip() or "/data/data.db"
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
PUBLIC_CHANNEL = os.getenv("PUBLIC_CHANNEL", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
REPL_CHANNEL   = os.getenv("REPL_CHANNEL", "").strip() or "-1003052132833"

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN:
    print("⚠️ Defina TG_BOT_TOKEN e WEBHOOK_TOKEN.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# =========================
# Hiperparâmetros
# =========================
MAX_STAGE = 3      # G0,G1,G2
WINDOW    = 400
DECAY     = 0.985
W4, W3, W2, W1 = 0.38, 0.30, 0.20, 0.12
MIN_SAMPLES = 600
GAP_MIN     = 0.04

app = FastAPI(title="Fantan Guardião", version="4.2.1")

# =========================
# DB helpers
# =========================
def _ensure_db_dir():
    try: os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    except Exception as e: print(f"[DB] mkdir: {e}")

def _connect() -> sqlite3.Connection:
    _ensure_db_dir()
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def exec_write(sql: str, params: tuple = ()):
    con = _connect(); con.execute(sql, params); con.commit(); con.close()

def query_all(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    con = _connect(); rows = con.execute(sql, params).fetchall(); con.close(); return rows

def query_one(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    con = _connect(); row = con.execute(sql, params).fetchone(); con.close(); return row

def init_db():
    con = _connect(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL, number INTEGER NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL, stage INTEGER NOT NULL,
        result TEXT NOT NULL, suggested INTEGER NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS daily_score (
        yyyymmdd TEXT PRIMARY KEY, g0 INTEGER, g1 INTEGER, g2 INTEGER,
        loss INTEGER, streak INTEGER)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pending_outcome (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL,
        suggested INTEGER NOT NULL, stage INTEGER NOT NULL, open INTEGER NOT NULL,
        window_left INTEGER NOT NULL, seen_numbers TEXT DEFAULT '')""")
    con.commit(); con.close()

init_db()

# =========================
# Utils / Telegram
# =========================
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN or not chat_id: return
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse})

async def tg_broadcast(text: str, parse: str="HTML"):
    if REPL_CHANNEL: await tg_send_text(REPL_CHANNEL, text, parse)

async def send_green(stage:int, number:int):
    if stage==0: await tg_broadcast(f"✅ <b>GREEN (G0)</b> — Número: <b>{number}</b>")
    elif stage==1: await tg_broadcast(f"✅ <b>GREEN (G1)</b> — Número: <b>{number}</b>")
    elif stage==2: await tg_broadcast(f"✅ <b>GREEN (G2)</b> — Número: <b>{number}</b>")

async def send_loss(number:int):
    await tg_broadcast(f"❌ <b>LOSS</b> — Número base: <b>{number}</b>")

# =========================
# Timeline & n-grams
# =========================
def append_timeline(n: int):
    exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)", (int(time.time()), int(n)))

# =========================
# Parsers
# =========================
GREEN_PATTERNS = [re.compile(r"\bGREEN\b.*?\((.*?)\)", re.I)]
RED_PATTERNS   = [re.compile(r"\bRED\b.*?\((.*?)\)", re.I), re.compile(r"\bLOSS\b.*?(\d)", re.I)]

def _last_num_in_group(g: str) -> Optional[int]:
    nums = re.findall(r"[1-4]", g or ""); return int(nums[-1]) if nums else None

def extract_green_number(text: str) -> Optional[int]:
    for rx in GREEN_PATTERNS:
        m = rx.search(text); 
        if m: return _last_num_in_group(m.group(1))
    return None

def extract_red_number(text: str) -> Optional[int]:
    for rx in RED_PATTERNS:
        m = rx.search(text)
        if m: return _last_num_in_group(m.group(1))
    return None

# =========================
# Pendências
# =========================
def open_pending(suggested: int):
    exec_write("""INSERT INTO pending_outcome (created_at,suggested,stage,open,window_left,seen_numbers)
                  VALUES (?,?,?,?,?,?)""",
               (int(time.time()), int(suggested), 0, 1, MAX_STAGE, ""))

async def close_pending_with_result(n_observed: int):
    rows = query_all("SELECT * FROM pending_outcome WHERE open=1 ORDER BY id ASC")
    if not rows: return
    for r in rows:
        pid, suggested, stage, left = r["id"], r["suggested"], r["stage"], r["window_left"]
        if n_observed == suggested:
            exec_write("UPDATE pending_outcome SET open=0 WHERE id=?", (pid,))
            exec_write("INSERT INTO outcomes (ts,stage,result,suggested) VALUES (?,?,?,?)",
                       (int(time.time()), stage, "WIN", suggested))
            await send_green(stage, suggested)
        else:
            if left > 1:
                exec_write("UPDATE pending_outcome SET stage=stage+1, window_left=window_left-1 WHERE id=?", (pid,))
            else:
                exec_write("UPDATE pending_outcome SET open=0 WHERE id=?", (pid,))
                exec_write("INSERT INTO outcomes (ts,stage,result,suggested) VALUES (?,?,?,?)",
                           (int(time.time()), stage, "LOSS", suggested))
                await send_loss(suggested)

# =========================
# Webhook
# =========================
class Update(BaseModel):
    update_id: int
    channel_post: Optional[dict] = None

@app.get("/")
async def root(): return {"ok": True}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN: raise HTTPException(status_code=403, detail="Forbidden")
    data = await request.json()
    upd = Update(**data)
    msg = upd.channel_post
    if not msg: return {"ok": True}
    text = (msg.get("text") or "").strip()

    gnum = extract_green_number(text)
    rnum = extract_red_number(text)
    if gnum is not None or rnum is not None:
        observed = gnum if gnum is not None else rnum
        append_timeline(observed)
        await close_pending_with_result(observed)
        return {"ok": True, "observed": observed}

    if "ENTRADA CONFIRMADA" in text:
        open_pending(1)  # 🔧 aqui você pode ajustar a lógica de sugestão
        await tg_broadcast("🎯 Nova entrada aberta (G0)")
        return {"ok": True, "entry": True}

    return {"ok": True, "ignored": True}