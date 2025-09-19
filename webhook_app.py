#!/usr/bin/env python3
# -- coding: utf-8 --

"""
guardiao-auto-bot — GEN híbrido + relatório 1h + debug
-> Versão segura para Render: mostra cada mensagem recebida e IA aprende
"""

import os, time, json, sqlite3, asyncio, random
from datetime import datetime, timezone
from typing import List, Dict
import httpx
from fastapi import FastAPI, Request
import zoneinfo

# ===================== ENV =====================
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1003081474331").strip()  # 24 Fan Tan
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "-1002810508717").strip()  # Vidigal
DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"

if not TG_BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
app = FastAPI(title="guardiao-auto-bot (relatório 1h + debug)", version="3.3.0")

# ===================== Fuso =====================
TZ_LOCAL = zoneinfo.ZoneInfo("America/Sao_Paulo")
def now_local(): return datetime.now(TZ_LOCAL)
def day_key(dt=None): dt=dt or now_local(); return dt.strftime("%Y%m%d")
def hour_key(dt=None): dt=dt or now_local(); return dt.strftime("%Y%m%d%H")

# ===================== DB =====================
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

def _exec_write(sql: str, params: tuple=()):
    for attempt in range(6):
        try:
            con = _connect(); cur = con.cursor()
            cur.execute(sql, params)
            con.commit(); con.close(); return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                time.sleep(0.25*(attempt+1)); continue
            raise

def _query_all(sql: str, params: tuple=()) -> List[sqlite3.Row]:
    con = _connect(); cur = con.cursor()
    rows = cur.execute(sql, params).fetchall()
    con.close()
    return rows

def migrate_db():
    con = _connect(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS pending (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER,
        message TEXT,
        predicted TEXT,
        outcome TEXT
    )""")
    con.commit(); con.close()
migrate_db()

# ===================== TELEGRAM =====================
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text,
                                "parse_mode": parse, "disable_web_page_preview": True})

# ===================== IA SIMPLES =====================
def ia_predict(message: str) -> str:
    msg = message.lower()
    if "green" in msg: return "GREEN"
    if "loss" in msg: return "LOSS"

    rows = _query_all("""
        SELECT predicted, outcome FROM pending
        WHERE created_at >= strftime('%s','now','start of day')
        ORDER BY id DESC LIMIT 50
    """)
    if not rows: 
        return random.choice(["GREEN","LOSS"])
    
    g_corr = sum(1 for r in rows if r["predicted"]=="GREEN" and r["outcome"]=="GREEN")
    l_corr = sum(1 for r in rows if r["predicted"]=="LOSS" and r["outcome"]=="LOSS")
    total_corr = g_corr + l_corr
    if total_corr == 0: return random.choice(["GREEN","LOSS"])
    
    return "GREEN" if g_corr/total_corr >= 0.5 else "LOSS"

# ===================== PROCESSAMENTO =====================
async def process_message(message: str):
    predicted = ia_predict(message)
    ts = int(time.time())
    _exec_write("INSERT INTO pending (created_at, message, predicted) VALUES (?,?,?)",
                (ts, message, predicted))
    print(f"[DEBUG] Recebido: {message} → Predição IA: {predicted}")
    await tg_send_text(TARGET_CHANNEL, f"⚡ Sinal automático: {predicted}\nMensagem: {message}")

# ===================== WEBHOOK =====================
@app.post(f"/webhook/{{token}}")
async def webhook(token: str, req: Request):
    if token != WEBHOOK_TOKEN:
        return {"ok": False, "error": "invalid token"}
    data = await req.json()
    if "message" in data:
        chat_id = str(data["message"]["chat"]["id"])
        text = data["message"].get("text","")
        if chat_id == SOURCE_CHANNEL:
            await process_message(text)
    return {"ok": True}

# ===================== RELATÓRIO 1H =====================
REPORT_EVERY_SEC = 60*60

def _report_snapshot_day() -> Dict[str,int]:
    rows = _query_all("SELECT predicted, outcome FROM pending")
    g = sum(1 for r in rows if r["outcome"]=="GREEN")
    l = sum(1 for r in rows if r["outcome"]=="LOSS")
    total = g + l
    acc = (g/total) if total>0 else 0.0
    return {"green":g,"loss":l,"acc":acc}

def _day_mood(acc: float) -> str:
    if acc >= 0.8: return "Dia bom"
    if acc <= 0.5: return "Dia ruim"
    return "Dia neutro"

async def _reporter_loop():
    while True:
        try:
            snap = _report_snapshot_day()
            txt = (
                f"📈 <b>Relatório do dia</b>\n"
                f"📊 GREEN: <b>{snap['green']}</b> × LOSS: <b>{snap['loss']}</b>\n"
                f"Acurácia: {snap['acc']*100:.1f}%\n"
                f"{_day_mood(snap['acc'])}"
            )
            await tg_send_text(TARGET_CHANNEL, txt)
        except Exception as e:
            print(f"[RELATORIO] erro: {e}")
        await asyncio.sleep(REPORT_EVERY_SEC)

# ===================== STARTUP =====================
@app.on_event("startup")
async def _on_startup():
    print("[STARTUP] Bot iniciado, aguardando mensagens...")
    asyncio.create_task(_reporter_loop())

@app.get("/")
async def root():
    return {"ok": True, "service": "guardiao-auto-bot", "target": TARGET_CHANNEL}
