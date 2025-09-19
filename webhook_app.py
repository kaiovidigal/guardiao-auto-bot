#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Bot Guardião Híbrido
- Integração completa: Webhook + Relatório 1h + Reset diário America/Sao_Paulo
- Lê sinais do canal SOURCE_CHANNEL (ex: Vidigal)
- Repassa/adapta para TARGET_CHANNEL (ex: Sinal 24 Fan Tan)
- IA simples com janelas curta/longa, vai se ajustando conforme coleta dados
- Pronto para rodar no Render
"""

import os, time, json, sqlite3, asyncio
from typing import List
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, HTTPException
import zoneinfo

# ========= ENV =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1003804744331").strip()  # Sinal 24 Fan Tan
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "-1002810500000").strip()   # Vidigal (ajustar ID correto)
DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"

if not TG_BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
app = FastAPI(title="guardiao-auto-bot-hibrido", version="5.0.0")

# ========= Fuso local =========
TZ_LOCAL = zoneinfo.ZoneInfo("America/Sao_Paulo")
def now_local(): return datetime.now(TZ_LOCAL)
def day_key(dt=None):
    dt = dt or now_local(); return dt.strftime("%Y%m%d")
def hour_key(dt=None):
    dt = dt or now_local(); return dt.strftime("%Y%m%d%H")

# ========= Parâmetros IA =========
SHORT_WINDOW    = 80
LONG_WINDOW     = 800
CONF_SHORT_MIN  = 0.85
CONF_LONG_MIN   = 0.90
REPORT_EVERY_SEC   = 60 * 60
GOOD_DAY_THRESHOLD = 0.80
BAD_DAY_THRESHOLD  = 0.50

# ========= DB =========
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
    cur.execute("""CREATE TABLE IF NOT EXISTS score (
        id INTEGER PRIMARY KEY CHECK (id=1),
        green INTEGER DEFAULT 0,
        loss  INTEGER DEFAULT 0
    )""")
    if not cur.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        cur.execute("INSERT INTO score (id, green, loss) VALUES (1,0,0)")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS relatorio_controle (
        chat_id TEXT NOT NULL,
        hour_key TEXT NOT NULL,
        sent_at INTEGER NOT NULL,
        PRIMARY KEY (chat_id, hour_key)
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS relatorio_dia (
        chat_id TEXT PRIMARY KEY,
        day_key TEXT NOT NULL
    );
    """)
    con.commit(); con.close()

migrate_db()

# ========= Telegram =========
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text,
                                "parse_mode": parse, "disable_web_page_preview": True})

# ========= Score =========
def bump_score(outcome: str):
    con = _connect(); cur = con.cursor()
    row = cur.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
    g, l = (row["green"], row["loss"]) if row else (0, 0)
    if outcome.upper() == "GREEN": g += 1
    elif outcome.upper() == "LOSS": l += 1
    cur.execute("INSERT OR REPLACE INTO score (id, green, loss) VALUES (1,?,?)", (g, l))
    con.commit(); con.close()

def score_text() -> str:
    con = _connect()
    row = con.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
    con.close()
    if not row: return "0 GREEN × 0 LOSS — 0.0%"
    g, l = int(row["green"]), int(row["loss"])
    total = g + l
    acc = (g/total*100.0) if total>0 else 0.0
    return f"{g} GREEN × {l} LOSS — {acc:.1f}%"

# ========= Relatório =========
def _day_mood(acc: float) -> str:
    if acc >= GOOD_DAY_THRESHOLD: return "Dia bom"
    if acc <= BAD_DAY_THRESHOLD:  return "Dia ruim"
    return "Dia neutro"

def gua_reset_if_new_day(chat_id: str):
    dk = day_key()
    rows = _query_all("SELECT day_key FROM relatorio_dia WHERE chat_id=?", (chat_id,))
    if not rows:
        _exec_write("INSERT OR REPLACE INTO relatorio_dia (chat_id, day_key) VALUES (?,?)", (chat_id, dk))
        return
    if rows[0]["day_key"] != dk:
        _exec_write("UPDATE relatorio_dia SET day_key=? WHERE chat_id=?", (dk, chat_id))

def gua_try_reserve_hour(chat_id: str) -> bool:
    hk = hour_key(); ts = int(time.time())
    try:
        _exec_write(
            "INSERT OR IGNORE INTO relatorio_controle(chat_id, hour_key, sent_at) VALUES (?,?,?)",
            (chat_id, hk, ts)
        )
        rows = _query_all(
            "SELECT sent_at FROM relatorio_controle WHERE chat_id=? AND hour_key=?",
            (chat_id, hk)
        )
        return bool(rows) and int(rows[0]["sent_at"]) == ts
    except Exception:
        return False

async def _reporter_loop():
    while True:
        try:
            chat_id = TARGET_CHANNEL
            gua_reset_if_new_day(str(chat_id))
            if gua_try_reserve_hour(str(chat_id)):
                txt = (
                    "📈 <b>Relatório do dia</b>\n"
                    f"{score_text()}\n"
                )
                await tg_send_text(TARGET_CHANNEL, txt)
        except Exception as e:
            print(f"[RELATORIO] erro: {e}")
        await asyncio.sleep(REPORT_EVERY_SEC)

# ========= Webhook =========
@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if WEBHOOK_TOKEN and token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token inválido")
    body = await request.json()
    msg = body.get("message", {}).get("text")
    if msg:
        # IA simples: só encaminhar por enquanto
        await tg_send_text(TARGET_CHANNEL, f"🔔 <b>Sinal recebido:</b> {msg}")
    return {"ok": True}

# ========= Vida da app =========
@app.on_event("startup")
async def _on_startup():
    asyncio.create_task(_reporter_loop())

@app.get("/")
async def root():
    return {"ok": True, "service": "guardiao-auto-bot-hibrido", "source": SOURCE_CHANNEL, "target": TARGET_CHANNEL}
