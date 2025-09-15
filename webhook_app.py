#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time, json, sqlite3
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

if not TG_BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")
if not WEBHOOK_TOKEN:
    raise RuntimeError("Defina WEBHOOK_TOKEN no ambiente.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
app = FastAPI(title="guardiao-auto-bot (GEN híbrido)", version="3.0")

# ========= CONFIG HÍBRIDO =========
SHORT_WINDOW   = 500
LONG_WINDOW    = 3000
CONF_SHORT_MIN = 0.49
CONF_LONG_MIN  = 0.55
GAP_MIN        = 0.040

# ========= TIMEOUTS =========
FINAL_TIMEOUT  = 120  # fecha por X se não veio o 3º número
FORCE_CLOSE_3  = 45   # fecha forçado após 45s no 3º número

def now_ts() -> int:
    return int(time.time())

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
    cur.execute("""CREATE TABLE IF NOT EXISTS pending (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER,
        suggested INTEGER,
        stage INTEGER DEFAULT 0 NOT NULL,
        open INTEGER DEFAULT 1,
        seen TEXT,
        opened_at INTEGER,
        last_conf_short REAL,
        last_conf_long REAL,
        d_final INTEGER
    )""")
    # patch em caso de NULL antigos
    try:
        cur.execute("UPDATE pending SET stage=0 WHERE stage IS NULL")
    except: pass
    con.commit(); con.close()
migrate_db()

# ========= TELEGRAM =========
async def tg_send_text(chat_id: str, text: str):
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"})

# ========= N-GRAM / HÍBRIDO (mesma lógica anterior) =========
def get_tail(window:int) -> List[int]:
    con = _connect()
    rows = con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (window,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def choose_single_number_hybrid(after: Optional[int]) -> Tuple[Optional[int],float,float,int]:
    tail_s = get_tail(SHORT_WINDOW)
    tail_l = get_tail(LONG_WINDOW)
    # ... (mesma lógica do best_conf_gap) ...
    return best, conf_s, conf_l, len(tail_s)

# ========= PENDING HELPERS =========
def get_open_pending() -> Optional[sqlite3.Row]:
    con = _connect()
    row = con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone()
    con.close()
    return row

def _seen_list(row: sqlite3.Row) -> List[str]:
    return [s for s in (row["seen"] or "").split("-") if s]

def _close(row, seen, outcome, stage_lbl):
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE pending SET open=0, seen=? WHERE id=?",
                ("-".join(seen[:3]), row["id"]))
    con.commit(); con.close()
    msg = f"{'🟢' if outcome=='GREEN' else '🔴'} <b>{outcome}</b> — ({stage_lbl}) observados={'-'.join(seen[:3])}"
    return msg

def _maybe_force_close():
    row = get_open_pending()
    if not row: return None
    seen = _seen_list(row)
    if len(seen) == 2 and row["d_final"] and now_ts() > row["d_final"]:
        return _close(row, seen+["X"], "LOSS", "timeout")
    if len(seen) == 3 and row["opened_at"] and now_ts() - row["opened_at"] > FORCE_CLOSE_3:
        return _close(row, seen, "LOSS", "force45s")
    return None

# ========= ROUTES =========
@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    # sempre checa destravamento
    forced = _maybe_force_close()
    if forced:
        await tg_send_text(TARGET_CHANNEL, forced)

    data = await request.json()
    text = (data.get("message") or {}).get("text","").strip()
    if not text: return {"ok":True}

    # TODO: mesma lógica parse ENTRADA e CLOSE de antes
    # Fechamento imediato GREEN quando bate, LOSS no 3º, e fallback timeout

    return {"ok": True}