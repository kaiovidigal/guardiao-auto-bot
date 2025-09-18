#!/usr/bin/env python3
# -- coding: utf-8 --

"""
guardiao-auto-bot (GEN híbrido + estratégia + guard-rails)
- Fuso America/Sao_Paulo
- Relatório 1x/h por grupo (anti-flood)
- Reset diário local 00:00 BRT
- NÃO encaminha a mensagem original do canal
- Só posta no destino depois de ANALISAR e escolher 1 número seco
"""

import os, re, time, json, sqlite3, asyncio
from typing import List, Optional, Dict
from datetime import datetime, timezone, timedelta
import zoneinfo
import httpx
from fastapi import FastAPI, Request, HTTPException

# ================= ENV =================
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "-1002810508717").strip()   # Vidigal
TARGET_CHANNEL = "-1003081474331"  # Supergrupo Sinal 24 fan tan
DB_PATH        = (os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db")

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN e WEBHOOK_TOKEN no ambiente.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
app = FastAPI(title="guardiao-auto-bot (GEN híbrido + estratégia)", version="3.3.0")

# ================= FUSO / HORÁRIOS =================
TZ_LOCAL = zoneinfo.ZoneInfo("America/Sao_Paulo")
def now_local(): return datetime.now(TZ_LOCAL)
def day_key(dt=None): return (dt or now_local()).strftime("%Y%m%d")
def hour_key(dt=None): return (dt or now_local()).strftime("%Y%m%d%H")
def now_ts(): return int(time.time())
def ts_utc_str(ts=None):
    if ts is None: ts = now_ts()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ================= PARÂMETROS =================
FINAL_TIMEOUT   = 45
REPORT_EVERY_SEC = 60*60
GOOD_DAY_THRESHOLD = 0.80
BAD_DAY_THRESHOLD  = 0.50
QUIET_HOURS = [(0,5)]
_last_cooldown_until = 0

# ================= DB =================
def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

def _exec_write(sql, params=()):
    for attempt in range(6):
        try:
            con = _connect(); cur = con.cursor()
            cur.execute(sql, params); con.commit(); con.close(); return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower(): time.sleep(0.25*(attempt+1)); continue
            raise

def _query_all(sql, params=()): 
    con = _connect(); cur = con.cursor()
    rows = cur.execute(sql, params).fetchall(); con.close(); return rows

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
        opened_at INTEGER,
        d_final INTEGER,
        base TEXT,
        closed_at INTEGER,
        outcome TEXT,
        stage TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS score (
        id INTEGER PRIMARY KEY CHECK (id=1),
        green INTEGER DEFAULT 0,
        loss INTEGER DEFAULT 0
    )""")
    if not cur.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        cur.execute("INSERT INTO score (id, green, loss) VALUES (1,0,0)")
    cur.execute("""CREATE TABLE IF NOT EXISTS relatorio_controle (
        chat_id TEXT NOT NULL,
        hour_key TEXT NOT NULL,
        sent_at INTEGER NOT NULL,
        PRIMARY KEY(chat_id, hour_key)
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS relatorio_dia (
        chat_id TEXT PRIMARY KEY,
        day_key TEXT NOT NULL
    )""")
    con.commit(); con.close()

migrate_db()

# ================= TELEGRAM =================
async def tg_send_text(chat_id, text, parse="HTML"):
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse, "disable_web_page_preview": True})

# ================= UTILIDADES =================
def append_timeline(seq: List[int]):
    for n in seq: _exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)", (now_ts(), int(n)))

def get_tail(window: int):
    rows = _query_all("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (window,))
    return [int(r["number"]) for r in rows][::-1]

def get_open_pending():
    rows = _query_all("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1")
    return rows[0] if rows else None

def _set_seen(row_id, seen_list):
    _exec_write("UPDATE pending SET seen=? WHERE id=?", ("-".join(seen_list[:3]), row_id))

def _close_now(row, suggested, final_seen):
    obs_nums = [int(x) for x in final_seen if x.isdigit()]
    outcome = "GREEN" if suggested in obs_nums else "LOSS"
    _exec_write("UPDATE pending SET open=0, seen=?, closed_at=?, outcome=?, stage=? WHERE id=?",
                ("-".join(final_seen[:3]), now_ts(), outcome, "G0", int(row["id"])))
    _exec_write("UPDATE score SET green=green+1 WHERE id=1" if outcome=="GREEN" else "UPDATE score SET loss=loss+1 WHERE id=1")
    return f"{'🟢' if outcome=='GREEN' else '🔴'} <b>{outcome}</b> — nosso={suggested}, observados={'-'.join(final_seen[:3])}"

def open_pending(suggested, base: List[int]):
    _exec_write("INSERT INTO pending (created_at, suggested, open, seen, opened_at, base) VALUES (?,?,?,?,?,?)",
                (now_ts(), suggested, 1, "", now_ts(), json.dumps(base)))

def parse_entry_text(text):
    t = re.sub(r"\s+", " ", text).strip()
    if "ENTRADA CONFIRMADA" not in t.upper(): return None
    seq_match = re.search(r"Sequ[eê]ncia:\s*([1-4|\s]+)", t, re.I)
    seq = [int(x) for x in re.findall(r"[1-4]", seq_match.group(1))] if seq_match else []
    ssh_match = re.search(r"\bSS?H\s*([1-4])(?:-([1-4]))?(?:-([1-4]))?", t, re.I)
    base = [int(g) for g in ssh_match.groups() if g] if ssh_match else [1,2,3,4]
    return {"seq": seq, "base": base}

# ================= RELATÓRIO =================
def _report_snapshot(last_secs=3600):
    since = now_ts() - max(60, int(last_secs))
    rows = _query_all("SELECT outcome FROM pending WHERE closed_at IS NOT NULL AND closed_at >= ?", (since,))
    g = sum(1 for r in rows if r["outcome"]=="GREEN")
    l = sum(1 for r in rows if r["outcome"]=="LOSS")
    total = g+l; acc = g/total if total>0 else 0.0
    return {"green":g,"loss":l,"acc":acc}

def _day_mood(acc: float):
    if acc >= GOOD_DAY_THRESHOLD: return "🔥 <b>Dia bom</b>"
    if acc <= BAD_DAY_THRESHOLD:  return "⚠️ <b>Dia ruim</b>"
    return "🔎 <b>Dia neutro</b>"

def gua_reset_if_new_day(chat_id):
    dk = day_key()
    rows = _query_all("SELECT day_key FROM relatorio_dia WHERE chat_id=?", (chat_id,))
    if not rows: _exec_write("INSERT OR REPLACE INTO relatorio_dia (chat_id, day_key) VALUES (?,?)", (chat_id, dk))
    elif rows[0]["day_key"] != dk: _exec_write("UPDATE relatorio_dia SET day_key=? WHERE chat_id=?", (dk, chat_id))

async def _reporter_loop():
    while True:
        try:
            gua_reset_if_new_day(TARGET_CHANNEL)
            hk = hour_key(); ts = now_ts()
            _exec_write("INSERT OR IGNORE INTO relatorio_controle(chat_id,hour_key,sent_at) VALUES (?,?,?)", (TARGET_CHANNEL,hk,ts))
            snap = _report_snapshot()
            txt = f"📈 <b>Relatório (última 1h)</b>\nGREEN: {snap['green']} / LOSS: {snap['loss']}\n📊 Dia: {snap['green']} GREEN ×
