# -*- coding: utf-8 -*-
# Guardião Auto — Alertas por contagem de ausência (1..4)
#
# - Lê mensagens do canal de origem via webhook do Telegram (qualquer chat)
# - Extrai números "observados" (1..4) de mensagens do tipo GREEN/LOSS/resultados
# - Mantém timeline em SQLite
# - Envia:
#   (1) Alerta pontual quando algum número atinge N sem vir (default 7)
#   (2) Boletim de 5 em 5 min com números >=8 ("quase 10 sem vir")
#   (3) Mostra recordes históricos (por número e global)
#
# Env:
#   TG_BOT_TOKEN, WEBHOOK_TOKEN, ALERT_CHANNEL
#   DB_PATH=/data/auto.db (default)
#   ALERT_THRESHOLD=7 (default)
#   ALMOST_INTERVAL_MIN=5 (default)

import os, re, time, sqlite3, asyncio
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# -------------------------
# Config / Env
# -------------------------
TG_BOT_TOKEN  = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "").strip()
ALERT_CHANNEL = os.getenv("ALERT_CHANNEL", "").strip()

DB_PATH = os.getenv("DB_PATH", "/data/auto.db").strip() or "/data/auto.db"
ALERT_THRESHOLD = int(os.getenv("ALERT_THRESHOLD", "7"))
ALMOST_INTERVAL_MIN = int(os.getenv("ALMOST_INTERVAL_MIN", "5"))

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

app = FastAPI(title="Guardião Auto — Contagem de Ausência", version="1.0.0")

# -------------------------
# DB helpers
# -------------------------
def _ensure_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def _connect() -> sqlite3.Connection:
    _ensure_db()
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def exec_write(sql: str, params: tuple = ()):
    con = _connect(); cur = con.cursor()
    cur.execute(sql, params); con.commit(); con.close()

def query_all(sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    con = _connect(); rows = con.execute(sql, params).fetchall(); con.close(); return rows

def query_one(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    con = _connect(); row = con.execute(sql, params).fetchone(); con.close(); return row

def init_db():
    con = _connect(); cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS timeline(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts INTEGER NOT NULL,
          number INTEGER NOT NULL CHECK(number BETWEEN 1 AND 4)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alert_state(
          number INTEGER PRIMARY KEY,
          last_sent_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS record_absence(
          number INTEGER PRIMARY KEY,
          max_absence INTEGER NOT NULL DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS meta(
          key TEXT PRIMARY KEY,
          value TEXT
        )
    """)
    con.commit(); con.close()
init_db()

# -------------------------
# Telegram helpers
# -------------------------
_httpx: Optional[httpx.AsyncClient] = None
async def client() -> httpx.AsyncClient:
    global _httpx
    if _httpx is None:
        _httpx = httpx.AsyncClient(timeout=10)
    return _httpx

@app.on_event("shutdown")
async def _shutdown():
    global _httpx
    if _httpx:
        try: await _httpx.aclose()
        except: pass
        _httpx = None

async def tg_send(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN or not chat_id:
        return
    try:
        c = await client()
        await c.post(f"{TELEGRAM_API}/sendMessage",
                     json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                           "disable_web_page_preview": True})
    except Exception as e:
        print(f"[TG] send error: {e}")

# -------------------------
# Parsing de mensagens para achar número observado (1..4)
# (usamos padrões simples e robustos)
# -------------------------
GREEN_RX = [
    re.compile(r"\bGREEN\b.*?N[uú]mero[:\s]*([1-4])", re.I|re.S),
    re.compile(r"\(([1-4])\)\s*$", re.I|re.S),
]
LOSS_RX = [
    re.compile(r"\bLOSS\b.*?N[uú]mero[:\s]*([1-4])", re.I|re.S),
    re.compile(r"\bRED\b.*?\(([1-4])\)", re.I|re.S),
]

def extract_observed(text: str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text).strip()
    for rx in GREEN_RX:
        m = rx.search(t)
        if m: return int(m.group(1))
    for rx in LOSS_RX:
        m = rx.search(t)
        if m: return int(m.group(1))
    # fallback: última linha "(em G0)" com número:
    m = re.search(r"N[uú]mero[:\s]*([1-4])", t, flags=re.I)
    if m: return int(m.group(1))
    return None

# -------------------------
# Lógica de contagem
# -------------------------
def append_timeline(n: int):
    exec_write("INSERT INTO timeline(ts, number) VALUES (?,?)", (int(time.time()), int(n)))

def total_rows() -> int:
    r = query_one("SELECT COUNT(1) AS c FROM timeline")
    return int(r["c"] if r else 0)

def last_id_for(n: int) -> Optional[int]:
    r = query_one("SELECT MAX(id) AS m FROM timeline WHERE number=?", (int(n),))
    return int(r["m"]) if r and r["m"] is not None else None

def current_absences() -> Dict[int, int]:
    """Quantos resultados desde a última vez que cada número apareceu"""
    tot = total_rows()
    out = {}
    for n in (1,2,3,4):
        last = last_id_for(n)
        out[n] = tot - last if last is not None else tot
    return out

def bump_record_if_needed(absences: Dict[int,int]) -> Tuple[int, Dict[int,int]]:
    # recorde por número
    for n,c in absences.items():
        r = query_one("SELECT max_absence FROM record_absence WHERE number=?", (n,))
        old = int(r["max_absence"]) if r else 0
        if c > old:
            exec_write("""
                INSERT INTO record_absence(number, max_absence)
                VALUES(?,?)
                ON CONFLICT(number) DO UPDATE SET max_absence=excluded.max_absence
            """, (n,c))
    # recorde global
    max_now = max(absences.values()) if absences else 0
    r = query_one("SELECT value FROM meta WHERE key='global_record'")
    old = int(r["value"]) if r and r["value"] and r["value"].isdigit() else 0
    if max_now > old:
        exec_write("INSERT OR REPLACE INTO meta(key,value) VALUES('global_record',?)", (str(max_now),))
    # retornar recordes por número
    recs = {}
    rows = query_all("SELECT number,max_absence FROM record_absence ORDER BY number")
    for x in rows: recs[int(x["number"])] = int(x["max_absence"])
    return (max_now, recs)

def last_sent_count(n:int) -> int:
    r = query_one("SELECT last_sent_count FROM alert_state WHERE number=?", (n,))
    return int(r["last_sent_count"]) if r else 0

def set_last_sent_count(n:int, c:int):
    exec_write("""
        INSERT INTO alert_state(number,last_sent_count)
        VALUES(?,?)
        ON CONFLICT(number) DO UPDATE SET last_sent_count=excluded.last_sent_count
    """, (int(n), int(c)))

# -------------------------
# Mensagens
# -------------------------
def fmt_record_line(global_rec:int, per_num:Dict[int,int]) -> str:
    # Ex.: "🔢 Recorde global: 15 | Por número: 1→12, 2→9, 3→15, 4→7"
    parts = [f"{n}→{per_num.get(n,0)}" for n in (1,2,3,4)]
    return f"🔢 <b>Recorde global:</b> {global_rec} | <b>Por número:</b> " + ", ".join(parts)

async def send_threshold_alert(n:int, count:int, global_rec:int, per_num:Dict[int,int]):
    txt = (
        f"⚠️ <b>Alerta</b>: número <b>{n}</b> chegou a <b>{count} sem vir</b>.\n"
        f"{fmt_record_line(global_rec, per_num)}"
    )
    await tg_send(ALERT_CHANNEL, txt)

async def send_almost_list(almost: List[Tuple[int,int]], global_rec:int, per_num:Dict[int,int]):
    # almost = [(n, count), ...] com count >= 8
    if not almost: return
    body = " | ".join([f"{n}: {c} sem vir" for (n,c) in almost])
    txt = f"⏱️ <b>Quase 10 sem vir</b> → {body}\n{fmt_record_line(global_rec, per_num)}"
    await tg_send(ALERT_CHANNEL, txt)

# -------------------------
# Loops de background
# -------------------------
_last_almost_ts: int = 0

async def loop_watch_absences():
    """Checa a cada ~10s e dispara alertas pontuais quando atravessar o threshold"""
    while True:
        try:
            absn = current_absences()
            glob_rec, per_rec = bump_record_if_needed(absn)
            for n,c in absn.items():
                if c >= ALERT_THRESHOLD and c > last_sent_count(n):
                    await send_threshold_alert(n, c, glob_rec, per_rec)
                    set_last_sent_count(n, c)
        except Exception as e:
            print("[loop_watch_absences] error:", e)
        await asyncio.sleep(10)

async def loop_almost_bulletin():
    """A cada ALMOST_INTERVAL_MIN, lista números com >=8 sem vir"""
    global _last_almost_ts
    interval = max(1, ALMOST_INTERVAL_MIN) * 60
    while True:
        try:
            now = int(time.time())
            if now - _last_almost_ts >= interval:
                absn = current_absences()
                glob_rec, per_rec = bump_record_if_needed(absn)
                almost = [(n,c) for (n,c) in absn.items() if c >= 8]
                almost.sort(key=lambda x: x[1], reverse=True)
                await send_almost_list(almost, glob_rec, per_rec)
                _last_almost_ts = now
        except Exception as e:
            print("[loop_almost_bulletin] error:", e)
        await asyncio.sleep(5)

# -------------------------
# Webhook
# -------------------------
class Update(BaseModel):
    update_id: int
    message: Optional[dict] = None
    channel_post: Optional[dict] = None
    edited_message: Optional[dict] = None
    edited_channel_post: Optional[dict] = None

@app.get("/")
async def root():
    return {"ok": True, "detail": "Use POST /webhook/<WEBHOOK_TOKEN>"}

@app.get("/ping_alert")
async def ping_alert():
    await tg_send(ALERT_CHANNEL, "🛠️ Teste: canal de alertas OK.")
    return {"ok": True, "sent_to": ALERT_CHANNEL}

@app.on_event("startup")
async def _boot():
    try: asyncio.create_task(loop_watch_absences())
    except Exception as e: print("startup watch error:", e)
    try: asyncio.create_task(loop_almost_bulletin())
    except Exception as e: print("startup bulletin error:", e)

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    payload = await request.json()
    upd = Update(**payload)
    msg = upd.channel_post or upd.message or upd.edited_channel_post or upd.edited_message
    if not msg: return {"ok": True, "skipped": "no_message"}

    text = (msg.get("text") or msg.get("caption") or "").strip()
    if not text: return {"ok": True, "skipped": "empty"}

    n = extract_observed(text)
    if n is None:
        return {"ok": True, "skipped": "no_number"}

    append_timeline(n)

    # Quando um número vem, zeramos o "last_sent" para ele (evita acumular alertas antigos)
    set_last_sent_count(n, 0)

    return {"ok": True, "observed": n}