# -*- coding: utf-8 -*-
# Guardião — Estratégia "10 sem aparecer" (com pré-alerta 8) — sem repetição de relatório
#
# ENVs:
#   TG_BOT_TOKEN
#   WEBHOOK_TOKEN
#   ANALYZE_CHANNEL   (fonte)  ex.: -1002810508717
#   SIGNAL_CHANNEL    (destino) ex.: -1002796105884
#   DB_PATH           (opcional) default: /data/gap10.db
#   WINDOW            (opcional) default: 2000
#
import os, re, time, sqlite3, asyncio
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

DB_PATH       = os.getenv("DB_PATH", "/data/gap10.db")
TG_BOT_TOKEN  = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "").strip()
ANALYZE_ID    = int(os.getenv("ANALYZE_CHANNEL", "-1002810508717"))
SIGNAL_ID     = int(os.getenv("SIGNAL_CHANNEL",  "-1002796105884"))
WINDOW        = int(os.getenv("WINDOW", "2000"))

FIRE_ABSENCE_THRESHOLD = 10    # dispara sinal
PRE_ALERT_AT           = 8     # alerta imediato (1x por dia/numero)
ALMOST_MIN, ALMOST_MAX = 8, 9  # faixa para relatório
REPORT_PERIOD_SECONDS  = 5*60  # a cada 5 min (mas só se houver mudança)

TELEGRAM_API  = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

app = FastAPI(title="Guardião 10x (no-repeat)", version="1.2.0")

# ---------------- DB
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def exec_write(sql: str, params: tuple = (), retries: int = 8, wait: float = 0.2):
    for _ in range(retries):
        try:
            con = _connect(); con.execute(sql, params); con.commit(); con.close(); return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                time.sleep(wait); continue
            raise
    raise sqlite3.OperationalError("DB bloqueado")

def query_one(sql: str, params: tuple = ()):
    con = _connect(); row = con.execute(sql, params).fetchone(); con.close(); return row

def query_all(sql: str, params: tuple = ()):
    con = _connect(); rows = con.execute(sql, params).fetchall(); con.close(); return rows

def init_db():
    con = _connect(); c = con.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS alerts (
        kind TEXT NOT NULL,    -- 'fire10' | 'pre8' | 'almost_state'
        number INTEGER NOT NULL,
        key TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        PRIMARY KEY (kind, number, key)
    )""")
    con.commit(); con.close()
init_db()

# ---------------- Telegram
_httpx: Optional[httpx.AsyncClient] = None
async def http() -> httpx.AsyncClient:
    global _httpx
    if _httpx is None:
        _httpx = httpx.AsyncClient(timeout=10)
    return _httpx

@app.on_event("shutdown")
async def _shutdown():
    global _httpx
    try:
        if _httpx: await _httpx.aclose()
    except Exception:
        pass

async def tg_send(chat_id: int, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN: return
    try:
        client = await http()
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                "disable_web_page_preview": True})
    except Exception as e:
        print(f"[TG] send error: {e}")

# ---------------- timeline ingest
def now_ts() -> int: return int(time.time())

def append_timeline(n:int):
    exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)", (now_ts(), int(n)))

def get_tail(k:int=WINDOW) -> List[int]:
    rows = query_all("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (k,))
    return [r["number"] for r in rows][::-1]

SEQ_RX = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
def parse_sequence_numbers(text: str) -> List[int]:
    m = SEQ_RX.search(text)
    s = m.group(1) if m else text
    parts = re.findall(r"[1-4]", s)
    if not parts: return []
    # no seu canal, último exibido é o mais recente
    return [int(x) for x in parts][::-1]

# ---------------- gaps
def absence_streaks(tail: List[int]) -> Dict[int,int]:
    out = {1:0,2:0,3:0,4:0}
    for n in (1,2,3,4):
        c=0
        for x in reversed(tail):
            if x == n: break
            c += 1
        out[n]=c
    return out

def _already(kind:str, number:int, key:str) -> bool:
    return bool(query_one("SELECT 1 FROM alerts WHERE kind=? AND number=? AND key=?", (kind, number, key)))

def _mark(kind:str, number:int, key:str):
    exec_write("INSERT OR REPLACE INTO alerts (kind,number,key,created_at) VALUES (?,?,?,?)",
               (kind, number, key, now_ts()))

# ---------------- main checks
async def check_and_fire_if_needed():
    tail = get_tail()
    if not tail: return
    gaps = absence_streaks(tail)
    daykey = datetime.now(timezone.utc).strftime("%Y%m%d")

    # pré alerta 8 (1x/dia por número)
    for n,s in gaps.items():
        if s == PRE_ALERT_AT and not _already("pre8", n, daykey):
            await tg_send(SIGNAL_ID, f"⚠️ Quase 10: número {n} está há 8 sem vir (estratégia teste)")
            _mark("pre8", n, daykey)

    # fire >=10 (1x/dia por número)
    for n,s in gaps.items():
        if s >= FIRE_ABSENCE_THRESHOLD and not _already("fire10", n, daykey):
            await tg_send(SIGNAL_ID, f"Entrar no número {n}  - estratégia teste")
            _mark("fire10", n, daykey)

_last_report_state: Optional[str] = None

async def periodic_report_loop():
    global _last_report_state
    while True:
        try:
            tail = get_tail()
            if tail:
                gaps = absence_streaks(tail)
                quase = sorted([(n,s) for n,s in gaps.items() if ALMOST_MIN <= s <= ALMOST_MAX],
                               key=lambda t: t[1], reverse=True)
                # estado textual (ordenado) para dedupe por conteúdo
                state = "|".join(f"{n}:{s}" for n,s in quase) if quase else ""
                if state and state != _last_report_state and not _already("almost_state", 0, state):
                    txt = " | ".join(f"{n} ({s} sem vir)" for n,s in quase)
                    await tg_send(SIGNAL_ID, f"⏱️ <b>Quase 10 sem vir</b>: {txt}")
                    _mark("almost_state", 0, state)
                    _last_report_state = state
        except Exception as e:
            print(f"[report] erro: {e}")
        await asyncio.sleep(REPORT_PERIOD_SECONDS)

# ---------------- FastAPI
class Update(BaseModel):
    update_id: int
    channel_post: Optional[dict] = None
    message: Optional[dict] = None
    edited_channel_post: Optional[dict] = None
    edited_message: Optional[dict] = None

@app.get("/")
async def root():
    return {"ok": True, "detail": "Guardião 10x — POST /webhook/<WEBHOOK_TOKEN>", "analyze": ANALYZE_ID, "signal": SIGNAL_ID}

@app.get("/ping")
async def ping():
    await tg_send(SIGNAL_ID, "🔧 Bot ON (10x no-repeat)")
    return {"ok": True}

@app.on_event("startup")
async def _boot():
    asyncio.create_task(periodic_report_loop())

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    data = await request.json()
    upd = Update(**data)
    msg = upd.channel_post or upd.message or upd.edited_channel_post or upd.edited_message
    if not msg: return {"ok": True}

    chat = msg.get("chat", {}) or msg.get("sender_chat", {}) or {}
    chat_id = int(chat.get("id") or 0)
    if chat_id != ANALYZE_ID:
        return {"ok": True, "ignored_from": chat_id}

    text = (msg.get("text") or msg.get("caption") or "").strip()
    nums = parse_sequence_numbers(text)
    added = 0
    for n in nums:
        if n in (1,2,3,4):
            append_timeline(n); added += 1

    await check_and_fire_if_needed()
    return {"ok": True, "added": added}
