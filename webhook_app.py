# -*- coding: utf-8 -*-
# Guardião - Contador de Ausências + Alertas
# - Atualiza ausência por número usando mensagens de resultado do canal de origem
# - Alerta imediato se ausência >= ABSENCE_ALERT_AT (padrão 7)
# - Boletim a cada 5 minutos com "Quase 10" (ausência >= 8)
# - Mostra recorde histórico de ausência
#
# ENV obrigatórias:
#   TG_BOT_TOKEN
#   WEBHOOK_TOKEN
#   ALERT_CHANNEL
# ENV opcionais:
#   DB_PATH (default: /data/data.db)
#   ABSENCE_ALERT_AT (default: 7)
#   BULLETIN_INTERVAL_SEC (default: 300)

import os, re, json, time, asyncio, sqlite3
from typing import Optional, List, Dict
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# --------------------
# Config
# --------------------
DB_PATH = os.getenv("DB_PATH", "/data/data.db").strip() or "/data/data.db"
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "").strip()
ALERT_CHANNEL = os.getenv("ALERT_CHANNEL", "").strip()

ABSENCE_ALERT_AT = int(os.getenv("ABSENCE_ALERT_AT", "7"))
BULLETIN_INTERVAL_SEC = int(os.getenv("BULLETIN_INTERVAL_SEC", "300"))

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

app = FastAPI(title="Guardião - Ausências", version="1.0.0")

# --------------------
# DB helpers
# --------------------
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def exec_write(sql: str, params: tuple = ()):
    con = _connect()
    try:
        con.execute(sql, params)
        con.commit()
    finally:
        con.close()

def query_all(sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    con = _connect()
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()

def query_one(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    con = _connect()
    try:
        return con.execute(sql, params).fetchone()
    finally:
        con.close()

def init_db():
    con = _connect()
    cur = con.cursor()
    # timeline simples (opcional – útil se quiser auditar)
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL CHECK(number BETWEEN 1 AND 4)
    )""")
    # ausências por número
    cur.execute("""CREATE TABLE IF NOT EXISTS absences (
        number INTEGER PRIMARY KEY CHECK(number BETWEEN 1 AND 4),
        current_absence INTEGER NOT NULL DEFAULT 0,
        record_absence  INTEGER NOT NULL DEFAULT 0,
        last_alerted    INTEGER NOT NULL DEFAULT 0
    )""")
    con.commit()
    # garantir 1..4
    for n in (1,2,3,4):
        cur.execute("""INSERT OR IGNORE INTO absences (number,current_absence,record_absence,last_alerted)
                       VALUES (?,0,0,0)""", (n,))
    con.commit()
    con.close()

init_db()

# --------------------
# Telegram helpers
# --------------------
_client: Optional[httpx.AsyncClient] = None

async def get_httpx() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=12)
    return _client

@app.on_event("shutdown")
async def _shutdown():
    global _client
    try:
        if _client:
            await _client.aclose()
    except Exception:
        pass

async def tg_send_text(chat_id: str, text: str, parse: str = "HTML"):
    if not TG_BOT_TOKEN or not chat_id:
        return
    try:
        client = await get_httpx()
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                "disable_web_page_preview": True})
    except Exception as e:
        print(f"[TG] send error: {e}")

def now_ts() -> int:
    return int(time.time())

# --------------------
# Parsers (resultado do canal de origem)
# Procuramos o número realmente sorteado na mensagem.
# --------------------
GREEN_PATTERNS = [
    re.compile(r"\bGREEN\b.*?N[úu]mero[:\s]*([1-4])", re.I | re.S),
    re.compile(r"\bGREEN\b.*?\(([1-4])\)", re.I | re.S),
]
RED_PATTERNS = [
    # Algumas mensagens de RED trazem o número que saiu dentro de parênteses; se não houver, ignoramos
    re.compile(r"\bRED\b.*?\(([1-4])\)", re.I | re.S),
    re.compile(r"\bLOSS\b.*?N[úu]mero[:\s]*([1-4])", re.I | re.S),
]

def extract_observed_number(text: str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text or "").strip()
    for rx in GREEN_PATTERNS:
        m = rx.search(t)
        if m:
            try: return int(m.group(1))
            except: pass
    for rx in RED_PATTERNS:
        m = rx.search(t)
        if m:
            try: return int(m.group(1))
            except: pass
    return None

# --------------------
# Ausências
# --------------------
def _get_absences() -> Dict[int, Dict[str, int]]:
    rows = query_all("SELECT number, current_absence, record_absence, last_alerted FROM absences ORDER BY number")
    out: Dict[int, Dict[str,int]] = {}
    for r in rows:
        out[int(r["number"])] = {
            "current": int(r["current_absence"]),
            "record":  int(r["record_absence"]),
            "alerted": int(r["last_alerted"])
        }
    return out

def _set_absence(n: int, current: int = None, record: int = None, alerted: int = None):
    row = query_one("SELECT current_absence, record_absence, last_alerted FROM absences WHERE number=?", (n,))
    if not row: return
    c = int(row["current_absence"])
    r = int(row["record_absence"])
    a = int(row["last_alerted"])
    if current is not None: c = current
    if record  is not None: r = record
    if alerted is not None: a = alerted
    exec_write("""UPDATE absences SET current_absence=?, record_absence=?, last_alerted=? WHERE number=?""",
               (c, r, a, n))

def update_absences_with_observed(observed: int):
    # incrementa todos, zera o sorteado
    rows = query_all("SELECT number, current_absence, record_absence FROM absences")
    for r in rows:
        n = int(r["number"])
        cur = int(r["current_absence"])
        rec = int(r["record_absence"])
        if n == observed:
            cur = 0
        else:
            cur = cur + 1
            if cur > rec:
                rec = cur
        exec_write("UPDATE absences SET current_absence=?, record_absence=? WHERE number=?",
                   (cur, rec, n))

def build_alert_text(n: int, cur: int, rec: int) -> str:
    return (f"⏰ <b>Alerta de ausência</b>\n"
            f"• Número: <b>{n}</b>\n"
            f"• Sem vir: <b>{cur}</b>\n"
            f"• Recorde histórico: <b>{rec}</b>")

def build_bulletin_text(filt: Dict[int,int], records: Dict[int,int]) -> Optional[str]:
    # Exibe "Quase 10 sem vir" para ausências >= 8
    quase = [n for n,v in filt.items() if v >= 8]
    if not quase:
        return None
    linhas = [f"🕒 <b>Quase 10 sem vir</b>"]
    for n in sorted(quase):
        linhas.append(f"• {n} (<b>{filt[n]}</b> sem vir) • Recorde: <b>{records.get(n,0)}</b>")
    return "\n".join(linhas)

async def maybe_send_immediate_alerts():
    absx = _get_absences()
    for n, d in absx.items():
        cur, rec, alerted = d["current"], d["record"], d["alerted"]
        if cur >= ABSENCE_ALERT_AT and cur > alerted:
            await tg_send_text(ALERT_CHANNEL, build_alert_text(n, cur, rec))
            _set_absence(n, alerted=cur)

# --------------------
# Loops em background
# --------------------
async def loop_bulletin():
    while True:
        try:
            await asyncio.sleep(BULLETIN_INTERVAL_SEC)
            absx = _get_absences()
            filt = {n: d["current"] for n,d in absx.items()}
            recs = {n: d["record"] for n,d in absx.items()}
            txt = build_bulletin_text(filt, recs)
            if txt:
                await tg_send_text(ALERT_CHANNEL, txt)
        except Exception as e:
            print(f"[loop_bulletin] error: {e}")

@app.on_event("startup")
async def _boot():
    # dispara boletim periódico
    asyncio.create_task(loop_bulletin())

# --------------------
# API
# --------------------
class Update(BaseModel):
    update_id: int
    channel_post: Optional[dict] = None
    message: Optional[dict] = None
    edited_channel_post: Optional[dict] = None
    edited_message: Optional[dict] = None

@app.get("/")
async def root():
    return {"ok": True, "hint": "POST /webhook/<WEBHOOK_TOKEN>"}

@app.get("/ping")
async def ping():
    try:
        await tg_send_text(ALERT_CHANNEL, "🔧 Bot online (ping).")
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "to": ALERT_CHANNEL}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    data = await request.json()
    upd = Update(**data)
    msg = upd.channel_post or upd.message or upd.edited_channel_post or upd.edited_message
    if not msg:
        return {"ok": True, "note": "no message"}

    text = (msg.get("text") or msg.get("caption") or "").strip()
    # Tenta extrair o número realmente sorteado das mensagens do canal de origem
    observed = extract_observed_number(text)
    if observed is None:
        return {"ok": True, "parsed": False}

    # timeline (opcional)
    exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)", (now_ts(), int(observed)))

    # atualiza ausências e checa alertas
    update_absences_with_observed(int(observed))
    await maybe_send_immediate_alerts()

    # retorno com snapshot atual
    absx = _get_absences()
    snap = {n: d["current"] for n,d in absx.items()}
    recs = {n: d["record"] for n,d in absx.items()}
    return {"ok": True, "observed": observed, "current_absences": snap, "records": recs}