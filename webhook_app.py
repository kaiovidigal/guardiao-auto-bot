# -*- coding: utf-8 -*-
# Guardião — Estratégia "10 sem aparecer" (análise -> envio)
# - Lê apenas do canal de análise (ANALYZE_CHANNEL).
# - Envia sinais e relatórios para SIGNAL_CHANNEL.
# - Sinal imediato: quando algum número (1..4) fica >=10 rodadas sem aparecer.
#   Mensagem: "Entrar no número X  - estratégia teste"
# - Relatório a cada 5 minutos: números com sequência de ausência 8 ou 9 ("quase chegando 10").
#
# Requisitos: fastapi, httpx, pydantic
#
# ENVs:
#   TG_BOT_TOKEN
#   WEBHOOK_TOKEN
#   ANALYZE_CHANNEL   (ex.: -1002810508717)  [de onde chegam as sequências]
#   SIGNAL_CHANNEL    (ex.: -1002796105884)  [pra onde enviar os sinais]
#   DB_PATH (opcional, default: /data/gap10.db)
#   WINDOW  (tamanho da cauda usada, default: 2000)
#
import os, re, time, json, sqlite3, asyncio
from typing import Optional, List, Dict
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# =========================
# Config
# =========================
DB_PATH       = os.getenv("DB_PATH", "/data/gap10.db")
TG_BOT_TOKEN  = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "").strip()
ANALYZE_ID    = int(os.getenv("ANALYZE_CHANNEL", "-1002810508717"))
SIGNAL_ID     = int(os.getenv("SIGNAL_CHANNEL",  "-1002796105884"))
WINDOW        = int(os.getenv("WINDOW", "2000"))

TELEGRAM_API  = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

FIRE_ABSENCE_THRESHOLD = 10     # dispara sinal
ALMOST_MIN, ALMOST_MAX = 8, 9   # para relatório quinquenal
REPORT_PERIOD_SECONDS  = 5*60   # 5 minutos

app = FastAPI(title="Guardião — 10 sem aparecer", version="1.0.0")

# =========================
# DB helpers
# =========================
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

def query_all(sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    con = _connect(); rows = con.execute(sql, params).fetchall(); con.close(); return rows

def query_one(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    con = _connect(); row  = con.execute(sql, params).fetchone();  con.close(); return row

def init_db():
    con = _connect(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    # evita repetir alertas iguais muitas vezes
    cur.execute("""CREATE TABLE IF NOT EXISTS alerts (
        kind TEXT NOT NULL,    -- 'fire10' ou 'almost'
        number INTEGER NOT NULL,
        key TEXT NOT NULL,     -- chave de dedupe (e.g., dia ou bucket horário)
        created_at INTEGER NOT NULL,
        PRIMARY KEY (kind, number, key)
    )""")
    con.commit(); con.close()
init_db()

# =========================
# Telegram
# =========================
_httpx_client: Optional[httpx.AsyncClient] = None
async def http() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(timeout=10)
    return _httpx_client

@app.on_event("shutdown")
async def _shutdown():
    global _httpx_client
    try:
        if _httpx_client: await _httpx_client.aclose()
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

# =========================
# Timeline / ingest
# =========================
def now_ts() -> int: return int(time.time())

def append_timeline(n: int):
    exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)", (now_ts(), int(n)))

def get_tail(k:int=WINDOW) -> List[int]:
    rows = query_all("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (k,))
    return [r["number"] for r in rows][::-1]

SEQ_RX = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
def parse_sequence_numbers(text: str) -> List[int]:
    # tenta linha 'Sequência: ...'
    m = SEQ_RX.search(text)
    s = m.group(1) if m else text
    parts = re.findall(r"[1-4]", s)
    if not parts: return []
    nums = [int(x) for x in parts]
    # assumir que no seu canal o último que aparece é o mais recente
    return nums[::-1]

# =========================
# Lógica: ausência consecutiva
# =========================
def absence_streaks(tail: List[int]) -> Dict[int, int]:
    # quantas rodadas desde a última ocorrência de cada número
    out = {1:0,2:0,3:0,4:0}
    for n in [1,2,3,4]:
        c = 0
        for x in reversed(tail):
            if x == n:
                break
            c += 1
        out[n] = c
    return out

def _already_alerted(kind:str, number:int, key:str) -> bool:
    row = query_one("SELECT 1 FROM alerts WHERE kind=? AND number=? AND key=?", (kind, number, key))
    return bool(row)

def _mark_alert(kind:str, number:int, key:str):
    exec_write("INSERT OR REPLACE INTO alerts (kind,number,key,created_at) VALUES (?,?,?,?)",
               (kind, int(number), key, now_ts()))

# =========================
# Sinal imediato (>=10 sem aparecer)
# =========================
async def check_and_fire_if_needed():
    tail = get_tail()
    if not tail: return
    streaks = absence_streaks(tail)
    # chave diária para não repetir spam do mesmo número
    daykey = datetime.now(timezone.utc).strftime("%Y%m%d")
    for n, s in streaks.items():
        if s >= FIRE_ABSENCE_THRESHOLD and not _already_alerted("fire10", n, daykey):
            msg = f"Entrar no número {n}  - estratégia teste"
            await tg_send(SIGNAL_ID, msg, parse="HTML")
            _mark_alert("fire10", n, daykey)

# =========================
# Relatório 5/5 minutos (8-9 sem aparecer)
# =========================
async def periodic_report_loop():
    while True:
        try:
            tail = get_tail()
            if tail:
                streaks = absence_streaks(tail)
                quase = [(n,s) for n,s in streaks.items() if ALMOST_MIN <= s <= ALMOST_MAX]
                if quase:
                    quase.sort(key=lambda t: t[1], reverse=True)
                    # chave por bucket de 5 minutos para não repetir igual
                    bucket = datetime.utcnow().strftime("%Y%m%d%H%M")
                    bucket = bucket[:-1] + "0"  # arredonda minuto para dezena
                    key = "q5_"+bucket+"_"+",".join(f"{n}:{s}" for n,s in quase)
                    if not _already_alerted("almost", 0, key):
                        txt_items = " | ".join([f"{n} ({s} sem vir)" for n,s in quase])
                        msg = f"⏱️ <b>Quase 10 sem vir</b>: {txt_items}"
                        await tg_send(SIGNAL_ID, msg)
                        _mark_alert("almost", 0, key)
        except Exception as e:
            print(f"[report] erro: {e}")
        await asyncio.sleep(REPORT_PERIOD_SECONDS)

# =========================
# FastAPI models & routes
# =========================
class Update(BaseModel):
    update_id: int
    channel_post: Optional[dict] = None
    message: Optional[dict] = None
    edited_channel_post: Optional[dict] = None
    edited_message: Optional[dict] = None

@app.get("/")
async def root():
    return {"ok": True, "detail": "Guardião 10x — POST /webhook/<WEBHOOK_TOKEN> e bot no canal de análise"}

@app.get("/ping")
async def ping():
    await tg_send(SIGNAL_ID, "🔧 Bot ON (10x)")
    return {"ok": True, "signal_to": SIGNAL_ID, "analyze_from": ANALYZE_ID}

@app.on_event("startup")
async def _boot():
    # inicia loop de relatório periódico 5/5
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

    text = (msg.get("text") or msg.get("caption") or "").strip()

    # Filtra: só processa mensagens vindas do canal de análise
    if chat_id != ANALYZE_ID:
        return {"ok": True, "ignored_from": chat_id}

    # Extrai números da sequência e registra na timeline
    nums = parse_sequence_numbers(text)
    added = 0
    for n in nums:
        if n in (1,2,3,4):
            append_timeline(n); added += 1

    # Checa disparo imediato (>=10 sem vir)
    await check_and_fire_if_needed()

    return {"ok": True, "added": added, "chat_id": chat_id}
