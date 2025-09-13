# -*- coding: utf-8 -*-
"""
Guardião Dry-Alert — monitor de “seca” por número (1–4)

Funções:
- Lê mensagens do canal-fonte (SOURCE_CHANNEL_ID) via webhook
- Extrai sequências com números 1..4 e alimenta timeline
- Calcula “dry count” (quantas jogadas desde a última ocorrência)
- Dispara alertas em 8, 9 e 10+ sem vir
- Envia relatório a cada 5 minutos (8–9 sem vir)
- Mantém e exibe o Máximo Histórico de seca (valor e número)
"""

import os, re, time, sqlite3, asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# =========================
# ENV
# =========================
TG_BOT_TOKEN      = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN     = os.getenv("WEBHOOK_TOKEN", "").strip()
SOURCE_CHANNEL_ID = os.getenv("SOURCE_CHANNEL_ID", "-1002810508717").strip()
SEND_CHANNEL_ID   = os.getenv("SEND_CHANNEL_ID",   "-1002796105884").strip()
DB_PATH           = os.getenv("DB_PATH", "/var/data/dryalert.db").strip()

if not TG_BOT_TOKEN:
    raise RuntimeError("Env TG_BOT_TOKEN ausente.")
if not WEBHOOK_TOKEN:
    raise RuntimeError("Env WEBHOOK_TOKEN ausente.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# thresholds
ALERT_8 = 8
ALERT_9 = 9
ALERT_10 = 10

SUMMARY_EVERY_SEC = 5 * 60   # 5 minutos

# =========================
# DB helpers
# =========================
def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_db():
    con = _conn(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        n  INTEGER NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS meta(
        k TEXT PRIMARY KEY,
        v TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS alerts_state(
        number INTEGER PRIMARY KEY,
        last_level INTEGER NOT NULL DEFAULT 0,   -- 0=none, 8,9,10
        last_ts INTEGER NOT NULL DEFAULT 0
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS dedupe(
        key TEXT PRIMARY KEY,
        sig TEXT,
        ts INTEGER NOT NULL
    )""")
    # bootstrap for 1..4
    for i in (1,2,3,4):
        cur.execute("INSERT OR IGNORE INTO alerts_state(number,last_level,last_ts) VALUES(?,?,?)",(i,0,0))
    con.commit(); con.close()
init_db()

def db_set(key:str, val:str):
    con = _conn()
    con.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)",(key,val))
    con.commit(); con.close()

def db_get(key:str, default:str="") -> str:
    con = _conn()
    row = con.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
    con.close()
    return row["v"] if row and row["v"] is not None else default

def append_timeline(nums: List[int]):
    if not nums: return
    ts = int(time.time())
    con = _conn()
    con.executemany("INSERT INTO timeline(ts,n) VALUES(?,?)", [(ts, int(x)) for x in nums])
    con.commit(); con.close()

def get_tail(limit:int=2000) -> List[int]:
    con = _conn()
    rows = con.execute("SELECT n FROM timeline ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [r["n"] for r in rows][::-1]

def last_seen_index(tail: List[int], number:int) -> Optional[int]:
    for i in range(len(tail)-1, -1, -1):
        if tail[i] == number:
            return i
    return None

def get_dry_counts() -> Dict[int,int]:
    tail = get_tail()
    if not tail:
        return {1:0,2:0,3:0,4:0}
    dry: Dict[int,int] = {}
    last_idx_total = len(tail)-1
    for num in (1,2,3,4):
        li = last_seen_index(tail, num)
        if li is None:
            dry[num] = len(tail)  # nunca visto → tudo seco
        else:
            dry[num] = last_idx_total - li
    return dry

def update_max_dry_if_needed(dry:Dict[int,int]) -> Tuple[int,int]:
    """Retorna (max_val, max_num) após possível atualização."""
    cur_max_val = int(db_get("max_dry_val","0") or "0")
    cur_max_num = int(db_get("max_dry_num","0") or "0")
    # melhor atual
    best_num = max(dry, key=lambda k: dry[k])
    best_val = dry[best_num]
    if best_val > cur_max_val:
        db_set("max_dry_val", str(best_val))
        db_set("max_dry_num", str(best_num))
        return best_val, best_num
    return cur_max_val, cur_max_num

def get_max_dry() -> Tuple[int,int]:
    return int(db_get("max_dry_val","0") or "0"), int(db_get("max_dry_num","0") or "0")

def get_alert_state(number:int) -> Tuple[int,int]:
    con = _conn()
    row = con.execute("SELECT last_level,last_ts FROM alerts_state WHERE number=?", (number,)).fetchone()
    con.close()
    if not row: return 0,0
    return int(row["last_level"]), int(row["last_ts"])

def set_alert_state(number:int, level:int):
    con = _conn()
    con.execute("UPDATE alerts_state SET last_level=?, last_ts=? WHERE number=?", (level, int(time.time()), number))
    con.commit(); con.close()

def reset_alert_state(number:int):
    set_alert_state(number, 0)

def signature_changed(key:str, sig:str) -> bool:
    con = _conn()
    row = con.execute("SELECT sig FROM dedupe WHERE key=?", (key,)).fetchone()
    if row and row["sig"] == sig:
        con.close(); return False
    con.execute("INSERT OR REPLACE INTO dedupe(key,sig,ts) VALUES(?,?,?)",(key,sig,int(time.time())))
    con.commit(); con.close()
    return True

# =========================
# Telegram
# =========================
_client: Optional[httpx.AsyncClient] = None

async def http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=10)
    return _client

async def tg_send(chat_id:str, text:str, parse_mode:str="HTML"):
    if not chat_id: return
    try:
        client = await http()
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode,
                                "disable_web_page_preview": True})
    except Exception as e:
        print(f"[TG] send error: {e}")

# =========================
# Parsing mensagens do canal
# =========================
SEQ_RX = re.compile(r"Sequ[eê]ncia\s*:\s*([^\n\r]+)", re.I)
DIG_RX = re.compile(r"[1-4]")

def extract_sequence_numbers(text:str) -> List[int]:
    """
    Procura por 'Sequência: ...' e extrai os dígitos 1..4.
    Considera que o texto traz ordem do mais antigo -> mais novo OU vice-versa.
    Para não errar, apenas injeta na ordem que aparecer (já funciona).
    """
    m = SEQ_RX.search(text)
    if not m:  # fallback: pega todos os [1-4] do texto
        parts = DIG_RX.findall(text)
    else:
        parts = DIG_RX.findall(m.group(1))
    return [int(x) for x in parts]

# =========================
# Mensagens formatadas
# =========================
def fmt_max_hist() -> str:
    mv, mn = get_max_dry()
    if mv <= 0:
        return "🏆 Máximo histórico: —"
    return f"🏆 Máximo histórico: <b>{mv}</b> sem vir (número <b>{mn}</b>)"

def fmt_alert_8(n:int, dry:int) -> str:
    return f"⏱️ Alerta: número <b>{n}</b> chegou a <b>{dry}</b> sem vir\n{fmt_max_hist()}"

def fmt_alert_9(n:int, dry:int) -> str:
    return f"⏱️ Quase 10 sem vir: <b>{n}</b> (<b>{dry}</b> sem vir)\n{fmt_max_hist()}"

def fmt_alert_10(n:int, dry:int) -> str:
    return f"🚨 <b>Entrar no número {n}</b> — estratégia <b>teste</b> (<b>{dry}</b> sem vir)\n{fmt_max_hist()}"

def fmt_summary(dry:Dict[int,int]) -> str:
    near = [(n,c) for n,c in dry.items() if c in (ALERT_8, ALERT_9)]
    if not near:
        body = "Sem números em 8–9 no momento."
    else:
        near = sorted(near, key=lambda t: (-t[1], t[0]))  # maiores primeiro
        body = " • ".join([f"{n}:{c}" for n,c in near])
    return f"⏱️ Monitor de seca (8–9): {body}\n{fmt_max_hist()}"

# =========================
# FastAPI
# =========================
app = FastAPI(title="Guardião Dry-Alert", version="1.0.0")

class Update(BaseModel):
    update_id: int
    message: Optional[dict] = None
    channel_post: Optional[dict] = None
    edited_message: Optional[dict] = None
    edited_channel_post: Optional[dict] = None

@app.on_event("shutdown")
async def _close_http():
    global _client
    try:
        if _client: await _client.aclose()
    except Exception:
        pass

@app.get("/")
async def root():
    return {"ok": True, "detail": "Use POST /webhook/<WEBHOOK_TOKEN>"}

@app.get("/ping")
async def ping():
    dry = get_dry_counts()
    return {"ok": True, "dry": dry, "max_hist": {"val": get_max_dry()[0], "num": get_max_dry()[1]}}

# Loop de resumo 5/5 min
async def summary_loop():
    while True:
        try:
            dry = get_dry_counts()
            # update max hist sempre que rodar
            update_max_dry_if_needed(dry)
            text = fmt_summary(dry)
            sig = text  # assinatura simples
            if signature_changed("summary", sig):
                await tg_send(SEND_CHANNEL_ID, text)
        except Exception as e:
            print(f"[SUM] error: {e}")
        await asyncio.sleep(SUMMARY_EVERY_SEC)

@app.on_event("startup")
async def _boot():
    asyncio.create_task(summary_loop())

# Core: trata mensagens do canal-fonte
@app.post("/webhook/{token}")
async def webhook(token:str, request:Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    data = await request.json()
    upd = Update(**data)
    msg = upd.channel_post or upd.message or upd.edited_channel_post or upd.edited_message
    if not msg:
        return {"ok": True}

    chat = msg.get("chat") or {}
    chat_id = str(chat.get("id",""))
    text = (msg.get("text") or msg.get("caption") or "").strip()

    # só processa o canal-fonte
    if chat_id != SOURCE_CHANNEL_ID:
        return {"ok": True, "skip": True}

    # 1) extrai números e inclui na timeline
    nums = extract_sequence_numbers(text)
    if nums:
        append_timeline(nums)

        # 2) calcula seca e dispara alertas
        dry = get_dry_counts()
        max_val, max_num = update_max_dry_if_needed(dry)

        # para cada número, vê se cruzou thresholds
        for n in (1,2,3,4):
            c = dry.get(n,0)
            last_level, _ = get_alert_state(n)

            # reset automático se voltou a sair (c == 0)
            if c == 0 and last_level != 0:
                reset_alert_state(n)
                continue

            # 10+ primeiro
            if c >= ALERT_10 and last_level < ALERT_10:
                await tg_send(SEND_CHANNEL_ID, fmt_alert_10(n, c))
                set_alert_state(n, ALERT_10)
                continue

            # 9
            if c == ALERT_9 and last_level < ALERT_9:
                await tg_send(SEND_CHANNEL_ID, fmt_alert_9(n, c))
                set_alert_state(n, ALERT_9)
                continue

            # 8
            if c == ALERT_8 and last_level < ALERT_8:
                await tg_send(SEND_CHANNEL_ID, fmt_alert_8(n, c))
                set_alert_state(n, ALERT_8)
                continue

        return {"ok": True, "added": len(nums)}

    return {"ok": True, "no_seq": True}