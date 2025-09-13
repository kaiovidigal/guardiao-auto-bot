# -*- coding: utf-8 -*-
"""
Guardião — Bot com rastreio de AUSÊNCIA + alertas e separação de canais
- Armazena resultados reais na SQLite (timeline)
- Mantém estatística persistida de ausência por número (current/max)
- Envia alertas quando algum número atinge >=8 sem vir, com recorde histórico
- Envia sinais/alertas apenas para IA_CHANNEL (ou PUBLIC_CHANNEL se IA_CHANNEL vazio)
- Processa mensagens do canal de origem via webhook (GREEN/LOSS/ANALISANDO)

ENV obrigatórias:
- TG_BOT_TOKEN       : token do bot do Telegram
- WEBHOOK_TOKEN      : segredo do endpoint (POST /webhook/<token>)
- IA_CHANNEL         : chat_id destino dos alertas/sinais (ex: -1002796105884)
- PUBLIC_CHANNEL     : opcional (fallback se IA_CHANNEL vazio)
- DB_PATH            : opcional (default: /data/data.db)

Suba com Uvicorn, ex: uvicorn guardiao_auto_absence:app --host 0.0.0.0 --port 10000
"""
import os, re, json, time, sqlite3, asyncio
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# =========================
# ENV / CONFIG
# =========================
DB_PATH         = os.getenv("DB_PATH", "/data/data.db").strip() or "/data/data.db"
TG_BOT_TOKEN    = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN   = os.getenv("WEBHOOK_TOKEN", "").strip()
IA_CHANNEL      = os.getenv("IA_CHANNEL", "").strip()
PUBLIC_CHANNEL  = os.getenv("PUBLIC_CHANNEL", "").strip()

TELEGRAM_API    = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
ALERT_INTERVAL_SECONDS = int(os.getenv("ALERT_INTERVAL_SECONDS", "300"))  # 5 min
ALERT_MIN_STREAK       = int(os.getenv("ALERT_MIN_STREAK", "8"))          # alerta a partir de 8
REPEAT_ALERT_EVERY     = int(os.getenv("REPEAT_ALERT_EVERY", "1"))        # repete se aumentar streak

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

def exec_write(sql: str, params: tuple = ()):
    con = _connect(); con.execute(sql, params); con.commit(); con.close()

def query_all(sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    con = _connect(); rows = con.execute(sql, params).fetchall(); con.close(); return rows

def query_one(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    con = _connect(); row = con.execute(sql, params).fetchone(); con.close(); return row

def now_ts() -> int:
    return int(time.time())

# =========================
# Schema
# =========================
def init_db():
    con = _connect(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS absence_stats (
        number INTEGER PRIMARY KEY,
        current_streak INTEGER NOT NULL DEFAULT 0,
        max_streak INTEGER NOT NULL DEFAULT 0,
        updated_at INTEGER NOT NULL
    )""")
    # Garante 4 linhas na absence_stats
    for n in (1,2,3,4):
        cur.execute("""INSERT OR IGNORE INTO absence_stats
                       (number,current_streak,max_streak,updated_at)
                       VALUES (?,?,?,?)""", (n,0,0,now_ts()))
    con.commit(); con.close()

def append_timeline_and_update_absence(n: int):
    n = int(n)
    exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)", (now_ts(), n))
    # Atualiza ausências persistidas
    for k in (1,2,3,4):
        row = query_one("SELECT current_streak, max_streak FROM absence_stats WHERE number=?", (k,))
        cur = int(row["current_streak"] or 0) if row else 0
        mx  = int(row["max_streak"] or 0) if row else 0
        if k == n:
            cur = 0
        else:
            cur += 1
            if cur > mx: mx = cur
        exec_write("""UPDATE absence_stats
                      SET current_streak=?, max_streak=?, updated_at=?
                      WHERE number=?""", (cur, mx, now_ts(), k))

def get_absence_summary() -> Dict[str, int]:
    rows = query_all("SELECT number,current_streak,max_streak FROM absence_stats ORDER BY number ASC")
    if not rows: return {"top_now_num":0, "top_now_streak":0, "record_num":0, "record_streak":0}
    # top atual
    top_now = max(rows, key=lambda r: int(r["current_streak"]))
    # recorde histórico
    top_hist = max(rows, key=lambda r: int(r["max_streak"]))
    return {
        "top_now_num": int(top_now["number"]),
        "top_now_streak": int(top_now["current_streak"]),
        "record_num": int(top_hist["number"]),
        "record_streak": int(top_hist["max_streak"]),
    }

# =========================
# Telegram helpers
# =========================
_httpx_client: Optional[httpx.AsyncClient] = None

async def get_httpx() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(timeout=10)
    return _httpx_client

async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN or not chat_id:
        return
    try:
        client = await get_httpx()
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                "disable_web_page_preview": True})
    except Exception as e:
        print(f"[TG] send error: {e}")

def target_channel() -> str:
    return IA_CHANNEL or PUBLIC_CHANNEL

# =========================
# Parsers do canal de origem
# =========================
GREEN_PATTERNS = [
    re.compile(r"APOSTA\s+ENCERRADA.*?\bGREEN\b.*?\(([1-4])\)", re.I | re.S),
    re.compile(r"\bGREEN\b.*?Número[:\s]*([1-4])", re.I | re.S),
]
RED_PATTERNS = [
    re.compile(r"APOSTA\s+ENCERRADA.*?\bRED\b.*?\(([1-4])\)", re.I | re.S),
    re.compile(r"\bLOSS\b.*?Número[:\s]*([1-4])", re.I | re.S),
]

def extract_green_number(text: str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text or "")
    for rx in GREEN_PATTERNS:
        m = rx.search(t)
        if m: return int(m.group(1))
    return None

def extract_red_last_left(text: str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text or "")
    for rx in RED_PATTERNS:
        m = rx.search(t)
        if m: return int(m.group(1))
    return None

def is_analise(text:str) -> bool:
    return bool(re.search(r"\bANALISANDO\b", text or "", flags=re.I))

def extract_sequence(text: str) -> List[int]:
    """Extrai números da esquerda para direita (mais antigo -> mais recente) de uma sequência postada."""
    m = re.search(r"Sequ[eê]ncia:\s*([^\n\r]+)", text or "", flags=re.I)
    if not m: return []
    parts = re.findall(r"[1-4]", m.group(1))
    return [int(x) for x in parts]

# =========================
# FastAPI
# =========================
class Update(BaseModel):
    update_id: int
    channel_post: Optional[dict] = None
    message: Optional[dict] = None
    edited_channel_post: Optional[dict] = None
    edited_message: Optional[dict] = None

app = FastAPI(title="Guardião — Auto Ausências", version="1.0.0")

@app.on_event("startup")
async def _boot():
    init_db()
    # inicia task de alertas
    async def _alert_loop():
        last_announced: Dict[int,int] = {}  # {numero: ultimo_streak_enviado}
        while True:
            try:
                s = get_absence_summary()
                n   = s["top_now_num"]
                cur = s["top_now_streak"]
                recn= s["record_num"]
                recr= s["record_streak"]
                if n in (1,2,3,4) and cur >= ALERT_MIN_STREAK:
                    # evita repetir a mesma contagem; só repete se aumentou pelo passo definido
                    last = last_announced.get(n, -1)
                    if (last == -1) or (cur >= last + REPEAT_ALERT_EVERY):
                        txt = (f"⏱️ <b>Quase 10 sem vir</b>: <b>{n}</b> (<b>{cur}</b> sem vir)\n"
                               f"📊 <i>Recorde histórico</i>: {recn} ficou {recr} sem aparecer")
                        await tg_send_text(target_channel(), txt)
                        last_announced[n] = cur
            except Exception as e:
                print(f"[ALERT] loop error: {e}")
            await asyncio.sleep(ALERT_INTERVAL_SECONDS)
    try:
        asyncio.create_task(_alert_loop())
    except Exception as e:
        print(f"[STARTUP] alert loop error: {e}")

@app.on_event("shutdown")
async def _shutdown():
    global _httpx_client
    try:
        if _httpx_client:
            await _httpx_client.aclose()
    except Exception:
        pass

@app.get("/")
async def root():
    return {"ok": True, "detail": "Use POST /webhook/<WEBHOOK_TOKEN> e /ping"}

@app.get("/ping")
async def ping():
    try:
        s = get_absence_summary()
        await tg_send_text(target_channel(),
                           f"🔧 Ping OK | top agora: {s['top_now_num']} ({s['top_now_streak']}) | "
                           f"recorde: {s['record_num']} ({s['record_streak']})")
        return {"ok": True, "sent": True, "to": target_channel()}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    data = await request.json()
    upd = Update(**data)
    msg = upd.channel_post or upd.message or upd.edited_channel_post or upd.edited_message
    if not msg: return {"ok": True}

    text = (msg.get("text") or msg.get("caption") or "").strip()

    # 1) GREEN/RED -> atualiza timeline/ausência
    gnum = extract_green_number(text)
    rnum = extract_red_last_left(text)
    if gnum is not None or rnum is not None:
        n = gnum if gnum is not None else rnum
        append_timeline_and_update_absence(int(n))
        return {"ok": True, "observed": int(n)}

    # 2) ANALISANDO -> se vier sequência, também alimenta
    if is_analise(text):
        seq = extract_sequence(text)
        if seq:
            # sequência vem como "esquerda para direita"; queremos alimentar na ordem real (antigo->recente)
            for n in seq[::-1]:
                append_timeline_and_update_absence(int(n))
        return {"ok": True, "analise_feed": bool(seq)}

    return {"ok": True, "skipped": True}
