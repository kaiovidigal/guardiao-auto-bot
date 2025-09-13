# -*- coding: utf-8 -*-
"""
Guardião — Relatório Only (sem IA, sem GREEN/LOSS)
- Lê o canal de origem via webhook (mensagens que contenham GREEN/LOSS ou "Sequência: ...")
- Atualiza contagem de ausência por número (1..4) em SQLite
- Envia ALERTA quando atingir 8 ou 9 sem vir
- Envia ENTRADA quando atingir 10 sem vir
- Não publica nenhum GREEN/LOSS, só alertas/entradas por ausência

ENV necessárias:
- TG_BOT_TOKEN       : token do bot (Telegram)
- WEBHOOK_TOKEN      : segredo do webhook (rota: POST /webhook/<WEBHOOK_TOKEN>)
- IA_CHANNEL         : chat_id destino (ex.: -1002796105884). Se vazio, usa PUBLIC_CHANNEL
- PUBLIC_CHANNEL     : opcional (fallback)
- DB_PATH            : opcional (default: /data/data.db)

Parâmetros de comportamento (opcionais):
- ALERT_MIN_STREAK   : padrão 8  (gera alerta em 8 e 9)
- ENTRY_STREAK       : padrão 10 (gera "Entrar no número X — estratégia: Relatório" no 10)
- ALERT_INTERVAL_SECONDS : padrão 300 (loop periódico)
- STARTUP_SUMMARY    : 1/0 (padrão 1) — manda resumo ao iniciar

Execução (exemplo):
uvicorn guardiao_relatorio_only:app --host 0.0.0.0 --port $PORT
"""
import os, re, time, sqlite3, asyncio
from typing import Optional, List, Dict
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# =========================
# ENV / CONFIG
# =========================
DB_PATH = os.getenv("DB_PATH", "/data/data.db").strip() or "/data/data.db"
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "").strip()
IA_CHANNEL = os.getenv("IA_CHANNEL", "").strip()
PUBLIC_CHANNEL = os.getenv("PUBLIC_CHANNEL", "").strip()

ALERT_MIN_STREAK = int(os.getenv("ALERT_MIN_STREAK", "8"))
ENTRY_STREAK = int(os.getenv("ENTRY_STREAK", "10"))
ALERT_INTERVAL_SECONDS = int(os.getenv("ALERT_INTERVAL_SECONDS", "300"))
STARTUP_SUMMARY = os.getenv("STARTUP_SUMMARY", "1").lower() in ("1","true","yes","on")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

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

def _exec(sql: str, p: tuple = ()):
    con = _conn(); con.execute(sql, p); con.commit(); con.close()

def _one(sql: str, p: tuple = ()):
    con = _conn(); r = con.execute(sql, p).fetchone(); con.close(); return r

def _all(sql: str, p: tuple = ()):
    con = _conn(); r = con.execute(sql, p).fetchall(); con.close(); return r

def now_ts() -> int: return int(time.time())

def init_db():
    con = _conn(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL CHECK(number BETWEEN 1 AND 4)
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS absence_stats (
        number INTEGER PRIMARY KEY CHECK(number BETWEEN 1 AND 4),
        current_streak INTEGER NOT NULL DEFAULT 0,
        max_streak INTEGER NOT NULL DEFAULT 0
    )""")
    # Garante 4 linhas
    for n in (1,2,3,4):
        cur.execute("INSERT OR IGNORE INTO absence_stats (number,current_streak,max_streak) VALUES (?,?,?)", (n,0,0))
    con.commit(); con.close()

init_db()

# =========================
# Telegram helpers
# =========================
_http: Optional[httpx.AsyncClient] = None

async def http_client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=10)
    return _http

def target_chat() -> str:
    return IA_CHANNEL or PUBLIC_CHANNEL

async def send_text(text: str):
    chat = target_chat()
    if not TG_BOT_TOKEN or not chat:
        print("[SEND] missing TG_BOT_TOKEN or chat_id")
        return
    try:
        client = await http_client()
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat, "text": text, "parse_mode":"HTML",
                                "disable_web_page_preview": True})
    except Exception as e:
        print("[SEND] error:", e)

# =========================
# Ausência — atualização e leitura
# =========================
def register_observation(n: int):
    """Grava o número na timeline e atualiza ausência (1..4)."""
    _exec("INSERT INTO timeline (created_at, number) VALUES (?,?)", (now_ts(), int(n)))
    # atualiza streaks
    rows = _all("SELECT number,current_streak,max_streak FROM absence_stats ORDER BY number")
    cur_map = {int(r["number"]): int(r["current_streak"]) for r in rows}
    max_map = {int(r["number"]): int(r["max_streak"]) for r in rows}
    for k in (1,2,3,4):
        cur = cur_map.get(k, 0)
        mx  = max_map.get(k, 0)
        if k == n:
            cur = 0
        else:
            cur += 1
            if cur > mx: mx = cur
        _exec("UPDATE absence_stats SET current_streak=?, max_streak=? WHERE number=?", (cur, mx, k))

def get_summary() -> Dict[str,int]:
    rows = _all("SELECT number,current_streak,max_streak FROM absence_stats ORDER BY number")
    if not rows:
        return {"top_now_num": 0, "top_now_streak": 0, "record_num": 0, "record_streak": 0}
    # top atual
    top = max(rows, key=lambda r: int(r["current_streak"]))
    rec = max(rows, key=lambda r: int(r["max_streak"]))
    return {
        "top_now_num": int(top["number"]),
        "top_now_streak": int(top["current_streak"]),
        "record_num": int(rec["number"]),
        "record_streak": int(rec["max_streak"]),
    }

_last_sent: Dict[int,int] = {1:-1,2:-1,3:-1,4:-1}  # guarda última contagem enviada para cada número

async def check_and_alert():
    """Verifica ausências e envia alerta/entrada conforme limiares. Evita repetição."""
    rows = _all("SELECT number,current_streak,max_streak FROM absence_stats ORDER BY number")
    if not rows: return
    # pega recorde
    max_row = max(rows, key=lambda r: int(r["max_streak"]))
    rec_n, rec_s = int(max_row["number"]), int(max_row["max_streak"])
    for r in rows:
        n, cur = int(r["number"]), int(r["current_streak"])
        last = _last_sent.get(n, -1)
        # só age quando a contagem cresce
        if cur <= 0 or cur == last:
            continue
        # Entrada (10+)
        if cur >= ENTRY_STREAK:
            _last_sent[n] = cur
            txt = (f"🏁 <b>Entrar no número {n}</b> — estratégia: <b>Relatório</b>\n"
                   f"📊 Recorde histórico: {rec_n} ficou {rec_s} sem aparecer")
            await send_text(txt)
            continue
        # Alerta (8 ou 9)
        if cur >= ALERT_MIN_STREAK and cur < ENTRY_STREAK:
            _last_sent[n] = cur
            txt = (f"⏱️ <b>Quase 10 sem vir</b>: <b>{n}</b> (<b>{cur}</b> sem vir)\n"
                   f"📊 Recorde histórico: {rec_n} ficou {rec_s} sem aparecer")
            await send_text(txt)

# =========================
# Parsing
# =========================
GREEN_PATTERNS = [
    re.compile(r"\bGREEN\b.*?Número[:\s]*([1-4])", re.I | re.S),
    re.compile(r"\bGREEN\b.*?\(([1-4])\)", re.I | re.S),
]
RED_PATTERNS = [
    re.compile(r"\bLOSS\b.*?Número[:\s]*([1-4])", re.I | re.S),
    re.compile(r"\bRED\b.*?\(([1-4])\)", re.I | re.S),
]
SEQ_RX = re.compile(r"Sequ[eê]ncia:\s*([\d\s\|\-]+)", re.I)

def extract_numbers_from_text(text: str) -> List[int]:
    t = re.sub(r"\s+", " ", text or "").strip()
    # GREEN
    for rx in GREEN_PATTERNS:
        m = rx.search(t)
        if m:
            nums = re.findall(r"[1-4]", m.group(1))
            if nums: return [int(nums[0])]
    # RED (usamos para recuperar último jogado)
    for rx in RED_PATTERNS:
        m = rx.search(t)
        if m:
            nums = re.findall(r"[1-4]", m.group(1))
            if nums: return [int(nums[0])]
    # Sequência: extrai lista esquerda->direita e converte para ordem antiga->recente
    m = SEQ_RX.search(t)
    if m:
        parts = re.findall(r"[1-4]", m.group(1))
        seq = [int(x) for x in parts][::-1]  # joga do mais antigo ao mais novo
        return seq
    return []

# =========================
# API
# =========================
class Update(BaseModel):
    update_id: int
    channel_post: Optional[dict] = None
    message: Optional[dict] = None
    edited_channel_post: Optional[dict] = None
    edited_message: Optional[dict] = None

app = FastAPI(title="Guardião — Relatório Only", version="1.0.0")

@app.on_event("startup")
async def _startup():
    if STARTUP_SUMMARY:
        s = get_summary()
        await send_text(f"🚀 Iniciei. Maior ausência atual: {s['top_now_num']} ({s['top_now_streak']}). "
                        f"Recorde: {s['record_num']} ({s['record_streak']}).")
    async def _loop():
        while True:
            try:
                await check_and_alert()
            except Exception as e:
                print("[ALERT LOOP] error:", e)
            await asyncio.sleep(ALERT_INTERVAL_SECONDS)
    asyncio.create_task(_loop())

@app.on_event("shutdown")
async def _shutdown():
    global _http
    try:
        if _http: await _http.aclose()
    except Exception:
        pass

@app.get("/")
async def index():
    return {"ok": True, "detail": "Use POST /webhook/<WEBHOOK_TOKEN> | /ping?token=..."}

@app.get("/ping")
async def ping(token: str):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    s = get_summary()
    await send_text(f"🔧 Bot online. Maior ausência: {s['top_now_num']} ({s['top_now_streak']}). "
                    f"Recorde: {s['record_num']} ({s['record_streak']}).")
    return {"ok": True}

@app.get("/force_alert")
async def force_alert(token: str):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    await check_and_alert()
    return {"ok": True}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    data = await request.json()
    upd = Update(**data)
    msg = upd.channel_post or upd.message or upd.edited_channel_post or upd.edited_message
    if not msg: return {"ok": True}
    text = (msg.get("text") or msg.get("caption") or "").strip()
    if not text: return {"ok": True}

    nums = extract_numbers_from_text(text)
    if not nums:
        return {"ok": True, "parsed": False}

    for n in nums:
        register_observation(int(n))
    # checa imediatamente (além do loop periódico)
    await check_and_alert()
    return {"ok": True, "ingested": len(nums)}