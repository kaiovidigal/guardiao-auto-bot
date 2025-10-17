#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GuardiAo Auto Bot — webhook_app.py (v7.3)

Inclui:
- /health, /debug_cfg
- /admin/echo-token/{token}          -> diagnosticar token do webhook
- /admin/fantan-status               -> painel do capturador (site)
- POST /ingest/fantan/{token}        -> ingestão dos números (site -> webhook)
- POST /webhook/{token}              -> Telegram

Render:
  Build: pip install -r requirements.txt
  Start: uvicorn webhook_app:app --host 0.0.0.0 --port $PORT

ENVs (Render -> Environment):
  WEBHOOK_TOKEN    = meusegredo123
  TG_BOT_TOKEN     = 8315...ztK4               (opcional para enviar mensagens)
  TARGET_CHANNEL   = -1003052132833            (se quiser avisos no canal)
  SOURCE_CHANNEL   = (opcional p/ filtrar)
  SHOW_DEBUG       = True/False (default False)
  MAX_GALE         = 1
  OBS_TIMEOUT_SEC  = 420
  DEDUP_WINDOW_SEC = 40
"""

import os, re, json, time, sqlite3, datetime, hashlib
from typing import List, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# ===================== ENV =====================
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip() or "meusegredo123"
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()    # opcional
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "").strip()    # opcional

SHOW_DEBUG       = os.getenv("SHOW_DEBUG", "False").strip().lower() == "true"
MAX_GALE         = int(os.getenv("MAX_GALE", "1"))
OBS_TIMEOUT_SEC  = int(os.getenv("OBS_TIMEOUT_SEC", "420"))
DEDUP_WINDOW_SEC = int(os.getenv("DEDUP_WINDOW_SEC", "40"))

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}" if TG_BOT_TOKEN else ""
DB_PATH = "/opt/render/project/src/main.sqlite"

# ===================== APP =====================
app = FastAPI(title="GuardiAo Auto Bot (webhook)", version="7.3")

# ===================== DB ======================
def _con():
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

def db_init():
    con = _con(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS processed(
        update_id TEXT PRIMARY KEY, seen_at INTEGER NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL, number INTEGER NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pending(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER, opened_at INTEGER,
        suggested INTEGER, stage INTEGER DEFAULT 0, seen TEXT DEFAULT '', open INTEGER DEFAULT 1)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS score(
        id INTEGER PRIMARY KEY CHECK(id=1), green INTEGER DEFAULT 0, loss INTEGER DEFAULT 0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS dedupe(
        kind TEXT NOT NULL, dkey TEXT NOT NULL, ts INTEGER NOT NULL, PRIMARY KEY (kind, dkey))""")
    if not con.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        con.execute("INSERT INTO score(id,green,loss) VALUES(1,0,0)")
    con.commit(); con.close()
db_init()

def _timeline_tail(n:int=400)->List[int]:
    con = _con()
    rows = con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?",(n,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def _append_seq(seq: List[int]):
    if not seq: return
    con = _con()
    now = int(time.time())
    con.executemany("INSERT INTO timeline(created_at,number) VALUES(?,?)",[(now,int(x)) for x in seq])
    con.commit(); con.close()

def _timeline_size()->int:
    con=_con(); row=con.execute("SELECT COUNT(*) c FROM timeline").fetchone(); con.close()
    return int(row["c"] or 0)

def _score_add(outcome:str):
    con = _con()
    row = con.execute("SELECT green,loss FROM score WHERE id=1").fetchone()
    g,l = (int(row["green"]), int(row["loss"])) if row else (0,0)
    if outcome.upper()=="GREEN": g+=1
    elif outcome.upper()=="LOSS": l+=1
    con.execute("INSERT OR REPLACE INTO score(id,green,loss) VALUES(1,?,?)",(g,l))
    con.commit(); con.close()

def _score_text()->str:
    con = _con(); row = con.execute("SELECT green,loss FROM score WHERE id=1").fetchone(); con.close()
    if not row: return "0 GREEN × 0 LOSS — 0.0%"
    g,l = int(row["green"]), int(row["loss"])
    tot = g+l; acc = (g/tot*100.0) if tot>0 else 0.0
    return f"{g} GREEN × {l} LOSS — {acc:.1f}%"

# ============ DEDUPE ============
def _dedupe_key(text: str) -> str:
    base = re.sub(r"\s+", " ", (text or "")).strip().lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _seen_recent(kind: str, dkey: str) -> bool:
    now = int(time.time())
    con = _con()
    row = con.execute("SELECT ts FROM dedupe WHERE kind=? AND dkey=?", (kind, dkey)).fetchone()
    if row and now - int(row["ts"]) <= DEDUP_WINDOW_SEC:
        con.close()
        return True
    con.execute("INSERT OR REPLACE INTO dedupe(kind, dkey, ts) VALUES (?,?,?)", (kind, dkey, now))
    con.commit(); con.close()
    return False

# ============ IA compacta ============
def _norm(d: Dict[int,float])->Dict[int,float]:
    s=sum(d.values()) or 1e-9
    return {k:v/s for k,v in d.items()}

def _post_freq(tail:List[int], k:int)->Dict[int,float]:
    if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
    win = tail[-k:] if len(tail)>=k else tail
    tot=max(1,len(win))
    return _norm({c:win.count(c)/tot for c in (1,2,3,4)})

def _post_e1_ngram(tail:List[int])->Dict[int,float]:
    mix={c:0.0 for c in (1,2,3,4)}
    for k,w in ((8,0.25),(21,0.35),(55,0.40)):
        pk=_post_freq(tail,k)
        for c in (1,2,3,4): mix[c]+=w*pk[c]
    return _norm(mix)

def _post_e2_short(tail):  return _post_freq(tail, 60)
def _post_e3_long(tail):   return _post_freq(tail, 300)
def _post_e4_llm(tail):    return {1:0.25,2:0.25,3:0.25,4:0.25}

def _hedge(p1,p2,p3,p4, w=(0.40,0.25,0.25,0.10)):
    cands=(1,2,3,4)
    out={c: w[0]*p1.get(c,0)+w[1]*p2.get(c,0)+w[2]*p3.get(c,0)+w[3]*p4.get(c,0) for c in cands}
    return _norm(out)

def _conf_floor(post:Dict[int,float], floor=0.30, cap=0.95):
    post=_norm({c:float(post.get(c,0)) for c in (1,2,3,4)})
    b=max(post,key=post.get); mx=post[b]
    if mx<floor:
        others=[c for c in (1,2,3,4) if c!=b]
        s=sum(post[c] for c in others)
        take=min(floor-mx, s)
        if s>0:
            scale=(s-take)/s
            for c in others: post[c]*=scale
        post[b]=min(cap, mx+take)
    if post[b]>cap:
        ex=post[b]-cap; post[b]=cap
        add=ex/3.0
        for c in (1,2,3,4):
            if c!=b: post[c]+=add
    return _norm(post)

def _choose_number()->Tuple[int,float,int,Dict[int,float],float,str]:
    tail=_timeline_tail(400)
    p1=_post_e1_ngram(tail); p2=_post_e2_short(tail); p3=_post_e3_long(tail); p4=_post_e4_llm(tail)
    base=_hedge(p1,p2,p3,p4)
    post=_conf_floor(base, 0.30, 0.95)
    best=max(post,key=post.get); conf=float(post[best])
    r=sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    gap=(r[0][1]-r[1][1]) if len(r)>=2 else r[0][1]
    return best, conf, _timeline_size(), post, gap, "IA"

def _ngram_snapshot(suggested:int)->str:
    tail=_timeline_tail(400)
    post=_post_e1_ngram(tail)
    pct=lambda x:f"{x*100:.1f}%"
    p1,p2,p3,p4 = pct(post[1]), pct(post[2]), pct(post[3]), pct(post[4])
    conf=pct(post.get(int(suggested),0.0))
    return (f"📈 Amostra: {_timeline_size()} • Conf: {conf}\n"
            f"🔎 E1(n-gram proxy): 1 {p1} | 2 {p2} | 3 {p3} | 4 {p4}")

# ============ Telegram helpers ============
async def tg_send(chat_id: str, text: str, parse="HTML"):
    if not TELEGRAM_API:
        print("ℹ️ (tg_send) TELEGRAM_API vazio — defina TG_BOT_TOKEN para enviar mensagens.")
        return
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            await cli.post(f"{TELEGRAM_API}/sendMessage",
                           json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                 "disable_web_page_preview": True})
    except Exception as e:
        print("tg_send error:", e)

# ============ Parser ============
RX_ENTRADA = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
RX_ANALISE = re.compile(r"\bANALISANDO\b", re.I)
RX_FECHA   = re.compile(r"APOSTA\s+ENCERRADA", re.I)

RX_SEQ     = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
RX_NUMS    = re.compile(r"[1-4]")
RX_AFTER   = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)
RX_PAREN   = re.compile(r"\(([^\)]*)\)\s*$")

def _parse_seq_list(text:str)->List[int]:
    m=RX_SEQ.search(text or "")
    if not m: return []
    return [int(x) for x in RX_NUMS.findall(m.group(1))]

# ===================== Rotas básicas + admin =====================
@app.get("/")
async def root():
    return {"ok": True, "service": "GuardiAo Auto Bot", "version": "7.3",
            "time": datetime.datetime.utcnow().isoformat()+"Z"}

@app.get("/health")
async def health():
    return {"ok": True, "db_exists": os.path.exists(DB_PATH), "db_path": DB_PATH}

@app.get("/debug_cfg")
async def debug_cfg():
    return {"MAX_GALE": MAX_GALE, "OBS_TIMEOUT_SEC": OBS_TIMEOUT_SEC,
            "DEDUP_WINDOW_SEC": DEDUP_WINDOW_SEC, "SOURCE_CHANNEL": SOURCE_CHANNEL,
            "TARGET_SET": bool(TARGET_CHANNEL)}

@app.get("/admin/echo-token/{token}")
async def echo_token(token: str):
    return {"match": token == WEBHOOK_TOKEN, "received": token, "expected": WEBHOOK_TOKEN}

# ===================== Ingestão Fan Tan (site -> webhook) =====================
class FanTanIn(BaseModel):
    numbers: Optional[List[int]] = None   # ex: [1, 3]
    text: Optional[str] = None            # ex: "Sequência: 1 | 3"

FAN_TAN_STATUS = {
    "source": "pinbet-fantan",
    "last_pull_ts": None,
    "last_numbers": None,
    "total_appended": 0,
}

def _now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

@app.get("/admin/fantan-status")
async def fantan_status():
    tail = _timeline_tail(20)
    return {
        "ok": True,
        "source": FAN_TAN_STATUS["source"],
        "last_pull_ts": FAN_TAN_STATUS["last_pull_ts"],
        "last_numbers": FAN_TAN_STATUS["last_numbers"],
        "total_appended": FAN_TAN_STATUS["total_appended"],
        "timeline_tail": tail,
        "timeline_size": _timeline_size(),
        "now": _now_iso(),
    }

@app.post("/ingest/fantan/{token}")
async def ingest_fantan(token: str, payload: FanTanIn):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    seq: List[int] = []
    if payload.numbers and isinstance(payload.numbers, list):
        seq = [int(x) for x in payload.numbers if int(x) in (1,2,3,4)]
    elif payload.text:
        seq = _parse_seq_list(payload.text)

    if not seq:
        return {"ok": False, "error": "no_numbers_found", "hint": "Envie numbers=[1,3] ou text='Sequência: 1 | 3'"}

    _append_seq(seq)
    FAN_TAN_STATUS["last_pull_ts"] = _now_iso()
    FAN_TAN_STATUS["last_numbers"] = seq
    FAN_TAN_STATUS["total_appended"] = int(FAN_TAN_STATUS["total_appended"] or 0) + len(seq)

    if TARGET_CHANNEL:
        await tg_send(TARGET_CHANNEL, f"🌐 Pinbet (Fan Tan) ingest: {seq} — amostra={_timeline_size()}")

    return {"ok": True, "ingested": seq, "timeline_size": _timeline_size()}

# ===================== Webhook Telegram =====================
@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        data = await request.json()
    except Exception:
        data = {}

    print("📩 webhook payload:", json.dumps(data)[:600])

    msg = data.get("channel_post") or data.get("message") \
        or data.get("edited_channel_post") or data.get("edited_message") or {}
    chat = msg.get("chat") or {}
    chat_id = str(chat.get("id") or "")
    text = (msg.get("text") or msg.get("caption") or "").strip()

    if SOURCE_CHANNEL and chat_id and chat_id != SOURCE_CHANNEL:
        return {"ok": True, "skipped": "wrong_source"}

    if not text:
        return {"ok": True, "skipped": "no_text"}

    if re.search(r"ANALISANDO", text, re.I):
        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)
        return {"ok": True, "analise_seq": len(seq)}

    if re.search(r"APOSTA\s+ENCERRADA|GREEN|✅|RED|❌", text, re.I):
        return {"ok": True, "noted_close": True}

    if re.search(r"ENTRADA\s+CONFIRMADA", text, re.I):
        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)
        best, conf, samples, post, gap, reason = _choose_number()
        if TARGET_CHANNEL:
            await tg_send(TARGET_CHANNEL,
                f"🤖 <b>IA SUGERE</b> — <b>{best}</b>\n"
                f"📊 <b>Conf:</b> {conf*100:.2f}% | <b>Amostra≈</b>{samples} | <b>gap≈</b>{gap*100:.1f}pp\n"
                f"🧠 <b>Modo:</b> {reason}\n{_ngram_snapshot(best)}")
        return {"ok": True, "entry_opened": True, "best": best, "conf": conf}

    return {"ok": True, "skipped": "unmatched"}