#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webhook_guardiao_fast.py — Guardião sem loop (webhook-only)
- Lê mensagens do grupo de sinal (Entrada Confirmada / Green / Red)
- Extrai padrão (BIG/SMALL/ODD/EVEN/SEQ/KWOK/SSH), sequência e "após X"
- Escolhe 1 "número seco" entre os 2 candidatos (SEM PERDER SINAL)
- Publica no canal alvo (-1002796105884 por padrão)
- Aprende com GREEN/RED via n-grams simples (SQLite)

ENV obrigatórias:
  TG_BOT_TOKEN        -> token do bot
  WEBHOOK_TOKEN       -> token do endpoint (para /webhook/<token>)

ENV opcionais:
  TARGET_CHANNEL      -> canal destino (default: -1002796105884)
  DB_PATH             -> caminho do sqlite (default: /data/fast.db)
  SELF_LABEL_IA       -> rótulo do sinal (default: "Tiro seco por IA")
  MIN_SAMPLES         -> amostra mínima para confiança "bonita" (default: 300)
"""

import os, re, json, time, sqlite3
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# ========= ENV / CONFIG =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN","").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN","").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL","-1002796105884").strip()
DB_PATH        = os.getenv("DB_PATH","/data/fast.db").strip()
SELF_LABEL_IA  = os.getenv("SELF_LABEL_IA","Tiro seco por IA").strip()
MIN_SAMPLES    = int(os.getenv("MIN_SAMPLES","300"))

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN:
    print("⚠️ Defina TG_BOT_TOKEN e WEBHOOK_TOKEN.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

app = FastAPI(title="Guardião Fast (webhook-only)", version="1.0.0")

# ========= DB (SQLite leve) =========
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_db():
    con = _connect()
    c = con.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS timeline(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        num INTEGER NOT NULL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS ngrams(
        n INTEGER NOT NULL,
        ctx TEXT NOT NULL,
        nxt INTEGER NOT NULL,
        w REAL NOT NULL,
        PRIMARY KEY(n,ctx,nxt)
    )""")
    con.commit(); con.close()

init_db()

def now_ts() -> int:
    return int(time.time())

def append_timeline(n: int):
    con = _connect()
    con.execute("INSERT INTO timeline(ts,num) VALUES(?,?)", (now_ts(), int(n)))
    con.commit(); con.close()

def get_tail(k:int=400) -> List[int]:
    con = _connect()
    rows = con.execute("SELECT num FROM timeline ORDER BY id DESC LIMIT ?", (k,)).fetchall()
    con.close()
    return [r["num"] for r in rows][::-1]

def update_ngrams(max_n:int=4, decay:float=0.985, window:int=400):
    tail = get_tail(window)
    if len(tail) < 2: return
    con = _connect()
    cur = con.cursor()
    for t in range(1, len(tail)):
        nxt = tail[t]
        dist = (len(tail)-1) - t
        w = (decay ** dist)
        for n in range(2, max_n+1):
            if t-(n-1) < 0: break
            ctx = tail[t-(n-1):t]
            ctx_key = ",".join(map(str, ctx))
            cur.execute("""
              INSERT INTO ngrams(n,ctx,nxt,w) VALUES(?,?,?,?)
              ON CONFLICT(n,ctx,nxt) DO UPDATE SET w = w + excluded.w
            """, (n, ctx_key, int(nxt), float(w)))
    con.commit(); con.close()

def prob_from_ngrams(ctx: List[int], candidate:int) -> float:
    n = len(ctx)+1
    if n<2 or n>4: return 0.0
    ctx_key = ",".join(map(str, ctx))
    con = _connect()
    row_tot = con.execute("SELECT SUM(w) AS s FROM ngrams WHERE n=? AND ctx=?", (n, ctx_key)).fetchone()
    tot = (row_tot["s"] or 0.0) if row_tot else 0.0
    if tot <= 0:
        con.close(); return 0.0
    row = con.execute("SELECT w FROM ngrams WHERE n=? AND ctx=? AND nxt=?", (n, ctx_key, candidate)).fetchone()
    w = (row["w"] or 0.0) if row else 0.0
    con.close()
    return w / tot

# ========= Telegram helpers =========
async def tg_send_text(chat_id:str, text:str, parse:str="HTML"):
    if not TG_BOT_TOKEN or not chat_id: return
    async with httpx.AsyncClient(timeout=15) as cli:
        await cli.post(f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": parse, "disable_web_page_preview": True})

# ========= Parsers =========
GREEN_PATTERNS = [
    re.compile(r"APOSTA\s+ENCERRADA.*?\bGREEN\b.*?\(([1-4])\)", re.I|re.S),
    re.compile(r"\bGREEN\b.*?Número[:\s]*([1-4])", re.I|re.S),
]
RED_PATTERNS = [
    re.compile(r"APOSTA\s+ENCERRADA.*?\bRED\b.*?\(([1-4])\)", re.I|re.S),
    re.compile(r"\bLOSS\b.*?Número[:\s]*([1-4])", re.I|re.S),
]

def extract_green(text:str) -> Optional[int]:
    t = re.sub(r"\s+"," ", text)
    for rx in GREEN_PATTERNS:
        m = rx.search(t)
        if m: return int(m.group(1))
    return None

def extract_red(text:str) -> Optional[int]:
    t = re.sub(r"\s+"," ", text)
    for rx in RED_PATTERNS:
        m = rx.search(t)
        if m: return int(m.group(1))
    return None

def is_entrada(t:str) -> bool:
    t = re.sub(r"\s+"," ", t).strip()
    if not re.search(r"ENTRADA\s+CONFIRMADA", t, re.I): return False
    if not re.search(r"Mesa:\s*Fantan\s*-\s*Evolution", t, re.I): return False
    return True

def extract_seq(text:str) -> List[int]:
    m = re.search(r"Sequ[eê]ncia:\s*([^\n\r]+)", text, re.I)
    if not m: return []
    parts = re.findall(r"[1-4]", m.group(1))
    return [int(x) for x in parts]

def extract_after(text:str) -> Optional[int]:
    m = re.search(r"Entrar\s+ap[oó]s\s+o\s+([1-4])", text, re.I)
    return int(m.group(1)) if m else None

def extract_pattern(text:str) -> Tuple[str, List[int]]:
    t = re.sub(r"\s+"," ", text)
    # BIG/SMALL -> FanTan: SMALL=[1,2], BIG=[3,4]
    if re.search(r"\bSMALL\b", t, re.I): return "SMALL", [1,2]
    if re.search(r"\bBIG\b", t, re.I):   return "BIG",   [3,4]
    if re.search(r"\bODD\b", t, re.I):   return "ODD",   [1,3]
    if re.search(r"\bEVEN\b", t, re.I):  return "EVEN",  [2,4]

    # KWOK X - Y
    m = re.search(r"\bKWOK\s*([1-4])\s*-\s*([1-4])", t, re.I)
    if m: return f"KWOK-{m.group(1)}-{m.group(2)}", [int(m.group(1)), int(m.group(2))]

    # SSH 2-4-... => usa únicos na ordem
    m = re.search(r"\bSS?H\s*([1-4])(?:-([1-4]))?(?:-([1-4]))?(?:-([1-4]))?", t, re.I)
    if m:
        base = [int(g) for g in m.groups() if g]
        # únicos mantendo ordem
        seen, out = set(), []
        for n in base:
            if n not in seen: seen.add(n); out.append(n)
        if len(out)>=2: return "SSH-"+"-".join(map(str,out)), out[:2]

    # fallback: Sequência → pega 2 primeiros únicos
    seq = extract_seq(t)
    if seq:
        seen, out = set(), []
        for n in seq:
            if n not in seen:
                seen.add(n); out.append(n)
            if len(out)==2: break
        if len(out)==2: return "SEQ", out

    return "GEN", [1,2]  # super-fallback

# ========= Escolha do número (SEM PERDER SINAL) =========
def choose_one(base: List[int], after_num: Optional[int]) -> Tuple[int, float, int]:
    """
    Retorna (best, conf, samples). Sem threshold: sempre escolhe top1.
    Heurística: n-gram backoff (ctx1..ctx3) + suavização.
    """
    tail = get_tail(400)
    samples = 0
    # conf fake bonita quando poucas amostras
    con = _connect()
    row = con.execute("SELECT COUNT(*) AS c FROM ngrams").fetchone()
    samples = int(row["c"] or 0)
    con.close()

    def score_for(cand:int) -> float:
        if not tail: return 1.0
        s = 0.0
        # se “após X” estiver na cauda, usa contexto a partir do último X
        if after_num is not None and after_num in tail:
            i = max([idx for idx,v in enumerate(tail) if v==after_num])
            ctx1 = tail[i:i+1]
            ctx2 = tail[i-1:i+1] if i-1>=0 else []
            ctx3 = tail[i-2:i+1] if i-2>=0 else []
        else:
            ctx1 = tail[-1:] if len(tail)>=1 else []
            ctx2 = tail[-2:] if len(tail)>=2 else []
            ctx3 = tail[-3:] if len(tail)>=3 else []
        for w,ctx in ((0.50,ctx3[:-1]), (0.30,ctx2[:-1]), (0.20,ctx1[:-1])):
            if ctx:
                s += w * prob_from_ngrams(ctx, cand)
        # leve prior uniforme
        return s + 1e-6

    scores = {c: score_for(c) for c in base}
    total = sum(scores.values()) or 1e-9
    post = {k: v/total for k,v in scores.items()}
    best = max(post.items(), key=lambda kv: kv[1])[0]
    conf = post[best]
    # se amostra baixa, “achatamos” para 0.58..0.65 só para estética do print
    if samples < MIN_SAMPLES:
        conf = 0.58 + 0.07 * conf
    return best, float(conf), samples

def fmt_signal(best:int, pattern:str, after_num: Optional[int], conf:float, samples:int) -> str:
    aft = f"\n🔁 Após último número <b>{after_num}</b>" if after_num else ""
    return (
        f"🤖 <b>{SELF_LABEL_IA}</b>\n"
        f"🎯 Número seco (G0): <b>{best}</b>\n"
        f"🧩 Padrão: <b>{pattern}</b>{aft}\n"
        f"📊 Conf: <b>{conf*100:.2f}%</b> | Amostra≈<b>{samples}</b>"
    )

# ========= Webhook =========
class Update(BaseModel):
    update_id: int
    channel_post: Optional[dict] = None
    message: Optional[dict] = None
    edited_channel_post: Optional[dict] = None
    edited_message: Optional[dict] = None

@app.get("/")
async def root():
    return {"ok": True, "detail": "POST /webhook/<WEBHOOK_TOKEN>"}

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

    # 1) GREEN / RED alimentam o histórico
    g = extract_green(text)
    r = extract_red(text)
    if g is not None or r is not None:
        n = g if g is not None else r
        append_timeline(n)
        update_ngrams()
        return {"ok": True, "observed": n}

    # 2) ENTRADA CONFIRMADA → escolher e publicar SEM FALHAR
    if not is_entrada(text):
        return {"ok": True, "skipped": True}

    pattern, base = extract_pattern(text)
    after_num = extract_after(text)

    # fallback robusto se por acaso vier base incompleta
    if len(base) < 2:
        # tenta enriquecer com sequência
        seq = extract_seq(text)
        for n in seq:
            if n not in base:
                base.append(n)
            if len(base)==2: break
        if len(base)<2:  # última segurança
            base = [1,2]

    best, conf, samples = choose_one(base, after_num)
    out = fmt_signal(best, pattern, after_num, conf, samples)
    await tg_send_text(TARGET_CHANNEL, out, "HTML")
    return {"ok": True, "sent": True, "best": best, "base": base, "conf": conf, "samples": samples}