#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Webhook único — MODO GEN PURO
# - Gatilho: "💰 ENTRADA CONFIRMADA 💰" do canal-fonte
# - Decide 1 número (G0) usando n-grams (GEN) em {1,2,3,4}
# - Mantém uma pendência (G0→G2). Só abre nova após GREEN/LOSS
# - Aprende com "Sequência:" para alimentar histórico
# - Placar a cada 5 min e reset 00:00 UTC
#
# ENV obrigatórias:
#   TG_BOT_TOKEN, WEBHOOK_TOKEN
# ENV recomendadas:
#   TARGET_CHANNEL (ex.: -1002796105884)
#   SOURCE_CHANNEL_ID (ex.: -1002810508717)  -> filtra apenas o canal-fonte
#   DB_PATH (ex.: /var/data/mini_ref.db)

import os, re, json, time, sqlite3, asyncio
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import FastAPI, Request, HTTPException

# ================== ENV / CONFIG ==================
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()

TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1002796105884").strip()       # destino
SOURCE_CHANNEL_ID = os.getenv("SOURCE_CHANNEL_ID", "").strip()               # fonte (opcional)

DB_PATH        = os.getenv("DB_PATH", "/var/data/mini_ref.db").strip() or "/var/data/mini_ref.db"
TELEGRAM_API   = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# n-grams
WINDOW = 400
DECAY  = 0.985
W4, W3, W2, W1 = 0.40, 0.30, 0.20, 0.10

# pendência (G0 + até G2 = 3 tentativas)
MAX_STAGE = 3  # G0(0), G1(1), G2(2)

# scoreboard
SCOREBOARD_INTERVAL_SEC = 300  # 5 min

app = FastAPI(title="guardiao — GEN puro (webhook)", version="2.1.0")

# ================== DB helpers ==================
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=20.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_db():
    con = _connect(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS ngram (
        n INTEGER NOT NULL, ctx TEXT NOT NULL, nxt INTEGER NOT NULL, w REAL NOT NULL,
        PRIMARY KEY (n, ctx, nxt)
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pending (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        suggested INTEGER NOT NULL,
        stage INTEGER NOT NULL,   -- 0=G0,1=G1,2=G2
        open INTEGER NOT NULL,    -- 1 em aberto
        seen TEXT NOT NULL DEFAULT '' -- números reais vistos na janela
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS daily_score (
        yyyymmdd TEXT PRIMARY KEY,
        g0 INTEGER NOT NULL DEFAULT 0,
        loss INTEGER NOT NULL DEFAULT 0,
        streak INTEGER NOT NULL DEFAULT 0
    )""")
    con.commit(); con.close()

init_db()

def now_ts() -> int:
    return int(time.time())

def today_key_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")

# ================== n-grams ==================
def get_tail(limit:int=WINDOW) -> List[int]:
    con = _connect()
    rows = con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def append_seq(seq: List[int]):
    if not seq: return
    con = _connect(); cur = con.cursor()
    for n in seq:
        cur.execute("INSERT INTO timeline (created_at, number) VALUES (?,?)", (now_ts(), int(n)))
    con.commit(); con.close()
    _update_ngrams()

def _update_ngrams(decay: float=DECAY, max_n:int=5, window:int=WINDOW):
    tail = get_tail(window)
    if len(tail) < 2: return
    con = _connect(); cur = con.cursor()
    for t in range(1, len(tail)):
        nxt = int(tail[t])
        dist = (len(tail)-1) - t
        w = decay ** dist
        for n in range(2, max_n+1):
            if t-(n-1) < 0: break
            ctx = tail[t-(n-1):t]
            ctx_key = ",".join(str(x) for x in ctx)
            cur.execute("""
              INSERT INTO ngram (n, ctx, nxt, w)
              VALUES (?,?,?,?)
              ON CONFLICT(n, ctx, nxt) DO UPDATE SET w = w + excluded.w
            """, (n, ctx_key, nxt, float(w)))
    con.commit(); con.close()

def _prob_from_ngrams(ctx: List[int], cand: int) -> float:
    n = len(ctx) + 1
    if n < 2 or n > 5: return 0.0
    ctx_key = ",".join(str(x) for x in ctx)
    con = _connect()
    row_tot = con.execute("SELECT SUM(w) AS s FROM ngram WHERE n=? AND ctx=?", (n, ctx_key)).fetchone()
    tot = (row_tot["s"] or 0.0) if row_tot else 0.0
    if tot <= 0:
        con.close(); return 0.0
    row_c = con.execute("SELECT w FROM ngram WHERE n=? AND ctx=? AND nxt=?", (n, ctx_key, int(cand))).fetchone()
    w = (row_c["w"] or 0.0) if row_c else 0.0
    con.close()
    return w / tot

def _ngram_backoff(tail: List[int], cand:int, after: Optional[int]) -> float:
    if not tail: return 0.0
    if after is not None and after in tail:
        idxs = [i for i,v in enumerate(tail) if v == after]
        i = idxs[-1]
        ctx1 = tail[max(0,i):i+1]
        ctx2 = tail[max(0,i-1):i+1] if i-1>=0 else []
        ctx3 = tail[max(0,i-2):i+1] if i-2>=0 else []
        ctx4 = tail[max(0,i-3):i+1] if i-3>=0 else []
    else:
        ctx4 = tail[-4:] if len(tail)>=4 else []
        ctx3 = tail[-3:] if len(tail)>=3 else []
        ctx2 = tail[-2:] if len(tail)>=2 else []
        ctx1 = tail[-1:] if len(tail)>=1 else []
    s = 0.0
    if len(ctx4)==4: s += W4 * _prob_from_ngrams(ctx4[:-1], cand)
    if len(ctx3)==3: s += W3 * _prob_from_ngrams(ctx3[:-1], cand)
    if len(ctx2)==2: s += W2 * _prob_from_ngrams(ctx2[:-1], cand)
    if len(ctx1)==1: s += W1 * _prob_from_ngrams(ctx1[:-1], cand)
    return s

def choose_gen_number(after: Optional[int]) -> Tuple[int, float, int]:
    """Escolhe 1 número em {1,2,3,4} pelo GEN (n-grams) — nunca abstem."""
    cands = [1,2,3,4]
    tail = get_tail(WINDOW)
    scores = {c: _ngram_backoff(tail, c, after) for c in cands}
    if all(v == 0.0 for v in scores.values()):
        # desempate: saiu menos nos últimos 50
        last = tail[-50:] if len(tail)>=50 else tail
        freq = {c: last.count(c) for c in cands}
        best = sorted(cands, key=lambda x: (freq.get(x,0), x))[0]
        return best, 0.50, len(tail)
    total = sum(scores.values()) or 1e-9
    post = {k: v/total for k,v in scores.items()}
    best = max(post.items(), key=lambda kv: kv[1])[0]
    return best, post[best], len(tail)

# ================== Parsers ==================
ENTRY_RX = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
SEQ_RX   = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
AFTER_RX = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)

# Resultados comuns nos canais
GREEN_PATTERNS = [
    re.compile(r"\bGREEN\b.*?\(([1-4])\)", re.I|re.S),
    re.compile(r"APOSTA\s+ENCERRADA.*?\bGREEN\b.*?\(([1-4])\)", re.I|re.S),
]
RED_PATTERNS = [
    re.compile(r"\bRED\b.*?\(([1-4])\)", re.I|re.S),
    re.compile(r"\bLOSS\b.*?Número[:\s]*([1-4])", re.I|re.S),
    re.compile(r"APOSTA\s+ENCERRADA.*?\bRED\b.*?\(([1-4])\)", re.I|re.S),
]

GALE1_RX = re.compile(r"Estamos\s+no\s*1[oº]\s*gale", re.I)
GALE2_RX = re.compile(r"Estamos\s+no\s*2[oº]\s*gale", re.I)

def parse_entry(text: str) -> Optional[Dict]:
    t = re.sub(r"\s+", " ", text).strip()
    if not ENTRY_RX.search(t): return None
    seq, after = [], None
    mseq = SEQ_RX.search(t)
    if mseq:
        parts = re.findall(r"[1-4]", mseq.group(1))
        seq = [int(x) for x in parts]
    mafter = AFTER_RX.search(t)
    after = int(mafter.group(1)) if mafter else None
    return {"seq": seq, "after": after, "raw": t}

def parse_green(text:str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text)
    for rx in GREEN_PATTERNS:
        m = rx.search(t)
        if m:
            nums = re.findall(r"[1-4]", m.group(1))
            if nums: return int(nums[0])
    return None

def parse_red(text:str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text)
    for rx in RED_PATTERNS:
        m = rx.search(t)
        if m:
            nums = re.findall(r"[1-4]", m.group(1))
            if nums: return int(nums[0])
    return None

def extract_seq(text:str) -> List[int]:
    m = SEQ_RX.search(text)
    if not m: return []
    return [int(x) for x in re.findall(r"[1-4]", m.group(1))]

# ================== Pendências & Placar ==================
def has_open_pending() -> Optional[sqlite3.Row]:
    con = _connect()
    r = con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id LIMIT 1").fetchone()
    con.close()
    return r

def open_pending(suggested:int):
    con = _connect(); cur = con.cursor()
    cur.execute("""INSERT INTO pending (created_at, suggested, stage, open, seen)
                   VALUES (?,?,?,?,?)""", (now_ts(), int(suggested), 0, 1, ""))
    con.commit(); con.close()

def register_seen(n:int):
    r = has_open_pending()
    if not r: return
    pid = int(r["id"])
    seen = (r["seen"] or "")
    seen2 = (seen + ("|" if seen else "") + str(int(n)))
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE pending SET seen=? WHERE id=?", (seen2, pid))
    con.commit(); con.close()

def _update_score(win: bool):
    y = today_key_utc()
    con = _connect(); cur = con.cursor()
    row = cur.execute("SELECT g0,loss,streak FROM daily_score WHERE yyyymmdd=?", (y,)).fetchone()
    g0 = (row["g0"] if row else 0); loss = (row["loss"] if row else 0); streak = (row["streak"] if row else 0)
    if win:
        g0 += 1; streak += 1
    else:
        loss += 1; streak = 0
    cur.execute("""INSERT OR REPLACE INTO daily_score (yyyymmdd,g0,loss,streak)
                   VALUES (?,?,?,?)""", (y, g0, loss, streak))
    con.commit(); con.close()
    return g0, loss, streak

async def _send_scoreboard():
    y = today_key_utc()
    con = _connect()
    row = con.execute("SELECT g0,loss,streak FROM daily_score WHERE yyyymmdd=?", (y,)).fetchone()
    con.close()
    g0 = (row["g0"] if row else 0); loss = (row["loss"] if row else 0); streak = (row["streak"] if row else 0)
    total = g0 + loss
    acc = (g0/total*100.0) if total else 0.0
    txt = (f"📊 <b>Placar do dia</b>\n"
           f"🟢 G0:{g0}  🔴 Loss:{loss}\n"
           f"✅ Acerto: {acc:.2f}%\n"
           f"🔥 Streak: {streak} GREEN(s)")
    await tg_send_text(TARGET_CHANNEL, txt, "HTML")

def _reset_daily():
    y = today_key_utc()
    con = _connect(); cur = con.cursor()
    cur.execute("""INSERT OR REPLACE INTO daily_score (yyyymmdd,g0,loss,streak)
                   VALUES (?,?,?,?)""", (y,0,0,0))
    con.commit(); con.close()

async def _loop_scoreboard():
    while True:
        try:
            await _send_scoreboard()
        except Exception:
            pass
        await asyncio.sleep(SCOREBOARD_INTERVAL_SEC)

async def _loop_midnight_reset():
    while True:
        now = datetime.now(timezone.utc)
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        wait = max(1.0, (midnight - now).total_seconds())
        await asyncio.sleep(wait)
        try:
            _reset_daily()
            await tg_send_text(TARGET_CHANNEL, "🕛 Reset diário executado (00:00 UTC)", "HTML")
        except Exception:
            pass

# ================== Telegram helpers ==================
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN or not chat_id: return
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                "disable_web_page_preview": True})

# ================== App lifecycle ==================
@app.on_event("startup")
async def _boot():
    try:
        asyncio.create_task(_loop_scoreboard())
        asyncio.create_task(_loop_midnight_reset())
    except Exception:
        pass

# ================== Routes ==================
@app.get("/")
async def root():
    return {"ok": True, "service": "guardiao-auto-bot (GEN puro)"}

def _from_source_chat(msg: dict) -> bool:
    if not SOURCE_CHANNEL_ID:
        return True
    chat = msg.get("chat") or {}
    cid = str(chat.get("id", ""))
    return cid == SOURCE_CHANNEL_ID

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()

    # Extrai o pacote de mensagem
    msg = data.get("channel_post") or data.get("message") \
          or data.get("edited_channel_post") or data.get("edited_message") or {}

    # Filtra por origem (se configurado)
    if not _from_source_chat(msg):
        return {"ok": True, "skipped": "wrong_source"}

    text = (msg.get("text") or msg.get("caption") or "").strip()
    if not text:
        return {"ok": True, "skipped": "sem_texto"}

    # 1) GREEN/RED → fecha/avança pendência
    gnum = parse_green(text)
    rnum = parse_red(text)
    openp = has_open_pending()

    if gnum is not None or rnum is not None:
        n_observed = gnum if gnum is not None else rnum
        if openp:
            # registra número real visto
            register_seen(n_observed)
            sug  = int(openp["suggested"])
            stg  = int(openp["stage"])
            pid  = int(openp["id"])

            if n_observed == sug:
                # GREEN
                g0, loss, streak = _update_score(True)
                await tg_send_text(TARGET_CHANNEL, f"✅ <b>GREEN</b> em <b>G{stg}</b> — Número: <b>{sug}</b>", "HTML")
                # fecha
                con=_connect(); con.execute("UPDATE pending SET open=0 WHERE id=?", (pid,)); con.commit(); con.close()
            else:
                # MISS → avança estágio ou fecha LOSS
                stg2 = stg + 1
                if stg2 >= MAX_STAGE:
                    g0, loss, streak = _update_score(False)
                    await tg_send_text(TARGET_CHANNEL, f"❌ <b>LOSS</b> — Número: <b>{sug}</b>", "HTML")
                    con=_connect(); con.execute("UPDATE pending SET open=0, stage=? WHERE id=?", (stg2, pid)); con.commit(); con.close()
                else:
                    con=_connect(); con.execute("UPDATE pending SET stage=? WHERE id=?", (stg2, pid)); con.commit(); con.close()
        return {"ok": True, "observed": n_observed, "green": gnum is not None}

    # 2) Alimenta histórico com "Sequência:" (sem disparar nada)
    seq_only = extract_seq(text)
    if seq_only:
        append_seq(seq_only)

    # 3) ENTRADA CONFIRMADA → se NÃO há pendência aberta, decide GEN e dispara G0
    parsed = parse_entry(text)
    if parsed:
        if has_open_pending():
            # já existe janela; não bagunçar
            return {"ok": True, "skipped": "pending_open"}
        seq = parsed["seq"] or []
        after = parsed["after"]
        # alimenta histórico antes de decidir
        append_seq(seq)
        # decide GEN
        best, conf, samples = choose_gen_number(after)
        # abre pendência
        open_pending(best)
        aft_txt = f" após {after}" if after else ""
        out = (f"🎯 <b>Número seco (G0):</b> <b>{best}</b>\n"
               f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
               f"📊 Conf: <b>{conf*100:.2f}%</b> | Amostra≈<b>{samples}</b>")
        await tg_send_text(TARGET_CHANNEL, out, "HTML")
        return {"ok": True, "posted": True, "best": best, "conf": conf, "samples": samples}

    return {"ok": True, "noop": True}
