#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webhook único (GEN 100%) — com SQLite robusto (retries) e placar periódico.
- Gatilho: mensagens "ENTRADA CONFIRMADA" do canal-fonte
- Decide SEMPRE 1 número seco via n-grams (GEN), ignora pares/ímpares/big/small
- Mantém janela aberta (G0→G2); não aceita novo gatilho até fechar com GREEN/LOSS
- Captura GREEN/LOSS do próprio canal (APOSTA ENCERRADA/LOSS) para fechar a janela
- Placar a cada 5 minutos, reset às 00:00 UTC

ENV esperadas:
  TG_BOT_TOKEN      (obrigatório)
  WEBHOOK_TOKEN     (obrigatório)
  TARGET_CHANNEL    (canal destino; ex: -1002796105884)
  DB_PATH           (opcional, default: /data/mini_ref.db)
"""
import os, re, json, time, sqlite3, asyncio, random
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import FastAPI, Request, HTTPException

# ================= ENV / Telegram =================
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1002796105884").strip()
DB_PATH        = os.getenv("DB_PATH", "/data/mini_ref.db").strip() or "/data/mini_ref.db"
TELEGRAM_API   = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# ================= FastAPI =================
app = FastAPI(title="Guardião GEN — Webhook único", version="2.2.0")

# ================= Util =================
def now_ts() -> int: return int(time.time())
def today_key_utc() -> str: return datetime.now(timezone.utc).strftime("%Y%m%d")

# ================= SQLite helpers (robustos) =================
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False, isolation_level=None)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA busy_timeout=60000;")  # 60s
    return con

def exec_write(sql: str, params: tuple = (), retries: int = 8, base_wait: float = 0.15):
    """Executa escrita com retries exponenciais se DB estiver bloqueado."""
    last_err = None
    for i in range(retries):
        try:
            con = _connect()
            with con:
                con.execute(sql, params)
            con.close()
            return
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            last_err = e
            if "locked" in msg or "busy" in msg or "database is locked" in msg:
                time.sleep(base_wait * (2**i) + random.random()*0.05)
                continue
            raise
    raise last_err or sqlite3.OperationalError("DB write failed")

def query_all(sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    con = _connect()
    rows = con.execute(sql, params).fetchall()
    con.close()
    return rows

def query_one(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    con = _connect()
    row = con.execute(sql, params).fetchone()
    con.close()
    return row

# ================= DB init =================
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
        opened_at INTEGER NOT NULL,
        suggested INTEGER NOT NULL,
        stage INTEGER NOT NULL,    -- 0=g0,1=g1,2=g2
        open INTEGER NOT NULL      -- 1=open, 0=closed
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS daily_score (
        yyyymmdd TEXT PRIMARY KEY,
        g0 INTEGER NOT NULL DEFAULT 0,
        g1 INTEGER NOT NULL DEFAULT 0,
        g2 INTEGER NOT NULL DEFAULT 0,
        loss INTEGER NOT NULL DEFAULT 0
    )""")
    con.close()

init_db()

# ================= n-grams (GEN) =================
WINDOW = 400
DECAY  = 0.985
W4,W3,W2,W1 = 0.40,0.30,0.20,0.10

def get_tail(limit:int=WINDOW) -> List[int]:
    rows = query_all("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (limit,))
    return [int(r["number"]) for r in rows][::-1]

def append_seq(seq: List[int]):
    if not seq: return
    for n in seq:
        exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)", (now_ts(), int(n)))
    _update_ngrams()

def _update_ngrams(max_n:int=5):
    tail = get_tail(WINDOW)
    if len(tail) < 2: return
    for t in range(1, len(tail)):
        nxt = tail[t]
        dist = (len(tail)-1) - t
        w = DECAY ** dist
        for n in range(2, max_n+1):
            if t-(n-1) < 0: break
            ctx = tail[t-(n-1):t]
            ctx_key = ",".join(str(x) for x in ctx)
            exec_write("""
              INSERT INTO ngram (n, ctx, nxt, w)
              VALUES (?,?,?,?)
              ON CONFLICT(n,ctx,nxt) DO UPDATE SET w = w + excluded.w
            """, (n, ctx_key, int(nxt), float(w)))

def _prob_from_ngrams(ctx: List[int], cand: int) -> float:
    n = len(ctx)+1
    if n<2 or n>5: return 0.0
    ctx_key = ",".join(str(x) for x in ctx)
    row_tot = query_one("SELECT SUM(w) s FROM ngram WHERE n=? AND ctx=?", (n, ctx_key))
    tot = float(row_tot["s"] or 0.0) if row_tot else 0.0
    if tot <= 0: return 0.0
    row_c = query_one("SELECT w FROM ngram WHERE n=? AND ctx=? AND nxt=?", (n, ctx_key, int(cand)))
    w = float(row_c["w"] or 0.0) if row_c else 0.0
    return w/tot if tot>0 else 0.0

def choose_single_number(after: Optional[int]) -> Tuple[int,float,int]:
    tail = get_tail(WINDOW)
    # Recorte por "após X" se existir na cauda
    if after is not None and after in tail:
        i = max(i for i,v in enumerate(tail) if v==after)
        ctx1 = tail[max(0,i):i+1]
        ctx2 = tail[max(0,i-1):i+1] if i-1>=0 else []
        ctx3 = tail[max(0,i-2):i+1] if i-2>=0 else []
        ctx4 = tail[max(0,i-3):i+1] if i-3>=0 else []
    else:
        ctx4 = tail[-4:] if len(tail)>=4 else []
        ctx3 = tail[-3:] if len(tail)>=3 else []
        ctx2 = tail[-2:] if len(tail)>=2 else []
        ctx1 = tail[-1:] if len(tail)>=1 else []
    scores = {}
    for c in (1,2,3,4):
        s=0.0
        if len(ctx4)==4: s += W4*_prob_from_ngrams(ctx4[:-1], c)
        if len(ctx3)==3: s += W3*_prob_from_ngrams(ctx3[:-1], c)
        if len(ctx2)==2: s += W2*_prob_from_ngrams(ctx2[:-1], c)
        if len(ctx1)==1: s += W1*_prob_from_ngrams(ctx1[:-1], c)
        scores[c]=s
    if all(v==0.0 for v in scores.values()):
        last = tail[-50:] if len(tail)>=50 else tail
        freq = {c:last.count(c) for c in (1,2,3,4)}
        best = sorted((1,2,3,4), key=lambda x:(freq.get(x,0), x))[0]
        return best, 0.50, len(tail)
    tot = sum(scores.values()) or 1e-9
    post = {k: v/tot for k,v in scores.items()}
    best = max(post.items(), key=lambda kv: kv[1])[0]
    return best, float(post[best]), len(tail)

# ================= Parsers canal-fonte =================
ENTRY_RX = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
SEQ_RX   = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
AFTER_RX = re.compile(r"Entrar\s+ap[oó]s\s+o\s+([1-4])", re.I)
GALE1_RX = re.compile(r"Estamos\s+no\s*1[ºo]\s*gale", re.I)
GALE2_RX = re.compile(r"Estamos\s+no\s*2[ºo]\s*gale", re.I)

GREEN_PATTERNS = [
    re.compile(r"APOSTA\s+ENCERRADA.*?\bGREEN\b.*?\(([1-4])\)", re.I | re.S),
    re.compile(r"\bGREEN\b.*?Número[:\s]*([1-4])", re.I | re.S),
]
LOSS_PATTERNS = [
    re.compile(r"\bLOSS\b.*?Número[:\s]*([1-4])", re.I | re.S),
    re.compile(r"APOSTA\s+ENCERRADA.*?\bRED\b", re.I | re.S),
]

def parse_entry(text:str) -> Optional[Dict]:
    t = re.sub(r"\s+", " ", text).strip()
    if not ENTRY_RX.search(t):
        return None
    seq=[]; after=None
    mseq = SEQ_RX.search(t)
    if mseq:
        parts = re.findall(r"[1-4]", mseq.group(1))
        seq = [int(x) for x in parts]
    mafter = AFTER_RX.search(t)
    if mafter: after = int(mafter.group(1))
    return {"seq":seq, "after":after, "raw":t}

def extract_green(text:str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text)
    for rx in GREEN_PATTERNS:
        m = rx.search(t)
        if m:
            nums = re.findall(r"[1-4]", m.group(1))
            if nums: return int(nums[0])
    return None

def is_loss_text(text:str) -> bool:
    t = re.sub(r"\s+", " ", text)
    return any(rx.search(t) for rx in LOSS_PATTERNS)

# ================= Telegram send =================
async def tg_send_text(chat_id: str, text: str, parse: str = "HTML"):
    if not TG_BOT_TOKEN: return
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                "disable_web_page_preview": True})

# ================= Placar =================
def _bump_score(stage:int, won:bool):
    y = today_key_utc()
    row = query_one("SELECT g0,g1,g2,loss FROM daily_score WHERE yyyymmdd=?", (y,))
    g0=g1=g2=loss=0
    if row: g0,g1,g2,loss = row["g0"],row["g1"],row["g2"],row["loss"]
    if won:
        if stage==0: g0+=1
        elif stage==1: g1+=1
        else: g2+=1
    else:
        loss+=1
    exec_write("""
      INSERT INTO daily_score (yyyymmdd,g0,g1,g2,loss)
      VALUES (?,?,?,?,?)
      ON CONFLICT(yyyymmdd) DO UPDATE SET
        g0=excluded.g0, g1=excluded.g1, g2=excluded.g2, loss=excluded.loss
    """, (y,g0,g1,g2,loss))

async def _scoreboard_task():
    while True:
        try:
            y = today_key_utc()
            row = query_one("SELECT g0,g1,g2,loss FROM daily_score WHERE yyyymmdd=?", (y,))
            g0=g1=g2=loss=0
            if row: g0,g1,g2,loss = row["g0"],row["g1"],row["g2"],row["loss"]
            total = g0+g1+g2+loss
            acc = ((g0+g1+g2)/total*100.0) if total else 0.0
            txt = (f"📊 <b>Placar do dia</b>\n"
                   f"G0:{g0}  G1:{g1}  G2:{g2}  Loss:{loss}\n"
                   f"✅ Acerto: {acc:.2f}% (total {total})")
            await tg_send_text(TARGET_CHANNEL, txt, "HTML")
        except Exception as e:
            # silencioso no loop
            pass
        # espera até completar 5 min exatos (reduz drift)
        await asyncio.sleep(300)

async def _midnight_reset_task():
    while True:
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        wait = (tomorrow - now).total_seconds()
        await asyncio.sleep(max(1, wait))
        # reset placar
        y = today_key_utc()
        exec_write("""INSERT OR REPLACE INTO daily_score (yyyymmdd,g0,g1,g2,loss) VALUES (?,0,0,0,0)""",(y,))
        await tg_send_text(TARGET_CHANNEL, "🕛 Reset diário executado (00:00 UTC).", "HTML")

@app.on_event("startup")
async def _boot():
    asyncio.create_task(_scoreboard_task())
    asyncio.create_task(_midnight_reset_task())

# ================ Core: janela (G0→G2) =================
def _has_open_pending() -> Optional[sqlite3.Row]:
    return query_one("SELECT id,suggested,stage FROM pending WHERE open=1 ORDER BY id LIMIT 1")

def _open_pending(suggested:int):
    exec_write("INSERT INTO pending (opened_at,suggested,stage,open) VALUES (?,?,0,1)",
               (now_ts(), int(suggested)))

def _advance_or_close(hit: bool):
    row = _has_open_pending()
    if not row: return
    pid, sug, stage = int(row["id"]), int(row["suggested"]), int(row["stage"])
    if hit:
        _bump_score(stage, True)
        exec_write("UPDATE pending SET open=0 WHERE id=?", (pid,))
    else:
        if stage >= 2:
            _bump_score(stage, False)
            exec_write("UPDATE pending SET open=0 WHERE id=?", (pid,))
        else:
            exec_write("UPDATE pending SET stage=stage+1 WHERE id=?", (pid,))

# ================ Rotas =================
@app.get("/")
async def root():
    return {"ok": True, "service": "guardiao-gen-webhook"}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()
    msg = data.get("channel_post") or data.get("message") \
          or data.get("edited_channel_post") or data.get("edited_message") or {}
    text = (msg.get("text") or msg.get("caption") or "").strip()
    if not text:
        return {"ok": True, "skipped": "no_text"}

    # 1) GREEN/LOSS fecha/avança janela aberta
    green_num = extract_green(text)
    if green_num is not None or is_loss_text(text):
        row = _has_open_pending()
        if row:
            sug = int(row["suggested"])
            hit = (green_num == sug) if green_num is not None else False
            _advance_or_close(hit)
            if green_num is not None:
                await tg_send_text(TARGET_CHANNEL,
                    f"✅ <b>Resultado:</b> {'GREEN' if hit else 'LOSS'} — Número real <b>{green_num}</b> (sug: <b>{sug}</b>)",
                    "HTML")
        return {"ok": True, "result_event": True}

    # 2) “Estamos no 1º/2º gale” → avança sem fechar (se aberto e se não foi hit)
    if GALE1_RX.search(text) or GALE2_RX.search(text):
        _advance_or_close(False)  # avança stage
        return {"ok": True, "gale_progress": True}

    # 3) ENTRADA CONFIRMADA → se não há pendência, decide GEN e publica
    parsed = parse_entry(text)
    if parsed:
        if _has_open_pending():
            # ignorar novo gatilho até fechar janela atual
            return {"ok": True, "ignored": "pending_open"}

        seq = parsed["seq"] or []
        after = parsed["after"]
        # Alimenta histórico com a sequência anterior
        append_seq(seq)
        best, conf, samples = choose_single_number(after)
        _open_pending(best)
        aft_txt = f" após {after}" if after else ""
        out = (f"🎯 <b>Número seco (G0):</b> <b>{best}</b>\n"
               f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
               f"📊 Conf: <b>{conf*100:.2f}%</b> | Amostra≈<b>{samples}</b>")
        await tg_send_text(TARGET_CHANNEL, out, "HTML")
        return {"ok": True, "posted": True, "best": best, "conf": conf, "samples": samples}

    return {"ok": True, "skipped": "unrelated"}
