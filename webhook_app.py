#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webhook_app.py — Guardião (modo GEN apenas)
- Não gera sinal sozinho; reage somente ao canal-fonte.
- Lê "ENTRADA CONFIRMADA" do canal-fonte, escolhe 1 número (1..4) por n-grams (histórico),
  publica no TARGET_CHANNEL e acompanha G0→G1→G2 até GREEN ou LOSS.
- Ignora novas entradas enquanto houver pendência aberta.
- Resumo/placar a cada 5 minutos e reset diário 00:00 UTC.

ENV obrigatórias:
  TG_BOT_TOKEN     -> token do bot
  WEBHOOK_TOKEN    -> token simples usado na rota /webhook/<token>

ENV recomendadas:
  TARGET_CHANNEL       -> chat_id destino (ex.: -1002796105884)
  SOURCE_CHANNEL_ID    -> chat_id do canal-fonte (ex.: -1002810508717). Se setado, só aceita dele.
  DB_PATH              -> caminho do sqlite (default: /data/mini_ref.db)

Endpoints:
  GET  /                 ping
  POST /webhook/<token>  recebe updates do Telegram
"""

import os, re, json, time, sqlite3, asyncio
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import FastAPI, Request, HTTPException

# ========= ENV =========
TG_BOT_TOKEN       = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN      = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL     = os.getenv("TARGET_CHANNEL", "").strip() or "-1002796105884"
SOURCE_CHANNEL_ID  = os.getenv("SOURCE_CHANNEL_ID", "").strip()   # opcional (se setado, filtra)
DB_PATH            = os.getenv("DB_PATH", "/data/mini_ref.db").strip() or "/data/mini_ref.db"

TELEGRAM_API       = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# ========= App =========
app = FastAPI(title="Guardião GEN — webhook", version="2.0.0")

# ========= DB helpers =========
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=20.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_db():
    con = _connect()
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    # n-grams 2..5
    cur.execute("""CREATE TABLE IF NOT EXISTS ngram (
        n INTEGER NOT NULL, ctx TEXT NOT NULL, nxt INTEGER NOT NULL, w REAL NOT NULL,
        PRIMARY KEY (n, ctx, nxt)
    )""")
    # pendência atual (uma por vez)
    cur.execute("""CREATE TABLE IF NOT EXISTS pending_outcome (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        suggested INTEGER NOT NULL,
        stage INTEGER NOT NULL,       -- 0=G0, 1=G1, 2=G2
        open INTEGER NOT NULL,        -- 1 aberto, 0 fechado
        seen_numbers TEXT DEFAULT '', -- concat de números observados: "3|1|4"
        announced INTEGER NOT NULL DEFAULT 0
    )""")
    # placar diário (G0 only + loss), como combinado
    cur.execute("""CREATE TABLE IF NOT EXISTS daily_score (
        yyyymmdd TEXT PRIMARY KEY,
        g0 INTEGER NOT NULL DEFAULT 0,
        loss INTEGER NOT NULL DEFAULT 0,
        streak INTEGER NOT NULL DEFAULT 0
    )""")
    con.commit()
    con.close()

init_db()

def now_ts() -> int:
    return int(time.time())

def today_key_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")

# ========= n-grams =========
def get_tail(limit:int=400) -> List[int]:
    con = _connect()
    rows = con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def append_seq(seq: List[int]):
    if not seq: return
    con = _connect()
    cur = con.cursor()
    for n in seq:
        cur.execute("INSERT INTO timeline (created_at, number) VALUES (?,?)", (now_ts(), int(n)))
    con.commit()
    con.close()
    _update_ngrams()

def append_one(n: int):
    append_seq([int(n)])

def _update_ngrams(decay: float=0.985, max_n:int=5, window:int=400):
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

# pesos backoff
W4, W3, W2, W1 = 0.40, 0.30, 0.20, 0.10

def _ngram_backoff(tail: List[int], cand:int, after: Optional[int]) -> float:
    if not tail: return 0.0
    # Se “após X”, corta o contexto no último X (se existir)
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

def choose_single_number_GEN(after: Optional[int]) -> Tuple[int, float, int]:
    """Escolhe 1 número dentre [1,2,3,4] SEM restrição por BIG/SMALL/ODD/EVEN."""
    tail = get_tail(400)
    cands = [1,2,3,4]
    scores = {c: _ngram_backoff(tail, c, after) for c in cands}

    # se tudo zero, desempata por menor frequência recente (variância)
    if all(v == 0.0 for v in scores.values()):
        last = tail[-50:] if len(tail) >= 50 else tail
        freq = {c: last.count(c) for c in cands}
        best = sorted(cands, key=lambda x: (freq.get(x,0), x))[0]
        conf = 0.50
        return best, conf, len(tail)

    total = sum(scores.values()) or 1e-9
    post = {k: v/total for k,v in scores.items()}
    best = max(post.items(), key=lambda kv: kv[1])[0]
    conf = post[best]
    return best, conf, len(tail)

# ========= Parsers =========
ENTRY_RX = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
SEQ_RX   = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
AFTER_RX = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)

GALE1_RX = re.compile(r"Estamos\s+no\s*1[ºo]\s*gale", re.I)
GALE2_RX = re.compile(r"Estamos\s+no\s*2[ºo]\s*gale", re.I)

GREEN_PATTERNS = [
    re.compile(r"APOSTA\s+ENCERRADA.*?\bGREEN\b.*?\(([1-4])\)", re.I | re.S),
    re.compile(r"\bGREEN\b.*?Número[:\s]*([1-4])", re.I | re.S),
]
RED_PATTERNS = [
    re.compile(r"APOSTA\s+ENCERRADA.*?\b(RED|LOSS)\b.*?\(([1-4])\)", re.I | re.S),
    re.compile(r"\b(RED|LOSS)\b.*?Número[:\s]*([1-4])", re.I | re.S),
]

def parse_entry(text: str) -> Optional[Dict]:
    t = re.sub(r"\s+", " ", text).strip()
    if not ENTRY_RX.search(t):
        return None
    seq = []
    mseq = SEQ_RX.search(t)
    if mseq:
        nums = re.findall(r"[1-4]", mseq.group(1))
        seq = [int(x) for x in nums]
    mafter = AFTER_RX.search(t)
    after = int(mafter.group(1)) if mafter else None
    return {"seq": seq, "after": after, "raw": t}

def extract_green(text: str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text)
    for rx in GREEN_PATTERNS:
        m = rx.search(t)
        if m:
            return int(m.group(1))
    return None

def extract_red_number(text: str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text)
    for rx in RED_PATTERNS:
        m = rx.search(t)
        if m:
            # grupo 2 é o número
            try:
                return int(m.group(2))
            except Exception:
                pass
    return None

# ========= Pending & Score =========
def has_open_pending() -> bool:
    con = _connect()
    row = con.execute("SELECT 1 FROM pending_outcome WHERE open=1 LIMIT 1").fetchone()
    con.close()
    return bool(row)

def open_pending(suggested:int):
    con = _connect()
    con.execute("""INSERT INTO pending_outcome (created_at, suggested, stage, open, seen_numbers, announced)
                   VALUES (?,?,?,?,?,?)""",
                (now_ts(), int(suggested), 0, 1, "", 1))
    con.commit(); con.close()

def _append_seen(pid:int, n:int):
    con = _connect()
    row = con.execute("SELECT seen_numbers FROM pending_outcome WHERE id=?", (pid,)).fetchone()
    seen = (row["seen_numbers"] or "") if row else ""
    seen2 = (seen + ("|" if seen else "") + str(int(n)))
    con.execute("UPDATE pending_outcome SET seen_numbers=? WHERE id=?", (seen2, pid))
    con.commit(); con.close()

def _get_open():
    con = _connect()
    row = con.execute("""SELECT id, suggested, stage, open, seen_numbers FROM pending_outcome
                         WHERE open=1 ORDER BY id LIMIT 1""").fetchone()
    con.close()
    return row

def _set_stage(pid:int, stage:int):
    con = _connect()
    con.execute("UPDATE pending_outcome SET stage=? WHERE id=?", (int(stage), pid))
    con.commit(); con.close()

def _close_pending(pid:int):
    con = _connect()
    con.execute("UPDATE pending_outcome SET open=0 WHERE id=?", (pid,))
    con.commit(); con.close()

def update_daily_score(hit: bool, stage: int):
    y = today_key_utc()
    con = _connect()
    row = con.execute("SELECT g0,loss,streak FROM daily_score WHERE yyyymmdd=?", (y,)).fetchone()
    g0 = row["g0"] if row else 0
    loss = row["loss"] if row else 0
    streak = row["streak"] if row else 0

    if hit and stage == 0:
        g0 += 1; streak += 1
    elif not hit and stage == 0:
        loss += 1; streak = 0
    # se GREEN sair em G1/G2, a conversão G0/Loss -> GREEN não é feita aqui (modo simples)

    con.execute("""INSERT OR REPLACE INTO daily_score (yyyymmdd,g0,loss,streak)
                   VALUES (?,?,?,?)""", (y, g0, loss, streak))
    con.commit(); con.close()

async def send_scoreboard():
    y = today_key_utc()
    con = _connect()
    row = con.execute("SELECT g0,loss,streak FROM daily_score WHERE yyyymmdd=?", (y,)).fetchone()
    con.close()
    g0 = row["g0"] if row else 0
    loss = row["loss"] if row else 0
    streak = row["streak"] if row else 0
    total = g0 + loss
    acc = (g0/total*100.0) if total else 0.0
    txt = (f"📊 <b>Placar do dia</b>\n"
           f"🟢 G0:{g0}  🔴 Loss:{loss}\n"
           f"✅ Acerto: {acc:.2f}%\n"
           f"🔥 Streak: {streak} GREEN(s)")
    await tg_send_text(TARGET_CHANNEL, txt, "HTML")

# ========= Telegram =========
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN or not chat_id: return
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                "disable_web_page_preview": True})

# ========= Tarefas periódicas =========
async def _scoreboard_task():
    while True:
        try:
            await send_scoreboard()
        except Exception as e:
            print(f"[scoreboard] erro: {e}")
        await asyncio.sleep(300)  # 5 min

async def _daily_reset_task():
    while True:
        try:
            now = datetime.now(timezone.utc)
            tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            await asyncio.sleep(max(1.0, (tomorrow - now).total_seconds()))
            # zera placar
            y = today_key_utc()
            con = _connect()
            con.execute("""INSERT OR REPLACE INTO daily_score (yyyymmdd,g0,loss,streak)
                           VALUES (?,0,0,0)""", (y,))
            con.commit(); con.close()
            await tg_send_text(TARGET_CHANNEL, "🕛 <b>Reset diário (00:00 UTC)</b> — placar zerado.", "HTML")
        except Exception as e:
            print(f"[reset] erro: {e}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def _boot():
    # inicia placar 5min + reset 00:00
    try:
        asyncio.create_task(_scoreboard_task())
    except Exception as e:
        print(f"[boot] scoreboard_task erro: {e}")
    try:
        asyncio.create_task(_daily_reset_task())
    except Exception as e:
        print(f"[boot] daily_reset_task erro: {e}")

# ========= Rotas =========
@app.get("/")
async def root():
    return {"ok": True, "service": "Guardião GEN — webhook"}

def _from_source_chat(msg: Dict) -> bool:
    """Se SOURCE_CHANNEL_ID estiver setado, garante que a msg vem dele."""
    try:
        chat = msg.get("chat") or {}
        cid = str(chat.get("id", ""))
        if SOURCE_CHANNEL_ID:
            return cid == SOURCE_CHANNEL_ID
        return True
    except Exception:
        return not bool(SOURCE_CHANNEL_ID)

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()

    msg = data.get("channel_post") or data.get("message") \
          or data.get("edited_channel_post") or data.get("edited_message") or {}
    text = (msg.get("text") or msg.get("caption") or "").strip()

    # filtra por canal-fonte, se configurado
    if not _from_source_chat(msg):
        return {"ok": True, "skipped": "not_from_source"}

    if not text:
        return {"ok": True, "skipped": "no_text"}

    # 1) GREEN/RED -> registra número real e fecha/atualiza pendência
    g = extract_green(text)
    r = extract_red_number(text)
    if g is not None or r is not None:
        n_observed = g if g is not None else r
        append_one(n_observed)  # alimenta histórico
        # fecha/avança pendência
        row = _get_open()
        if not row:
            return {"ok": True, "observed": n_observed, "note": "no_open_pending"}
        pid, suggested, stage = int(row["id"]), int(row["suggested"]), int(row["stage"])
        _append_seen(pid, n_observed)

        hit = (n_observed == suggested)
        if hit:
            # GREEN em stage atual
            await tg_send_text(TARGET_CHANNEL, f"✅ <b>GREEN</b> em <b>G{stage}</b> — Número: <b>{suggested}</b>", "HTML")
            if stage == 0:
                update_daily_score(True, 0)  # conta só G0/Loss
            _close_pending(pid)
        else:
            # não bateu no stage atual
            if stage >= 2:
                # fim em G2 -> LOSS
                await tg_send_text(TARGET_CHANNEL, f"❌ <b>LOSS</b> — Número: <b>{suggested}</b> (até G2)", "HTML")
                if stage == 0:
                    update_daily_score(False, 0)
                _close_pending(pid)
            else:
                # avança p/ próximo gale (G1 ou G2)
                new_stage = stage + 1
                _set_stage(pid, new_stage)

        return {"ok": True, "observed": n_observed, "hit": hit}

    # 2) Marcadores de gale (espelho do fonte)
    if GALE1_RX.search(text):
        if has_open_pending():
            await tg_send_text(TARGET_CHANNEL, "🔁 Estamos no <b>1° gale</b>", "HTML")
        return {"ok": True, "gale": 1}
    if GALE2_RX.search(text):
        if has_open_pending():
            await tg_send_text(TARGET_CHANNEL, "🔁 Estamos no <b>2° gale</b>", "HTML")
        return {"ok": True, "gale": 2}

    # 3) ENTRADA CONFIRMADA — só se NÃO houver pendência aberta
    parsed = parse_entry(text)
    if parsed:
        if has_open_pending():
            return {"ok": True, "skipped": "pending_open"}

        # alimenta histórico com a sequência passada
        seq = parsed["seq"] or []
        after = parsed["after"]
        if seq:
            append_seq(seq)

        # escolhe 1 número (GEN puro, 1..4)
        best, conf, samples = choose_single_number_GEN(after)

        # abre pendência e publica
        open_pending(best)
        aft_txt = f" após {after}" if after else ""
        out = (f"🎯 <b>Número seco (G0):</b> <b>{best}</b>\n"
               f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
               f"📊 Conf: <b>{conf*100:.2f}%</b> | Amostra≈<b>{samples}</b>")
        await tg_send_text(TARGET_CHANNEL, out, "HTML")
        return {"ok": True, "sent": True, "best": best, "conf": conf, "samples": samples}

    # 4) Se a mensagem tiver só "Sequência: ..." (ex.: análises), usamos para histórico
    mseq = SEQ_RX.search(text)
    if mseq and not ENTRY_RX.search(text):
        nums = re.findall(r"[1-4]", mseq.group(1))
        seq = [int(x) for x in nums]
        append_seq(seq)
        return {"ok": True, "fed_seq": len(seq)}

    # nada relevante
    return {"ok": True, "skipped": "unmatched"}
