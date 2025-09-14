#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webhook_app.py
--------------
FastAPI + Telegram webhook para refletir sinais do canal-fonte e publicar
um "número seco" (modo GEN = sem restrição de paridade/tamanho) no canal-alvo.
Também acompanha os gales (G1/G2) com base nas mensagens do canal-fonte.

Agora com:
- Placar diário (zera 00:00 America/Sao_Paulo)
- Mensagens de GREEN/LOSS contendo estágio (G0/G1/G2) + placar do dia
- Publicação automática do placar a cada hora
"""

import os, re, time, sqlite3, asyncio
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import FastAPI, Request, HTTPException
from zoneinfo import ZoneInfo

# ========= ENV =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1002796105884").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()  # se vazio, não filtra

DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"
TELEGRAM_API   = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

if not TG_BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")
if not WEBHOOK_TOKEN:
    raise RuntimeError("Defina WEBHOOK_TOKEN no ambiente.")

# ========= App =========
app = FastAPI(title="guardiao-auto-bot (GEN webhook)", version="3.0.0")

# ========= Utils =========
SP_TZ = ZoneInfo("America/Sao_Paulo")

def now_ts() -> int:
    return int(time.time())

def ts_str(ts=None) -> str:
    if ts is None: ts = now_ts()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%SZ")

def today_sp_str() -> str:
    """YYYY-MM-DD na timezone America/Sao_Paulo"""
    return datetime.now(SP_TZ).strftime("%Y-%m-%d")

# ========= DB helpers =========
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    # reduzir "database is locked"
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

def _column_exists(con: sqlite3.Connection, table: str, col: str) -> bool:
    r = con.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == col or row[1] == col for row in r)

def migrate_db():
    con = _connect(); cur = con.cursor()
    # timeline
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    # ngram
    cur.execute("""CREATE TABLE IF NOT EXISTS ngram (
        n INTEGER NOT NULL, ctx TEXT NOT NULL, nxt INTEGER NOT NULL, w REAL NOT NULL,
        PRIMARY KEY (n, ctx, nxt)
    )""")
    # pending
    cur.execute("""CREATE TABLE IF NOT EXISTS pending (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER,
        suggested INTEGER,
        stage INTEGER DEFAULT 0,
        open INTEGER DEFAULT 1,
        seen TEXT
    )""")
    # garantir colunas (idempotente)
    if not _column_exists(con, "pending", "created_at"):
        cur.execute("ALTER TABLE pending ADD COLUMN created_at INTEGER")
    if not _column_exists(con, "pending", "suggested"):
        cur.execute("ALTER TABLE pending ADD COLUMN suggested INTEGER")
    if not _column_exists(con, "pending", "stage"):
        cur.execute("ALTER TABLE pending ADD COLUMN stage INTEGER DEFAULT 0")
    if not _column_exists(con, "pending", "open"):
        cur.execute("ALTER TABLE pending ADD COLUMN open INTEGER DEFAULT 1")
    if not _column_exists(con, "pending", "seen"):
        cur.execute("ALTER TABLE pending ADD COLUMN seen TEXT")

    # stats_day: placar por dia (zera naturalmente ao trocar a data)
    cur.execute("""CREATE TABLE IF NOT EXISTS stats_day (
        day   TEXT PRIMARY KEY,   -- 'YYYY-MM-DD' America/Sao_Paulo
        total INTEGER NOT NULL,
        green INTEGER NOT NULL,
        loss  INTEGER NOT NULL
    )""")

    con.commit(); con.close()

migrate_db()

# ========= Stats por dia (placar) =========
def _stats_day_get(d: str) -> Dict[str, int]:
    con = _connect()
    row = con.execute("SELECT total, green, loss FROM stats_day WHERE day=?", (d,)).fetchone()
    con.close()
    if not row:
        return {"total": 0, "green": 0, "loss": 0}
    return {"total": int(row["total"]), "green": int(row["green"]), "loss": int(row["loss"])}

def _stats_day_inc(outcome: str):
    d = today_sp_str()
    for attempt in range(6):
        try:
            con = _connect(); cur = con.cursor()
            cur.execute("""INSERT INTO stats_day(day,total,green,loss)
                           VALUES (?,0,0,0)
                           ON CONFLICT(day) DO NOTHING""", (d,))
            cur.execute("UPDATE stats_day SET total = total + 1 WHERE day=?", (d,))
            if (outcome or "").upper() == "GREEN":
                cur.execute("UPDATE stats_day SET green = green + 1 WHERE day=?", (d,))
            else:
                cur.execute("UPDATE stats_day SET loss  = loss  + 1 WHERE day=?", (d,))
            con.commit(); con.close(); return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                time.sleep(0.25*(attempt+1)); continue
            raise

def format_scoreboard_today() -> str:
    d = today_sp_str()
    s = _stats_day_get(d)
    total, green, loss = s["total"], s["green"], s["loss"]
    acc = (green / total * 100.0) if total > 0 else 0.0
    return (f"📆 {d} (America/Sao_Paulo)\n"
            f"📈 Placar geral: {green} GREEN / {loss} LOSS — Acerto: {acc:.1f}% (N={total})")

# ========= N-gram memória =========
def get_tail(limit:int=400) -> List[int]:
    con = _connect()
    rows = con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def _exec_write(sql: str, params: tuple=()):
    # robust com retry para "database is locked"
    for attempt in range(6):
        try:
            con = _connect()
            cur = con.cursor()
            cur.execute(sql, params)
            con.commit()
            con.close()
            return
        except sqlite3.OperationalError as e:
            emsg = str(e).lower()
            if "locked" in emsg or "busy" in emsg:
                time.sleep(0.25*(attempt+1))
                continue
            raise

def append_seq(seq: List[int]):
    if not seq: return
    for n in seq:
        _exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)",
                    (now_ts(), int(n)))
    _update_ngrams()

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
    row_tot = con.execute("SELECT SUM(w) AS s FROM ngram WHERE n=? AND ctx=?",
                          (n, ctx_key)).fetchone()
    tot = (row_tot["s"] or 0.0) if row_tot else 0.0
    if tot <= 0:
        con.close(); return 0.0
    row_c = con.execute("SELECT w FROM ngram WHERE n=? AND ctx=? AND nxt=?",
                        (n, ctx_key, int(cand))).fetchone()
    w = (row_c["w"] or 0.0) if row_c else 0.0
    con.close()
    return w / tot

W4, W3, W2, W1 = 0.40, 0.30, 0.20, 0.10
def _ngram_backoff(tail: List[int], after: Optional[int], cand:int) -> float:
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

def choose_single_number(after: Optional[int]) -> Tuple[int, float, int]:
    cands = [1,2,3,4]
    tail = get_tail(400)
    scores = {c: _ngram_backoff(tail, after, c) for c in cands}
    if all(v == 0.0 for v in scores.values()):
        last = tail[-50:] if len(tail) >= 50 else tail
        freq = {c: last.count(c) for c in cands}
        # desempata pelo menor índice para dar variedade
        best = sorted(cands, key=lambda x: (-freq.get(x,0), -x))[0]
        conf = 0.50
        return best, conf, len(tail)
    total = sum(scores.values()) or 1e-9
    post = {k: v/total for k,v in scores.items()}
    best = max(post.items(), key=lambda kv: kv[1])[0]
    conf = float(post[best])
    return best, conf, len(tail)

# ========= Regex =========
ENTRY_RX = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
SEQ_RX = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
AFTER_RX = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)

GALE1_RX = re.compile(r"Estamos\s+no\s*1[ºo]\s*gale", re.I)
GALE2_RX = re.compile(r"Estamos\s+no\s*2[ºo]\s*gale", re.I)

# Desfechos (mais exigentes para evitar falsos positivos em posts de resumo)
GREEN_RX = re.compile(r"(resultado|final|GREEN)\s*.*?(green|✅|win)", re.I)
LOSS_RX  = re.compile(r"(resultado|final|LOSS)\s*.*?(loss|perdemos|❌)", re.I)

def parse_entry_text(text: str) -> Optional[Dict]:
    t = re.sub(r"\s+", " ", text).strip()
    if not ENTRY_RX.search(t):
        return None
    mseq = SEQ_RX.search(t)
    seq = []
    if mseq:
        parts = re.findall(r"[1-4]", mseq.group(1))
        seq = [int(x) for x in parts]
    mafter = AFTER_RX.search(t)
    after_num = int(mafter.group(1)) if mafter else None
    return {"seq": seq, "after": after_num, "raw": t}

# ========= Pending helpers =========
def get_open_pending() -> Optional[sqlite3.Row]:
    con = _connect()
    row = con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone()
    con.close()
    return row

def open_pending(suggested: int):
    _exec_write("""INSERT INTO pending (created_at, suggested, stage, open, seen)
                   VALUES (?,?,?,?,?)""",
                (now_ts(), int(suggested), 0, 1, ""))

def set_stage(stage:int):
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE pending SET stage=? WHERE open=1", (int(stage),))
    con.commit(); con.close()

def close_pending(outcome:str):
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE pending SET open=0, seen=? WHERE open=1", (outcome,))
    con.commit(); con.close()

# ========= Telegram =========
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN: return
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                "disable_web_page_preview": True})

# ========= Tarefa periódica: placar horário =========
async def hourly_scoreboard_publisher():
    # publica no “fechar da hora”: 01:00, 02:00, 03:00, ...
    while True:
        now = datetime.now(SP_TZ)
        next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
        await asyncio.sleep(max(1.0, (next_hour - now).total_seconds()))
        try:
            sb = format_scoreboard_today()
            await tg_send_text(TARGET_CHANNEL, sb)
        except Exception:
            # evita travar o loop se der erro transitório
            pass

@app.on_event("startup")
async def _startup_tasks():
    asyncio.create_task(hourly_scoreboard_publisher())

# ========= Rotas =========
@app.get("/")
async def root():
    return {"ok": True, "service": "guardiao-auto-bot (GEN webhook)"}

@app.get("/health")
async def health():
    pend = bool(get_open_pending())
    return {"ok": True, "db": DB_PATH, "pending_open": pend, "time": ts_str()}

@app.get("/placar")
async def placar():
    d = today_sp_str()
    s = _stats_day_get(d)
    acc = (s["green"]/s["total"]*100.0) if s["total"]>0 else 0.0
    return {"ok": True, "day": d, "tz": "America/Sao_Paulo",
            "total": s["total"], "green": s["green"], "loss": s["loss"],
            "accuracy_pct": round(acc, 2)}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()
    msg = data.get("channel_post") or data.get("message") \
        or data.get("edited_channel_post") or data.get("edited_message") or {}

    text = (msg.get("text") or msg.get("caption") or "").strip()
    chat = msg.get("chat") or {}
    chat_id = str(chat.get("id") or "")

    # se SOURCE_CHANNEL estiver definido, filtra por ele
    if SOURCE_CHANNEL and chat_id != str(SOURCE_CHANNEL):
        return {"ok": True, "skipped": "outro_chat"}

    if not text:
        return {"ok": True, "skipped": "sem_texto"}

    # Comando manual para mostrar o placar do dia
    if text.strip().lower() in ("/placar", "placar", "/placar@seubot"):
        sb = format_scoreboard_today()
        await tg_send_text(TARGET_CHANNEL, sb)
        return {"ok": True, "posted": "placar_hoje"}

    # === Eventos de Gale
    if GALE1_RX.search(text):
        if get_open_pending():
            set_stage(1)
            await tg_send_text(TARGET_CHANNEL, "🔁 Estamos no <b>1° gale (G1)</b>")
        return {"ok": True, "noted": "g1"}

    if GALE2_RX.search(text):
        if get_open_pending():
            set_stage(2)
            await tg_send_text(TARGET_CHANNEL, "🔁 Estamos no <b>2° gale (G2)</b>")
        return {"ok": True, "noted": "g2"}

    # === Desfechos GREEN / LOSS (com estágio + placar do dia)
    if GREEN_RX.search(text):
        pend = get_open_pending()
        if pend:
            stage_val = int(pend["stage"] or 0)
            stage_label = f"G{stage_val}" if stage_val in (1,2) else "G0"
            close_pending("GREEN")
            _stats_day_inc("GREEN")
            sb = format_scoreboard_today()
            await tg_send_text(
                TARGET_CHANNEL,
                f"🟢 <b>GREEN</b> (<b>{stage_label}</b>) — finalizado.\n" + sb
            )
        return {"ok": True, "closed": "green"}

    if LOSS_RX.search(text):
        pend = get_open_pending()
        if pend:
            stage_val = int(pend["stage"] or 0)
            stage_label = f"G{stage_val}" if stage_val in (1,2) else "G0"
            close_pending("LOSS")
            _stats_day_inc("LOSS")
            sb = format_scoreboard_today()
            await tg_send_text(
                TARGET_CHANNEL,
                f"🔴 <b>LOSS</b> (<b>{stage_label}</b>) — finalizado.\n" + sb
            )
        return {"ok": True, "closed": "loss"}

    # === Nova entrada
    parsed = parse_entry_text(text)
    if not parsed:
        return {"ok": True, "skipped": "nao_eh_entrada_confirmada"}

    # Se já existe pendência aberta:
    pend = get_open_pending()
    if pend:
        # heurística: se já estávamos em G2 e chegou outra entrada, considera LOSS da anterior
        if int(pend["stage"] or 0) >= 2:
            stage_val = int(pend["stage"] or 0)  # deve ser 2 aqui
            stage_label = f"G{stage_val}" if stage_val in (1,2) else "G0"
            close_pending("LOSS")
            _stats_day_inc("LOSS")
            sb = format_scoreboard_today()
            await tg_send_text(
                TARGET_CHANNEL,
                f"🔴 <b>LOSS</b> (<b>{stage_label}</b>) — anterior encerrada.\n" + sb
            )
        else:
            # ignora abertura até encerrar
            return {"ok": True, "ignored": "ja_existe_pendente"}

    # Alimenta memória de sequência (se vier algo), antes de decidir
    seq = parsed["seq"] or []
    if seq:
        append_seq(seq)

    after = parsed["after"]
    best, conf, samples = choose_single_number(after)

    # Abre pendência e publica
    open_pending(best)
    aft_txt = f" após {after}" if after else ""
    txt = (
        f"🎯 <b>Número seco (G0):</b> <b>{best}</b>\n"
        f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
        f"📊 <b>Conf:</b> {conf*100:.2f}% | <b>Amostra≈</b>{samples}"
    )
    await tg_send_text(TARGET_CHANNEL, txt)

    return {"ok": True, "posted": True, "best": best, "conf": conf, "samples": samples}
