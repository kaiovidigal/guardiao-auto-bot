#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webhook_app.py — v4.3.2 (Ajuste de Confiança para TESTE mais agressivo)
----------------------------------------------------------------------
FastAPI + Telegram webhook que:
- Lê mensagens do canal-fonte e publica um "número seco" (1..4) no canal-alvo.
- Fluxo ESTRITO: nunca abre novo sinal enquanto o anterior não fechou de verdade.
- Acompanha gales (G1/G2) e fecha robusto (GREEN/LOSS) por observados.
- Aprende online com feedback: penaliza erro e REFORÇA o número vencedor real.
- Usa ENSEMBLE (Hedge) de 4 especialistas (n-grama+feedback, freq curta, freq longa, tendencia).
- Consciência de sequência (loss_streak) e jogada anti-tilt.
- Dedupe por update_id, abertura transacional, timeout (fecha com "X" só por tempo quando já há 2 observados).

ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN
ENV opcionais:    TARGET_CHANNEL, SOURCE_CHANNEL, DB_PATH
Webhook:          POST /webhook/{WEBHOOK_TOKEN}
"""
import os, re, time, sqlite3, math
from contextlib import contextmanager
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, HTTPException

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
app = FastAPI(title="guardiao-auto-bot (GEN webhook)", version="4.3.2")

# ========= Parâmetros =========
DECAY = 0.980
W4, W3, W2, W1 = 0.35, 0.35, 0.20, 0.10
OBS_TIMEOUT_SEC = 240

# ======== Gates (LIGADOS para diminuir o LOSS) ========
CONF_MIN    = 0.45 # <<<<< AJUSTADO PARA TESTE >>>>>
GAP_MIN     = 0.08
H_MAX       = 0.80
FREQ_WINDOW = 120

# ======== Cooldown após RED ========
COOLDOWN_N     = 3
CD_CONF_BOOST  = 0.00 # <<<<< TEMPORARIAMENTE REMOVIDO PARA TESTE >>>>>
CD_GAP_BOOST   = 0.03

# ======== Modo "sempre entrar" (DESLIGADO para filtrar sinais) ========
ALWAYS_ENTER = False

# ======== Online Learning (feedback) ========
FEED_BETA   = 0.55
FEED_POS    = 1.1
FEED_NEG    = 1.2
FEED_DECAY  = 0.995
WF4, WF3, WF2, WF1 = W4, W3, W2, W1

# ======== Empate técnico (legado) ========
GAP_SOFT = 0.010

# ======== Ensemble Hedge ========
HEDGE_ETA = 0.8
K_SHORT   = 60
K_LONG    = 300

# ========= Utils =========
def now_ts() -> int:
    return int(time.time())

def ts_str(ts=None) -> str:
    if ts is None: ts = now_ts()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _entropy_norm(post: Dict[int, float]) -> float:
    """Entropia normalizada (base 4). 0=concentrada, 1=uniforme."""
    eps = 1e-12
    H = -sum((p+eps) * math.log(p+eps, 4) for p in post.values())
    return H

# ========= DB helpers =========
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

def _column_exists(con: sqlite3.Connection, table: str, col: str) -> bool:
    r = con.execute(f"PRAGMA table_info({table})").fetchall()
    return any((row["name"] if isinstance(row, sqlite3.Row) else row[1]) == col for row in r)

@contextmanager
def _tx():
    con = _connect()
    try:
        con.execute("BEGIN IMMEDIATE")
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

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
        seen TEXT,
        opened_at INTEGER,
        "after" INTEGER,
        ctx1 TEXT, ctx2 TEXT, ctx3 TEXT, ctx4 TEXT
    )""")
    # score
    cur.execute("""CREATE TABLE IF NOT EXISTS score (
        id INTEGER PRIMARY KEY CHECK (id=1),
        green INTEGER DEFAULT 0,
        loss  INTEGER DEFAULT 0
    )""")
    row = con.execute("SELECT 1 FROM score WHERE id=1").fetchone()
    if not row:
        cur.execute("INSERT INTO score (id, green, loss) VALUES (1,0,0)")
    # feedback
    cur.execute("""CREATE TABLE IF NOT EXISTS feedback (
        n INTEGER NOT NULL, ctx TEXT NOT NULL, nxt INTEGER NOT NULL, w REAL NOT NULL,
        PRIMARY KEY (n, ctx, nxt)
    )""")
    # processed (dedupe)
    cur.execute("""CREATE TABLE IF NOT EXISTS processed (
        update_id TEXT PRIMARY KEY,
        seen_at   INTEGER NOT NULL
    )""")
    # state (cooldown + loss_streak)
    cur.execute("""CREATE TABLE IF NOT EXISTS state (
        id INTEGER PRIMARY KEY CHECK (id=1),
        cooldown_left INTEGER DEFAULT 0,
        loss_streak   INTEGER DEFAULT 0
    )""")
    row = con.execute("SELECT 1 FROM state WHERE id=1").fetchone()
    if not row:
        cur.execute("INSERT INTO state (id, cooldown_left, loss_streak) VALUES (1,0,0)")
    try:
        cur.execute("ALTER TABLE state ADD COLUMN loss_streak INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    # expert weights (Hedge)
    cur.execute("""CREATE TABLE IF NOT EXISTS expert_w (
        id INTEGER PRIMARY KEY CHECK (id=1),
        w1 REAL NOT NULL,
        w2 REAL NOT NULL,
        w3 REAL NOT NULL,
        w4 REAL NOT NULL
    )""")
    row = con.execute("SELECT 1 FROM expert_w WHERE id=1").fetchone()
    if not row:
        cur.execute("INSERT INTO expert_w (id, w1, w2, w3, w4) VALUES (1, 1.0, 1.0, 1.0, 1.0)")
    else:
        if not _column_exists(con, 'expert_w', 'w4'):
            cur.execute("ALTER TABLE expert_w ADD COLUMN w4 REAL DEFAULT 1.0")
    con.commit(); con.close()
migrate_db()

def _exec_write(sql: str, params: tuple=()):
    for attempt in range(6):
        try:
            with _tx() as con:
                con.execute(sql, params)
            return
        except sqlite3.OperationalError as e:
            emsg = str(e).lower()
            if "locked" in emsg or "busy" in emsg:
                time.sleep(0.25*(attempt+1))
                continue
            raise

# ========= Dedupe =========
def _is_processed(update_id: str) -> bool:
    if not update_id: return False
    con = _connect()
    row = con.execute("SELECT 1 FROM processed WHERE update_id=?", (update_id,)).fetchone()
    con.close()
    return bool(row)

def _mark_processed(update_id: str):
    if not update_id: return
    _exec_write("INSERT OR IGNORE INTO processed (update_id, seen_at) VALUES (?,?)",
                (str(update_id), now_ts()))

# ========= Score helpers =========
def bump_score(outcome: str) -> Tuple[int, int]:
    with _tx() as con:
        row = con.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
        g, l = (row["green"], row["loss"]) if row else (0, 0)
        if outcome.upper() == "GREEN":
            g += 1
        elif outcome.upper() == "LOSS":
            l += 1
        con.execute("INSERT OR REPLACE INTO score (id, green, loss) VALUES (1,?,?)", (g, l))
        return g, l

def score_text() -> str:
    con = _connect()
    row = con.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
    con.close()
    if not row:
        return "0 GREEN × 0 LOSS — 0.0%"
    g, l = int(row["green"]), int(row["loss"])
    total = g + l
    acc = (g/total*100.0) if total > 0 else 0.0
    return f"{g} GREEN × {l} LOSS — {acc:.1f}%"

# ========= State helpers =========
def _get_cooldown() -> int:
    con = _connect()
    row = con.execute("SELECT cooldown_left FROM state WHERE id=1").fetchone()
    con.close()
    return int((row["cooldown_left"] if row else 0) or 0)

def _set_cooldown(v:int):
    _exec_write("UPDATE state SET cooldown_left=? WHERE id=1", (int(v),))

def _dec_cooldown():
    with _tx() as con:
        row = con.execute("SELECT cooldown_left FROM state WHERE id=1").fetchone()
        cur = int((row["cooldown_left"] if row else 0) or 0)
        cur = max(0, cur-1)
        con.execute("UPDATE state SET cooldown_left=? WHERE id=1", (cur,))

def _get_loss_streak() -> int:
    con = _connect()
    row = con.execute("SELECT loss_streak FROM state WHERE id=1").fetchone()
    con.close()
    return int((row["loss_streak"] if row else 0) or 0)

def _set_loss_streak(v:int):
    _exec_write("UPDATE state SET loss_streak=? WHERE id=1", (int(v),))

def _bump_loss_streak(reset: bool):
    if reset:
        _set_loss_streak(0)
    else:
        cur = _get_loss_streak()
        _set_loss_streak(cur + 1)

# ========= N-gram & Feedback =========
def timeline_size() -> int:
    con = _connect()
    row = con.execute("SELECT COUNT(*) AS c FROM timeline").fetchone()
    con.close()
    return int(row["c"] or 0)

def get_tail(limit:int=400) -> List[int]:
    con = _connect()
    rows = con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def append_seq(seq: List[int]):
    if not seq: return
    with _tx() as con:
        for n in seq:
            con.execute("INSERT INTO timeline (created_at, number) VALUES (?,?)",
                        (now_ts(), int(n)))
    _update_ngrams()

def _update_ngrams(decay: float=DECAY, max_n:int=5, window:int=400):
    tail = get_tail(window)
    if len(tail) < 2: return
    with _tx() as con:
        for t in range(1, len(tail)):
            nxt = int(tail[t])
            dist = (len(tail)-1) - t
            w = decay ** dist
            for n in range(2, max_n+1):
                if t-(n-1) < 0: break
                ctx = tail[t-(n-1):t]
                ctx_key = ",".join(str(x) for x in ctx)
                con.execute("""
                  INSERT INTO ngram (n, ctx, nxt, w)
                  VALUES (?,?,?,?)
                  ON CONFLICT(n, ctx, nxt) DO UPDATE SET w = w + excluded.w
                """, (n, ctx_key, nxt, float(w)))

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

def _ctx_to_key(ctx: List[int]) -> str:
    return ",".join(str(x) for x in ctx) if ctx else ""

def _feedback_upsert(n:int, ctx_key:str, nxt:int, delta
