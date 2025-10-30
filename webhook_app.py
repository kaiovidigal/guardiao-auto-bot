#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webhook_app.py — v4.5.0-g6 (Final)
- Fluxo estrito + Anti-tilt sem reduzir sinais + robustez de canal
- 7 Especialistas (incluindo IA local/LLM) + Hedge (Ensemble)
- "ANALISANDO": aprende sequência, pode adiantar fechamento
- Placar zera todo dia às 00:00 (fuso TZ_NAME, default America/Sao_Paulo)
- CHAIN_ON: após fechar, abre automaticamente um novo sinal
- Suporte até G6 (7 observações: G0..G6)

ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN
ENV opcionais:    TARGET_CHANNEL, SOURCE_CHANNEL, DB_PATH, DEBUG_MSG, BYPASS_SOURCE
                  LLM_ENABLED, LLM_MODEL_PATH, LLM_CTX_TOKENS, LLM_N_THREADS, LLM_TEMP, LLM_TOP_P
                  TZ_NAME, CHAIN_ON

Webhook:          POST /webhook/{WEBHOOK_TOKEN}
"""
import os, re, time, sqlite3, math, json
from contextlib import contextmanager
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone

# ---- timezone (reset diário) ----
TZ_NAME = os.getenv("TZ_NAME", "America/Sao_Paulo").strip()
try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo(TZ_NAME)
except ImportError:
    # Fallback for Python versions < 3.9
    _TZ = timezone.utc

import httpx
from fastapi import FastAPI, Request, HTTPException

# ========= ENV =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1002796105884").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()  # se vazio, não filtra
DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"
DEBUG_MSG      = os.getenv("DEBUG_MSG", "0").strip().lower() in ("1","true","yes") # Desativado por padrão ("0")
BYPASS_SOURCE  = os.getenv("BYPASS_SOURCE", "0").strip().lower() in ("1","true","yes")
TELEGRAM_API   = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# ===== LLM (IA local tipo ChatGPT) =====
LLM_ENABLED    = os.getenv("LLM_ENABLED", "1").strip().lower() in ("1","true","yes")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/phi-3-mini.gguf").strip()
LLM_CTX_TOKENS = int(os.getenv("LLM_CTX_TOKENS", "2048"))
LLM_N_THREADS  = int(os.getenv("LLM_N_THREADS", "4"))
LLM_TEMP       = float(os.getenv("LLM_TEMP", "0.2"))
LLM_TOP_P      = float(os.getenv("LLM_TOP_P", "0.95"))

# ===== Encadeamento =====
CHAIN_ON       = os.getenv("CHAIN_ON", "1").strip().lower() in ("1","true","yes")

if not TG_BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")
if not WEBHOOK_TOKEN:
    raise RuntimeError("Defina WEBHOOK_TOKEN no ambiente.")

# ========= App =========
app = FastAPI(title="guardiao-auto-bot (GEN webhook)", version="4.5.0-g6")

# ========= Parâmetros =========
DECAY = 0.980
W4, W3, W2, W1 = 0.42, 0.30, 0.18, 0.10
# NOVOS PESOS
W5, W6, W7 = 0.10, 0.10, 0.10
OBS_TIMEOUT_SEC = 240  # timeout adaptado p/ completar se faltar 1 observação

# ======== Limites de gales/observações ========
MAX_G   = 6                 # até G6
OBS_MAX = MAX_G + 1         # nº total de observações (G0..G6) = 7

# ======== Gates (não bloqueiam abertura) ========
CONF_MIN    = 0.70
GAP_MIN     = 0.12
H_MAX       = 0.80
FREQ_WINDOW = 90

# ======== Cooldown após RED (sem cortar fluxo) ========
COOLDOWN_N     = 5
CD_CONF_BOOST  = 0.04
CD_GAP_BOOST   = 0.03

# ======== Modo "sempre entrar" ========
ALWAYS_ENTER = True  # mantido: fluxo estrito = 1 pendência por vez, sem cortar sinal

# ======== Online Learning (feedback) ========
FEED_BETA   = 0.40
FEED_POS    = 0.70
FEED_NEG    = 1.20
FEED_DECAY  = 0.995
WF4, WF3, WF2, WF1 = W4, W3, W2, W1

# ======== Ensemble Hedge ========
HEDGE_ETA = 0.6
K_SHORT   = 60
K_LONG    = 300

# ========= Utils =========
def now_ts() -> int:
    return int(time.time())

def ts_str(ts=None) -> str:
    if ts is None: ts = now_ts()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def tz_today_ymd() -> str:
    dt = datetime.now(_TZ)
    return dt.strftime("%Y-%m-%d")

def _entropy_norm(post: Dict[int, float]) -> float:
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

def _ensure_column(con: sqlite3.Connection, table: str, col: str, decl: str):
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    names = {r["name"] for r in rows}
    if col not in names:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

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
        "after" INTEGER
    )""")
    _ensure_column(con, "pending", "after", "INTEGER")
    _ensure_column(con, "pending", "ctx1", "TEXT")
    _ensure_column(con, "pending", "ctx2", "TEXT")
    _ensure_column(con, "pending", "ctx3", "TEXT")
    _ensure_column(con, "pending", "ctx4", "TEXT")
    _ensure_column(con, "pending", "wait_notice_sent", "INTEGER DEFAULT 0")

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

    # state
    cur.execute("""CREATE TABLE IF NOT EXISTS state (
        id INTEGER PRIMARY KEY CHECK (id=1),
        cooldown_left INTEGER DEFAULT 0,
        loss_streak   INTEGER DEFAULT 0,
        last_reset_ymd TEXT DEFAULT ''
    )""")
    row = con.execute("SELECT 1 FROM state WHERE id=1").fetchone()
    if not row:
        cur.execute("INSERT INTO state (id, cooldown_left, loss_streak, last_reset_ymd) VALUES (1,0,0,'')")

    # expert weights
    cur.execute("""CREATE TABLE IF NOT EXISTS expert_w (
        id INTEGER PRIMARY KEY CHECK (id=1),
        w1 REAL NOT NULL,
        w2 REAL NOT NULL,
        w3 REAL NOT NULL,
        w4 REAL NOT NULL
    )""")
    row = con.execute("SELECT 1 FROM expert_w WHERE id=1").fetchone()
    if not row:
        cur.execute("INSERT INTO expert_w (id, w1, w2, w3, w4) VALUES (1,1.0,1.0,1.0,1.0)")
    else:
        _ensure_column(con, "expert_w", "w4", "REAL NOT NULL DEFAULT 1.0")
        _ensure_column(con, "expert_w", "w5", "REAL NOT NULL DEFAULT 1.0")
        _ensure_column(con, "expert_w", "w6", "REAL NOT NULL DEFAULT 1.0")
        _ensure_column(con, "expert_w", "w7", "REAL NOT NULL DEFAULT 1.0")

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

def reset_score():
    _exec_write("INSERT OR REPLACE INTO score (id, green, loss) VALUES (1,0,0)")

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

# ========= Daily reset =========
def _get_last_reset_ymd() -> str:
    con = _connect()
    row = con.execute("SELECT last_reset_ymd FROM state WHERE id=1").fetchone()
    con.close()
    return (row["last_reset_ymd"] or "") if row else ""

def _set_last_reset_ymd(ymd: str):
    _exec_write("UPDATE state SET last_reset_ymd=? WHERE id=1", (ymd,))

def check_and_maybe_reset_score():
    today = tz_today_ymd()
    last = _get_last_reset_ymd()
    if last != today:
        reset_score()
        _set_last_reset_ymd(today)

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

def _feedback_upsert(n:int, ctx_key:str, nxt:int, delta:float):
    with _tx() as con:
        con.execute("UPDATE feedback SET w = w * ?", (FEED_DECAY,))
        con.execute("""
          INSERT INTO feedback (n, ctx, nxt, w)
          VALUES (?,?,?,?)
          ON CONFLICT(n, ctx, nxt) DO UPDATE SET w = w + excluded.w
        """, (n, ctx_key, int(nxt), float(delta)))

def _feedback_prob(n:int, ctx: List[int], cand:int) -> float:
    if not ctx: return 0.0
    ctx_key = _ctx_to_key(ctx)
    con = _connect()
    row_tot = con.execute("SELECT SUM(w) AS s FROM feedback WHERE n=? AND ctx=?", (n, ctx_key)).fetchone()
    tot = (row_tot["s"] or 0.0) if row_tot else 0.0
    if tot <= 0:
        con.close(); return 0.0
    row_c = con.execute("SELECT w FROM feedback WHERE n=? AND ctx=? AND nxt=?", (n, ctx_key, int(cand))).fetchone()
    w = (row_c["w"] or 0.0) if row_c else 0.0
    con.close()
    tot = abs(tot)
    return max(0.0, w) / (tot if tot > 0 else 1e-9)

def _decision_context(after: Optional[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
    tail = get_tail(400)
    if tail and after is not None and after in tail:
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
    return ctx1, ctx2, ctx3, ctx4

def _post_from_tail(tail: List[int], after: Optional[int]) -> Dict[int, float]:
    cands = [1,2,3,4]
    scores = {c: 0.0 for c in cands}
    if not tail:
        return {c: 0.25 for c in cands}
    if after is not None and after in tail:
        idxs = [i for i,v in enumerate(tail) if v == after]
        i = idxs[-1]
        ctx1 = tail[i:i+1]
        ctx2 = tail[max(0,i-1):i+1] if i-1>=0 else []
        ctx3 = tail[max(0,i-2):i+1] if i-2>=0 else []
        ctx4 = tail[max(0,i-3):i+1] if i-3>=0 else []
    else:
        ctx4 = tail[-4:] if len(tail)>=4 else []
        ctx3 = tail[-3:] if len(tail)>=3 else []
        ctx2 = tail[-2:] if len(tail)>=2 else []
        ctx1 = tail[-1:] if len(tail)>=1 else []
    for c in cands:
        s = 0.0
        if len(ctx4)==4: s += W4 * _prob_from_ngrams(ctx4[:-1], c)
        if len(ctx3)==3: s += W3 * _prob_from_ngrams(ctx3[:-1], c)
        if len(ctx2)==2: s += W2 * _prob_from_ngrams(ctx2[:-1], c)
        if len(ctx1)==1: s += W1 * _prob_from_ngrams(ctx1[:-1], c)
        if len(ctx4)==4: s += FEED_BETA * WF4 * _feedback_prob(4, ctx4[:-1], c)
        if len(ctx3)==3: s += FEED_BETA * WF3 * _feedback_prob(3, ctx3[:-1], c)
        if len(ctx2)==2: s += FEED_BETA * WF2 * _feedback_prob(2, ctx2[:-1], c)
        if len(ctx1)==1: s += FEED_BETA * WF1 * _feedback_prob(1, ctx1[:-1], c)
        scores[c] = s
    tot = sum(scores.values()) or 1e-9
    return {k: v/tot for k,v in scores.items()}

# ========= NOVOS ESPECIALISTAS (5, 6, 7) =========
def _norm_dict(d: Dict[int,float]) -> Dict[int,float]:
    s = sum(d.values()) or 1e-9
    return {k: v/s for k,v in d.items()}

def _repetition_expert(tail: List[int]) -> Dict[int, float]:
    probs = {c: 0.25 for c in [1, 2, 3, 4]}
    if len(tail) < 3: return probs

    last_three = tail[-3:]
    if last_three[0] == last_three[2] and last_three[0] != last_three[1]:
        # Padrão A, B, A -> aposta em B e na repetição A
        probs[last_three[1]] += 0.1
        probs[last_three[0]] += 0.05
        # Reduz os outros
        for c in probs:
            if c != last_three[1] and c != last_three[0]:
                probs[c] -= 0.075

    return _norm_dict(probs)

def _parity_freq_expert(tail: List[int]) -> Dict[int, float]:
    probs = {c: 0.25 for c in [1, 2, 3, 4]}
    if not tail: return probs

    last_20 = tail[-20:]
    par_freq = { 'odd': 0, 'even': 0 }

    for n in last_20:
        if n % 2 == 0: # 2, 4
            par_freq['even'] += 1
        else:          # 1, 3
            par_freq['odd'] += 1

    total = len(last_20) or 1

    odd_score = par_freq['odd'] / total
    even_score = par_freq['even'] / total

    # Se a frequência de pares/ímpares está desequilibrada (> 15% de diferença), aposta no oposto
    if even_score > odd_score + 0.15:
        probs[1] += 0.1; probs[3] += 0.1
        probs[2] -= 0.1; probs[4] -= 0.1
    elif odd_score > even_score + 0.15:
        probs[2] += 0.1; probs[4] += 0.1
        probs[1] -= 0.1; probs[3] -= 0.1

    return _norm_dict(probs)

def _gap_analysis_expert(tail: List[int]) -> Dict[int, float]:
    probs = {c: 0.25 for c in [1, 2, 3, 4]}
    if not tail or len(tail) < 50: return probs

    last_50 = tail[-50:]
    last_gaps = {c: -1 for c in [1,2,3,4]} # -1 = not seen in 50

    # Encontra o último índice de cada número
    for i in range(len(last_50)-1, -1, -1):
        n = last_50[i]
        if last_gaps[n] == -1:
            last_gaps[n] = len(last_50) - 1 - i

    # O número que não aparece há mais tempo tem um voto extra
    last_gaps_valid = {k: v for k, v in last_gaps.items() if v != -1}
    if not last_gaps_valid: return probs # Evita erro se a janela for muito pequena

    most_overdue = sorted(last_gaps_valid.items(), key=lambda x: x[1], reverse=True)[0][0]

    # Se está atrasado o suficiente (ex: 10 posições ou mais)
    if last_gaps[most_overdue] >= 10:
        probs[most_overdue] += 0.2
        for c in probs:
            if c != most_overdue:
                probs[c] -= 0.06
    elif last_gaps[most_overdue] >= 5: # Atraso menor
        probs[most_overdue] += 0.1
        for c in probs:
            if c != most_overdue:
                probs[c] -= 0.03

    return _norm_dict(probs)

# ========= LLM local (Especialista 4) =========
try:
    from llama_cpp import Llama
    _LLM = None
    def _llm_load():
        global _LLM
        if _LLM is None and LLM_ENABLED and os.path.exists(LLM_MODEL_PATH):
            _LLM = Llama(
                model_path=LLM_MODEL_PATH,
                n_ctx=LLM_CTX_TOKENS,
                n_threads=LLM_N_THREADS,
                verbose=False
            )
        return _LLM
except Exception:
    _LLM = None
    def _llm_load():
        return None

_LLM_SYSTEM = (
    "Você é um assistente que prevê o próximo número de um stream discreto com classes {1,2,3,4}.\n"
    "Responda APENAS um JSON com as probabilidades normalizadas em porcentagem, "
    "com esta forma exata: {\"1\":p1,\"2\":p2,\"3\":p3,\"4\":p4}. Sem texto extra."
)

def _llm_probs_from_tail(tail: List[int]) -> Dict[int,float]:
    llm = _llm_load()
    if llm is None or not LLM_ENABLED:
        return {}
    win60  = tail[-60:] if len(tail) >= 60 else tail
    win300 = tail[-300:] if len(tail) >= 300 else tail
    def freq(win, n): return win.count(n)/max(1, len(win))
    feats = {
        "len": len(tail),
        "last10": tail[-10:],
        "freq60": {n: freq(win60, n) for n in (1,2,3,4)},
        "freq300": {n: freq(win300, n) for n in (1,2,3,4)},
    }
    user = (
        f"Historico_Recente={tail[-50:]}\n"
        f"Feats={feats}\n"
        "Devolva JSON com as probabilidades (%) para o PRÓXIMO número. "
        "Formato: {\"1\":p1,\"2\":p2,\"3\":p3,\"4\":p4}"
    )
    try:
        out = llm.create_chat_completion(
            messages=[
                {"role":"system","content":_LLM_SYSTEM},
                {"role":"user","content":user}
            ],
            temperature=LLM_TEMP,
            top_p=LLM_TOP_P,
            max_tokens=128
        )
        text = out["choices"][0]["message"]["content"].strip()
        m = re.search(r"\{.*\}", text, re.S)
        jtxt = m.group(0) if m else text
        data = json.loads(jtxt)
        raw = {int(k): float(v) for k,v in data.items() if str(k) in ("1","2","3","4")}
        S = sum(raw.values()) or 1e-9
        return {k: max(0.0, v/S) for k,v in raw.items() if k in (1,2,3,4)}
    except Exception:
        return {}

# ========= Ensemble (Hedge) — 7 especialistas =========

def _post_freq_k(tail: List[int], k: int) -> Dict[int,float]:
    if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
    win = tail[-k:] if len(tail) >= k else tail
    tot = max(1, len(win))
    return _norm_dict({c: win.count(c)/tot for c in [1,2,3,4]})

def _get_expert_w():
    con = _connect()
    row = con.execute("SELECT w1,w2,w3,w4,w5,w6,w7 FROM expert_w WHERE id=1").fetchone()
    con.close()
    if not row: return (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    return float(row["w1"]), float(row["w2"]), float(row["w3"]), float(row["w4"]), float(row["w5"]), float(row["w6"]), float(row["w7"])

def _set_expert_w(w1, w2, w3, w4, w5, w6, w7):
    _exec_write("UPDATE expert_w SET w1=?, w2=?, w3=?, w4=?, w5=?, w6=?, w7=? WHERE id=1",
                (float(w1), float(w2), float(w3), float(w4), float(w5), float(w6), float(w7)))

def _hedge_blend7(p1, p2, p3, p4, p5, p6, p7):
    w1, w2, w3, w4, w5, w6, w7 = _get_expert_w()
    S = (w1 + w2 + w3 + w4 + w5 + w6 + w7) or 1e-9
    w1, w2, w3, w4, w5, w6, w7 = (w1/S, w2/S, w3/S, w4/S, w5/S, w6/S, w7/S)
    blended = {c: w1*p1.get(c,0)+w2*p2.get(c,0)+w3*p3.get(c,0)+w4*p4.get(c,0)+w5*p5.get(c,0)+w6*p6.get(c,0)+w7*p7.get(c,0) for c in [1,2,3,4]}
    s2 = sum(blended.values()) or 1e-9
    return {k: v/s2 for k,v in blended.items()}, (w1,w2,w3,w4,w5,w6,w7)

def _hedge_update7(true_c:int, p1, p2, p3, p4, p5, p6, p7):
    w1, w2, w3, w4, w5, w6, w7 = _get_expert_w()
    l = lambda p: 1.0 - p.get(true_c, 0.0)
    from math import exp
    w1n = w1 * exp(-HEDGE_ETA * l(p1))
    w2n = w2 * exp(-HEDGE_ETA * l(p2))
    w3n = w3 * exp(-HEDGE_ETA * l(p3))
    w4n = w4 * exp(-HEDGE_ETA * l(p4))
    w5n = w5 * exp(-HEDGE_ETA * l(p5))
    w6n = w6 * exp(-HEDGE_ETA * l(p6))
    w7n = w7 * exp(-HEDGE_ETA * l(p7))
    S = (w1n + w2n + w3n + w4n + w5n + w6n + w7n) or 1e-9
    _set_expert_w(w1n/S, w2n/S, w3n/S, w4n/S, w5n/S, w6n/S, w7n/S)

# ========= Anti-tilt (sem reduzir sinais) =========
def _streak_adjust_choice(post:Dict[int,float], gap:float, ls:int) -> Tuple[int,str,Dict[int,float]]:
    reason = "IA"
    ranking = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    best = ranking[0][0]
    if ls >= 3:
        comp = _norm_dict({c: max(1e-9, 1.0 - post[c]) for c in [1,2,3,4]})
        post = _norm_dict({c: 0.7*post[c] + 0.3*comp[c] for c in [1,2,3,4]})
        ranking = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
        best = ranking[0][0]
        reason = "IA_anti_tilt_mix"
    if ls >= 2:
        top2 = ranking[:2]
        if len(top2) == 2 and gap < 0.05:
            best = top2[1][0]
            reason = "IA_runnerup_ls2"
    return best, reason, post

def choose_single_number(after: Optional[int]):
    tail = get_tail(400)
    post_e1 = _post_from_tail(tail, after)          # n-grama + feedback
    post_e2 = _post_freq_k(tail, K_SHORT)           # freq curta
    post_e3 = _post_freq_k(tail, K_LONG)            # freq longa
    post_e4 = _llm_probs_from_tail(tail) or {1:0.25,2:0.25,3:0.25,4:0.25}  # IA local
    # Novos especialistas
    post_e5 = _repetition_expert(tail)
    post_e6 = _parity_freq_expert(tail)
    post_e7 = _gap_analysis_expert(tail)

    post, (w1,w2,w3,w4,w5,w6,w7) = _hedge_blend7(post_e1, post_e2, post_e3, post_e4, post_e5, post_e6, post_e7)
    ranking = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    top2 = ranking[:2]
    gap = (top2[0][1] - top2[1][1]) if len(top2) >= 2 else ranking[0][1]
    base_best = ranking[0][0]
    conf = float(post[base_best])
    ls = _get_loss_streak()
    best, reason, post_adj = _streak_adjust_choice(post, gap, ls)
    conf = float(post_adj[best])
    r2 = sorted(post_adj.items(), key=lambda kv: kv[1], reverse=True)[:2]
    gap2 = (r2[0][1] - r2[1][1]) if len(r2) == 2 else r2[0][1]
    return best, conf, timeline_size(), post_adj, gap2, reason, ls

# ========= Parse (com keycaps e ANALISANDO) =========
ENTRY_RX       = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
SEQ_RX         = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
AFTER_RX       = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)
ANALISANDO_RX  = re.compile(r"\bANALISANDO\b", re.I)
GALE1_RX = re.compile(r"Estamos\s+no\s*1[ºo]\s*gale", re.I)
GALE2_RX = re.compile(r"Estamos\s+no\s*2[ºo]\s*gale", re.I)
GALE3_RX = re.compile(r"Estamos\s+no\s*3[ºo]\s*gale", re.I)
GALE4_RX = re.compile(r"Estamos\s+no\s*4[ºo]\s*gale", re.I)
GALE5_RX = re.compile(r"Estamos\s+no\s*5[ºo]\s*gale", re.I)
GALE6_RX = re.compile(r"Estamos\s+no\s*6[ºo]\s*gale", re.I)

GREEN_RX = re.compile(r"(?:\bgr+e+e?n\b|\bwin\b|✅)", re.I)
LOSS_RX  = re.compile(r"(?:\blo+s+s?\b|\bred\b|❌|\bperdemos\b)", re.I)

PAREN_GROUP_RX = re.compile(r"\(([^)]*)\)")
ANY_14_RX      = re.compile(r"[1-4]")

KEYCAP_MAP = {"1️⃣":"1","2️⃣":"2","3️⃣":"3","4️⃣":"4"}
def _normalize_keycaps(s: str) -> str:
    return "".join(KEYCAP_MAP.get(ch, ch) for ch in (s or ""))

def parse_entry_text(text: str) -> Optional[Dict]:
    t = _normalize_keycaps(re.sub(r"\s+", " ", text).strip())
    if not ENTRY_RX.search(t):
        return None
    mseq = SEQ_RX.search(t)
    seq = []
    if mseq:
        parts = re.findall(r"[1-4]", _normalize_keycaps(mseq.group(1)))
        seq = [int(x) for x in parts]
    mafter = AFTER_RX.search(t)
    after_num = int(mafter.group(1)) if mafter else None
    return {"seq": seq, "after": after_num, "raw": t}

def parse_close_numbers(text: str) -> List[int]:
    t = _normalize_keycaps(re.sub(r"\s+", " ", text))
    groups = PAREN_GROUP_RX.findall(t)
    if groups:
        last = groups[-1]
        nums = re.findall(r"[1-4]", _normalize_keycaps(last))
        return [int(x) for x in nums][:OBS_MAX]
    nums = ANY_14_RX.findall(t)
    return [int(x) for x in nums][:OBS_MAX]

def parse_analise_seq(text: str) -> List[int]:
    if not ANALISANDO_RX.search(_normalize_keycaps(text or "")):
        return []
    mseq = SEQ_RX.search(_normalize_keycaps(text or ""))
    if not mseq:
        return []
    return [int(x) for x in re.findall(r"[1-4]", mseq.group(1))]

# ========= Pending helpers =========
def get_open_pending() -> Optional[sqlite3.Row]:
    con = _connect()
    row = con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone()
    con.close()
    return row

def set_stage(stage:int):
    with _tx() as con:
        con.execute("UPDATE pending SET stage=? WHERE open=1", (int(stage),))

def _seen_list(row: sqlite3.Row) -> List[str]:
    seen = (row["seen"] or "").strip()
    return [s for s in seen.split("-") if s]

def _seen_append(row: sqlite3.Row, new_items: List[str]):
    cur_seen = _seen_list(row)
    for it in new_items:
        if len(cur_seen) >= OBS_MAX: break
        if it not in cur_seen:
            cur_seen.append(it)
    seen_txt = "-".join(cur_seen[:OBS_MAX])
    with _tx() as con:
        con.execute("UPDATE pending SET seen=? WHERE id=?", (seen_txt, int(row["id"])))

def _stage_from_observed(suggested: int, obs: List[int]) -> Tuple[str, str]:
    """Retorna (outcome, labelG) considerando até G6 (OBS_MAX=7)."""
    if not obs:
        return ("LOSS", f"G{MAX_G}")
    # primeiro acerto conta
    for k in range(min(len(obs), OBS_MAX)):
        if obs[k] == suggested:
            return ("GREEN", f"G{k}")
    # sem acerto em nenhuma tentativa
    return ("LOSS", f"G{MAX_G}")

def _ngram_snapshot_text(suggested: int) -> str:
    tail = get_tail(400)
    post = _post_from_tail(tail, after=None)
    def pct(x: float) -> str:
        try: return f"{x*100:.1f}%"
        except Exception: return "0.0%"
    p1 = pct(post.get(1, 0.0)); p2 = pct(post.get(2, 0.0))
    p3 = pct(post.get(3, 0.0)); p4 = pct(post.get(4, 0.0))
    conf = pct(post.get(int(suggested), 0.0))
    amostra = timeline_size()
    line1 = f"📈 Amostra: {amostra} • Conf: {conf}"
    line2 = f"🔎 E1(n-gram+fb): 1 {p1} | 2 {p2} | 3 {p3} | 4 {p4}"
    return line1 + "\n\n" + line2

def _format_seen(lst: List[str]) -> str:
    return "-".join(lst[:OBS_MAX]) if lst else ""

def _open_pending_with_ctx(suggested:int, after:Optional[int], ctx1, ctx2, ctx3, ctx4) -> bool:
    """Abertura transacional: só abre se não existir pendência aberta."""
    with _tx() as con:
        row = con.execute("SELECT 1 FROM pending WHERE open=1 LIMIT 1").fetchone()
        if row:
            return False
        con.execute("""INSERT INTO pending (created_at, suggested, stage, open, seen, opened_at, after, ctx1, ctx2, ctx3, ctx4, wait_notice_sent)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,0)""",
                    (now_ts(), int(suggested), 0, 1, "", now_ts(), after,
                     _ctx_to_key(ctx1), _ctx_to_key(ctx2), _ctx_to_key(ctx3), _ctx_to_key(ctx4)))
        return True

# ========= Telegram =========
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN: return
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(f"{TELEGRAM_API}/sendMessage",
                              json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                    "disable_web_page_preview": True})
    except Exception:
        pass

def _feedback_apply_on_close(row: sqlite3.Row, outcome: str, final_seen: str, suggested: int):
    """Atualiza feedback/hedge/timeline ao fechar."""
    try:
        # Extrai contextos salvos na abertura
        ctxs = []
        for ncol in ("ctx1","ctx2","ctx3","ctx4"):
            v = (row[ncol] or "").strip()
            ctx = [int(x) for x in v.split(",") if x.strip().isdigit()]
            ctxs.append(ctx)
        ctx1, ctx2, ctx3, ctx4 = ctxs

        if outcome.upper() == "GREEN":
            # peso por quão cedo acertou
            try:
                obs_nums = [int(x) for x in final_seen.split("-") if x.isdigit()]
            except Exception:
                obs_nums = []
            k_hit = None
            for k, n in enumerate(obs_nums[:OBS_MAX]):
                if n == suggested:
                    k_hit = k; break
            stage_weight = {
                0: 1.00, 1: 0.85, 2: 0.70, 3: 0.55, 4: 0.45, 5: 0.35, 6: 0.25
            }.get(k_hit if k_hit is not None else MAX_G, 0.50)
            delta = FEED_POS * stage_weight
            for (n,ctx) in [(1,ctx1),(2,ctx2),(3,ctx3),(4,ctx4)]:
                if len(ctx)>=1:
                    _feedback_upsert(n, _ctx_to_key(ctx[:-1]), suggested, delta)
        else:
            try:
                true_first = next(int(x) for x in final_seen.split("-") if x.isdigit())
            except StopIteration:
                true_first = None
            for (n,ctx) in [(1,ctx1),(2,ctx2),(3,ctx3),(4,ctx4)]:
                if len(ctx)>=1:
                    _feedback_upsert(n, _ctx_to_key(ctx[:-1]), suggested, -1.5*FEED_NEG)
                    if true_first is not None:
                        _feedback_upsert(n, _ctx_to_key(ctx[:-1]), true_first, +1.2*FEED_POS)

        # timeline a partir do fechamento
        obs_add = [int(x) for x in final_seen.split("-") if x.isdigit()]
        append_seq(obs_add)
    except Exception:
        pass

    # Hedge update — 7 especialistas
    try:
        true_first = None
        try:
            true_first = next(int(x) for x in final_seen.split("-") if x.isdigit())
        except StopIteration:
            pass
        if true_first is not None:
            tail_now = get_tail(400)
            post_e1 = _post_from_tail(tail_now, after=None)
            post_e2 = _post_freq_k(tail_now, K_SHORT)
            post_e3 = _post_freq_k(tail_now, K_LONG)
            post_e4 = _llm_probs_from_tail(tail_now) or {1:0.25,2:0.25,3:0.25,4:0.25}
            post_e5 = _repetition_expert(tail_now)
            post_e6 = _parity_freq_expert(tail_now)
            post_e7 = _gap_analysis_expert(tail_now)
            _hedge_update7(true_first, post_e1, post_e2, post_e3, post_e4, post_e5, post_e6, post_e7)
    except Exception:
        pass

def _compose_close_msg(outcome: str, stage_lbl: str, suggested: int, final_seen: str) -> str:
    our_num_display = suggested if outcome.upper()=="GREEN" else "X"
    snapshot = _ngram_snapshot_text(int(suggested))
    return (
        f"{'🟢' if outcome.upper()=='GREEN' else '🔴'} "
        f"<b>{outcome.upper()}</b> — finalizado "
        f"(<b>{stage_lbl}</b>, nosso={our_num_display}, observados={final_seen}).\n"
        f"📊 Geral: {score_text()}\n\n"
        f"{snapshot}"
    )

def _close_with_outcome(row: sqlite3.Row, outcome: str, final_seen: str, stage_lbl: str, suggested: int):
    # fecha na base
    with _tx() as con:
        con.execute("UPDATE pending SET open=0, seen=? WHERE id=?", (final_seen, int(row["id"])))

    # atualiza placar e estados
    bump_score(outcome.upper())
    try:
        if outcome.upper() == "LOSS":
            _set_cooldown(COOLDOWN_N)
            _bump_loss_streak(reset=False)
        else:
            _dec_cooldown()
            _bump_loss_streak(reset=True)
    except Exception:
        pass

    # feedback/hedge/timeline
    _feedback_apply_on_close(row, outcome, final_seen, suggested)

    # mensagem
    return _compose_close_msg(outcome, stage_lbl, suggested, final_seen)

def _maybe_close_by_timeout():
    """
    Se passou muito tempo e temos EXATAMENTE OBS_MAX-1 observados, completa com X e fecha.
    (ex.: para G6, OBS_MAX=7 → se houver 6 observados e o 7º não vier, completa)
    """
    row = get_open_pending()
    if not row: return None
    opened_at = int(row["opened_at"] or row["created_at"] or now_ts())
    if now_ts() - opened_at < OBS_TIMEOUT_SEC:
        return None
    seen_list = _seen_list(row)
    if len(seen_list) == OBS_MAX - 1:
        seen_list.append("X")
        final_seen = _format_seen(seen_list)
        suggested = int(row["suggested"] or 0)
        obs_nums = [int(x) for x in seen_list if x.isdigit()]
        outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
        return _close_with_outcome(row, outcome, final_seen, stage_lbl, suggested)
    return None

def close_pending():
    """Força fechar preenchendo X até OBS_MAX observações."""
    row = get_open_pending()
    if not row: return
    seen_list = _seen_list(row)
    while len(seen_list) < OBS_MAX:
        seen_list.append("X")
    final_seen = _format_seen(seen_list)
    suggested = int(row["suggested"] or 0)
    obs_nums = [int(x) for x in seen_list if x.isdigit()]
    outcome2, stage_lbl = _stage_from_observed(suggested, obs_nums)
    return _close_with_outcome(row, outcome2, final_seen, stage_lbl, suggested)

# ========= Helpers de abertura automática (cadeia) =========
async def _open_suggestion(after: Optional[int], origin_tag: str):
    """Abre a sugestão com o motor atual, se não houver pendência."""
    if get_open_pending():
        return {"ok": True, "skipped": "pending_open"}

    best, conf, samples, post, gap, reason, ls = choose_single_number(after)
    ctx1, ctx2, ctx3, ctx4 = _decision_context(after)
    if not _open_pending_with_ctx(best, after, ctx1, ctx2, ctx3, ctx4):
        return {"ok": True, "skipped": "race_lost"}

    # --- MODELO 2: Foco em Ação e Risco ---
    aft_txt = f"{after}" if after else "X"
    txt = (
        f"🚀 **ENTRADA IMEDIATA:** Número **{best}** (G0)\n"
        f"🌟 **CONFIANÇA** — **{conf*100:.1f}%**\n"
        f"📊 **Base:** Padrão {aft_txt} | Amostra ≈{samples}\n"
        f"💡 **Modo:** {reason} | Gap {gap*100:.1f}pp\n"
        f"🧠 **Streak RED:** {ls}"
    )
    # --- FIM DO MODELO 2 ---

    await tg_send_text(TARGET_CHANNEL, txt)
    return {"ok": True, "posted": True, "best": best}

# ========= Rotas =========
@app.get("/")
async def root():
    check_and_maybe_reset_score()
    return {"ok": True, "service": "guardiao-auto-bot (GEN webhook)"}

@app.get("/health")
async def health():
    check_and_maybe_reset_score()
    pend = get_open_pending()
    pend_open = bool(pend)
    seen = (pend["seen"] if pend else "")
    return {"ok": True, "db": DB_PATH, "pending_open": pend_open, "pending_seen": seen, "time": ts_str(), "tz": TZ_NAME}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    try:
        if token != WEBHOOK_TOKEN:
            raise HTTPException(status_code=403, detail="Forbidden")

        check_and_maybe_reset_score()

        data = await request.json()

        # DEDUPE por update_id
        upd_id = str(data.get("update_id", "")) if isinstance(data, dict) else ""
        if _is_processed(upd_id):
            return {"ok": True, "skipped": "duplicate_update"}
        _mark_processed(upd_id)

        # timeout pode fechar pendência antiga (apenas se faltar 1 observação)
        timeout_msg = _maybe_close_by_timeout()
        if timeout_msg:
            await tg_send_text(TARGET_CHANNEL, timeout_msg)

        msg = data.get("channel_post") or data.get("message") \
            or data.get("edited_channel_post") or data.get("edited_message") or {}

        text = (msg.get("text") or msg.get("caption") or "").strip()
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id") or "")

        # Filtro de fonte (bypass para diagnóstico)
        if SOURCE_CHANNEL and not BYPASS_SOURCE and chat_id != str(SOURCE_CHANNEL):
            if DEBUG_MSG:
                await tg_send_text(TARGET_CHANNEL, f"DEBUG: Ignorando chat {chat_id}. Fonte esperada: {SOURCE_CHANNEL}")
            return {"ok": True, "skipped": "outro_chat"}
        if not text:
            return {"ok": True, "skipped": "sem_texto"}

        # 0) ANALISANDO — aprende, pode adiantar e fechar; nunca abre novo SE já houver pendência
        if ANALISANDO_RX.search(_normalize_keycaps(text)):
            seq = parse_analise_seq(text)
            if seq:
                append_seq(seq)  # aprende padrão

                pend = get_open_pending()
                if pend:
                    # tentar adiantar/fechar
                    cur_seen = _seen_list(pend)
                    need = OBS_MAX - len(cur_seen)
                    if need > 0:
                        to_add = [str(n) for n in seq[:need]]
                        if to_add:
                            _seen_append(pend, to_add)
                            pend = get_open_pending()
                            cur_seen = _seen_list(pend)
                            if len(cur_seen) >= OBS_MAX:
                                suggested = int(pend["suggested"] or 0)
                                obs_nums = [int(x) for x in cur_seen if x.isdigit()]
                                outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
                                final_seen = _format_seen(cur_seen)
                                out_msg = _close_with_outcome(pend, outcome, final_seen, stage_lbl, suggested)
                                await tg_send_text(TARGET_CHANNEL, out_msg)
                                # CHAIN_ON: abrir novo APÓS fechar
                                if CHAIN_ON:
                                    mafter2 = AFTER_RX.search(_normalize_keycaps(text or ""))
                                    after2 = int(mafter2.group(1)) if mafter2 else None
                                    return await _open_suggestion(after2, origin_tag="analise_after_close")
                                return {"ok": True, "closed_from_analise": True, "seen": final_seen}
                else:
                    # sem pendência → pode abrir
                    mafter = AFTER_RX.search(_normalize_keycaps(text or ""))
                    after = int(mafter.group(1)) if mafter else None
                    return await _open_suggestion(after, origin_tag="analise")
            return {"ok": True, "analise_seen": len(seq)}

        # 1) Gales (informativo)
        for rx, stage_n in [(GALE1_RX,1),(GALE2_RX,2),(GALE3_RX,3),(GALE4_RX,4),(GALE5_RX,5),(GALE6_RX,6)]:
            if rx.search(text):
                if get_open_pending():
                    set_stage(stage_n)
                    await tg_send_text(TARGET_CHANNEL, f"🔁 Estamos no <b>{stage_n}° gale (G{stage_n})</b>")
                return {"ok": True, "noted": f"g{stage_n}"}

        # 2) Fechamentos do fonte (GREEN/LOSS) — usamos os números entre parênteses, até OBS_MAX
        if GREEN_RX.search(text) or LOSS_RX.search(text):
            pend = get_open_pending()
            if pend:
                nums = parse_close_numbers(text)  # pode ter 1..OBS_MAX observações
                if nums:
                    _seen_append(pend, [str(n) for n in nums])
                    pend = get_open_pending()
                seen_list = _seen_list(pend) if pend else []
                if pend and len(seen_list) >= OBS_MAX:
                    suggested = int(pend["suggested"] or 0)
                    obs_nums = [int(x) for x in seen_list if x.isdigit()]
                    outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
                    final_seen = _format_seen(seen_list)
                    out_msg = _close_with_outcome(pend, outcome, final_seen, stage_lbl, suggested)
                    await tg_send_text(TARGET_CHANNEL, out_msg)
                    # CHAIN_ON: abrir novo APÓS fechar
                    if CHAIN_ON:
                        mafter = AFTER_RX.search(_normalize_keycaps(text or ""))
                        after = int(mafter.group(1)) if mafter else None
                        return await _open_suggestion(after, origin_tag="after_close")
                    return {"ok": True, "closed": outcome.lower(), "seen": final_seen}
            return {"ok": True, "noted_close": True}

        # 3) Nova ENTRADA CONFIRMADA (Fluxo estrito)
        parsed = parse_entry_text(text)
        if not parsed:
            if DEBUG_MSG:
                await tg_send_text(TARGET_CHANNEL, "DEBUG: Mensagem não reconhecida como ENTRADA/FECHAMENTO/ANALISANDO.")
            return {"ok": True, "skipped": "nao_eh_entrada_confirmada"}

        # se existir pendência e estiver incompleta, NÃO abre novo
        pend = get_open_pending()
        if pend:
            seen_list = _seen_list(pend)
            if len(seen_list) >= OBS_MAX:
                suggested = int(pend["suggested"] or 0)
                obs_nums = [int(x) for x in seen_list if x.isdigit()]
                outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
                final_seen = _format_seen(seen_list)
                out_msg = _close_with_outcome(pend, outcome, final_seen, stage_lbl, suggested)
                await tg_send_text(TARGET_CHANNEL, out_msg)
                if CHAIN_ON:
                    after_chain = parsed["after"]
                    return await _open_suggestion(after_chain, origin_tag="entry_after_close")
            else:
                # mantém pendência aguardando fechamento
                return {"ok": True, "kept_open_waiting_close": True}

        # Alimenta memória com a sequência (se houver)
        seq = parsed["seq"] or []
        if seq: append_seq(seq)

        after = parsed["after"]
        best, conf, samples, post, gap, reason, ls = choose_single_number(after)

        # Abertura
        ctx1, ctx2, ctx3, ctx4 = _decision_context(after)
        opened = _open_pending_with_ctx(best, after, ctx1, ctx2, ctx3, ctx4)
        if not opened:
            if DEBUG_MSG:
                await tg_send_text(TARGET_CHANNEL, "DEBUG: Já existe pending open — não abri novo.")
            return {"ok": True, "skipped": "pending_already_open"}

        # --- MODELO 2: Foco em Ação e Risco ---
        aft_txt = f"{after}" if after else "X"
        txt = (
            f"🚀 **ENTRADA IMEDIATA:** Número **{best}** (G0)\n"
            f"🌟 **CONFIANÇA** — **{conf*100:.1f}%**\n"
            f"📊 **Base:** Padrão {aft_txt} | Amostra ≈{samples}\n"
            f"💡 **Modo:** {reason} | Gap {gap*100:.1f}pp\n"
            f"🧠 **Streak RED:** {ls}"
        )
        # --- FIM DO MODELO 2 ---

        await tg_send_text(TARGET_CHANNEL, txt)
        return {"ok": True, "posted": True, "best": best, "conf": conf, "gap": gap, "samples": samples}

    except HTTPException:
        # deixa o 403/405 etc. do FastAPI passar normal
        raise
    except Exception as e:
        # evita 502 no Render e loga o erro
        if DEBUG_MSG:
            try:
                await tg_send_text(TARGET_CHANNEL, f"DEBUG EXC: {e.__class__.__name__}: {e}")
            except Exception:
                pass
        return {"ok": True, "skipped": "internal_error_caught"}
