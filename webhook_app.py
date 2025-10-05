#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webhook_app.py — v4.5.0 (Fibonacci + SEQ3 + ConfFloor 30% + E8–E12 + M0)

- Fluxo estrito + anti-tilt conservador (opcional, via ENV)
- Especialista curto multi-janela em Fibonacci (opcional, via ENV)
- Regra de sequência de 3 números (trio -> próximo) com support/conf/boost/force (opcional, via ENV)
- IA local (LLM) como 4º especialista (opcional)
- Pós-ajuste com piso de confiança (min 30%) sem quebrar o fluxo
- Novos especialistas: E8 (Regime), E9 (Condicional), E10 (Dirichlet), E11 (Contrarian), E12 (Momentum)
- IA Regente (M0) que calibra pesos on-line (sem EV nova)
- "ANALISANDO": aprende sequência e pode adiantar fechamento
- Placar zera todo dia às 00:00 (fuso TZ_NAME)

ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN
ENV opcionais:    TARGET_CHANNEL, SOURCE_CHANNEL, DB_PATH, DEBUG_MSG, BYPASS_SOURCE
                  LLM_ENABLED, LLM_MODEL_PATH, LLM_CTX_TOKENS, LLM_N_THREADS, LLM_TEMP, LLM_TOP_P
                  TZ_NAME, FIB_FREQ_ENABLED, FIB_ANTITILT_ENABLED, HEDGE_FIB_SEED
                  SEQ3_ENABLED, SEQ3_MIN_SUPPORT, SEQ3_MIN_CONF, SEQ3_BOOST, SEQ3_FORCE
                  DEALER_KL_THRESH, DEALER_ALPHA
"""
import os, re, time, sqlite3, math, json
from contextmanager import contextmanager
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone

# ---- timezone (reset diário) ----
TZ_NAME = os.getenv("TZ_NAME", "America/Sao_Paulo").strip()
try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo(TZ_NAME)
except Exception:
    _TZ = timezone.utc

import httpx
from fastapi import FastAPI, Request, HTTPException

# ========= ENV =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1002796105884").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()  # se vazio, não filtra
DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"
DEBUG_MSG      = os.getenv("DEBUG_MSG", "0").lower() in ("1","true","yes")
BYPASS_SOURCE  = os.getenv("BYPASS_SOURCE", "0").lower() in ("1","true","yes")
TELEGRAM_API   = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# ===== Flags (safe by default: OFF) =====
FIB_FREQ_ENABLED     = os.getenv("FIB_FREQ_ENABLED", "0").lower() in ("1","true","yes")
FIB_ANTITILT_ENABLED = os.getenv("FIB_ANTITILT_ENABLED", "0").lower() in ("1","true","yes")
HEDGE_FIB_SEED       = os.getenv("HEDGE_FIB_SEED", "0").lower() in ("1","true","yes")

# ===== SEQ3 (trio->próximo) =====
SEQ3_ENABLED      = os.getenv("SEQ3_ENABLED", "0").lower() in ("1","true","yes")
SEQ3_MIN_SUPPORT  = int(os.getenv("SEQ3_MIN_SUPPORT", "12"))
SEQ3_MIN_CONF     = float(os.getenv("SEQ3_MIN_CONF", "0.78"))
SEQ3_BOOST        = float(os.getenv("SEQ3_BOOST", "0.18"))
SEQ3_FORCE        = os.getenv("SEQ3_FORCE", "0").lower() in ("1","true","yes")

# ===== Pós-ajuste (piso de confiança) =====
MIN_CONF_FLOOR = 0.30  # nunca exibir < 30% após ajustes
H_MAX          = 0.95  # teto de confiança

# ===== LLM (IA local tipo ChatGPT) =====
LLM_ENABLED    = os.getenv("LLM_ENABLED", "1").lower() in ("1","true","yes")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/phi-3-mini.gguf").strip()
LLM_CTX_TOKENS = int(os.getenv("LLM_CTX_TOKENS", "2048"))
LLM_N_THREADS  = int(os.getenv("LLM_N_THREADS", "4"))
LLM_TEMP       = float(os.getenv("LLM_TEMP", "0.10"))
LLM_TOP_P      = float(os.getenv("LLM_TOP_P", "0.90"))

if not TG_BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")
if not WEBHOOK_TOKEN:
    raise RuntimeError("Defina WEBHOOK_TOKEN no ambiente.")

# ========= App =========
app = FastAPI(title="guardiao-auto-bot (GEN webhook)", version="4.5.0")

# ========= Parâmetros =========
DECAY = 0.980
W4, W3, W2, W1 = 0.42, 0.30, 0.18, 0.10
OBS_TIMEOUT_SEC = 240  # fecha por timeout só se já houver 2 observados

# ======== Gates (diagnóstico; não bloqueiam abertura) ========
CONF_MIN    = 0.62
GAP_MIN     = 0.06
FREQ_WINDOW = 80

# ======== Cooldown após RED (sem cortar fluxo) ========
COOLDOWN_N     = 7
CD_CONF_BOOST  = 0.04
CD_GAP_BOOST   = 0.03

# ======== Modo "sempre entrar" ========
ALWAYS_ENTER = True

# ======== Online Learning (feedback) ========
FEED_BETA   = 0.40
FEED_POS    = 0.85
FEED_NEG    = 1.20
FEED_DECAY  = 0.995
WF4, WF3, WF2, WF1 = W4, W3, W2, W1

# ======== Ensemble Hedge ========
HEDGE_ETA = 0.75
K_SHORT   = 60
K_LONG    = 300

# ========= Utils =========
def now_ts() -> int: return int(time.time())

def ts_str(ts=None) -> str:
    if ts is None: ts = now_ts()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def tz_today_ymd() -> str:
    dt = datetime.now(_TZ); return dt.strftime("%Y-%m-%d")

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
        con.rollback(); raise
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
        ctx1 TEXT, ctx2 TEXT, ctx3 TEXT, ctx4 TEXT,
        wait_notice_sent INTEGER DEFAULT 0
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
    # expert weights (4 especialistas p/ LLM)
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
        try:
            con.execute("ALTER TABLE expert_w ADD COLUMN w4 REAL NOT NULL DEFAULT 1.0")
        except sqlite3.OperationalError:
            pass
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
                time.sleep(0.25*(attempt+1)); continue
            raise

# --- dedupe helpers (faltavam) ---
def _is_processed(update_id: str) -> bool:
    if not update_id:
        return False
    con = _connect()
    row = con.execute("SELECT 1 FROM processed WHERE update_id=?", (update_id,)).fetchone()
    con.close()
    return bool(row)

def _mark_processed(update_id: str):
    if not update_id:
        return
    _exec_write("INSERT OR IGNORE INTO processed (update_id, seen_at) VALUES (?,?)",
                (str(update_id), now_ts()))

# --- (1) Prior do Hedge com pesos de Fibonacci [1,1,2,3] normalizados (opcional) ---
def _get_expert_w():
    con = _connect()
    row = con.execute("SELECT w1,w2,w3,w4 FROM expert_w WHERE id=1").fetchone()
    con.close()
    if not row: return (1.0, 1.0, 1.0, 1.0)
    return float(row["w1"]), float(row["w2"]), float(row["w3"]), float(row["w4"])

def _set_expert_w(w1, w2, w3, w4):
    _exec_write("UPDATE expert_w SET w1=?, w2=?, w3=?, w4=? WHERE id=1",
                (float(w1), float(w2), float(w3), float(w4)))

def _seed_expert_w_fib():
    try:
        w1, w2, w3, w4 = _get_expert_w()
    except Exception:
        return
    def _is_flat(ws, tol=1e-6): return all(abs(x-1.0) < tol for x in ws)
    if _is_flat([w1, w2, w3, w4]):
        base = [1, 1, 2, 3]; s = sum(base)
        _set_expert_w(*(x/s for x in base))

try:
    if HEDGE_FIB_SEED:
        _seed_expert_w_fib()
except Exception:
    pass

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
        ctx1 = tail[max(0,i):i+1]
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

# ========= Ensemble helpers =========
def _norm_dict(d: Dict[int,float]) -> Dict[int,float]:
    s = sum(d.values()) or 1e-9
    return {k: v/s for k,v in d.items()}

# --- Frequência multi-janela com janelas de Fibonacci (conservador) ---
FIB_WINS = [21, 34, 55, 89]
_FIB_WIN_WEIGHTS = [1, 1, 1, 2]
_FIB_WIN_WEIGHTS = [w/sum(_FIB_WIN_WEIGHTS) for w in _FIB_WIN_WEIGHTS]

def _post_freq_k(tail: List[int], k: int) -> Dict[int,float]:
    if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
    win = tail[-k:] if len(tail) >= k else tail
    tot = max(1, len(win))
    return _norm_dict({c: win.count(c)/tot for c in [1,2,3,4]})

def _post_freq_fib(tail: List[int]) -> Dict[int, float]:
    if not tail:
        return {1:0.25, 2:0.25, 3:0.25, 4:0.25}
    acc = {1:0.0, 2:0.0, 3:0.0, 4:0.0}
    for k, wk in zip(FIB_WINS, _FIB_WIN_WEIGHTS):
        pk = _post_freq_k(tail, k)
        for c in (1,2,3,4):
            acc[c] += wk * pk[c]
    tot = sum(acc.values()) or 1e-9
    return {c: acc[c]/tot for c in (1,2,3,4)}

# ======== E8–E12 + M0 (sem EV novas) =========================================
def _last_runlength(tail: List[int]) -> int:
    if not tail: return 0
    x = tail[-1]; r = 1
    for i in range(len(tail)-2, -1, -1):
        if tail[i] == x: r += 1
        else: break
    return r

def _get_gale_stage_safe() -> int:
    row = get_open_pending()
    if not row: return 0
    try:
        st = int(row["stage"] or 0)
        return 2 if st >= 2 else (1 if st == 1 else 0)
    except Exception:
        return 0

# --- E8: Detector de Regime (usa 'Crupiê' e entropia) ------------------------
def _regime_prior() -> Dict[str, float]:
    snap = _dealer_snapshot()
    kl = float(snap.get("kl", 0.0))
    ent_s = float(snap.get("ent_s", 0.0))
    ent_l = float(snap.get("ent_l", 0.0))
    thr = float(os.getenv("DEALER_KL_THRESH", "0.22"))

    if kl > thr and ent_s < 0.70:
        regime = "tendencioso"
        w_mom = 0.65; w_ctr = 0.15; w_cond = 0.20
    elif ent_s > 0.85 and ent_l > 0.85:
        regime = "caotico"
        w_mom = 0.30; w_ctr = 0.40; w_cond = 0.30
    else:
        regime = "estavel"
        w_mom = 0.45; w_ctr = 0.20; w_cond = 0.35

    return {"regime": regime, "w_momentum": w_mom, "w_contrarian": w_ctr, "w_conditional": w_cond}

# --- E9: Transição Condicional (After + Runlength) ---------------------------
def _post_conditional(tail: List[int], after: Optional[int]) -> Dict[int, float]:
    if len(tail) < 6:
        return {1:0.25, 2:0.25, 3:0.25, 4:0.25}

    last = tail[-1]
    rl = _last_runlength(tail)
    rl_bucket = 1 if rl == 1 else (2 if rl == 2 else 3)

    counts = {1:1e-9, 2:1e-9, 3:1e-9, 4:1e-9}
    total = 0
    for i in range(3, len(tail)-1):
        a,b,c = tail[i-3], tail[i-2], tail[i-1]
        nxt = tail[i]
        _last = c
        _rl = 1
        if a == b == c: _rl = 3
        elif b == c:    _rl = 2

        cond_ok = True
        if after is not None:
            cond_ok = (_last == after)
        if cond_ok and _last == last and _rl == rl_bucket:
            counts[nxt] = counts.get(nxt, 0.0) + 1.0
            total += 1

    if total == 0:
        for i in range(len(tail)-1):
            if tail[i] == last:
                counts[tail[i+1]] = counts.get( tail[i+1], 0.0) + 1.0
                total += 1
        if total == 0:
            return {1:0.25, 2:0.25, 3:0.25, 4:0.25}

    return _norm_dict(counts)

# --- E10: Suavizador Bayesiano (Dirichlet) -----------------------------------
def _dirichlet_smooth(post: Dict[int, float], samples: int) -> Dict[int, float]:
    post = _norm_dict(post)
    alpha = max(0.2, min(1.2, 1.0 - 0.0005 * float(samples)))
    eff = {c: post[c] * max(1.0, float(samples)) for c in (1,2,3,4)}
    smooth = {c: (eff[c] + alpha) for c in (1,2,3,4)}
    tot = sum(smooth.values()) or 1e-9
    return {c: smooth[c]/tot for c in (1,2,3,4)}

# --- E11: Contrarian (reversão à média) --------------------------------------
def _post_contrarian(post_base: Dict[int, float]) -> Dict[int, float]:
    if not post_base: return {1:0.25,2:0.25,3:0.25,4:0.25}
    inv = {c: max(1e-9, 1.0 - float(post_base.get(c,0.0))) for c in (1,2,3,4)}
    return _norm_dict(inv)

# --- E12: Momentum de Curto Prazo (Burst) ------------------------------------
def _post_momentum_burst(tail: List[int]) -> Dict[int, float]:
    if not tail:
        return {1:0.25,2:0.25,3:0.25,4:0.25}
    wins = [8, 13, 21]
    weights = [0.2, 0.3, 0.5]
    acc = {1:0.0,2:0.0,3:0.0,4:0.0}
    for k, wk in zip(wins, weights):
        pk = _post_freq_k(tail, k)
        for c in (1,2,3,4):
            acc[c] += wk * pk[c]
    return _norm_dict(acc)

# --- M0: IA Regente (calibra pesos on-line) ----------------------------------
def _m0_blend_extended(base_post: Dict[int,float], tail: List[int], after: Optional[int], samples: int) -> Tuple[Dict[int,float], str]:
    pri = _regime_prior()
    w_mom = pri["w_momentum"]; w_ctr = pri["w_contrarian"]; w_cnd = pri["w_conditional"]

    p_cnd = _post_conditional(tail, after)
    p_mom = _post_momentum_burst(tail)
    p_ctr = _post_contrarian(base_post)

    w_base = max(0.0, 1.0 - (w_mom + w_ctr + w_cnd))
    mix = {
        c: w_base*base_post.get(c,0.0) + w_cnd*p_cnd.get(c,0.0) + w_mom*p_mom.get(c,0.0) + w_ctr*p_ctr.get(c,0.0)
        for c in (1,2,3,4)
    }
    mix = _norm_dict(mix)
    smooth = _dirichlet_smooth(mix, samples)
    tag = f"| M0:{pri['regime']} w0={w_base:.2f} wCnd={w_cnd:.2f} wMom={w_mom:.2f} wCtr={w_ctr:.2f}"
    return smooth, tag

# --- Anti-tilt com intensidade guiada por Fibonacci (conservador) ------------
def _fib(n: int) -> int:
    if n <= 1: return 1
    a, b = 1, 1
    for _ in range(n-1):
        a, b = b, a + b
    return a

def _streak_adjust_choice(post:Dict[int,float], gap:float, ls:int) -> Tuple[int,str,Dict[int,float]]:
    reason = "IA"
    ranking = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    best = ranking[0][0]

    if FIB_ANTITILT_ENABLED and ls >= 4 and gap < 0.04:
        k = min(3, _fib(ls))
        mix = min(0.30, 0.20 + 0.04 * k)
        comp = _norm_dict({c: max(1e-9, 1.0 - post[c]) for c in [1,2,3,4]})
        post = _norm_dict({c: (1.0 - mix)*post[c] + mix*comp[c] for c in [1,2,3,4]})
        ranking = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
        best = ranking[0][0]
        reason = f"IA_anti_tilt_fib_soft{k}"

    if ls >= 2:
        top2 = ranking[:2]
        if len(top2) == 2 and gap < 0.05:
            best = top2[1][0]
            reason = "IA_runnerup_ls2"
    return best, reason, post

# --- Pós-ajuste: garantir piso de confiança (30%) e cap no H_MAX -------------
def _apply_conf_floor(post: Dict[int,float], floor: float = MIN_CONF_FLOOR, cap: float = H_MAX) -> Tuple[Dict[int,float], str]:
    if not post: return post, ""
    post = _norm_dict({c: float(post.get(c,0.0)) for c in (1,2,3,4)})
    best = max(post, key=post.get)
    mx = post[best]
    tag = ""
    if mx < floor:
        others = [c for c in (1,2,3,4) if c != best]
        sum_others = sum(post[c] for c in others)
        need = floor - mx
        take = min(need, sum_others)
        if sum_others > 0:
            scale = (sum_others - take) / sum_others
            for c in others:
                post[c] *= scale
        post[best] = min(cap, mx + take)
        post = _norm_dict(post)
        tag = f"| conf_floor→{post[best]*100:.0f}%"
    if post[best] > cap:
        excess = post[best] - cap
        others = [c for c in (1,2,3,4) if c != best]
        add = excess / len(others)
        post[best] = cap
        for c in others:
            post[c] += add
        post = _norm_dict(post)
        tag += f"| cap{int(cap*100)}"
    return post, tag

# ========= (NOVO) Print do Crupiê =========
def _kl_divergence(p: Dict[int, float], q: Dict[int, float]) -> float:
    eps = 1e-12; s = 0.0
    for c in (1,2,3,4):
        pc = max(eps, float(p.get(c, 0.0)))
        qc = max(eps, float(q.get(c, 0.0)))
        s += pc * math.log(pc / qc)
    return s

def _dealer_snapshot() -> Dict[str, float]:
    tail = get_tail(400)
    p_short = _post_freq_k(tail, K_SHORT)
    p_long  = _post_freq_k(tail, K_LONG)
    kl  = _kl_divergence(p_short, p_long)
    dom_s = max(p_short, key=p_short.get)
    dom_l = max(p_long,  key=p_long.get)
    ent_s = _entropy_norm(p_short)
    ent_l = _entropy_norm(p_long)
    drift = kl > float(os.getenv("DEALER_KL_THRESH", "0.22"))
    return {"kl": kl, "drift": drift, "dom_s": dom_s, "dom_l": dom_l, "ent_s": ent_s, "ent_l": ent_l}

def _dealer_print_line() -> str:
    snap = _dealer_snapshot()
    try:
        thr = float(os.getenv("DEALER_KL_THRESH", "0.22"))
    except Exception:
        thr = 0.22
    alpha = os.getenv("DEALER_ALPHA", "0.0")
    status = "DRIFT ALTO" if snap["drift"] else "OK"
    return (f"🧑‍💼 <b>Crupiê</b>: KL={snap['kl']:.3f} (thr {thr}) • {status} • "
            f"curto→{snap['dom_s']} | longo→{snap['dom_l']} • Hs={snap['ent_s']:.2f} Hl={snap['ent_l']:.2f} • α={alpha}")

# ========= Hedge (blend/atualização) =========
def _hedge_blend4(p1:Dict[int,float], p2:Dict[int,float], p3:Dict[int,float], p4:Dict[int,float]):
    w1, w2, w3, w4 = _get_expert_w()
    S = (w1 + w2 + w3 + w4) or 1e-9
    w1, w2, w3, w4 = (w1/S, w2/S, w3/S, w4/S)
    blended = {c: w1*p1.get(c,0)+w2*p2.get(c,0)+w3*p3.get(c,0)+w4*p4.get(c,0) for c in [1,2,3,4]}
    s2 = sum(blended.values()) or 1e-9
    return {k: v/s2 for k,v in blended.items()}, (w1,w2,w3,w4)

def _hedge_update4(true_c:int, p1:Dict[int,float], p2:Dict[int,float], p3:Dict[int,float], p4:Dict[int,float]):
    w1, w2, w3, w4 = _get_expert_w()
    l = lambda p: 1.0 - p.get(true_c, 0.0)
    from math import exp
    w1n = w1 * exp(-HEDGE_ETA * (1.0 - l(p1)))
    w2n = w2 * exp(-HEDGE_ETA * (1.0 - l(p2)))
    w3n = w3 * exp(-HEDGE_ETA * (1.0 - l(p3)))
    w4n = w4 * exp(-HEDGE_ETA * (1.0 - l(p4)))
    S = (w1n + w2n + w3n + w4n) or 1e-9
    _set_expert_w(w1n/S, w2n/S, w3n/S, w4n/S)

# ========= Score/State helpers =========
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
        reset_score(); _set_last_reset_ymd(today)

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
    if reset: _set_loss_streak(0)
    else:     _set_loss_streak(_get_loss_streak() + 1)

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
        if len(cur_seen) >= 3: break
        if it not in cur_seen:
            cur_seen.append(it)
    seen_txt = "-".join(cur_seen[:3])
    with _tx() as con:
        con.execute("UPDATE pending SET seen=? WHERE id=?", (seen_txt, int(row["id"])))

def _stage_from_observed(suggested: int, obs: List[int]) -> Tuple[str, str]:
    if not obs: return ("LOSS", "G2")
    if len(obs) >= 1 and obs[0] == suggested: return ("GREEN", "G0")
    if len(obs) >= 2 and obs[1] == suggested: return ("GREEN", "G1")
    if len(obs) >= 3 and obs[2] == suggested: return ("GREEN", "G2")
    return ("LOSS", "G2")

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

def _close_with_outcome(row: sqlite3.Row, outcome: str, final_seen: str, stage_lbl: str, suggested: int):
    our_num_display = suggested if outcome.upper()=="GREEN" else "X"

    with _tx() as con:
        con.execute("UPDATE pending SET open=0, seen=? WHERE id=?", (final_seen, int(row["id"])))

    bump_score(outcome.upper())

    try:
        if outcome.upper() == "LOSS":
            _set_cooldown(COOLDOWN_N); _bump_loss_streak(reset=False)
        else:
            _dec_cooldown(); _bump_loss_streak(reset=True)
    except Exception:
        pass

    # Feedback e timeline
    try:
        ctxs = []
        for ncol in ("ctx1","ctx2","ctx3","ctx4"):
            v = (row[ncol] or "").strip()
            ctx = [int(x) for x in v.split(",") if x.strip().isdigit()]
            ctxs.append(ctx)
        ctx1, ctx2, ctx3, ctx4 = ctxs

        if outcome.upper() == "GREEN":
            stage_weight = {"G0": 1.00, "G1": 0.65, "G2": 0.40}.get(stage_lbl, 0.50)
            delta = FEED_POS * stage_weight
            for (n,ctx) in [(1,ctx1),(2,ctx2),(3,ctx3),(4,ctx4)]:
                if len(ctx)>=1:
                    _feedback_upsert(n, _ctx_to_key(ctx[:-1]), suggested, delta)
        else:
            true_first = None
            try:
                true_first = next(int(x) for x in final_seen.split("-") if x.isdigit())
            except StopIteration:
                pass
            for (n,ctx) in [(1,ctx1),(2,ctx2),(3,ctx3),(4,ctx4)]:
                if len(ctx)>=1:
                    _feedback_upsert(n, _ctx_to_key(ctx[:-1]), suggested, -1.5*FEED_NEG)
                    if true_first is not None:
                        _feedback_upsert(n, _ctx_to_key(ctx[:-1]), true_first, +1.2*FEED_POS)

        obs_add = [int(x) for x in final_seen.split("-") if x.isdigit()]
        append_seq(obs_add)
    except Exception:
        pass

    # Hedge update — 4 especialistas
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
            _hedge_update4(true_first, post_e1, post_e2, post_e3, post_e4)
    except Exception:
        pass

    snapshot = _ngram_snapshot_text(int(suggested))
    msg = (
        f"{'🟢' if outcome.upper()=='GREEN' else '🔴'} "
        f"<b>{outcome.upper()}</b> — finalizado "
        f"(<b>{stage_lbl}</b>, nosso={our_num_display}, observados={final_seen}).\n"
        f"📊 Geral: {score_text()}\n\n"
        f"{snapshot}\n\n"
        f"{_dealer_print_line()}"
    )
    return msg

def _maybe_close_by_timeout():
    row = get_open_pending()
    if not row: return None
    opened_at = int(row["opened_at"] or row["created_at"] or now_ts())
    if now_ts() - opened_at < OBS_TIMEOUT_SEC:
        return None
    seen_list = _seen_list(row)
    if len(seen_list) == 2:
        seen_list.append("X")
        final_seen = "-".join(seen_list[:3])
        suggested = int(row["suggested"] or 0)
        obs_nums = [int(x) for x in seen_list if x.isdigit()]
        outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
        return _close_with_outcome(row, outcome, final_seen, stage_lbl, suggested)
    return None

def close_pending(outcome:str):
    row = get_open_pending()
    if not row: return
    seen_list = _seen_list(row)
    while len(seen_list) < 3:
        seen_list.append("X")
    final_seen = "-".join(seen_list[:3])
    suggested = int(row["suggested"] or 0)
    obs_nums = [int(x) for x in seen_list if x.isdigit()]
    outcome2, stage_lbl = _stage_from_observed(suggested, obs_nums)
    return _close_with_outcome(row, outcome2, final_seen, stage_lbl, suggested)

def _open_pending_with_ctx(suggested:int, after:Optional[int], ctx1,ctx2,ctx3,ctx4) -> bool:
    with _tx() as con:
        row = con.execute("SELECT 1 FROM pending WHERE open=1 LIMIT 1").fetchone()
        if row: return False
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

# ========= Decisão principal =========
def _apply_seq3_boost(post: Dict[int,float], tail: List[int]) -> Tuple[Dict[int,float], str]:
    if not SEQ3_ENABLED:
        return post, ""
    if len(tail) < 4:
        return post, ""
    a,b,c = tail[-3:]
    support_counts = {1:0,2:0,3:0,4:0}
    support_total = 0
    for i in range(len(tail)-3):
        t0,t1,t2, nxt = tail[i], tail[i+1], tail[i+2], tail[i+3]
        if (t0,t1,t2) == (a,b,c):
            support_counts[nxt] += 1
            support_total += 1
    if support_total == 0:
        return post, ""
    best = max(support_counts, key=support_counts.get)
    conf = support_counts[best] / support_total
    if support_total >= SEQ3_MIN_SUPPORT and conf >= SEQ3_MIN_CONF:
        if SEQ3_FORCE:
            forced = {1:0.0,2:0.0,3:0.0,4:0.0}; forced[int(best)] = 1.0
            return forced, f"| SEQ3_FORCE s={support_total} c={conf:.2f}→{best}"
        boosted = {c_: (1.0-SEQ3_BOOST)*post.get(c_,0.0) for c_ in (1,2,3,4)}
        boosted[int(best)] += SEQ3_BOOST
        return _norm_dict(boosted), f"| SEQ3_BOOST s={support_total} c={conf:.2f}→{best}"
    return post, ""

def choose_single_number(after: Optional[int]):
    tail = get_tail(400)

    # Especialistas base
    post_e1 = _post_from_tail(tail, after)
    post_e2 = _post_freq_fib(tail) if FIB_FREQ_ENABLED else _post_freq_k(tail, K_SHORT)
    post_e3 = _post_freq_k(tail, K_LONG)
    post_e4 = _llm_probs_from_tail(tail) or {1:0.25,2:0.25,3:0.25,4:0.25}

    # Blend (Hedge original)
    post_base, (w1,w2,w3,w4) = _hedge_blend4(post_e1, post_e2, post_e3, post_e4)

    # SEQ3 (trio->próximo) como pós-mix opcional
    post_seq3, seq3_tag = _apply_seq3_boost(post_base, tail)

    # Gap de diagnóstico (pré-M0)
    ranking0 = sorted(post_seq3.items(), key=lambda kv: kv[1], reverse=True)
    top20 = ranking0[:2]
    gap0 = (top20[0][1] - top20[1][1]) if len(top20) >= 2 else ranking0[0][1]

    # ===== M0: IA Regente (E8/E9/E10/E11/E12) =====
    post_m0, m0_tag = _m0_blend_extended(post_seq3, tail, after, timeline_size())

    # Ranking/gap após M0
    ranking = sorted(post_m0.items(), key=lambda kv: kv[1], reverse=True)
    top2 = ranking[:2]
    gap = (top2[0][1] - top2[1][1]) if len(top2) >= 2 else ranking[0][1]
    base_best = ranking[0][0]
    conf = float(post_m0[base_best])

    # Anti-tilt final
    ls = _get_loss_streak()
    best, reason, post_adj = _streak_adjust_choice(post_m0, gap, ls)

    # Piso/cap de confiança
    post_final, floor_tag = _apply_conf_floor(post_adj, MIN_CONF_FLOOR, H_MAX)
    best = max(post_final, key=post_final.get)
    conf = float(post_final[best])

    r2 = sorted(post_final.items(), key=lambda kv: kv[1], reverse=True)[:2]
    gap2 = (r2[0][1] - r2[1][1]) if len(r2) == 2 else r2[0][1]

    # Motivo (tags)
    reason_tags = []
    if seq3_tag: reason_tags.append(seq3_tag)
    if m0_tag:  reason_tags.append(m0_tag)
    if floor_tag: reason_tags.append(floor_tag)
    reason_full = (" ".join(reason_tags)).strip()
    if reason_full:
        reason = f"{reason} {reason_full}".strip()

    return best, conf, timeline_size(), post_final, gap2, reason

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
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    check_and_maybe_reset_score()

    data = await request.json()

    # DEDUPE por update_id
    upd_id = str(data.get("update_id", "")) if isinstance(data, dict) else ""
    if _is_processed(upd_id):
        return {"ok": True, "skipped": "duplicate_update"}
    _mark_processed(upd_id)

    # timeout pode fechar pendência antiga (apenas se já houver 2 observados)
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

    # 0) ANALISANDO — aprender e ADIANTAR FECHAMENTO (sem abrir/fechar por si só)
    ANALISANDO_RX = re.compile(r"\bANALISANDO\b", re.I)
    if ANALISANDO_RX.search(("".join({"1️⃣":"1","2️⃣":"2","3️⃣":"3","4️⃣":"4"}.get(ch, ch) for ch in text))):
        def _normalize_keycaps(s: str) -> str:
            return "".join({"1️⃣":"1","2️⃣":"2","3️⃣":"3","4️⃣":"4"}.get(ch, ch) for ch in (s or ""))
        def parse_analise_seq(txt: str) -> List[int]:
            SEQ_RX = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
            if not ANALISANDO_RX.search(_normalize_keycaps(txt or "")): return []
            mseq = SEQ_RX.search(_normalize_keycaps(txt or ""))
            if not mseq: return []
            return [int(x) for x in re.findall(r"[1-4]", mseq.group(1))]
        seq = parse_analise_seq(text)
        if seq:
            append_seq(seq)
            pend = get_open_pending()
            if pend:
                to_add = []
                cur_seen = _seen_list(pend)
                need = 3 - len(cur_seen)
                if need > 0:
                    to_add = [str(n) for n in seq[:need]]
                    if to_add:
                        _seen_append(pend, to_add)
                        pend = get_open_pending()
                        cur_seen = _seen_list(pend)
                        if len(cur_seen) >= 3:
                            suggested = int(pend["suggested"] or 0)
                            obs_nums = [int(x) for x in cur_seen if x.isdigit()]
                            outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
                            final_seen = "-".join(cur_seen[:3])
                            out_msg = _close_with_outcome(pend, outcome, final_seen, stage_lbl, suggested)
                            await tg_send_text(TARGET_CHANNEL, out_msg)
                            return {"ok": True, "closed_from_analise": True, "seen": final_seen}
        return {"ok": True, "analise_seen": len(seq)}

    # Regexes gerais
    ENTRY_RX = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
    SEQ_RX   = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
    AFTER_RX = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)
    GALE1_RX = re.compile(r"Estamos\s+no\s*1[ºo]\s*gale", re.I)
    GALE2_RX = re.compile(r"Estamos\s+no\s*2[ºo]\s*gale", re.I)
    GREEN_RX = re.compile(r"(?:\bgr+e+e?n\b|\bwin\b|✅)", re.I)
    LOSS_RX  = re.compile(r"(?:\blo+s+s?\b|\bred\b|❌|\bperdemos\b)", re.I)
    PAREN_GROUP_RX = re.compile(r"\(([^)]*)\)")
    ANY_14_RX      = re.compile(r"[1-4]")
    KEYCAP_MAP = {"1️⃣":"1","2️⃣":"2","3️⃣":"3","4️⃣":"4"}
    def _normalize_keycaps(s: str) -> str:
        return "".join(KEYCAP_MAP.get(ch, ch) for ch in (s or ""))

    def parse_entry_text(txt: str) -> Optional[Dict]:
        t = _normalize_keycaps(re.sub(r"\s+", " ", txt).strip())
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

    def parse_close_numbers(txt: str) -> List[int]:
        t = _normalize_keycaps(re.sub(r"\s+", " ", txt))
        groups = PAREN_GROUP_RX.findall(t)
        if groups:
            last = groups[-1]
            nums = re.findall(r"[1-4]", _normalize_keycaps(last))
            return [int(x) for x in nums][:3]
        nums = ANY_14_RX.findall(t)
        return [int(x) for x in nums][:3]

    # 1) Gales (informativo)
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

    # 2) Fechamentos do fonte (GREEN/LOSS)
    if GREEN_RX.search(text) or LOSS_RX.search(text):
        pend = get_open_pending()
        if pend:
            nums = parse_close_numbers(text)
            if nums:
                _seen_append(pend, [str(n) for n in nums])
                pend = get_open_pending()
            seen_list = _seen_list(pend) if pend else []
            if pend and len(seen_list) >= 3:
                suggested = int(pend["suggested"] or 0)
                obs_nums = [int(x) for x in seen_list if x.isdigit()]
                outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
                final_seen = "-".join(seen_list[:3])
                out_msg = _close_with_outcome(pend, outcome, final_seen, stage_lbl, suggested)
                await tg_send_text(TARGET_CHANNEL, out_msg)
                return {"ok": True, "closed": outcome.lower(), "seen": final_seen}
        return {"ok": True, "noted_close": True}

    # 3) Nova ENTRADA CONFIRMADA (Fluxo estrito)
    parsed = parse_entry_text(text)
    if not parsed:
        if DEBUG_MSG:
            await tg_send_text(TARGET_CHANNEL, "DEBUG: Mensagem não reconhecida como ENTRADA/FECHAMENTO/ANALISANDO.")
        return {"ok": True, "skipped": "nao_eh_entrada_confirmada"}

    # se existir pendência e estiver incompleta, NÃO abre novo (sem avisos)
    pend = get_open_pending()
    if pend:
        seen_list = _seen_list(pend)
        if len(seen_list) >= 3:
            suggested = int(pend["suggested"] or 0)
            obs_nums = [int(x) for x in seen_list if x.isdigit()]
            outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
            final_seen = "-".join(seen_list[:3])
            out_msg = _close_with_outcome(pend, outcome, final_seen, stage_lbl, suggested)
            await tg_send_text(TARGET_CHANNEL, out_msg)
        else:
            return {"ok": True, "kept_open_waiting_close": True}

    # Alimenta memória com a sequência (se houver)
    seq = parsed["seq"] or []
    if seq: append_seq(seq)

    after = parsed["after"]
    best, conf, samples, post, gap, reason = choose_single_number(after)

    # Abertura (ALWAYS_ENTER=True)
    ctx1, ctx2, ctx3, ctx4 = _decision_context(after)
    opened = _open_pending_with_ctx(best, after, ctx1, ctx2, ctx3, ctx4)
    if not opened:
        if DEBUG_MSG:
            await tg_send_text(TARGET_CHANNEL, "DEBUG: Já existe pending open — não abri novo.")
        return {"ok": True, "skipped": "pending_already_open"}

    aft_txt = f" após {after}" if after else ""
    ls = _get_loss_streak()
    txt = (
        f"🤖 <b>IA SUGERE</b> — <b>{best}</b>\n"
        f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
        f"📊 <b>Conf:</b> {conf*100:.2f}% | <b>Amostra≈</b>{samples} | <b>gap≈</b>{gap*100:.1f}pp\n"
        f"🧠 <b>Modo:</b> {reason} | <b>streak RED:</b> {ls}\n"
        f"{_dealer_print_line()}"
    )
    await tg_send_text(TARGET_CHANNEL, txt)
    return {"ok": True, "posted": True, "best": best, "conf": conf, "gap": gap, "samples": samples}