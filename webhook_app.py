#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webhook_app.py — v4.7.0 (Gale Dinâmico por Sinal + Parser Estrito + Timeout Neutro + LLM status/debug)

- NEW: Gale dinâmico por SINAL (lê "Fazer até X gales" e usa required_obs=X+1)
- Fecha G0/G1/G2 conforme required_obs da pendência (1/2/3 observações)
- Parser estrito: captura 1, 2 ou 3 números no FINAL (com/sem parênteses)
  separadores aceitos: -, –, —, x, /, |, .
- Timeout neutro quando faltava apenas 1 observado (não marca LOSS)
- Endpoints /llm_status e /debug_cfg
- IA local (LLM) opcional como 4º especialista (mantido)
- Tokens/canais por ENV; demais configs fixas/seguras

ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN
ENV opcionais:    TARGET_CHANNEL, SOURCE_CHANNEL, DB_PATH (/var/data/data.db),
                  SHOW_DEBUG (0/1), OBS_TIMEOUT_SEC (420)
"""

import os, re, time, sqlite3, math, json
from contextlib import contextmanager
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone

# ====== Fuso horário ======
TZ_NAME = "America/Sao_Paulo"
try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo(TZ_NAME)
except Exception:
    _TZ = timezone.utc

import httpx
from fastapi import FastAPI, Request, HTTPException

# ========= OBRIGATÓRIAS (ENV) =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "").strip()  # ex: -100123...
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()  # se vazio, não filtra

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN e WEBHOOK_TOKEN no ambiente.")

TELEGRAM_API   = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# ========= CONFIG =========
DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"
SHOW_DEBUG     = os.getenv("SHOW_DEBUG", "0").lower() in ("1","true","yes")
OBS_TIMEOUT_SEC = int(os.getenv("OBS_TIMEOUT_SEC", "420"))

# ===== IA (LLM) =====
LLM_ENABLED    = False  # deixe True se tiver o modelo local
LLM_MODEL_PATH = "models/phi-3-mini.gguf"
LLM_CTX_TOKENS = 2048
LLM_N_THREADS  = 4
LLM_TEMP       = 0.10
LLM_TOP_P      = 0.90

# ===== Limites Globais =====
MAX_GALE_GLOBAL = 2  # limite teórico do sistema; o "required_obs" manda por sinal

# ===== Recursos estatísticos (mantidos) =====
DECAY = 0.980
W4, W3, W2, W1 = 0.42, 0.30, 0.18, 0.10
CONF_MIN    = 0.62
GAP_MIN     = 0.06
FREQ_WINDOW = 80
COOLDOWN_N     = 7
CD_CONF_BOOST  = 0.04
CD_GAP_BOOST   = 0.03
ALWAYS_ENTER = True
FEED_BETA   = 0.40
FEED_POS    = 0.85
FEED_NEG    = 1.20
FEED_DECAY  = 0.995
WF4, WF3, WF2, WF1 = W4, W3, W2, W1
HEDGE_ETA = 0.75
K_SHORT   = 60
K_LONG    = 300

MIN_CONF_FLOOR = 0.30
H_MAX          = 0.95

# ===== Freq/Anti-tilt/SEQ3 =====
FIB_FREQ_ENABLED     = True
FIB_ANTITILT_ENABLED = True
HEDGE_FIB_SEED       = True

SEQ3_ENABLED      = True
SEQ3_MIN_SUPPORT  = 12
SEQ3_MIN_CONF     = 0.78
SEQ3_BOOST        = 0.18
SEQ3_FORCE        = False

# ========= App =========
app = FastAPI(title="guardiao-auto-bot (GEN webhook)", version="4.7.0")

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

# ========= DB =========
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

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
        try:
            con.execute("ALTER TABLE expert_w ADD COLUMN w4 REAL NOT NULL DEFAULT 1.0")
        except sqlite3.OperationalError:
            pass
    # v4.7.0: coluna required_obs por pendência
    try:
        cur.execute("ALTER TABLE pending ADD COLUMN required_obs INTEGER")
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
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                time.sleep(0.25*(attempt+1)); continue
            raise

# --- dedupe ---
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
            nxt = int(tail[t]); dist = (len(tail)-1) - t
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
    row_tot = con.execute("SELECT SUM(w) AS s FROM ngram WHERE n=? AND ctx=?", (n, ctx_key)).fetchone()
    tot = (row_tot["s"] or 0.0) if row_tot else 0.0
    if tot <= 0:
        con.close(); return 0.0
    row_c = con.execute("SELECT w FROM ngram WHERE n=? AND ctx=? AND nxt=?",
                        (n, ctx_key, int(cand))).fetchone()
    w = (row_c["w"] or 0.0) if row_c else 0.0
    con.close()
    return w / max(tot, 1e-9)

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

# ========= IA local (opcional) =========
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
    feats = {"len": len(tail), "last10": tail[-10:], "freq60": {n: freq(win60, n) for n in (1,2,3,4)}, "freq300": {n: freq(win300, n) for n in (1,2,3,4)}}
    user = (f"Historico_Recente={tail[-50:]}\nFeats={feats}\nDevolva JSON com as probabilidades (%) para o PRÓXIMO número. Formato: {{\"1\":p1,\"2\":p2,\"3\":p3,\"4\":p4}}")
    try:
        out = llm.create_chat_completion(
            messages=[{"role":"system","content":_LLM_SYSTEM},{"role":"user","content":user}],
            temperature=LLM_TEMP, top_p=LLM_TOP_P, max_tokens=128
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

FIB_WINS = [21, 34, 55, 89]
_FIB_WIN_WEIGHTS = [1, 1, 1, 2]
_FIB_WIN_WEIGHTS = [w/sum(_FIB_WIN_WEIGHTS) for w in _FIB_WIN_WEIGHTS]

def _post_freq_k(tail: List[int], k: int) -> Dict[int,float]:
    if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
    win = tail[-k:] if len(tail) >= k else tail
    tot = max(1, len(win))
    return _norm_dict({c: win.count(c)/tot for c in [1,2,3,4]})

def _post_freq_fib(tail: List[int]) -> Dict[int, float]:
    if not tail: return {1:0.25, 2:0.25, 3:0.25, 4:0.25}
    acc = {1:0.0,2:0.0,3:0.0,4:0.0}
    for k, wk in zip(FIB_WINS, _FIB_WIN_WEIGHTS):
        pk = _post_freq_k(tail, k)
        for c in (1,2,3,4):
            acc[c] += wk * pk[c]
    tot = sum(acc.values()) or 1e-9
    return {c: acc[c]/tot for c in (1,2,3,4)}

# ========= E8–E12 + M0 =========
def _last_runlength(tail: List[int]) -> int:
    if not tail: return 0
    x = tail[-1]; r = 1
    for i in range(len(tail)-2, -1, -1):
        if tail[i] == x: r += 1
        else: break
    return r

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
    drift = kl > 0.22
    return {"kl": kl, "drift": drift, "dom_s": dom_s, "dom_l": dom_l, "ent_s": ent_s, "ent_l": ent_l}

def _dealer_print_line() -> str:
    snap = _dealer_snapshot()
    status = "DRIFT ALTO" if snap["drift"] else "OK"
    return (f"🧑‍💼 <b>Crupiê</b>: KL={snap['kl']:.3f} (thr 0.22) • {status} • "
            f"curto→{snap['dom_s']} | longo→{snap['dom_l']} • Hs={snap['ent_s']:.2f} Hl={snap['ent_l']:.2f} • α=0.0")

def _regime_prior() -> Dict[str, float]:
    snap = _dealer_snapshot()
    kl = float(snap.get("kl", 0.0))
    ent_s = float(snap.get("ent_s", 0.0))
    ent_l = float(snap.get("ent_l", 0.0))
    thr = 0.22
    if kl > thr and ent_s < 0.70:
        regime = "tendencioso"; w_mom = 0.65; w_ctr = 0.15; w_cnd = 0.20
    elif ent_s > 0.85 and ent_l > 0.85:
        regime = "caotico";     w_mom = 0.30; w_ctr = 0.40; w_cnd = 0.30
    else:
        regime = "estavel";     w_mom = 0.45; w_ctr = 0.20; w_cnd = 0.35
    return {"regime": regime, "w_momentum": w_mom, "w_contrarian": w_ctr, "w_conditional": w_cnd}

def _post_conditional(tail: List[int], after: Optional[int]) -> Dict[int, float]:
    if len(tail) < 6: return {1:0.25, 2:0.25, 3:0.25, 4:0.25}
    last = tail[-1]
    rl = _last_runlength(tail)
    rl_bucket = 1 if rl == 1 else (2 if rl == 2 else 3)
    counts = {1:1e-9, 2:1e-9, 3:1e-9, 4:1e-9}; total = 0
    for i in range(3, len(tail)-1):
        a,b,c = tail[i-3], tail[i-2], tail[i-1]
        nxt = tail[i]
        _last = c
        _rl = 3 if a==b==c else (2 if b==c else 1)
        cond_ok = (_last == last and _rl == rl_bucket)
        if after is not None: cond_ok = cond_ok and (_last == after)
        if cond_ok:
            counts[nxt] = counts.get(nxt, 0.0) + 1.0; total += 1
    if total == 0:
        for i in range(len(tail)-1):
            if tail[i] == last:
                counts[tail[i+1]] = counts.get( tail[i+1], 0.0) + 1.0; total += 1
        if total == 0: return {1:0.25,2:0.25,3:0.25,4:0.25}
    return _norm_dict(counts)

def _dirichlet_smooth(post: Dict[int, float], samples: int) -> Dict[int, float]:
    post = _norm_dict(post)
    alpha = max(0.2, min(1.2, 1.0 - 0.0005 * float(samples)))
    eff = {c: post[c] * max(1.0, float(samples)) for c in (1,2,3,4)}
    smooth = {c: (eff[c] + alpha) for c in (1,2,3,4)}
    tot = sum(smooth.values()) or 1e-9
    return {c: smooth[c]/tot for c in (1,2,3,4)}

def _post_contrarian(post_base: Dict[int, float]) -> Dict[int, float]:
    if not post_base: return {1:0.25,2:0.25,3:0.25,4:0.25}
    inv = {c: max(1e-9, 1.0 - float(post_base.get(c,0.0))) for c in (1,2,3,4)}
    return _norm_dict(inv)

def _post_momentum_burst(tail: List[int]) -> Dict[int, float]:
    if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
    wins = [8, 13, 21]; weights = [0.2, 0.3, 0.5]
    acc = {1:0.0,2:0.0,3:0.0,4:0.0}
    for k, wk in zip(wins, weights):
        pk = _post_freq_k(tail, k)
        for c in (1,2,3,4): acc[c] += wk * pk[c]
    return _norm_dict(acc)

def _m0_blend_extended(base_post: Dict[int,float], tail: List[int], after: Optional[int], samples: int) -> Tuple[Dict[int,float], str]:
    pri = _regime_prior()
    p_cnd = _post_conditional(tail, after)
    p_mom = _post_momentum_burst(tail)
    p_ctr = _post_contrarian(base_post)
    w_mom = pri["w_momentum"]; w_ctr = pri["w_contrarian"]; w_cnd = pri["w_conditional"]
    w_base = max(0.0, 1.0 - (w_mom + w_ctr + w_cnd))
    mix = {c: w_base*base_post.get(c,0.0) + w_cnd*p_cnd.get(c,0.0) + w_mom*p_mom.get(c,0.0) + w_ctr*p_ctr.get(c,0.0) for c in (1,2,3,4)}
    mix = _norm_dict(mix)
    smooth = _dirichlet_smooth(mix, samples)
    tag = f"| M0:{pri['regime']} w0={w_base:.2f} wCnd={w_cnd:.2f} wMom={w_mom:.2f} wCtr={w_ctr:.2f}"
    return smooth, tag

# ========= Hedge =========
def _get_expert_w():
    con = _connect()
    row = con.execute("SELECT w1,w2,w3,w4 FROM expert_w WHERE id=1").fetchone()
    con.close()
    if not row: return (1.0,1.0,1.0,1.0)
    return float(row["w1"]), float(row["w2"]), float(row["w3"]), float(row["w4"])

def _set_expert_w(w1, w2, w3, w4):
    with _tx() as con:
        con.execute("UPDATE expert_w SET w1=?, w2=?, w3=?, w4=? WHERE id=1",
                    (float(w1), float(w2), float(w3), float(w4)))

def _seed_expert_w_fib():
    try:
        w1, w2, w3, w4 = _get_expert_w()
    except Exception:
        return
    if all(abs(x-1.0)<1e-6 for x in (w1,w2,w3,w4)):
        base = [1,1,2,3]; s=sum(base)
        _set_expert_w(*(x/s for x in base))
if HEDGE_FIB_SEED:
    try: _seed_expert_w_fib()
    except Exception: pass

def _hedge_blend4(p1:Dict[int,float], p2:Dict[int,float], p3:Dict[int,float], p4:Dict[int,float]):
    w1, w2, w3, w4 = _get_expert_w()
    S = (w1+w2+w3+w4) or 1e-9
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
    w3n = w3n = w3 * exp(-HEDGE_ETA * (1.0 - l(p3)))
    w4n = w4 * exp(-HEDGE_ETA * (1.0 - l(p4)))
    S = (w1n+w2n+w3n+w4n) or 1e-9
    _set_expert_w(w1n/S, w2n/S, w3n/S, w4n/S)

# ========= Score/State =========
def bump_score(outcome: str) -> Tuple[int, int]:
    with _tx() as con:
        row = con.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
        g, l = (row["green"], row["loss"]) if row else (0, 0)
        if outcome.upper() == "GREEN": g += 1
        elif outcome.upper() == "LOSS": l += 1
        con.execute("INSERT OR REPLACE INTO score (id, green, loss) VALUES (1,?,?)", (g, l))
        return g, l

def reset_score():
    with _tx() as con:
        con.execute("INSERT OR REPLACE INTO score (id, green, loss) VALUES (1,0,0)")

def score_text() -> str:
    con = _connect()
    row = con.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
    con.close()
    if not row: return "0 GREEN × 0 LOSS — 0.0%"
    g, l = int(row["green"]), int(row["loss"]); total = g + l
    acc = (g/total*100.0) if total > 0 else 0.0
    return f"{g} GREEN × {l} LOSS — {acc:.1f}%"

def _get_last_reset_ymd() -> str:
    con = _connect(); row = con.execute("SELECT last_reset_ymd FROM state WHERE id=1").fetchone(); con.close()
    return (row["last_reset_ymd"] or "") if row else ""
def _set_last_reset_ymd(ymd: str):
    with _tx() as con: con.execute("UPDATE state SET last_reset_ymd=? WHERE id=1", (ymd,))
def check_and_maybe_reset_score():
    today = tz_today_ymd()
    if _get_last_reset_ymd() != today:
        reset_score(); _set_last_reset_ymd(today)
def _get_cooldown() -> int:
    con = _connect(); row = con.execute("SELECT cooldown_left FROM state WHERE id=1").fetchone(); con.close()
    return int((row["cooldown_left"] if row else 0) or 0)
def _set_cooldown(v:int):
    with _tx() as con: con.execute("UPDATE state SET cooldown_left=? WHERE id=1", (int(v),))
def _dec_cooldown():
    with _tx() as con:
        row = con.execute("SELECT cooldown_left FROM state WHERE id=1").fetchone()
        cur = int((row["cooldown_left"] if row else 0) or 0)
        con.execute("UPDATE state SET cooldown_left=? WHERE id=1", (max(0, cur-1),))
def _get_loss_streak() -> int:
    con = _connect(); row = con.execute("SELECT loss_streak FROM state WHERE id=1").fetchone(); con.close()
    return int((row["loss_streak"] if row else 0) or 0)
def _set_loss_streak(v:int):
    with _tx() as con: con.execute("UPDATE state SET loss_streak=? WHERE id=1", (int(v),))
def _bump_loss_streak(reset: bool):
    if reset: _set_loss_streak(0)
    else:     _set_loss_streak(_get_loss_streak() + 1)

# ========= Pending (com required_obs) =========
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

def _pending_required_obs(row: sqlite3.Row) -> int:
    try:
        v = int(row["required_obs"] or 0)
        return v if 1 <= v <= 3 else 2  # fallback seguro: 2 (G1)
    except Exception:
        return 2

def _seen_append(row: sqlite3.Row, new_items: List[str]):
    cur_seen = _seen_list(row)
    need = _pending_required_obs(row)
    for it in new_items:
        if len(cur_seen) >= need:
            break
        if it not in cur_seen:
            cur_seen.append(it)
    seen_txt = "-".join(cur_seen[:need])
    with _tx() as con:
        con.execute("UPDATE pending SET seen=? WHERE id=?", (seen_txt, int(row["id"])))

def _seen_is_complete(row: sqlite3.Row) -> bool:
    return len(_seen_list(row)) >= _pending_required_obs(row)

def _stage_from_observed(suggested: int, obs: List[int], required_obs:int) -> Tuple[str, str]:
    """required_obs=1->G0, 2->G1, 3->G2"""
    stages = ["G0", "G1", "G2"]
    lim = min(MAX_GALE_GLOBAL, max(0, min(2, required_obs-1)))
    for i in range(0, lim + 1):
        if len(obs) >= i + 1 and obs[i] == suggested:
            return ("GREEN", stages[i])
    return ("LOSS", stages[lim])

def _ngram_snapshot_text(suggested: int) -> str:
    tail = get_tail(400)
    post = _post_from_tail(tail, after=None)
    pct = lambda x: f"{x*100:.1f}%"
    p1 = pct(post.get(1, 0.0)); p2 = pct(post.get(2, 0.0))
    p3 = pct(post.get(3, 0.0)); p4 = pct(post.get(4, 0.0))
    conf = pct(post.get(int(suggested), 0.0))
    amostra = timeline_size()
    return f"📈 Amostra: {amostra} • Conf: {conf}\n\n🔎 E1(n-gram proxy): 1 {p1} | 2 {p2} | 3 {p3} | 4 {p4}"

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
    # feedback + append timeline
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
                if len(ctx)>=1: _feedback_upsert(n, _ctx_to_key(ctx[:-1]), suggested, delta)
        else:
            true_first = None
            try: true_first = next(int(x) for x in final_seen.split("-") if x.isdigit())
            except StopIteration: pass
            for (n,ctx) in [(1,ctx1),(2,ctx2),(3,ctx3),(4,ctx4)]:
                if len(ctx)>=1:
                    _feedback_upsert(n, _ctx_to_key(ctx[:-1]), suggested, -1.5*FEED_NEG)
                    if true_first is not None:
                        _feedback_upsert(n, _ctx_to_key(ctx[:-1]), true_first, +1.2*FEED_POS)
        obs_add = [int(x) for x in final_seen.split("-") if x.isdigit()]
        append_seq(obs_add)
    except Exception:
        pass
    # hedge update
    try:
        true_first = None
        try: true_first = next(int(x) for x in final_seen.split("-") if x.isdigit())
        except StopIteration: pass
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
    msg = (f"{'🟢' if outcome.upper()=='GREEN' else '🔴'} <b>{outcome.upper()}</b> — finalizado "
           f"(<b>{stage_lbl}</b>, nosso={our_num_display}, observados={final_seen}).\n"
           f"📊 Geral: {score_text()}\n\n{snapshot}\n\n{_dealer_print_line()}")
    return msg

def _maybe_close_by_timeout():
    row = get_open_pending()
    if not row: 
        return None
    opened_at = int(row["opened_at"] or row["created_at"] or now_ts())
    if now_ts() - opened_at < OBS_TIMEOUT_SEC:
        return None
    need = _pending_required_obs(row)
    seen_list = _seen_list(row)
    # Timeout NEUTRO quando faltava apenas 1 observado
    if len(seen_list) == need - 1:
        final_seen = "-".join(seen_list + ["X"])
        with _tx() as con:
            con.execute("UPDATE pending SET open=0, seen=? WHERE id=?", (final_seen, int(row["id"])))
        if SHOW_DEBUG:
            try:
                import asyncio as _asyncio
                _asyncio.create_task(tg_send_text(TARGET_CHANNEL, f"⏳ Timeout neutro (faltava 1). seen={final_seen}"))
            except Exception:
                pass
        return None
    return None

def close_pending(outcome:str):
    row = get_open_pending()
    if not row: return
    need = _pending_required_obs(row)
    seen_list = _seen_list(row)
    while len(seen_list) < need:
        seen_list.append("X")
    final_seen = "-".join(seen_list[:need])
    suggested = int(row["suggested"] or 0)
    obs_nums = [int(x) for x in seen_list if x.isdigit()]
    outcome2, stage_lbl = _stage_from_observed(suggested, obs_nums, need)
    return _close_with_outcome(row, outcome2, final_seen, stage_lbl, suggested)

def _open_pending_with_ctx(suggested:int, after:Optional[int], ctx1,ctx2,ctx3,ctx4, required_obs:int) -> bool:
    need = max(1, min(3, int(required_obs or 2)))  # fallback 2 (G1)
    with _tx() as con:
        row = con.execute("SELECT 1 FROM pending WHERE open=1 LIMIT 1").fetchone()
        if row: 
            return False
        con.execute("""
            INSERT INTO pending (created_at, suggested, stage, open, seen, opened_at, after, 
                                 ctx1, ctx2, ctx3, ctx4, wait_notice_sent, required_obs)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (now_ts(), int(suggested), 0, 1, "", now_ts(), after,
         _ctx_to_key(ctx1), _ctx_to_key(ctx2), _ctx_to_key(ctx3), _ctx_to_key(ctx4),
         0, need))
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

# ========= Decisão =========
def _apply_conf_floor(post: Dict[int,float], floor: float = MIN_CONF_FLOOR, cap: float = H_MAX) -> Tuple[Dict[int,float], str]:
    if not post: return post, ""
    post = _norm_dict({c: float(post.get(c,0.0)) for c in (1,2,3,4)})
    best = max(post, key=post.get); mx = post[best]; tag = ""
    if mx < floor:
        others = [c for c in (1,2,3,4) if c != best]
        sum_others = sum(post[c] for c in others)
        need = floor - mx; take = min(need, sum_others)
        if sum_others > 0:
            scale = (sum_others - take) / sum_others
            for c in others: post[c] *= scale
        post[best] = min(cap, mx + take)
        post = _norm_dict(post); tag = f"| conf_floor→{post[best]*100:.0f}%"
    if post[best] > cap:
        excess = post[best] - cap; others = [c for c in (1,2,3,4) if c != best]
        add = excess / len(others)
        post[best] = cap
        for c in others: post[c] += add
        post = _norm_dict(post); tag += f"| cap{int(cap*100)}"
    return post, tag

def _fib(n: int) -> int:
    if n <= 1: return 1
    a, b = 1, 1
    for _ in range(n-1): a, b = b, a + b
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

def _apply_seq3_boost(post: Dict[int,float], tail: List[int]) -> Tuple[Dict[int,float], str]:
    if not SEQ3_ENABLED or len(tail) < 4:
        return post, ""
    a,b,c = tail[-3:]
    support_counts = {1:0,2:0,3:0,4:0}; support_total = 0
    for i in range(len(tail)-3):
        t0,t1,t2, nxt = tail[i], tail[i+1], tail[i+2], tail[i+3]
        if (t0,t1,t2) == (a,b,c):
            support_counts[nxt] += 1; support_total += 1
    if support_total == 0: return post, ""
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

def choose_single_number(after: Optional[int]):
    tail = get_tail(400)
    post_e1 = _post_from_tail(tail, after)
    post_e2 = _post_freq_fib(tail) if FIB_FREQ_ENABLED else _post_freq_k(tail, K_SHORT)
    post_e3 = _post_freq_k(tail, K_LONG)
    post_e4 = _llm_probs_from_tail(tail) or {1:0.25,2:0.25,3:0.25,4:0.25}
    post_base, (w1,w2,w3,w4) = _hedge_blend4(post_e1, post_e2, post_e3, post_e4)
    post_seq3, seq3_tag = _apply_seq3_boost(post_base, tail)
    post_m0, m0_tag = _m0_blend_extended(post_seq3, tail, after, timeline_size())
    ranking = sorted(post_m0.items(), key=lambda kv: kv[1], reverse=True)
    top2 = ranking[:2]
    gap = (top2[0][1] - top2[1][1]) if len(top2) >= 2 else ranking[0][1]
    base_best = ranking[0][0]
    conf = float(post_m0[base_best])
    ls = _get_loss_streak()
    best, reason, post_adj = _streak_adjust_choice(post_m0, gap, ls)
    post_final, floor_tag = _apply_conf_floor(post_adj, MIN_CONF_FLOOR, H_MAX)
    best = max(post_final, key=post_final.get); conf = float(post_final[best])
    r2 = sorted(post_final.items(), key=lambda kv: kv[1], reverse=True)[:2]
    gap2 = (r2[0][1] - r2[1][1]) if len(r2) == 2 else r2[0][1]
    reason_tags = [tag for tag in (seq3_tag, m0_tag, floor_tag) if tag]
    if reason_tags: reason = f"{reason} {' '.join(reason_tags)}".strip()
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
    return {"ok": True, "db": DB_PATH, "pending_open": bool(pend), "pending_seen": (pend["seen"] if pend else ""), "time": ts_str(), "tz": TZ_NAME}

@app.get("/llm_status")
async def llm_status():
    llm = _llm_load()
    exists = os.path.exists(LLM_MODEL_PATH)
    return {"enabled": bool(LLM_ENABLED), "model_path": LLM_MODEL_PATH, "model_exists": bool(exists), "loaded": bool(llm is not None), "ctx_tokens": LLM_CTX_TOKENS, "threads": LLM_N_THREADS}

@app.get("/debug_cfg")
async def debug_cfg():
    return {"MAX_GALE_GLOBAL": MAX_GALE_GLOBAL, "OBS_TIMEOUT_SEC": OBS_TIMEOUT_SEC, "DB_PATH": DB_PATH}

# ========= Parser e Webhook =========
GRUPO_FINAL_PAREN_RX = re.compile(r"\(([^\)]*)\)\s*$")
SEP = r"[-–—xX/|\.]"
PAR_SOLTO_FIM_RX    = re.compile(rf"\b([1-4])\s*{SEP}\s*([1-4])\s*$")
TRINCA_SOLTA_FIM_RX = re.compile(rf"\b([1-4])\s*{SEP}\s*([1-4])\s*{SEP}\s*([1-4])\s*$")
PAR_NUMS_RX    = re.compile(rf"\b([1-4])\s*{SEP}\s*([1-4])\b")
TRINCA_NUMS_RX = re.compile(rf"\b([1-4])\s*{SEP}\s*([1-4])\s*{SEP}\s*([1-4])\b")
NUM_SOLTO_FIM_RX = re.compile(r"\b([1-4])\s*$")

ENTRY_RX = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
SEQ_RX   = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
AFTER_RX = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)
GALE1_RX = re.compile(r"Estamos\s+no\s*1[ºo]\s*gale", re.I)
GALE2_RX = re.compile(r"Estamos\s+no\s*2[ºo]\s*gale", re.I)
GREEN_RX = re.compile(r"(?:\bgr+e+e?n\b|\bwin\b|✅)", re.I)
LOSS_RX  = re.compile(r"(?:\blo+s+s?\b|\bred\b|❌|\bperdemos\b)", re.I)
GALES_RX = re.compile(r"Fazer\s+até\s+(\d+)\s+gales?", re.I)
ANALISANDO_RX = re.compile(r"\bANALISANDO\b", re.I)

KEYCAP_MAP = {"1️⃣":"1","2️⃣":"2","3️⃣":"3","4️⃣":"4"}
def _normalize_keycaps(s: str) -> str:
    return "".join(KEYCAP_MAP.get(ch, ch) for ch in (s or ""))

def parse_obs_estrito(txt: str, need: int) -> List[int]:
    t = _normalize_keycaps(re.sub(r"\s+", " ", txt or ""))
    m = GRUPO_FINAL_PAREN_RX.search(t)
    if m:
        seg = m.group(1)
        if need == 1:
            m1 = re.search(r"\b([1-4])\b", seg)
            if m1: return [int(m1.group(1))]
        if need == 2:
            mp = PAR_NUMS_RX.search(seg)
            if mp: return [int(mp.group(1)), int(mp.group(2))]
        if need == 3:
            m3 = TRINCA_NUMS_RX.search(seg)
            if m3: return [int(m3.group(1)), int(m3.group(2)), int(m3.group(3))]
    if need == 1:
        ms1 = NUM_SOLTO_FIM_RX.search(t)
        if ms1: return [int(ms1.group(1))]
    if need == 2:
        ms = PAR_SOLTO_FIM_RX.search(t)
        if ms: return [int(ms.group(1)), int(ms.group(2))]
    if need == 3:
        ms3 = TRINCA_SOLTA_FIM_RX.search(t)
        if ms3: return [int(ms3.group(1)), int(ms3.group(2)), int(ms3.group(3))]
    return []

def parse_entry_text(txt: str) -> Optional[Dict]:
    t = _normalize_keycaps(re.sub(r"\s+", " ", txt).strip())
    if not ENTRY_RX.search(t): 
        return None
    mseq = SEQ_RX.search(t); seq = []
    if mseq:
        parts = re.findall(r"[1-4]", _normalize_keycaps(mseq.group(1)))
        seq = [int(x) for x in parts]
    mafter = AFTER_RX.search(t)
    after_num = int(mafter.group(1)) if mafter else None
    mgale = GALES_RX.search(t)
    required_obs = None
    if mgale:
        try:
            g = max(0, min(2, int(mgale.group(1))))
            required_obs = g + 1
        except Exception:
            required_obs = None
    return {"seq": seq, "after": after_num, "required_obs": required_obs, "raw": t}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    check_and_maybe_reset_score()

    data = await request.json()
    upd_id = str(data.get("update_id", "")) if isinstance(data, dict) else ""
    if _is_processed(upd_id):
        return {"ok": True, "skipped": "duplicate_update"}
    _mark_processed(upd_id)

    _ = _maybe_close_by_timeout()

    msg = data.get("channel_post") or data.get("message") \
        or data.get("edited_channel_post") or data.get("edited_message") or {}
    text = (msg.get("text") or msg.get("caption") or "").strip()
    chat = msg.get("chat") or {}
    chat_id = str(chat.get("id") or "")

    # Filtro de origem
    if SOURCE_CHANNEL and chat_id != str(SOURCE_CHANNEL):
        if SHOW_DEBUG:
            await tg_send_text(TARGET_CHANNEL, f"DEBUG: Ignorando chat {chat_id}. Fonte esperada: {SOURCE_CHANNEL}")
        return {"ok": True, "skipped": "outro_chat"}
    if not text:
        return {"ok": True, "skipped": "sem_texto"}

    # === Fechamento por observações (antes de tags GREEN/LOSS) ===
    pend = get_open_pending()
    if pend:
        need = _pending_required_obs(pend)
        nums = parse_obs_estrito(text, need)
        if nums:
            _seen_append(pend, [str(n) for n in nums])
            pend = get_open_pending()
            if pend and _seen_is_complete(pend):
                need = _pending_required_obs(pend)
                suggested = int(pend["suggested"] or 0)
                obs_nums = [int(x) for x in _seen_list(pend) if x.isdigit()]
                outcome, stage_lbl = _stage_from_observed(suggested, obs_nums, need)
                final_seen = "-".join(_seen_list(pend)[:need])
                out_msg = _close_with_outcome(pend, outcome, final_seen, stage_lbl, suggested)
                await tg_send_text(TARGET_CHANNEL, out_msg)
                return {"ok": True, "closed_by_obs_strict": True, "seen": final_seen, "outcome": outcome}

    # --- Tags gerais ---
    if ANALISANDO_RX.search(_normalize_keycaps(text)):
        # Alimenta timeline se tiver "Sequência: ..."
        mseq = SEQ_RX.search(_normalize_keycaps(text))
        seq = []
        if mseq:
            seq = [int(x) for x in re.findall(r"[1-4]", mseq.group(1))]
        if seq:
            append_seq(seq)
            pend = get_open_pending()
            if pend and not _seen_is_complete(pend):
                need = _pending_required_obs(pend)
                cur_seen = _seen_list(pend)
                falta = max(0, need - len(cur_seen))
                if falta > 0:
                    _seen_append(pend, [str(n) for n in seq[:falta]])
                    pend = get_open_pending()
                    if pend and _seen_is_complete(pend):
                        need = _pending_required_obs(pend)
                        suggested = int(pend["suggested"] or 0)
                        obs_nums = [int(x) for x in _seen_list(pend) if x.isdigit()]
                        outcome, stage_lbl = _stage_from_observed(suggested, obs_nums, need)
                        final_seen = "-".join(_seen_list(pend)[:need])
                        out_msg = _close_with_outcome(pend, outcome, final_seen, stage_lbl, suggested)
                        await tg_send_text(TARGET_CHANNEL, out_msg)
                        return {"ok": True, "closed_from_analise": True, "seen": final_seen}
        return {"ok": True, "analise_seen": len(seq)}

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

    if GREEN_RX.search(text) or LOSS_RX.search(text):
        pend = get_open_pending()
        if pend:
            need = _pending_required_obs(pend)
            nums = parse_obs_estrito(text, need)
            if nums:
                _seen_append(pend, [str(n) for n in nums])
                pend = get_open_pending()
            if pend and _seen_is_complete(pend):
                need = _pending_required_obs(pend)
                suggested = int(pend["suggested"] or 0)
                obs_nums = [int(x) for x in _seen_list(pend) if x.isdigit()]
                outcome, stage_lbl = _stage_from_observed(suggested, obs_nums, need)
                final_seen = "-".join(_seen_list(pend)[:need])
                out_msg = _close_with_outcome(pend, outcome, final_seen, stage_lbl, suggested)
                await tg_send_text(TARGET_CHANNEL, out_msg)
                return {"ok": True, "closed": outcome.lower(), "seen": final_seen}
        return {"ok": True, "noted_close": True}

    # === Nova ENTRADA CONFIRMADA ===
    parsed = parse_entry_text(text)
    if not parsed:
        if SHOW_DEBUG:
            await tg_send_text(TARGET_CHANNEL, "DEBUG: Mensagem não reconhecida como ENTRADA/FECHAMENTO/ANALISANDO.")
        return {"ok": True, "skipped": "nao_eh_entrada_confirmada"}

    # Se existir pendência aberta e completa, fecha; senão mantém aberta
    pend = get_open_pending()
    if pend:
        if _seen_is_complete(pend):
            need = _pending_required_obs(pend)
            suggested = int(pend["suggested"] or 0)
            obs_nums = [int(x) for x in _seen_list(pend) if x.isdigit()]
            outcome, stage_lbl = _stage_from_observed(suggested, obs_nums, need)
            final_seen = "-".join(_seen_list(pend)[:need])
            out_msg = _close_with_outcome(pend, outcome, final_seen, stage_lbl, suggested)
            await tg_send_text(TARGET_CHANNEL, out_msg)
        else:
            return {"ok": True, "kept_open_waiting_close": True}

    # Alimenta memória com a sequência (se houver)
    seq = parsed["seq"] or []
    if seq: append_seq(seq)

    # Decide
    after = parsed["after"]
    best, conf, samples, post, gap, reason = choose_single_number(after)

    # required_obs dinâmico
    required_obs = parsed.get("required_obs") or 2  # fallback G1 se não houver a linha na mensagem

    # Abertura
    ctx1, ctx2, ctx3, ctx4 = _decision_context(after:=after if after is not None else None)
    opened = _open_pending_with_ctx(best, after, ctx1, ctx2, ctx3, ctx4, required_obs)
    if not opened:
        if SHOW_DEBUG:
            await tg_send_text(TARGET_CHANNEL, "DEBUG: Já existe pending open — não abri novo.")
        return {"ok": True, "skipped": "pending_already_open"}

    aft_txt = f" após {after}" if after else ""
    ls = _get_loss_streak()
    txt = (
        f"🤖 <b>IA SUGERE</b> — <b>{best}</b>\n"
        f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
        f"📊 <b>Conf:</b> {conf*100:.2f}% | <b>Amostra≈</b>{samples} | <b>gap≈</b>{gap*100:.1f}pp | <b>need:</b>{required_obs}\n"
        f"🧠 <b>Modo:</b> {reason} | <b>streak RED:</b> {ls}\n"
        f"{_dealer_print_line()}"
    )
    await tg_send_text(TARGET_CHANNEL, txt)
    return {"ok": True, "posted": True, "best": best, "conf": conf, "gap": gap, "samples": samples}