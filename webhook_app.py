#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
guardiao-auto-bot (GEN híbrido + estratégia + timeout 45s + relatório 5min + ML por padrão)
Arquivo consolidado com:
- thresholds endurecidos e filtros de qualidade
- cooldown de proteção por sequência/dia ruim
- ensemble com suporte a modelos por padrão (GEN/KWOK/SSH/SEQ/ODD/EVEN) via registry.json
"""

import os, re, time, json, sqlite3, asyncio, pickle
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, Request, HTTPException

# novo: joblib para carregar modelos
import joblib

# ========= ENV =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1002796105884").strip()  # destino
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "-1002810508717").strip()  # fonte
DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"

# ======== Reset diário (timezone) ========
RESET_TZ       = os.getenv("RESET_TZ", "America/Sao_Paulo").strip()

# ========= ML (opcional) =========
MODEL_PATH    = os.getenv("MODEL_PATH", "/var/data/model.pkl").strip()  # legado/global
ML_MIN_PROBA  = float(os.getenv("ML_MIN_PROBA", "0.78"))   # limiar mínimo p/ aceitar previsão
ML_GAP_MIN    = float(os.getenv("ML_GAP_MIN", "0.08"))     # gap vs segundo melhor
ENSEMBLE_MODE = os.getenv("ENSEMBLE_MODE", "blend").strip() # "gate" | "blend" | "ml_only"

# modelos por padrão
MODEL_DIR = os.getenv("MODEL_DIR", "/var/data").strip() or "/var/data"
_MODELS_BY_PATTERN: Dict[str, object] = {}
_REGISTRY: Optional[Dict] = None

if not TG_BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")
if not WEBHOOK_TOKEN:
    raise RuntimeError("Defina WEBHOOK_TOKEN no ambiente.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
app = FastAPI(title="guardiao-auto-bot (GEN híbrido + estratégia + ML por padrão)", version="3.3.0-ml-pattern")

# ========= HÍBRIDO (curta/longa) =========
SHORT_WINDOW    = 40
LONG_WINDOW     = 600

# thresholds mais fortes (qualidade > quantidade)
CONF_SHORT_MIN  = 0.56      # antes 0.49
CONF_LONG_MIN   = 0.64      # antes 0.59
GAP_MIN         = 0.05      # antes 0.025

# ========= Filtros de qualidade =========
MIN_SUPPORT_SHORT = 12      # amostras mínimas na janela curta
MIN_SUPPORT_LONG  = 60      # amostras mínimas na janela longa
MIN_TOTAL_CTX     = 180     # mínimo de registros totais na timeline (segurança)

# thresholds por padrão (opcionais, sobrepõem os gerais)
PATTERN_THRESHOLDS = {
    "KWOK": {"conf_s": 0.58, "conf_l": 0.66, "gap": 0.06},
    "SSH":  {"conf_s": 0.56, "conf_l": 0.64, "gap": 0.05},
    "SEQ":  {"conf_s": 0.58, "conf_l": 0.66, "gap": 0.05},
    "ODD":  {"conf_s": 0.60, "conf_l": 0.68, "gap": 0.06},
    "EVEN": {"conf_s": 0.60, "conf_l": 0.68, "gap": 0.06},
    "GEN":  {"conf_s": 0.62, "conf_l": 0.70, "gap": 0.07},
}

GAP_MIN         = float(os.getenv("GAP_MIN", str(GAP_MIN)))
CONF_SHORT_MIN  = float(os.getenv("CONF_SHORT_MIN", str(CONF_SHORT_MIN)))
CONF_LONG_MIN   = float(os.getenv("CONF_LONG_MIN", str(CONF_LONG_MIN)))

FINAL_TIMEOUT   = int(os.getenv("FINAL_TIMEOUT", "45"))  # começa quando houver 2 observados

# ========= Guard rails de risco =========
MAX_LOSS_STREAK   = int(os.getenv("MAX_LOSS_STREAK", "3"))        # pausa após 3 reds seguidos
COOLDOWN_SECONDS  = int(os.getenv("COOLDOWN_SECONDS", str(30*60)))# 30 min parado
MIN_DAY_ACC_GO    = float(os.getenv("MIN_DAY_ACC_GO", "0.55"))    # se o dia < 55% pausa
_last_cooldown_until = 0

# ========= Relatório / Sinais do dia =========
REPORT_EVERY_SEC   = 5 * 60
GOOD_DAY_THRESHOLD = 0.70
BAD_DAY_THRESHOLD  = 0.40

# ========= Utils =========
def now_ts() -> int: return int(time.time())
def ts_str(ts=None) -> str:
    if ts is None: ts = now_ts()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ========= DB =========
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

def _exec_write(sql: str, params: tuple=()):
    for attempt in range(6):
        try:
            con = _connect(); cur = con.cursor()
            cur.execute(sql, params)
            con.commit(); con.close(); return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                time.sleep(0.25*(attempt+1)); continue
            raise

def _query_all(sql: str, params: tuple=()) -> List[sqlite3.Row]:
    con = _connect(); cur = con.cursor()
    rows = cur.execute(sql, params).fetchall()
    con.close()
    return rows

def migrate_db():
    con = _connect(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pending (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER,
        suggested INTEGER,
        open INTEGER DEFAULT 1,
        seen TEXT,
        opened_at INTEGER,
        last_post_short TEXT,
        last_post_long  TEXT,
        last_conf_short REAL,
        last_conf_long  REAL,
        d_final INTEGER,
        base TEXT,
        pattern_key TEXT,
        closed_at INTEGER,
        outcome TEXT,
        stage TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS score (
        id INTEGER PRIMARY KEY CHECK (id=1),
        green INTEGER DEFAULT 0,
        loss  INTEGER DEFAULT 0
    )""")
    if not cur.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        cur.execute("INSERT INTO score (id, green, loss) VALUES (1,0,0)")

    # migrações idempotentes
    for col, ddl in [
        ("d_final","ALTER TABLE pending ADD COLUMN d_final INTEGER"),
        ("last_post_short","ALTER TABLE pending ADD COLUMN last_post_short TEXT"),
        ("last_post_long","ALTER TABLE pending ADD COLUMN last_post_long TEXT"),
        ("last_conf_short","ALTER TABLE pending ADD COLUMN last_conf_short REAL"),
        ("last_conf_long","ALTER TABLE pending ADD COLUMN last_conf_long REAL"),
        ("base","ALTER TABLE pending ADD COLUMN base TEXT"),
        ("pattern_key","ALTER TABLE pending ADD COLUMN pattern_key TEXT"),
        ("closed_at","ALTER TABLE pending ADD COLUMN closed_at INTEGER"),
        ("outcome","ALTER TABLE pending ADD COLUMN outcome TEXT"),
        ("stage","ALTER TABLE pending ADD COLUMN stage TEXT"),
    ]:
        try: cur.execute(f"SELECT {col} FROM pending LIMIT 1")
        except sqlite3.OperationalError:
            try: cur.execute(ddl)
            except sqlite3.OperationalError: pass

    # tabela para logs de ML
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER NOT NULL,
            pending_id INTEGER,
            after INTEGER,
            base TEXT,
            pattern_key TEXT,
            conf_short REAL,
            conf_long REAL,
            gap_short REAL,
            gap_long REAL,
            samples_short INTEGER,
            proba_json TEXT,
            chosen INTEGER,
            chosen_by TEXT,
            label INTEGER,
            stage TEXT,
            outcome TEXT
        )
    """)

    # corrige linhas antigas
    try:
        cur.execute("UPDATE pending SET stage='OPEN'      WHERE stage IS NULL OR stage=''")
        cur.execute("UPDATE pending SET outcome='PENDING' WHERE outcome IS NULL OR outcome=''")
    except sqlite3.OperationalError:
        pass

    con.commit(); con.close()

migrate_db()

# ========= Telegram =========
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                "disable_web_page_preview": True})

# ========= Score =========
def bump_score(outcome: str):
    con = _connect(); cur = con.cursor()
    row = cur.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
    g, l = (row["green"], row["loss"]) if row else (0, 0)
    if outcome.upper() == "GREEN": g += 1
    elif outcome.upper() == "LOSS": l += 1
    cur.execute("INSERT OR REPLACE INTO score (id, green, loss) VALUES (1,?,?)", (g, l))
    con.commit(); con.close()

def reset_score(): _exec_write("INSERT OR REPLACE INTO score (id, green, loss) VALUES (1,0,0)")

def score_text() -> str:
    con = _connect()
    row = con.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
    con.close()
    if not row: return "0 GREEN × 0 LOSS — 0.0%"
    g, l = int(row["green"]), int(row["loss"])
    total = g + l
    acc = (g/total*100.0) if total>0 else 0.0
    return f"{g} GREEN × {l} LOSS — {acc:.1f}%"

# ========= Timeline / N-gram =========
def append_timeline(seq: List[int]):
    for n in seq:
        _exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)", (now_ts(), int(n)))

def get_tail(window:int) -> List[int]:
    con = _connect()
    rows = con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (window,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def _ctx_counts(tail: List[int], ctx: List[int]) -> Dict[int,int]:
    k = len(ctx)
    cnt = {1:0,2:0,3:0,4:0}
    if k == 0 or len(tail) <= k: return cnt
    for i in range(k, len(tail)):
        if tail[i-k:i] == ctx:
            nxt = tail[i]
            if nxt in cnt: cnt[nxt] += 1
    return cnt

def _post_from_tail(tail: List[int], after: Optional[int], candidates: List[int]) -> Dict[int,float]:
    if not tail:
        return {c: 1.0/len(candidates) for c in candidates}
    W = [0.46, 0.30, 0.16, 0.08]
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

    posts = {c: 0.0 for c in candidates}
    ctxs  = [(4,ctx4),(3,ctx3),(2,ctx2),(1,ctx1)]
    for lvl, ctx in ctxs:
        if not ctx: continue
        counts = _ctx_counts(tail, ctx[:-1])
        tot = sum(counts.values())
        if tot == 0: continue
        for n in candidates:
            posts[n] += W[4-lvl] * (counts.get(n,0)/tot)
    s = sum(posts.values()) or 1e-9
    return {k: v/s for k,v in posts.items()}

def _best_conf_gap(post: Dict[int,float]) -> Tuple[int,float,float]:
    top = sorted(post.items(), key=lambda kv: kv[1], reverse=True)[:2]
    best = top[0][0]; conf = top[0][1]
    gap  = top[0][1] - (top[1][1] if len(top)>1 else 0.0)
    return best, conf, gap

def _support_from_tail(tail: List[int], ctx_len: int) -> int:
    # aproximação do total de pares/trincas úteis para posterior
    return max(0, len(tail) - max(1, ctx_len))

# ========= Parsers/estratégia =========
ENTRY_RX = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
SEQ_RX   = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
AFTER_RX = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)
KWOK_RX  = re.compile(r"\bKWOK\s*([1-4])\s*-\s*([1-4])", re.I)
SSH_RX   = re.compile(r"\bSS?H\s*([1-4])(?:-([1-4]))?(?:-([1-4]))?(?:-([1-4]))?", re.I)
ODD_RX   = re.compile(r"\bODD\b", re.I)
EVEN_RX  = re.compile(r"\bEVEN\b", re.I)

GREEN_RX = re.compile(r"(?:\bgr+e+e?n\b|\bwin\b|✅)", re.I)
LOSS_RX  = re.compile(r"(?:\blo+s+s?\b|\bred\b|❌|\bperdemos\b)", re.I)

PAREN_GROUP_RX = re.compile(r"\(([^)]*)\)")
ANY_14_RX      = re.compile(r"[1-4]")

def parse_candidates_and_pattern(t: str) -> Tuple[List[int], str]:
    m = KWOK_RX.search(t)
    if m:
        a,b = int(m.group(1)), int(m.group(2))
        base = sorted(list({a,b}))
        return base, f"KWOK-{a}-{b}"
    if ODD_RX.search(t):  return [1,3], "ODD"
    if EVEN_RX.search(t): return [2,4], "EVEN"
    m = SSH_RX.search(t)
    if m:
        nums = [int(g) for g in m.groups() if g]
        base = sorted(list(dict.fromkeys(nums)))[:4]
        return base, "SSH-" + "-".join(str(x) for x in base)
    m = SEQ_RX.search(t)
    if m:
        parts = [int(x) for x in re.findall(r"[1-4]", m.group(1))]
        seen, base = set(), []
        for n in parts:
            if n not in seen:
                seen.add(n); base.append(n)
            if len(base) == 3: break
        if base: return base, "SEQ"
    return [1,2,3,4], "GEN"

def parse_entry_text(text: str) -> Optional[Dict]:
    t = re.sub(r"\s+", " ", text).strip()
    if not ENTRY_RX.search(t): return None
    base, pattern_key = parse_candidates_and_pattern(t)
    mseq = SEQ_RX.search(t)
    seq = [int(x) for x in re.findall(r"[1-4]", mseq.group(1))] if mseq else []
    mafter = AFTER_RX.search(t)
    after_num = int(mafter.group(1)) if mafter else None
    return {"seq": seq, "after": after_num, "raw": t, "base": base, "pattern_key": pattern_key}

def parse_close_numbers(text: str) -> List[int]:
    t = re.sub(r"\s+", " ", text)
    groups = PAREN_GROUP_RX.findall(t)
    if groups:
        nums = re.findall(r"[1-4]", groups[-1])
        return [int(x) for x in nums][:3]
    nums = ANY_14_RX.findall(t)
    return [int(x) for x in nums][:3]

# ========= Decisor (híbrido + estratégia) =========
def choose_single_number_hybrid(after: Optional[int], candidates: List[int], pattern_key: str = "GEN") -> Tuple[Optional[int], float, float, int, Dict[int,float], Dict[int,float]]:
    candidates = sorted(list(dict.fromkeys([c for c in candidates if c in (1,2,3,4)]))) or [1,2,3,4]
    tail_s = get_tail(SHORT_WINDOW)
    tail_l = get_tail(LONG_WINDOW)

    # dados suficientes?
    total_ctx = len(get_tail(LONG_WINDOW))
    if total_ctx < MIN_TOTAL_CTX:
        return None, 0.0, 0.0, len(tail_s), {}, {}

    post_s = _post_from_tail(tail_s, after, candidates)
    post_l = _post_from_tail(tail_l, after, candidates)
    b_s, c_s, g_s = _best_conf_gap(post_s)
    b_l, c_l, g_l = _best_conf_gap(post_l)

    # suporte mínimo por janela
    sup_s = _support_from_tail(tail_s, 2)
    sup_l = _support_from_tail(tail_l, 3)
    if sup_s < MIN_SUPPORT_SHORT or sup_l < MIN_SUPPORT_LONG:
        return None, c_s, c_l, len(tail_s), post_s, post_l

    # thresholds por padrão
    pk = (pattern_key or "GEN").upper()
    # normaliza chave base (ODD/EVEN/GEN) para mapping
    base_pk = "GEN"
    if pk.startswith("KWOK"): base_pk = "KWOK"
    elif pk.startswith("SSH"): base_pk = "SSH"
    elif pk.startswith("SEQ"): base_pk = "SEQ"
    elif pk == "ODD": base_pk = "ODD"
    elif pk == "EVEN": base_pk = "EVEN"
    thr = PATTERN_THRESHOLDS.get(base_pk, {"conf_s": CONF_SHORT_MIN, "conf_l": CONF_LONG_MIN, "gap": GAP_MIN})
    conf_s_min = max(CONF_SHORT_MIN,  thr.get("conf_s", CONF_SHORT_MIN))
    conf_l_min = max(CONF_LONG_MIN,   thr.get("conf_l", CONF_LONG_MIN))
    gap_min    = max(GAP_MIN,         thr.get("gap", GAP_MIN))

    best = None
    if b_s == b_l and c_s >= conf_s_min and c_l >= conf_l_min and g_s >= gap_min and g_l >= gap_min:
        best = b_s

    return best, c_s, c_l, len(tail_s), post_s, post_l

# ========= ML helpers =========
_MODEL = None  # legado (não usado com modelos por padrão)

def _top2_from_post(post: Dict[int,float]) -> Tuple[float,float]:
    vals = sorted(post.values(), reverse=True)
    a = vals[0] if vals else 0.0
    b = vals[1] if len(vals)>1 else 0.0
    return a, b

def build_features_for_candidates(after: Optional[int], base: List[int],
                                  conf_s: float, conf_l: float,
                                  post_s: Dict[int,float], post_l: Dict[int,float],
                                  samples_s: int, pattern_key: str) -> Dict[str, float]:
    a_s, b_s = _top2_from_post(post_s)
    a_l, b_l = _top2_from_post(post_l)
    gap_s = max(0.0, a_s - b_s)
    gap_l = max(0.0, a_l - b_l)
    pk = (pattern_key or "GEN").upper()
    return {
        "after": float(after or 0),
        "has_after": 1.0 if after else 0.0,
        "conf_short": float(conf_s or 0.0),
        "conf_long":  float(conf_l or 0.0),
        "gap_short":  float(gap_s),
        "gap_long":   float(gap_l),
        "samples_short": float(samples_s or 0),
        "base_len": float(len(base or [])),
        "pat_GEN":   1.0 if pk.startswith("GEN") else 0.0,
        "pat_KWOK":  1.0 if pk.startswith("KWOK") else 0.0,
        "pat_SSH":   1.0 if pk.startswith("SSH") else 0.0,
        "pat_SEQ":   1.0 if pk.startswith("SEQ") else 0.0,
        "pat_ODD":   1.0 if pk == "ODD" else 0.0,
        "pat_EVEN":  1.0 if pk == "EVEN" else 0.0,
    }

def expand_candidate_features(base_feats: Dict[str,float], cand: int,
                              post_s: Dict[int,float], post_l: Dict[int,float]) -> Dict[str,float]:
    x = dict(base_feats)
    x.update({
        "cand": float(cand),
        "post_s": float(post_s.get(cand, 0.0)),
        "post_l": float(post_l.get(cand, 0.0)),
    })
    return x

def _load_registry():
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    try:
        with open(os.path.join(MODEL_DIR, "registry.json"), "r", encoding="utf-8") as f:
            _REGISTRY = json.load(f)
    except Exception:
        _REGISTRY = {"models": {}, "skipped": {}}
    return _REGISTRY

def load_model_for(pattern_key: str):
    """Retorna o modelo específico do padrão (se existir), senão GLOBAL, senão None."""
    pk = (pattern_key or "GEN").upper()
    base_pk = "GEN"
    if pk.startswith("KWOK"): base_pk = "KWOK"
    elif pk.startswith("SSH"): base_pk = "SSH"
    elif pk.startswith("SEQ"): base_pk = "SEQ"
    elif pk == "ODD": base_pk = "ODD"
    elif pk == "EVEN": base_pk = "EVEN"

    reg = _load_registry()

    # 1) tenta modelo específico
    entry = reg.get("models", {}).get(base_pk)
    if entry:
        path = entry.get("path")
        if path and os.path.exists(path):
            if base_pk not in _MODELS_BY_PATTERN:
                _MODELS_BY_PATTERN[base_pk] = joblib.load(path)
            return _MODELS_BY_PATTERN[base_pk]

    # 2) fallback GLOBAL
    glob = reg.get("models", {}).get("GLOBAL")
    if glob:
        gpath = glob.get("path")
        if gpath and os.path.exists(gpath):
            if "GLOBAL" not in _MODELS_BY_PATTERN:
                _MODELS_BY_PATTERN["GLOBAL"] = joblib.load(gpath)
            return _MODELS_BY_PATTERN["GLOBAL"]

    # 3) sem modelo → None (o bot cai pro híbrido)
    return None

def ml_score_candidates(candidates: List[int],
                        after: Optional[int],
                        base: List[int],
                        pattern_key: str,
                        conf_s: float, conf_l: float,
                        post_s: Dict[int,float], post_l: Dict[int,float],
                        samples_s: int) -> Tuple[Optional[int], Dict[int,float], float]:
    model = load_model_for(pattern_key)
    if model is None:
        return None, {}, 0.0

    base_feats = build_features_for_candidates(after, base, conf_s, conf_l, post_s, post_l, samples_s, pattern_key)
    X, order = [], []
    for c in candidates:
        feats = expand_candidate_features(base_feats, c, post_s, post_l)
        keys = sorted(feats.keys())
        X.append([feats[k] for k in keys])
        order.append((c, keys))
    try:
        proba = model.predict_proba(X)
        if len(proba.shape) == 2 and proba.shape[1] >= 2:
            p_green = [float(p[1]) for p in proba]
        else:
            p_green = [float(p[0]) for p in proba]
    except Exception as e:
        print(f"[ML] predict_proba falhou: {e}")
        return None, {}, 0.0

    proba_by_cand = {order[i][0]: p_green[i] for i in range(len(order))}
    top = sorted(proba_by_cand.items(), key=lambda kv: kv[1], reverse=True)[:2]
    if not top:
        return None, proba_by_cand, 0.0
    best_cand, best_p = top[0]
    second_p = top[1][1] if len(top)>1 else 0.0
    gap = max(0.0, best_p - second_p)
    if best_p >= ML_MIN_PROBA and gap >= ML_GAP_MIN:
        return best_cand, proba_by_cand, gap
    return None, proba_by_cand, gap

# ========= Pending helpers =========
def get_open_pending() -> Optional[sqlite3.Row]:
    con = _connect()
    row = con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone()
    con.close()
    return row

def _seen_list(row: sqlite3.Row) -> List[str]:
    seen = (row["seen"] or "").strip()
    return [s for s in seen.split("-") if s]

def _set_seen(row_id:int, seen_list:List[str]):
    _exec_write("UPDATE pending SET seen=? WHERE id=?", ("-".join(seen_list[:3]), row_id))

def _ensure_final_deadline_when_two(row: sqlite3.Row):
    if int(row["d_final"] or 0) > 0: return
    if len(_seen_list(row)) == 2:
        _exec_write("UPDATE pending SET d_final=? WHERE id=?", (now_ts() + FINAL_TIMEOUT, int(row["id"])))

def _close_now(row: sqlite3.Row, suggested:int, final_seen:List[str]):
    obs_nums = [int(x) for x in final_seen if x.isdigit()]
    if len(obs_nums) >= 1 and obs_nums[0] == suggested:
        outcome, stage_lbl = "GREEN", "G0"
    elif len(obs_nums) >= 2 and obs_nums[1] == suggested:
        outcome, stage_lbl = "GREEN", "G1"
    elif len(obs_nums) >= 3 and obs_nums[2] == suggested:
        outcome, stage_lbl = "GREEN", "G2"
    else:
        outcome, stage_lbl = "LOSS", "G2"

    _exec_write("""UPDATE pending
                      SET open=0, seen=?, closed_at=?, outcome=?, stage=?
                    WHERE id=?""",
                ("-".join(final_seen[:3]), now_ts(), outcome, stage_lbl, int(row["id"])))
    bump_score(outcome.upper())

    # Preenche rótulo no ml_log (se existir)
    try:
        _exec_write("""
            UPDATE ml_log
               SET label = ?, stage = ?, outcome = ?
             WHERE pending_id = ?
        """, (1 if outcome.upper()=="GREEN" else 0, stage_lbl, outcome.upper(), int(row["id"])))
    except Exception as e:
        print(f"[ML] log label update fail: {e}")

    our_num_display = suggested if outcome=="GREEN" else "X"
    msg = (f"{'🟢' if outcome=='GREEN' else '🔴'} <b>{outcome}</b> — finalizado "
           f"(<b>{stage_lbl}</b>, nosso={our_num_display}, observados={'-'.join(final_seen[:3])}).\n"
           f"📊 Geral: {score_text()}")
    return msg

def open_pending(suggested: int, conf_short:float, conf_long:float,
                 post_short:Dict[int,float], post_long:Dict[int,float],
                 base:List[int], pattern_key:str):
    _exec_write("""INSERT INTO pending
        (created_at, suggested, open, seen, opened_at,
         last_post_short, last_post_long, last_conf_short, last_conf_long,
         d_final, base, pattern_key, closed_at, outcome, stage)
        VALUES (?,?,?,?,?,?,?,?,?,NULL,?,?,NULL,'PENDING','OPEN')
    """, (now_ts(), int(suggested), 1, "", now_ts(),
          json.dumps(post_short), json.dumps(post_long),
          float(conf_short), float(conf_long),
          json.dumps(base), pattern_key))

def _maybe_close_by_final_timeout():
    row = get_open_pending()
    if not row: return None
    d_final = int(row["d_final"] or 0)
    if d_final <= 0 or now_ts() < d_final: return None
    seen_list = _seen_list(row)
    if len(seen_list) == 2:
        seen_list.append("X")
        return _close_now(row, int(row["suggested"] or 0), seen_list)
    return None

# ========= Relatório 5 em 5 minutos =========
def _report_snapshot(last_secs:int=300) -> Dict[str,int]:
    since = now_ts() - max(60, int(last_secs))
    rows = _query_all("""
        SELECT outcome, stage FROM pending
        WHERE closed_at IS NOT NULL AND closed_at >= ?
    """, (since,))
    g0=g1=g2=0; l0=l1=l2=0
    for r in rows:
        oc = (r["outcome"] or "").upper()
        st = (r["stage"] or "").upper()
        if oc == "GREEN":
            if st == "G0": g0 += 1
            elif st == "G1": g1 += 1
            else: g2 += 1
        elif oc == "LOSS":
            if st == "G0": l0 += 1
            elif st == "G1": l1 += 1
            else: l2 += 1
    row = _query_all("SELECT green, loss FROM score WHERE id=1")
    row = row[0] if row else None
    g = int(row["green"] if row else 0); l = int(row["loss"] if row else 0)
    total = g + l
    acc = (g/total) if total>0 else 0.0
    return {"g0":g0,"g1":g1,"g2":g2,"l0":l0,"l1":l1,"l2":l2,"day_green":g,"day_loss":l,"day_acc":acc}

def _day_mood(acc: float) -> str:
    if acc >= GOOD_DAY_THRESHOLD: return "🔥 <b>Dia bom</b> — continuando bem!"
    if acc <= BAD_DAY_THRESHOLD:  return "⚠️ <b>Dia ruim</b> — atenção!"
    return "🔎 <b>Dia neutro</b> — operação regular."

async def _reporter_loop():
    while True:
        try:
            snap = _report_snapshot(300)
            gtot = snap["g0"] + snap["g1"] + snap["g2"]
            ltot = snap["l0"] + snap["l1"] + snap["l2"]
            txt = (
                "📈 <b>Relatório (últimos 5 min)</b>\n"
                f"G0: <b>{snap['g0']}</b> GREEN / <b>{snap['l0']}</b> LOSS\n"
                f"G1: <b>{snap['g1']}</b> GREEN / <b>{snap['l1']}</b> LOSS\n"
                f"G2: <b>{snap['g2']}</b> GREEN / <b>{snap['l2']}</b> LOSS\n"
                f"Total (5min): <b>{gtot}</b> GREEN × <b>{ltot}</b> LOSS\n"
                "—\n"
                f"📊 <b>Dia</b>: <b>{snap['day_green']}</b> GREEN × <b>{snap['day_loss']}</b> LOSS — "
                f"{snap['day_acc']*100:.1f}%\n"
                f"{_day_mood(snap['day_acc'])}"
            )
            await tg_send_text(TARGET_CHANNEL, txt)
        except Exception as e:
            print(f"[RELATORIO] erro: {e}")
        await asyncio.sleep(REPORT_EVERY_SEC)

# ========= Reset diário (00:00 no fuso definido) =========
async def _daily_reset_loop():
    tz = None
    try:
        tz = ZoneInfo(RESET_TZ)
    except Exception:
        tz = timezone.utc
    while True:
        try:
            now = datetime.now(tz)
            tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            await asyncio.sleep(max(1.0, (tomorrow - now).total_seconds()))
            reset_score()
            tzname = getattr(tz, "key", "UTC")
            await tg_send_text(TARGET_CHANNEL, f"🕛 <b>Reset diário (00:00 {tzname})</b>\n📊 Placar zerado para o novo dia.")
        except Exception as e:
            print(f"[RESET] erro: {e}")
            await asyncio.sleep(60)

# ========= Rotas =========
@app.get("/")
async def root():
    return {"ok": True, "service": "guardiao-auto-bot (GEN híbrido + estratégia + ML por padrão)"}

@app.on_event("startup")
async def _boot_tasks():
    try: asyncio.create_task(_reporter_loop())
    except Exception as e: print(f"[START] reporter error: {e}")
    try: asyncio.create_task(_daily_reset_loop())
    except Exception as e: print(f"[START] reset error: {e}")

@app.get("/health")
async def health():
    pend = get_open_pending()
    return {
        "ok": True,
        "db": DB_PATH,
        "pending_open": bool(pend),
        "pending_seen": (pend["seen"] if pend else ""),
        "d_final": int(pend["d_final"] or 0) if pend else 0,
        "time": ts_str()
    }

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    msg_timeout = _maybe_close_by_final_timeout()
    if msg_timeout: await tg_send_text(TARGET_CHANNEL, msg_timeout)

    data = await request.json()
    msg = data.get("channel_post") or data.get("message") \
        or data.get("edited_channel_post") or data.get("edited_message") or {}
    text = (msg.get("text") or msg.get("caption") or "").strip()
    chat = msg.get("chat") or {}
    chat_id = str(chat.get("id") or "")
    if SOURCE_CHANNEL and chat_id != str(SOURCE_CHANNEL):
        return {"ok": True, "skipped": "outro_chat"}
    if not text:
        return {"ok": True, "skipped": "sem_texto"}

    # Fechamentos
    if GREEN_RX.search(text) or LOSS_RX.search(text):
        pend = get_open_pending()
        nums = parse_close_numbers(text)
        if pend and nums:
            seen = _seen_list(pend)
            for n in nums:
                if len(seen) >= 3: break
                seen.append(str(int(n)))
                suggested = int(pend["suggested"] or 0)
                obs_nums = [int(x) for x in seen if x.isdigit()]
                if (len(obs_nums) >= 1 and obs_nums[0] == suggested) or \
                   (len(obs_nums) >= 2 and obs_nums[1] == suggested) or \
                   (len(obs_nums) >= 3 and obs_nums[2] == suggested):
                    out = _close_now(pend, suggested, seen)
                    await tg_send_text(TARGET_CHANNEL, out)
                    return {"ok": True, "closed": "green_imediato"}
            _set_seen(int(pend["id"]), seen)
            pend = get_open_pending()
            if pend and len(_seen_list(pend)) == 2:
                _ensure_final_deadline_when_two(pend)
            pend = get_open_pending()
            if pend and len(_seen_list(pend)) >= 3:
                out = _close_now(pend, int(pend["suggested"] or 0), _seen_list(pend))
                await tg_send_text(TARGET_CHANNEL, out)
                return {"ok": True, "closed": "loss_3_observados"}
        return {"ok": True, "noted_close": True}

    # Entrada confirmada
    parsed = parse_entry_text(text)
    if not parsed:
        return {"ok": True, "skipped": "nao_eh_entrada_confirmada"}

    msg_timeout = _maybe_close_by_final_timeout()
    if msg_timeout: await tg_send_text(TARGET_CHANNEL, msg_timeout)

    seq = parsed["seq"] or []
    if seq: append_timeline(seq)

    # ===== Guard-rails de risco / cooldown =====
    global _last_cooldown_until
    now = now_ts()
    if now < _last_cooldown_until:
        return {"ok": True, "skipped": "cooldown"}
    if _loss_streak() >= MAX_LOSS_STREAK or (not _day_acc_ok()):
        _last_cooldown_until = now + COOLDOWN_SECONDS
        await tg_send_text(TARGET_CHANNEL, "⛔️ <b>Pausa de proteção ativada</b> (sequência de loss/dia ruim). Retomamos automaticamente depois.")
        return {"ok": True, "cooldown_started": True}

    after       = parsed["after"]
    base        = parsed["base"] or [1,2,3,4]
    pattern_key = parsed["pattern_key"] or "GEN"

    # 1) decisão híbrida (endurecida + thresholds por padrão)
    best_h, conf_s, conf_l, samples_s, post_s, post_l = choose_single_number_hybrid(after, base, pattern_key)

    # 2) decisão ML (por padrão, se houver modelo)
    best_ml, proba_by_cand, ml_gap = ml_score_candidates(base, after, base, pattern_key, conf_s, conf_l, post_s, post_l, samples_s)

    # 3) ensemble
    chosen = None
    chosen_by = None
    if ENSEMBLE_MODE == "ml_only":
        if best_ml is not None:
            chosen, chosen_by = best_ml, "ml"
    elif ENSEMBLE_MODE == "blend":
        # concordância → perfeito
        if (best_h is not None) and (best_ml is not None) and (best_h == best_ml):
            chosen, chosen_by = best_h, "ensemble"
        # ML muito confiante leva
        elif best_ml is not None and proba_by_cand.get(best_ml, 0.0) >= ML_MIN_PROBA:
            chosen, chosen_by = best_ml, "ml"
        # só híbrido forte quando ML fraco/indisponível
        elif (best_h is not None) and ((best_ml is None) or (proba_by_cand.get(best_ml,0.0) < ML_MIN_PROBA*0.9)):
            chosen, chosen_by = best_h, "hybrid"
        else:
            chosen, chosen_by = None, "no_consensus"
    else:  # "gate" (conservador)
        if best_h is not None:
            if best_ml is not None and best_ml != best_h:
                p_h = proba_by_cand.get(best_h, 0.0)
                if p_h >= ML_MIN_PROBA * 0.9:
                    chosen, chosen_by = best_h, "hybrid"
                else:
                    chosen, chosen_by = None, "blocked_by_ml"
            else:
                chosen, chosen_by = best_h, "hybrid"
        else:
            if best_ml is not None:
                chosen, chosen_by = best_ml, "ml"

    if chosen is None:
        return {"ok": True, "skipped_low_conf_or_disagree": True}

    # 4) abre pendência + log de ML
    open_pending(chosen, conf_s, conf_l, post_s, post_l, base, pattern_key)

    try:
        gap_s_vals = sorted(post_s.values(), reverse=True)
        gap_l_vals = sorted(post_l.values(), reverse=True)
        gap_s = (gap_s_vals[0] - (gap_s_vals[1] if len(gap_s_vals)>1 else 0.0)) if gap_s_vals else 0.0
        gap_l = (gap_l_vals[0] - (gap_l_vals[1] if len(gap_l_vals)>1 else 0.0)) if gap_l_vals else 0.0
        _exec_write("""
            INSERT INTO ml_log (created_at, pending_id, after, base, pattern_key,
                                conf_short, conf_long, gap_short, gap_long, samples_short,
                                proba_json, chosen, chosen_by, label, stage, outcome)
            VALUES (?, (SELECT id FROM pending ORDER BY id DESC LIMIT 1), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
        """, (now_ts(),
              int(after or 0),
              json.dumps(base or []),
              str(pattern_key or "GEN"),
              float(conf_s), float(conf_l),
              float(max(0.0, gap_s)),
              float(max(0.0, gap_l)),
              int(samples_s or 0),
              json.dumps({int(k): float(v) for k,v in (proba_by_cand or {}).items()}),
              int(chosen),
              str(chosen_by or "hybrid")))
    except Exception as e:
        print(f"[ML] log insert fail: {e}")

    base_txt = ", ".join(str(x) for x in base) if base else "—"
    aft_txt  = f" após {after}" if after else ""
    txt = (
        f"🎯 <b>Número seco (G0):</b> <b>{chosen}</b>\n"
        f"🧩 <b>Padrão:</b> {pattern_key}{aft_txt}\n"
        f"🧮 <b>Base:</b> [{base_txt}]\n"
        f"📊 <b>Conf (curta/longa):</b> {conf_s*100:.2f}% / {conf_l*100:.2f}% | <b>Amostra≈</b>{samples_s}\n"
        f"🤖 <b>Decisão:</b> {chosen_by.upper() if chosen_by else 'HYBRID'}"
    )
    await tg_send_text(TARGET_CHANNEL, txt)
    return {"ok": True, "posted": True, "best": chosen}
