#!/usr/bin/env python3
# -- coding: utf-8 --

"""
Webhook (GEN híbrido + estratégia + timeout 45s + relatório 1h + reset diário America/Sao_Paulo)
"""

import os, re, time, json, sqlite3, asyncio
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone, timedelta
import httpx
from fastapi import FastAPI, Request, HTTPException
import zoneinfo  # fuso local

# ========= ENV =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1002796105884").strip()  # destino
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "-1002810508717").strip()  # fonte
DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"

if not TG_BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")
if not WEBHOOK_TOKEN:
    raise RuntimeError("Defina WEBHOOK_TOKEN no ambiente.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
app = FastAPI(title="guardiao-auto-bot (GEN híbrido + estratégia)", version="3.2.0")

# ========= Fuso local =========
TZ_LOCAL = zoneinfo.ZoneInfo("America/Sao_Paulo")
def now_local():
    return datetime.now(TZ_LOCAL)
def day_key(dt=None):
    dt = dt or now_local()
    return dt.strftime("%Y%m%d")
def hour_key(dt=None):
    dt = dt or now_local()
    return dt.strftime("%Y%m%d%H")

# ========= HÍBRIDO (curta/longa) =========
# Parâmetros mais conservadores para reduzir overtrade e loss
SHORT_WINDOW    = 80       # era 40
LONG_WINDOW     = 800      # era 1000
CONF_SHORT_MIN  = 0.85     # era 0.70
CONF_LONG_MIN   = 0.90     # era 0.80
GAP_MIN         = 0.050    # era 0.020
FINAL_TIMEOUT   = 45       # começa quando houver 2 observados

# Guard-rails adicionais (robustez da entrada)
MIN_SAMPLES_SHORT = 120
MIN_SAMPLES_LONG  = 400
MAX_LOSS_STREAK   = 2
COOLDOWN_SECONDS  = 120
QUIET_HOURS       = [(0, 5)]  # evita operar entre 00:00 e 05:59
_last_cooldown_until = 0  # memória

# ========= Relatório / Sinais do dia =========
# Troca para 1x por hora (anti-flood) e thresholds mais rígidos
REPORT_EVERY_SEC   = 60 * 60
GOOD_DAY_THRESHOLD = 0.80
BAD_DAY_THRESHOLD  = 0.50

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

    # corrige linhas antigas com NULL/vazio
    try:
        cur.execute("UPDATE pending SET stage='OPEN'    WHERE stage IS NULL OR stage=''")
        cur.execute("UPDATE pending SET outcome='PENDING' WHERE outcome IS NULL OR outcome=''")
    except sqlite3.OperationalError:
        pass

    # ===== PATCH-GUA schema de relatório/controle =====
    cur.execute("""
    CREATE TABLE IF NOT EXISTS relatorio_controle (
        chat_id TEXT NOT NULL,
        hour_key TEXT NOT NULL,
        sent_at INTEGER NOT NULL,
        PRIMARY KEY (chat_id, hour_key)
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS relatorio_dia (
        chat_id TEXT PRIMARY KEY,
        day_key TEXT NOT NULL
    );
    """)
    # ===== PATCH-GUA (fim) =====

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
def choose_single_number_hybrid(after: Optional[int], candidates: List[int]) -> Tuple[Optional[int], float, float, int, Dict[int,float], Dict[int,float]]:
    candidates = sorted(list(dict.fromkeys([c for c in candidates if c in (1,2,3,4)]))) or [1,2,3,4]
    tail_s = get_tail(SHORT_WINDOW)
    tail_l = get_tail(LONG_WINDOW)
    post_s = _post_from_tail(tail_s, after, candidates)
    post_l = _post_from_tail(tail_l, after, candidates)
    b_s, c_s, g_s = _best_conf_gap(post_s)
    b_l, c_l, g_l = _best_conf_gap(post_l)
    best = None
    # Exige consenso curto=long, confs altas e gaps reais
    if b_s == b_l and c_s >= CONF_SHORT_MIN and c_l >= CONF_LONG_MIN and g_s >= GAP_MIN and g_l >= GAP_MIN:
        best = b_s
    return best, c_s, c_l, len(tail_s), post_s, post_l

# ========= Guard-rails: decisão final de entrada =========
def get_loss_streak(limit=20):
    rows = _query_all(
        "SELECT outcome FROM pending WHERE closed_at IS NOT NULL ORDER BY id DESC LIMIT ?",
        (int(limit),)
    )
    s = 0
    for r in rows:
        oc = (r["outcome"] or "").upper()
        if oc == "LOSS":
            s += 1
        else:
            break
    return s

def gua_can_enter(conf_s, conf_l, n_s, n_l, gap):
    global _last_cooldown_until
    if time.time() < _last_cooldown_until:
        return False, "cooldown_timer"

    hour_now = now_local().hour
    for h0, h1 in QUIET_HOURS:
        if h0 <= hour_now <= h1:
            return False, "quiet_hours"

    if n_s < MIN_SAMPLES_SHORT or n_l < MIN_SAMPLES_LONG:
        return False, "few_samples"

    if conf_s < CONF_SHORT_MIN:
        return False, "low_conf_short"
    if conf_l < CONF_LONG_MIN:
        return False, "low_conf_long"
    if abs(conf_s - conf_l) < GAP_MIN:
        return False, "low_gap"

    ls = get_loss_streak()
    if ls >= MAX_LOSS_STREAK:
        _last_cooldown_until = time.time() + COOLDOWN_SECONDS
        return False, "loss_streak_cooldown"

    return True, "ok"

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
    our_num_display = suggested if outcome=="GREEN" else "X"
    msg = (f"{'🟢' if outcome=='GREEN' else '🔴'} <b>{outcome}</b> — finalizado "
           f"(<b>{stage_lbl}</b>, nosso={our_num_display}, observados={'-'.join(final_seen[:3])}).\n"
           f"📊 Geral: {score_text()}")
    return msg

def open_pending(suggested: int, conf_short:float, conf_long:float,
                 post_short:Dict[int,float], post_long:Dict[int,float],
                 base:List[int], pattern_key:str):
    # grava stage/outcome não nulos (compat)
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

# ========= Controle de reset diário por grupo e anti-duplicação por hora =========
def gua_reset_if_new_day(chat_id: str):
    dk = day_key()
    rows = _query_all("SELECT day_key FROM relatorio_dia WHERE chat_id=?", (chat_id,))
    if not rows:
        _exec_write("INSERT OR REPLACE INTO relatorio_dia (chat_id, day_key) VALUES (?,?)", (chat_id, dk))
        return
    if rows[0]["day_key"] != dk:
        # Se quiser zerar contadores por grupo, faça aqui (ex.: limpar tabela específica por chat_id)
        _exec_write("UPDATE relatorio_dia SET day_key=? WHERE chat_id=?", (dk, chat_id))

def gua_try_reserve_hour(chat_id: str) -> bool:
    hk = hour_key()
    ts = int(time.time())
    try:
        _exec_write(
            "INSERT OR IGNORE INTO relatorio_controle(chat_id, hour_key, sent_at) VALUES (?,?,?)",
            (chat_id, hk, ts)
        )
        rows = _query_all(
            "SELECT sent_at FROM relatorio_controle WHERE chat_id=? AND hour_key=?",
            (chat_id, hk)
        )
        return bool(rows) and int(rows[0]["sent_at"]) == ts
    except Exception:
        return False

# ========= Relatório 1x por hora (por grupo) =========
def _report_snapshot_day(chat_id: str) -> Dict[str,int]:
    # janela do dia local (00:00 -> agora)
    start = now_local().replace(hour=0, minute=0, second=0, microsecond=0)
    start_ts = int(start.astimezone(timezone.utc).timestamp())

    rows = _query_all("""
        SELECT outcome, stage FROM pending
        WHERE closed_at IS NOT NULL AND closed_at >= ?
    """, (start_ts,))

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
    g = int(row[0]["green"] if row else 0)
    l = int(row[0]["loss"] if row else 0)
    total = g + l
    acc = (g/total) if total>0 else 0.0
    return {"g0":g0,"g1":g1,"g2":g2,"l0":l0,"l1":l1,"l2":l2,"day_green":g,"day_loss":l,"day_acc":acc}

def _day_mood(acc: float) -> str:
    if acc >= GOOD_DAY_THRESHOLD: return "Dia bom"
    if acc <= BAD_DAY_THRESHOLD:  return "Dia ruim"
    return "Dia neutro"

async def _reporter_loop():
    while True:
        try:
            # pega snapshot da última 1h
            snap = _report_snapshot(3600)
            gtot = snap["g0"] + snap["g1"] + snap["g2"]
            ltot = snap["l0"] + snap["l1"] + snap["l2"]

            txt = (
                "📈 <b>Relatório do dia (última 1h)</b>\n"
                f"G0: <b>{snap['g0']}</b> GREEN / <b>{snap['l0']}</b> LOSS\n"
                f"G1: <b>{snap['g1']}</b> GREEN / <b>{snap['l1']}</b> LOSS\n"
                f"G2: <b>{snap['g2']}</b> GREEN / <b>{snap['l2']}</b> LOSS\n"
                f"Total (1h): <b>{gtot}</b> GREEN × <b>{ltot}</b> LOSS\n"
                "—\n"
                f"📊 <b>Dia</b>: <b>{snap['day_green']}</b> GREEN × <b>{snap['day_loss']}</b> LOSS — "
                f"{snap['day_acc']*100:.1f}%\n"
                f"{_day_mood(snap['day_acc'])}"
            )

            await tg_send_text(TARGET_CHANNEL, txt)
        except Exception as e:
            print(f"[RELATORIO] erro: {e}")
        await asyncio.sleep(3600)  # 1h
