# -*- coding: utf-8 -*-
# Guardião — separação de canais (PUBLIC_CHANNEL vs IA_CHANNEL)
# - PUBLIC_CHANNEL: recebe apenas o G0 normal do canal
# - IA_CHANNEL: recebe apenas sinais da IA (FIRE) + resultados (GREEN/LOSS/G1/G2) da IA
# - REPL desativado por padrão
#
import os, re, json, time, sqlite3, asyncio, shutil
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone, timedelta
from collections import Counter

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# =========================
# ENV / CONFIG
# =========================
DB_PATH = os.getenv("DB_PATH", "/data/data.db").strip() or "/data/data.db"

OLD_DB_CANDIDATES = [
    "/var/data/data.db",
    "/opt/render/project/src/data.db",
    "/opt/render/project/src/data/data.db",
    "/data/data.db",
]
def _ensure_db_dir():
    try: os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    except Exception as e: print(f"[DB] Falha ao criar dir DB: {e}")
def _migrate_old_db_if_needed():
    if os.path.exists(DB_PATH): return
    for src in OLD_DB_CANDIDATES:
        if os.path.exists(src):
            try:
                _ensure_db_dir(); shutil.copy2(src, DB_PATH)
                print(f"[DB] Migrado {src} -> {DB_PATH}"); return
            except Exception as e:
                print(f"[DB] Erro migrando {src} -> {DB_PATH}: {e}")
_ensure_db_dir(); _migrate_old_db_if_needed()

TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
PUBLIC_CHANNEL = os.getenv("PUBLIC_CHANNEL", "").strip()   # -100...
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()

# Canais
IA_CHANNEL     = os.getenv("IA_CHANNEL", "").strip()
REPL_ENABLED   = os.getenv("REPL_ENABLED", "0").strip().lower() in ("1","true","yes","on")  # off por padrão
REPL_CHANNEL   = os.getenv("REPL_CHANNEL", "").strip()

TELEGRAM_API   = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# Modo
ALWAYS_FREE_FIRE = os.getenv("ALWAYS_FREE_FIRE", "1").strip().lower() in ("1","true","yes","on")

# Hiperparâmetros
WINDOW = int(os.getenv("WINDOW", "400"))
DECAY  = float(os.getenv("DECAY", "0.985"))
W4, W3, W2, W1 = 0.38, 0.30, 0.20, 0.12
ALPHA, BETA, GAMMA = 1.05, 0.70, 0.40
GAP_MIN = 0.08

MIN_CONF_G0 = float(os.getenv("MIN_CONF_G0", "0.55"))
MIN_GAP_G0  = float(os.getenv("MIN_GAP_G0", "0.04"))
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "1000"))

INTEL_DIR = os.getenv("INTEL_DIR", "/var/data/ai_intel").rstrip("/")
INTEL_MAX_BYTES = int(os.getenv("INTEL_MAX_BYTES", "1000000000"))
INTEL_SIGNAL_INTERVAL = float(os.getenv("INTEL_SIGNAL_INTERVAL", "20"))
INTEL_ANALYZE_INTERVAL = float(os.getenv("INTEL_ANALYZE_INTERVAL", "0.2"))
SELF_LABEL_IA = os.getenv("SELF_LABEL_IA", "Tiro seco por IA")

MAX_STAGE     = 3
FAST_RESULT   = True
SCORE_G0_ONLY = True

app = FastAPI(title="Guardião — canais separados", version="3.12.1")

# =========================
# SQLite
# =========================
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def exec_write(sql: str, params: tuple = (), retries: int = 8, wait: float = 0.25):
    for _ in range(retries):
        try:
            con = _connect(); con.execute(sql, params); con.commit(); con.close(); return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower(): time.sleep(wait); continue
            raise
    raise sqlite3.OperationalError("Banco bloqueado após várias tentativas.")

def query_all(sql: str, params: tuple = ()) -> list:
    con = _connect(); rows = con.execute(sql, params).fetchall(); con.close(); return rows

def query_one(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    con = _connect(); row = con.execute(sql, params).fetchone(); con.close(); return row

def init_db():
    con = _connect(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS ngram_stats (
        n INTEGER NOT NULL, ctx TEXT NOT NULL, next INTEGER NOT NULL, weight REAL NOT NULL,
        PRIMARY KEY (n, ctx, next)
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS stats_pattern (
        pattern_key TEXT NOT NULL, number INTEGER NOT NULL,
        wins INTEGER NOT NULL DEFAULT 0, losses INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (pattern_key, number)
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS stats_strategy (
        strategy TEXT NOT NULL, number INTEGER NOT NULL,
        wins INTEGER NOT NULL DEFAULT 0, losses INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (strategy, number)
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS suggestions (
        source_msg_id INTEGER PRIMARY KEY, strategy TEXT, seq_raw TEXT,
        context_key TEXT, pattern_key TEXT, base TEXT,
        suggested_number INTEGER, stage TEXT, sent_at INTEGER
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS last_by_strategy (
        strategy TEXT PRIMARY KEY, source_msg_id INTEGER, suggested_number INTEGER,
        context_key TEXT, pattern_key TEXT, stage TEXT, created_at INTEGER
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS daily_score (
        yyyymmdd TEXT PRIMARY KEY,
        g0 INTEGER NOT NULL DEFAULT 0,
        g1 INTEGER NOT NULL DEFAULT 0,
        g2 INTEGER NOT NULL DEFAULT 0,
        loss INTEGER NOT NULL DEFAULT 0,
        streak INTEGER NOT NULL DEFAULT 0
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pending_outcome (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL,
        strategy TEXT, suggested INTEGER NOT NULL, stage INTEGER NOT NULL,
        open INTEGER NOT NULL, window_left INTEGER NOT NULL,
        seen_numbers TEXT DEFAULT '',
        announced INTEGER NOT NULL DEFAULT 0,
        source TEXT NOT NULL DEFAULT 'CHAN'
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS daily_score_ia (
        yyyymmdd TEXT PRIMARY KEY,
        g0 INTEGER NOT NULL DEFAULT 0,
        loss INTEGER NOT NULL DEFAULT 0,
        streak INTEGER NOT NULL DEFAULT 0
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS recov_g1_stats (
        number INTEGER PRIMARY KEY,
        wins INTEGER NOT NULL DEFAULT 0,
        losses INTEGER NOT NULL DEFAULT 0
    )""")
    con.commit()
    for alter in [
        "ALTER TABLE pending_outcome ADD COLUMN seen_numbers TEXT DEFAULT ''",
        "ALTER TABLE pending_outcome ADD COLUMN announced INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE pending_outcome ADD COLUMN source TEXT NOT NULL DEFAULT 'CHAN'",
    ]:
        try: cur.execute(alter); con.commit()
        except sqlite3.OperationalError: pass
    con.close()
init_db()

# =========================
# Telegram helpers
# =========================
_httpx_client: Optional[httpx.AsyncClient] = None
async def get_httpx() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(timeout=10)
    return _httpx_client

@app.on_event("shutdown")
async def _shutdown():
    global _httpx_client
    try:
        if _httpx_client: await _httpx_client.aclose()
    except Exception as e:
        print(f"[HTTPX] close error: {e}")

async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN or not chat_id:
        return
    try:
        client = await get_httpx()
        asyncio.create_task(
            client.post(f"{TELEGRAM_API}/sendMessage",
                        json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                              "disable_web_page_preview": True})
        )
    except Exception as e:
        print(f"[TG] send error: {e}")

async def tg_broadcast(text: str, parse: str="HTML"):
    # Só canal principal; espelha apenas se REPL estiver ON e com canal válido
    if PUBLIC_CHANNEL:
        await tg_send_text(PUBLIC_CHANNEL, text, parse)
    if REPL_ENABLED and REPL_CHANNEL:
        await tg_send_text(REPL_CHANNEL, text, parse)

async def send_ia_text(text: str, parse: str="HTML"):
    if IA_CHANNEL:
        await tg_send_text(IA_CHANNEL, text, parse)

# Mensagens de resultado
async def send_green_channel(n:int, stage_txt:str="G0"):
    await tg_broadcast(f"✅ <b>GREEN</b> em <b>{stage_txt}</b> — Número: <b>{n}</b>")

async def send_loss_channel(n:int, stage_txt:str="G0"):
    await tg_broadcast(f"❌ <b>LOSS</b> — Número: <b>{n}</b> (em {stage_txt})")

async def send_green_ia(n:int, stage_txt:str="G0"):
    await send_ia_text(f"✅ <b>GREEN</b> em <b>{stage_txt}</b> — Número: <b>{n}</b>")

async def send_loss_ia(n:int, stage_txt:str="G0"):
    await send_ia_text(f"❌ <b>LOSS</b> — Número: <b>{n}</b> (em {stage_txt})")

# =========================
# Timeline & modelos
# =========================
def now_ts() -> int: return int(time.time())
def utc_iso() -> str: return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
def today_key_local() -> str: return datetime.now(timezone.utc).strftime("%Y%m%d")

def append_timeline(n: int):
    exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)", (now_ts(), int(n)))

def get_recent_tail(window: int = WINDOW) -> List[int]:
    rows = query_all("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (window,))
    return [r["number"] for r in rows][::-1]

def update_ngrams(decay: float = DECAY, max_n: int = 5, window: int = WINDOW):
    tail = get_recent_tail(window)
    if len(tail) < 2: return
    for t in range(1, len(tail)):
        nxt = tail[t]; dist = (len(tail)-1) - t; w = (decay ** dist)
        for n in range(2, max_n+1):
            if t-(n-1) < 0: break
            ctx = tail[t-(n-1):t]
            ctx_key = ",".join(str(x) for x in ctx)
            exec_write("""
                INSERT INTO ngram_stats (n, ctx, next, weight)
                VALUES (?,?,?,?)
                ON CONFLICT(n, ctx, next) DO UPDATE SET weight = weight + excluded.weight
            """, (n, ctx_key, int(nxt), float(w)))

def prob_from_ngrams(ctx: List[int], candidate: int) -> float:
    n = len(ctx) + 1
    if n < 2 or n > 5: return 0.0
    ctx_key = ",".join(str(x) for x in ctx)
    row = query_one("SELECT SUM(weight) AS w FROM ngram_stats WHERE n=? AND ctx=?", (n, ctx_key))
    tot = (row["w"] or 0.0) if row else 0.0
    if tot <= 0: return 0.0
    row2 = query_one("SELECT weight FROM ngram_stats WHERE n=? AND ctx=? AND next=?", (n, ctx_key, candidate))
    w = (row2["weight"] or 0.0) if row2 else 0.0
    return w / tot

# =========================
# Parsers do canal
# =========================
MUST_HAVE = (r"ENTRADA\s+CONFIRMADA", r"Mesa:\s*Fantan\s*-\s*Evolution")
MUST_NOT  = (r"\bANALISANDO\b", r"\bPlacar do dia\b", r"\bAPOSTA ENCERRADA\b")

def is_real_entry(text: str) -> bool:
    t = re.sub(r"\s+", " ", text).strip()
    for bad in MUST_NOT:
        if re.search(bad, t, flags=re.I): return False
    for good in MUST_HAVE:
        if not re.search(good, t, flags=re.I): return False
    has_ctx = any(re.search(p, t, flags=re.I) for p in [
        r"Sequ[eê]ncia:\s*[\d\s\|\-]+",
        r"\bKWOK\s*[1-4]\s*-\s*[1-4]",
        r"\bSS?H\s*[1-4](?:-[1-4]){0,3}",
        r"\bODD\b|\bEVEN\b",
        r"Entrar\s+ap[oó]s\s+o\s+[1-4]"
    ])
    return bool(has_ctx)

def extract_strategy(text: str) -> Optional[str]:
    m = re.search(r"Estrat[eé]gia:\s*(\d+)", text, flags=re.I)
    return m.group(1) if m else None

def extract_seq_raw(text: str) -> Optional[str]:
    m = re.search(r"Sequ[eê]ncia:\s*([^\n\r]+)", text, flags=re.I)
    return m.group(1).strip() if m else None

def extract_after_num(text: str) -> Optional[int]:
    m = re.search(r"Entrar\s+ap[oó]s\s+o\s+([1-4])", text, flags=re.I)
    return int(m.group(1)) if m else None

def bases_from_sequence_left_recent(seq_left_recent: List[int], k: int = 3) -> List[int]:
    seen, base = set(), []
    for n in seq_left_recent:
        if n not in seen:
            seen.add(n); base.append(n)
        if len(base) == k: break
    return base

def parse_bases_and_pattern(text: str) -> Tuple[List[int], str]:
    t = re.sub(r"\s+", " ", text).strip()
    m = re.search(r"\bKWOK\s*([1-4])\s*-\s*([1-4])", t, flags=re.I)
    if m: a,b = int(m.group(1)), int(m.group(2)); return [a,b], f"KWOK-{a}-{b}"
    if re.search(r"\bODD\b", t, flags=re.I):  return [1,3], "ODD"
    if re.search(r"\bEVEN\b", t, flags=re.I): return [2,4], "EVEN"
    m = re.search(r"\bSS?H\s*([1-4])(?:-([1-4]))?(?:-([1-4]))?(?:-([1-4]))?", t, flags=re.I)
    if m:
        nums = [int(g) for g in m.groups() if g]
        return nums, "SSH-" + "-".join(str(x) for x in nums)
    m = re.search(r"Sequ[eê]ncia:\s*([\d\s\|\-]+)", t, flags=re.I)
    if m:
        parts = re.findall(r"[1-4]", m.group(1))
        base = bases_from_sequence_left_recent([int(x) for x in parts], 3)
        if base: return base, "SEQ"
    return [], "GEN"

GREEN_PATTERNS = [
    re.compile(r"APOSTA\s+ENCERRADA.*?\bGREEN\b.*?\((\d)\)", re.I | re.S),
    re.compile(r"\bGREEN\b.*?Número[:\s]*([1-4])", re.I | re.S),
    re.compile(r"\bGREEN\b.*?\(([1-4])\)", re.I | re.S),
]
RED_PATTERNS = [
    re.compile(r"APOSTA\s+ENCERRADA.*?\bRED\b.*?\((.*?)\)", re.I | re.S),
    re.compile(r"\bLOSS\b.*?Número[:\s]*([1-4])", re.I | re.S),
    re.compile(r"\bRED\b.*?\(([1-4])\)", re.I | re.S),
]

def extract_green_number(text: str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text)
    for rx in GREEN_PATTERNS:
        m = rx.search(t)
        if m:
            nums = re.findall(r"[1-4]", m.group(1))
            if nums: return int(nums[0])
    return None

def extract_red_last_left(text: str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text)
    for rx in RED_PATTERNS:
        m = rx.search(t)
        if m:
            nums = re.findall(r"[1-4]", m.group(1))
            if nums: return int(nums[0])
    return None

def is_analise(text:str) -> bool:
    return bool(re.search(r"\bANALISANDO\b", text, flags=re.I))

# =========================
# Estatísticas
# =========================
def laplace_ratio(wins:int, losses:int) -> float:
    return (wins + 1.0) / (wins + losses + 2.0)

def bump_pattern(pattern_key: str, number: int, won: bool):
    row = query_one("SELECT wins, losses FROM stats_pattern WHERE pattern_key=? AND number=?", (pattern_key, number))
    w = row["wins"] if row else 0; l = row["losses"] if row else 0
    if won: w += 1
    else:   l += 1
    exec_write("""
      INSERT INTO stats_pattern (pattern_key, number, wins, losses)
      VALUES (?,?,?,?)
      ON CONFLICT(pattern_key, number)
      DO UPDATE SET wins=excluded.wins, losses=excluded.losses
    """, (pattern_key, number, w, l))

def bump_strategy(strategy: str, number: int, won: bool):
    row = query_one("SELECT wins, losses FROM stats_strategy WHERE strategy=? AND number=?", (strategy, number))
    w = row["wins"] if row else 0; l = row["losses"] if row else 0
    if won: w += 1
    else:   l += 1
    exec_write("""
      INSERT INTO stats_strategy (strategy, number, wins, losses)
      VALUES (?,?,?,?)
      ON CONFLICT(strategy, number)
      DO UPDATE SET wins=excluded.wins, losses=excluded.losses
    """, (strategy, number, w, l))

def update_daily_score(stage: Optional[int], won: bool):
    y = today_key_local()
    row = query_one("SELECT g0,g1,g2,loss,streak FROM daily_score WHERE yyyymmdd=?", (y,))
    if not row: g0=g1=g2=loss=streak=0
    else:       g0,g1,g2,loss,streak = row["g0"],row["g1"],row["g2"],row["loss"],row["streak"]

    if SCORE_G0_ONLY:
        if won and stage == 0: g0 += 1; streak += 1
        elif not won:          loss += 1; streak = 0
    else:
        if won:
            if stage == 0: g0 += 1
            elif stage == 1: g1 += 1
            elif stage == 2: g2 += 1
            streak += 1
        else:
            loss += 1; streak = 0

    exec_write("""
      INSERT INTO daily_score (yyyymmdd,g0,g1,g2,loss,streak)
      VALUES (?,?,?,?,?,?)
      ON CONFLICT(yyyymmdd) DO UPDATE SET
        g0=excluded.g0, g1=excluded.g1, g2=excluded.g2, loss=excluded.loss, streak=excluded.streak
    """, (y,g0,g1,g2,loss,streak))

# =========================
# Heurística extra: top-2 da cauda (40)
# =========================
def tail_top2_boost(tail: List[int], k:int=40) -> Dict[int, float]:
    boosts = {1:1.00, 2:1.00, 3:1.00, 4:1.00}
    if not tail: return boosts
    tail_k = tail[-k:] if len(tail) >= k else tail[:]
    c = Counter(tail_k)
    if not c: return boosts
    freq = c.most_common()
    if len(freq) >= 1: boosts[freq[0][0]] = 1.04
    if len(freq) >= 2: boosts[freq[1][0]] = 1.02
    return boosts

# =========================
# Predição
# =========================
def ngram_backoff_score(tail: List[int], after_num: Optional[int], candidate: int) -> float:
    score = 0.0
    if not tail: return 0.0
    if after_num is not None:
        idxs = [i for i,v in enumerate(tail) if v == after_num]
        if not idxs:
            ctx4 = tail[-4:] if len(tail) >= 4 else []
            ctx3 = tail[-3:] if len(tail) >= 3 else []
            ctx2 = tail[-2:] if len(tail) >= 2 else []
            ctx1 = tail[-1:] if len(tail) >= 1 else []
        else:
            i = idxs[-1]
            ctx1 = tail[max(0,i):i+1]
            ctx2 = tail[max(0,i-1):i+1] if i-1>=0 else []
            ctx3 = tail[max(0,i-2):i+1] if i-2>=0 else []
            ctx4 = tail[max(0,i-3):i+1] if i-3>=0 else []
    else:
        ctx4 = tail[-4:] if len(tail) >= 4 else []
        ctx3 = tail[-3:] if len(tail) >= 3 else []
        ctx2 = tail[-2:] if len(tail) >= 2 else []
        ctx1 = tail[-1:] if len(tail) >= 1 else []

    parts = []
    if len(ctx4)==4: parts.append((W4, prob_from_ngrams(ctx4[:-1], candidate)))
    if len(ctx3)==3: parts.append((W3, prob_from_ngrams(ctx3[:-1], candidate)))
    if len(ctx2)==2: parts.append((W2, prob_from_ngrams(ctx2[:-1], candidate)))
    if len(ctx1)==1: parts.append((W1, prob_from_ngrams(ctx1[:-1], candidate)))
    for w,p in parts: score += w*p
    return score

def confident_best(post: Dict[int,float], gap: float = GAP_MIN) -> Optional[int]:
    a = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    if not a: return None
    if len(a)==1: return a[0][0]
    return a[0][0] if (a[0][1]-a[1][1]) >= gap else None

def suggest_number(base: List[int], pattern_key: str, strategy: Optional[str], after_num: Optional[int]) -> Tuple[Optional[int],float,int,Dict[int,float]]:
    if not base: base = [1,2,3,4]
    hour_block = int(datetime.now(timezone.utc).hour // 2)
    pat_key = f"{pattern_key}|h{hour_block}"

    tail = get_recent_tail(WINDOW)
    scores: Dict[int, float] = {}

    if after_num is not None:
        try:
            last_idx = max([i for i, v in enumerate(tail) if v == after_num])
        except ValueError:
            return None, 0.0, len(tail), {k: 1/len(base) for k in base}
        if (len(tail)-1 - last_idx) > 60:
            return None, 0.0, len(tail), {k: 1/len(base) for k in base}

    boost_tail = tail_top2_boost(tail, k=40)

    for c in base:
        ng = ngram_backoff_score(tail, after_num, c)
        rowp = query_one("SELECT wins, losses FROM stats_pattern WHERE pattern_key=? AND number=?", (pat_key, c))
        pw = rowp["wins"] if rowp else 0; pl = rowp["losses"] if rowp else 0
        p_pat = laplace_ratio(pw, pl)

        p_str = 1/len(base)
        if strategy:
            rows = query_one("SELECT wins, losses FROM stats_strategy WHERE strategy=? AND number=?", (strategy, c))
            sw = rows["wins"] if rows else 0; sl = rows["losses"] if rows else 0
            p_str = laplace_ratio(sw, sl)

        boost_pat = 1.05 if p_pat >= 0.60 else (0.97 if p_pat <= 0.40 else 1.00)
        boost_hist = boost_tail.get(c, 1.00)
        prior = 1.0/len(base)
        score = (prior) * ((ng or 1e-6) ** ALPHA) * (p_pat ** BETA) * (p_str ** GAMMA) * boost_pat * boost_hist
        scores[c] = score

    total = sum(scores.values()) or 1e-9
    post = {k: v/total for k,v in scores.items()}
    number = confident_best(post, gap=GAP_MIN)
    conf = post.get(number, 0.0) if number is not None else 0.0

    roww = query_one("SELECT SUM(weight) AS s FROM ngram_stats")
    samples = int((roww["s"] or 0) if roww else 0)

    last = query_one("SELECT suggested, announced FROM pending_outcome ORDER BY id DESC LIMIT 1")
    if last and number is not None and last["suggested"] == number and (last["announced"] or 0) == 1:
        if post.get(number, 0.0) < (MIN_CONF_G0 + 0.08):
            return None, 0.0, samples, post

    top = sorted(post.items(), key=lambda kv: kv[1], reverse=True)[:2]
    if len(top) == 2 and (top[0][1] - top[1][1]) < 0.015:
        return None, 0.0, samples, post

    top2 = sorted(post.items(), key=lambda kv: kv[1], reverse=True)[:2]
    gap = (top2[0][1] - (top2[1][1] if len(top2) > 1 else 0.0)) if top2 else 0.0
    enough_samples = samples >= MIN_SAMPLES
    should_abstain = enough_samples and (
        (number is None) or
        (post.get(number, 0.0) < MIN_CONF_G0) or
        (gap < MIN_GAP_G0)
    )
    if should_abstain:
        return None, 0.0, samples, post

    return number, conf, samples, post

def build_suggestion_msg(number:int, base:List[int], pattern_key:str,
                         after_num:Optional[int], conf:float, samples:int, stage:str="G0") -> str:
    base_txt = ", ".join(str(x) for x in base) if base else "—"
    aft_txt = f" após {after_num}" if after_num else ""
    return (
        f"🎯 <b>Número seco ({stage}):</b> <b>{number}</b>\n"
        f"🧩 <b>Padrão:</b> {pattern_key}{aft_txt}\n"
        f"🧮 <b>Base:</b> [{base_txt}]\n"
        f"📊 Conf: {conf*100:.2f}% | Amostra≈{samples}"
    )

# =========================
# IA loop e controle (globals corrigidos)
# =========================
IA2_TIER_STRICT             = 0.62
IA2_GAP_SAFETY              = 0.08
IA2_DELTA_GAP               = 0.03
IA2_MAX_PER_HOUR            = 30
IA2_COOLDOWN_AFTER_LOSS     = 8
IA2_MIN_SECONDS_BETWEEN_FIRE= 2

_ia2_blocked_until_ts: int = 0
_ia2_sent_this_hour: int = 0
_ia2_hour_bucket: Optional[int] = None
_ia2_last_fire_ts: int = 0

def _ia2_hour_key() -> int: return int(datetime.now(timezone.utc).strftime("%Y%m%d%H"))

def _ia2_reset_hour():
    global _ia2_sent_this_hour, _ia2_hour_bucket
    hb = _ia2_hour_key()
    if _ia2_hour_bucket != hb:
        _ia2_hour_bucket = hb
        _ia2_sent_this_hour = 0

def _ia2_antispam_ok() -> bool:
    _ia2_reset_hour()
    return _ia2_sent_this_hour < IA2_MAX_PER_HOUR

def _ia2_mark_sent():
    global _ia2_sent_this_hour
    _ia2_sent_this_hour += 1

def _ia2_blocked_now() -> bool:
    return now_ts() < _ia2_blocked_until_ts

def _ia_set_post_loss_block():
    global _ia2_blocked_until_ts
    _ia2_blocked_until_ts = now_ts() + int(IA2_COOLDOWN_AFTER_LOSS)

def _ia2_can_fire_now() -> bool:
    if _ia2_blocked_now(): return False
    if not _ia2_antispam_ok(): return False
    if now_ts() - _ia2_last_fire_ts < IA2_MIN_SECONDS_BETWEEN_FIRE: return False
    return True

def _ia2_mark_fire_sent():
    global _ia2_last_fire_ts
    _ia2_mark_sent()
    _ia2_last_fire_ts = now_ts()

async def ia2_send_signal(best:int, conf:float, tail_len:int, mode:str):
    # IA: só IA_CHANNEL
    txt = (
        f"🤖 <b>{SELF_LABEL_IA} [{mode}]</b>\n"
        f"🎯 Número seco (G0): <b>{best}</b>\n"
        f"📈 Conf: <b>{conf*100:.2f}%</b> | Amostra≈<b>{tail_len}</b>"
    )
    await send_ia_text(txt)

def _get_scalar(sql: str, params: tuple = (), default: int|float = 0):
    row = query_one(sql, params)
    if not row: return default
    try: return row[0] if row[0] is not None else default
    except Exception:
        keys = row.keys()
        return row[keys[0]] if keys and row[keys[0]] is not None else default

def _has_open_pending() -> bool:
    if ALWAYS_FREE_FIRE: return False
    open_cnt = int(_get_scalar("SELECT COUNT(*) FROM pending_outcome WHERE open=1"))
    return open_cnt >= 3

class _IntelStub:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(os.path.join(self.base_dir, "snapshots"), exist_ok=True)
    def start_signal(self, suggested: int, strategy: Optional[str] = None):
        try:
            path = os.path.join(self.base_dir, "snapshots", "latest_top.json")
            with open(path, "w") as f:
                json.dump({"ts": now_ts(), "suggested": suggested, "strategy": strategy}, f)
        except Exception as e:
            print(f"[INTEL] snapshot error: {e}")
    def stop_signal(self): pass

INTEL = _IntelStub(INTEL_DIR)

def open_pending(strategy: Optional[str], suggested: int, source: str = "CHAN"):
    exec_write("""
        INSERT INTO pending_outcome
        (created_at,strategy,suggested,stage,open,window_left,seen_numbers,announced,source)
        VALUES (?,?,?,?,1,?, '', 0, ?)
    """, (now_ts(), strategy or "", int(suggested), 0, MAX_STAGE, source))
    try: INTEL.start_signal(suggested=int(suggested), strategy=strategy)
    except Exception as e: print(f"[INTEL] start_signal error: {e}")

async def ia2_process_once():
    # IA avalia em background
    base, pattern_key, strategy, after_num = [], "GEN", None, None
    tail = get_recent_tail(WINDOW)
    best, conf_raw, tail_len, post = suggest_number(base, pattern_key, strategy, after_num)
    if best is None:
        return
    gap = 1.0
    if post and len(post) >= 2:
        top = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
        if len(top) >= 2: gap = top[0][1] - top[1][1]
    if _has_open_pending(): return
    if _ia2_blocked_now(): return
    if not _ia2_antispam_ok(): return
    if now_ts() - _ia2_last_fire_ts < IA2_MIN_SECONDS_BETWEEN_FIRE: return

    # critério
    if (conf_raw >= IA2_TIER_STRICT or (conf_raw >= IA2_TIER_STRICT - IA2_DELTA_GAP and gap >= IA2_GAP_SAFETY)) and tail_len >= MIN_SAMPLES and _ia2_can_fire_now():
        open_pending(strategy, best, source="IA")
        await ia2_send_signal(best, conf_raw, tail_len, "FIRE")
        _ia2_mark_fire_sent()

# =========================
# Fechamento de pendências
# =========================
async def close_pending_with_result(n_observed: int, event_kind: str):
    try:
        rows = query_all("""
            SELECT id, created_at, strategy, suggested, stage, open, window_left,
                   seen_numbers, announced, source
            FROM pending_outcome
            WHERE open=1
            ORDER BY id ASC
        """)
        if not rows: return {"ok": True, "no_open": True}

        for r in rows:
            pid        = r["id"]
            suggested  = int(r["suggested"])
            stage      = int(r["stage"])
            left       = int(r["window_left"])
            src        = (r["source"] or "CHAN").upper()
            seen       = (r["seen_numbers"] or "").strip()
            seen_new   = (seen + ("," if seen else "") + str(int(n_observed)))

            if int(n_observed) == suggested:
                exec_write("UPDATE pending_outcome SET open=0, window_left=0, seen_numbers=? WHERE id=?",
                           (seen_new, pid))
                # vitória
                update_daily_score(0 if stage==0 else stage, True)
                if src == "IA":
                    await send_green_ia(suggested, "G0" if stage==0 else f"G{stage}")
                else:
                    await send_green_channel(suggested, "G0" if stage==0 else f"G{stage}")
            else:
                # avança estágio ou encerra
                if left > 1:
                    exec_write("""
                        UPDATE pending_outcome
                           SET stage = stage + 1, window_left = window_left - 1, seen_numbers=?
                         WHERE id=?
                    """, (seen_new, pid))
                    if stage == 0:
                        update_daily_score(0, False)
                        _ia_set_post_loss_block()
                        if src == "IA":
                            await send_loss_ia(suggested, "G0")
                        else:
                            await send_loss_channel(suggested, "G0")
                else:
                    exec_write("UPDATE pending_outcome SET open=0, window_left=0, seen_numbers=? WHERE id=?",
                               (seen_new, pid))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# =========================
# Webhook models & routes
# =========================
class Update(BaseModel):
    update_id: int
    channel_post: Optional[dict] = None
    message: Optional[dict] = None
    edited_channel_post: Optional[dict] = None
    edited_message: Optional[dict] = None

@app.get("/")
async def root():
    return {"ok": True, "detail": "Use POST /webhook/<WEBHOOK_TOKEN>"}

@app.get("/ping_ia")
async def ping_ia():
    try:
        await send_ia_text("🔧 Teste IA: ok")
        return {"ok": True, "sent": True, "to": IA_CHANNEL}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.on_event("startup")
async def _boot_tasks():
    async def _loop():
        while True:
            try: await ia2_process_once()
            except Exception as e: print(f"[IA2] analyzer error: {e}")
            await asyncio.sleep(max(0.2, INTEL_ANALYZE_INTERVAL))
    try: asyncio.create_task(_loop())
    except Exception as e: print(f"[IA2] startup error: {e}")

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    data = await request.json()
    upd = Update(**data)
    msg = upd.channel_post or upd.message or upd.edited_channel_post or upd.edited_message
    if not msg: return {"ok": True}

    text = (msg.get("text") or msg.get("caption") or "").strip()
    t = re.sub(r"\s+", " ", text)

    # GREEN/RED -> fecha pendências + timeline
    gnum = extract_green_number(t); redn = extract_red_last_left(t)
    if gnum is not None or redn is not None:
        n_observed = gnum if gnum is not None else redn
        append_timeline(n_observed); update_ngrams()
        await close_pending_with_result(n_observed, "GREEN" if gnum is not None else "RED")
        return {"ok": True, "observed": n_observed}

    # ANALISANDO -> apenas timeline
    if is_analise(t):
        seq_raw = extract_seq_raw(t)
        if seq_raw:
            parts = re.findall(r"[1-4]", seq_raw)
            seq_left_recent = [int(x) for x in parts]
            seq_old_to_new  = seq_left_recent[::-1]
            for n in seq_old_to_new: append_timeline(n)
            update_ngrams()
        return {"ok": True, "analise": True}

    # ENTRADA CONFIRMADA -> gera G0 (canal principal)
    if not is_real_entry(t):
        return {"ok": True, "skipped": True}

    source_msg_id = msg.get("message_id")
    if query_one("SELECT 1 FROM suggestions WHERE source_msg_id=?", (source_msg_id,)):
        return {"ok": True, "dup": True}

    strategy  = extract_strategy(t) or ""
    seq_raw   = extract_seq_raw(t) or ""
    after_num = extract_after_num(t)
    base, pattern_key = parse_bases_and_pattern(t)
    if not base: base=[1,2,3,4]; pattern_key="GEN"

    number, conf, samples, post = suggest_number(base, pattern_key, strategy, after_num)
    if number is None:
        return {"ok": True, "skipped_low_conf": True}

    out = build_suggestion_msg(int(number), base, pattern_key, after_num, conf, samples, stage="G0")
    await tg_broadcast(out)   # <- SOMENTE canal principal (e opcional REPL)

    exec_write("""
      INSERT OR REPLACE INTO suggestions
      (source_msg_id, strategy, seq_raw, context_key, pattern_key, base, suggested_number, stage, sent_at)
      VALUES (?,?,?,?,?,?,?,?,?)
    """, (source_msg_id, strategy, seq_raw, "CTX", pattern_key, json.dumps(base), int(number), "G0", now_ts()))
    exec_write("""
      INSERT OR REPLACE INTO last_by_strategy
      (strategy, source_msg_id, suggested_number, context_key, pattern_key, stage, created_at)
      VALUES (?,?,?,?,?,?,?)
    """, (strategy, source_msg_id, int(number), "CTX", pattern_key, "G0", now_ts()))
    open_pending(strategy, int(number), source="CHAN")

    return {"ok": True, "sent": True, "number": number, "conf": conf, "samples": samples}