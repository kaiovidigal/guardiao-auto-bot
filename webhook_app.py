# -*- coding: utf-8 -*-
# Fan Tan — Guardião (G0 + Recuperação oculta) — enxuto
# - IA com warm-up em 0% e thresholds dinâmicos (sobem com aprendizado)
# - Cap de 100% na exibição de confiança
# - Boost leve do top-2 da cauda (40)
# - GREEN/RED: salva o ÚLTIMO número entre parênteses (p.ex. "(1|4|2)" -> 2)
# - ANALISANDO: só registra sequência, sem sinal
#
# Rotas:
#   POST /webhook/<WEBHOOK_TOKEN>
#   GET  /debug/state, /debug/reason, /debug/samples, /debug/flush

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
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
PUBLIC_CHANNEL = os.getenv("PUBLIC_CHANNEL", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
FLUSH_KEY      = os.getenv("FLUSH_KEY", "meusegredo123").strip()

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN:
    print("⚠️ Defina TG_BOT_TOKEN e WEBHOOK_TOKEN.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
REPL_ENABLED, REPL_CHANNEL = True, "-1003052132833"  # espelho

# =========================
# MODO & Hiperparâmetros
# =========================
MAX_STAGE     = 3
FAST_RESULT   = True
SCORE_G0_ONLY = True

WINDOW = 400
DECAY  = 0.985
W4, W3, W2, W1 = 0.38, 0.30, 0.20, 0.12
ALPHA, BETA, GAMMA = 1.05, 0.70, 0.40
GAP_MIN = 0.08

MIN_CONF_G0 = 0.55
MIN_GAP_G0  = 0.04
MIN_SAMPLES = 1000

# IA em disco
INTEL_DIR = os.getenv("INTEL_DIR", "/var/data/ai_intel").rstrip("/")
INTEL_MAX_BYTES = int(os.getenv("INTEL_MAX_BYTES", "1000000000"))
INTEL_SIGNAL_INTERVAL = float(os.getenv("INTEL_SIGNAL_INTERVAL", "20"))
INTEL_ANALYZE_INTERVAL = float(os.getenv("INTEL_ANALYZE_INTERVAL", "2"))

SELF_LABEL_IA = os.getenv("SELF_LABEL_IA", "Tiro seco por IA")

# Caps e warm-up
CONF_CAP, GAP_CAP = 0.999, 1.0       # exibição/proteção
WARMUP_SAMPLES    = 5_000            # até aqui, thresholds começam em 0% e sobem

app = FastAPI(title="Fantan Guardião — FIRE-only (G0 + Recuperação oculta)", version="3.11.0")

# =========================
# DB helpers
# =========================
OLD_DB_CANDIDATES = ["/var/data/data.db","/opt/render/project/src/data.db",
                     "/opt/render/project/src/data/data.db","/data/data.db"]
def _ensure_db_dir():
    try: os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    except Exception as e: print(f"[DB] mkdir: {e}")

def _migrate_old_db_if_needed():
    if os.path.exists(DB_PATH): return
    for src in OLD_DB_CANDIDATES:
        if os.path.exists(src):
            try: _ensure_db_dir(); shutil.copy2(src, DB_PATH); print(f"[DB] Migrado {src} -> {DB_PATH}"); return
            except Exception as e: print(f"[DB] Migração falhou {src}: {e}")

_ensure_db_dir(); _migrate_old_db_if_needed()

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

def query_all(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    con = _connect(); rows = con.execute(sql, params).fetchall(); con.close(); return rows

def query_one(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    con = _connect(); row = con.execute(sql, params).fetchone(); con.close(); return row

def init_db():
    con = _connect(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL, number INTEGER NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS ngram_stats (
        n INTEGER NOT NULL, ctx TEXT NOT NULL, next INTEGER NOT NULL, weight REAL NOT NULL,
        PRIMARY KEY (n, ctx, next))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS stats_pattern (
        pattern_key TEXT NOT NULL, number INTEGER NOT NULL, wins INTEGER NOT NULL DEFAULT 0, losses INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (pattern_key, number))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS stats_strategy (
        strategy TEXT NOT NULL, number INTEGER NOT NULL, wins INTEGER NOT NULL DEFAULT 0, losses INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (strategy, number))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS suggestions (
        source_msg_id INTEGER PRIMARY KEY, strategy TEXT, seq_raw TEXT, context_key TEXT, pattern_key TEXT, base TEXT,
        suggested_number INTEGER, stage TEXT, sent_at INTEGER)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS last_by_strategy (
        strategy TEXT PRIMARY KEY, source_msg_id INTEGER, suggested_number INTEGER,
        context_key TEXT, pattern_key TEXT, stage TEXT, created_at INTEGER)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS daily_score (
        yyyymmdd TEXT PRIMARY KEY, g0 INTEGER NOT NULL DEFAULT 0, g1 INTEGER NOT NULL DEFAULT 0,
        g2 INTEGER NOT NULL DEFAULT 0, loss INTEGER NOT NULL DEFAULT 0, streak INTEGER NOT NULL DEFAULT 0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pending_outcome (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL, strategy TEXT, suggested INTEGER NOT NULL,
        stage INTEGER NOT NULL, open INTEGER NOT NULL, window_left INTEGER NOT NULL, seen_numbers TEXT DEFAULT '',
        announced INTEGER NOT NULL DEFAULT 0, source TEXT NOT NULL DEFAULT 'CHAN')""")
    cur.execute("""CREATE TABLE IF NOT EXISTS daily_score_ia (
        yyyymmdd TEXT PRIMARY KEY, g0 INTEGER NOT NULL DEFAULT 0, loss INTEGER NOT NULL DEFAULT 0, streak INTEGER NOT NULL DEFAULT 0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS recov_g1_stats (
        number INTEGER PRIMARY KEY, wins INTEGER NOT NULL DEFAULT 0, losses INTEGER NOT NULL DEFAULT 0)""")
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
# Utils / Telegram
# =========================
def now_ts() -> int: return int(time.time())
def utc_iso() -> str: return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
def today_key_local() -> str: return datetime.now(timezone.utc).strftime("%Y%m%d")

async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN or not chat_id: return
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse, "disable_web_page_preview": True})

async def tg_broadcast(text: str, parse: str="HTML"):
    if REPL_ENABLED and REPL_CHANNEL: await tg_send_text(REPL_CHANNEL, text, parse)

async def send_green_imediato(sugerido:int, stage_txt:str="G0"):
    await tg_broadcast(f"✅ <b>GREEN</b> em <b>{stage_txt}</b> — Número: <b>{sugerido}</b>")

async def send_loss_imediato(sugerido:int, stage_txt:str="G0"):
    await tg_broadcast(f"❌ <b>LOSS</b> — Número: <b>{sugerido}</b> (em {stage_txt})")

async def ia2_send_signal(best:int, conf:float, tail_len:int, mode:str):
    conf_capped = max(0.0, min(float(conf), CONF_CAP))
    txt = (f"🤖 <b>{SELF_LABEL_IA} [{mode}]</b>\n"
           f"🎯 Número seco (G0): <b>{best}</b>\n"
           f"📈 Conf: <b>{conf_capped*100:.2f}%</b> | Amostra≈<b>{tail_len}</b>")
    await tg_broadcast(txt)

# =========================
# Timeline & n-grams
# =========================
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
    tot = (query_one("SELECT SUM(weight) AS w FROM ngram_stats WHERE n=? AND ctx=?", (n, ctx_key)) or {"w":0})["w"] or 0.0
    if tot <= 0: return 0.0
    w = (query_one("SELECT weight FROM ngram_stats WHERE n=? AND ctx=? AND next=?", (n, ctx_key, candidate)) or {"weight":0})["weight"] or 0.0
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
    m = re.search(r"Estrat[eé]gia:\s*(\d+)", text, flags=re.I); return m.group(1) if m else None

def extract_seq_raw(text: str) -> Optional[str]:
    m = re.search(r"Sequ[eê]ncia:\s*([^\n\r]+)", text, flags=re.I); return m.group(1).strip() if m else None

def extract_after_num(text: str) -> Optional[int]:
    m = re.search(r"Entrar\s+ap[oó]s\s+o\s+([1-4])", text, flags=re.I); return int(m.group(1)) if m else None

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
        nums = [int(g) for g in m.groups() if g]; return nums, "SSH-" + "-".join(str(x) for x in nums)
    m = re.search(r"Sequ[eê]ncia:\s*([\d\s\|\-]+)", t, flags=re.I)
    if m:
        parts = re.findall(r"[1-4]", m.group(1)); base = bases_from_sequence_left_recent([int(x) for x in parts], 3)
        if base: return base, "SEQ"
    return [], "GEN"

# GREEN/RED — salva o ÚLTIMO número entre parênteses
GREEN_PATTERNS = [
    re.compile(r"APOSTA\s+ENCERRADA.*?\bGREEN\b.*?\((.*?)\)", re.I | re.S),
    re.compile(r"\bGREEN\b.*?\((.*?)\)", re.I | re.S),
    re.compile(r"\bGREEN\b.*?Número[:\s]*([1-4])", re.I | re.S),
]
RED_PATTERNS = [
    re.compile(r"APOSTA\s+ENCERRADA.*?\bRED\b.*?\((.*?)\)", re.I | re.S),
    re.compile(r"\bRED\b.*?\((.*?)\)", re.I | re.S),
    re.compile(r"\bLOSS\b.*?Número[:\s]*([1-4])", re.I | re.S),
]

def _last_num_in_group(g: str) -> Optional[int]:
    if not g: return None
    nums = re.findall(r"[1-4]", g)
    return int(nums[-1]) if nums else None

def extract_green_number(text: str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text)
    for rx in GREEN_PATTERNS:
        m = rx.search(t)
        if m:
            # se capturou bloco "(1 | 4 | 2)" pega o último; se foi "Número: 3", pega direto
            g = m.group(1) if m.lastindex else None
            return _last_num_in_group(g) if g is not None else _last_num_in_group(m.group(0))
    return None

def extract_red_last_left(text: str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text)
    for rx in RED_PATTERNS:
        m = rx.search(t)
        if m:
            g = m.group(1) if m.lastindex else None
            return _last_num_in_group(g) if g is not None else _last_num_in_group(m.group(0))
    return None

def is_analise(text:str) -> bool:
    return bool(re.search(r"\bANALISANDO\b", text, flags=re.I))

# =========================
# Placar e métricas
# =========================
def laplace_ratio(wins:int, losses:int) -> float: return (wins + 1.0) / (wins + losses + 2.0)

def bump_pattern(pattern_key: str, number: int, won: bool):
    row = query_one("SELECT wins, losses FROM stats_pattern WHERE pattern_key=? AND number=?", (pattern_key, number))
    w = (row["wins"] if row else 0) + (1 if won else 0)
    l = (row["losses"] if row else 0) + (0 if won else 1)
    exec_write("""
      INSERT INTO stats_pattern (pattern_key, number, wins, losses)
      VALUES (?,?,?,?)
      ON CONFLICT(pattern_key, number) DO UPDATE SET wins=excluded.wins, losses=excluded.losses
    """, (pattern_key, number, w, l))

def bump_strategy(strategy: str, number: int, won: bool):
    row = query_one("SELECT wins, losses FROM stats_strategy WHERE strategy=? AND number=?", (strategy, number))
    w = (row["wins"] if row else 0) + (1 if won else 0)
    l = (row["losses"] if row else 0) + (0 if won else 1)
    exec_write("""
      INSERT INTO stats_strategy (strategy, number, wins, losses)
      VALUES (?,?,?,?)
      ON CONFLICT(strategy, number) DO UPDATE SET wins=excluded.wins, losses=excluded.losses
    """, (strategy, number, w, l))

def update_daily_score(stage: Optional[int], won: bool):
    y = today_key_local()
    row = query_one("SELECT g0,g1,g2,loss,streak FROM daily_score WHERE yyyymmdd=?", (y,))
    g0,g1,g2,loss,streak = (row["g0"],row["g1"],row["g2"],row["loss"],row["streak"]) if row else (0,0,0,0,0)
    if SCORE_G0_ONLY:
        if won and stage == 0: g0 += 1; streak += 1
        elif not won: loss += 1; streak = 0
    else:
        if won:
            (g0,g1,g2)[stage] += 1  # não usado, mantido p/compat
            streak += 1
        else: loss += 1; streak = 0
    exec_write("""INSERT INTO daily_score (yyyymmdd,g0,g1,g2,loss,streak) VALUES (?,?,?,?,?,?)
                   ON CONFLICT(yyyymmdd) DO UPDATE SET g0=excluded.g0,g1=excluded.g1,g2=excluded.g2,loss=excluded.loss,streak=excluded.streak""",
               (y,g0,g1,g2,loss,streak))
    total = g0 + loss
    acc = (g0/total) if total else 0.0
    return g0, loss, acc, streak

def update_daily_score_ia(stage: Optional[int], won: bool):
    y = today_key_local()
    row = query_one("SELECT g0,loss,streak FROM daily_score_ia WHERE yyyymmdd=?", (y,))
    g0,loss,streak = (row["g0"],row["loss"],row["streak"]) if row else (0,0,0)
    if won and stage == 0: g0 += 1; streak += 1
    elif not won: loss += 1; streak = 0
    exec_write("""INSERT INTO daily_score_ia (yyyymmdd,g0,loss,streak) VALUES (?,?,?,?)
                   ON CONFLICT(yyyymmdd) DO UPDATE SET g0=excluded.g0,loss=excluded.loss,streak=excluded.streak""",
               (y,g0,loss,streak))
    total = g0 + loss
    acc = (g0/total) if total else 0.0
    return g0, loss, acc, streak

def _convert_last_loss_to_green():
    y = today_key_local()
    row = query_one("SELECT g0,g1,g2,loss,streak FROM daily_score WHERE yyyymmdd=?", (y,))
    if not row: exec_write("""INSERT OR REPLACE INTO daily_score (yyyymmdd,g0,g1,g2,loss,streak) VALUES (?,0,0,0,0,0)""",(y,)); return
    g0,loss,streak = row["g0"] or 0, row["loss"] or 0, row["streak"] or 0
    if loss>0: loss-=1; g0+=1; streak+=1
    exec_write("""INSERT OR REPLACE INTO daily_score (yyyymmdd,g0,g1,g2,loss,streak) VALUES (?,?,?,?,?,?)""",
               (y,g0,row["g1"] or 0,row["g2"] or 0,loss,streak))

def _convert_last_loss_to_green_ia():
    y = today_key_local()
    row = query_one("SELECT g0,loss,streak FROM daily_score_ia WHERE yyyymmdd=?", (y,))
    if not row: exec_write("""INSERT OR REPLACE INTO daily_score_ia (yyyymmdd,g0,loss,streak) VALUES (?,0,0,0)""",(y,)); return
    g0,loss,streak = row["g0"] or 0, row["loss"] or 0, row["streak"] or 0
    if loss>0: loss-=1; g0+=1; streak+=1
    exec_write("""INSERT OR REPLACE INTO daily_score_ia (yyyymmdd,g0,loss,streak) VALUES (?,?,?,?)""",(y,g0,loss,streak))

def _ia_acc_days(days:int=7) -> Tuple[int,int,float]:
    days = max(1, min(int(days), 30))
    rows = query_all("""SELECT g0, loss FROM daily_score_ia ORDER BY yyyymmdd DESC LIMIT ?""",(days,))
    g = sum((r["g0"] or 0) for r in rows); l = sum((r["loss"] or 0) for r in rows)
    return int(g), int(l), (g/(g+l)) if (g+l)>0 else 0.0

# =========================
# Health + debug
# =========================
def _fmt_bytes(n: int) -> str:
    try: n=float(n)
    except: return "—"
    for u in ["B","KB","MB","GB","TB","PB"]:
        if n<1024.0: return f"{n:.1f} {u}"
        n/=1024.0
    return f"{n:.1f} EB"

def _dir_size_bytes(path: str) -> int:
    total=0
    try:
        for root,_d,files in os.walk(path):
            for f in files:
                fp=os.path.join(root,f)
                try: total+=os.path.getsize(fp)
                except: pass
    except: return 0
    return total

def _get_scalar(sql: str, params: tuple = (), default: int|float = 0):
    row = query_one(sql, params)
    if not row: return default
    try: return row[0] if row[0] is not None else default
    except: keys=row.keys(); return row[keys[0]] if keys and row[keys[0]] is not None else default

# ===== Fases adaptativas =====
IA2_TIER_STRICT             = 0.62
IA2_GAP_SAFETY              = 0.08
IA2_DELTA_GAP               = 0.03
IA2_MAX_PER_HOUR            = 10
IA2_COOLDOWN_AFTER_LOSS     = 20
IA2_MIN_SECONDS_BETWEEN_FIRE= 15

_S_A_MAX=20_000; _S_B_MAX=100_000
_ACC_TO_B=0.55; _ACC_TO_C=0.58; _ACC_FALL_B=0.52; _ACC_FALL_C=0.55
_PHASE_DWELL=60*60
_phase_name, _phase_since_ts = "A", 0

def _current_samples() -> int:
    row = query_one("SELECT SUM(weight) AS s FROM ngram_stats"); return int((row["s"] or 0) if row else 0)

def _decide_phase(samples:int, acc7:float) -> str:
    if samples < _S_A_MAX: return "A"
    if samples < _S_B_MAX: return "B" if acc7>=_ACC_TO_B else "A"
    if acc7 >= _ACC_TO_C: return "C"
    if acc7 >= _ACC_TO_B: return "B"
    return "A" if acc7 < _ACC_FALL_B else "B"

def _adaptive_phase() -> str:
    global _phase_name, _phase_since_ts
    s=_current_samples(); _g,_l,acc7=_ia_acc_days(7); target=_decide_phase(s,acc7)
    now=now_ts()
    if _phase_since_ts==0: _phase_name, _phase_since_ts = target, now; return _phase_name
    if target==_phase_name: return _phase_name
    if target>_phase_name:
        if now-_phase_since_ts < _PHASE_DWELL: return _phase_name
        _phase_name,_phase_since_ts=target,now; return _phase_name
    if _phase_name=="C" and acc7<_ACC_FALL_C: _phase_name,_phase_since_ts="B",now; return _phase_name
    if _phase_name=="B" and acc7<_ACC_FALL_B: _phase_name,_phase_since_ts="A",now; return _phase_name
    if now-_phase_since_ts < _PHASE_DWELL: return _phase_name
    _phase_name,_phase_since_ts=target,now; return _phase_name

def _eff() -> dict:
    p=_adaptive_phase()
    if p=="A":
        return {"PHASE":"A","MAX_CONCURRENT_PENDINGS":2,"IA2_MAX_PER_HOUR":30,"IA2_MIN_SECONDS_BETWEEN_FIRE":7,
                "IA2_COOLDOWN_AFTER_LOSS":10,"IA2_TIER_STRICT":0.59,"IA2_DELTA_GAP":0.04,"IA2_GAP_SAFETY":IA2_GAP_SAFETY,
                "MIN_CONF_G0":0.52,"MIN_GAP_G0":0.035}
    if p=="B":
        return {"PHASE":"B","MAX_CONCURRENT_PENDINGS":2,"IA2_MAX_PER_HOUR":20,"IA2_MIN_SECONDS_BETWEEN_FIRE":10,
                "IA2_COOLDOWN_AFTER_LOSS":12,"IA2_TIER_STRICT":0.60,"IA2_DELTA_GAP":0.04,"IA2_GAP_SAFETY":IA2_GAP_SAFETY,
                "MIN_CONF_G0":0.53,"MIN_GAP_G0":0.035}
    return {"PHASE":"C","MAX_CONCURRENT_PENDINGS":1,"IA2_MAX_PER_HOUR":12,"IA2_MIN_SECONDS_BETWEEN_FIRE":12,
            "IA2_COOLDOWN_AFTER_LOSS":15,"IA2_TIER_STRICT":0.62,"IA2_DELTA_GAP":0.03,"IA2_GAP_SAFETY":IA2_GAP_SAFETY,
            "MIN_CONF_G0":0.55,"MIN_GAP_G0":0.040}

# ===== Razões / health text
_ia2_last_reason, _ia2_last_reason_ts = "—", 0
def _mark_reason(r: str):
    global _ia2_last_reason,_ia2_last_reason_ts
    _ia2_last_reason, _ia2_last_reason_ts = r, now_ts()

def _daily_score_snapshot():
    y=today_key_local()
    row=query_one("SELECT g0,loss,streak FROM daily_score WHERE yyyymmdd=?", (y,))
    g0=(row["g0"] if row else 0); loss=(row["loss"] if row else 0); streak=(row["streak"] if row else 0)
    total=g0+loss; acc=(g0/total*100.0) if total else 0.0
    return g0, loss, streak, total, acc

def _daily_score_snapshot_ia():
    y=today_key_local()
    row=query_one("SELECT g0,loss,streak FROM daily_score_ia WHERE yyyymmdd=?", (y,))
    g0=(row["g0"] if row else 0); loss=(row["loss"] if row else 0); total=g0+loss; acc=(g0/total*100.0) if total else 0.0
    return g0, loss, total, acc

def _last_snapshot_info():
    try:
        p=os.path.join(INTEL_DIR,"snapshots","latest_top.json")
        if not os.path.exists(p): return "—"
        ts=os.path.getmtime(p); return datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc).isoformat()
    except: return "—"

def _health_text() -> str:
    timeline_cnt=_get_scalar("SELECT COUNT(*) FROM timeline")
    ngram_rows=_get_scalar("SELECT COUNT(*) FROM ngram_stats")
    ngram_samples=_get_scalar("SELECT SUM(weight) FROM ngram_stats")
    pat_rows=_get_scalar("SELECT COUNT(*) FROM stats_pattern")
    pat_events=_get_scalar("SELECT SUM(wins+losses) FROM stats_pattern")
    strat_rows=_get_scalar("SELECT COUNT(*) FROM stats_strategy")
    strat_events=_get_scalar("SELECT SUM(wins+losses) FROM stats_strategy")
    pend_open=_get_scalar("SELECT COUNT(*) FROM pending_outcome WHERE open=1")
    intel_size=_dir_size_bytes(INTEL_DIR)
    last_snap_ts=_last_snapshot_info()
    g0,loss,streak,total,acc=_daily_score_snapshot()
    ia_g0,ia_loss,ia_total,ia_acc=_daily_score_snapshot_ia()
    try: last_reason_age=max(0, now_ts()-(_ia2_last_reason_ts or now_ts())); last_line=f"🤖 Motivo último NÃO-FIRE/FIRE: {_ia2_last_reason} (há {last_reason_age}s)"
    except: last_line="🤖 Motivo último NÃO-FIRE/FIRE: —"
    try:
        eff=_eff(); _g7,_l7,acc7=_ia_acc_days(7)
        phase_line=(f"🧭 Fase: {eff['PHASE']} | Max/h: {eff['IA2_MAX_PER_HOUR']} | Conc.: {eff['MAX_CONCURRENT_PENDINGS']} | "
                    f"Conf≥{eff['MIN_CONF_G0']:.2f}/Gap≥{eff['MIN_GAP_G0']:.3f} | IA 7d: {(_g7+_l7)} ops • {acc7*100:.2f}%")
    except: phase_line="🧭 Fase: —"
    return ("🩺 <b>Saúde do Guardião</b>\n"
            f"⏱️ UTC: <code>{utc_iso()}</code>\n—\n"
            f"🗄️ timeline: <b>{timeline_cnt}</b>\n"
            f"📚 ngram_stats: <b>{ngram_rows}</b> | amostras≈<b>{int(ngram_samples or 0)}</b>\n"
            f"🧩 stats_pattern: chaves=<b>{pat_rows}</b> | eventos=<b>{int(pat_events or 0)}</b>\n"
            f"🧠 stats_strategy: chaves=<b>{strat_rows}</b> | eventos=<b>{int(strat_events or 0)}</b>\n"
            f"⏳ pendências abertas: <b>{pend_open}</b>\n"
            f"💾 INTEL dir: <b>{_fmt_bytes(intel_size)}</b> | último snapshot: <code>{last_snap_ts}</code>\n—\n"
            f"📊 Placar (hoje - G0 only): G0=<b>{g0}</b> | Loss=<b>{loss}</b> | Total=<b>{total}</b> | ✅ {acc:.2f}% | 🔥 {streak}\n"
            f"🤖 IA: G0=<b>{ia_g0}</b> | Loss=<b>{ia_loss}</b> | Total=<b>{ia_total}</b> | ✅ {ia_acc:.2f}%\n"
            f"{last_line}\n{phase_line}\n")

async def _health_reporter_task():
    while True:
        try: await tg_broadcast(_health_text()); 
        except Exception as e: print(f"[HEALTH] erro: {e}")
        await asyncio.sleep(30*60)

async def _daily_reset_task():
    while True:
        try:
            now=datetime.now(timezone.utc)
            tomorrow=(now+timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0)
            await asyncio.sleep(max(1.0,(tomorrow-now).total_seconds()))
            y=today_key_local()
            exec_write("""INSERT OR REPLACE INTO daily_score (yyyymmdd,g0,g1,g2,loss,streak) VALUES (?,0,0,0,0,0)""",(y,))
            exec_write("""INSERT OR REPLACE INTO daily_score_ia (yyyymmdd,g0,loss,streak) VALUES (?,0,0,0)""",(y,))
            await tg_broadcast("🕛 <b>Reset diário (00:00 UTC)</b> — placares zerados.")
        except Exception as e: print(f"[RESET] erro: {e}"); await asyncio.sleep(60)

# =========================
# Top-2 cauda (40) — boost leve
# =========================
def tail_top2_boost(tail: List[int], k:int=40) -> Dict[int, float]:
    boosts={1:1.00,2:1.00,3:1.00,4:1.00}
    if not tail: return boosts
    tail_k = tail[-k:] if len(tail)>=k else tail[:]
    c=Counter(tail_k); freq=c.most_common()
    if len(freq)>=1: boosts[freq[0][0]]=1.04
    if len(freq)>=2: boosts[freq[1][0]]=1.02
    return boosts

# =========================
# Predição (número seco)
# =========================
def ngram_backoff_score(tail: List[int], after_num: Optional[int], candidate: int) -> float:
    if not tail: return 0.0
    if after_num is not None:
        idxs=[i for i,v in enumerate(tail) if v==after_num]
        if not idxs:
            ctx4=tail[-4:] if len(tail)>=4 else []; ctx3=tail[-3:] if len(tail)>=3 else []
            ctx2=tail[-2:] if len(tail)>=2 else [];  ctx1=tail[-1:] if len(tail)>=1 else []
        else:
            i=idxs[-1]
            ctx1=tail[max(0,i):i+1]; ctx2=tail[max(0,i-1):i+1] if i-1>=0 else []
            ctx3=tail[max(0,i-2):i+1] if i-2>=0 else []; ctx4=tail[max(0,i-3):i+1] if i-3>=0 else []
    else:
        ctx4=tail[-4:] if len(tail)>=4 else []; ctx3=tail[-3:] if len(tail)>=3 else []
        ctx2=tail[-2:] if len(tail)>=2 else [];  ctx1=tail[-1:] if len(tail)>=1 else []
    parts=[]
    if len(ctx4)==4: parts.append((W4, prob_from_ngrams(ctx4[:-1], candidate)))
    if len(ctx3)==3: parts.append((W3, prob_from_ngrams(ctx3[:-1], candidate)))
    if len(ctx2)==2: parts.append((W2, prob_from_ngrams(ctx2[:-1], candidate)))
    if len(ctx1)==1: parts.append((W1, prob_from_ngrams(ctx1[:-1], candidate)))
    return sum(w*p for w,p in parts)

def confident_best(post: Dict[int,float], gap: float = GAP_MIN) -> Optional[int]:
    a=sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    if not a: return None
    if len(a)==1: return a[0][0]
    return a[0][0] if (a[0][1]-a[1][1]) >= gap else None

def suggest_number(base: List[int], pattern_key: str, strategy: Optional[str], after_num: Optional[int]) -> Tuple[Optional[int],float,int,Dict[int,float]]:
    if not base: base=[1,2,3,4]
    hour_block=int(datetime.now(timezone.utc).hour // 2)
    pat_key=f"{pattern_key}|h{hour_block}"

    tail=get_recent_tail(WINDOW); scores: Dict[int,float]={}

    if after_num is not None:
        try: last_idx=max([i for i,v in enumerate(tail) if v==after_num])
        except ValueError: return None, 0.0, len(tail), {k:1/len(base) for k in base}
        if (len(tail)-1 - last_idx) > 60: return None, 0.0, len(tail), {k:1/len(base) for k in base}

    boost_tail = tail_top2_boost(tail, k=40)

    for c in base:
        ng = ngram_backoff_score(tail, after_num, c)
        rowp = query_one("SELECT wins, losses FROM stats_pattern WHERE pattern_key=? AND number=?", (pat_key, c))
        pw,pl = (rowp["wins"] if rowp else 0),(rowp["losses"] if rowp else 0)
        p_pat = laplace_ratio(pw, pl)

        p_str = 1/len(base)
        if strategy:
            rows = query_one("SELECT wins, losses FROM stats_strategy WHERE strategy=? AND number=?", (strategy, c))
            sw,sl = (rows["wins"] if rows else 0),(rows["losses"] if rows else 0)
            p_str = laplace_ratio(sw, sl)

        boost_pat = 1.05 if p_pat>=0.60 else (0.97 if p_pat<=0.40 else 1.00)
        prior = 1.0/len(base)
        score = prior * ((ng or 1e-6) ** ALPHA) * (p_pat ** BETA) * (p_str ** GAMMA) * boost_pat * boost_tail.get(c,1.00)
        scores[c]=score

    total=sum(scores.values()) or 1e-9
    post={k:v/total for k,v in scores.items()}
    number=confident_best(post, gap=GAP_MIN)
    conf = post.get(number, 0.0) if number is not None else 0.0

    roww=query_one("SELECT SUM(weight) AS s FROM ngram_stats"); samples=int((roww["s"] or 0) if roww else 0)

    # anti-repique
    last = query_one("SELECT suggested, announced FROM pending_outcome ORDER BY id DESC LIMIT 1")
    if last and number is not None and last["suggested"]==number and (last["announced"] or 0)==1:
        effx=_eff()
        if post.get(number,0.0) < (effx["MIN_CONF_G0"] + 0.08): return None, 0.0, samples, post

    # paraquedas: evita empate técnico
    top=sorted(post.items(), key=lambda kv: kv[1], reverse=True)[:2]
    if len(top)==2 and (top[0][1]-top[1][1])<0.015: return None, 0.0, samples, post

    # filtros dinâmicos por fase aplicam depois
    return number, conf, samples, post

# =========================
# IA2 — processo (FIRE-only + fila única)
# =========================
IA2_BLOCK_UNTIL=0; IA2_SENT_THIS_HOUR=0; IA2_HOUR_BUCKET=None; IA2_LAST_FIRE_TS=0
def _ia2_hour_key() -> int: return int(datetime.now(timezone.utc).strftime("%Y%m%d%H"))
def _ia2_reset_hour():
    global IA2_SENT_THIS_HOUR, IA2_HOUR_BUCKET
    hb=_ia2_hour_key()
    if IA2_HOUR_BUCKET!=hb: IA2_HOUR_BUCKET=hb; IA2_SENT_THIS_HOUR=0
def _ia2_antispam_ok() -> bool:
    eff=_eff(); _ia2_reset_hour(); return IA2_SENT_THIS_HOUR < eff["IA2_MAX_PER_HOUR"]
def _ia2_mark_sent():
    global IA2_SENT_THIS_HOUR; IA2_SENT_THIS_HOUR += 1
def _ia2_blocked_now() -> bool: return now_ts() < IA2_BLOCK_UNTIL
def _ia_set_post_loss_block():
    global IA2_BLOCK_UNTIL; eff=_eff(); IA2_BLOCK_UNTIL = now_ts() + int(eff["IA2_COOLDOWN_AFTER_LOSS"])
def _has_open_pending() -> bool:
    eff=_eff(); open_cnt=int(_get_scalar("SELECT COUNT(*) FROM pending_outcome WHERE open=1")); return open_cnt >= eff["MAX_CONCURRENT_PENDINGS"]
def _ia2_can_fire_now() -> bool:
    eff=_eff()
    if _ia2_blocked_now(): return False
    if not _ia2_antispam_ok(): return False
    if now_ts() - (IA2_LAST_FIRE_TS or 0) < eff["IA2_MIN_SECONDS_BETWEEN_FIRE"]: return False
    return True
def _ia2_mark_fire_sent():
    global IA2_LAST_FIRE_TS; _ia2_mark_sent(); IA2_LAST_FIRE_TS=now_ts()

def _dyn_thresholds() -> tuple[float, float]:
    """ thresholds dinâmicos: começam em 0%/0.0 e sobem com acc(7d)+amostra, respeitando a fase """
    eff=_eff()
    g7,l7,acc7=_ia_acc_days(7); samples=_current_samples()
    if samples < WARMUP_SAMPLES:
        base_conf, base_gap = 0.00, 0.000
    else:
        base_conf = 0.50 + min(0.10, acc7*0.12) + min(0.05, (samples/200_000.0)*0.05)
        base_gap  = 0.030 + min(0.02, acc7*0.025) + min(0.010,(samples/100_000.0)*0.010)
    min_conf = max(base_conf, eff["MIN_CONF_G0"]); min_gap = max(base_gap, eff["MIN_GAP_G0"])
    min_conf = min(min_conf, 0.68); min_gap = min(min_gap, 0.060)   # caps de endurecimento
    min_conf = max(0.0, min(float(min_conf), CONF_CAP)); min_gap = max(0.0, min(float(min_gap), GAP_CAP))
    return round(min_conf,3), round(min_gap,3)

def _relevance_ok(conf_raw: float, gap: float, tail_len: int) -> bool:
    min_conf_dyn, min_gap_dyn = _dyn_thresholds()
    conf_c = max(0.0, min(float(conf_raw), CONF_CAP))
    gap_c  = max(0.0, min(float(gap),      GAP_CAP))
    return (conf_c >= min_conf_dyn) and (gap_c >= min_gap_dyn) and (tail_len >= MIN_SAMPLES)

# Pendências
def open_pending(strategy: Optional[str], suggested: int, source: str = "CHAN"):
    exec_write("""INSERT INTO pending_outcome (created_at,strategy,suggested,stage,open,window_left,seen_numbers,announced,source)
                  VALUES (?,?,?,?,1,?, '', 0, ?)""", (now_ts(), strategy or "", int(suggested), 0, MAX_STAGE, source))

async def close_pending_with_result(n_observed: int, _event_kind: str):
    rows=query_all("""SELECT id, strategy, suggested, stage, window_left, seen_numbers, announced, source
                      FROM pending_outcome WHERE open=1 ORDER BY id ASC""")
    if not rows: return {"ok": True, "no_open": True}
    for r in rows:
        pid, suggested, stage, left = r["id"], int(r["suggested"]), int(r["stage"]), int(r["window_left"])
        src = (r["source"] or "CHAN").upper()
        seen_new = ((r["seen_numbers"] or "") + ("," if (r["seen_numbers"] or "") else "") + str(int(n_observed))).strip()
        if int(n_observed)==suggested:
            exec_write("UPDATE pending_outcome SET open=0, window_left=0, seen_numbers=? WHERE id=?", (seen_new, pid))
            if stage==0:
                update_daily_score(0, True)
                if src=="IA": update_daily_score_ia(0, True)
                await send_green_imediato(suggested,"G0")
            else:
                _convert_last_loss_to_green()
                if src=="IA": _convert_last_loss_to_green_ia()
        else:
            if left>1:
                exec_write("""UPDATE pending_outcome SET stage=stage+1, window_left=window_left-1, seen_numbers=? WHERE id=?""",(seen_new,pid))
                if stage==0:
                    update_daily_score(0, False)
                    if src=="IA": update_daily_score_ia(0, False)
                    await send_loss_imediato(suggested,"G0"); _ia_set_post_loss_block()
            else:
                exec_write("UPDATE pending_outcome SET open=0, window_left=0, seen_numbers=? WHERE id=?", (seen_new, pid))
    return {"ok": True}

# =========================
# IA loop (com EXPLORAÇÃO no warm-up)
# =========================
async def ia2_process_once():
    base, pattern_key, strategy, after_num = [], "GEN", None, None
    tail = get_recent_tail(WINDOW)
    best, conf_raw, tail_len, post = suggest_number(base, pattern_key, strategy, after_num)

    # gap atual
    gap = 1.0
    if post and len(post) >= 2:
        top = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
        if len(top) >= 2: gap = top[0][1] - top[1][1]

    # se já há pendência, não abre outra
    if _has_open_pending():
        _mark_reason("pendencia_aberta"); return

    # se bloqueado ou antispam, respeita
    eff=_eff()
    if _ia2_blocked_now(): _mark_reason("cooldown_pos_loss"); return
    if not _ia2_antispam_ok(): _mark_reason("limite_hora_atingido"); return
    if now_ts() - (IA2_LAST_FIRE_TS or 0) < eff["IA2_MIN_SECONDS_BETWEEN_FIRE"]:
        _mark_reason("espacamento_minimo"); return

    # Caso 1: temos best e passa nos thresholds dinâmicos
    if best is not None and _relevance_ok(conf_raw, gap, tail_len) and _ia2_can_fire_now():
        open_pending(strategy, best, source="IA")
        await ia2_send_signal(best, conf_raw, tail_len, "FIRE")
        _ia2_mark_fire_sent()
        _mark_reason(f"FIRE(best={best}, conf={conf_raw:.3f}, gap={gap:.3f}, tail={tail_len})")
        return

    # Caso 2: EXPLORAÇÃO durante warm-up (envia mesmo sem confiança)
    samples=_current_samples()
    if samples < WARMUP_SAMPLES and _ia2_can_fire_now():
        # escolhe o top-1 da cauda(40) se existir; senão, 1
        boosts = tail_top2_boost(tail, k=40)
        expl = max(boosts.items(), key=lambda kv: kv[1])[0] if tail else 1
        open_pending(None, expl, source="IA")
        await ia2_send_signal(expl, 0.0, tail_len, "WARMUP")
        _ia2_mark_fire_sent()
        _mark_reason(f"WARMUP_FIRE(expl={expl}, tail={tail_len}, samples={samples})")
        return

    # Caso 3: não disparou
    if tail_len < MIN_SAMPLES: _mark_reason(f"amostra_insuficiente({tail_len}<{MIN_SAMPLES})")
    else: _mark_reason(f"reprovado(conf={conf_raw:.3f}, gap={gap:.3f})")

# =========================
# INTEL (stub)
# =========================
class _IntelStub:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(os.path.join(self.base_dir, "snapshots"), exist_ok=True)
    def start_signal(self, suggested: int, strategy: Optional[str] = None):
        try:
            path = os.path.join(self.base_dir, "snapshots", "latest_top.json")
            with open(path, "w") as f: json.dump({"ts": now_ts(), "suggested": suggested, "strategy": strategy}, f)
        except Exception as e: print(f"[INTEL] snapshot error: {e}")
    def stop_signal(self): pass

INTEL = _IntelStub(INTEL_DIR)

# =========================
# Relatório Canal x IA
# =========================
def _read_daily_score_days(limit_days:int=7):
    return query_all("""SELECT yyyymmdd, g0, g1, g2, loss, streak FROM daily_score ORDER BY yyyymmdd DESC LIMIT ?""",(limit_days,))
def _read_daily_score_ia_days(limit_days:int=7):
    return query_all("""SELECT yyyymmdd, g0, loss, streak FROM daily_score_ia ORDER BY yyyymmdd DESC LIMIT ?""",(limit_days,))
def _fmt_pct(num:int, den:int) -> float: return (num/den*100.0) if den else 0.0

def _mk_relatorio_text(days:int=7) -> str:
    days=max(1,min(int(days),30)); y=today_key_local()
    r=query_one("SELECT g0,loss,streak FROM daily_score WHERE yyyymmdd=?", (y,))
    g0=(r["g0"] if r else 0); loss=(r["loss"] if r else 0); acc_today=_fmt_pct(g0,g0+loss)
    ria=query_one("SELECT g0,loss,streak FROM daily_score_ia WHERE yyyymmdd=?", (y,))
    ia_g0=(ria["g0"] if ria else 0); ia_loss=(ria["loss"] if ria else 0); ia_acc_today=_fmt_pct(ia_g0, ia_g0+ia_loss)
    chan_rows=_read_daily_score_days(days); ia_rows=_read_daily_score_ia_days(days)
    sum_g0=sum((rr["g0"] or 0) for rr in chan_rows); sum_loss=sum((rr["loss"] or 0) for rr in chan_rows)
    sum_ia_g0=sum((rr["g0"] or 0) for rr in ia_rows); sum_ia_loss=sum((rr["loss"] or 0) for rr in ia_rows)
    acc_ndays=_fmt_pct(sum_g0, sum_g0+sum_loss); ia_acc_ndays=_fmt_pct(sum_ia_g0, sum_ia_g0+sum_ia_loss)
    return ("📈 <b>Relatório de Desempenho</b>\n"
            f"🗓️ Janela: <b>hoje</b> e últimos <b>{days}</b> dias\n—\n"
            f"📣 Canal (hoje): G0=<b>{g0}</b> | Loss=<b>{loss}</b> | Acerto=<b>{acc_today:.2f}%</b>\n"
            f"🤖 IA (hoje): G0=<b>{ia_g0}</b> | Loss=<b>{ia_loss}</b> | Acerto=<b>{ia_acc_today:.2f}%</b>\n—\n"
            f"📣 Canal ({days}d): G0=<b>{sum_g0}</b> | Loss=<b>{sum_loss}</b> | Acerto=<b>{acc_ndays:.2f}%</b>\n"
            f"🤖 IA ({days}d): G0=<b>{sum_ia_g0}</b> | Loss=<b>{sum_ia_loss}</b> | Acerto=<b>{ia_acc_ndays:.2f}%</b>\n")

# =========================
# Webhook models & rotas
# =========================
class Update(BaseModel):
    update_id: int
    channel_post: Optional[dict] = None
    message: Optional[dict] = None
    edited_channel_post: Optional[dict] = None
    edited_message: Optional[dict] = None

@app.get("/")
async def root(): return {"ok": True, "detail": "Use POST /webhook/<WEBHOOK_TOKEN>"}

@app.on_event("startup")
async def _boot_tasks():
    try: asyncio.create_task(_health_reporter_task())
    except Exception as e: print(f"[HEALTH] startup error: {e}")
    try: asyncio.create_task(_daily_reset_task())
    except Exception as e: print(f"[RESET] startup error: {e}")
    async def _ia2_loop():
        while True:
            try: await ia2_process_once()
            except Exception as e: print(f"[IA2] analyzer error: {e}")
            await asyncio.sleep(max(0.2, INTEL_ANALYZE_INTERVAL))
    try: asyncio.create_task(_ia2_loop())
    except Exception as e: print(f"[IA2] startup error: {e}")

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN: raise HTTPException(status_code=403, detail="Forbidden")
    data = await request.json()
    upd = Update(**data)
    msg = upd.channel_post or upd.message or upd.edited_channel_post or upd.edited_message
    if not msg: return {"ok": True}

    text = (msg.get("text") or msg.get("caption") or "").strip()
    t = re.sub(r"\s+", " ", text)

    # 1) GREEN / RED — salva o ÚLTIMO número do parênteses
    gnum = extract_green_number(t)
    redn = extract_red_last_left(t)
    if gnum is not None or redn is not None:
        n_observed = gnum if gnum is not None else redn
        append_timeline(n_observed)
        update_ngrams()
        await close_pending_with_result(n_observed, "GREEN" if gnum is not None else "RED")

        strat = extract_strategy(t) or ""
        row = query_one("SELECT suggested_number, pattern_key FROM last_by_strategy WHERE strategy=?", (strat,))
        if row and gnum is not None:
            suggested = int(row["suggested_number"]); pat_key = row["pattern_key"] or "GEN"
            won = (suggested == int(gnum))
            bump_pattern(pat_key, suggested, won)
            if strat: bump_strategy(strat, suggested, won)

        await ia2_process_once()
        return {"ok": True, "observed": n_observed}

    # 2) ANALISANDO — só registra sequência, sem sinal
    if is_analise(t):
        seq_raw = extract_seq_raw(t)
        if seq_raw:
            parts = re.findall(r"[1-4]", seq_raw)
            seq_left_recent = [int(x) for x in parts]
            seq_old_to_new  = seq_left_recent[::-1]
            for n in seq_old_to_new: append_timeline(n)
            update_ngrams()
        await ia2_process_once()
        return {"ok": True, "analise": True}

    # 3) ENTRADA CONFIRMADA — processa G0 (pode abster por qualidade)
    if not is_real_entry(t):
        await ia2_process_once()
        return {"ok": True, "skipped": True}

    source_msg_id = msg.get("message_id")
    if query_one("SELECT 1 FROM suggestions WHERE source_msg_id=?", (source_msg_id,)):
        await ia2_process_once()
        return {"ok": True, "dup": True}

    strategy  = extract_strategy(t) or ""
    seq_raw   = extract_seq_raw(t) or ""
    after_num = extract_after_num(t)
    base, pattern_key = parse_bases_and_pattern(t)
    if not base: base=[1,2,3,4]; pattern_key="GEN"

    number, conf, samples, post = suggest_number(base, pattern_key, strategy, after_num)

    # thresholds dinâmicos
    min_conf_dyn, min_gap_dyn = _dyn_thresholds()
    gap = 1.0
    if post and len(post) >= 2:
        top = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
        if len(top) >= 2: gap = top[0][1] - top[1][1]

    if number is None or conf < min_conf_dyn or gap < min_gap_dyn or samples < MIN_SAMPLES:
        await ia2_process_once()
        return {"ok": True, "skipped_low_conf": True, "conf_min": min_conf_dyn, "gap_min": min_gap_dyn}

    exec_write("""INSERT OR REPLACE INTO suggestions
                  (source_msg_id, strategy, seq_raw, context_key, pattern_key, base, suggested_number, stage, sent_at)
                  VALUES (?,?,?,?,?,?,?,?,?)""",
               (source_msg_id, strategy, seq_raw, "CTX", pattern_key, json.dumps(base), int(number), "G0", now_ts()))
    exec_write("""INSERT OR REPLACE INTO last_by_strategy
                  (strategy, source_msg_id, suggested_number, context_key, pattern_key, stage, created_at)
                  VALUES (?,?,?,?,?,?,?)""",
               (strategy, source_msg_id, int(number), "CTX", pattern_key, "G0", now_ts()))

    open_pending(strategy, int(number), source="CHAN")
    conf_capped = max(0.0, min(float(conf), CONF_CAP))
    out = (f"🎯 <b>Número seco (G0):</b> <b>{int(number)}</b>\n"
           f"🧩 <b>Padrão:</b> {pattern_key}{(' após '+str(after_num)) if after_num else ''}\n"
           f"🧮 <b>Base:</b> [{', '.join(str(x) for x in base)}]\n"
           f"📊 Conf: {conf_capped*100:.2f}% | Amostra≈{samples}")
    await tg_broadcast(out)

    await ia2_process_once()
    return {"ok": True, "sent": True, "number": number, "conf": conf_capped, "samples": samples}

# =========================
# DEBUG endpoints
# =========================
@app.get("/debug/samples")
async def debug_samples():
    row = query_one("SELECT SUM(weight) AS s FROM ngram_stats")
    samples = int((row["s"] or 0) if row else 0)
    return {"samples": samples, "MIN_SAMPLES": MIN_SAMPLES, "enough_samples": samples >= MIN_SAMPLES}

@app.get("/debug/reason")
async def debug_reason():
    try:
        age=max(0, now_ts() - (_ia2_last_reason_ts or now_ts()))
        return {"last_reason": _ia2_last_reason, "last_reason_age_seconds": age}
    except Exception as e: return {"error": str(e)}

@app.get("/debug/state")
async def debug_state():
    try:
        samples=int((_get_scalar("SELECT SUM(weight) FROM ngram_stats") or 0))
        pend_open=int(_get_scalar("SELECT COUNT(*) FROM pending_outcome WHERE open=1"))
        reason_age=max(0, now_ts() - (_ia2_last_reason_ts or now_ts()))
        y=today_key_local()
        r=query_one("SELECT g0,loss,streak FROM daily_score WHERE yyyymmdd=?", (y,))
        g0=(r["g0"] if r else 0); loss=(r["loss"] if r else 0); acc=(g0/(g0+loss)) if (g0+loss)>0 else 0.0
        ria=query_one("SELECT g0,loss,streak FROM daily_score_ia WHERE yyyymmdd=?", (y,))
        ia_g0=(ria["g0"] if ria else 0); ia_loss=(ria["loss"] if ria else 0)
        ia_acc=(ia_g0/(ia_g0+ia_loss)) if (ia_g0+ia_loss)>0 else 0.0
        hb=_ia2_hour_key()
        cooldown_remaining=max(0, IA2_BLOCK_UNTIL - now_ts())
        last_fire_age=max(0, now_ts() - (IA2_LAST_FIRE_TS or 0))
        eff=_eff()
        g7,l7,acc7=_ia_acc_days(7)
        return {
            "samples": samples,
            "enough_samples": samples >= MIN_SAMPLES,
            "pendencias_abertas": pend_open,
            "last_reason": _ia2_last_reason,
            "last_reason_age_seconds": int(reason_age),
            "hour_bucket": hb,
            "fires_enviados_nesta_hora": IA2_SENT_THIS_HOUR,
            "cooldown_pos_loss_seconds": int(cooldown_remaining),
            "ultimo_fire_ha_seconds": int(last_fire_age),
            "placar_canal_hoje": {"g0": int(g0), "loss": int(loss), "acc_pct": round(acc*100, 2)},
            "placar_ia_hoje": {"g0": int(ia_g0), "loss": int(ia_loss), "acc_pct": round(ia_acc*100, 2)},
            "ia_acc_7d": {"greens": g7, "losses": l7, "acc_pct": round(acc7*100, 2)},
            "config": {
                "PHASE": eff["PHASE"], "MIN_SAMPLES": MIN_SAMPLES,
                "IA2_TIER_STRICT": eff["MIN_CONF_G0"], "IA2_GAP_MIN": eff["MIN_GAP_G0"],
                "IA2_MAX_PER_HOUR": eff["IA2_MAX_PER_HOUR"], "IA2_MIN_SECONDS_BETWEEN_FIRE": eff["IA2_MIN_SECONDS_BETWEEN_FIRE"],
                "IA2_COOLDOWN_AFTER_LOSS": eff["IA2_COOLDOWN_AFTER_LOSS"], "MAX_CONCURRENT_PENDINGS": eff["MAX_CONCURRENT_PENDINGS"],
                "WARMUP_SAMPLES": WARMUP_SAMPLES
            }
        }
    except Exception as e: return {"ok": False, "error": str(e)}

_last_flush_ts=0
@app.get("/debug/flush")
async def debug_flush(request: Request, days: int = 7, key: str = ""):
    global _last_flush_ts
    try:
        if not key or key != FLUSH_KEY: return {"ok": False, "error": "unauthorized"}
        now=now_ts()
        if now - (_last_flush_ts or 0) < 60:
            return {"ok": False,"error": "flush_cooldown","retry_after_seconds": 60 - (now - (_last_flush_ts or 0))}
        try: await tg_broadcast(_health_text())
        except Exception as e: print(f"[FLUSH] saúde erro: {e}")
        try: await tg_broadcast(_mk_relatorio_text(days=max(1,min(days,30))))
        except Exception as e: print(f"[FLUSH] relatorio erro: {e}")
        _last_flush_ts=now; return {"ok": True, "flushed": True, "days": max(1,