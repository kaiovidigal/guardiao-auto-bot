# -*- coding: utf-8 -*-
# Fan Tan — Guardião (G0 + Recuperação G1/G2) — sem IA autônoma
# - Sugestão de número G0 baseada no histórico do próprio canal (n-gram)
# - Sem loop de IA disparando sozinho
# - GREEN/RED: salva o ÚLTIMO número entre parênteses. ANALISANDO só registra sequência
# - Recuperação: não conta Loss parcial; só conta Loss quando esgota G2
# - Placar automático a cada 30 minutos (últimos 30m): G0, G1, G2 e Loss

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
DB_PATH       = os.getenv("DB_PATH", "/data/data.db").strip() or "/data/data.db"
TG_BOT_TOKEN  = os.getenv("TG_BOT_TOKEN", "").strip()
PUBLIC_CHANNEL= os.getenv("PUBLIC_CHANNEL", "").strip()
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "").strip()
FLUSH_KEY     = os.getenv("FLUSH_KEY", "meusegredo123").strip()

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN:
    print("⚠️ Defina TG_BOT_TOKEN e WEBHOOK_TOKEN.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
REPL_ENABLED = True
REPL_CHANNEL = (os.getenv("REPL_CHANNEL", "-1003052132833") or "-1003052132833").strip()

# =========================
# Hiperparâmetros do sugeridor (CANAL)
# =========================
MAX_STAGE = 3               # G0, G1, G2 (janela total = 3)
WINDOW    = 400             # histórico usado
DECAY     = 0.985
W4, W3, W2, W1 = 0.38, 0.30, 0.20, 0.12

# thresholds (mais "soltos" pra enviar mais)
MIN_SAMPLES = 300           # antes 600
GAP_MIN     = 0.03          # antes 0.04

# antispam do canal
CHAN_MAX_PER_MIN = 18       # antes 12
_last_min_bucket = None
_chan_sent_this_min = 0

app = FastAPI(title="Fantan Guardião — Canal-only (G0 + Recuperação)", version="4.3.0")

# =========================
# DB helpers
# =========================
OLD_DB_CANDIDATES = [
    "/var/data/data.db",
    "/opt/render/project/src/data.db",
    "/opt/render/project/src/data/data.db",
    "/data/data.db"
]

def _ensure_db_dir():
    try:
        db_dir = os.path.dirname(DB_PATH)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)   # <<< garante /data
    except Exception as e:
        print(f"[DB] mkdir: {e}")

def _migrate_old_db_if_needed():
    if os.path.exists(DB_PATH):
        return
    for src in OLD_DB_CANDIDATES:
        if os.path.exists(src):
            try:
                _ensure_db_dir()
                shutil.copy2(src, DB_PATH)
                print(f"[DB] Migrado {src} -> {DB_PATH}")
                return
            except Exception as e:
                print(f"[DB] Migração falhou {src}: {e}")

_ensure_db_dir()
_migrate_old_db_if_needed()

def _connect() -> sqlite3.Connection:
    # Garante diretório antes de abrir o SQLite
    _ensure_db_dir()
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def exec_write(sql: str, params: tuple = (), retries: int = 8, wait: float = 0.25):
    for _ in range(retries):
        try:
            con = _connect()
            con.execute(sql, params)
            con.commit()
            con.close()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                time.sleep(wait)
                continue
            raise
    raise sqlite3.OperationalError("Banco bloqueado após várias tentativas.")

def query_all(sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    con = _connect()
    rows = con.execute(sql, params).fetchall()
    con.close()
    return rows

def query_one(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    con = _connect()
    row = con.execute(sql, params).fetchone()
    con.close()
    return row

def init_db():
    con = _connect()
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL, number INTEGER NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS ngram_stats (
        n INTEGER NOT NULL, ctx TEXT NOT NULL, next INTEGER NOT NULL, weight REAL NOT NULL,
        PRIMARY KEY (n, ctx, next))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS suggestions (
        source_msg_id INTEGER PRIMARY KEY, strategy TEXT, seq_raw TEXT, context_key TEXT, pattern_key TEXT, base TEXT,
        suggested_number INTEGER, stage TEXT, sent_at INTEGER)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS last_by_strategy (
        strategy TEXT PRIMARY KEY, source_msg_id INTEGER, suggested_number INTEGER,
        context_key TEXT, pattern_key TEXT, stage TEXT, created_at INTEGER)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL, stage INTEGER NOT NULL,
        result TEXT NOT NULL, suggested INTEGER NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS daily_score (
        yyyymmdd TEXT PRIMARY KEY, g0 INTEGER NOT NULL DEFAULT 0, g1 INTEGER NOT NULL DEFAULT 0,
        g2 INTEGER NOT NULL DEFAULT 0, loss INTEGER NOT NULL DEFAULT 0, streak INTEGER NOT NULL DEFAULT 0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pending_outcome (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL, strategy TEXT, suggested INTEGER NOT NULL,
        stage INTEGER NOT NULL, open INTEGER NOT NULL, window_left INTEGER NOT NULL,
        seen_numbers TEXT DEFAULT '', announced INTEGER NOT NULL DEFAULT 0)""")
    con.commit()
    # migrações leves (idempotentes)
    for alter in [
        "ALTER TABLE pending_outcome ADD COLUMN seen_numbers TEXT DEFAULT ''",
        "ALTER TABLE pending_outcome ADD COLUMN announced INTEGER NOT NULL DEFAULT 0",
    ]:
        try:
            cur.execute(alter)
            con.commit()
        except sqlite3.OperationalError:
            pass
    con.close()

init_db()

# =========================
# Utils / Telegram
# =========================
def now_ts() -> int: return int(time.time())
def today_key_local() -> str: return datetime.now(timezone.utc).strftime("%Y%m%d")

async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN or not chat_id:
        return
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse,
                "disable_web_page_preview": True
            }
        )

async def tg_broadcast(text: str, parse: str="HTML"):
    if REPL_ENABLED and REPL_CHANNEL:
        await tg_send_text(REPL_CHANNEL, text, parse)

async def send_green(stage:int, number:int):
    if stage==0:
        await tg_broadcast(f"✅ <b>GREEN (G0)</b> — Número: <b>{number}</b>")
    elif stage==1:
        await tg_broadcast(f"✅ <b>GREEN (recuperação G1)</b> — Número: <b>{number}</b>")
    elif stage==2:
        await tg_broadcast(f"✅ <b>GREEN (recuperação G2)</b> — Número: <b>{number}</b>")

async def send_loss_final(number:int):
    await tg_broadcast(f"❌ <b>LOSS</b> — Número base: <b>{number}</b> (após G2)")

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
    if len(tail) < 2:
        return
    for t in range(1, len(tail)):
        nxt = tail[t]
        dist = (len(tail)-1) - t
        w = (decay ** dist)
        for n in range(2, max_n+1):
            if t-(n-1) < 0:
                break
            ctx = tail[t-(n-1):t]
            ctx_key = ",".join(str(x) for x in ctx)
            exec_write("""
                INSERT INTO ngram_stats (n, ctx, next, weight)
                VALUES (?,?,?,?)
                ON CONFLICT(n, ctx, next) DO UPDATE SET weight = weight + excluded.weight
            """, (n, ctx_key, int(nxt), float(w)))

def prob_from_ngrams(ctx: List[int], candidate: int) -> float:
    n = len(ctx) + 1
    if n < 2 or n > 5:
        return 0.0
    ctx_key = ",".join(str(x) for x in ctx)
    tot = (query_one("SELECT SUM(weight) AS w FROM ngram_stats WHERE n=? AND ctx=?", (n, ctx_key)) or {"w":0})["w"] or 0.0
    if tot <= 0:
        return 0.0
    w = (query_one("SELECT weight FROM ngram_stats WHERE n=? AND ctx=? AND next=?", (n, ctx_key, candidate)) or {"weight":0})["weight"] or 0.0
    return w / tot

def tail_top2_boost(tail: List[int], k:int=40) -> Dict[int, float]:
    boosts={1:1.00,2:1.00,3:1.00,4:1.00}
    if not tail:
        return boosts
    tail_k = tail[-k:] if len(tail)>=k else tail[:]
    c=Counter(tail_k)
    freq=c.most_common()
    if len(freq)>=1: boosts[freq[0][0]]=1.04
    if len(freq)>=2: boosts[freq[1][0]]=1.02
    return boosts

def _chan_antispam_ok() -> bool:
    global _last_min_bucket, _chan_sent_this_min
    now_min = datetime.utcnow().strftime("%Y%m%d%H%M")
    if _last_min_bucket != now_min:
        _last_min_bucket = now_min
        _chan_sent_this_min = 0
    if _chan_sent_this_min >= CHAN_MAX_PER_MIN:
        return False
    _chan_sent_this_min += 1
    return True

def suggest_number(base: List[int]) -> Tuple[Optional[int], float, Dict[int,float]]:
    # backoff 4-3-2-1 + boost da cauda
    tail = get_recent_tail(WINDOW)
    if len(tail) < 1:
        return None, 0.0, {}
    boosts = tail_top2_boost(tail, k=40)
    scores: Dict[int,float] = {}
    for c in base:
        p4 = prob_from_ngrams(tail[-4:-1], c) if len(tail)>=4 else 0.0
        p3 = prob_from_ngrams(tail[-3:-1], c) if len(tail)>=3 else 0.0
        p2 = prob_from_ngrams(tail[-2:-1], c) if len(tail)>=2 else 0.0
        p1 = prob_from_ngrams(tail[-1:],    c) if len(tail)>=1 else 0.0
        score = (W4*p4 + W3*p3 + W2*p2 + W1*p1) * boosts.get(c,1.0)
        scores[c] = score if score>0 else 1e-9
    tot = sum(scores.values()) or 1.0
    post = {k: v/tot for k,v in scores.items()}
    top = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    if not top:
        return None, 0.0, post
    gap = (top[0][1] - top[1][1]) if len(top)>=2 else 1.0
    if len(tail) < MIN_SAMPLES or gap < GAP_MIN:
        return None, 0.0, post
    return top[0][0], top[0][1], post

# =========================
# Parsers do canal
# =========================
MUST_HAVE = (r"ENTRADA\s+CONFIRMADA", r"Mesa:\s*Fantan\s*-\s*Evolution")
MUST_NOT  = (r"\bANALISANDO\b", r"\bPlacar do dia\b", r"\bAPOSTA ENCERRADA\b")

def is_real_entry(text: str) -> bool:
    t = re.sub(r"\s+", " ", text).strip()
    for bad in MUST_NOT:
        if re.search(bad, t, flags=re.I):
            return False
    for good in MUST_HAVE:
        if not re.search(good, t, flags=re.I):
            return False
    return True

def extract_seq_raw(text: str) -> Optional[str]:
    m = re.search(r"Sequ[eê]ncia:\s*([^\n\r]+)", text, flags=re.I)
    return m.group(1).strip() if m else None

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
    if not g:
        return None
    nums = re.findall(r"[1-4]", g)
    return int(nums[-1]) if nums else None

def extract_green_number(text: str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text)
    for rx in GREEN_PATTERNS:
        m = rx.search(t)
        if m:
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
def _daily_row():
    y = today_key_local()
    row = query_one("SELECT g0,g1,g2,loss,streak FROM daily_score WHERE yyyymmdd=?", (y,))
    if not row:
        exec_write("INSERT INTO daily_score (yyyymmdd,g0,g1,g2,loss,streak) VALUES (?,?,?,?,?,?)",
                   (y,0,0,0,0,0))
        row = query_one("SELECT g0,g1,g2,loss,streak FROM daily_score WHERE yyyymmdd=?", (y,))
    return dict(row)

def update_score_on_result(stage: Optional[int], won: bool):
    y = today_key_local()
    row = _daily_row()
    g0,g1,g2,loss,streak = row["g0"],row["g1"],row["g2"],row["loss"],row["streak"]
    if won:
        if stage==0: g0+=1
        elif stage==1: g1+=1
        elif stage==2: g2+=1
        streak += 1
    else:
        loss += 1
        streak = 0
    exec_write("""INSERT OR REPLACE INTO daily_score (yyyymmdd,g0,g1,g2,loss,streak)
                  VALUES (?,?,?,?,?,?)""",(y,g0,g1,g2,loss,streak))
    total = g0+g1+g2+loss
    acc = (g0+g1+g2)/total if total else 0.0
    return g0,g1,g2,loss,acc,streak

def record_outcome(stage:int, result:str, suggested:int):
    exec_write("INSERT INTO outcomes (ts,stage,result,suggested) VALUES (?,?,?,?)",
               (now_ts(), int(stage), result.upper(), int(suggested)))

def _score_last_30m():
    since = now_ts() - 30*60
    rows = query_all("SELECT stage,result FROM outcomes WHERE ts>=? ORDER BY id ASC", (since,))
    g0=g1=g2=los=0
    for r in rows:
        if r["result"]=="WIN":
            if r["stage"]==0: g0+=1
            elif r["stage"]==1: g1+=1
            elif r["stage"]==2: g2+=1
        else:
            los+=1
    total = g0+g1+g2+los
    acc = ((g0+g1+g2)/total*100.0) if total else 0.0
    # streak atual (do fim p/ o começo)
    rows2 = query_all("SELECT result FROM outcomes ORDER BY id DESC LIMIT 200", ())
    streak=0
    for rr in rows2:
        if rr["result"]=="WIN": streak+=1
        else: break
    return g0,g1,g2,los,acc,streak

async def _scoreboard_30m_task():
    while True:
        try:
            g0,g1,g2,los,acc,streak = _score_last_30m()
            txt = (f"📊 <b>Placar (30m)</b>\n"
                   f"🟢 G0:{g0} • G1:{g1} • G2:{g2} • 🔴 Loss:{los}\n"
                   f"✅ {acc:.2f}% • 🔥 {streak}")
            await tg_broadcast(txt)
        except Exception as e:
            print(f"[SB30] erro: {e}")
        await asyncio.sleep(30*60)

# =========================
# Pendências
# =========================
def open_pending(suggested: int):
    exec_write("""INSERT INTO pending_outcome (created_at,strategy,suggested,stage,open,window_left,seen_numbers,announced)
                  VALUES (?,?,?,?,1,?, '', 0)""",
               (now_ts(), "", int(suggested), 0, MAX_STAGE))

async def close_pending_with_result(n_observed: int):
    rows=query_all("""SELECT id, suggested, stage, window_left, seen_numbers
                      FROM pending_outcome WHERE open=1 ORDER BY id ASC""")
    if not rows:
        return
    for r in rows:
        pid, suggested, stage, left = r["id"], int(r["suggested"]), int(r["stage"]), int(r["window_left"])
        seen_new = ((r["seen_numbers"] or "") + ("," if (r["seen_numbers"] or "") else "") + str(int(n_observed))).strip()
        if int(n_observed)==suggested:
            exec_write("UPDATE pending_outcome SET open=0, window_left=0, seen_numbers=? WHERE id=?", (seen_new, pid))
            record_outcome(stage, "WIN", suggested)
            update_score_on_result(stage, True)
            await send_green(stage, suggested)
        else:
            if left>1:
                exec_write("""UPDATE pending_outcome SET stage=stage+1, window_left=window_left-1, seen_numbers=? WHERE id=?""",
                           (seen_new,pid))
            else:
                exec_write("UPDATE pending_outcome SET open=0, window_left=0, seen_numbers=? WHERE id=?", (seen_new, pid))
                record_outcome(stage, "LOSS", suggested)
                update_score_on_result(None, False)
                await send_loss_final(suggested)

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
async def root():
    return {"ok": True, "detail": "Use POST /webhook/<WEBHOOK_TOKEN>"}

@app.on_event("startup")
async def _boot_tasks():
    try:
        asyncio.create_task(_scoreboard_30m_task())
    except Exception as e:
        print(f"[SB30] startup error: {e}")

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    data = await request.json()
    upd = Update(**data)
    msg = upd.channel_post or upd.message or upd.edited_channel_post or upd.edited_message
    if not msg:
        return {"ok": True}

    text = (msg.get("text") or msg.get("caption") or "").strip()
    t = re.sub(r"\s+", " ", text)

    # 1) GREEN / RED — fecha pendências
    gnum = extract_green_number(t)
    redn = extract_red_last_left(t)
    if gnum is not None or redn is not None:
        n_observed = gnum if gnum is not None else redn
        append_timeline(n_observed)
        update_ngrams()
        await close_pending_with_result(n_observed)
        return {"ok": True, "observed": n_observed}

    # 2) ANALISANDO — só registra sequência
    if is_analise(t):
        seq_raw = extract_seq_raw(t)
        if seq_raw:
            parts = re.findall(r"[1-4]", seq_raw)
            seq_left_recent = [int(x) for x in parts]
            seq_old_to_new  = seq_left_recent[::-1]
            for n in seq_old_to_new:
                append_timeline(n)
            update_ngrams()
        return {"ok": True, "analise": True}

    # 3) ENTRADA CONFIRMADA — gera sugestão e abre pendência
    if not is_real_entry(t):
        return {"ok": True, "skipped": True}

    # antispam mínimo
    if not _chan_antispam_ok():
        return {"ok": True, "rate_limited": True}

    base=[1,2,3,4]
    number, conf, post = suggest_number(base)
    if number is None:
        return {"ok": True, "skipped_low_conf": True}

    source_msg_id = msg.get("message_id")
    if not query_one("SELECT 1 FROM suggestions WHERE source_msg_id=?", (source_msg_id,)):
        exec_write("""INSERT OR REPLACE INTO suggestions
                      (source_msg_id, strategy, seq_raw, context_key, pattern_key, base, suggested_number, stage, sent_at)
                      VALUES (?,?,?,?,?,?,?,?,?)""",
                   (source_msg_id, "", "", "CTX", "GEN", json.dumps(base), int(number), "G0", now_ts()))
    open_pending(int(number))

    out = (f"🎯 <b>Número seco (G0):</b> <b>{int(number)}</b>\n"
           f"📊 Conf: {conf*100:.2f}% (histórico do canal)")
    await tg_broadcast(out)
    return {"ok": True, "sent": True, "number": number, "conf": conf}