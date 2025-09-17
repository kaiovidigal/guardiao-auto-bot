#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webhook (GEN híbrido + estratégia + placar diário + relatório 5/5min)

- Respeita a estratégia do canal-fonte (KWOK/SSH/ODD/EVEN/Sequência) restringindo os candidatos.
- Decide o "número seco" usando n-gram híbrido: cauda curta (1000) + longa (5000).
- Publica G0 imediatamente; fecha assim que nosso número aparecer (G0/G1/G2).
- Timeout único: ao ter 2 observados, inicia contagem de 45s; se 3º não vier, fecha com X.
- Numeração de jogada por dia (Jogada 001, 002, ...).
- Placar diário (G0/G1/G2/LOSS) e relatório automático a cada 5 minutos.
"""

import os, re, time, json, sqlite3, asyncio
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone
import httpx
from fastapi import FastAPI, Request, HTTPException

# ========= ENV =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1002796105884").strip()  # destino
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "-1002810508717").strip()  # fonte
DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"

if not TG_BOT_TOKEN:   raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")
if not WEBHOOK_TOKEN:  raise RuntimeError("Defina WEBHOOK_TOKEN no ambiente.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
app = FastAPI(title="guardiao-auto-bot (GEN híbrido + estratégia + relatório)", version="3.2.0")

# ========= HÍBRIDO (curta/longa) =========
SHORT_WINDOW    = 40     # cauda curta (mais responsiva)
LONG_WINDOW     = 200     # cauda longa  (mais estável)
CONF_SHORT_MIN  = 0.35     # confiança mínima na curta
CONF_LONG_MIN   = 0.45     # confiança mínima na longa
GAP_MIN         = 0.020    # gap mínimo (top1 - top2) nas duas
FINAL_TIMEOUT   = 45       # segundos; começa quando houver 2 observados

# Relatório 5/5min: faixas de status do dia
GOOD_THRESH = 0.70
BAD_THRESH  = 0.50

# ========= Utils =========
def now_ts() -> int:
    return int(time.time())

def ts_str(ts=None) -> str:
    if ts is None: ts = now_ts()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def today_key() -> str:
    # reset natural "à meia-noite" via chave de dia YYYYMMDD (UTC)
    return datetime.utcnow().strftime("%Y%m%d")

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

def migrate_db():
    con = _connect(); cur = con.cursor()
    # Histórico simples
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    # Pendência (com timer final) + metadados
    cur.execute("""CREATE TABLE IF NOT EXISTS pending (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER,
        suggested INTEGER,
        open INTEGER DEFAULT 1,
        seen TEXT,                -- "1-3" (dois) | "1-3-2" (três)
        opened_at INTEGER,
        last_post_short TEXT,
        last_post_long  TEXT,
        last_conf_short REAL,
        last_conf_long  REAL,
        d_final INTEGER,
        base TEXT,
        pattern_key TEXT,
        day_key TEXT,
        play_no INTEGER
    )""")
    # Placar diário por estágio
    cur.execute("""CREATE TABLE IF NOT EXISTS daily_score (
        day_key TEXT PRIMARY KEY,
        g0 INTEGER NOT NULL DEFAULT 0,
        g1 INTEGER NOT NULL DEFAULT 0,
        g2 INTEGER NOT NULL DEFAULT 0,
        loss INTEGER NOT NULL DEFAULT 0
    )""")
    # Contador de jogadas por dia
    cur.execute("""CREATE TABLE IF NOT EXISTS day_counter (
        day_key TEXT PRIMARY KEY,
        next_no INTEGER NOT NULL
    )""")
    # Backfills idempotentes:
    for col, ddl in [
        ("d_final","ALTER TABLE pending ADD COLUMN d_final INTEGER"),
        ("last_post_short","ALTER TABLE pending ADD COLUMN last_post_short TEXT"),
        ("last_post_long","ALTER TABLE pending ADD COLUMN last_post_long TEXT"),
        ("last_conf_short","ALTER TABLE pending ADD COLUMN last_conf_short REAL"),
        ("last_conf_long","ALTER TABLE pending ADD COLUMN last_conf_long REAL"),
        ("base","ALTER TABLE pending ADD COLUMN base TEXT"),
        ("pattern_key","ALTER TABLE pending ADD COLUMN pattern_key TEXT"),
        ("day_key","ALTER TABLE pending ADD COLUMN day_key TEXT"),
        ("play_no","ALTER TABLE pending ADD COLUMN play_no INTEGER"),
    ]:
        try: cur.execute(f"SELECT {col} FROM pending LIMIT 1")
        except sqlite3.OperationalError:
            try: cur.execute(ddl)
            except sqlite3.OperationalError: pass
    con.commit(); con.close()

migrate_db()

# ========= Telegram =========
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                "disable_web_page_preview": True})

# ========= Placar / Jogadas =========
def _ensure_day_rows(day: str):
    con = _connect(); cur = con.cursor()
    if not cur.execute("SELECT 1 FROM daily_score WHERE day_key=?", (day,)).fetchone():
        cur.execute("INSERT INTO daily_score (day_key,g0,g1,g2,loss) VALUES (?,?,?, ?,?)",
                    (day,0,0,0,0))
    if not cur.execute("SELECT 1 FROM day_counter WHERE day_key=?", (day,)).fetchone():
        cur.execute("INSERT INTO day_counter (day_key,next_no) VALUES (?,?)", (day,1))
    con.commit(); con.close()

def _next_play_no(day: str) -> int:
    _ensure_day_rows(day)
    con = _connect(); cur = con.cursor()
    row = cur.execute("SELECT next_no FROM day_counter WHERE day_key=?", (day,)).fetchone()
    n = int(row["next_no"]) if row else 1
    cur.execute("UPDATE day_counter SET next_no=? WHERE day_key=?", (n+1, day))
    con.commit(); con.close()
    return n

def bump_daily(stage: Optional[str], won: bool):
    day = today_key()
    _ensure_day_rows(day)
    con = _connect(); cur = con.cursor()
    row = cur.execute("SELECT g0,g1,g2,loss FROM daily_score WHERE day_key=?", (day,)).fetchone()
    g0,g1,g2,ls = (row["g0"], row["g1"], row["g2"], row["loss"]) if row else (0,0,0,0)
    if won:
        if stage == "G0": g0 += 1
        elif stage == "G1": g1 += 1
        elif stage == "G2": g2 += 1
    else:
        ls += 1
    cur.execute("UPDATE daily_score SET g0=?,g1=?,g2=?,loss=? WHERE day_key=?",
                (g0,g1,g2,ls,day))
    con.commit(); con.close()

def read_daily() -> Tuple[int,int,int,int,float]:
    day = today_key()
    _ensure_day_rows(day)
    con = _connect()
    row = con.execute("SELECT g0,g1,g2,loss FROM daily_score WHERE day_key=?", (day,)).fetchone()
    con.close()
    g0,g1,g2,ls = (row["g0"], row["g1"], row["g2"], row["loss"]) if row else (0,0,0,0)
    greens = g0+g1+g2
    total  = greens + ls
    acc = (greens/total) if total>0 else 0.0
    return g0,g1,g2,ls,acc

# ========= Timeline / N-gram simplificado =========
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
    # contextos (após X se existir na cauda)
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
    if ODD_RX.search(t):
        return [1,3], "ODD"
    if EVEN_RX.search(t):
        return [2,4], "EVEN"
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
        if base:
            return base, "SEQ"
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
    if b_s == b_l and c_s >= CONF_SHORT_MIN and c_l >= CONF_LONG_MIN and g_s >= GAP_MIN and g_l >= GAP_MIN:
        best = b_s
    return best, c_s, c_l, len(tail_s), post_s, post_l

# ========= Pending helpers (timer único 45s) =========
def get_open_pending() -> Optional[sqlite3.Row]:
    con = _connect()
    row = con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone()
    con.close()
    return row

def _seen_list(row: sqlite3.Row) -> List[str]:
    seen = (row["seen"] or "").strip()
    return [s for s in seen.split("-") if s]

def _set_seen(row_id:int, seen_list:List[str]):
    seen_txt = "-".join(seen_list[:3])
    _exec_write("UPDATE pending SET seen=? WHERE id=?", (seen_txt, row_id))

def _ensure_final_deadline_when_two(row: sqlite3.Row):
    if int(row["d_final"] or 0) > 0: 
        return
    seen_list = _seen_list(row)
    if len(seen_list) == 2:
        _exec_write("UPDATE pending SET d_final=? WHERE id=?", (now_ts() + FINAL_TIMEOUT, int(row["id"])))

def _close_now(row: sqlite3.Row, suggested:int, final_seen:List[str]):
    # determina estágio e outcome
    obs_nums = [int(x) for x in final_seen if x.isdigit()]
    if len(obs_nums) >= 1 and obs_nums[0] == suggested:
        outcome, stage_lbl = "GREEN", "G0"
    elif len(obs_nums) >= 2 and obs_nums[1] == suggested:
        outcome, stage_lbl = "GREEN", "G1"
    elif len(obs_nums) >= 3 and obs_nums[2] == suggested:
        outcome, stage_lbl = "GREEN", "G2"
    else:
        outcome, stage_lbl = "LOSS", "G2"

    _exec_write("UPDATE pending SET open=0, seen=? WHERE id=?",
                ("-".join(final_seen[:3]), int(row["id"])))
    bump_daily(stage_lbl if outcome=="GREEN" else None, won=(outcome=="GREEN"))

    play_no = int(row["play_no"] or 0)
    play_tag = f"{today_key()}-{play_no:03d}" if play_no else today_key()
    our_num_display = suggested if outcome=="GREEN" else "X"
    msg = (f"{'🟢' if outcome=='GREEN' else '🔴'} <b>{outcome}</b> — Jogada <b>{play_tag}</b> "
           f"(<b>{stage_lbl}</b>, nosso={our_num_display}, observados={'-'.join(final_seen[:3])}).")
    # anexa placar compacto do dia
    g0,g1,g2,ls,acc = read_daily()
    msg += f"\n📊 Dia: G0={g0} • G1={g1} • G2={g2} • Loss={ls} • Acerto={acc*100:.1f}%"
    return msg

def open_pending(suggested: int, conf_short:float, conf_long:float,
                 post_short:Dict[int,float], post_long:Dict[int,float],
                 base:List[int], pattern_key:str):
    day = today_key()
    play_no = _next_play_no(day)
    _exec_write("""INSERT INTO pending
        (created_at, suggested, open, seen, opened_at,
         last_post_short, last_post_long, last_conf_short, last_conf_long,
         d_final, base, pattern_key, day_key, play_no)
        VALUES (?,?,?,?,?,?,?,?,?,NULL,?,?,?,?)
    """, (now_ts(), int(suggested), 1, "", now_ts(),
          json.dumps(post_short), json.dumps(post_long),
          float(conf_short), float(conf_long),
          json.dumps(base), pattern_key, day, int(play_no)))
    return day, play_no

def _maybe_close_by_final_timeout():
    row = get_open_pending()
    if not row: return None
    d_final = int(row["d_final"] or 0)
    if d_final <= 0 or now_ts() < d_final:
        return None
    seen_list = _seen_list(row)
    if len(seen_list) == 2:
        seen_list.append("X")
        msg = _close_now(row, int(row["suggested"] or 0), seen_list)
        return msg
    return None

# ========= Relatório 5/5min =========
async def send_5min_report():
    g0,g1,g2,ls,acc = read_daily()
    greens = g0+g1+g2
    status = "🟢 Bom" if acc>=GOOD_THRESH else ("🟠 Neutro" if acc>=BAD_THRESH else "🔴 Ruim")
    txt = (
        "⏱️ <b>Relatório (5 min)</b>\n"
        f"📅 Dia: <code>{today_key()}</code>\n"
        f"🟢 G0={g0} | G1={g1} | G2={g2} · 🔴 Loss={ls}\n"
        f"✅ Acerto: <b>{acc*100:.2f}%</b> • Total jogadas: <b>{greens+ls}</b>\n"
        f"📣 Status do dia: <b>{status}</b>"
    )
    await tg_send_text(TARGET_CHANNEL, txt)

async def report_loop():
    while True:
        try:
            await send_5min_report()
        except Exception as e:
            print(f"[RELATÓRIO] erro: {e}")
        await asyncio.sleep(300)  # 5 minutos

@app.on_event("startup")
async def _boot():
    try:
        asyncio.create_task(report_loop())
    except Exception as e:
        print(f"[BOOT] report loop error: {e}")

# ========= Rotas =========
@app.get("/")
async def root():
    return {"ok": True, "service": "guardiao-auto-bot (GEN híbrido + estratégia + timeout 45s + relatório 5/5min)"}

@app.get("/health")
async def health():
    pend = get_open_pending()
    g0,g1,g2,ls,acc = read_daily()
    return {
        "ok": True,
        "db": DB_PATH,
        "pending_open": bool(pend),
        "pending_seen": (pend["seen"] if pend else ""),
        "d_final": int(pend["d_final"] or 0) if pend else 0,
        "day": today_key(),
        "score": {"g0":g0,"g1":g1,"g2":g2,"loss":ls,"acc_pct":round(acc*100,2)},
        "time": ts_str()
    }

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Destravamento por timeout (se houver 2 observados e 45s vencidos)
    msg_timeout = _maybe_close_by_final_timeout()
    if msg_timeout:
        await tg_send_text(TARGET_CHANNEL, msg_timeout)

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

    # 1) Fechamentos do fonte (GREEN/LOSS — 1,2,3 números)
    if GREEN_RX.search(text) or LOSS_RX.search(text):
        pend = get_open_pending()
        nums = parse_close_numbers(text)
        if pend and nums:
            seen = _seen_list(pend)
            for n in nums:
                if len(seen) >= 3: break
                seen.append(str(int(n)))
                # GREEN imediato se bater nosso número
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

    # 2) ENTRADA CONFIRMADA — decide novo número (híbrido + estratégia)
    parsed = parse_entry_text(text)
    if not parsed:
        return {"ok": True, "skipped": "nao_eh_entrada_confirmada"}

    # antes de abrir novo, tenta fechar por timeout se pendência ficou com 2 e já venceu
    msg_timeout = _maybe_close_by_final_timeout()
    if msg_timeout:
        await tg_send_text(TARGET_CHANNEL, msg_timeout)

    # Alimenta histórico com sequência crua, se houver
    seq = parsed["seq"] or []
    if seq: append_timeline(seq)

    after       = parsed["after"]
    base        = parsed["base"] or [1,2,3,4]
    pattern_key = parsed["pattern_key"] or "GEN"

    best, conf_s, conf_l, samples_s, post_s, post_l = choose_single_number_hybrid(after, base)
    if best is None:
        return {"ok": True, "skipped_low_conf_or_disagree": True}

    day, play_no = open_pending(best, conf_s, conf_l, post_s, post_l, base, pattern_key)

    base_txt = ", ".join(str(x) for x in base) if base else "—"
    aft_txt  = f" após {after}" if after else ""
    play_tag = f"{day}-{play_no:03d}"
    txt = (
        f"🎮 <b>Jogada</b> <b>{play_tag}</b>\n"
        f"🎯 <b>Número seco (G0):</b> <b>{best}</b>\n"
        f"🧩 <b>Padrão:</b> {pattern_key}{aft_txt}\n"
        f"🧮 <b>Base:</b> [{base_txt}]\n"
        f"📊 <b>Conf (curta/longa):</b> {conf_s*100:.2f}% / {conf_l*100:.2f}% "
        f"| <b>Amostra≈</b>{samples_s}"
    )
    await tg_send_text(TARGET_CHANNEL, txt)
    return {"ok": True, "posted": True, "best": best, "play": play_tag}