 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time, json, sqlite3
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone
import httpx
from fastapi import FastAPI, Request, HTTPException

# ========= ENV =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1002796105884").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()
DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"

# Reset diário: "UTC" (default) ou "LOCAL"
RESET_CLOCK    = os.getenv("RESET_CLOCK", "UTC").strip().upper()  # "UTC" | "LOCAL"

if not TG_BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")
if not WEBHOOK_TOKEN:
    raise RuntimeError("Defina WEBHOOK_TOKEN no ambiente.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
app = FastAPI(title="guardiao-auto-bot (GEN Hibrido)", version="2.8.0")

# ========= HÍBRIDO (curta/longa) – mantém sua lógica anterior =========
SHORT_WINDOW = 100
LONG_WINDOW  = 5000
CONF_SHORT_MIN = 0.55
CONF_LONG_MIN  = 0.58
GAP_MIN        = 0.020

# ======== TIMEOUT ÚNICO: conta a partir de 2 observados ========
FINAL_TIMEOUT = 120  # segundos

def now_ts() -> int:
    return int(time.time())

def ts_str(ts=None) -> str:
    if ts is None: ts = now_ts()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _today_key() -> str:
    """Chave de data 'YYYY-MM-DD' para o reset diário."""
    if RESET_CLOCK == "LOCAL":
        return datetime.now().strftime("%Y-%m-%d")
    # default: UTC
    return datetime.utcnow().strftime("%Y-%m-%d")

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
    # timeline
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    # pendências
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
        d_final INTEGER
    )""")
    # migrações leves (idempotentes)
    for col, ddl in [
        ("d_final","ALTER TABLE pending ADD COLUMN d_final INTEGER"),
        ("last_post_short","ALTER TABLE pending ADD COLUMN last_post_short TEXT"),
        ("last_post_long","ALTER TABLE pending ADD COLUMN last_post_long TEXT"),
        ("last_conf_short","ALTER TABLE pending ADD COLUMN last_conf_short REAL"),
        ("last_conf_long","ALTER TABLE pending ADD COLUMN last_conf_long REAL"),
    ]:
        try:
            cur.execute(f"SELECT {col} FROM pending LIMIT 1")
        except sqlite3.OperationalError:
            try: cur.execute(ddl)
            except sqlite3.OperationalError: pass

    # placar
    cur.execute("""CREATE TABLE IF NOT EXISTS score (
        id INTEGER PRIMARY KEY CHECK (id=1),
        green INTEGER DEFAULT 0,
        loss  INTEGER DEFAULT 0
    )""")
    if not cur.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        cur.execute("INSERT INTO score (id, green, loss) VALUES (1,0,0)")

    # meta (para reset diário)
    cur.execute("""CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        val TEXT
    )""")
    if not cur.execute("SELECT 1 FROM meta WHERE key='last_reset'").fetchone():
        cur.execute("INSERT INTO meta (key,val) VALUES ('last_reset', ?)", (_today_key(),))

    con.commit(); con.close()

migrate_db()

# ========= Reset diário (zera score à meia-noite) =========
def maybe_reset_score_daily():
    con = _connect(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, val TEXT)""")
    row = cur.execute("SELECT val FROM meta WHERE key='last_reset'").fetchone()
    today = _today_key()
    if (not row) or (row["val"] != today):
        # zera placar
        cur.execute("UPDATE score SET green=0, loss=0 WHERE id=1")
        # grava marca de reset
        cur.execute("INSERT OR REPLACE INTO meta (key,val) VALUES ('last_reset',?)", (today,))
        con.commit()
    con.close()

# ========= Telegram =========
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                "disable_web_page_preview": True})

# ========= Score helpers =========
def bump_score(outcome: str):
    con = _connect(); cur = con.cursor()
    row = cur.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
    g, l = (row["green"], row["loss"]) if row else (0, 0)
    if outcome.upper() == "GREEN": g += 1
    elif outcome.upper() == "LOSS": l += 1
    cur.execute("INSERT OR REPLACE INTO score (id, green, loss) VALUES (1,?,?)", (g, l))
    con.commit(); con.close()

def score_text() -> str:
    con = _connect()
    row = con.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
    con.close()
    if not row: return "0 GREEN × 0 LOSS — 0.0%"
    g, l = int(row["green"]), int(row["loss"])
    total = g + l
    acc = (g/total*100.0) if total>0 else 0.0
    return f"{g} GREEN × {l} LOSS — {acc:.1f}%"

# ========= HÍBRIDO — n-gram simplificado por janela =========
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

def _post_from_tail(tail: List[int], after: Optional[int]) -> Dict[int,float]:
    if not tail:
        return {1:0.25,2:0.25,3:0.25,4:0.25}
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
    posts = {1:0.0,2:0.0,3:0.0,4:0.0}
    ctxs  = [(4,ctx4),(3,ctx3),(2,ctx2),(1,ctx1)]
    for lvl, ctx in ctxs:
        if not ctx: continue
        counts = _ctx_counts(tail, ctx[:-1])
        tot = sum(counts.values())
        if tot == 0: continue
        for n in [1,2,3,4]:
            posts[n] += W[4-lvl] * (counts[n]/tot)
    s = sum(posts.values()) or 1e-9
    return {k: v/s for k,v in posts.items()}

def _best_conf_gap(post: Dict[int,float]) -> Tuple[int,float,float]:
    top = sorted(post.items(), key=lambda kv: kv[1], reverse=True)[:2]
    best = top[0][0]; conf = top[0][1]
    gap  = top[0][1] - (top[1][1] if len(top)>1 else 0.0)
    return best, conf, gap

def choose_single_number_hybrid(after: Optional[int]):
    tail_s = get_tail(SHORT_WINDOW)
    tail_l = get_tail(LONG_WINDOW)
    post_s = _post_from_tail(tail_s, after)
    post_l = _post_from_tail(tail_l, after)
    b_s, c_s, g_s = _best_conf_gap(post_s)
    b_l, c_l, g_l = _best_conf_gap(post_l)

    # precisa concordar (mesmo número) e passar limiares de cada base
    if b_s == b_l and c_s >= CONF_SHORT_MIN and c_l >= CONF_LONG_MIN and g_s >= GAP_MIN and g_l >= GAP_MIN:
        best = b_s
    else:
        best = None
    return best, c_s, c_l, len(tail_s), post_s, post_l

# ========= Parsers =========
ENTRY_RX = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
SEQ_RX   = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
AFTER_RX = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)

GREEN_RX = re.compile(r"(?:\bgr+e+e?n\b|\bwin\b|✅)", re.I)
LOSS_RX  = re.compile(r"(?:\blo+s+s?\b|\bred\b|❌|\bperdemos\b)", re.I)
PAREN_GROUP_RX = re.compile(r"\(([^)]*)\)")
ANY_14_RX      = re.compile(r"[1-4]")

def parse_entry_text(text: str) -> Optional[Dict]:
    t = re.sub(r"\s+", " ", text).strip()
    if not ENTRY_RX.search(t): return None
    mseq = SEQ_RX.search(t)
    seq = [int(x) for x in re.findall(r"[1-4]", mseq.group(1))] if mseq else []
    mafter = AFTER_RX.search(t)
    after_num = int(mafter.group(1)) if mafter else None
    return {"seq": seq, "after": after_num, "raw": t}

def parse_close_numbers(text: str) -> List[int]:
    t = re.sub(r"\s+", " ", text)
    groups = PAREN_GROUP_RX.findall(t)
    if groups:
        nums = re.findall(r"[1-4]", groups[-1])
        return [int(x) for x in nums][:3]
    nums = ANY_14_RX.findall(t)
    return [int(x) for x in nums][:3]

# ========= Pending helpers (COM TIMER ÚNICO) =========
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
    """Assim que houver 2 observados, abre o cronômetro de 120s para forçar X se o 3º não vier."""
    if int(row["d_final"] or 0) > 0:
        return
    seen_list = _seen_list(row)
    if len(seen_list) == 2:
        _exec_write("UPDATE pending SET d_final=? WHERE id=?", (now_ts() + FINAL_TIMEOUT, int(row["id"])))

def _close_now(row: sqlite3.Row, suggested:int, final_seen:List[str]):
    obs_nums = [int(x) for x in final_seen if x.isdigit()]
    # GREEN imediato assim que nosso número aparecer:
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
    bump_score(outcome.upper())
    our_num_display = suggested if outcome=="GREEN" else "X"
    msg = (f"{'🟢' if outcome=='GREEN' else '🔴'} <b>{outcome}</b> — finalizado "
           f"(<b>{stage_lbl}</b>, nosso={our_num_display}, observados={'-'.join(final_seen[:3])}).\n"
           f"📊 Geral: {score_text()}")
    return msg

def open_pending(suggested: int, conf_short:float, conf_long:float, post_short:Dict[int,float], post_long:Dict[int,float]):
    _exec_write("""INSERT INTO pending
        (created_at, suggested, open, seen, opened_at, last_post_short, last_post_long, last_conf_short, last_conf_long, d_final)
        VALUES (?,?,?,?,?,?,?,?,?,NULL)
    """, (now_ts(), int(suggested), 1, "", now_ts(),
          json.dumps(post_short), json.dumps(post_long),
          float(conf_short), float(conf_long)))

def _maybe_close_by_final_timeout():
    """Se há pendência aberta com 2 observados e d_final expirou, força X e fecha."""
    row = get_open_pending()
    if not row: return None
    d_final = int(row["d_final"] or 0)
    if d_final <= 0: return None
    if now_ts() < d_final: return None

    seen_list = _seen_list(row)
    if len(seen_list) == 2:
        seen_list.append("X")  # força X no 3º
        msg = _close_now(row, int(row["suggested"] or 0), seen_list)
        return msg
    return None

# ========= Rotas =========
@app.get("/")
async def root():
    return {"ok": True, "service": "guardiao-auto-bot (GEN híbrido c/ timeout único 120s)"}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    # reset diário sempre que chegar um update
    maybe_reset_score_daily()

    # destrava pendência antiga por timeout único (se aplicável)
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

                # GREEN imediato: se bater nosso número, fecha já
                suggested = int(pend["suggested"] or 0)
                obs_nums = [int(x) for x in seen if x.isdigit()]
                if (len(obs_nums) >= 1 and obs_nums[0] == suggested) or \
                   (len(obs_nums) >= 2 and obs_nums[1] == suggested) or \
                   (len(obs_nums) >= 3 and obs_nums[2] == suggested):
                    out = _close_now(pend, suggested, seen)
                    await tg_send_text(TARGET_CHANNEL, out)
                    return {"ok": True, "closed": "green_imediato"}

            # se ficou com 2 observados, arma/renova o d_final = agora+120
            _set_seen(int(pend["id"]), seen)
            pend = get_open_pending()
            if pend and len(_seen_list(pend)) == 2:
                _ensure_final_deadline_when_two(pend)

            # com 3 observados sem bater → fecha LOSS
            pend = get_open_pending()
            if pend and len(_seen_list(pend)) >= 3:
                out = _close_now(pend, int(pend["suggested"] or 0), _seen_list(pend))
                await tg_send_text(TARGET_CHANNEL, out)
                return {"ok": True, "closed": "loss_3_observados"}

        return {"ok": True, "noted_close": True}

    # 2) ENTRADA CONFIRMADA — decide novo número (híbrido)
    parsed = parse_entry_text(text)
    if not parsed:
        return {"ok": True, "skipped": "nao_eh_entrada_confirmada"}

    # Se houver pendência com 2 observados e timer já estourou, fecha antes de abrir nova
    msg_timeout = _maybe_close_by_final_timeout()
    if msg_timeout:
        await tg_send_text(TARGET_CHANNEL, msg_timeout)

    # Alimenta memória com a sequência bruta, se tiver
    seq = parsed["seq"] or []
    if seq:
        append_timeline(seq)

    after = parsed["after"]
    best, conf_s, conf_l, samples_s, post_s, post_l = choose_single_number_hybrid(after)
    if best is None:
        return {"ok": True, "skipped_low_conf_or_disagree": True}

    # Abre pendência nova (sem d_final; ele só nasce com 2 observados)
    open_pending(best, conf_s, conf_l, post_s, post_l)

    txt = (
        f"🎯 <b>Número seco (G0):</b> <b>{best}</b>\n"
        f"🧩 <b>Padrão:</b> GEN{' após ' + str(after) if after else ''}\n"
        f"📊 <b>Conf (curta/longa):</b> {conf_s*100:.2f}% / {conf_l*100:.2f}% "
        f"| <b>Amostra≈</b>{samples_s}"
    )
    await tg_send_text(TARGET_CHANNEL, txt)
    return {"ok": True, "posted": True, "best": best}

# ===== Health =====
@app.get("/health")
async def health():
    pend = get_open_pending()
    return {
        "ok": True,
        "db": DB_PATH,
        "pending_open": bool(pend),
        "pending_seen": (pend["seen"] if pend else ""),
        "d_final": int(pend["d_final"] or 0) if pend else 0,
        "reset_clock": RESET_CLOCK,
        "time": ts_str()
    }