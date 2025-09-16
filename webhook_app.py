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

# destino (onde vamos publicar)
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1002796105884").strip()

# fonte (de onde lemos). Deixe vazio "" para NÃO filtrar.
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()

DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"

if not TG_BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")
if not WEBHOOK_TOKEN:
    raise RuntimeError("Defina WEBHOOK_TOKEN no ambiente.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
app = FastAPI(title="guardiao-auto-bot (GEN híbrido)", version="3.1.0")

# ========= HÍBRIDO (curta/longa) =========
SHORT_WINDOW   = 40      # cauda curta (sensível ao momento)
LONG_WINDOW    = 4000     # cauda longa (estável)
CONF_SHORT_MIN = 0.40     # mínimo de confiança no modelo curto
CONF_LONG_MIN  = 0.50     # mínimo de confiança no modelo longo
GAP_MIN        = 0.020    # distância top1-top2 mínima (anti-empate)

# ========= TIMEOUTS =========
FINAL_TIMEOUT  = 120  # inicia quando houver 2 observados; se não vier o 3º, força X
FORCE_CLOSE_3  = 45   # se já temos 3 observados “mal-formado/sem número”, força fechar em 45s

# ========= Utils =========
def now_ts() -> int:
    return int(time.time())

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

def migrate_db():
    con = _connect(); cur = con.cursor()
    # timeline de números para alimentar n-gram
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    # pendência do tiro aberto
    cur.execute("""CREATE TABLE IF NOT EXISTS pending (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER,
        suggested INTEGER,
        stage INTEGER NOT NULL DEFAULT 0,  -- G0=0, G1=1, G2=2
        open INTEGER DEFAULT 1,
        seen TEXT,                -- "1-3" (dois observados), "1-3-4" (fechado)
        opened_at INTEGER,
        last_conf_short REAL,
        last_conf_long REAL,
        d_final INTEGER           -- deadline para forçar “X” no 3º
    )""")
    # patch em DB antigo que tinha stage sem NOT NULL
    try:
        cur.execute("UPDATE pending SET stage=0 WHERE stage IS NULL")
    except: pass

    # placar simples
    cur.execute("""CREATE TABLE IF NOT EXISTS score (
        id INTEGER PRIMARY KEY CHECK (id=1),
        green INTEGER DEFAULT 0,
        loss  INTEGER DEFAULT 0
    )""")
    if not cur.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        cur.execute("INSERT INTO score (id, green, loss) VALUES (1,0,0)")
    con.commit(); con.close()

migrate_db()

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

# ========= Timeline / n-gram simplificado =========
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
    # contexto (padrão “após X” se der)
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

    # precisa concordar (mesmo número) e passar limiares
    if b_s == b_l and c_s >= CONF_SHORT_MIN and c_l >= CONF_LONG_MIN and g_s >= GAP_MIN and g_l >= GAP_MIN:
        best = b_s
    else:
        best = None
    return best, c_s, c_l, len(tail_s), post_s, post_l

# ========= Parsers =========
ENTRY_RX = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
SEQ_RX   = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
AFTER_RX = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)

GREEN_RX = re.compile(r"(?:\bgr+e+e?n\b|\bwin\b|✅)", re.I)   # GREEN/GREEM… WIN, ✅
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
    seen_txt = "-".join(seen_list[:3])
    _exec_write("UPDATE pending SET seen=? WHERE id=?", (seen_txt, row_id))

def _ensure_final_deadline_when_two(row: sqlite3.Row):
    """Assim que houver 2 observados, abre o cronômetro de FINAL_TIMEOUT para forçar X se o 3º não vier."""
    if int(row["d_final"] or 0) > 0:
        return
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

    _exec_write("UPDATE pending SET open=0, seen=? WHERE id=?",
                ("-".join(final_seen[:3]), int(row["id"])))
    bump_score(outcome.upper())
    our_num_display = suggested if outcome=="GREEN" else "X"
    msg = (f"{'🟢' if outcome=='GREEN' else '🔴'} <b>{outcome}</b> — finalizado "
           f"(<b>{stage_lbl}</b>, nosso={our_num_display}, observados={'-'.join(final_seen[:3])}).\n"
           f"📊 Geral: {score_text()}")
    return msg

def open_pending(suggested: int, conf_short:float, conf_long:float):
    _exec_write("""INSERT INTO pending
        (created_at, suggested, stage, open, seen, opened_at, last_conf_short, last_conf_long, d_final)
        VALUES (?,?,?,?,?,?,?, ?, NULL)
    """, (now_ts(), int(suggested), 0, 1, "", now_ts(), float(conf_short), float(conf_long)))

def _maybe_force_close():
    """
    Destrava pendência:
      • Se há 2 observados e d_final expirou → adiciona 'X' e fecha.
      • Se há 3 observados mas ficou travado (sem número) por > FORCE_CLOSE_3 → fecha.
    """
    row = get_open_pending()
    if not row: return None
    seen_list = _seen_list(row)

    # caso 2 observados e cronômetro estourou: força X no 3º
    d_final = int(row["d_final"] or 0)
    if len(seen_list) == 2 and d_final > 0 and now_ts() >= d_final:
        seen_list.append("X")
        return _close_now(row, int(row["suggested"] or 0), seen_list)

    # caso 3 observados “pendurados” há muito tempo
    if len(seen_list) >= 3:
        opened_at = int(row["opened_at"] or now_ts())
        if now_ts() - opened_at >= FORCE_CLOSE_3:
            return _close_now(row, int(row["suggested"] or 0), seen_list)

    return None

# ========= Rotas =========
@app.get("/")
async def root():
    return {"ok": True, "service": "guardiao-auto-bot (GEN híbrido c/ timeout único)"}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    # destrava pendências antigas sempre que chegar algo
    forced = _maybe_force_close()
    if forced:
        await tg_send_text(TARGET_CHANNEL, forced)

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

    # 1) Fechamentos do fonte (GREEN/LOSS — números entre parênteses ou soltos)
    if GREEN_RX.search(text) or LOSS_RX.search(text):
        pend = get_open_pending()
        nums = parse_close_numbers(text)
        if pend and nums:
            seen = _seen_list(pend)
            for n in nums:
                if len(seen) >= 3: break
                seen.append(str(int(n)))

                # GREEN imediato: se bater nosso número em qualquer posição, fecha já
                suggested = int(pend["suggested"] or 0)
                obs_nums = [int(x) for x in seen if x.isdigit()]
                if (len(obs_nums) >= 1 and obs_nums[0] == suggested) or \
                   (len(obs_nums) >= 2 and obs_nums[1] == suggested) or \
                   (len(obs_nums) >= 3 and obs_nums[2] == suggested):
                    out = _close_now(pend, suggested, seen)
                    await tg_send_text(TARGET_CHANNEL, out)
                    return {"ok": True, "closed": "green_imediato"}

            # salva vistos
            _set_seen(int(pend["id"]), seen)

            # se ficou com 2 observados, arma/renova o deadline final
            pend = get_open_pending()
            if pend and len(_seen_list(pend)) == 2:
                _ensure_final_deadline_when_two(pend)

            # se já tem 3 observados e não bateu, fecha LOSS
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

    # antes de abrir novo, tenta destravar novamente
    forced2 = _maybe_force_close()
    if forced2:
        await tg_send_text(TARGET_CHANNEL, forced2)

    # Alimenta memória com a sequência bruta, se houver
    seq = parsed["seq"] or []
    if seq:
        append_timeline(seq)

    after = parsed["after"]
    best, conf_s, conf_l, samples_s, _post_s, _post_l = choose_single_number_hybrid(after)
    if best is None:
        return {"ok": True, "skipped_low_conf_or_disagree": True}

    # abre pendência nova
    open_pending(best, conf_s, conf_l)

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
        "ok": True, "db": DB_PATH,
        "pending_open": bool(pend),
        "pending_seen": (pend["seen"] if pend else ""),
        "d_final": int(pend["d_final"] or 0) if pend else 0,
        "time": ts_str()
    }