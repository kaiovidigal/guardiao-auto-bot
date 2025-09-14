#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webhook_app.py
--------------
FastAPI + Telegram webhook para refletir sinais do canal-fonte e publicar
um "número seco" (modo GEN = sem restrição de paridade/tamanho) no canal-alvo.
Também acompanha os gales (G1/G2) com base nas mensagens do canal-fonte.

Regras reforçadas:
- Só fecha GREEN/LOSS após conferir 3 números.
- Se o fonte encerrar antes (ex.: GREEN no G1), o bot espera a próxima
  ENTRADA CONFIRMADA e usa os 3 primeiros números da nova "🚥 Sequência" para
  completar a conferência do sinal anterior.
"""

import os, re, json, time, sqlite3, asyncio
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, HTTPException

# ========= ENV =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1002796105884").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()  # se vazio, não filtra

DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"
TELEGRAM_API   = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

if not TG_BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")
if not WEBHOOK_TOKEN:
    raise RuntimeError("Defina WEBHOOK_TOKEN no ambiente.")

# ========= App =========
app = FastAPI(title="guardiao-auto-bot (GEN webhook)", version="2.2.0")

# ========= Utils =========
def now_ts() -> int:
    return int(time.time())

def ts_str(ts=None) -> str:
    if ts is None: ts = now_ts()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ========= DB helpers =========
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    # tentar reduzir "database is locked"
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

def _column_exists(con: sqlite3.Connection, table: str, col: str) -> bool:
    r = con.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == col or row[1] == col for row in r)

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
        seen TEXT
    )""")
    # garantir colunas (idempotente)
    if not _column_exists(con, "pending", "created_at"):
        cur.execute("ALTER TABLE pending ADD COLUMN created_at INTEGER")
    if not _column_exists(con, "pending", "suggested"):
        cur.execute("ALTER TABLE pending ADD COLUMN suggested INTEGER")
    if not _column_exists(con, "pending", "stage"):
        cur.execute("ALTER TABLE pending ADD COLUMN stage INTEGER DEFAULT 0")
    if not _column_exists(con, "pending", "open"):
        cur.execute("ALTER TABLE pending ADD COLUMN open INTEGER DEFAULT 1")
    if not _column_exists(con, "pending", "seen"):
        cur.execute("ALTER TABLE pending ADD COLUMN seen TEXT")

    # score (geral de GREEN/LOSS)
    cur.execute("""CREATE TABLE IF NOT EXISTS score (
        id INTEGER PRIMARY KEY CHECK (id=1),
        green INTEGER DEFAULT 0,
        loss INTEGER DEFAULT 0
    )""")
    # seed linha 1 se precisar
    row = con.execute("SELECT 1 FROM score WHERE id=1").fetchone()
    if not row:
        cur.execute("INSERT INTO score (id, green, loss) VALUES (1,0,0)")

    con.commit(); con.close()

migrate_db()

# ========= Score helpers (GERAL) =========
def bump_score(outcome: str) -> Tuple[int, int]:
    con = _connect(); cur = con.cursor()
    row = cur.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
    g, l = (row["green"], row["loss"]) if row else (0, 0)
    if outcome.upper() == "GREEN":
        g += 1
    elif outcome.upper() == "LOSS":
        l += 1
    cur.execute("INSERT OR REPLACE INTO score (id, green, loss) VALUES (1,?,?)", (g, l))
    con.commit(); con.close()
    return g, l

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

# ========= N-gram memória =========
def get_tail(limit:int=400) -> List[int]:
    con = _connect()
    rows = con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def _exec_write(sql: str, params: tuple=()):
    # robust com retry para "database is locked"
    for attempt in range(6):
        try:
            con = _connect()
            cur = con.cursor()
            cur.execute(sql, params)
            con.commit()
            con.close()
            return
        except sqlite3.OperationalError as e:
            emsg = str(e).lower()
            if "locked" in emsg or "busy" in emsg:
                time.sleep(0.25*(attempt+1))
                continue
            raise

def append_seq(seq: List[int]):
    if not seq: return
    for n in seq:
        _exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)",
                    (now_ts(), int(n)))
    _update_ngrams()

def _update_ngrams(decay: float=0.985, max_n:int=5, window:int=400):
    tail = get_tail(window)
    if len(tail) < 2: return
    con = _connect(); cur = con.cursor()
    for t in range(1, len(tail)):
        nxt = int(tail[t])
        dist = (len(tail)-1) - t
        w = decay ** dist
        for n in range(2, max_n+1):
            if t-(n-1) < 0: break
            ctx = tail[t-(n-1):t]
            ctx_key = ",".join(str(x) for x in ctx)
            cur.execute("""
              INSERT INTO ngram (n, ctx, nxt, w)
              VALUES (?,?,?,?)
              ON CONFLICT(n, ctx, nxt) DO UPDATE SET w = w + excluded.w
            """, (n, ctx_key, nxt, float(w)))
    con.commit(); con.close()

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

W4, W3, W2, W1 = 0.40, 0.30, 0.20, 0.10
def _ngram_backoff(tail: List[int], after: Optional[int], cand:int) -> float:
    if not tail: return 0.0
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
    s = 0.0
    if len(ctx4)==4: s += W4 * _prob_from_ngrams(ctx4[:-1], cand)
    if len(ctx3)==3: s += W3 * _prob_from_ngrams(ctx3[:-1], cand)
    if len(ctx2)==2: s += W2 * _prob_from_ngrams(ctx2[:-1], cand)
    if len(ctx1)==1: s += W1 * _prob_from_ngrams(ctx1[:-1], cand)
    return s

def choose_single_number(after: Optional[int]) -> Tuple[int, float, int]:
    cands = [1,2,3,4]
    tail = get_tail(400)
    scores = {c: _ngram_backoff(tail, after, c) for c in cands}
    if all(v == 0.0 for v in scores.values()):
        last = tail[-50:] if len(tail) >= 50 else tail
        freq = {c: last.count(c) for c in cands}
        best = sorted(cands, key=lambda x: (freq.get(x,0), x))[0]
        conf = 0.50
        return best, conf, len(tail)
    total = sum(scores.values()) or 1e-9
    post = {k: v/total for k,v in scores.items()}
    best = max(post.items(), key=lambda kv: kv[1])[0]
    conf = float(post[best])
    return best, conf, len(tail)

# ========= Parse =========
ENTRY_RX = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
SEQ_RX = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
AFTER_RX = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)
GALE1_RX = re.compile(r"Estamos\s+no\s*1[ºo]\s*gale", re.I)
GALE2_RX = re.compile(r"Estamos\s+no\s*2[ºo]\s*gale", re.I)

# Resultados no "APOSTA ENCERRADA": extrai números dentro de parênteses, ex. "(1 | 4 | 4)" ou "(4 | 2)"
PAREN_GROUP_RX = re.compile(r"\(([^)]*)\)")
DIGIT_RX = re.compile(r"[1-4]")

def parse_entry_text(text: str) -> Optional[Dict]:
    t = re.sub(r"\s+", " ", text).strip()
    if not ENTRY_RX.search(t):
        return None
    mseq = SEQ_RX.search(t)
    seq = []
    if mseq:
        parts = re.findall(r"[1-4]", mseq.group(1))
        seq = [int(x) for x in parts]
    mafter = AFTER_RX.search(t)
    after_num = int(mafter.group(1)) if mafter else None
    return {"seq": seq, "after": after_num, "raw": t}

def extract_result_numbers(text: str) -> List[int]:
    """Tenta pegar os números de resultado do 'APOSTA ENCERRADA' (GREEN/RED) do fonte."""
    t = re.sub(r"\s+", " ", text)
    groups = PAREN_GROUP_RX.findall(t)
    if not groups:
        return []
    # Usa o último grupo de parênteses (normalmente o que contém 1|4|4 ou 4|2)
    last = groups[-1]
    nums = DIGIT_RX.findall(last)
    return [int(x) for x in nums]

# ========= Pending helpers =========
def get_open_pending() -> Optional[sqlite3.Row]:
    con = _connect()
    row = con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone()
    con.close()
    return row

def open_pending(suggested: int):
    _exec_write("""INSERT INTO pending (created_at, suggested, stage, open, seen)
                   VALUES (?,?,?,?,?)""",
                (now_ts(), int(suggested), 0, 1, ""))

def set_stage(stage:int):
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE pending SET stage=? WHERE open=1", (int(stage),))
    con.commit(); con.close()

def set_seen_list(lst: List[int]):
    s = ",".join(str(x) for x in lst[:3])
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE pending SET seen=? WHERE open=1", (s,))
    con.commit(); con.close()

def append_seen_numbers(nums: List[int]):
    if not nums:
        return
    con = _connect(); cur = con.cursor()
    row = cur.execute("SELECT seen FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        con.close(); return
    seen = (row["seen"] or "").strip()
    current = [int(x) for x in seen.split(",") if x.strip().isdigit()]
    for n in nums:
        if len(current) >= 3:
            break
        current.append(int(n))
    s = ",".join(str(x) for x in current[:3])
    cur.execute("UPDATE pending SET seen=? WHERE open=1", (s,))
    con.commit(); con.close()

def get_seen_list_from_row(pend_row: sqlite3.Row) -> List[int]:
    seen = (pend_row["seen"] or "").strip()
    return [int(x) for x in seen.split(",") if x.strip().isdigit()]

def close_pending(outcome:str):
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE pending SET open=0, seen=? WHERE open=1", (outcome,))
    con.commit(); con.close()

def _stage_label_from_index(idx: int) -> str:
    return "G0" if idx == 0 else ("G1" if idx == 1 else "G2")

def _stage_label(stage_val: Optional[int]) -> str:
    try:
        s = int(stage_val or 0)
    except Exception:
        s = 0
    return "G0" if s == 0 else ("G1" if s == 1 else "G2")

def _resolve_if_ready_and_close():
    """Se já houver 3 números em seen, resolve imediatamente e fecha."""
    pend = get_open_pending()
    if not pend:
        return None
    seen_list = get_seen_list_from_row(pend)
    if len(seen_list) < 3:
        return None

    suggested = int(pend["suggested"])
    outcome = "LOSS"
    stage_lbl = "G2"
    # encontra primeira ocorrência do sugerido entre os 3:
    try:
        idx = seen_list.index(suggested)
        if idx in (0,1,2):
            outcome = "GREEN"
            stage_lbl = _stage_label_from_index(idx)
    except ValueError:
        outcome = "LOSS"
        stage_lbl = "G2"

    close_pending(outcome)
    bump_score(outcome)

    # Mensagem com número sugerido e geral:
    icon = "🟢" if outcome == "GREEN" else "🔴"
    txt = (
        f"{icon} <b>{outcome}</b> — finalizado (<b>{stage_lbl}</b>, número <b>{suggested}</b>).\n"
        f"📊 Geral: {score_text()}"
    )
    return txt

# ========= Telegram =========
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN: return
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                "disable_web_page_preview": True})

# ========= Rotas =========
@app.get("/")
async def root():
    return {"ok": True, "service": "guardiao-auto-bot (GEN webhook)"}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()
    msg = data.get("channel_post") or data.get("message") \
        or data.get("edited_channel_post") or data.get("edited_message") or {}

    text = (msg.get("text") or msg.get("caption") or "").strip()
    chat = msg.get("chat") or {}
    chat_id = str(chat.get("id") or "")
    # se SOURCE_CHANNEL estiver definido, filtra
    if SOURCE_CHANNEL and chat_id != str(SOURCE_CHANNEL):
        return {"ok": True, "skipped": "outro_chat"}

    if not text:
        return {"ok": True, "skipped": "sem_texto"}

    t_norm = re.sub(r"\s+", " ", text)

    # ===== 0) Se for "APOSTA ENCERRADA": tente coletar números do resultado (1..3) e resolver se conseguir 3
    if "APOSTA ENCERRADA" in t_norm.upper():
        nums = extract_result_numbers(text)  # pode retornar 1, 2 ou 3 números
        if nums and get_open_pending():
            append_seen_numbers(nums)
            # tenta resolver se já tem 3
            maybe_msg = _resolve_if_ready_and_close()
            if maybe_msg:
                await tg_send_text(TARGET_CHANNEL, maybe_msg)
                return {"ok": True, "closed_by_result": True}
        # mesmo sem números suficientes, só anota — fechamento ficará para a próxima ENTRADA
        return {"ok": True, "result_noted": True, "nums": nums}

    # ===== 1) Gales — apenas anotam estágio; não fecham
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

    # ===== 2) Nova ENTRADA CONFIRMADA
    parsed = parse_entry_text(text)
    if not parsed:
        return {"ok": True, "skipped": "nao_eh_entrada_confirmada"}

    # (2.a) Se existir pendência aberta e ela ainda não tem 3 números,
    # use os 3 primeiros da nova "Sequência" para completar a conferência do sinal anterior.
    pend = get_open_pending()
    if pend:
        seen_list = get_seen_list_from_row(pend)
        if len(seen_list) < 3:
            seq = parsed["seq"] or []
            fill = seq[:3 - len(seen_list)]
            if fill:
                new_seen = seen_list + fill
                set_seen_list(new_seen)
            # tenta resolver agora
            maybe_msg = _resolve_if_ready_and_close()
            if maybe_msg:
                await tg_send_text(TARGET_CHANNEL, maybe_msg)
            else:
                # se mesmo assim não completou 3, mantém pendente e não abre novo
                if len(get_seen_list_from_row(get_open_pending())) < 3:
                    return {"ok": True, "kept_open_waiting_3": True}

            # após fechar (se fechou), continua e abre a nova pendência normalmente

        else:
            # já tinha 3 por algum motivo; resolve e depois abre novo
            maybe_msg = _resolve_if_ready_and_close()
            if maybe_msg:
                await tg_send_text(TARGET_CHANNEL, maybe_msg)

        # Recarrega pendência — pode ter fechado
        pend = get_open_pending()

        # Se ainda existir aberta aqui, é porque não conseguiu completar 3; então não abra outra
        if pend:
            return {"ok": True, "ignored": "ja_existe_pendente_sem_3_numeros"}

    # (2.b) Alimenta memória de sequência (se vier algo), antes de decidir
    seq_new = parsed["seq"] or []
    if seq_new:
        append_seq(seq_new)

    after = parsed["after"]
    best, conf, samples = choose_single_number(after)

    # (2.c) Abre pendência e publica novo tiro seco
    open_pending(best)
    aft_txt = f" após {after}" if after else ""
    txt = (
        f"🎯 <b>Número seco (G0):</b> <b>{best}</b>\n"
        f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
        f"📊 <b>Conf:</b> {conf*100:.2f}% | <b>Amostra≈</b>{samples}"
    )
    await tg_send_text(TARGET_CHANNEL, txt)

    return {"ok": True, "posted": True, "best": best, "conf": conf, "samples": samples}

# ===== Debug/help endpoints (opcionais) =====
@app.get("/health")
async def health():
    pend = bool(get_open_pending())
    return {"ok": True, "db": DB_PATH, "pending_open": pend, "time": ts_str()}