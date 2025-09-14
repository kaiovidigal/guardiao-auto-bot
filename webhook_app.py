#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webhook_app.py
--------------
FastAPI + Telegram webhook para refletir sinais do canal-fonte e publicar
um "número seco" (modo GEN = sem restrição de paridade/tamanho) no canal-alvo.
Também acompanha os gales (G1/G2) com base nas mensagens do canal-fonte.

Fechamento robusto:
- Fecha SEMPRE comparando pelo(s) número(s) observados do canal-fonte:
  • GREEN se nosso número aparecer em 1º, 2º ou 3º -> G0/G1/G2.
  • LOSS se, após 3 observados, nosso número não aparece.
  • Se não surgir o 3º observado em tempo hábil (timeout), fecha como LOSS com "X" no(s) faltante(s).
Mensagens finais:
  🟢 GREEN — finalizado (G1, nosso=3, observados=1-3-4).
  🔴 LOSS  — finalizado (G2, nosso=2, observados=1-4-X).
E sempre adiciona "📊 Geral: <greens> GREEN × <loss> LOSS — <acc>%".

Qualidade (modelo “sem travas de ritmo, mas com filtros”):
- decay = 0.980
- W4=0.42, W3=0.30, W2=0.18, W1=0.10
- MIN_SAMPLES = 1500
- CONF_MIN = 0.60
- GAP_MIN = 0.050
- GAP_SOFT (empate técnico) = 0.015 → cai para fallback leve por frequência
- *Sem antispam/limites de hora ou cooldown pós-loss (aproveita todos os sinais)*

ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN
ENV opcionais: TARGET_CHANNEL, SOURCE_CHANNEL, DB_PATH
Webhook: POST /webhook/{WEBHOOK_TOKEN}
"""

import os, re, time, sqlite3
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
app = FastAPI(title="guardiao-auto-bot (GEN webhook)", version="2.3.1")

# ========= Parâmetros de qualidade =========
DECAY = 0.980
W4, W3, W2, W1 = 0.42, 0.30, 0.18, 0.10
MIN_SAMPLES = 1500
CONF_MIN    = 0.60
GAP_MIN     = 0.050         # gap mínimo para atirar
GAP_SOFT    = 0.015         # anti-empate técnico (se <, usa fallback leve)
OBS_TIMEOUT_SEC = 180       # se não vier 3º número em ~3min, fecha com X

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
    # minimizar "database is locked"
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

def _column_exists(con: sqlite3.Connection, table: str, col: str) -> bool:
    r = con.execute(f"PRAGMA table_info({table})").fetchall()
    return any((row["name"] if isinstance(row, sqlite3.Row) else row[1]) == col for row in r)

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
    # pending (1 por vez)
    cur.execute("""CREATE TABLE IF NOT EXISTS pending (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER,
        suggested INTEGER,
        stage INTEGER DEFAULT 0,
        open INTEGER DEFAULT 1,
        seen TEXT,
        opened_at INTEGER,
        observed_from_close INTEGER DEFAULT 0
    )""")
    # garantir colunas (idempotente)
    for col, ddl in [
        ("created_at", "ALTER TABLE pending ADD COLUMN created_at INTEGER"),
        ("suggested",  "ALTER TABLE pending ADD COLUMN suggested INTEGER"),
        ("stage",      "ALTER TABLE pending ADD COLUMN stage INTEGER DEFAULT 0"),
        ("open",       "ALTER TABLE pending ADD COLUMN open INTEGER DEFAULT 1"),
        ("seen",       "ALTER TABLE pending ADD COLUMN seen TEXT"),
        ("opened_at",  "ALTER TABLE pending ADD COLUMN opened_at INTEGER"),
        ("observed_from_close", "ALTER TABLE pending ADD COLUMN observed_from_close INTEGER DEFAULT 0"),
    ]:
        if not _column_exists(con, "pending", col):
            try: cur.execute(ddl)
            except sqlite3.OperationalError: pass

    # score (geral GREEN/LOSS)
    cur.execute("""CREATE TABLE IF NOT EXISTS score (
        id INTEGER PRIMARY KEY CHECK (id=1),
        green INTEGER DEFAULT 0,
        loss  INTEGER DEFAULT 0
    )""")
    row = con.execute("SELECT 1 FROM score WHERE id=1").fetchone()
    if not row:
        cur.execute("INSERT INTO score (id, green, loss) VALUES (1,0,0)")

    con.commit(); con.close()
migrate_db()

def _exec_write(sql: str, params: tuple=()):
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

# ========= Score helpers =========
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

def append_seq(seq: List[int]):
    if not seq: return
    for n in seq:
        _exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)",
                    (now_ts(), int(n)))
    _update_ngrams()

def _update_ngrams(decay: float=DECAY, max_n:int=5, window:int=400):
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

# ======== boosts/tie/temperatura (NÃO mudam estrutura) ========
def _sharpen(post: Dict[int,float], tau: float=0.75) -> Dict[int,float]:
    raw = {k: (v**(1.0/max(1e-6, tau))) for k,v in post.items()}
    s = sum(raw.values()) or 1e-9
    return {k: v/s for k,v in raw.items()}

def _rare_rising_boost(tail: List[int], post: Dict[int,float]) -> Dict[int,float]:
    if not tail: return post
    long = tail[-200:] if len(tail)>=200 else tail
    short= tail[-40:]  if len(tail)>=40  else tail
    from collections import Counter
    cL, cS = Counter(long), Counter(short)
    m = {}
    for n,p in post.items():
        rateL = (cL[n]/max(1,len(long)))
        rateS = (cS[n]/max(1,len(short)))
        boost = 1.03 if (rateS > rateL*1.25 and rateS < 0.40) else 1.0
        m[n] = p*boost
    s = sum(m.values()) or 1e-9
    return {k: v/s for k,v in m.items()}

def _time_bucket_bias(post: Dict[int,float]) -> Dict[int,float]:
    bucket = int(datetime.now(timezone.utc).minute >= 30)
    bias = {1:1.00, 2:(1.01 if bucket==0 else 0.995), 3:1.00, 4:(1.01 if bucket==1 else 0.995)}
    m = {k: post[k]*bias.get(k,1.0) for k in post}
    s = sum(m.values()) or 1e-9
    return {k: v/s for k,v in m.items()}

def _after_laplace_boost(tail: List[int], after: Optional[int], post: Dict[int,float]) -> Dict[int,float]:
    if after is None or after not in tail: return post
    pairs = sum(1 for i in range(len(tail)-1) if tail[i]==after)
    from collections import Counter
    c = Counter(tail[i+1] for i in range(len(tail)-1) if tail[i]==after)
    m={}
    for k,p in post.items():
        lap = (c.get(k,0)+1.0)/(pairs+4.0) if pairs>0 else 0.25
        boost = 1.0 + min(0.03, max(0.0, (lap-0.25)*0.20))
        m[k]=p*boost
    s=sum(m.values()) or 1e-9
    return {k:v/s for k,v in m.items()}

def _anti_runlen(tail: List[int], post: Dict[int,float]) -> Dict[int,float]:
    last = tail[-10:] if len(tail)>=10 else tail
    from collections import Counter
    c = Counter(last)
    m={k:(v*(0.95 if c.get(k,0)>=3 else 1.0)) for k,v in post.items()}
    s=sum(m.values()) or 1e-9
    return {k:v/s for k,v in m.items()}

# ========= posterior bruto =========
def _post_from_tail(tail: List[int], after: Optional[int]) -> Dict[int, float]:
    cands = [1,2,3,4]
    scores = {c: 0.0 for c in cands}
    if not tail:
        return {c: 0.25 for c in cands}
    # contexto
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
        scores[c] = s
    tot = sum(scores.values()) or 1e-9
    return {k: v/tot for k,v in scores.items()}

def choose_single_number(after: Optional[int]) -> Tuple[int, float, int, Dict[int,float]]:
    tail = get_tail(400)
    post = _post_from_tail(tail, after)

    # Empate técnico → fallback leve por frequência (como já estava)
    top2 = sorted(post.items(), key=lambda kv: kv[1], reverse=True)[:2]
    if len(top2) == 2 and (top2[0][1] - top2[1][1]) < GAP_SOFT:
        last = tail[-50:] if len(tail) >= 50 else tail
        if last:
            freq = {c: last.count(c) for c in [1,2,3,4]}
            best = sorted([1,2,3,4], key=lambda x: (freq.get(x,0), x))[0]
            conf = 0.50
            return best, conf, len(tail), post

    # >>>>>>> melhorias de qualidade (não mudam estrutura) <<<<<<<
    post = _rare_rising_boost(tail, post)
    post = _after_laplace_boost(tail, after, post)
    post = _anti_runlen(tail, post)

    # Desempate por meia-hora se top1~top2 < 0.02
    top = sorted(post.items(), key=lambda x: x[1], reverse=True)
    if len(top)>=2 and (top[0][1]-top[1][1])<0.02:
        post = _time_bucket_bias(post)

    # Temperaturas (duas afiadas leves)
    post = _sharpen(post, tau=0.75)
    post = _sharpen(post, tau=0.90)

    best = max(post.items(), key=lambda kv: kv[1])[0]
    conf = float(post[best])
    return best, conf, len(tail), post

# ========= Parse =========
ENTRY_RX = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
SEQ_RX   = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
AFTER_RX = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)
GALE1_RX = re.compile(r"Estamos\s+no\s*1[ºo]\s*gale", re.I)
GALE2_RX = re.compile(r"Estamos\s+no\s*2[ºo]\s*gale", re.I)
GREEN_RX = re.compile(r"(green|✅|win)", re.I)
LOSS_RX  = re.compile(r"(loss|red|❌|perdemos)", re.I)
CLOSE_NUMS_RX = re.compile(r"\(([^)]+)\)")

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

def parse_close_numbers(text: str) -> List[int]:
    """Extrai números dentro de parênteses no fechamento. Ex.: '(1 | 4 | 4)' -> [1,4,4]."""
    m = CLOSE_NUMS_RX.search(re.sub(r"\s+", " ", text))
    if not m:
        return []
    nums = re.findall(r"[1-4]", m.group(1))
    return [int(x) for x in nums][:3]

# ========= Pending helpers =========
def get_open_pending() -> Optional[sqlite3.Row]:
    con = _connect()
    row = con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone()
    con.close()
    return row

def open_pending(suggested: int):
    _exec_write("""INSERT INTO pending (created_at, suggested, stage, open, seen, opened_at, observed_from_close)
                   VALUES (?,?,?,?,?,?,?)""",
                (now_ts(), int(suggested), 0, 1, "", now_ts(), 0))

def set_stage(stage:int):
    # apenas informativo; o stage final usado na msg vem dos 3 observados
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE pending SET stage=? WHERE open=1", (int(stage),))
    con.commit(); con.close()

def _seen_list(row: sqlite3.Row) -> List[str]:
    seen = (row["seen"] or "").strip()
    return [s for s in seen.split("-") if s]

def _seen_append(row: sqlite3.Row, new_items: List[str], from_close: bool=False):
    cur_seen = _seen_list(row)
    for it in new_items:
        if len(cur_seen) >= 3: break
        cur_seen.append(it)
    seen_txt = "-".join(cur_seen[:3])
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE pending SET seen=?, observed_from_close=? WHERE id=?",
                (seen_txt, 1 if from_close else (row["observed_from_close"] or 0), int(row["id"])))
    con.commit(); con.close()

def _stage_label(stage_val: Optional[int]) -> str:
    try:
        s = int(stage_val or 0)
    except Exception:
        s = 0
    return "G0" if s == 0 else ("G1" if s == 1 else "G2")

def _stage_from_observed(suggested: int, obs: List[int]) -> Tuple[str, str]:
    """
    Retorna (outcome, stage_label) dados os observados (até 3) e o nosso sugerido.
    """
    if not obs:
        return ("LOSS", "G2")
    if len(obs) >= 1 and obs[0] == suggested:
        return ("GREEN", "G0")
    if len(obs) >= 2 and obs[1] == suggested:
        return ("GREEN", "G1")
    if len(obs) >= 3 and obs[2] == suggested:
        return ("GREEN", "G2")
    return ("LOSS", "G2")

def _close_with_outcome(row: sqlite3.Row, outcome: str, final_seen: str, our_stage_label: str, our_num: int|str):
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE pending SET open=0, seen=? WHERE id=?", (final_seen, int(row["id"])))
    con.commit(); con.close()

    bump_score(outcome.upper())
    msg = (
        f"{'🟢' if outcome.upper()=='GREEN' else '🔴'} "
        f"<b>{outcome.upper()}</b> — finalizado "
        f"(<b>{our_stage_label}</b>, nosso={our_num}, observados={final_seen}).\n"
        f"📊 Geral: {score_text()}"
    )
    return msg

def _maybe_close_by_timeout():
    """Se passou muito tempo e só temos 1-2 observados, completa com X e fecha."""
    row = get_open_pending()
    if not row: return None
    opened_at = int(row["opened_at"] or row["created_at"] or now_ts())
    if now_ts() - opened_at < OBS_TIMEOUT_SEC:
        return None
    seen_list = _seen_list(row)
    if 1 <= len(seen_list) < 3:
        # completa com X e fecha
        while len(seen_list) < 3:
            seen_list.append("X")
        final_seen = "-".join(seen_list[:3])
        suggested = int(row["suggested"] or 0)
        obs_nums = [int(x) for x in seen_list if x.isdigit()]
        outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
        msg = _close_with_outcome(row, outcome, final_seen, stage_lbl, suggested)
        return msg
    return None

def close_pending(outcome:str):
    """Compat: força fechar preenchendo X até 3 observados."""
    row = get_open_pending()
    if not row: return
    seen_list = _seen_list(row)
    while len(seen_list) < 3:
        seen_list.append("X")
    final_seen = "-".join(seen_list[:3])
    suggested = int(row["suggested"] or 0)
    obs_nums = [int(x) for x in seen_list if x.isdigit()]
    outcome2, stage_lbl = _stage_from_observed(suggested, obs_nums)
    msg = _close_with_outcome(row, outcome2, final_seen, stage_lbl, suggested)
    return msg

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

    # fecha por timeout (se travado aguardando 3º número)
    timeout_msg = _maybe_close_by_timeout()
    if timeout_msg:
        await tg_send_text(TARGET_CHANNEL, timeout_msg)

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

    # 1) Mensagens informativas de gales
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

    # 2) Fechamentos do fonte: GREEN/LOSS + números (quando houver)
    if re.search(r"(GREEN|WIN|✅|LOSS|RED|❌|perdemos)", text, re.I):
        pend = get_open_pending()
        if pend:
            nums = parse_close_numbers(text)  # [n1, n2, n3]? às vezes vem só 1-2
            if nums:
                _seen_append(pend, [str(n) for n in nums], from_close=True)
                pend = get_open_pending()
            seen_list = _seen_list(pend) if pend else []
            # fecha apenas quando já temos 3 observados (ou timeout fará X)
            if pend and len(seen_list) >= 3:
                suggested = int(pend["suggested"] or 0)
                obs_nums = [int(x) for x in seen_list if x.isdigit()]
                outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
                final_seen = "-".join(seen_list[:3])
                out_msg = _close_with_outcome(pend, outcome, final_seen, stage_lbl, suggested)
                await tg_send_text(TARGET_CHANNEL, out_msg)
                return {"ok": True, "closed": outcome.lower(), "seen": final_seen}
        return {"ok": True, "noted_close": True}

    # 3) Nova ENTRADA
    parsed = parse_entry_text(text)
    if not parsed:
        return {"ok": True, "skipped": "nao_eh_entrada_confirmada"}

    # Se já existe pendência aberta:
    pend = get_open_pending()
    if pend:
        # se está com 1-2 observados pendentes, completa com X e fecha agora
        seen_list = _seen_list(pend)
        if 1 <= len(seen_list) < 3:
            while len(seen_list) < 3:
                seen_list.append("X")
            final_seen = "-".join(seen_list[:3])
            suggested = int(pend["suggested"] or 0)
            obs_nums = [int(x) for x in seen_list if x.isdigit()]
            outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
            out_msg = _close_with_outcome(pend, outcome, final_seen, stage_lbl, suggested)
            await tg_send_text(TARGET_CHANNEL, out_msg)
        else:
            # se já tinha 3 observados mas faltou fechar por algum motivo, força fechar
            if len(seen_list) >= 3:
                final_seen = "-".join(seen_list[:3])
                suggested = int(pend["suggested"] or 0)
                obs_nums = [int(x) for x in seen_list if x.isdigit()]
                outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
                out_msg = _close_with_outcome(pend, outcome, final_seen, stage_lbl, suggested)
                await tg_send_text(TARGET_CHANNEL, out_msg)
        # segue para abrir a nova entrada

    # Alimenta memória de sequência (se vier algo), antes de decidir
    seq = parsed["seq"] or []
    if seq:
        append_seq(seq)

    after = parsed["after"]
    best, conf, samples, post = choose_single_number(after)

    # Critérios de qualidade (sem travas de ritmo)
    top2 = sorted(post.items(), key=lambda kv: kv[1], reverse=True)[:2]
    gap = (top2[0][1] - (top2[1][1] if len(top2)>1 else 0.0)) if top2 else 0.0
    if samples < MIN_SAMPLES:
        return {"ok": True, "abstain": "amostra_insuficiente", "samples": samples}
    if conf < CONF_MIN or gap < GAP_MIN:
        return {"ok": True, "abstain": "conf_ou_gap_baixo", "conf": conf, "gap": gap}

    # Abre pendência e publica (SEM antispam/cooldown)
    open_pending(best)
    aft_txt = f" após {after}" if after else ""
    txt = (
        f"🎯 <b>Número seco (G0):</b> <b>{best}</b>\n"
        f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
        f"📊 <b>Conf:</b> {conf*100:.2f}% | <b>Amostra≈</b>{samples}"
    )
    await tg_send_text(TARGET_CHANNEL, txt)

    return {"ok": True, "posted": True, "best": best, "conf": conf, "samples": samples, "gap": gap}

# ===== Debug/help endpoint =====
@app.get("/health")
async def health():
    pend = get_open_pending()
    pend_open = bool(pend)
    seen = (pend["seen"] if pend else "")
    return {
        "ok": True,
        "db": DB_PATH,
        "pending_open": pend_open,
        "pending_seen": seen,
        "time": ts_str()
    }