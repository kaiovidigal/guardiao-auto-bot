#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webhook_app.py
--------------
FastAPI + Telegram webhook para refletir sinais do canal-fonte e publicar
um "número seco" (modo GEN = sem restrição de paridade/tamanho) no canal-alvo.
Também acompanha os gales (G1/G2) com base nas mensagens do canal-fonte.

Fechamento robusto (mantido):
- Fecha SEMPRE comparando pelo(s) número(s) observados do canal-fonte:
  • GREEN se nosso número aparecer em 1º, 2º ou 3º -> G0/G1/G2.
  • LOSS se, após 3 observados, nosso número não aparece.
  • Se vierem apenas 2 observados, o 3º vira "X" (no próximo ENTRADA CONFIRMADA
    OU por timeout), e fecha. Se só houver 1 observado, mantém aberto.
Mensagens finais (agora com N-gram):
  🟢/🔴 + Geral + "📈 Amostra: N • Conf: X%"
  🔎 N-gram context (última análise):
     1 → a% | 2 → b% | 3 → c% | 4 → d%  (com o escolhido em negrito)

Qualidade (aproveita todos os sinais; filtros leves apenas no empate técnico):
- decay = 0.982   (mais estável, prioriza um pouco mais sequência longa)
- W4=0.46, W3=0.30, W2=0.16, W1=0.08
- GAP_SOFT (empate técnico) = 0.015 → fallback leve por frequência
- SEM bloqueio por MIN_SAMPLES/CONF_MIN/GAP_MIN
- Timeout para completar com X (apenas se já houver 2 observados): 180s

ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN
ENV opcionais: TARGET_CHANNEL, SOURCE_CHANNEL, DB_PATH
Webhook: POST /webhook/{WEBHOOK_TOKEN}
"""

import os, re, time, sqlite3, json
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
app = FastAPI(title="guardiao-auto-bot (GEN webhook)", version="2.4.0")

# ========= Parâmetros =========
# (ajuste cirúrgico na qualidade)
DECAY = 0.982
W4, W3, W2, W1 = 0.46, 0.30, 0.16, 0.08
GAP_SOFT       = 0.015         # anti-empate técnico (fallback leve)
OBS_TIMEOUT_SEC= 180           # completa com X (apenas se já houver 2 observados)

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
    # pending
    cur.execute("""CREATE TABLE IF NOT EXISTS pending (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER,
        suggested INTEGER,
        stage INTEGER DEFAULT 0,
        open INTEGER DEFAULT 1,
        seen TEXT,
        opened_at INTEGER,
        last_conf REAL,
        last_samples INTEGER,
        last_post TEXT
    )""")
    for col, ddl in [
        ("created_at", "ALTER TABLE pending ADD COLUMN created_at INTEGER"),
        ("suggested",  "ALTER TABLE pending ADD COLUMN suggested INTEGER"),
        ("stage",      "ALTER TABLE pending ADD COLUMN stage INTEGER DEFAULT 0"),
        ("open",       "ALTER TABLE pending ADD COLUMN open INTEGER DEFAULT 1"),
        ("seen",       "ALTER TABLE pending ADD COLUMN seen TEXT"),
        ("opened_at",  "ALTER TABLE pending ADD COLUMN opened_at INTEGER"),
        ("last_conf",  "ALTER TABLE pending ADD COLUMN last_conf REAL"),
        ("last_samples","ALTER TABLE pending ADD COLUMN last_samples INTEGER"),
        ("last_post",  "ALTER TABLE pending ADD COLUMN last_post TEXT"),
    ]:
        if not _column_exists(con, "pending", col):
            try: cur.execute(ddl)
            except sqlite3.OperationalError: pass

    # score
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

# ========= N-gram =========
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
    # empate técnico → fallback leve por frequência nos últimos 50
    top2 = sorted(post.items(), key=lambda kv: kv[1], reverse=True)[:2]
    if len(top2) == 2 and (top2[0][1] - top2[1][1]) < GAP_SOFT:
        last = tail[-50:] if len(tail) >= 50 else tail
        if last:
            freq = {c: last.count(c) for c in [1,2,3,4]}
            best = sorted([1,2,3,4], key=lambda x: (freq.get(x,0), x))[0]
            conf = 0.50
            return best, conf, len(tail), post
    best = max(post.items(), key=lambda kv: kv[1])[0]
    conf = float(post[best])
    return best, conf, len(tail), post

# ========= Parse =========
ENTRY_RX = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
SEQ_RX   = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
AFTER_RX = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)
GALE1_RX = re.compile(r"Estamos\s+no\s*1[ºo]\s*gale", re.I)
GALE2_RX = re.compile(r"Estamos\s+no\s*2[ºo]\s*gale", re.I)

# Fechamento tolerante
GREEN_RX = re.compile(r"(?:\bgr+e+e?n\b|\bwin\b|✅)", re.I)   # GREEN/GREEM/GREN, WIN, ✅
LOSS_RX  = re.compile(r"(?:\blo+s+s?\b|\bred\b|❌|\bperdemos\b)", re.I)  # LOSS/LOS, RED, ❌, PERDEMOS

# Fecha: números dentro de parênteses; se não houver, tenta ler números soltos 1..4
PAREN_GROUP_RX = re.compile(r"\(([^)]*)\)")
ANY_14_RX      = re.compile(r"[1-4]")

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
    """
    1) Extrai o último grupo entre parênteses: (1 | 4 | 4) -> [1,4,4]
    2) Se não houver parênteses, captura até 3 dígitos válidos 1..4 no texto inteiro.
    """
    t = re.sub(r"\s+", " ", text)
    groups = PAREN_GROUP_RX.findall(t)
    if groups:
        last = groups[-1]
        nums = re.findall(r"[1-4]", last)
        return [int(x) for x in nums][:3]
    nums = ANY_14_RX.findall(t)
    return [int(x) for x in nums][:3]

# ========= Pending helpers =========
def get_open_pending() -> Optional[sqlite3.Row]:
    con = _connect()
    row = con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone()
    con.close()
    return row

def open_pending(suggested: int, conf: float=0.0, samples: int=0, post: Optional[Dict[int,float]]=None):
    post_txt = json.dumps(post or {}, ensure_ascii=False)
    _exec_write("""INSERT INTO pending
                   (created_at, suggested, stage, open, seen, opened_at, last_conf, last_samples, last_post)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (now_ts(), int(suggested), 0, 1, "", now_ts(), float(conf), int(samples), post_txt))

def set_stage(stage:int):
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE pending SET stage=? WHERE open=1", (int(stage),))
    con.commit(); con.close()

def _seen_list(row: sqlite3.Row) -> List[str]:
    seen = (row["seen"] or "").strip()
    return [s for s in seen.split("-") if s]

def _seen_append(row: sqlite3.Row, new_items: List[str]):
    cur_seen = _seen_list(row)
    for it in new_items:
        if len(cur_seen) >= 3: break
        cur_seen.append(it)
    seen_txt = "-".join(cur_seen[:3])
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE pending SET seen=? WHERE id=?", (seen_txt, int(row["id"])))
    con.commit(); con.close()

def _stage_from_observed(suggested: int, obs: List[int]) -> Tuple[str, str]:
    if not obs:
        return ("LOSS", "G2")
    if len(obs) >= 1 and obs[0] == suggested: return ("GREEN", "G0")
    if len(obs) >= 2 and obs[1] == suggested: return ("GREEN", "G1")
    if len(obs) >= 3 and obs[2] == suggested: return ("GREEN", "G2")
    return ("LOSS", "G2")

def _fmt_ngram_context(post: Dict[int,float], best:int, samples:int, conf:float) -> str:
    # formata: "📈 Amostra: N • Conf: X%\n\n🔎 N-gram context (última análise):\n1 → a% | 2 → b% | 3 → c% | 4 → d%"
    lines = []
    lines.append(f"📈 Amostra: {int(samples)} • Conf: {conf*100:.1f}%")
    # ordena por chave 1..4 mantendo ordem visual 1-2-3-4
    parts = []
    for n in [1,2,3,4]:
        p = post.get(n, 0.0) * 100.0
        label = f"<b>{n}</b> → {p:.1f}%"
        parts.append(label)
    body = " | ".join(parts)
    return f"{lines[0]}\n\n🔎 N-gram context (última análise):\n{body}"

def _close_with_outcome(row: sqlite3.Row, outcome: str, final_seen: str, stage_lbl: str, suggested: int):
    our_num_display = suggested if outcome.upper()=="GREEN" else "X"

    # snapshot da análise para imprimir no fechamento
    try:
        conf = float(row["last_conf"] or 0.0)
    except Exception:
        conf = 0.0
    try:
        samples = int(row["last_samples"] or 0)
    except Exception:
        samples = 0
    try:
        post = json.loads(row["last_post"] or "{}")
        # chaves do sqlite vêm como strings; normaliza para int
        post = {int(k): float(v) for k,v in post.items()}
    except Exception:
        post = {}

    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE pending SET open=0, seen=? WHERE id=?", (final_seen, int(row["id"])))
    con.commit(); con.close()

    bump_score(outcome.upper())

    base_msg = (
        f"{'🟢' if outcome.upper()=='GREEN' else '🔴'} "
        f"<b>{outcome.upper()}</b> — finalizado "
        f"(<b>{stage_lbl}</b>, nosso={our_num_display}, observados={final_seen}).\n"
        f"📊 Geral: {score_text()}"
    )

    # anexa bloco N-gram (amostra dinâmica + distribuição)
    ngram_block = _fmt_ngram_context(post, suggested, samples, conf)
    msg = f"{base_msg}\n\n{ngram_block}"
    return msg

def _maybe_close_by_timeout():
    """
    Se passou muito tempo e temos EXATAMENTE 2 observados, completa com X e fecha.
    Se só houver 1 observado, mantém aberto (não coloca X na 2ª posição).
    """
    row = get_open_pending()
    if not row: return None
    opened_at = int(row["opened_at"] or row["created_at"] or now_ts())
    if now_ts() - opened_at < OBS_TIMEOUT_SEC:
        return None
    seen_list = _seen_list(row)
    if len(seen_list) == 2:
        seen_list.append("X")
        final_seen = "-".join(seen_list[:3])
        suggested = int(row["suggested"] or 0)
        obs_nums = [int(x) for x in seen_list if x.isdigit()]
        outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
        return _close_with_outcome(row, outcome, final_seen, stage_lbl, suggested)
    return None

def close_pending_force_fill_third_with_X_if_two():
    """
    Fecha imediatamente com X na 3ª posição se já houver 2 observados.
    Usado ao chegar a próxima ENTRADA CONFIRMADA.
    """
    row = get_open_pending()
    if not row: return None
    seen_list = _seen_list(row)
    if len(seen_list) == 2:
        seen_list.append("X")
        final_seen = "-".join(seen_list[:3])
        suggested = int(row["suggested"] or 0)
        obs_nums = [int(x) for x in seen_list if x.isdigit()]
        outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
        return _close_with_outcome(row, outcome, final_seen, stage_lbl, suggested)
    return None

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

    # timeout pode fechar pendência antiga (apenas se já houver 2 observados)
    timeout_msg = _maybe_close_by_timeout()
    if timeout_msg:
        await tg_send_text(TARGET_CHANNEL, timeout_msg)

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

    # 1) Gales (informativo)
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

    # 2) Fechamentos do fonte (GREEN/LOSS, com 1, 2 ou 3 números; variações greem/los/red)
    if GREEN_RX.search(text) or LOSS_RX.search(text):
        pend = get_open_pending()
        if pend:
            nums = parse_close_numbers(text)  # pode ter 1, 2 ou 3
            if nums:
                _seen_append(pend, [str(n) for n in nums])
                pend = get_open_pending()

            seen_list = _seen_list(pend) if pend else []
            if pend and len(seen_list) >= 3:
                suggested = int(pend["suggested"] or 0)
                obs_nums = [int(x) for x in seen_list if x.isdigit()]
                outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
                final_seen = "-".join(seen_list[:3])
                out_msg = _close_with_outcome(pend, outcome, final_seen, stage_lbl, suggested)
                await tg_send_text(TARGET_CHANNEL, out_msg)
                return {"ok": True, "closed": outcome.lower(), "seen": final_seen}
        return {"ok": True, "noted_close": True}

    # 3) Nova ENTRADA: NÃO usa a nova sequência para completar;
    #    se houver 2 observados no sinal anterior, fecha agora com X no 3º.
    parsed = parse_entry_text(text)
    if not parsed:
        return {"ok": True, "skipped": "nao_eh_entrada_confirmada"}

    pend = get_open_pending()
    if pend:
        # se já houver 3, força fechar antes de qualquer coisa
        seen_list = _seen_list(pend)
        if len(seen_list) >= 3:
            suggested = int(pend["suggested"] or 0)
            obs_nums = [int(x) for x in seen_list if x.isdigit()]
            outcome, stage_lbl = _stage_from_observed(suggested, obs_nums)
            final_seen = "-".join(seen_list[:3])
            out_msg = _close_with_outcome(pend, outcome, final_seen, stage_lbl, suggested)
            await tg_send_text(TARGET_CHANNEL, out_msg)
            pend = get_open_pending()

        # se ficou com 2 observados, fecha com X no 3º AGORA
        if pend:
            msgx = close_pending_force_fill_third_with_X_if_two()
            if msgx:
                await tg_send_text(TARGET_CHANNEL, msgx)
                pend = get_open_pending()

        # se ainda existir pendente (ex.: só 1 observado), NÃO abre novo
        if pend:
            return {"ok": True, "kept_open_waiting_more_observed": True}

    # Alimenta memória e decide novo tiro (sem travas por min/conf/gap)
    seq = parsed["seq"] or []
    if seq: append_seq(seq)

    after = parsed["after"]
    best, conf, samples, post = choose_single_number(after)

    # Publica novo tiro (agora salvando snapshot p/ imprimir no fechamento)
    open_pending(best, conf=conf, samples=samples, post=post)
    aft_txt = f" após {after}" if after else ""
    txt = (
        f"🎯 <b>Número seco (G0):</b> <b>{best}</b>\n"
        f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
        f"📊 <b>Conf:</b> {conf*100:.2f}% | <b>Amostra≈</b>{samples}"
    )
    await tg_send_text(TARGET_CHANNEL, txt)
    return {"ok": True, "posted": True, "best": best, "conf": conf, "samples": samples}

# ===== Debug/help endpoint =====
@app.get("/health")
async def health():
    pend = get_open_pending()
    pend_open = bool(pend)
    seen = (pend["seen"] if pend else "")
    return {"ok": True, "db": DB_PATH, "pending_open": pend_open, "pending_seen": seen, "time": ts_str()}