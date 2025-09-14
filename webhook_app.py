#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webhook_app.py
--------------
FastAPI + Telegram webhook para refletir sinais do canal-fonte e publicar
um "número seco" (modo GEN = sem restrição de paridade/tamanho) no canal-alvo.
Também acompanha os gales (G1/G2) com base nas mensagens do canal-fonte.

Mudanças (v2.2.0):
- Fechamento baseado APENAS no número observado no canal-fonte:
  * 1º número após a entrada = G0
  * 2º número após a entrada = G1
  * 3º número após a entrada = G2
  * Se nenhum desses iguala ao sugerido → LOSS ao registrar o 3º diferente
- Mensagens GREEN/LOSS incluem o número que saiu e o estágio correto (G0/G1/G2)
- Placar geral ao lado da mensagem, sem alterar a estrutura do DB

Principais pontos:
- ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN
- ENV opcionais:
    TARGET_CHANNEL   -> canal onde será publicado o tiro seco (ex.: -1002796105884)
    SOURCE_CHANNEL   -> canal-fonte que disparam os gatilhos (ex.: -1002810508717)
    DB_PATH          -> caminho do sqlite (default: /var/data/data.db)
- Webhook: POST /webhook/{WEBHOOK_TOKEN}
  Configure no Telegram com setWebhook apontando para essa URL.

- Banco:
  * timeline / ngram -> memória leve para n-grams (2..5) com decaimento
  * pending -> controle do sinal em aberto (um por vez):
      id, created_at, suggested, stage(0|1|2), open(0|1), seen(TEXT CSV com resultados observados)
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

# Observação de resultado (aceita muitos formatos simples; ignora quando for sequência):
# Regra: se NÃO for mensagem de "Sequência:" e houver exatamente UM dígito [1-4] no texto,
# interpretamos como o número que saiu na mesa.
RESULT_ONE_RX = re.compile(r"[1-4]")

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

def extract_result_number(text: str) -> Optional[int]:
    """Retorna o número 1..4 quando o texto aparenta ser o resultado da rodada.
       Heurística: ignorar mensagens que contenham 'Sequência:'; se o resto do texto
       tiver exatamente 1 dígito 1..4, tomamos como resultado.
    """
    t = re.sub(r"\s+", " ", text).strip()
    if SEQ_RX.search(t):
        return None
    nums = RESULT_ONE_RX.findall(t)
    # manter apenas 1..4
    nums = [int(x) for x in nums if x in ("1","2","3","4")]
    # deduplicar mantendo ordem
    seen = []
    for x in nums:
        if x not in seen:
            seen.append(x)
    if len(seen) == 1:
        return seen[0]
    return None

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

def _append_seen_and_stage(res_number:int):
    """Acrescenta o número observado em pending.seen (CSV) e atualiza pending.stage
       para refletir quantos resultados já passaram (0 = G0, 1 = G1, 2 = G2)."""
    con = _connect(); cur = con.cursor()
    row = cur.execute("SELECT id, seen FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        con.close(); return
    pid = row["id"]
    seen = (row["seen"] or "").strip()
    new_seen = (f"{seen},{res_number}" if seen else f"{res_number}")
    stage_idx = new_seen.count(",")  # 0 => G0, 1 => G1, 2 => G2
    cur.execute("UPDATE pending SET seen=?, stage=? WHERE id=?", (new_seen, stage_idx, pid))
    con.commit(); con.close()

def close_pending(outcome:str):
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE pending SET open=0, seen=? WHERE open=1", (outcome,))
    con.commit(); con.close()

def _stage_label(stage_val: Optional[int]) -> str:
    try:
        s = int(stage_val or 0)
    except Exception:
        s = 0
    return "G0" if s == 0 else ("G1" if s == 1 else "G2")

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

    # 0) Se for um "resultado" (um único número 1..4 fora de 'Sequência:'), processa conferência
    res = extract_result_number(text)
    if res is not None:
        pend = get_open_pending()
        if pend:
            suggested = int(pend["suggested"] or 0)
            # Anota o resultado e atualiza stage conforme quantos já saíram
            _append_seen_and_stage(res)
            # re-lê a pendência para saber o stage atualizado
            pend2 = get_open_pending()
            if pend2:
                stage_lbl = _stage_label(pend2["stage"])
                # GREEN se bateu
                if res == suggested:
                    close_pending("GREEN")
                    bump_score("GREEN")
                    await tg_send_text(
                        TARGET_CHANNEL,
                        f"🟢 <b>GREEN</b> — número: <b>{res}</b> (<b>{stage_lbl}</b>).\n"
                        f"📊 Geral: {score_text()}"
                    )
                else:
                    # Se já computamos 3 resultados (G2 é o 3º) sem bater → LOSS
                    # pend2['stage'] já representa 0 (G0), 1 (G1), 2 (G2)
                    if int(pend2["stage"] or 0) >= 2:
                        close_pending("LOSS")
                        bump_score("LOSS")
                        await tg_send_text(
                            TARGET_CHANNEL,
                            f"🔴 <b>LOSS</b> — número: <b>{res}</b> (<b>{stage_lbl}</b>).\n"
                            f"📊 Geral: {score_text()}"
                        )
        return {"ok": True, "noted": "resultado", "value": res}

    # 1) ENTRADA CONFIRMADA → abre pendência e publica o G0
    parsed = parse_entry_text(text)
    if parsed:
        # Se já existe pendência aberta, não abrimos outra.
        pend = get_open_pending()
        if pend:
            return {"ok": True, "ignored": "ja_existe_pendente"}

        # Alimenta memória de sequência (se vier algo), antes de decidir
        seq = parsed["seq"] or []
        if seq:
            append_seq(seq)

        after = parsed["after"]
        best, conf, samples = choose_single_number(after)

        # Abre pendência e publica
        open_pending(best)
        aft_txt = f" após {after}" if after else ""
        txt = (
            f"🎯 <b>Número seco (G0):</b> <b>{best}</b>\n"
            f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
            f"📊 <b>Conf:</b> {conf*100:.2f}% | <b>Amostra≈</b>{samples}"
        )
        await tg_send_text(TARGET_CHANNEL, txt)

        return {"ok": True, "posted": True, "best": best, "conf": conf, "samples": samples}

    # 2) Se vier "Sequência:" fora de entrada confirmada, apenas alimenta memória
    if SEQ_RX.search(text):
        parts = re.findall(r"[1-4]", text)
        seq = [int(x) for x in parts]
        if seq:
            append_seq(seq)
        return {"ok": True, "noted": "sequencia", "len": len(seq)}

    # Outros textos: ignorar
    return {"ok": True, "skipped": "texto_nao_relevante"}

# ===== Debug/help endpoints (opcionais) =====
@app.get("/health")
async def health():
    pend = bool(get_open_pending())
    return {"ok": True, "db": DB_PATH, "pending_open": pend, "time": ts_str()}