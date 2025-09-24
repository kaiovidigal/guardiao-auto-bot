#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webhook_app.py — v5.0.0
---------------------------------------------
FastAPI + Telegram webhook com fluxo de sinal robusto:

• Lê posts do canal-fonte (ou mensagens) e publica "número seco" (1..4) no canal-alvo.
• Fluxo ESTRITO: nunca abre novo sinal enquanto existir um aberto.
• Gales G1/G2 com fechamento GREEN/LOSS (ou por timeout).
• Aprendizado online limpo (feedback) + Hedge (4 especialistas) realmente atualizando pesos.
• Anti-tilt/cooldown configuráveis e gates de qualidade (confiança, gap, entropia).
• Deduplicação por update_id em TODOS os caminhos.
• Suporte a updates de canal (channel_post) e chat (message).
• Mensagens de "ignorado" controladas por DEBUG_MSG (sem poluir canal).

ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN
ENV opcionais:    TARGET_CHANNEL, SOURCE_CHANNEL, DB_PATH, DEBUG_MSG
Webhook:          POST /webhook/{WEBHOOK_TOKEN}
"""
import os, re, time, sqlite3, math
from contextlib import contextmanager
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, HTTPException

# ========= ENV =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1002796105884").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()  # se vazio, não filtra
DB_PATH        = (os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db")
DEBUG_MSG      = os.getenv("DEBUG_MSG", "0").strip() in ("1", "true", "True", "yes", "YES")
TELEGRAM_API   = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

if not TG_BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")
if not WEBHOOK_TOKEN:
    raise RuntimeError("Defina WEBHOOK_TOKEN no ambiente.")

# ========= App =========
app = FastAPI(title="guardiao-auto-bot (GEN webhook)", version="5.0.0")

# ========= Parâmetros =========
DECAY = 0.980
W4, W3, W2, W1 = 0.35, 0.35, 0.20, 0.10
OBS_TIMEOUT_SEC = 240

# ======== Gates (ajuste fino) ========
CONF_MIN    = 0.40      # subir/baixar agressividade
GAP_MIN     = 0.10
H_MAX       = 0.80      # entropia normalizada (base 4); >H_MAX = muito difuso
FREQ_WINDOW = 120

# ======== Cooldown / Anti-tilt ========
COOLDOWN_N       = 3    # após LOSS
ANTI_TILT_ON     = True
ANTI_TILT_MAXRED = 3    # não abre novo se loss_streak >= 3

# ======== Modo "sempre entrar" ========
ALWAYS_ENTER = False

# ======== Online Learning (feedback) ========
FEED_POS    = 1.1
FEED_NEG    = 1.2
FEED_DECAY  = 0.995     # esquecimento suave

# ======== Ensemble Hedge ========
HEDGE_ETA = 0.8
K_SHORT   = 60
K_LONG    = 300

# ========= Utils =========
def now_ts() -> int:
    return int(time.time())

def ts_str(ts=None) -> str:
    if ts is None: ts = now_ts()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _entropy_norm(post: Dict[int, float]) -> float:
    """Entropia normalizada (base 4). 0=concentrada, 1=uniforme."""
    eps = 1e-12
    return -sum((p+eps) * math.log(p+eps, 4) for p in post.values())

def _escape_md(s: str) -> str:
    # Escape básico para parse_mode=Markdown
    return re.sub(r'([_*`])', r'\\\1', s or "")

def _is_valid_nxt(n: Any) -> bool:
    try:
        n = int(n)
    except Exception:
        return False
    return n in (1, 2, 3, 4)

# ========= DB helpers =========
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

@contextmanager
def _tx():
    con = _connect()
    try:
        con.execute("BEGIN IMMEDIATE")
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

def migrate_db():
    with _tx() as con:
        cur = con.cursor()
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
            "after" INTEGER,
            ctx1 INTEGER, ctx2 INTEGER, ctx3 INTEGER, ctx4 INTEGER
        )""")
        # score
        cur.execute("""CREATE TABLE IF NOT EXISTS score (
            id INTEGER PRIMARY KEY CHECK (id=1),
            green INTEGER DEFAULT 0,
            loss  INTEGER DEFAULT 0
        )""")
        # init row
        row = cur.execute("SELECT 1 FROM score WHERE id=1").fetchone()
        if not row:
            cur.execute("INSERT INTO score (id, green, loss) VALUES (1,0,0)")
        # feedback
        cur.execute("""CREATE TABLE IF NOT EXISTS feedback (
            n INTEGER NOT NULL, ctx TEXT NOT NULL, nxt INTEGER NOT NULL, w REAL NOT NULL,
            PRIMARY KEY (n, ctx, nxt)
        )""")
        # processed (dedupe)
        cur.execute("""CREATE TABLE IF NOT EXISTS processed (
            update_id TEXT PRIMARY KEY,
            seen_at   INTEGER NOT NULL
        )""")
        # state (cooldown + loss_streak)
        cur.execute("""CREATE TABLE IF NOT EXISTS state (
            id INTEGER PRIMARY KEY CHECK (id=1),
            cooldown_left INTEGER DEFAULT 0,
            loss_streak   INTEGER DEFAULT 0
        )""")
        row = cur.execute("SELECT 1 FROM state WHERE id=1").fetchone()
        if not row:
            cur.execute("INSERT INTO state (id, cooldown_left, loss_streak) VALUES (1,0,0)")
        # expert weights (Hedge)
        cur.execute("""CREATE TABLE IF NOT EXISTS expert_w (
            id INTEGER PRIMARY KEY CHECK (id=1),
            w1 REAL NOT NULL,
            w2 REAL NOT NULL,
            w3 REAL NOT NULL,
            w4 REAL NOT NULL
        )""")
        row = cur.execute("SELECT 1 FROM expert_w WHERE id=1").fetchone()
        if not row:
            cur.execute("INSERT INTO expert_w (id, w1, w2, w3, w4) VALUES (1,1.0,1.0,1.0,1.0)")

migrate_db()

def _exec_write(sql: str, params: tuple=()):
    for attempt in range(6):
        try:
            with _tx() as con:
                con.execute(sql, params)
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                time.sleep(0.25*(attempt+1)); continue
            raise

# ========= Dedupe =========
def _is_processed(update_id: str) -> bool:
    if not update_id: return False
    con = _connect()
    row = con.execute("SELECT 1 FROM processed WHERE update_id=?", (str(update_id),)).fetchone()
    con.close()
    return bool(row)

def _mark_processed(update_id: str):
    if not update_id: return
    _exec_write("INSERT OR IGNORE INTO processed (update_id, seen_at) VALUES (?,?)",
                (str(update_id), now_ts()))

# ========= Score helpers =========
def bump_score(outcome: str) -> Tuple[int, int]:
    with _tx() as con:
        row = con.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
        g, l = (row["green"], row["loss"]) if row else (0, 0)
        if outcome.upper() == "GREEN":
            g += 1
        elif outcome.upper() == "LOSS":
            l += 1
        con.execute("INSERT OR REPLACE INTO score (id, green, loss) VALUES (1,?,?)", (g, l))
        return g, l

def score_text() -> str:
    con = _connect()
    row = con.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
    con.close()
    g, l = (int(row["green"]), int(row["loss"])) if row else (0, 0)
    total = g + l
    acc = (g/total*100.0) if total > 0 else 0.0
    return f"{g} GREEN × {l} LOSS — {acc:.1f}%"

# ========= State helpers =========
def _get_cooldown() -> int:
    con = _connect()
    row = con.execute("SELECT cooldown_left FROM state WHERE id=1").fetchone()
    con.close()
    return int((row["cooldown_left"] if row else 0) or 0)

def _set_cooldown(v:int):
    _exec_write("UPDATE state SET cooldown_left=? WHERE id=1", (int(v),))

def _dec_cooldown():
    with _tx() as con:
        row = con.execute("SELECT cooldown_left FROM state WHERE id=1").fetchone()
        cur = int((row["cooldown_left"] if row else 0) or 0)
        cur = max(0, cur-1)
        con.execute("UPDATE state SET cooldown_left=? WHERE id=1", (cur,))

def _get_loss_streak() -> int:
    con = _connect()
    row = con.execute("SELECT loss_streak FROM state WHERE id=1").fetchone()
    con.close()
    return int((row["loss_streak"] if row else 0) or 0)

def _set_loss_streak(v:int):
    _exec_write("UPDATE state SET loss_streak=? WHERE id=1", (int(v),))

def _bump_loss_streak(reset: bool):
    if reset:
        _set_loss_streak(0)
    else:
        _set_loss_streak(_get_loss_streak() + 1)

# ========= N-gram & Feedback =========
def timeline_size() -> int:
    con = _connect()
    row = con.execute("SELECT COUNT(*) AS c FROM timeline").fetchone()
    con.close()
    return int(row["c"] or 0)

def get_tail(limit:int=400) -> List[int]:
    con = _connect()
    rows = con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def append_seq(seq: List[int]):
    if not seq: return
    with _tx() as con:
        for n in seq:
            con.execute("INSERT INTO timeline (created_at, number) VALUES (?,?)",
                        (now_ts(), int(n)))
    _update_ngrams()

def _update_ngrams(decay: float=DECAY, max_n:int=5, window:int=400):
    tail = get_tail(window)
    if len(tail) < 2: return
    with _tx() as con:
        for t in range(1, len(tail)):
            nxt = int(tail[t])
            dist = (len(tail)-1) - t
            w = decay ** dist
            for n in range(2, max_n+1):
                if t-(n-1) < 0: break
                ctx = tail[t-(n-1):t]
                ctx_key = ",".join(str(x) for x in ctx)
                con.execute("""
                  INSERT INTO ngram (n, ctx, nxt, w)
                  VALUES (?,?,?,?)
                  ON CONFLICT(n, ctx, nxt) DO UPDATE SET w = w + excluded.w
                """, (n, ctx_key, nxt, float(w)))

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

def _ctx_to_key(ctx: List[int]) -> str:
    return ",".join(str(x) for x in ctx) if ctx else ""

def _feedback_upsert(n:int, ctx_key:str, nxt:int, factor: float) -> None:
    with _tx() as con:
        con.execute("""
          INSERT INTO feedback (n, ctx, nxt, w)
          VALUES (?,?,?,?)
          ON CONFLICT(n, ctx, nxt) DO UPDATE SET w = w * excluded.w
        """, (n, ctx_key, nxt, float(factor)))

def _feedback_apply(ctx: List[int], nxt: int) -> None:
    if not _is_valid_nxt(nxt): return
    if len(ctx) >= 4:
        _feedback_upsert(4, _ctx_to_key(ctx[-4:]), nxt, FEED_POS)
    if len(ctx) >= 3:
        _feedback_upsert(3, _ctx_to_key(ctx[-3:]), nxt, FEED_POS)
    if len(ctx) >= 2:
        _feedback_upsert(2, _ctx_to_key(ctx[-2:]), nxt, FEED_POS)
    _feedback_upsert(1, _ctx_to_key([]), nxt, FEED_POS)

def _feedback_penalize(ctx: List[int], nxt: int) -> None:
    if not _is_valid_nxt(nxt): return
    if len(ctx) >= 4:
        _feedback_upsert(4, _ctx_to_key(ctx[-4:]), nxt, 1.0/FEED_NEG)
    if len(ctx) >= 3:
        _feedback_upsert(3, _ctx_to_key(ctx[-3:]), nxt, 1.0/FEED_NEG)
    if len(ctx) >= 2:
        _feedback_upsert(2, _ctx_to_key(ctx[-2:]), nxt, 1.0/FEED_NEG)
    _feedback_upsert(1, _ctx_to_key([]), nxt, 1.0/FEED_NEG)

def _decay_feedback(decay: float=FEED_DECAY):
    _exec_write("UPDATE feedback SET w = w * ? WHERE w > 0.01", (decay,))

# ========= Expert Weights (Hedge) =========
def _get_expert_w() -> Tuple[float, float, float, float]:
    con = _connect()
    row = con.execute("SELECT w1, w2, w3, w4 FROM expert_w WHERE id=1").fetchone()
    con.close()
    return (row["w1"], row["w2"], row["w3"], row["w4"]) if row else (1.0, 1.0, 1.0, 1.0)

def _update_expert_w(outcome: str, experts: Dict[str, int]):
    if not experts: return
    with _tx() as con:
        row = con.execute("SELECT w1, w2, w3, w4 FROM expert_w WHERE id=1").fetchone()
        w1, w2, w3, w4 = (row["w1"], row["w2"], row["w3"], row["w4"]) if row else (1.0,1.0,1.0,1.0)

        # Reforça o especialista que coincidiu com o sugerido (e_winner)
        def upd(w, e):
            return w * (1 + HEDGE_ETA) if e == experts.get("e_winner") else w * (1 - HEDGE_ETA)

        w1 = upd(w1, experts.get("e1"))
        w2 = upd(w2, experts.get("e2"))
        w3 = upd(w3, experts.get("e3"))
        w4 = upd(w4, experts.get("e4"))

        # Normaliza soma constante (4.0)
        total = max(1e-9, (w1 + w2 + w3 + w4))
        k = 4.0 / total
        w1, w2, w3, w4 = w1*k, w2*k, w3*k, w4*k

        con.execute("UPDATE expert_w SET w1=?, w2=?, w3=?, w4=? WHERE id=1",
                    (w1, w2, w3, w4))

# ========= Business Logic =========
def _get_last_open_signal() -> Optional[Dict]:
    con = _connect()
    row = con.execute("SELECT * FROM pending WHERE open=1 ORDER BY created_at DESC LIMIT 1").fetchone()
    con.close()
    return dict(row) if row else None

def _reconstruct_ctx_experts(p: Dict) -> Tuple[List[int], Dict[str, int]]:
    ctx = [int(x) for x in (p.get("ctx1"), p.get("ctx2"), p.get("ctx3"), p.get("ctx4")) if _is_valid_nxt(x)]
    experts = {
        "e1": p.get("ctx1"),
        "e2": p.get("ctx2"),
        "e3": p.get("ctx3"),
        "e4": p.get("ctx4"),
        "e_winner": p.get("suggested"),
    }
    return ctx, experts

def _close_signal(
    signal_id: int,
    outcome: str,
    actual_number: int,
    stage: int,
    ctx: List[int],
    experts: Dict[str, int],
    suggested: int,
):
    _exec_write("UPDATE pending SET open=0, after=? WHERE id=?", (int(actual_number), signal_id))

    # perda/ganho e streaks
    is_green = outcome.upper() == "GREEN"
    _bump_loss_streak(reset=is_green)
    bump_score(outcome)

    # feedback limpo
    if is_green:
        _feedback_apply(ctx, actual_number)           # reforça o real vencedor
    else:
        _feedback_penalize(ctx, suggested)            # penaliza o sugerido
        if _is_valid_nxt(actual_number):
            _feedback_penalize(ctx, actual_number)    # penaliza também o real se fizer sentido

    # hedge
    _update_expert_w(outcome, experts)

    # cooldown
    if outcome.upper() == "LOSS":
        _set_cooldown(COOLDOWN_N)

async def _send_telegram(text: str, channel_id: str = TARGET_CHANNEL):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={"chat_id": channel_id, "text": _escape_md(text), "parse_mode": "Markdown"}
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"Erro ao enviar mensagem para o Telegram: {e}")
        return None

def _get_prediction(tail: List[int]) -> Tuple[int, Dict[int, float], Dict]:
    # Especialista 1: N-grama + pesos base
    n_gram_probs = {1:0.0, 2:0.0, 3:0.0, 4:0.0}
    ctx = tail[-4:]
    # pesos por ordem (W4..W1) somam 1.0 — aqui usamos como "mistura por ordem"
    for i in range(2, 6):
        if len(ctx) >= i-1:
            n_ctx = ctx[-(i-1):]
            n_gram_probs[1] += _prob_from_ngrams(n_ctx, 1) * W4
            n_gram_probs[2] += _prob_from_ngrams(n_ctx, 2) * W3
            n_gram_probs[3] += _prob_from_ngrams(n_ctx, 3) * W2
            n_gram_probs[4] += _prob_from_ngrams(n_ctx, 4) * W1

    # Especialista 2: Frequência Curta
    k_short_tail = tail[-K_SHORT:] if len(tail) >= 1 else [1]
    short_freq_probs = {n: k_short_tail.count(n) / len(k_short_tail) for n in range(1, 5)}

    # Especialista 3: Frequência Longa
    k_long_tail = tail[-K_LONG:] if len(tail) >= 1 else [1]
    long_freq_probs = {n: k_long_tail.count(n) / len(k_long_tail) for n in range(1, 5)}

    # Especialista 4: Tendência simples (cíclica)
    trend_probs = {1:0.25, 2:0.25, 3:0.25, 4:0.25}
    if len(tail) >= 3 and tail[-3:] == [1,2,3]: trend_probs[4] += 0.5
    if len(tail) >= 3 and tail[-3:] == [2,3,4]: trend_probs[1] += 0.5
    if len(tail) >= 3 and tail[-3:] == [3,4,1]: trend_probs[2] += 0.5
    if len(tail) >= 3 and tail[-3:] == [4,1,2]: trend_probs[3] += 0.5

    # Hedge
    w1, w2, w3, w4 = _get_expert_w()

    final_probs = {
        1: n_gram_probs[1]*w1 + short_freq_probs[1]*w2 + long_freq_probs[1]*w3 + trend_probs[1]*w4,
        2: n_gram_probs[2]*w1 + short_freq_probs[2]*w2 + long_freq_probs[2]*w3 + trend_probs[2]*w4,
        3: n_gram_probs[3]*w1 + short_freq_probs[3]*w2 + long_freq_probs[3]*w3 + trend_probs[3]*w4,
        4: n_gram_probs[4]*w1 + short_freq_probs[4]*w2 + long_freq_probs[4]*w3 + trend_probs[4]*w4,
    }

    # Normaliza
    s = sum(final_probs.values())
    if s > 0:
        for k in final_probs:
            final_probs[k] /= s
    else:
        final_probs = {1:0.25, 2:0.25, 3:0.25, 4:0.25}

    suggested = max(final_probs, key=lambda x: final_probs[x])

    experts = {
        "e1": max(n_gram_probs, key=n_gram_probs.get),
        "e2": max(short_freq_probs, key=short_freq_probs.get),
        "e3": max(long_freq_probs, key=long_freq_probs.get),
        "e4": max(trend_probs, key=trend_probs.get),
        "e_winner": suggested
    }
    return suggested, final_probs, experts

async def _process_incoming(msg: Dict, update_id: str):
    # dedupe na entrada
    if _is_processed(update_id):
        print("update_id já processado, ignorando.")
        return

    # valida estrutura
    if "chat" not in msg or "id" not in msg["chat"]:
        _mark_processed(update_id); return

    chat_id = str(msg["chat"]["id"])
    text = (msg.get("text") or "").strip()

    # filtra canal-fonte (se configurado)
    if SOURCE_CHANNEL and chat_id != SOURCE_CHANNEL:
        print(f"Ignorando mensagem de canal não-fonte: {chat_id}")
        _mark_processed(update_id); return

    last_signal = _get_last_open_signal()

    # tenta extrair número 1..4
    m = re.search(r"\b(1|2|3|4)\b", text)
    if not m:
        # Se não há número e existe sinal ABERTO, avalia timeout
        if last_signal and last_signal["open"] and (now_ts() - int(last_signal["opened_at"])) > OBS_TIMEOUT_SEC:
            await _send_telegram(f"Sinal `X` fechado por tempo. [ID:{last_signal['id']}]")
            # fecha como LOSS por tempo (penaliza apenas o sugerido)
            ctx, experts = _reconstruct_ctx_experts(last_signal)
            s = int(last_signal["suggested"])
            _close_signal(last_signal["id"], "LOSS", 0, int(last_signal["stage"]), ctx, experts, s)
        _mark_processed(update_id)
        return

    # há número na mensagem
    number_played = int(m.group(1))

    # ========= Se há sinal aberto, processa fechamento / avanço de stage =========
    if last_signal and last_signal["open"]:
        suggested = int(last_signal["suggested"])
        stage = int(last_signal["stage"])
        seen = [int(s) for s in (last_signal["seen"].split(",") if last_signal["seen"] else [])]
        if number_played in seen:
            print(f"Número {number_played} já visto no sinal {last_signal['id']}, ignorando.")
            _mark_processed(update_id); return

        seen = sorted(set(seen + [number_played]))
        _exec_write("UPDATE pending SET seen=? WHERE id=?", (",".join(str(s) for s in seen), last_signal["id"]))

        ctx, experts = _reconstruct_ctx_experts(last_signal)

        if number_played == suggested:
            outcome = "GREEN"
            msg_text = (
                f"Sinal `VERDE` no {stage+1}º stage. 🎉\n\n"
                f"Número: {suggested}\nObservado: {number_played}\n\n[ID:{last_signal['id']}]"
            )
            await _send_telegram(msg_text)
            _close_signal(last_signal["id"], outcome, number_played, stage, ctx, experts, suggested)

        elif stage == 0:
            outcome = "G1"
            _exec_write("UPDATE pending SET stage=1 WHERE id=?", (last_signal["id"],))
            await _send_telegram(f"Sinal `G1` para {suggested}...\n\nÚltimo: {number_played}\n\n[ID:{last_signal['id']}]")

        elif stage == 1:
            outcome = "G2"
            _exec_write("UPDATE pending SET stage=2 WHERE id=?", (last_signal["id"],))
            await _send_telegram(f"Sinal `G2` para {suggested}...\n\nÚltimo: {number_played}\n\n[ID:{last_signal['id']}]")

        else:  # stage >= 2
            outcome = "LOSS"
            await _send_telegram(
                f"Sinal `LOSS`. 💔\n\nNúmero: {suggested}\nÚltimo: {number_played}\n\n[ID:{last_signal['id']}]"
            )
            _close_signal(last_signal["id"], outcome, number_played, stage, ctx, experts, suggested)

        # registra observação na timeline e encerra
        append_seq([number_played])
        _mark_processed(update_id)
        return

    # ========= Não há sinal aberto: considera abrir um novo =========
    append_seq([number_played])   # registramos o observado atual
    tail = get_tail()

    if len(tail) < 20:
        if DEBUG_MSG:
            await _send_telegram("Sequência muito curta, aguardando mais dados...")
        _mark_processed(update_id); return

    # cooldown
    if _get_cooldown() > 0:
        cd = _get_cooldown()
        _dec_cooldown()
        if DEBUG_MSG:
            await _send_telegram(f"Em cooldown... faltam {cd} sinais.")
        _mark_processed(update_id); return

    # anti-tilt
    if ANTI_TILT_ON and _get_loss_streak() >= ANTI_TILT_MAXRED and not ALWAYS_ENTER:
        if DEBUG_MSG:
            await _send_telegram(f"Sinal bloqueado por anti-tilt (streak RED={_get_loss_streak()}).")
        _mark_processed(update_id); return

    # predição
    suggested, probs, experts = _get_prediction(tail)

    # gates
    conf = float(probs[suggested])
    sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    gap = (sorted_probs[0][1] - sorted_probs[1][1]) if len(sorted_probs) > 1 else 0.0
    ent = _entropy_norm(probs)

    is_low_conf = conf < CONF_MIN
    is_low_gap = gap < GAP_MIN
    is_high_entropy = ent > H_MAX

    if not ALWAYS_ENTER and (is_low_conf or is_low_gap or is_high_entropy):
        if DEBUG_MSG:
            reason = "LOW_CONF" if is_low_conf else "LOW_GAP" if is_low_gap else "HIGH_ENTROPY"
            info_msg = (
                f"Sinal ignorado.\n"
                f"Motivo: {reason}\n"
                f"Conf: {conf*100:.2f}% | Gap: {gap*100:.1f}pp | H: {ent:.3f}\n"
                f"Cooldown: {_get_cooldown()} | Streak RED: {_get_loss_streak()}"
            )
            await _send_telegram(info_msg)
        _mark_processed(update_id)
        return

    # abrir sinal
    with _tx() as con:
        cur = con.execute("""
            INSERT INTO pending (created_at, suggested, opened_at, stage, open, seen, ctx1, ctx2, ctx3, ctx4)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (now_ts(), suggested, now_ts(), 0, 1, str(number_played),  # já marca o primeiro "visto"
              experts["e1"], experts["e2"], experts["e3"], experts["e4"]))
        signal_id = cur.lastrowid

    msg_text = (
        f"Sinal `NOVO` - `{suggested}`\n\n"
        f"Conf: {conf*100:.2f}% | Gap: {gap*100:.1f}pp | H: {ent:.3f}\n"
        f"Cooldown: {_get_cooldown()} | Streak RED: {_get_loss_streak()}\n"
        f"Score: {score_text()}\n"
        f"ID: {signal_id}"
    )
    await _send_telegram(msg_text)

    _mark_processed(update_id)

# ========= Webhook =========
@app.post("/webhook/{webhook_token}")
async def handle_webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token de webhook inválido.")

    body = await request.json()
    update_id = body.get("update_id", "")

    # Telegram pode mandar "message" (DM/grupo) ou "channel_post" (canal).
    message = body.get("message") or body.get("channel_post") or {}

    if not update_id:
        return {"ok": True, "message": "Sem update_id."}

    if message:
        await _process_incoming(message, update_id)
    else:
        # mesmo sem mensagem, marca e segue (evita retry loop)
        _mark_processed(update_id)

    # Decaimento suave do feedback (lazy)
    _decay_feedback()

    return {"ok": True}