#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webhook_app.py — v4.3.2 (Ajuste de Confiança para TESTE mais agressivo)
----------------------------------------------------------------------
FastAPI + Telegram webhook que:
- Lê mensagens do canal-fonte e publica um "número seco" (1..4) no canal-alvo.
- Fluxo ESTRITO: nunca abre novo sinal enquanto o anterior não fechou de verdade.
- Acompanha gales (G1/G2) e fecha robusto (GREEN/LOSS) por observados.
- Aprende online com feedback: penaliza erro e REFORÇA o número vencedor real.
- Usa ENSEMBLE (Hedge) de 4 especialistas (n-grama+feedback, freq curta, freq longa, tendencia).
- Consciência de sequência (loss_streak) e jogada anti-tilt.
- Dedupe por update_id, abertura transacional, timeout (fecha com "X" só por tempo quando já há 2 observados).

ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN
ENV opcionais:    TARGET_CHANNEL, SOURCE_CHANNEL, DB_PATH
Webhook:          POST /webhook/{WEBHOOK_TOKEN}
"""
import os, re, time, sqlite3, math
from contextlib import contextmanager
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
app = FastAPI(title="guardiao-auto-bot (GEN webhook)", version="4.3.2")

# ========= Parâmetros =========
DECAY = 0.980
W4, W3, W2, W1 = 0.35, 0.35, 0.20, 0.10
OBS_TIMEOUT_SEC = 240

# ======== Gates (LIGADOS para diminuir o LOSS) ========
CONF_MIN    = 0.35 # <<<<< AJUSTADO PARA TESTE >>>>>
GAP_MIN     = 0.08
H_MAX       = 0.80
FREQ_WINDOW = 120

# ======== Cooldown após RED ========
COOLDOWN_N     = 3
CD_CONF_BOOST  = 0.00 # <<<<< TEMPORARIAMENTE REMOVIDO PARA TESTE >>>>>
CD_GAP_BOOST   = 0.03

# ======== Modo "sempre entrar" (DESLIGADO para filtrar sinais) ========
ALWAYS_ENTER = False

# ======== Online Learning (feedback) ========
FEED_BETA   = 0.55
FEED_POS    = 1.1
FEED_NEG    = 1.2
FEED_DECAY  = 0.995
WF4, WF3, WF2, WF1 = W4, W3, W2, W1

# ======== Empate técnico (legado) ========
GAP_SOFT = 0.010

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
    H = -sum((p+eps) * math.log(p+eps, 4) for p in post.values())
    return H

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
        "after" INTEGER,
        ctx1 TEXT, ctx2 TEXT, ctx3 TEXT, ctx4 TEXT
    )""")
    # score
    cur.execute("""CREATE TABLE IF NOT EXISTS score (
        id INTEGER PRIMARY KEY CHECK (id=1),
        green INTEGER DEFAULT 0,
        loss  INTEGER DEFAULT 0
    )""")
    row = con.execute("SELECT 1 FROM score WHERE id=1").fetchone()
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
    row = con.execute("SELECT 1 FROM state WHERE id=1").fetchone()
    if not row:
        cur.execute("INSERT INTO state (id, cooldown_left, loss_streak) VALUES (1,0,0)")
    try:
        cur.execute("ALTER TABLE state ADD COLUMN loss_streak INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    # expert weights (Hedge)
    cur.execute("""CREATE TABLE IF NOT EXISTS expert_w (
        id INTEGER PRIMARY KEY CHECK (id=1),
        w1 REAL NOT NULL,
        w2 REAL NOT NULL,
        w3 REAL NOT NULL,
        w4 REAL NOT NULL
    )""")
    row = con.execute("SELECT 1 FROM expert_w WHERE id=1").fetchone()
    if not row:
        cur.execute("INSERT INTO expert_w (id, w1, w2, w3, w4) VALUES (1, 1.0, 1.0, 1.0, 1.0)")
    else:
        if not _column_exists(con, 'expert_w', 'w4'):
            cur.execute("ALTER TABLE expert_w ADD COLUMN w4 REAL DEFAULT 1.0")
    con.commit(); con.close()
migrate_db()

def _exec_write(sql: str, params: tuple=()):
    for attempt in range(6):
        try:
            with _tx() as con:
                con.execute(sql, params)
            return
        except sqlite3.OperationalError as e:
            emsg = str(e).lower()
            if "locked" in emsg or "busy" in emsg:
                time.sleep(0.25*(attempt+1))
                continue
            raise

# ========= Dedupe =========
def _is_processed(update_id: str) -> bool:
    if not update_id: return False
    con = _connect()
    row = con.execute("SELECT 1 FROM processed WHERE update_id=?", (update_id,)).fetchone()
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
    if not row:
        return "0 GREEN × 0 LOSS — 0.0%"
    g, l = int(row["green"]), int(row["loss"])
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
        cur = _get_loss_streak()
        _set_loss_streak(cur + 1)

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

def _feedback_upsert(n:int, ctx_key:str, nxt:int, delta) -> None:
    with _tx() as con:
        con.execute("""
          INSERT INTO feedback (n, ctx, nxt, w)
          VALUES (?,?,?,?)
          ON CONFLICT(n, ctx, nxt) DO UPDATE SET w = w * excluded.w
        """, (n, ctx_key, nxt, float(delta)))

def _feedback_apply(ctx: List[int], nxt: int) -> None:
    if len(ctx) >= 4:
        _feedback_upsert(4, _ctx_to_key(ctx[-4:]), nxt, FEED_POS)
    if len(ctx) >= 3:
        _feedback_upsert(3, _ctx_to_key(ctx[-3:]), nxt, FEED_POS)
    if len(ctx) >= 2:
        _feedback_upsert(2, _ctx_to_key(ctx[-2:]), nxt, FEED_POS)
    _feedback_upsert(1, _ctx_to_key([]), nxt, FEED_POS)

def _feedback_penalize(ctx: List[int], nxt: int) -> None:
    if len(ctx) >= 4:
        _feedback_upsert(4, _ctx_to_key(ctx[-4:]), nxt, 1.0/FEED_NEG)
    if len(ctx) >= 3:
        _feedback_upsert(3, _ctx_to_key(ctx[-3:]), nxt, 1.0/FEED_NEG)
    if len(ctx) >= 2:
        _feedback_upsert(2, _ctx_to_key(ctx[-2:]), nxt, 1.0/FEED_NEG)
    _feedback_upsert(1, _ctx_to_key([]), nxt, 1.0/FEED_NEG)

def _get_feedback_w(n:int, ctx_key:str, nxt:int) -> float:
    con = _connect()
    row = con.execute("SELECT w FROM feedback WHERE n=? AND ctx=? AND nxt=?",
                      (n, ctx_key, nxt)).fetchone()
    con.close()
    return (row["w"] or 1.0) if row else 1.0

def _decay_feedback(decay: float=FEED_DECAY):
    _exec_write("UPDATE feedback SET w = w * ? WHERE w > 0.01", (decay,))

# ========= Expert Weights (Hedge) =========
def _get_expert_w() -> Tuple[float, float, float, float]:
    con = _connect()
    row = con.execute("SELECT w1, w2, w3, w4 FROM expert_w WHERE id=1").fetchone()
    con.close()
    if not row: return (1.0, 1.0, 1.0, 1.0)
    return (row["w1"], row["w2"], row["w3"], row["w4"])

def _update_expert_w(outcome: str, experts: Dict[str, int]):
    if not experts: return
    with _tx() as con:
        w1, w2, w3, w4 = _get_expert_w()
        if experts.get("e1") == experts.get("e_winner"):
            w1 *= (1 + HEDGE_ETA)
        else:
            w1 *= (1 - HEDGE_ETA)
        if experts.get("e2") == experts.get("e_winner"):
            w2 *= (1 + HEDGE_ETA)
        else:
            w2 *= (1 - HEDGE_ETA)
        if experts.get("e3") == experts.get("e_winner"):
            w3 *= (1 + HEDGE_ETA)
        else:
            w3 *= (1 - HEDGE_ETA)
        if experts.get("e4") == experts.get("e_winner"):
            w4 *= (1 + HEDGE_ETA)
        else:
            w4 *= (1 - HEDGE_ETA)
        
        # Normalizar para que a soma seja constante (ex: 4.0)
        total_w = w1 + w2 + w3 + w4
        factor = 4.0 / total_w
        w1, w2, w3, w4 = w1 * factor, w2 * factor, w3 * factor, w4 * factor

        con.execute("UPDATE expert_w SET w1=?, w2=?, w3=?, w4=? WHERE id=1",
                    (w1, w2, w3, w4))

# ========= Business Logic =========
def _get_last_open_signal() -> Optional[Dict]:
    con = _connect()
    row = con.execute("SELECT * FROM pending WHERE open=1 ORDER BY created_at DESC LIMIT 1").fetchone()
    con.close()
    return dict(row) if row else None

def _close_signal(
    signal_id: int, outcome: str, actual_number: int, stage: int, ctx: List[int],
    experts: Dict[str, int]
):
    _exec_write("UPDATE pending SET open=0, after=? WHERE id=?", (int(actual_number), signal_id))
    _bump_loss_streak(outcome.upper() == "GREEN")
    bump_score(outcome)
    
    suggested = _get_last_open_signal()["suggested"] if _get_last_open_signal() else None
    
    # Feedback
    if outcome.upper() == "GREEN":
        _feedback_apply(ctx, actual_number)
    else:
        _feedback_penalize(ctx, suggested)
        _feedback_penalize(ctx, actual_number)
        
    # Expert Hedge
    _update_expert_w(outcome, experts)
    
    if outcome.upper() == "LOSS":
        _set_cooldown(COOLDOWN_N)

async def _send_telegram(text: str, channel_id: str = TARGET_CHANNEL):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={"chat_id": channel_id, "text": text, "parse_mode": "Markdown"}
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"Erro ao enviar mensagem para o Telegram: {e}")
        return None

def _get_prediction(tail: List[int]) -> Tuple[int, Dict[int, float], Dict]:
    # Experto 1: N-grama + Feedback
    n_gram_probs = {1:0.0, 2:0.0, 3:0.0, 4:0.0}
    ctx = tail[-4:]
    for i in range(2, 6):
        if len(ctx) >= i-1:
            n_ctx = ctx[-(i-1):]
            n_gram_probs[1] += _prob_from_ngrams(n_ctx, 1) * W4
            n_gram_probs[2] += _prob_from_ngrams(n_ctx, 2) * W3
            n_gram_probs[3] += _prob_from_ngrams(n_ctx, 3) * W2
            n_gram_probs[4] += _prob_from_ngrams(n_ctx, 4) * W1
    
    # Experto 2: Frequência Curta
    k_short_tail = tail[-K_SHORT:]
    short_freq_probs = {n: k_short_tail.count(n) / len(k_short_tail) for n in range(1, 5)}

    # Experto 3: Frequência Longa
    k_long_tail = tail[-K_LONG:]
    long_freq_probs = {n: k_long_tail.count(n) / len(k_long_tail) for n in range(1, 5)}

    # Experto 4: Tendência (ex: 1->2->3->4)
    trend_probs = {1:0.25, 2:0.25, 3:0.25, 4:0.25}
    if len(tail) >= 3 and tail[-3:] == [1,2,3]: trend_probs[4] += 0.5
    if len(tail) >= 3 and tail[-3:] == [2,3,4]: trend_probs[1] += 0.5
    if len(tail) >= 3 and tail[-3:] == [3,4,1]: trend_probs[2] += 0.5
    if len(tail) >= 3 and tail[-3:] == [4,1,2]: trend_probs[3] += 0.5

    # Hedge
    w1, w2, w3, w4 = _get_expert_w()
    
    final_probs = {1:0.0, 2:0.0, 3:0.0, 4:0.0}
    
    final_probs[1] = (n_gram_probs[1] * w1 + short_freq_probs[1] * w2 + long_freq_probs[1] * w3 + trend_probs[1] * w4)
    final_probs[2] = (n_gram_probs[2] * w1 + short_freq_probs[2] * w2 + long_freq_probs[2] * w3 + trend_probs[2] * w4)
    final_probs[3] = (n_gram_probs[3] * w1 + short_freq_probs[3] * w2 + long_freq_probs[3] * w3 + trend_probs[3] * w4)
    final_probs[4] = (n_gram_probs[4] * w1 + short_freq_probs[4] * w2 + long_freq_probs[4] * w3 + trend_probs[4] * w4)
    
    # Normalizar
    total_prob = sum(final_probs.values())
    if total_prob > 0:
        for n in final_probs:
            final_probs[n] /= total_prob
            
    sorted_probs = sorted(final_probs.items(), key=lambda item: item[1], reverse=True)
    suggested = sorted_probs[0][0]
    
    experts = {
        "e1": max(n_gram_probs, key=n_gram_probs.get),
        "e2": max(short_freq_probs, key=short_freq_probs.get),
        "e3": max(long_freq_probs, key=long_freq_probs.get),
        "e4": max(trend_probs, key=trend_probs.get),
        "e_winner": suggested
    }
    
    return suggested, final_probs, experts

async def _process_incoming(msg: Dict, update_id: str):
    if _is_processed(update_id):
        print("update_id já processado, ignorando.")
        return
        
    last_signal = _get_last_open_signal()
    
    if "text" not in msg or "chat" not in msg or "id" not in msg["chat"]:
        return
    
    chat_id = msg["chat"]["id"]
    text = msg["text"].strip()
    
    if SOURCE_CHANNEL and str(chat_id) != SOURCE_CHANNEL:
        print(f"Ignorando mensagem de canal não-fonte: {chat_id}")
        return
        
    number_match = re.search(r"\b(1|2|3|4)\b", text)
    if not number_match:
        # Tenta identificar "entrada fechada" por timeout
        if last_signal and last_signal["open"] and (now_ts() - last_signal["opened_at"]) > OBS_TIMEOUT_SEC:
            await _send_telegram(f"Sinal `X` fechado por tempo. [ID:{last_signal['id']}]")
            _close_signal(last_signal["id"], "LOSS", 0, 0, [], {})
        return
    
    number_played = int(number_match.group(1))
    
    # Fechar sinal pendente
    if last_signal and last_signal["open"]:
        suggested = int(last_signal["suggested"])
        stage = int(last_signal["stage"])
        seen = [int(s) for s in last_signal["seen"].split(",")] if last_signal["seen"] else []
        
        # Already seen?
        if number_played in seen:
            print(f"Número {number_played} já visto no sinal {last_signal['id']}, ignorando.")
            _mark_processed(update_id)
            return
            
        seen.append(number_played)
        _exec_write("UPDATE pending SET seen=? WHERE id=?",
                    (",".join(str(s) for s in seen), last_signal["id"]))

        if number_played == suggested:
            outcome = "GREEN"
            msg_text = f"Sinal `VERDE` no {stage+1}º stage. 🎉\n\nNúmero: {suggested}\nObservado: {number_played}\n\n[ID:{last_signal['id']}]"
            _set_cooldown(0) # Green reseta o cooldown
            
        elif stage == 0:
            outcome = "G1"
            msg_text = f"Sinal `G1` para {suggested}...\n\nÚltimo: {number_played}\n\n[ID:{last_signal['id']}]"
            _exec_write("UPDATE pending SET stage=1 WHERE id=?", (last_signal["id"],))
            
        elif stage == 1:
            outcome = "G2"
            msg_text = f"Sinal `G2` para {suggested}...\n\nÚltimo: {number_played}\n\n[ID:{last_signal['id']}]"
            _exec_write("UPDATE pending SET stage=2 WHERE id=?", (last_signal["id"],))

        else: # stage == 2
            outcome = "LOSS"
            msg_text = f"Sinal `LOSS`. 💔\n\nNúmero: {suggested}\nÚltimo: {number_played}\n\n[ID:{last_signal['id']}]"
            
        await _send_telegram(msg_text)
        
        if outcome in ["GREEN", "LOSS"]:
            _close_signal(last_signal["id"], outcome, number_played, stage, [], {})
            
        append_seq([number_played])
        _mark_processed(update_id)
        return
        
    # Abrir novo sinal
    tail = get_tail()
    append_seq([number_played])
    tail.append(number_played)
    
    if len(tail) < 20:
        print("Sequência muito curta, aguardando...")
        return
    
    if _get_cooldown() > 0:
        cd = _get_cooldown()
        _dec_cooldown()
        print(f"Em cooldown... faltam {cd} sinais.")
        _mark_processed(update_id)
        return
    
    suggested, probs, experts = _get_prediction(tail)
    
    # Gates para abrir sinal
    conf = probs[suggested]
    sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    gap = sorted_probs[0][1] - sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
    
    # Cooldown boost
    cd_conf_boost = CD_CONF_BOOST * _get_loss_streak()
    cd_gap_boost = CD_GAP_BOOST * _get_loss_streak()

    is_low_conf = (conf + cd_conf_boost) < CONF_MIN
    is_low_gap = (gap + cd_gap_boost) < GAP_MIN
    is_high_entropy = _entropy_norm(probs) > H_MAX

    if not ALWAYS_ENTER and (is_low_conf or is_low_gap or is_high_entropy):
        reason = "LOW_CONF" if is_low_conf else "LOW_GAP" if is_low_gap else "HIGH_ENTROPY"
        print(f"Sinal ignorado. Motivo: {reason}")
        
        info_msg = (
            f"Sinal ignorado.\n"
            f"Motivo: {reason}\n"
            f"Conf: {conf*100:.2f}% | Gap: {gap*100:.1f}pp\n"
            f"Cooldown: {_get_cooldown()} | Streak RED: {_get_loss_streak()}"
        )
        
        await _send_telegram(info_msg)
        _mark_processed(update_id)
        return
    
    # Abrir sinal
    with _tx() as con:
        cur = con.execute("""
            INSERT INTO pending (created_at, suggested, opened_at, stage, open, ctx1, ctx2, ctx3, ctx4)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (now_ts(), suggested, now_ts(), 0, 1,
              experts["e1"], experts["e2"], experts["e3"], experts["e4"]))
        signal_id = cur.lastrowid
        
    msg_text = (
        f"Sinal `NOVO` - `{suggested}`\n\n"
        f"Conf: {conf*100:.2f}% | Gap: {gap*100:.1f}pp\n"
        f"Cooldown: {_get_cooldown()} | Streak RED: {_get_loss_streak()}\n"
        f"Score: {score_text()}\n"
        f"ID: {signal_id}"
    )
    
    await _send_telegram(msg_text)
    
    _mark_processed(update_id)

@app.post("/webhook/{webhook_token}")
async def handle_webhook(webhook_token: str, request: Request):
    if webhook_token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Token de webhook inválido.")
    
    body = await request.json()
    if "update_id" not in body or "message" not in body:
        return {"ok": True, "message": "Nenhum dado de mensagem no corpo do request."}
    
    update_id = body.get("update_id", "")
    message = body.get("message", {})
    
    if message:
        await _process_incoming(message, update_id)
    
    return {"ok": True}

