#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Webhook único: replica sinais do canal-fonte, decide 1 número seco e publica,
# acompanhando G0 -> G1 -> G2 e marcando GREEN/LOSS ao final.

import os, re, json, time, sqlite3
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, HTTPException

# ========= ENV =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()

# Canal destino (onde você quer publicar o tiro seco)
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1002796105884").strip()   # @iafantan

# Banco local simples (persiste histórico leve + pendência)
DB_PATH        = os.getenv("DB_PATH", "/data/mini_ref.db").strip() or "/data/mini_ref.db"

TELEGRAM_API   = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

app = FastAPI(title="guardiao-auto-bot (webhook único com G1/G2 + GREEN/LOSS)", version="1.2.0")

# ========= DB helpers =========
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_db():
    con = _connect()
    cur = con.cursor()
    # histórico de números (para n-gram)
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    # n-grams 2..5
    cur.execute("""CREATE TABLE IF NOT EXISTS ngram (
        n INTEGER NOT NULL, ctx TEXT NOT NULL, nxt INTEGER NOT NULL, w REAL NOT NULL,
        PRIMARY KEY (n, ctx, nxt)
    )""")
    # pendência de uma janela (G0->G2), por origem
    cur.execute("""CREATE TABLE IF NOT EXISTS pending (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_msg_id INTEGER,
        stage INTEGER NOT NULL,             -- 0=G0, 1=G1, 2=G2
        suggested INTEGER NOT NULL,         -- número seco escolhido
        open INTEGER NOT NULL,              -- 1=aberta, 0=fechada
        created_at INTEGER NOT NULL
    )""")
    con.commit(); con.close()

init_db()

def now_ts() -> int:
    return int(time.time())

# ========= Timeline / ngram =========
def get_tail(limit:int=400) -> List[int]:
    con = _connect()
    rows = con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def append_seq(seq: List[int]):
    if not seq: return
    con = _connect()
    cur = con.cursor()
    for n in seq:
        cur.execute("INSERT INTO timeline (created_at, number) VALUES (?,?)", (now_ts(), int(n)))
    con.commit(); con.close()
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
    row_tot = con.execute("SELECT SUM(w) AS s FROM ngram WHERE n=? AND ctx=?", (n, ctx_key)).fetchone()
    tot = (row_tot["s"] or 0.0) if row_tot else 0.0
    if tot <= 0:
        con.close(); return 0.0
    row_c = con.execute("SELECT w FROM ngram WHERE n=? AND ctx=? AND nxt=?", (n, ctx_key, int(cand))).fetchone()
    w = (row_c["w"] or 0.0) if row_c else 0.0
    con.close()
    return w / tot

# ========= Parsers do canal-fonte =========
ENTRY_RX   = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
SEQ_RX     = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
AFTER_RX   = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)
TARGET_RX  = re.compile(r"apostar?\s+em\s+(BIG|SMALL|ODD|EVEN)", re.I)

GALE1_RX   = re.compile(r"Estamos\s+no\s*1[ºo]\s*gale", re.I)
GALE2_RX   = re.compile(r"Estamos\s+no\s*2[ºo]\s*gale", re.I)

# Encerramento GREEN/RED (vários formatos possíveis)
GREEN_RXS = [
    re.compile(r"APOSTA\s+ENCERRADA.*?\bGREEN\b.*?\(([1-4])\)", re.I | re.S),
    re.compile(r"\bGREEN\b.*?Número[:\s]*([1-4])", re.I | re.S),
    re.compile(r"\bGREEN\b.*?\(([1-4])\)", re.I | re.S),
]
RED_RXS = [
    re.compile(r"APOSTA\s+ENCERRADA.*?\bRED\b.*?\(([1-4])\)", re.I | re.S),
    re.compile(r"\bRED\b.*?\(([1-4])\)", re.I | re.S),
]

def parse_entry_text(text: str) -> Optional[Dict]:
    """Extrai os campos da Entrada Confirmada."""
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

    mtarget = TARGET_RX.search(t)
    target = mtarget.group(1).upper() if mtarget else None

    return {"seq": seq, "after": after_num, "target": target, "raw": t}

def is_gale1(text:str) -> bool:
    return bool(GALE1_RX.search(re.sub(r"\s+", " ", text)))

def is_gale2(text:str) -> bool:
    return bool(GALE2_RX.search(re.sub(r"\s+", " ", text)))

def extract_green(text:str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text)
    for rx in GREEN_RXS:
        m = rx.search(t); 
        if m:
            nums = re.findall(r"[1-4]", m.group(1))
            if nums: return int(nums[0])
    return None

def extract_red(text:str) -> Optional[int]:
    t = re.sub(r"\s+", " ", text)
    for rx in RED_RXS:
        m = rx.search(t)
        if m:
            nums = re.findall(r"[1-4]", m.group(1))
            if nums: return int(nums[0])
    return None

# ========= Estratégia de decisão (sempre escolhe 1 número) =========
W4, W3, W2, W1 = 0.40, 0.30, 0.20, 0.10

def target_to_candidates(target: Optional[str]) -> List[int]:
    if not target: return [1,2,3,4]
    if target == "SMALL": return [1,2]
    if target == "BIG":   return [3,4]
    if target == "ODD":   return [1,3]
    if target == "EVEN":  return [2,4]
    return [1,2,3,4]

def _ngram_backoff(tail: List[int], after: Optional[int], cand:int) -> float:
    if not tail: return 0.0
    # Se “após X”, corta o contexto no último X (se houver)
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

def choose_single_number(cands: List[int], after: Optional[int]) -> Tuple[int, float, int]:
    """Retorna (melhor, conf_normalizada, amostras~) — nunca abstem."""
    tail = get_tail(400)
    scores = {c: _ngram_backoff(tail, after, c) for c in cands}
    # Se todos 0, desempata por quem saiu MENOS nos últimos 50
    if all(v == 0.0 for v in scores.values()):
        last = tail[-50:] if len(tail) >= 50 else tail
        freq = {c: last.count(c) for c in cands}
        best = sorted(cands, key=lambda x: (freq.get(x,0), x))[0]
        conf = 0.50
        samples = len(tail)
        return best, conf, samples

    total = sum(scores.values()) or 1e-9
    post = {k: v/total for k,v in scores.items()}
    best = max(post.items(), key=lambda kv: kv[1])[0]
    conf = post[best]
    samples = len(tail)
    return best, conf, samples

# ========= Telegram =========
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN: return
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                "disable_web_page_preview": True})

# ========= Pendente (G0->G2) =========
def pend_get_open() -> Optional[sqlite3.Row]:
    con = _connect()
    row = con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone()
    con.close()
    return row

def pend_open(source_msg_id: Optional[int], suggested:int):
    con = _connect()
    con.execute("""INSERT INTO pending (source_msg_id, stage, suggested, open, created_at)
                   VALUES (?,?,?,?,?)""", (int(source_msg_id or 0), 0, int(suggested), 1, now_ts()))
    con.commit(); con.close()

def pend_stage(next_stage:int):
    con = _connect()
    con.execute("UPDATE pending SET stage=? WHERE open=1", (int(next_stage),))
    con.commit(); con.close()

def pend_close():
    con = _connect()
    con.execute("UPDATE pending SET open=0 WHERE open=1")
    con.commit(); con.close()

# ========= Rotas =========
@app.get("/")
async def root():
    return {"ok": True, "service": "guardiao-auto-bot (webhook único G0/G1/G2 + GREEN/LOSS)"}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()

    # Extrai o texto (canal/mensagem/edit)
    msg = data.get("channel_post") or data.get("message") \
          or data.get("edited_channel_post") or data.get("edited_message") or {}
    text = (msg.get("text") or msg.get("caption") or "").strip()
    source_msg_id = msg.get("message_id")

    if not text:
        return {"ok": True, "skipped": "sem texto"}

    tnorm = re.sub(r"\s+", " ", text)

    # 1) GREEN / RED (encerramento) — fecha pendência e publica resultado
    gnum = extract_green(tnorm)
    rnum = extract_red(tnorm)
    if gnum is not None or rnum is not None:
        pend = pend_get_open()
        if pend:
            sug = int(pend["suggested"])
            stage = int(pend["stage"])
            if gnum is not None:
                hit = (int(gnum) == sug)
                if hit:
                    await tg_send_text(
                        TARGET_CHANNEL,
                        f"✅ <b>GREEN</b> em <b>{'G0' if stage==0 else ('G1' if stage==1 else 'G2')}</b> — Número: <b>{sug}</b>"
                    )
                    pend_close()
                else:
                    # GREEN em outro número -> trata como LOSS desta janela (canal fonte foi green, mas não no nosso seco)
                    await tg_send_text(
                        TARGET_CHANNEL,
                        f"❌ <b>LOSS</b> — Nosso número: <b>{sug}</b> | Green do canal: <b>{gnum}</b>"
                    )
                    pend_close()
            else:
                # RED detectado: se já estávamos no G2, é loss; senão aguardará gale seguinte
                if stage >= 2:
                    await tg_send_text(TARGET_CHANNEL, f"❌ <b>LOSS</b> — Número: <b>{sug}</b> (em G2)")
                    pend_close()
        return {"ok": True, "observed": ("GREEN" if gnum is not None else "RED")}

    # 2) Mensagens de GALE — avança estágio e republica status
    if is_gale1(tnorm):
        pend = pend_get_open()
        if pend:
            pend_stage(1)
            sug = int(pend["suggested"])
            await tg_send_text(TARGET_CHANNEL, f"🔁 Indo para <b>G1</b> — mantendo número seco: <b>{sug}</b>")
        return {"ok": True, "gale": 1}

    if is_gale2(tnorm):
        pend = pend_get_open()
        if pend:
            pend_stage(2)
            sug = int(pend["suggested"])
            await tg_send_text(TARGET_CHANNEL, f"🔁 Indo para <b>G2</b> — mantendo número seco: <b>{sug}</b>")
        return {"ok": True, "gale": 2}

    # 3) ENTRADA CONFIRMADA — decide tiro seco (sempre) e publica (G0)
    parsed = parse_entry_text(text)
    if not parsed:
        return {"ok": True, "skipped": "nao_eh_evento_relevante"}

    # Alimenta histórico com a sequência passada, antes de decidir
    seq = parsed["seq"] or []
    after = parsed["after"]
    target = parsed["target"] or "GEN"
    append_seq(seq)

    cands = target_to_candidates(target)
    best, conf, samples = choose_single_number(cands, after)

    # Abre pendência
    pend_close()  # garante fila única
    pend_open(source_msg_id, best)

    # Monta e publica G0
    aft_txt = f" após {after}" if after else ""
    out = (
        f"🎯 <b>Número seco (G0):</b> <b>{best}</b>\n"
        f"🧩 <b>Padrão:</b> {target}{aft_txt}\n"
        f"📊 Conf: <b>{conf*100:.2f}%</b> | Amostra≈<b>{samples}</b>"
    )
    if TG_BOT_TOKEN and TARGET_CHANNEL:
        await tg_send_text(TARGET_CHANNEL, out)

    return {"ok": True, "posted": True, "best": best, "conf": conf, "samples": samples,
            "cands": cands, "target": target, "after": after, "source_msg_id": source_msg_id}