# -*- coding: utf-8 -*-
# Fan Tan — Guardião Híbrido (Canal-in + IA-out + Recuperação visível)
# - Escuta o canal de origem via webhook (não replica o texto do canal).
# - Atualiza o banco (timeline + ngrams) em tempo real.
# - IA sugere somente quando há "ENTRADA CONFIRMADA" no canal.
# - Envia APENAS a mensagem FIRE da IA ao canal réplica:
#     "🤖 Tiro seco por IA [FIRE]\nEntrar no <N>\nDepois do <X>" (se houver "após X" na origem)
# - RECUPERAÇÃO visível: se bater em G1/G2, envia "🟢 RECUPERAÇÃO G1/G2 — Número: N"
# - Latência mínima: processamento é síncrono no webhook (sem fila lenta).
#
# ENV necessárias (Render -> Environment):
#   TG_BOT_TOKEN       -> token do bot que POSTA no canal réplica
#   REPL_CHANNEL       -> ID do canal réplica (ex: -1002796105884)
#   WEBHOOK_TOKEN      -> segredo do endpoint (ex: meusegredo123)
#   DB_PATH            -> caminho no disco persistente (ex: /var/data/data.db/main.sqlite)
#   MIN_CONF_IA        -> conf mínima p/ FIRE (padrão: 0.30 = 30%)
#   MIN_SAMPLES        -> amostras mínimas p/ IA liberar FIRE (padrão: 200)
#   GAP_MIN            -> gap mínimo top1-top2 (padrão: 0.08)
#   FLUSH_KEY          -> chave simples p/ /debug endpoints (padrão: meusegredo123)
#
# Procfile (ex):
#   web: uvicorn guardiao_hibrido:app --host 0.0.0.0 --port $PORT
#
import os, re, json, time, sqlite3, asyncio
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone
from collections import Counter

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# ========= ENV =========
DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db/main.sqlite").strip()
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
REPL_CHANNEL   = os.getenv("REPL_CHANNEL", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "meusegredo123").strip()
SELF_LABEL_IA  = os.getenv("SELF_LABEL_IA", "Tiro seco por IA").strip()
FLUSH_KEY      = os.getenv("FLUSH_KEY", "meusegredo123").strip()

MIN_CONF_IA    = float(os.getenv("MIN_CONF_IA", "0.30"))   # 30%
MIN_SAMPLES    = int(os.getenv("MIN_SAMPLES", "200"))
GAP_MIN        = float(os.getenv("GAP_MIN", "0.08"))

if not TG_BOT_TOKEN:  print("⚠️ Defina TG_BOT_TOKEN.")
if not REPL_CHANNEL:  print("⚠️ Defina REPL_CHANNEL.")
if not WEBHOOK_TOKEN: print("⚠️ Defina WEBHOOK_TOKEN.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# ========= Modelo IA (n-grams + cauda 40) =========
WINDOW = 400
DECAY  = 0.985
W4, W3, W2, W1 = 0.38, 0.30, 0.20, 0.12
ALPHA, BETA, GAMMA = 1.05, 0.70, 0.40  # BETA/GAMMA reservados p/ futuro
CONF_CAP = 0.999

# ========= DB =========
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def exec_write(sql: str, params: tuple = ()):
    con = _connect()
    con.execute(sql, params)
    con.commit()
    con.close()

def query_all(sql: str, params: tuple = ()) -> list:
    con = _connect()
    rows = con.execute(sql, params).fetchall()
    con.close()
    return rows

def query_one(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    con = _connect()
    row = con.execute(sql, params).fetchone()
    con.close()
    return row

def init_db():
    con = _connect()
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS ngram_stats (
        n INTEGER NOT NULL, ctx TEXT NOT NULL, next INTEGER NOT NULL, weight REAL NOT NULL,
        PRIMARY KEY (n, ctx, next)
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pending_outcome (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        suggested INTEGER NOT NULL,
        stage INTEGER NOT NULL,         -- 0=G0,1=G1,2=G2
        window_left INTEGER NOT NULL,   -- quantos estágios restam (incluindo o atual)
        after_num INTEGER,              -- se houver "após X" na origem
        open INTEGER NOT NULL DEFAULT 1
    )""")
    con.commit()
    con.close()

init_db()

def now_ts() -> int:
    return int(time.time())

def append_timeline(n: int):
    exec_write("INSERT INTO timeline (created_at, number) VALUES (?,?)", (now_ts(), int(n)))

def get_recent_tail(window: int = WINDOW) -> List[int]:
    rows = query_all("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (window,))
    return [r["number"] for r in rows][::-1]

def update_ngrams(decay: float = DECAY, max_n: int = 5, window: int = WINDOW):
    tail = get_recent_tail(window)
    if len(tail) < 2: return
    for t in range(1, len(tail)):
        nxt  = tail[t]
        dist = (len(tail)-1) - t
        w = (decay ** dist)
        for n in range(2, max_n+1):
            if t-(n-1) < 0: break
            ctx = tail[t-(n-1):t]
            ctx_key = ",".join(str(x) for x in ctx)
            exec_write("""
                INSERT INTO ngram_stats (n, ctx, next, weight)
                VALUES (?,?,?,?)
                ON CONFLICT(n, ctx, next) DO UPDATE SET weight = weight + excluded.weight
            """, (n, ctx_key, int(nxt), float(w)))

def prob_from_ngrams(ctx: List[int], candidate: int) -> float:
    n = len(ctx) + 1
    if n < 2 or n > 5: return 0.0
    ctx_key = ",".join(str(x) for x in ctx)
    row = query_one("SELECT SUM(weight) AS w FROM ngram_stats WHERE n=? AND ctx=?", (n, ctx_key))
    tot = (row["w"] or 0.0) if row else 0.0
    if tot <= 0: return 0.0
    row2 = query_one("SELECT weight FROM ngram_stats WHERE n=? AND ctx=? AND next=?", (n, ctx_key, candidate))
    w = (row2["weight"] or 0.0) if row2 else 0.0
    return w / tot

def ngram_backoff_score(tail: List[int], candidate: int) -> float:
    if not tail: return 0.0
    ctx4 = tail[-4:] if len(tail)>=4 else []
    ctx3 = tail[-3:] if len(tail)>=3 else []
    ctx2 = tail[-2:] if len(tail)>=2 else []
    ctx1 = tail[-1:] if len(tail)>=1 else []
    parts=[] 
    if len(ctx4)==4: parts.append((W4, prob_from_ngrams(ctx4[:-1], candidate)))
    if len(ctx3)==3: parts.append((W3, prob_from_ngrams(ctx3[:-1], candidate)))
    if len(ctx2)==2: parts.append((W2, prob_from_ngrams(ctx2[:-1], candidate)))
    if len(ctx1)==1: parts.append((W1, prob_from_ngrams(ctx1[:-1], candidate)))
    return sum(w*p for w,p in parts)

def tail_top2_boost(tail: List[int], k:int=40) -> Dict[int, float]:
    boosts={1:1.00, 2:1.00, 3:1.00, 4:1.00}
    if not tail: return boosts
    c = Counter(tail[-k:] if len(tail)>=k else tail[:])
    freq = c.most_common()
    if len(freq)>=1: boosts[freq[0][0]]=1.04
    if len(freq)>=2: boosts[freq[1][0]]=1.02
    return boosts

def suggest_number() -> Tuple[Optional[int], float, int, Dict[int,float]]:
    base=[1,2,3,4]
    tail = get_recent_tail(WINDOW)
    boosts = tail_top2_boost(tail, k=40)
    scores={}
    for c in base:
        ng = ngram_backoff_score(tail, c)
        prior = 1.0/len(base)
        score = prior * ((ng or 1e-6) ** ALPHA) * boosts.get(c,1.0)
        scores[c]=score
    total=sum(scores.values()) or 1e-9
    post={k:v/total for k,v in scores.items()}
    a=sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    if not a:
        return None,0.0,len(tail),post
    gap = (a[0][1] - (a[1][1] if len(a)>1 else 0.0))
    number = a[0][0] if gap >= GAP_MIN else None
    conf = post.get(number,0.0) if number is not None else 0.0
    # amostras ~= soma de pesos
    row = query_one("SELECT SUM(weight) AS s FROM ngram_stats")
    samples = int((row["s"] or 0) if row else 0)
    if samples < MIN_SAMPLES:
        return None,0.0,samples,post
    # limiar de confiança
    if number is None or conf < MIN_CONF_IA:
        return None,0.0,samples,post
    return number, conf, samples, post

# ========= Telegram =========
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN or not chat_id:
        return
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": parse, "disable_web_page_preview": True},
        )

async def tg_broadcast(text: str, parse: str="HTML"):
    if REPL_CHANNEL:
        await tg_send_text(REPL_CHANNEL, text, parse)

# ========= Pendências (G0 + G1/G2 visíveis ao recuperar) =========
def open_pending(suggested:int, after_num: Optional[int]):
    exec_write("""
        INSERT INTO pending_outcome (created_at, suggested, stage, window_left, after_num, open)
        VALUES (?,?,?,?,?,1)
    """, (now_ts(), int(suggested), 0, 3, (int(after_num) if after_num else None)))

async def close_pending_with_observed(n_observed: int):
    # busca a mais antiga pendência aberta
    rows = query_all("SELECT id, suggested, stage, window_left, after_num FROM pending_outcome WHERE open=1 ORDER BY id ASC LIMIT 1")
    if not rows:
        return
    r = rows[0]
    pid, sug, stage, left, after_num = int(r["id"]), int(r["suggested"]), int(r["stage"]), int(r["window_left"]), r["after_num"]
    if int(n_observed) == sug:
        # GREEN (se stage>0 => recuperação visível)
        if stage == 0:
            # green direto
            pass
        else:
            lab = "G1" if stage==1 else "G2"
            await tg_broadcast(f"🟢 <b>RECUPERAÇÃO {lab}</b> — Número: <b>{sug}</b>")
        exec_write("UPDATE pending_outcome SET open=0, window_left=0 WHERE id=?", (pid,))
    else:
        # não bateu
        if left > 1:
            exec_write("UPDATE pending_outcome SET stage=stage+1, window_left=window_left-1 WHERE id=?", (pid,))
        else:
            # esgotou G2
            exec_write("UPDATE pending_outcome SET open=0, window_left=0 WHERE id=?", (pid,))

# ========= Parsers do canal =========
def is_analise(text:str) -> bool:
    return bool(re.search(r"\bANALISANDO\b", text, flags=re.I))

def extract_seq_raw(text: str) -> Optional[str]:
    m = re.search(r"Sequ[eê]ncia:\s*([^\n\r]+)", text, flags=re.I)
    return m.group(1).strip() if m else None

def extract_green_number(text: str) -> Optional[int]:
    pats = [
        re.compile(r"APOSTA\s+ENCERRADA.*?\bGREEN\b.*?\(([1-4])\)", re.I|re.S),
        re.compile(r"\bGREEN\b.*?N[uú]mero[:\s]*([1-4])", re.I|re.S),
        re.compile(r"\bGREEN\b.*?\(([1-4])\)", re.I|re.S),
    ]
    t = re.sub(r"\s+"," ", text)
    for rx in pats:
        m = rx.search(t)
        if m:
            return int(m.group(1))
    return None

def extract_red_last_left(text: str) -> Optional[int]:
    pats = [
        re.compile(r"APOSTA\s+ENCERRADA.*?\bRED\b.*?\(([^\)]+)\)", re.I|re.S),
        re.compile(r"\bLOSS\b.*?N[uú]mero[:\s]*([1-4])", re.I|re.S),
        re.compile(r"\bRED\b.*?\(([1-4])\)", re.I|re.S),
    ]
    t = re.sub(r"\s+"," ", text)
    for rx in pats:
        m = rx.search(t)
        if m:
            nums = re.findall(r"[1-4]", m.group(1))
            if nums: return int(nums[0])
    return None

def extract_after_num(text: str) -> Optional[int]:
    m = re.search(r"Entrar\s+ap[oó]s\s+o\s+([1-4])", text, flags=re.I)
    return int(m.group(1)) if m else None

def is_real_entry(text: str) -> bool:
    # sinal do canal que abre janela (não será replicado; só gatilha IA)
    must = (r"ENTRADA\s+CONFIRMADA", r"Mesa:\s*Fantan\s*-\s*Evolution")
    must_not = (r"\bANALISANDO\b", r"\bPlacar do dia\b", r"\bAPOSTA ENCERRADA\b")
    t = re.sub(r"\s+"," ", text).strip()
    for bad in must_not:
        if re.search(bad, t, flags=re.I): return False
    for good in must:
        if not re.search(good, t, flags=re.I): return False
    return True

# ========= FastAPI =========
class Update(BaseModel):
    update_id: int
    channel_post: Optional[dict] = None
    message: Optional[dict] = None
    edited_channel_post: Optional[dict] = None
    edited_message: Optional[dict] = None

app = FastAPI(title="Guardião Híbrido — IA-only Out", version="2.1.0")

@app.get("/")
async def root():
    row = query_one("SELECT COUNT(*) AS c FROM timeline")
    timeline_cnt = int(row["c"] or 0) if row else 0
    row2 = query_one("SELECT SUM(weight) AS s FROM ngram_stats")
    samples = int((row2["s"] or 0) if row2 else 0)
    return {"ok": True, "timeline": timeline_cnt, "samples": samples, "min_conf": MIN_CONF_IA}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()
    upd = Update(**data)
    msg = upd.channel_post or upd.message or upd.edited_channel_post or upd.edited_message
    if not msg:
        return {"ok": True}

    text = (msg.get("text") or msg.get("caption") or "").strip()

    # 0) ANALISANDO -> só alimentar timeline (cauda) rápido
    if is_analise(text):
        seq_raw = extract_seq_raw(text)
        if seq_raw:
            parts = re.findall(r"[1-4]", seq_raw)
            # vêm "da esquerda recente": jogar em ordem correta (antigo->novo)
            seq_old_to_new = [int(x) for x in parts][::-1]
            for n in seq_old_to_new:
                append_timeline(n)
            update_ngrams()
        return {"ok": True, "analise": True}

    # 1) GREEN/RED do canal -> fechar/avançar pendências + timeline
    gnum = extract_green_number(text)
    rnum = extract_red_last_left(text)
    if gnum is not None or rnum is not None:
        observed = gnum if gnum is not None else rnum
        append_timeline(int(observed))
        update_ngrams()
        await close_pending_with_observed(int(observed))
        return {"ok": True, "observed": int(observed)}

    # 2) ENTRADA CONFIRMADA -> rodar IA e ENVIAR só FIRE (sem replicar texto do canal)
    if is_real_entry(text):
        after_num = extract_after_num(text)

        num, conf, samples, post = suggest_number()
        if num is None:
            return {"ok": True, "skipped": "low_conf_or_samples", "samples": samples}

        conf_capped = max(0.0, min(float(conf), CONF_CAP))
        if after_num:
            txt = (f"🤖 <b>{SELF_LABEL_IA} [FIRE]</b>\n"
                   f"🎯 Entrar no <b>{num}</b>\n"
                   f"↪️ Depois do <b>{after_num}</b>\n"
                   f"📈 Conf: <b>{conf_capped*100:.2f}%</b> | Amostra≈<b>{samples}</b>")
        else:
            txt = (f"🤖 <b>{SELF_LABEL_IA} [FIRE]</b>\n"
                   f"🎯 Entrar no <b>{num}</b>\n"
                   f"📈 Conf: <b>{conf_capped*100:.2f}%</b> | Amostra≈<b>{samples}</b>")

        # enviar já (latência mínima)
        await tg_broadcast(txt)

        # abrir pendência (G0->G1->G2)
        open_pending(int(num), after_num)

        return {"ok": True, "fire": int(num), "conf": conf_capped, "samples": samples}

    # 3) Demais mensagens do canal: ignorar (mas sem travar o fluxo)
    return {"ok": True, "ignored": True}

# ========= Debug / Saúde =========
def _fmt_bytes(n: int) -> str:
    try:
        n = float(n)
    except Exception:
        return "—"
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} EB"

@app.get("/debug/health")
async def debug_health(key: str = ""):
    if not key or key != FLUSH_KEY:
        return {"ok": False, "error": "unauthorized"}
    try:
        row_t = query_one("SELECT COUNT(*) AS c FROM timeline")
        row_n = query_one("SELECT COUNT(*) AS r, SUM(weight) AS s FROM ngram_stats")
        row_p = query_one("SELECT COUNT(*) AS o FROM pending_outcome WHERE open=1")
        return {
            "ok": True,
            "timeline": int(row_t["c"] or 0),
            "ngrams_rows": int(row_n["r"] or 0),
            "ngrams_samples": int(row_n["s"] or 0),
            "pending_open": int(row_p["o"] or 0),
            "min_conf": MIN_CONF_IA,
            "min_samples": MIN_SAMPLES,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/debug/ping")
async def debug_ping(key: str = ""):
    if not key or key != FLUSH_KEY:
        return {"ok": False, "error": "unauthorized"}
    try:
        await tg_broadcast("🔔 Ping de teste: o bot está conseguindo postar no canal.")
        return {"ok": True, "sent": True, "channel": REPL_CHANNEL}
    except Exception as e:
        return {"ok": False, "error": str(e)}