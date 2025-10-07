#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GuardiAo Auto Bot — webhook_app.py
v7.1.1-g0paren  (G0-only; fecha pelo último número entre parênteses)
- Parser flexível (entrada/analisando/fechamento)
- Fecha G0 usando apenas o número dentro de parênteses (ex.: GREEN (3) ou RED (4))
- Anti-trava: timeout de pendência (fecha LOSS G0 X)
- Endpoints de emergência: /admin/status e /admin/unlock
- Dedupe por conteúdo e DB SQLite
"""

import os, re, time, sqlite3, datetime, hashlib
from typing import List, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException

# ================== ENV ==================
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "").strip()

SHOW_DEBUG       = os.getenv("SHOW_DEBUG", "False").strip().lower() == "true"
MAX_GALE         = int(os.getenv("MAX_GALE", "0"))        # G0 por padrão
OBS_TIMEOUT_SEC  = int(os.getenv("OBS_TIMEOUT_SEC", "420"))
DEDUP_WINDOW_SEC = int(os.getenv("DEDUP_WINDOW_SEC", "40"))

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN or not TARGET_CHANNEL:
    raise RuntimeError("Faltam ENV: TG_BOT_TOKEN, WEBHOOK_TOKEN, TARGET_CHANNEL.")
TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
DB_PATH = "/opt/render/project/src/main.sqlite"

# ================== APP ==================
app = FastAPI(title="GuardiAo Auto Bot (webhook)", version="7.1.1-g0paren")

# ================== DB ==================
def _con():
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

def db_init():
    con = _con(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS processed(
        update_id TEXT PRIMARY KEY,
        seen_at   INTEGER NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pending(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER,
        opened_at  INTEGER,
        suggested  INTEGER,
        seen       TEXT     DEFAULT '',
        open       INTEGER  DEFAULT 1
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS score(
        id INTEGER PRIMARY KEY CHECK(id=1),
        green INTEGER DEFAULT 0,
        loss  INTEGER DEFAULT 0
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS dedupe(
        kind TEXT NOT NULL,
        dkey TEXT NOT NULL,
        ts   INTEGER NOT NULL,
        PRIMARY KEY (kind, dkey)
    )""")
    if not con.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        con.execute("INSERT INTO score(id,green,loss) VALUES(1,0,0)")
    con.commit(); con.close()
db_init()

def _mark_processed(upd: str):
    try:
        con = _con()
        con.execute("INSERT OR IGNORE INTO processed(update_id,seen_at) VALUES(?,?)",
                    (str(upd), int(time.time())))
        con.commit(); con.close()
    except Exception:
        pass

def _timeline_tail(n:int=400)->List[int]:
    con=_con()
    rows=con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?",(n,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def _append_seq(seq: List[int]):
    if not seq: return
    con=_con(); now=int(time.time())
    con.executemany("INSERT INTO timeline(created_at,number) VALUES(?,?)",
                    [(now,int(x)) for x in seq])
    con.commit(); con.close()

def _timeline_size()->int:
    con=_con(); row=con.execute("SELECT COUNT(*) c FROM timeline").fetchone(); con.close()
    return int(row["c"] or 0)

def _score_add(outcome:str):
    con=_con()
    row=con.execute("SELECT green,loss FROM score WHERE id=1").fetchone()
    g,l = (int(row["green"]), int(row["loss"])) if row else (0,0)
    if outcome.upper()=="GREEN": g+=1
    elif outcome.upper()=="LOSS": l+=1
    con.execute("INSERT OR REPLACE INTO score(id,green,loss) VALUES(1,?,?)",(g,l))
    con.commit(); con.close()

def _score_text()->str:
    con=_con(); row=con.execute("SELECT green,loss FROM score WHERE id=1").fetchone(); con.close()
    g,l = (int(row["green"]), int(row["loss"])) if row else (0,0)
    tot=g+l; acc=(g/tot*100.0) if tot>0 else 0.0
    return f"{g} GREEN × {l} LOSS — {acc:.1f}%"

def _pending_get()->Optional[sqlite3.Row]:
    con=_con(); row=con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone(); con.close()
    return row

def _pending_open(suggested:int):
    if _pending_get(): return False
    con=_con(); now=int(time.time())
    con.execute("""INSERT INTO pending(created_at,opened_at,suggested,seen,open)
                   VALUES(?,?,?,?,1)""",(now, now, int(suggested), ""))
    con.commit(); con.close()
    return True

def _pending_seen_set(v:str):
    row=_pending_get()
    if not row: return
    con=_con(); con.execute("UPDATE pending SET seen=? WHERE id=?", (v, int(row["id"]))); con.commit(); con.close()

def _pending_close(final_seen: str, outcome: str, stage_lbl: str, suggested:int)->str:
    row=_pending_get()
    if not row: return ""
    con=_con()
    con.execute("UPDATE pending SET open=0, seen=? WHERE id=?", (final_seen, int(row["id"])))
    con.commit(); con.close()
    _score_add(outcome)
    obs=[int(x) for x in final_seen.split("-") if x.isdigit()]
    _append_seq(obs)
    our = suggested if outcome.upper()=="GREEN" else "X"
    snap=_ngram_snapshot(suggested)
    return (f"{'🟢' if outcome.upper()=='GREEN' else '🔴'} <b>{outcome.upper()}</b> — finalizado "
            f"(<b>{stage_lbl}</b>, nosso={our}, observados={final_seen}).\n"
            f"📊 Geral: {_score_text()}\n\n{snap}")

# ============== ANTI-TRAVA (timeout) ==============
def _pending_timeout_check() -> Optional[dict]:
    row = _pending_get()
    if not row:
        return None
    opened_at = int(row["opened_at"] or 0)
    now = int(time.time())
    if now - opened_at >= OBS_TIMEOUT_SEC:
        suggested = int(row["suggested"] or 0)
        final_seen = "X"
        outcome = "LOSS"
        stage_lbl = "G0"
        msg_txt = _pending_close(final_seen, outcome, stage_lbl, suggested)
        return {"timeout_closed": True, "final_seen": final_seen, "suggested": suggested, "msg": msg_txt}
    return None

# ============== DEDUPE ==============
def _dedupe_key(text: str) -> str:
    base = re.sub(r"\s+", " ", (text or "")).strip().lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _seen_recent(kind: str, dkey: str) -> bool:
    now=int(time.time())
    con=_con()
    row=con.execute("SELECT ts FROM dedupe WHERE kind=? AND dkey=?", (kind, dkey)).fetchone()
    if row and now - int(row["ts"]) <= DEDUP_WINDOW_SEC:
        con.close(); return True
    con.execute("INSERT OR REPLACE INTO dedupe(kind, dkey, ts) VALUES (?,?,?)", (kind, dkey, now))
    con.commit(); con.close()
    return False

# ============== IA compacta (4 especialistas + ajustes) ==============
def _norm(d: Dict[int,float])->Dict[int,float]:
    s=sum(d.values()) or 1e-9
    return {k:v/s for k,v in d.items()}

def _post_freq(tail:List[int], k:int)->Dict[int,float]:
    if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
    win = tail[-k:] if len(tail)>=k else tail
    tot=max(1,len(win))
    return _norm({c:win.count(c)/tot for c in (1,2,3,4)})

def _post_e1_ngram(tail:List[int])->Dict[int,float]:
    mix={c:0.0 for c in (1,2,3,4)}
    for k,w in ((8,0.25),(21,0.35),(55,0.40)):
        pk=_post_freq(tail,k)
        for c in (1,2,3,4): mix[c]+=w*pk[c]
    return _norm(mix)

def _post_e2_short(tail):  return _post_freq(tail, 60)
def _post_e3_long(tail):   return _post_freq(tail, 300)
def _post_e4_llm(tail):    return {1:0.25,2:0.25,3:0.25,4:0.25}

def _hedge(p1,p2,p3,p4, w=(0.40,0.25,0.25,0.10)):
    cands=(1,2,3,4)
    return _norm({c: w[0]*p1.get(c,0)+w[1]*p2.get(c,0)+w[2]*p3.get(c,0)+w[3]*p4.get(c,0) for c in cands})

def _runnerup_ls2(post:Dict[int,float], loss_streak:int)->Tuple[int,Dict[int,float],str]:
    rank=sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    best=rank[0][0]
    if loss_streak>=2 and len(rank)>=2 and (rank[0][1]-rank[1][1])<0.05:
        return rank[1][0], post, "IA_runnerup_ls2"
    return best, post, "IA"

def _conf_floor(post:Dict[int,float], floor=0.30, cap=0.95):
    post=_norm({c:float(post.get(c,0)) for c in (1,2,3,4)})
    b=max(post,key=post.get); mx=post[b]
    if mx<floor:
        others=[c for c in (1,2,3,4) if c!=b]
        s=sum(post[c] for c in others)
        take=min(floor-mx, s)
        if s>0:
            scale=(s-take)/s
            for c in others: post[c]*=scale
        post[b]=min(cap, mx+take)
    if post[b]>cap:
        ex=post[b]-cap; post[b]=cap
        add=ex/3.0
        for c in (1,2,3,4):
            if c!=b: post[c]+=add
    return _norm(post)

def _get_ls()->int: return 0

def _choose_number()->Tuple[int,float,int,Dict[int,float],float,str]:
    tail=_timeline_tail(400)
    p1=_post_e1_ngram(tail); p2=_post_e2_short(tail); p3=_post_e3_long(tail); p4=_post_e4_llm(tail)
    base=_hedge(p1,p2,p3,p4)
    best, post, reason=_runnerup_ls2(base, loss_streak=_get_ls())
    post=_conf_floor(post, 0.30, 0.95)
    best=max(post,key=post.get); conf=float(post[best])
    r=sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    gap=(r[0][1]-r[1][1]) if len(r)>=2 else r[0][1]
    return best, conf, _timeline_size(), post, gap, reason

def _ngram_snapshot(suggested:int)->str:
    tail=_timeline_tail(400); post=_post_e1_ngram(tail)
    pct=lambda x:f"{x*100:.1f}%"
    p1,p2,p3,p4=pct(post[1]), pct(post[2]), pct(post[3]), pct(post[4])
    conf=pct(post.get(int(suggested),0.0))
    return (f"📈 Amostra: {_timeline_size()} • Conf: {conf}\n"
            f"🔎 E1(n-gram proxy): 1 {p1} | 2 {p2} | 3 {p3} | 4 {p4}")

# ================== Telegram ==================
async def tg_send(chat_id: str, text: str, parse="HTML"):
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            await cli.post(f"{TELEGRAM_API}/sendMessage",
                           json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                 "disable_web_page_preview": True})
    except Exception:
        pass

async def tg_send_return(chat_id: str, text: str, parse="HTML") -> Optional[int]:
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            r = await cli.post(f"{TELEGRAM_API}/sendMessage",
                               json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                     "disable_web_page_preview": True})
            data = r.json()
            if isinstance(data, dict) and data.get("ok") and data.get("result", {}).get("message_id"):
                return int(data["result"]["message_id"])
    except Exception:
        pass
    return None

async def tg_delete(chat_id: str, message_id: int):
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            await cli.post(f"{TELEGRAM_API}/deleteMessage",
                           json={"chat_id": chat_id, "message_id": int(message_id)})
    except Exception:
        pass

# ================== Parser ==================
RX_ENTRADA = re.compile(r"(💰\s*)?ENTRADA.*CONFIRMADA|ENTRADA\s*OK", re.I)
RX_ANALISE = re.compile(r"\bANALIS(A|Á)NDO\b|ANALISE|🧩", re.I)
RX_FECHA   = re.compile(r"APOSTA.*ENCERRADA|RESULTADO|GREEN|RED|✅|❌", re.I)

RX_SEQ     = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
RX_NUMS    = re.compile(r"[1-4]")
RX_AFTER   = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)
RX_PAREN   = re.compile(r"\(([^\)]*)\)")

def _parse_seq_list(text:str)->List[int]:
    m=RX_SEQ.search(text or "");  return [int(x) for x in RX_NUMS.findall(m.group(1))] if m else []

def _parse_after(text:str)->Optional[int]:
    m=RX_AFTER.search(text or "");  return int(m.group(1)) if m else None

def _parse_paren_last_one(text:str)->Optional[int]:
    """
    Retorna o último número 1..4 encontrado ENTRE PARÊNTESES na mensagem.
    Ex.: '... GREEN (3)' -> 3 ; '... RED ❌ (4)' -> 4 ; se não achar, None.
    """
    nums=[]
    for m in RX_PAREN.finditer(text or ""):
        nums_in = [int(x) for x in RX_NUMS.findall(m.group(1))]
        if nums_in:
            nums.append(nums_in[-1])  # pega o último de cada parênteses
    return nums[-1] if nums else None

# ================== Rotas básicas ==================
@app.get("/")
async def root():
    return {"ok": True, "service": "GuardiAo Auto Bot", "time": datetime.datetime.utcnow().isoformat()+"Z"}

@app.get("/health")
async def health():
    return {"ok": True, "db_exists": os.path.exists(DB_PATH), "db_path": DB_PATH}

@app.get("/debug_cfg")
async def debug_cfg():
    return {"MAX_GALE": MAX_GALE, "OBS_TIMEOUT_SEC": OBS_TIMEOUT_SEC, "DEDUP_WINDOW_SEC": DEDUP_WINDOW_SEC}

# ----- Admin helpers -----
@app.get("/admin/status")
async def admin_status():
    pend = _pending_get()
    if not pend:
        return {"open": False}
    return {
        "open": True,
        "id": int(pend["id"]),
        "opened_at": int(pend["opened_at"] or 0),
        "age_sec": int(time.time()) - int(pend["opened_at"] or 0),
        "suggested": int(pend["suggested"] or 0),
        "seen": (pend["seen"] or "")
    }

@app.post("/admin/unlock")
async def admin_unlock():
    pend = _pending_get()
    if not pend:
        return {"ok": True, "message": "nenhuma pendência aberta"}
    suggested = int(pend["suggested"] or 0)
    msg_txt = _pending_close("X", "LOSS", "G0", suggested)
    return {"ok": True, "forced_close": True, "message": msg_txt}

# ================== Webhook ==================
@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()
    upd_id = str(data.get("update_id", "")); _mark_processed(upd_id)

    msg = data.get("channel_post") or data.get("message") \
        or data.get("edited_channel_post") or data.get("edited_message") or {}
    chat = msg.get("chat") or {}
    chat_id = str(chat.get("id") or "")
    text = (msg.get("text") or msg.get("caption") or "").strip()

    # watchdog anti-trava
    try:
        watchdog = _pending_timeout_check()
        if watchdog and SHOW_DEBUG:
            if watchdog.get("msg"):
                await tg_send(TARGET_CHANNEL, f"DEBUG: Timeout pendência — fechado automático.\n{watchdog['msg']}")
            else:
                await tg_send(TARGET_CHANNEL, "DEBUG: Timeout pendência — fechado automático (LOSS G0 X).")
    except Exception:
        pass

    if SOURCE_CHANNEL and chat_id and chat_id != SOURCE_CHANNEL:
        if SHOW_DEBUG:
            await tg_send(TARGET_CHANNEL, f"DEBUG: Ignorando chat {chat_id}. Fonte esperada: {SOURCE_CHANNEL}")
        return {"ok": True, "skipped": "wrong_source"}

    # -------- ANALISANDO --------
    if RX_ANALISE.search(text):
        if _seen_recent("analise", _dedupe_key(text)):
            return {"ok": True, "skipped": "analise_dupe"}
        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)
        if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, "DEBUG: Análise reconhecida ✅")
        return {"ok": True, "analise_seq": len(seq)}

    # -------- FECHAMENTO (G0 pelo parênteses) --------
    if RX_FECHA.search(text):
        if _seen_recent("fechamento", _dedupe_key(text)):
            return {"ok": True, "skipped": "fechamento_dupe"}

        pend=_pending_get()
        if pend:
            suggested=int(pend["suggested"] or 0)

            # Só usamos o ÚLTIMO número entre parênteses para decidir
            obs = _parse_paren_last_one(text)   # 1..4 ou None
            if obs is not None:
                _pending_seen_set(str(obs))

            # Decide G0
            seen = (_pending_get()["seen"] or "").strip()
            outcome="LOSS"; stage_lbl="G0"
            if seen.isdigit() and int(seen)==suggested:
                outcome="GREEN"

            # Fecha imediatamente (G0 only)
            final_seen = seen if seen else "X"
            msg_txt=_pending_close(final_seen, outcome, stage_lbl, suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)
            return {"ok": True, "closed": outcome, "seen": final_seen}

        if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, "DEBUG: Fechamento reconhecido ✅ — sem pendência aberta")
        return {"ok": True, "noted_close": True}

    # -------- ENTRADA --------
    if RX_ENTRADA.search(text):
        if _seen_recent("entrada", _dedupe_key(text)):
            if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, "DEBUG: Entrada duplicada ignorada (conteúdo repetido).")
            return {"ok": True, "skipped": "entrada_dupe"}

        # Alimenta memória a partir da "Sequência", mas isso NÃO decide o fechamento
        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)
        after=_parse_after(text)

        # Fecha pendência esquecida (como X)
        pend=_pending_get()
        if pend:
            suggested=int(pend["suggested"] or 0)
            seen=(pend["seen"] or "").strip()
            final_seen = seen if seen else "X"
            outcome="LOSS"; stage_lbl="G0"
            if seen.isdigit() and int(seen)==suggested:
                outcome="GREEN"
            msg_txt=_pending_close(final_seen, outcome, stage_lbl, suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)

        # Mensagem de “Analisando...”
        analyzing_id = await tg_send_return(TARGET_CHANNEL, "⏳ Analisando padrão, aguarde...")

        # Sugestão
        best, conf, samples, post, gap, reason = _choose_number()
        opened=_pending_open(best)
        if opened:
            aft_txt = f" após {after}" if after else ""
            txt=(f"🤖 <b>IA SUGERE</b> — <b>{best}</b>\n"
                 f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
                 f"📊 <b>Conf:</b> {conf*100:.2f}% | <b>Amostra≈</b>{samples} | <b>gap≈</b>{gap*100:.1f}pp\n"
                 f"🧠 <b>Modo:</b> {reason}\n"
                 f"{_ngram_snapshot(best)}")
            await tg_send(TARGET_CHANNEL, txt)
            if analyzing_id is not None: await tg_delete(TARGET_CHANNEL, analyzing_id)
            if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, "DEBUG: Entrada reconhecida ✅ — sugestão enviada.")
            return {"ok": True, "entry_opened": True, "best": best, "conf": conf}
        else:
            if analyzing_id is not None: await tg_delete(TARGET_CHANNEL, analyzing_id)
            if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, "DEBUG: pending já aberto; entrada ignorada.")
            return {"ok": True, "skipped": "pending_open"}

    # -------- Não reconhecido --------
    if SHOW_DEBUG:
        await tg_send(TARGET_CHANNEL, "DEBUG: Mensagem não reconhecida como ENTRADA/FECHAMENTO/ANALISANDO.")
    return {"ok": True, "skipped": "unmatched"}