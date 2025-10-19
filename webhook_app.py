#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GuardiAo Auto Bot — webhook_app.py
v7.1 (G1-only, fechamento estrito, IA Fibo-Mix dinâmica, estudo TOP-3, dedupe,
      fechamento correto G0/G1, “Analisando...” com auto-delete, DB SQLite)

ENV obrigatórias (Render -> Environment):
- TG_BOT_TOKEN
- WEBHOOK_TOKEN
- SOURCE_CHANNEL           ex: -1002810508717
- TARGET_CHANNEL           ex: -1003052132833

ENV opcionais:
- SHOW_DEBUG          (default False)
- MAX_GALE            (default 1)          # G0/G1
- OBS_TIMEOUT_SEC     (default 420)
- DEDUP_WINDOW_SEC    (default 40)
- FIBO_WINDOWS        (default "8,13,21,34,55")
- SHOW_TOP3           (default "True")

Start command:
  uvicorn webhook_app:app --host 0.0.0.0 --port $PORT
"""

import os, re, json, time, math, sqlite3, datetime, hashlib
from typing import List, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException

# ------------------------------------------------------
# ENV & constantes
# ------------------------------------------------------
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "").strip()

SHOW_DEBUG       = os.getenv("SHOW_DEBUG", "False").strip().lower() == "true"
MAX_GALE         = int(os.getenv("MAX_GALE", "1"))
OBS_TIMEOUT_SEC  = int(os.getenv("OBS_TIMEOUT_SEC", "420"))
DEDUP_WINDOW_SEC = int(os.getenv("DEDUP_WINDOW_SEC", "40"))

# >>> Novas ENVs
def _parse_fibo_env(txt: str) -> List[int]:
    out=[]
    for p in (txt or "").split(","):
        p=p.strip()
        if not p: continue
        try:
            k=int(p)
            if k>0: out.append(k)
        except:
            pass
    # fallback seguro
    return out or [8,13,21,34,55]

FIBO_WINDOWS = _parse_fibo_env(os.getenv("FIBO_WINDOWS", "8,13,21,34,55"))
SHOW_TOP3    = os.getenv("SHOW_TOP3", "True").strip().lower() == "true"

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN or not TARGET_CHANNEL:
    raise RuntimeError("Faltam ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN, TARGET_CHANNEL.")
TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

DB_PATH = "/opt/render/project/src/main.sqlite"

# ------------------------------------------------------
# App
# ------------------------------------------------------
app = FastAPI(title="GuardiAo Auto Bot (webhook)", version="7.1")

# ------------------------------------------------------
# DB helpers
# ------------------------------------------------------
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
        stage      INTEGER DEFAULT 0,
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
        con.execute("INSERT OR IGNORE INTO processed(update_id,seen_at) VALUES(?,?)",(str(upd), int(time.time())))
        con.commit(); con.close()
    except Exception:
        pass

def _timeline_tail(n:int=400)->List[int]:
    con = _con()
    rows = con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?",(n,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def _append_seq(seq: List[int]):
    if not seq: return
    try:
        con = _con()
        now = int(time.time())
        con.executemany("INSERT INTO timeline(created_at,number) VALUES(?,?)",[(now,int(x)) for x in seq])
        con.commit(); con.close()
    except Exception:
        pass

def _timeline_size()->int:
    con=_con(); row=con.execute("SELECT COUNT(*) c FROM timeline").fetchone(); con.close()
    return int(row["c"] or 0)

def _score_add(outcome:str):
    try:
        con = _con()
        row = con.execute("SELECT green,loss FROM score WHERE id=1").fetchone()
        g,l = (int(row["green"]), int(row["loss"])) if row else (0,0)
        if outcome.upper()=="GREEN": g+=1
        elif outcome.upper()=="LOSS": l+=1
        con.execute("INSERT OR REPLACE INTO score(id,green,loss) VALUES(1,?,?)",(g,l))
        con.commit(); con.close()
    except Exception:
        pass

def _score_text()->str:
    con = _con(); row = con.execute("SELECT green,loss FROM score WHERE id=1").fetchone(); con.close()
    if not row: return "0 GREEN × 0 LOSS — 0.0%"
    g,l = int(row["green"]), int(row["loss"])
    tot = g+l; acc = (g/tot*100.0) if tot>0 else 0.0
    return f"{g} GREEN × {l} LOSS — {acc:.1f}%"

def _pending_get()->Optional[sqlite3.Row]:
    con = _con(); row = con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone(); con.close()
    return row

def _pending_open(suggested:int):
    if _pending_get(): return False
    con = _con()
    now = int(time.time())
    con.execute("INSERT INTO pending(created_at,opened_at,suggested,stage,seen,open) VALUES(?,?,?,?,?,1)",
                (now, now, int(suggested), 0, ""))
    con.commit(); con.close()
    return True

def _pending_set_stage(stage:int):
    con = _con()
    con.execute("UPDATE pending SET stage=? WHERE open=1",(int(stage),))
    con.commit(); con.close()

def _pending_seen_append(nums: List[int], need:int=2):
    row = _pending_get()
    if not row: return
    seen = (row["seen"] or "").strip()
    arr = [s for s in seen.split("-") if s]
    for n in nums:
        if len(arr) >= need: break
        arr.append(str(int(n)))
    txt = "-".join(arr[:need])
    con = _con(); con.execute("UPDATE pending SET seen=? WHERE id=?", (txt, int(row["id"]))); con.commit(); con.close()

def _pending_close(final_seen: str, outcome: str, stage_lbl: str, suggested:int)->str:
    row = _pending_get()
    if not row: return ""
    con = _con()
    con.execute("UPDATE pending SET open=0, seen=? WHERE id=?", (final_seen, int(row["id"])))
    con.commit(); con.close()
    _score_add(outcome)
    # alimentar timeline com observados
    obs = [int(x) for x in final_seen.split("-") if x.isdigit()]
    _append_seq(obs)
    our = suggested if outcome.upper()=="GREEN" else "X"
    snap = _ngram_snapshot(suggested)
    msg = (f"{'🟢' if outcome.upper()=='GREEN' else '🔴'} <b>{outcome.upper()}</b> — finalizado "
           f"(<b>{stage_lbl}</b>, nosso={our}, observados={final_seen}).\n"
           f"📊 Geral: {_score_text()}\n\n{snap}")
    return msg

# ------------------ DEDUPE por conteúdo ------------------
def _dedupe_key(text: str) -> str:
    base = re.sub(r"\s+", " ", (text or "")).strip().lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _seen_recent(kind: str, dkey: str) -> bool:
    now = int(time.time())
    con = _con()
    row = con.execute("SELECT ts FROM dedupe WHERE kind=? AND dkey=?", (kind, dkey)).fetchone()
    if row and now - int(row["ts"]) <= DEDUP_WINDOW_SEC:
        con.close()
        return True
    con.execute("INSERT OR REPLACE INTO dedupe(kind, dkey, ts) VALUES (?,?,?)", (kind, dkey, now))
    con.commit(); con.close()
    return False

# ------------------------------------------------------
# IA — Fibo Mix + Short/Long + ajustes
# ------------------------------------------------------
def _norm(d: Dict[int,float])->Dict[int,float]:
    s=sum(d.values()) or 1e-9
    return {k:(v/s) for k,v in d.items()}

def _post_freq(tail:List[int], k:int)->Dict[int,float]:
    if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
    win = tail[-k:] if len(tail)>=k else tail
    tot=max(1,len(win))
    return _norm({c:win.count(c)/tot for c in (1,2,3,4)})

def _weights_for_fibo(ws: List[int])->Dict[int,float]:
    # peso = 1/sqrt(k) (suaviza preferência por janelas menores sem colapsar as grandes)
    raw={k:1.0/math.sqrt(k) for k in ws}
    s=sum(raw.values()) or 1e-9
    return {k:v/s for k,v in raw.items()}

def _post_e1_fibo_mix(tail: List[int], windows: List[int]) -> Dict[int,float]:
    mix={c:0.0 for c in (1,2,3,4)}
    wts=_weights_for_fibo(windows)
    for k, wk in wts.items():
        pk=_post_freq(tail,k)
        for c in (1,2,3,4):
            mix[c]+=wk*pk[c]
    return _norm(mix)

def _post_e2_short(tail):  return _post_freq(tail, 60)
def _post_e3_long(tail):   return _post_freq(tail, 300)
def _post_e4_llm(tail):    return {1:0.25,2:0.25,3:0.25,4:0.25}  # placeholder offline

def _hedge(p1,p2,p3,p4, w=(0.45,0.25,0.20,0.10)):
    # leve aumento do peso do E1 (Fibo-Mix) para refletir o pedido de “vários fibo”
    cands=(1,2,3,4)
    out={c: w[0]*p1.get(c,0)+w[1]*p2.get(c,0)+w[2]*p3.get(c,0)+w[3]*p4.get(c,0) for c in cands}
    return _norm(out)

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

def _top_k(post: Dict[int,float], k:int=3)->List[Tuple[int,float]]:
    return sorted(post.items(), key=lambda kv: kv[1], reverse=True)[:k]

def _choose_number()->Tuple[int,float,int,Dict[int,float],float,str,List[Tuple[int,float]],Dict[int,float]]:
    tail=_timeline_tail(400)
    p1=_post_e1_fibo_mix(tail, FIBO_WINDOWS)  # << FIBO MIX dinâmico
    p2=_post_e2_short(tail)
    p3=_post_e3_long(tail)
    p4=_post_e4_llm(tail)
    base=_hedge(p1,p2,p3,p4)
    best, post, reason = _runnerup_ls2(base, loss_streak=0)
    post=_conf_floor(post, 0.30, 0.95)
    best=max(post,key=post.get); conf=float(post[best])
    r=sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    gap=(r[0][1]-r[1][1]) if len(r)>=2 else r[0][1]
    top3=_top_k(post, 3)
    return best, conf, _timeline_size(), post, gap, reason, top3, p1  # p1 para snapshot E1

def _ngram_snapshot(suggested:int)->str:
    # agora mostrando snapshot do E1 (Fibo-Mix)
    tail=_timeline_tail(400)
    post=_post_e1_fibo_mix(tail, FIBO_WINDOWS)
    pct=lambda x:f"{x*100:.1f}%"
    p1,p2,p3,p4 = pct(post[1]), pct(post[2]), pct(post[3]), pct(post[4])
    wtxt=",".join(str(k) for k in FIBO_WINDOWS)
    conf=pct(post.get(int(suggested),0.0))
    return (f"📈 Amostra: {_timeline_size()} • Conf(E1): {conf}\n"
            f"🌀 E1(FiboMix {wtxt}): 1 {p1} | 2 {p2} | 3 {p3} | 4 {p4}")

# ------------------------------------------------------
# Telegram helpers (send + delete)
# ------------------------------------------------------
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

# ------------------------------------------------------
# Parser do canal-fonte
# ------------------------------------------------------
RX_ENTRADA = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
RX_ANALISE = re.compile(r"\bANALISANDO\b", re.I)

# compat (não fecha mais por estes, fechamento agora é estrito)
RX_FECHA   = re.compile(r"APOSTA\s+ENCERRADA", re.I)
RX_GREEN   = re.compile(r"GREEN|✅", re.I)
RX_RED     = re.compile(r"RED|❌", re.I)

RX_SEQ     = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
RX_NUMS    = re.compile(r"[1-4]")
RX_AFTER   = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)
RX_PAREN   = re.compile(r"\(([^\)]*)\)\s*$")

# >>> FECHAMENTO ESTRITO: "GREEN/RED (...) fecha"
RX_CLOSE_STRICT = re.compile(
    r"(?:✅\s*)?\b(GREEN|RED)\b.*?\(([1-4])\).*?(?:✅\s*)?fecha\b",
    re.I | re.S
)

def _parse_seq_list(text:str)->List[int]:
    m=RX_SEQ.search(text or "")
    if not m: return []
    return [int(x) for x in RX_NUMS.findall(m.group(1))]

def _parse_seq_pair(text:str, need:int=2)->List[int]:
    arr=_parse_seq_list(text)
    return arr[:need]

def _parse_after(text:str)->Optional[int]:
    m=RX_AFTER.search(text or "")
    if not m: return None
    try: return int(m.group(1))
    except: return None

def _parse_paren_pair(text:str, need:int=2)->List[int]:
    m=RX_PAREN.search(text or "")
    if not m: return []
    nums=[int(x) for x in re.findall(r"[1-4]", m.group(1))]
    return nums[:need]

# ------------------------------------------------------
# Rotas básicas
# ------------------------------------------------------
@app.get("/")
async def root():
    return {"ok": True, "service": "GuardiAo Auto Bot", "time": datetime.datetime.utcnow().isoformat()+"Z"}

@app.get("/health")
async def health():
    return {"ok": True, "db_exists": os.path.exists(DB_PATH), "db_path": DB_PATH}

@app.get("/debug_cfg")
async def debug_cfg():
    return {
        "MAX_GALE": MAX_GALE,
        "OBS_TIMEOUT_SEC": OBS_TIMEOUT_SEC,
        "DEDUP_WINDOW_SEC": DEDUP_WINDOW_SEC,
        "FIBO_WINDOWS": FIBO_WINDOWS,
        "SHOW_TOP3": SHOW_TOP3
    }

# ------------------------------------------------------
# Webhook principal
# ------------------------------------------------------
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

    # filtra fonte se configurado
    if SOURCE_CHANNEL and chat_id and chat_id != SOURCE_CHANNEL:
        if SHOW_DEBUG:
            await tg_send(TARGET_CHANNEL, f"DEBUG: Ignorando chat {chat_id}. Fonte esperada: {SOURCE_CHANNEL}")
        return {"ok": True, "skipped": "wrong_source"}

    # 1) ANALISANDO: apenas alimenta memória (com dedupe)
    if RX_ANALISE.search(text):
        if _seen_recent("analise", _dedupe_key(text)):
            return {"ok": True, "skipped": "analise_dupe"}
        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)
        return {"ok": True, "analise_seq": len(seq)}

    # 2) FECHAMENTO ESTRITO: somente GREEN/RED (...) fecha
    m_close = RX_CLOSE_STRICT.search(text)
    if m_close:
        if _seen_recent("fechamento", _dedupe_key(text)):
            return {"ok": True, "skipped": "fechamento_dupe"}

        pend=_pending_get()
        if pend:
            suggested=int(pend["suggested"] or 0)

            # observado/hit entre parênteses
            hit = None
            try:
                hit = int(m_close.group(2))
            except:
                pass

            # memória extra: quaisquer números entre parênteses -> timeline
            extra_tail = _parse_paren_pair(text, need=2)
            if extra_tail:
                _append_seq(extra_tail)

            # atualizar 'seen' com o hit
            if hit is not None:
                _pending_seen_append([hit], need=min(2, MAX_GALE+1))

            pend=_pending_get()
            seen = [s for s in (pend["seen"] or "").split("-") if s]

            # outcome do próprio texto
            raw_outcome = (m_close.group(1) or "").strip().upper()
            outcome = "GREEN" if raw_outcome == "GREEN" else "LOSS"

            # estágio inferido
            stage_lbl = "G0" if (seen and seen[0].isdigit() and int(seen[0])==suggested) else "G1"

            final_seen="-".join(seen[:min(2, MAX_GALE+1)]) if seen else (str(hit) if hit is not None else "X")
            msg_txt=_pending_close(final_seen, outcome, stage_lbl, suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)
            return {"ok": True, "closed": outcome, "seen": final_seen}

        return {"ok": True, "noted_close": True}

    # 3) ENTRADA CONFIRMADA (com dedupe + “Analisando...” auto-delete)
    if RX_ENTRADA.search(text):
        if _seen_recent("entrada", _dedupe_key(text)):
            if SHOW_DEBUG:
                await tg_send(TARGET_CHANNEL, "DEBUG: entrada duplicada ignorada (conteúdo repetido).")
            return {"ok": True, "skipped": "entrada_dupe"}

        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)               # memória
        after = _parse_after(text)             # usado apenas para exibir "após X"

        # fecha pendência anterior esquecida (marca X)
        pend=_pending_get()
        if pend:
            seen=[s for s in (pend["seen"] or "").split("-") if s]
            while len(seen)<min(2,MAX_GALE+1): seen.append("X")
            final_seen="-".join(seen[:min(2,MAX_GALE+1)])
            suggested=int(pend["suggested"] or 0)
            outcome="LOSS"; stage_lbl="G1"
            if len(seen)>=1 and seen[0].isdigit() and int(seen[0])==suggested:
                outcome="GREEN"; stage_lbl="G0"
            elif len(seen)>=2 and seen[1].isdigit() and int(seen[1])==suggested and MAX_GALE>=1:
                outcome="GREEN"; stage_lbl="G1"
            msg_txt=_pending_close(final_seen, outcome, stage_lbl, suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)

        # “Analisando...” (apaga depois)
        analyzing_id = await tg_send_return(TARGET_CHANNEL, "⏳ Analisando padrão, aguarde...")

        # escolhe nova sugestão (com estudo TOP-3)
        best, conf, samples, post, gap, reason, top3, e1mix = _choose_number()
        opened=_pending_open(best)
        if opened:
            aft_txt = f" após {after}" if after else ""
            def pct(x): return f"{x*100:.2f}%"
            top3_txt = " | ".join(f"{n} ({pct(p)})" for n,p in top3)
            wtxt=",".join(str(k) for k in FIBO_WINDOWS)
            txt=(f"🤖 <b>IA SUGERE</b> — <b>{best}</b>\n"
                 f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
                 f"📊 <b>Conf:</b> {conf*100:.2f}% | <b>Amostra≈</b>{samples} | <b>gap≈</b>{gap*100:.1f}pp\n"
                 f"🧠 <b>Modo:</b> IA (E1=FiboMix[{wtxt}] + Short/Long)\n")
            if SHOW_TOP3 and top3:
                txt += f"🔬 <b>Estudo 3 números:</b> {top3_txt}\n"
            txt += _ngram_snapshot(best)
            await tg_send(TARGET_CHANNEL, txt)

            if analyzing_id is not None:
                await tg_delete(TARGET_CHANNEL, analyzing_id)

            return {"ok": True, "entry_opened": True, "best": best, "conf": conf}
        else:
            if analyzing_id is not None:
                await tg_delete(TARGET_CHANNEL, analyzing_id)
            if SHOW_DEBUG:
                await tg_send(TARGET_CHANNEL, "DEBUG: pending já aberto; entrada ignorada.")
            return {"ok": True, "skipped": "pending_open"}

    # Não reconhecido
    if SHOW_DEBUG:
        await tg_send(TARGET_CHANNEL, "DEBUG: Mensagem não reconhecida como ENTRADA/FECHAMENTO/ANALISANDO.")
    return {"ok": True, "skipped": "unmatched"}