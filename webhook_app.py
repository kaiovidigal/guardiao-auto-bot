#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GuardiAo Auto Bot — webhook_app.py
v8.0 (G0, parser canal-fonte, IA hierárquica + 40 cálculos, calibração on-line,
      dedupe, “Analisando...” auto-delete, fechamento correto G0, DB SQLite)

ENV obrigatórias:
- TG_BOT_TOKEN
- WEBHOOK_TOKEN
- SOURCE_CHANNEL           ex: -1002810508717
- TARGET_CHANNEL           ex: -1003052132833

ENV opcionais:
- SHOW_DEBUG          (default False)
- OBS_TIMEOUT_SEC     (default 420)
- DEDUP_WINDOW_SEC    (default 40)
- CALIB_ENABLED       (default True)
- CALIB_LR            (default 0.08)   # taxa de aprendizado
- CALIB_MAX_SHIFT     (default 0.15)   # limite por rodada
- CALIB_MIN_DELTA     (default 0.02)   # ignora ajustes < 2pp
- G0_MODE             (default True)   # força fechamento G0
Start:
  uvicorn webhook_app:app --host 0.0.0.0 --port $PORT
"""

import os, re, json, time, math, sqlite3, datetime, hashlib
from typing import List, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException

# ========= ENV =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "").strip()

SHOW_DEBUG       = os.getenv("SHOW_DEBUG", "False").strip().lower() == "true"
OBS_TIMEOUT_SEC  = int(os.getenv("OBS_TIMEOUT_SEC", "420"))
DEDUP_WINDOW_SEC = int(os.getenv("DEDUP_WINDOW_SEC", "40"))
CALIB_ENABLED    = os.getenv("CALIB_ENABLED", "True").strip().lower() == "true"
CALIB_LR         = float(os.getenv("CALIB_LR", "0.08"))
CALIB_MAX_SHIFT  = float(os.getenv("CALIB_MAX_SHIFT", "0.15"))
CALIB_MIN_DELTA  = float(os.getenv("CALIB_MIN_DELTA", "0.02"))
G0_MODE          = os.getenv("G0_MODE", "True").strip().lower() == "true"

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN or not TARGET_CHANNEL:
    raise RuntimeError("Faltam ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN, TARGET_CHANNEL.")
TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
DB_PATH = "/opt/render/project/src/main.sqlite"

app = FastAPI(title="GuardiAo Auto Bot (webhook)", version="8.0")

# ========= DB =========
def _con():
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=20)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA busy_timeout=12000;")
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
    # pesos adaptativos (E1/E2/E3/E4 + P40)
    cur.execute("""CREATE TABLE IF NOT EXISTS expert_w(
        id INTEGER PRIMARY KEY CHECK(id=1),
        w_e1 REAL, w_e2 REAL, w_e3 REAL, w_e4 REAL, w_p40 REAL
    )""")
    row = con.execute("SELECT 1 FROM score WHERE id=1").fetchone()
    if not row:
        con.execute("INSERT INTO score(id,green,loss) VALUES(1,0,0)")
    row = con.execute("SELECT 1 FROM expert_w WHERE id=1").fetchone()
    if not row:
        # soma 1.0 -> normalizado depois
        con.execute("INSERT INTO expert_w(id,w_e1,w_e2,w_e3,w_e4,w_p40) VALUES(1,0.35,0.20,0.20,0.05,0.20)")
    con.commit(); con.close()

db_init()

# ========= Utils =========
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
    con = _con()
    now = int(time.time())
    con.executemany("INSERT INTO timeline(created_at,number) VALUES(?,?)",[(now,int(x)) for x in seq])
    con.commit(); con.close()

def _timeline_size()->int:
    con=_con(); row=con.execute("SELECT COUNT(*) c FROM timeline").fetchone(); con.close()
    return int(row["c"] or 0)

def _score_add(outcome:str):
    con = _con()
    row = con.execute("SELECT green,loss FROM score WHERE id=1").fetchone()
    g,l = (int(row["green"]), int(row["loss"])) if row else (0,0)
    if outcome.upper()=="GREEN": g+=1
    elif outcome.upper()=="LOSS": l+=1
    con.execute("INSERT OR REPLACE INTO score(id,green,loss) VALUES(1,?,?)",(g,l))
    con.commit(); con.close()

def _score_text()->str:
    con = _con(); row = con.execute("SELECT green,loss FROM score WHERE id=1").fetchone(); con.close()
    if not row: return "0 GREEN × 0 LOSS — 0.0%"
    g,l = int(row["green"]), int(row["loss"]); tot = g+l
    acc = (g/tot*100.0) if tot>0 else 0.0
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
    con = _con(); con.execute("UPDATE pending SET stage=? WHERE open=1",(int(stage),)); con.commit(); con.close()

def _pending_seen_append(nums: List[int], need:int=1):
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
    # alimenta timeline com observado G0 (já que G0-mode)
    obs = [int(x) for x in final_seen.split("-") if x.isdigit()]
    _append_seq(obs)
    our = suggested if outcome.upper()=="GREEN" else "X"
    snap = _ngram_snapshot(suggested)
    msg = (f"{'🟢' if outcome.upper()=='GREEN' else '🔴'} <b>{outcome.upper()}</b> — finalizado "
           f"(<b>{stage_lbl}</b>, nosso={our}, observados={final_seen}).\n"
           f"📊 Geral: {_score_text()}\n\n{snap}")
    return msg

# ========= DEDUPE =========
def _dedupe_key(text: str) -> str:
    base = re.sub(r"\s+", " ", (text or "")).strip().lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _seen_recent(kind: str, dkey: str) -> bool:
    now = int(time.time())
    con = _con()
    row = con.execute("SELECT ts FROM dedupe WHERE kind=? AND dkey=?", (kind, dkey)).fetchone()
    if row and now - int(row["ts"]) <= DEDUP_WINDOW_SEC:
        con.close(); return True
    con.execute("INSERT OR REPLACE INTO dedupe(kind, dkey, ts) VALUES (?,?,?)", (kind, dkey, now))
    con.commit(); con.close()
    return False

# ========= IA base (E1..E4) =========
def _norm(d: Dict[int,float])->Dict[int,float]:
    s=sum(d.values()) or 1e-9
    return {k:(v/s) for k,v in d.items()}

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
def _post_e4_llm(tail):    return {1:0.25,2:0.25,3:0.25,4:0.25}  # placeholder offline

# ========= 40 cálculos de probabilidade (multi-prob) =========
def _exp_decay_freq(tail: List[int], decay: float)->Dict[int,float]:
    # peso t mais recente = 1, anterior = decay, etc.
    w_sum = {1:0.0,2:0.0,3:0.0,4:0.0}
    w_tot = 0.0
    d = 1.0
    for x in reversed(tail):
        w_sum[x] += d
        w_tot += d
        d *= decay
    if w_tot<=0: return {1:0.25,2:0.25,3:0.25,4:0.25}
    return _norm({c:w_sum[c]/w_tot for c in (1,2,3,4)})

def _conditional_after_last(tail: List[int], span:int=200)->Dict[int,float]:
    if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
    last = tail[-1]
    cnt={1:1e-9,2:1e-9,3:1e-9,4:1e-9}
    T = tail[-span:] if len(tail)>span else tail
    for i in range(len(T)-1):
        if T[i]==last:
            cnt[T[i+1]] += 1.0
    return _norm(cnt)

def _pair_conditional(tail: List[int], span:int=240)->Dict[int,float]:
    if len(tail)<2: return {1:0.25,2:0.25,3:0.25,4:0.25}
    a,b = tail[-2], tail[-1]
    cnt={1:1e-9,2:1e-9,3:1e-9,4:1e-9}
    T = tail[-span:] if len(tail)>span else tail
    for i in range(len(T)-2):
        if T[i]==a and T[i+1]==b:
            cnt[T[i+2]] += 1.0
    return _norm(cnt)

def _triple_conditional(tail: List[int], span:int=280)->Dict[int,float]:
    if len(tail)<3: return {1:0.25,2:0.25,3:0.25,4:0.25}
    a,b,c = tail[-3], tail[-2], tail[-1]
    cnt={1:1e-9,2:1e-9,3:1e-9,4:1e-9}
    T = tail[-span:] if len(tail)>span else tail
    for i in range(len(T)-3):
        if T[i]==a and T[i+1]==b and T[i+2]==c:
            cnt[T[i+3]] += 1.0
    return _norm(cnt)

def _momentum(post:Dict[int,float])->Dict[int,float]:
    return dict(post)  # identidade (o próprio favorece dominância)

def _contrarian(post:Dict[int,float])->Dict[int,float]:
    inv = {c: max(1e-9, 1.0-post.get(c,0.0)) for c in (1,2,3,4)}
    return _norm(inv)

def _entropy(p:Dict[int,float])->float:
    eps=1e-12
    return -sum(max(eps,p.get(c,0.0))*math.log(max(eps,p.get(c,0.0)),4) for c in (1,2,3,4))

def _entropy_blend(p:Dict[int,float])->Dict[int,float]:
    # quanto menor a entropia, mais concentrado — aplicamos leve suavização
    H=_entropy(p)
    alpha = 0.15*(1.0 - H)  # 0..0.15
    uni={1:0.25,2:0.25,3:0.25,4:0.25}
    return _norm({c:(1-alpha)*p.get(c,0.0)+alpha*uni[c] for c in (1,2,3,4)})

def _kl(p:Dict[int,float], q:Dict[int,float])->float:
    eps=1e-12; s=0.0
    for c in (1,2,3,4):
        pc=max(eps,p.get(c,0.0)); qc=max(eps,q.get(c,0.0))
        s += pc*math.log(pc/qc)
    return s

def _drift_adjust(short:Dict[int,float], long:Dict[int,float])->Dict[int,float]:
    # se curto e longo divergem, puxa 10% pro curto
    if _kl(short,long) > 0.22:
        return _norm({c:0.9*long.get(c,0.0)+0.1*short.get(c,0.0) for c in (1,2,3,4)})
    return long

def _repeat3_support(tail:List[int])->Tuple[Optional[int], float, int]:
    if len(tail)<4: return (None, 0.0, 0)
    a,b,c = tail[-3], tail[-2], tail[-1]
    cnt={1:0,2:0,3:0,4:0}; tot=0
    for i in range(len(tail)-3):
        if tail[i]==a and tail[i+1]==b and tail[i+2]==c:
            cnt[tail[i+3]] += 1; tot += 1
    if tot==0: return (None, 0.0, 0)
    best = max(cnt, key=cnt.get); conf = cnt[best]/tot
    return (best, conf, tot)

def _ngram2(tail:List[int])->Dict[int,float]:
    return _pair_conditional(tail, span=400)

def _ngram3(tail:List[int])->Dict[int,float]:
    return _triple_conditional(tail, span=400)

def _freq_windows(tail:List[int], wins:List[int])->List[Dict[int,float]]:
    return [_post_freq(tail, k) for k in wins]

def _multi_prob40(tail: List[int]) -> Dict[int,float]:
    # 1) Frequências janeladas (10)
    wins = [5,8,13,21,34,55,89,144,60,300]
    Fs = _freq_windows(tail, wins)

    # 2) Frequências com decaimento exponencial (5)
    Ds = [_exp_decay_freq(tail, d) for d in (0.90,0.93,0.95,0.97,0.99)]

    # 3) Condicionais (10): last, pair, triple com spans variados
    C1 = [_conditional_after_last(tail, s) for s in (80,120,160,200,260)]
    C2 = [_pair_conditional(tail, s)        for s in (120,160,200,240,300)]
    C3 = [_triple_conditional(tail, s)      for s in (160,220,280,320,380)]

    # 4) N-gram proxies (4)
    NG = [_ngram2(tail), _ngram3(tail),
          _pair_conditional(tail, 180), _triple_conditional(tail, 220)]

    # 5) Momentum/Reversão (2) sobre base curto (k=21)
    base21 = _post_freq(tail, 21)
    MOM = [_momentum(base21), _contrarian(base21)]

    # 6) Entropia/Uniformização (1) sobre base longa (k=144)
    base144 = _post_freq(tail, 144)
    ENT = [_entropy_blend(base144)]

    # 7) Drift curto vs longo (1)
    drift = [_drift_adjust(_post_freq(tail,60), _post_freq(tail,300))]

    # Soma (10+5+15+4+2+1+1 = 38). Para fechar 40:
    # 8) Força de repetição-3 (suporte alto força 2%) (1)
    best3, conf3, tot3 = _repeat3_support(tail)
    REP3 = [{c:0.98*base21.get(c,0.0)+(0.02 if c==best3 and conf3>=0.78 and tot3>=12 else 0.0) for c in (1,2,3,4)}] if best3 else [base21]

    # 9) Priori global (1)
    prior = _post_freq(tail, max(1, len(tail)))  # tudo
    PRIOR = [prior]

    # Junta tudo
    buckets = Fs + Ds + C1 + C2 + C3 + NG + MOM + ENT + drift + REP3 + PRIOR  # 40
    acc={1:0.0,2:0.0,3:0.0,4:0.0}
    w = 1.0/len(buckets)
    for B in buckets:
        for c in (1,2,3,4):
            acc[c]+= w*B.get(c,0.0)
    return _norm(acc)

# ========= Hedge + RunnerUp + Floor =========
def _get_w():
    con=_con()
    row=con.execute("SELECT w_e1,w_e2,w_e3,w_e4,w_p40 FROM expert_w WHERE id=1").fetchone()
    con.close()
    if not row: return (0.35,0.20,0.20,0.05,0.20)
    w = [float(row[k]) for k in ("w_e1","w_e2","w_e3","w_e4","w_p40")]
    s = sum(w) or 1e-9
    return tuple(x/s for x in w)

def _set_w(w):
    w = list(w); s=sum(w) or 1e-9
    w = [x/s for x in w]
    con=_con()
    con.execute("INSERT OR REPLACE INTO expert_w(id,w_e1,w_e2,w_e3,w_e4,w_p40) VALUES(1,?,?,?,?,?)", tuple([1]+w))
    con.commit(); con.close()

def _hedge5(p1,p2,p3,p4,p40):
    we1,we2,we3,we4,wp40 = _get_w()
    out={c: we1*p1.get(c,0)+we2*p2.get(c,0)+we3*p3.get(c,0)+we4*p4.get(c,0)+wp40*p40.get(c,0) for c in (1,2,3,4)}
    return _norm(out), (we1,we2,we3,we4,wp40)

def _runnerup_ls2(post:Dict[int,float], loss_streak:int=0)->Tuple[int,Dict[int,float],str]:
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

# ========= Calibração on-line =========
def _calib_update(true_c:int, p_e1, p_e2, p_e3, p_e4, p_p40):
    if not CALIB_ENABLED: return
    # perda = 1 - p(true)
    l = lambda p: 1.0 - float(p.get(true_c,0.0))
    # gradiente multiplicativo (hedge) com LR e limites
    we = list(_get_w())
    losses = [l(p_e1), l(p_e2), l(p_e3), l(p_e4), l(p_p40)]
    # normaliza perda e aplica ajuste proporcional
    avg = sum(losses)/len(losses)
    deltas=[]
    for i,loss in enumerate(losses):
        # se a fonte foi melhor (perdeu menos), reforça; se pior, reduz
        delta = CALIB_LR * (avg - loss)  # positivo reforça, negativo reduz
        deltas.append(delta)
    # aplica limites e ignora micro-ajustes
    new=[]
    for w,dx in zip(we, deltas):
        if abs(dx) < CALIB_MIN_DELTA: dx = 0.0
        dx = max(-CALIB_MAX_SHIFT, min(CALIB_MAX_SHIFT, dx))
        new.append(max(0.01, w + dx))
    _set_w(new)

# ========= Escolha do número =========
def _choose_number()->Tuple[int,float,int,Dict[int,float],float,str,Tuple[Dict,...]]:
    tail=_timeline_tail(400)
    p1=_post_e1_ngram(tail)
    p2=_post_e2_short(tail)
    p3=_post_e3_long(tail)
    p4=_post_e4_llm(tail)
    p40=_multi_prob40(tail)
    base, w = _hedge5(p1,p2,p3,p4,p40)
    best, post, reason = _runnerup_ls2(base, loss_streak=0)
    post=_conf_floor(post, 0.30, 0.95)
    best=max(post,key=post.get); conf=float(post[best])
    r=sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    gap=(r[0][1]-r[1][1]) if len(r)>=2 else r[0][1]
    return best, conf, _timeline_size(), post, gap, reason,(p1,p2,p3,p4,p40)

def _ngram_snapshot(suggested:int)->str:
    tail=_timeline_tail(400)
    post=_post_e1_ngram(tail)
    pct=lambda x:f"{x*100:.1f}%"
    p1,p2,p3,p4 = pct(post[1]), pct(post[2]), pct(post[3]), pct(post[4])
    conf=pct(post.get(int(suggested),0.0))
    return (f"📈 Amostra: {_timeline_size()} • Conf: {conf}\n"
            f"🔎 E1(n-gram proxy): 1 {p1} | 2 {p2} | 3 {p3} | 4 {p4}")

# ========= Telegram =========
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

# ========= Parser do canal-fonte =========
RX_ENTRADA = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
RX_ANALISE = re.compile(r"\bANALISANDO\b", re.I)
RX_FECHA   = re.compile(r"APOSTA\s+ENCERRADA", re.I)

RX_SEQ     = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
RX_NUMS    = re.compile(r"[1-4]")
RX_AFTER   = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)
RX_GREEN   = re.compile(r"GREEN|✅", re.I)
RX_RED     = re.compile(r"RED|❌", re.I)
RX_PAREN   = re.compile(r"\(([^\)]*)\)\s*$")

def _parse_seq_list(text:str)->List[int]:
    m=RX_SEQ.search(text or "")
    if not m: return []
    return [int(x) for x in RX_NUMS.findall(m.group(1))]

def _parse_after(text:str)->Optional[int]:
    m=RX_AFTER.search(text or "")
    if not m: return None
    try: return int(m.group(1))
    except: return None

def _parse_paren_single(text:str)->Optional[int]:
    # G0: GREEN (x) ou RED (x)
    m=RX_PAREN.search(text or "")
    if not m: return None
    nums=[int(x) for x in re.findall(r"[1-4]", m.group(1))]
    return nums[0] if nums else None

# ========= Rotas =========
@app.get("/")
async def root():
    return {"ok": True, "service": "GuardiAo Auto Bot", "time": datetime.datetime.utcnow().isoformat()+"Z"}

@app.get("/health")
async def health():
    return {"ok": True, "db_exists": os.path.exists(DB_PATH), "db_path": DB_PATH}

@app.get("/debug_cfg")
async def debug_cfg():
    we1,we2,we3,we4,wp40 = _get_w()
    return {
        "OBS_TIMEOUT_SEC": OBS_TIMEOUT_SEC,
        "DEDUP_WINDOW_SEC": DEDUP_WINDOW_SEC,
        "CALIB_ENABLED": CALIB_ENABLED,
        "CALIB_LR": CALIB_LR,
        "CALIB_MAX_SHIFT": CALIB_MAX_SHIFT,
        "CALIB_MIN_DELTA": CALIB_MIN_DELTA,
        "G0_MODE": G0_MODE,
        "w_e1": we1, "w_e2": we2, "w_e3": we3, "w_e4": we4, "w_p40": wp40
    }

# ========= Webhook principal =========
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

    # filtra fonte
    if SOURCE_CHANNEL and chat_id and chat_id != SOURCE_CHANNEL:
        if SHOW_DEBUG:
            await tg_send(TARGET_CHANNEL, f"DEBUG: Ignorando chat {chat_id}. Fonte esperada: {SOURCE_CHANNEL}")
        return {"ok": True, "skipped": "wrong_source"}

    # ANALISANDO: só memória
    if RX_ANALISE.search(text):
        if _seen_recent("analise", _dedupe_key(text)):
            return {"ok": True, "skipped": "analise_dupe"}
        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)
        return {"ok": True, "analise_seq": len(seq)}

    # APOSTA ENCERRADA / GREEN / RED (fechamento G0)
    if RX_FECHA.search(text) or RX_GREEN.search(text) or RX_RED.search(text):
        if _seen_recent("fechamento", _dedupe_key(text)):
            return {"ok": True, "skipped": "fechamento_dupe"}
        pend=_pending_get()
        if pend:
            suggested=int(pend["suggested"] or 0)

            # G0: pega somente o número dentro de parênteses
            g0 = _parse_paren_single(text)
            if g0 is not None:
                _pending_seen_append([g0], need=1)

            # Reavalia pendência
            pend=_pending_get()
            seen = [s for s in (pend["seen"] or "").split("-") if s]
            outcome="LOSS"; stage_lbl="G0"
            if len(seen)>=1 and seen[0].isdigit() and int(seen[0])==suggested:
                outcome="GREEN"
            # fecha sempre no G0
            final_seen="-".join(seen[:1]) if seen else "X"

            # Calibração (recalcula fontes no fechamento)
            # Obs: recalcular aqui (tail ~ estável logo após a jogada) é suficiente
            if CALIB_ENABLED and seen and seen[0].isdigit():
                true_c = int(seen[0])
                tail_now = _timeline_tail(400)
                _calib_update(true_c,
                              _post_e1_ngram(tail_now),
                              _post_e2_short(tail_now),
                              _post_e3_long(tail_now),
                              _post_e4_llm(tail_now),
                              _multi_prob40(tail_now))

            msg_txt=_pending_close(final_seen, outcome, stage_lbl, suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)
            return {"ok": True, "closed": outcome, "seen": final_seen}
        return {"ok": True, "noted_close": True}

    # ENTRADA CONFIRMADA
    if RX_ENTRADA.search(text):
        if _seen_recent("entrada", _dedupe_key(text)):
            if SHOW_DEBUG:
                await tg_send(TARGET_CHANNEL, "DEBUG: entrada duplicada ignorada (conteúdo repetido).")
            return {"ok": True, "skipped": "entrada_dupe"}

        # memória de cauda
        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)
        after = _parse_after(text)  # apenas para frase "após X"

        # fecha pendência esquecida (G0-mode) com X
        pend=_pending_get()
        if pend:
            seen=[s for s in (pend["seen"] or "").split("-") if s]
            while len(seen)<1: seen.append("X")
            final_seen="-".join(seen[:1])
            suggested=int(pend["suggested"] or 0)
            outcome="LOSS"
            if len(seen)>=1 and seen[0].isdigit() and int(seen[0])==suggested:
                outcome="GREEN"
            msg_txt=_pending_close(final_seen, outcome, "G0", suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)

        analyzing_id = await tg_send_return(TARGET_CHANNEL, "⏳ Analisando padrão, aguarde...")

        # escolhe nova entrada (já com p40)
        best, conf, samples, post, gap, reason, _experts = _choose_number()
        opened=_pending_open(best)
        if opened:
            aft_txt = f" após {after}" if after else ""
            txt=(f"🤖 <b>IA SUGERE</b> — <b>{best}</b>\n"
                 f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
                 f"📊 <b>Conf:</b> {conf*100:.2f}% | <b>Amostra≈</b>{samples} | <b>gap≈</b>{gap*100:.1f}pp\n"
                 f"🧠 <b>Modo:</b> {reason}\n"
                 f"{_ngram_snapshot(best)}")
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

    if SHOW_DEBUG:
        await tg_send(TARGET_CHANNEL, "DEBUG: Mensagem não reconhecida como ENTRADA/FECHAMENTO/ANALISANDO.")
    return {"ok": True, "skipped": "unmatched"}