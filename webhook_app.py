#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GuardiAo Auto Bot — v6.0 (Blueprint ML)
- G1-only (fecha em G0/G1), parser canal-fonte, dedupe por conteúdo
- "Entrar após X" incluso no texto de saída quando houver
- Timeline+SQLite com treino on-line leve (40 features probabilísticas)
- Ensemble linear com softmax; retreino a cada TRAIN_INTERVAL rodadas

Start (Render): uvicorn webhook_app:app --host 0.0.0.0 --port $PORT
"""

import os, re, time, math, sqlite3, hashlib, json, datetime, random
from typing import List, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException

# =========================
# ENV & Constantes
# =========================
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "").strip()

SHOW_DEBUG       = os.getenv("SHOW_DEBUG", "False").strip().lower() == "true"
MAX_GALE         = int(os.getenv("MAX_GALE", "1"))               # G0/G1
OBS_TIMEOUT_SEC  = int(os.getenv("OBS_TIMEOUT_SEC", "420"))
DEDUP_WINDOW_SEC = int(os.getenv("DEDUP_WINDOW_SEC", "40"))

TRAIN_INTERVAL    = int(os.getenv("TRAIN_INTERVAL", "1000"))
ML_LEARNING_RATE  = float(os.getenv("ML_LEARNING_RATE", "0.05"))
MIN_ENTER_CONF    = float(os.getenv("MIN_ENTER_CONF", "0.30"))

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN or not TARGET_CHANNEL:
    raise RuntimeError("Faltam ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN, TARGET_CHANNEL.")
TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

DB_PATH = "/opt/render/project/src/main.sqlite"

# =========================
# App
# =========================
app = FastAPI(title="GuardiAo Auto Bot (webhook) — v6.0", version="6.0")

# =========================
# DB Helpers
# =========================
def _con():
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=20)
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
        after_hint INTEGER,
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
    # Pesos do ensemble ML (40 features x 4 classes)
    cur.execute("""CREATE TABLE IF NOT EXISTS ml_weights(
        fname TEXT PRIMARY KEY,
        w1 REAL NOT NULL, w2 REAL NOT NULL, w3 REAL NOT NULL, w4 REAL NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS ml_meta(
        id INTEGER PRIMARY KEY CHECK(id=1),
        last_train_at INTEGER DEFAULT 0,
        total_samples INTEGER DEFAULT 0
    )""")
    if not con.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        con.execute("INSERT INTO score(id,green,loss) VALUES(1,0,0)")
    if not con.execute("SELECT 1 FROM ml_meta WHERE id=1").fetchone():
        con.execute("INSERT INTO ml_meta(id,last_train_at,total_samples) VALUES(1,0,0)")
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
    con = _con()
    rows = con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?",(n,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def _append_seq(seq: List[int]):
    if not seq: return
    con = _con()
    now = int(time.time())
    con.executemany("INSERT INTO timeline(created_at,number) VALUES(?,?)",
                    [(now,int(x)) for x in seq])
    # atualiza total samples
    con.execute("UPDATE ml_meta SET total_samples = total_samples + ?", (len(seq),))
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
    g,l = int(row["green"]), int(row["loss"])
    tot = g+l; acc = (g/tot*100.0) if tot>0 else 0.0
    return f"{g} GREEN × {l} LOSS — {acc:.1f}%"

def _pending_get()->Optional[sqlite3.Row]:
    con = _con()
    row = con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone()
    con.close()
    return row

def _pending_open(suggested:int, after_hint:Optional[int]):
    if _pending_get(): return False
    con = _con()
    now = int(time.time())
    con.execute("""INSERT INTO pending(created_at,opened_at,suggested,stage,seen,after_hint,open)
                   VALUES(?,?,?,?,?,?,1)""",
                (now, now, int(suggested), 0, "", int(after_hint) if after_hint else None))
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

# =========================
# Dedupe por conteúdo
# =========================
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

# =========================
# Telegram
# =========================
async def tg_send(chat_id: str, text: str, parse="HTML"):
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            await cli.post(f"{TELEGRAM_API}/sendMessage",
                           json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                 "disable_web_page_preview": True})
    except Exception:
        pass

# =========================
# Parser do canal-fonte
# =========================
RX_ENTRADA = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
RX_ANALISE = re.compile(r"\bANALISANDO\b", re.I)
RX_FECHA   = re.compile(r"APOSTA\s+ENCERRADA", re.I)

RX_SEQ     = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
RX_NUMS    = re.compile(r"[1-4]")
RX_AFTER   = re.compile(r"Entrar\s+ap[oó]s\s+o\s+([1-4])", re.I)  # "Entrar após o 1"
RX_AFTER2  = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)           # fallback

RX_GREEN   = re.compile(r"GREEN|✅", re.I)
RX_RED     = re.compile(r"RED|❌", re.I)
RX_PAREN   = re.compile(r"\(([^\)]*)\)\s*$")

def _parse_seq(text:str)->List[int]:
    m=RX_SEQ.search(text or "")
    if not m: return []
    return [int(x) for x in RX_NUMS.findall(m.group(1))]

def _parse_after(text:str)->Optional[int]:
    for rx in (RX_AFTER, RX_AFTER2):
        m=rx.search(text or "")
        if m:
            try: return int(m.group(1))
            except: pass
    return None

def _parse_obs_final(text:str, need:int=2)->List[int]:
    m=RX_PAREN.search(text or "")
    if not m: return []
    nums=[int(x) for x in re.findall(r"[1-4]", m.group(1))]
    return nums[:need]

# =========================
# Prob/Features (40) + Ensemble
# =========================
def _norm(d: Dict[int,float])->Dict[int,float]:
    s = sum(d.values()) or 1e-9
    return {k: v/s for k,v in d.items()}

def _freq_win(tail:List[int], k:int)->Dict[int,float]:
    if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
    win = tail[-k:] if len(tail)>=k else tail
    tot=max(1,len(win))
    return _norm({c: win.count(c)/tot for c in (1,2,3,4)})

def _markov1(tail:List[int])->Dict[int,float]:
    if len(tail)<2: return {1:0.25,2:0.25,3:0.25,4:0.25}
    last=tail[-1]
    counts={1:1e-9,2:1e-9,3:1e-9,4:1e-9}
    for i in range(len(tail)-1):
        if tail[i]==last:
            counts[tail[i+1]]+=1
    return _norm(counts)

def _markov2(tail:List[int])->Dict[int,float]:
    if len(tail)<3: return {1:0.25,2:0.25,3:0.25,4:0.25}
    a,b=tail[-2],tail[-1]
    counts={1:1e-9,2:1e-9,3:1e-9,4:1e-9}
    for i in range(len(tail)-2):
        if tail[i]==a and tail[i+1]==b:
            counts[tail[i+2]]+=1
    return _norm(counts)

def _runlength_bias(tail:List[int])->Dict[int,float]:
    if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
    last=tail[-1]
    rl=1
    for i in range(len(tail)-2,-1,-1):
        if tail[i]==last: rl+=1
        else: break
    out={c:0.0 for c in (1,2,3,4)}
    # leve reversão quando rl>1; leve momentum quando rl==1
    if rl==1:
        out[last]=0.40
    else:
        for c in (1,2,3,4):
            out[c]=0.33 if c!=last else 0.01
    return _norm(out)

def _momentum_burst(tail:List[int])->Dict[int,float]:
    # Fibonacci wins
    mix={c:0.0 for c in (1,2,3,4)}
    for k,w in ((8,0.2),(13,0.3),(21,0.5)):
        pk=_freq_win(tail,k)
        for c in (1,2,3,4): mix[c]+=w*pk[c]
    return _norm(mix)

def _contrarian(p:Dict[int,float])->Dict[int,float]:
    inv={c: max(1e-9, 1.0-p.get(c,0.0)) for c in (1,2,3,4)}
    return _norm(inv)

def _dirichlet_smooth(p:Dict[int,float], samples:int)->Dict[int,float]:
    p=_norm(p)
    alpha=max(0.2, min(1.2, 1.0 - 0.0005*float(samples)))
    eff={c: p[c]*max(1.0,float(samples)) for c in (1,2,3,4)}
    sm={c: eff[c]+alpha for c in (1,2,3,4)}
    return _norm(sm)

def _seq3_support(tail:List[int])->Dict[int,float]:
    if len(tail)<4: return {1:0.25,2:0.25,3:0.25,4:0.25}
    a,b,c=tail[-3],tail[-2],tail[-1]
    counts={1:1e-9,2:1e-9,3:1e-9,4:1e-9}
    tot=0
    for i in range(len(tail)-3):
        if tail[i]==a and tail[i+1]==b and tail[i+2]==c:
            counts[tail[i+3]]+=1; tot+=1
    if tot==0: return {1:0.25,2:0.25,3:0.25,4:0.25}
    return _norm(counts)

# --- gera 40 features (nome -> prob dict) ---
def _features_40(tail:List[int])->Dict[str,Dict[int,float]]:
    feats={}
    # 1) Freq janelas (7)
    for k in [5,8,13,21,34,55,89]:
        feats[f"freq_{k}"]=_freq_win(tail,k)
    # 2) Markov (2)
    feats["mk1"]=_markov1(tail)
    feats["mk2"]=_markov2(tail)
    # 3) Momentum/Contrarian/Run (3)
    feats["momentum"]=_momentum_burst(tail)
    base=_freq_win(tail,60)
    feats["contrarian"]=_contrarian(base)
    feats["runbias"]=_runlength_bias(tail)
    # 4) Seq3 (1)
    feats["seq3"]=_seq3_support(tail)
    # 5) Freq longas (4)
    for k in [120,180,240,300]:
        feats[f"freq_{k}"]=_freq_win(tail,k)
    # 6) Dirichlet smooth sobre bases (5)
    samples=_timeline_size()
    for src in ["freq_13","freq_21","freq_55","mk1","mk2"]:
        feats[f"dir_{src}"]=_dirichlet_smooth(feats[src], samples)
    # 7) Variantes com leve ruído/perturbação (para diversidade) (8)
    rnd = random.Random( (tail[-1] if tail else 0) + samples )
    for i,basek in enumerate([8,21,34,55,89,120,180,240], start=1):
        p=_freq_win(tail,basek)
        # jitter sutil
        j={c: max(1e-9, p[c]*(0.98 + 0.04*rnd.random())) for c in (1,2,3,4)}
        feats[f"jitter_{basek}"]=_norm(j)
    # 8) Combos (10)
    def _mix(a,b,wa,wb):
        return _norm({c: wa*a[c]+wb*b[c] for c in (1,2,3,4)})
    combos=[
        ("mix_mk1_mom", "mk1","momentum",0.6,0.4),
        ("mix_mk2_mom", "mk2","momentum",0.6,0.4),
        ("mix_mk1_contra","mk1","contrarian",0.5,0.5),
        ("mix_seq3_mom","seq3","momentum",0.65,0.35),
        ("mix_run_mom","runbias","momentum",0.5,0.5),
        ("mix_f21_mk1","freq_21","mk1",0.55,0.45),
        ("mix_f34_mk2","freq_34","mk2",0.55,0.45) if "freq_34" in feats else None,
        ("mix_f55_dir13","freq_55","dir_freq_13",0.5,0.5),
        ("mix_f89_dir21","freq_89","dir_freq_21",0.5,0.5),
        ("mix_contra_dir55","contrarian","dir_freq_55",0.5,0.5),
    ]
    for tpl in combos:
        if not tpl: continue
        name,a,b,wa,wb=tpl
        if a in feats and b in feats:
            feats[name]=_mix(feats[a], feats[b], wa, wb)
    # até aqui ~40 no total
    return feats

# =========================
# ML Weights (carregar/treinar/aplicar)
# =========================
def _weights_load()->Dict[str,Tuple[float,float,float,float]]:
    con=_con()
    rows=con.execute("SELECT fname,w1,w2,w3,w4 FROM ml_weights").fetchall()
    con.close()
    if rows: return {r["fname"]: (float(r["w1"]),float(r["w2"]),float(r["w3"]),float(r["w4"])) for r in rows}
    # inicializa pesos uniformes para as features atuais
    feats=_features_40(_timeline_tail(400))
    init={}
    for f in feats.keys():
        init[f]=(0.25,0.25,0.25,0.25)
    _weights_save(init)
    return init

def _weights_save(wmap:Dict[str,Tuple[float,float,float,float]]):
    con=_con(); cur=con.cursor()
    for f,(w1,w2,w3,w4) in wmap.items():
        cur.execute("""INSERT INTO ml_weights(fname,w1,w2,w3,w4)
                       VALUES(?,?,?,?,?)
                       ON CONFLICT(fname) DO UPDATE SET w1=excluded.w1,w2=excluded.w2,
                       w3=excluded.w3,w4=excluded.w4""", (f,float(w1),float(w2),float(w3),float(w4)))
    con.commit(); con.close()

def _softmax(v:Dict[int,float])->Dict[int,float]:
    mx=max(v.values()) if v else 0.0
    ex={k: math.exp(v[k]-mx) for k in v}
    s=sum(ex.values()) or 1e-9
    return {k: ex[k]/s for k in v}

def _ensemble_predict(tail:List[int], wmap)->Tuple[Dict[int,float], Dict[str,Dict[int,float]]]:
    feats=_features_40(tail)
    # garante que toda feature tenha peso
    for f in list(feats.keys()):
        if f not in wmap:
            wmap[f]=(0.25,0.25,0.25,0.25)
    # score linear por classe
    score={1:0.0,2:0.0,3:0.0,4:0.0}
    for f,p in feats.items():
        w=wmap[f]; # w1..w4
        score[1]+=w[0]*p[1]; score[2]+=w[1]*p[2]
        score[3]+=w[2]*p[3]; score[4]+=w[3]*p[4]
    prob=_softmax(score)  # normaliza
    return prob, feats

def _train_if_needed():
    # retreina pesos levemente com base no último fechamento registrado
    con=_con()
    meta=con.execute("SELECT last_train_at,total_samples FROM ml_meta WHERE id=1").fetchone()
    last_train=int(meta["last_train_at"] or 0); total=int(meta["total_samples"] or 0)
    if total - last_train < TRAIN_INTERVAL:
        con.close(); return
    # pegue um batch simples de últimas N amostras (janelas sobre timeline)
    rows=con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (TRAIN_INTERVAL+5,)).fetchall()
    con.close()
    seq=[int(r["number"]) for r in rows][::-1]
    if len(seq)<6: 
        _meta_set_last_train(total); 
        return
    wmap=_weights_load()
    # Treino por janelas (online): para cada posição, predizer próximo e ajustar
    lr=ML_LEARNING_RATE
    tail=[]
    for i in range(len(seq)-1):
        tail.append(seq[i])
        if len(tail)<5: 
            continue
        true_next=seq[i+1]
        prob,_= _ensemble_predict(tail, wmap)
        # ajuste: para cada feature, subida de gradiente simples
        feats=_features_40(tail)
        for f,p in feats.items():
            w=list(wmap[f])
            # sinal de erro (target one-hot - prob)
            err={1:-prob[1],2:-prob[2],3:-prob[3],4:-prob[4]}
            err[true_next]+=1.0
            # atualiza pesos na direção do p (feature)
            w[0]+=lr*err[1]*p[1]; w[1]+=lr*err[2]*p[2]
            w[2]+=lr*err[3]*p[3]; w[3]+=lr*err[4]*p[4]
            # regularização leve para manter estável
            mean=sum(w)/4.0
            w=[0.98*x+0.02*mean for x in w]
            # normaliza (mantém soma ≈1)
            s=sum(w) or 1e-9
            w=[x/s for x in w]
            wmap[f]=tuple(w)
    _weights_save(wmap)
    _meta_set_last_train(total)

def _meta_set_last_train(total:int):
    con=_con(); con.execute("UPDATE ml_meta SET last_train_at=?, total_samples=? WHERE id=1", (total,total)); con.commit(); con.close()

# =========================
# IA de Decisão (usa ensemble + filtros)
# =========================
def _choose_number(after_hint:Optional[int])->Tuple[int,float,int,float,str]:
    tail=_timeline_tail(400)
    wmap=_weights_load()
    prob,_= _ensemble_predict(tail, wmap)
    # runner-up ls2 simples (placeholder; ls não persistido nesta versão)
    rank=sorted(prob.items(), key=lambda kv: kv[1], reverse=True)
    best,conf = rank[0][0], float(rank[0][1])
    gap = (rank[0][1]-rank[1][1]) if len(rank)>=2 else rank[0][1]
    mode="IA"
    # piso de confiança
    if conf < MIN_ENTER_CONF:
        # eleva piso apenas para o melhor
        others=[c for c in (1,2,3,4) if c!=best]
        take=min(MIN_ENTER_CONF-conf, sum(prob[c] for c in others))
        if take>0:
            scale=(sum(prob[c] for c in others)-take)/max(1e-9, sum(prob[c] for c in others))
            for c in others: prob[c]*=scale
            prob[best]=min(0.95, prob[best]+take)
        # renormaliza
        s=sum(prob.values()) or 1e-9
        prob={k:prob[k]/s for k in prob}
        best=max(prob, key=prob.get); conf=float(prob[best])
        mode+="|conf_floor"
    return best, conf, _timeline_size(), gap, mode

def _ngram_snapshot(suggested:int)->str:
    tail=_timeline_tail(400)
    p=_freq_win(tail,55)
    pct=lambda x:f"{x*100:.1f}%"
    p1,p2,p3,p4 = pct(p[1]), pct(p[2]), pct(p[3]), pct(p[4])
    conf=pct(p.get(int(suggested),0.0))
    return (f"📈 Amostra: {_timeline_size()} • Conf: {conf}\n"
            f"🔎 E(n-gram/Fibo55): 1 {p1} | 2 {p2} | 3 {p3} | 4 {p4}")

# =========================
# Rotas básicas
# =========================
@app.get("/")
async def root():
    return {"ok": True, "service": "GuardiAo Auto Bot v6.0", "time": datetime.datetime.utcnow().isoformat()+"Z"}

@app.get("/health")
async def health():
    return {"ok": True, "db_exists": os.path.exists(DB_PATH), "db_path": DB_PATH}

@app.get("/debug_cfg")
async def debug_cfg():
    return {
        "MAX_GALE": MAX_GALE, "OBS_TIMEOUT_SEC": OBS_TIMEOUT_SEC,
        "DEDUP_WINDOW_SEC": DEDUP_WINDOW_SEC, "TRAIN_INTERVAL": TRAIN_INTERVAL,
        "ML_LEARNING_RATE": ML_LEARNING_RATE, "MIN_ENTER_CONF": MIN_ENTER_CONF
    }

# =========================
# Webhook
# =========================
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

    # 0) treino on-line (lazy)
    _train_if_needed()

    # 1) ANALISANDO — só memória (com dedupe)
    if RX_ANALISE.search(text):
        if _seen_recent("analise", _dedupe_key(text)):
            return {"ok": True, "skipped": "analise_dupe"}
        seq=_parse_seq(text)
        if seq: _append_seq(seq)
        return {"ok": True, "analise_seq": len(seq)}

    # 2) FECHAMENTO — fecha G0/G1 (com dedupe)
    if RX_FECHA.search(text) or RX_GREEN.search(text) or RX_RED.search(text):
        if _seen_recent("fechamento", _dedupe_key(text)):
            return {"ok": True, "skipped": "fechamento_dupe"}
        pend=_pending_get()
        if pend:
            need=min(2, MAX_GALE+1)
            obs=_parse_obs_final(text, need=need)
            if obs: _pending_seen_append(obs, need=need)
            pend=_pending_get()
            seen=[s for s in (pend["seen"] or "").split("-") if s]
            suggested=int(pend["suggested"] or 0)
            outcome="LOSS"; stage_lbl="G1"
            if len(seen)>=1 and seen[0].isdigit() and int(seen[0])==suggested:
                outcome="GREEN"; stage_lbl="G0"
            elif len(seen)>=2 and seen[1].isdigit() and int(seen[1])==suggested and MAX_GALE>=1:
                outcome="GREEN"; stage_lbl="G1"
            final_seen="-".join(seen[:need]) if seen else "X"
            msg_txt=_pending_close(final_seen, outcome, stage_lbl, suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)
            return {"ok": True, "closed": outcome, "seen": final_seen}
        return {"ok": True, "noted_close": True}

    # 3) ENTRADA — abre pending e sugere, incluindo “após X” se existir
    if RX_ENTRADA.search(text):
        if _seen_recent("entrada", _dedupe_key(text)):
            if SHOW_DEBUG:
                await tg_send(TARGET_CHANNEL, "DEBUG: entrada duplicada ignorada.")
            return {"ok": True, "skipped": "entrada_dupe"}

        seq=_parse_seq(text)
        if seq: _append_seq(seq)
        after_hint=_parse_after(text)  # <= usado no texto

        # fecha pendência esquecida (X-fill)
        pend=_pending_get()
        if pend:
            need=min(2,MAX_GALE+1)
            seen=[s for s in (pend["seen"] or "").split("-") if s]
            while len(seen)<need: seen.append("X")
            final_seen="-".join(seen[:need])
            suggested=int(pend["suggested"] or 0)
            outcome="LOSS"; stage_lbl="G1"
            if len(seen)>=1 and seen[0].isdigit() and int(seen[0])==suggested:
                outcome="GREEN"; stage_lbl="G0"
            elif len(seen)>=2 and seen[1].isdigit() and int(seen[1])==suggested and MAX_GALE>=1:
                outcome="GREEN"; stage_lbl="G1"
            msg_txt=_pending_close(final_seen, outcome, stage_lbl, suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)

        # decide
        best, conf, samples, gap, mode = _choose_number(after_hint)
        opened=_pending_open(best, after_hint)
        if opened:
            padrao = f"GEN após {after_hint}" if after_hint else "GEN"
            txt=(f"🤖 <b>IA SUGERE</b> — <b>{best}</b>\n"
                 f"🧩 <b>Padrão:</b> {padrao}\n"
                 f"📊 <b>Conf:</b> {conf*100:.2f}% | <b>Amostra≈</b>{samples} | <b>gap≈</b>{gap*100:.1f}pp\n"
                 f"🧠 <b>Modo:</b> {mode}\n"
                 f"{_ngram_snapshot(best)}")
            await tg_send(TARGET_CHANNEL, txt)
            return {"ok": True, "entry_opened": True, "best": best, "conf": conf, "after_hint": after_hint}
        else:
            if SHOW_DEBUG:
                await tg_send(TARGET_CHANNEL, "DEBUG: pending já aberto; entrada ignorada.")
            return {"ok": True, "skipped": "pending_open"}

    # Não reconhecido
    if SHOW_DEBUG:
        await tg_send(TARGET_CHANNEL, "DEBUG: Mensagem não reconhecida como ENTRADA/FECHAMENTO/ANALISANDO.")
    return {"ok": True, "skipped": "unmatched"}