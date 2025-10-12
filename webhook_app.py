#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GuardiAo Auto Bot — webhook_app.py
v7.5.2-guard  (FLOW + Multi-horizonte + Anti-bias forte + Slate + ClassGuard + Ban)

- Fecha G0 pelo último número entre parênteses
- Ensemble: ProfoundSim + (curto, médio, longo, n-gram) + Fibo10 + T30
- Anti-viés (dominância / repetições) + empurrão de entropia
- Slate de opções (top-K): primário + alternativas curto/médio
- ClassCooldown + ClassGuard: evita grudar e corta classe em streaks ruins
- Bias clamp no reforço (evita “viciar” no 4)
- NeuroController FLOW: sem cooldown, corte por confiança; ε explora dentro do slate
- Ban temporário por classe (/admin/ban/{cls}/{plays})
- Reforço leve (EMA) por classe
- Timeout de pendência (anti-trava) e dedupe
- Admin: /health /debug_cfg /admin/status /admin/unlock /admin/flush_bias /admin/retune /admin/ban/{c}/{n}
"""

import os, re, time, sqlite3, datetime, hashlib, math, json, random
from typing import List, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException

# ================== ENV ==================
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "").strip()

SHOW_DEBUG       = os.getenv("SHOW_DEBUG", "False").strip().lower() == "true"
MAX_GALE         = int(os.getenv("MAX_GALE", "0"))        # G0-only
OBS_TIMEOUT_SEC  = int(os.getenv("OBS_TIMEOUT_SEC", "420"))
DEDUP_WINDOW_SEC = int(os.getenv("DEDUP_WINDOW_SEC", "40"))

# Controller FLOW
CTL_MIN_CONF     = float(os.getenv("CTL_MIN_CONF", "0.74"))
CTL_EPSILON      = float(os.getenv("CTL_EPSILON", "0.05"))

# Slate (opções alternativas)
SLATE_K          = int(os.getenv("SLATE_K", "3"))          # 1..4
SLATE_GAP_MAX    = float(os.getenv("SLATE_GAP_MAX", "0.12"))
SLATE_MINCONF    = float(os.getenv("SLATE_MINCONF", "0.20"))
MIN_ALT_GAP      = float(os.getenv("MIN_ALT_GAP", "0.05"))  # se gap <= isso, consideramos 2º

# Anti-viés/diversidade
DIV_LASTK       = int(os.getenv("DIVERSITY_LASTK", "40"))
DIV_MAX_SHARE   = float(os.getenv("DIVERSITY_MAX_SHARE", "0.30"))
DIV_MAX_SAME    = int(os.getenv("DIVERSITY_MAX_SAME", "1"))
ENTROPY_PUSH    = float(os.getenv("ANTIENTROPY_PUSH", "0.20"))

# ClassCooldown / Guard
CLASS_COOLDOWN_REPS  = int(os.getenv("CLASS_COOLDOWN_REPS", "2"))
CLASS_COOLDOWN_DELTA = float(os.getenv("CLASS_COOLDOWN_DELTA", "0.06"))
CLASS_GUARD_LASTN    = int(os.getenv("CLASS_GUARD_LASTN", "30"))
CLASS_GUARD_MIN_HIT  = float(os.getenv("CLASS_GUARD_MIN_HIT", "0.35"))
CLASS_GUARD_HARDLOSS = int(os.getenv("CLASS_GUARD_HARDLOSS", "2"))

# Bias clamp no reforço
BIAS_CLAMP = float(os.getenv("BIAS_CLAMP", "0.28"))

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN or not TARGET_CHANNEL:
    raise RuntimeError("Faltam ENV: TG_BOT_TOKEN, WEBHOOK_TOKEN, TARGET_CHANNEL.")
TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
DB_PATH = "/opt/render/project/src/main.sqlite"

# ================== APP ==================
app = FastAPI(title="GuardiAo Auto Bot (webhook)", version="7.5.2-guard")

# ================== DB ==================
def _con():
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

def db_init():
    con=_con(); cur=con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS processed(
        update_id TEXT PRIMARY KEY, seen_at INTEGER NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL, number INTEGER NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pending(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER, opened_at INTEGER,
        suggested INTEGER, seen TEXT DEFAULT '', open INTEGER DEFAULT 1)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS score(
        id INTEGER PRIMARY KEY CHECK(id=1),
        green INTEGER DEFAULT 0, loss INTEGER DEFAULT 0,
        streak_green INTEGER DEFAULT 0, streak_loss INTEGER DEFAULT 0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS dedupe(
        kind TEXT NOT NULL, dkey TEXT NOT NULL, ts INTEGER NOT NULL,
        PRIMARY KEY (kind,dkey))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS neural(
        id INTEGER PRIMARY KEY CHECK(id=1),
        temp REAL DEFAULT 0.85,
        bias_json TEXT DEFAULT '{"1":0.0,"2":0.0,"3":0.0,"4":0.0}',
        weight_neural REAL DEFAULT 0.70)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS neuroctl(
        id INTEGER PRIMARY KEY CHECK(id=1),
        epsilon REAL DEFAULT 0.06,
        min_conf REAL DEFAULT 0.58,
        cool_after_losses INTEGER DEFAULT 2,
        cool_secs INTEGER DEFAULT 180,
        throttle_until INTEGER DEFAULT 0,
        regime TEXT DEFAULT 'neutral')""")
    cur.execute("""CREATE TABLE IF NOT EXISTS meta(
        k TEXT PRIMARY KEY, v TEXT
    )""")
    if not con.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        con.execute("INSERT INTO score(id,green,loss,streak_green,streak_loss) VALUES(1,0,0,0,0)")
    if not con.execute("SELECT 1 FROM neural WHERE id=1").fetchone():
        con.execute("INSERT INTO neural(id,temp,bias_json,weight_neural) VALUES(1,0.85,'{\"1\":0.0,\"2\":0.0,\"3\":0.0,\"4\":0.0}',0.70)")
    if not con.execute("SELECT 1 FROM neuroctl WHERE id=1").fetchone():
        con.execute("INSERT INTO neuroctl(id,epsilon,min_conf,cool_after_losses,cool_secs,throttle_until,regime) VALUES(1,0.06,0.58,2,180,0,'neutral')")
    con.commit(); con.close()
db_init()

def _kv_get(k:str, default:str=""):
    con=_con(); row=con.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone(); con.close()
    return row["v"] if row and row["v"] is not None else default

def _kv_set(k:str, v:str):
    con=_con(); con.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", (k, str(v))); con.commit(); con.close()

def _mark_processed(upd: str):
    try:
        con=_con()
        con.execute("INSERT OR IGNORE INTO processed(update_id,seen_at) VALUES(?,?)",(str(upd),int(time.time())))
        con.commit(); con.close()
    except Exception:
        pass

# ===== timeline =====
def _timeline_tail(n:int=400)->List[int]:
    con=_con()
    rows=con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?",(n,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def _append_seq(seq: List[int]):
    if not seq: return
    con=_con(); now=int(time.time())
    con.executemany("INSERT INTO timeline(created_at,number) VALUES(?,?)",[(now,int(x)) for x in seq])
    con.commit(); con.close()

def _timeline_size()->int:
    con=_con(); row=con.execute("SELECT COUNT(*) c FROM timeline").fetchone(); con.close()
    return int(row["c"] or 0)

# ===== score =====
def _score_add(outcome:str):
    con=_con()
    row=con.execute("SELECT green,loss,streak_green,streak_loss FROM score WHERE id=1").fetchone()
    g,l,sg,sl=(int(row["green"]),int(row["loss"]),int(row["streak_green"]),int(row["streak_loss"])) if row else (0,0,0,0)
    if outcome.upper()=="GREEN": g+=1; sg+=1; sl=0
    elif outcome.upper()=="LOSS": l+=1; sl+=1; sg=0
    con.execute("INSERT OR REPLACE INTO score(id,green,loss,streak_green,streak_loss) VALUES(1,?,?,?,?)",(g,l,sg,sl))
    con.commit(); con.close()

def _score_text()->str:
    con=_con(); row=con.execute("SELECT green,loss FROM score WHERE id=1").fetchone(); con.close()
    g,l=(int(row["green"]),int(row["loss"])) if row else (0,0)
    tot=g+l; acc=(g/tot*100.0) if tot>0 else 0.0
    return f"{g} GREEN × {l} LOSS — {acc:.1f}%"

# ===== pending =====
def _pending_get()->Optional[sqlite3.Row]:
    con=_con(); row=con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone(); con.close()
    return row

def _pending_open(suggested:int):
    if _pending_get(): return False
    con=_con(); now=int(time.time())
    con.execute("""INSERT INTO pending(created_at,opened_at,suggested,seen,open)
                   VALUES(?,?,?,?,1)""",(now,now,int(suggested),""))
    con.commit(); con.close()
    # rastreia repetições do mesmo sugerido
    last=_kv_get("last_suggested","")
    reps=int(_kv_get("same_suggested_reps","0"))
    if last.isdigit() and int(last)==int(suggested): reps+=1
    else: reps=1
    _kv_set("last_suggested", str(suggested))
    _kv_set("same_suggested_reps", str(reps))
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
    _update_neural_feedback(suggested, outcome)
    _score_add(outcome)
    obs=[int(x) for x in final_seen.split("-") if x.isdigit()]
    _append_seq(obs)
    our = suggested if outcome.upper()=="GREEN" else "X"
    return (f"{'🟢' if outcome.upper()=='GREEN' else '🔴'} <b>{outcome.upper()}</b> — finalizado "
            f"(<b>{stage_lbl}</b>, nosso={our}, observados={final_seen}).\n"
            f"📊 Geral: {_score_text()}\n\n{_ngram_snapshot(suggested)}")

# ===== anti-trava =====
def _pending_timeout_check() -> Optional[dict]:
    row=_pending_get()
    if not row: return None
    opened_at=int(row["opened_at"] or 0); now=int(time.time())
    if now - opened_at >= OBS_TIMEOUT_SEC:
        suggested=int(row["suggested"] or 0)
        msg_txt=_pending_close("X","LOSS","G0",suggested)
        return {"timeout_closed": True, "final_seen": "X", "suggested": suggested, "msg": msg_txt}
    return None

# ===== dedupe =====
def _dedupe_key(text:str)->str:
    base=re.sub(r"\s+"," ",(text or "")).strip().lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _seen_recent(kind:str,dkey:str)->bool:
    now=int(time.time()); con=_con()
    row=con.execute("SELECT ts FROM dedupe WHERE kind=? AND dkey=?", (kind,dkey)).fetchone()
    if row and now-int(row["ts"])<= DEDUP_WINDOW_SEC:
        con.close(); return True
    con.execute("INSERT OR REPLACE INTO dedupe(kind,dkey,ts) VALUES (?,?,?)",(kind,dkey,now))
    con.commit(); con.close(); return False

# ================== Especialistas ==================
def _norm(d:Dict[int,float])->Dict[int,float]:
    s=sum(d.values()) or 1e-9
    return {k:v/s for k,v in d.items()}

def _post_freq(tail:List[int], k:int)->Dict[int,float]:
    if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
    win=tail[-k:] if len(tail)>=k else tail
    tot=max(1,len(win))
    return _norm({c:win.count(c)/tot for c in (1,2,3,4)})

def _post_e1_ngram(tail:List[int])->Dict[int,float]:
    mix={c:0.0 for c in (1,2,3,4)}
    for k,w in ((8,0.25),(21,0.35),(55,0.40)):
        pk=_post_freq(tail,k)
        for c in (1,2,3,4): mix[c]+=w*pk[c]
    return _norm(mix)

def _post_e2_short(tail):  return _post_freq(tail, 40)   # curto
def _post_e3_mid(tail):    return _post_freq(tail, 120)  # médio
def _post_e4_long(tail):   return _post_freq(tail, 300)  # longo

def _e_fibo10(tail:List[int])->Dict[int,float]:
    if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
    wins=(10,21,34,55)
    mix={c:0.0 for c in (1,2,3,4)}
    for k in wins:
        win=tail[-k:] if len(tail)>=k else tail
        best=max((1,2,3,4), key=lambda c: win.count(c))
        mix[best]+=1.0
    return _norm(mix)

def _e_t30()->Dict[int,float]:
    m=int(datetime.datetime.utcnow().timestamp()//1800)
    vals=[(m*i*17)%100 for i in (1,2,3,4)]
    s=sum(vals) or 1
    return {i+1: vals[i]/s for i in range(4)}

def _conf_floor(post:Dict[int,float], floor=0.30, cap=0.95):
    post=_norm({c:float(post.get(c,0)) for c in (1,2,3,4)})
    b=max(post,key=post.get); mx=post[b]
    if mx<floor:
        others=[c for c in (1,2,3,4) if c!=b]
        s=sum(post[c] for c in others); take=min(floor-mx,s)
        if s>0:
            scale=(s-take)/s
            for c in others: post[c]*=scale
        post[b]=min(cap,mx+take)
    if post[b]>cap:
        ex=post[b]-cap; post[b]=cap
        add=ex/3.0
        for c in (1,2,3,4):
            if c!=b: post[c]+=add
    return _norm(post)

# ================== ProfoundSim (neural “leve”) ==================
def _load_neural_params():
    con=_con(); row=con.execute("SELECT temp,bias_json,weight_neural FROM neural WHERE id=1").fetchone(); con.close()
    temp=float(row["temp"] if row else 0.85)
    wneu=float(row["weight_neural"] if row else 0.70)
    try: bias=json.loads(row["bias_json"]) if row and row["bias_json"] else {"1":0.0,"2":0.0,"3":0.0,"4":0.0}
    except Exception: bias={"1":0.0,"2":0.0,"3":0.0,"4":0.0}
    return temp, wneu, {int(k):float(v) for k,v in bias.items()}

def _save_neural_params(temp=None,wneu=None,bias=None):
    con=_con(); row=con.execute("SELECT temp,bias_json,weight_neural FROM neural WHERE id=1").fetchone()
    cur_temp=float(row["temp"] if row else 0.85)
    cur_w=float(row["weight_neural"] if row else 0.70)
    cur_bias=json.loads(row["bias_json"]) if row and row["bias_json"] else {"1":0.0,"2":0.0,"3":0.0,"4":0.0}
    if temp is not None: cur_temp=float(temp)
    if wneu is not None: cur_w=float(wneu)
    if bias is not None:
        # clamp de bias por classe
        for k in ("1","2","3","4"):
            if k in bias:
                bias[k]=max(-BIAS_CLAMP, min(BIAS_CLAMP, float(bias[k])))
        cur_bias={str(k):float(v) for k,v in bias.items()}
    con.execute("INSERT OR REPLACE INTO neural(id,temp,bias_json,weight_neural) VALUES(1,?,?,?)",(cur_temp,json.dumps(cur_bias),cur_w))
    con.commit(); con.close()

def _features_from_tail(tail:List[int])->List[float]:
    if not tail: return [0.25,0.25,0.25,0.25]+[0.0]*28
    L=len(tail)
    freq=[tail.count(c)/L for c in (1,2,3,4)]
    wins=[]
    for k in (8,13,21,34,55,89,144):
        win=tail[-k:] if L>=k else tail
        wins.extend([win.count(c)/max(1,len(win)) for c in (1,2,3,4)])
    trans=[[0]*4 for _ in range(4)]
    for a,b in zip(tail[:-1],tail[1:]): trans[a-1][b-1]+=1
    tn=[]
    for i in range(4):
        s=sum(trans[i]) or 1
        tn.extend([trans[i][j]/s for j in range(4)])
    streaks=[0,0,0,0]; cur=tail[-1]; s=0
    for x in reversed(tail):
        if x==cur: s+=1
        else: break
    streaks[cur-1]=s/max(1,min(20,L))
    mean=sum(tail)/L
    var=sum((x-mean)**2 for x in tail)/L
    ent=0.0
    for p in freq:
        if p>0: ent-=p*math.log(p+1e-12)
    return freq+wins+tn+streaks+[var/3.0, ent/1.4]

def _profoundsim_logits(feat:List[float], seed_base:int)->List[float]:
    H=512
    h=[0.0]*H
    for i in range(H):
        s=0.0
        for j,f in enumerate(feat):
            seed=seed_base + i*131 + j*17
            w=math.sin(seed*0.000113)*math.cos(seed*0.000071)
            s += f*w
        h[i]=math.tanh(1.2*s + 0.15*math.sin(s*3.0))
    logits=[0.0]*4
    for c in range(4):
        s=0.0
        for i,val in enumerate(h):
            seed=seed_base + (c+1)*997 + i*29
            w=math.sin(seed*0.000091)*math.cos(seed*0.000067)
            s += val*w
        logits[c]=s
    return logits

def _softmax(x:List[float], temp:float)->List[float]:
    m=max(x); ex=[math.exp((xi-m)/max(0.15,temp)) for xi in x]; s=sum(ex) or 1e-9
    return [e/s for e in ex]

def _neural_probs(tail:List[int])->Dict[int,float]:
    temp, _, bias=_load_neural_params()
    feat=_features_from_tail(tail)
    seed_base=len(tail)*1009 + (sum(tail)%997)
    logits=_profoundsim_logits(feat,seed_base)
    for idx,c in enumerate((1,2,3,4)):
        logits[idx]+=float(bias.get(c,0.0))
    probs=_softmax(logits,temp)
    return {c:float(probs[c-1]) for c in (1,2,3,4)}

def _calibrate_from_score():
    con=_con(); row=con.execute("SELECT green,loss FROM score WHERE id=1").fetchone(); con.close()
    if not row: return
    g,l=int(row["green"]),int(row["loss"]); tot=g+l
    if tot<50: return
    acc=g/max(1,tot)
    temp,wneu,_=_load_neural_params()
    new_temp=max(0.55,min(1.10,1.00-0.35*(acc-0.50)))
    new_wneu=max(0.55,min(0.85,0.60+0.50*(acc-0.50)))
    _save_neural_params(temp=new_temp,wneu=new_wneu,bias=None)

def _update_neural_feedback(suggested:int,outcome:str):
    _,_,bias=_load_neural_params()
    delta=0.10 if outcome.upper()=="GREEN" else -0.07
    ema=0.90
    cur=float(bias.get(suggested,0.0))
    new=ema*cur + (1-ema)*delta
    bias[suggested]=new
    _save_neural_params(bias=bias)
    _calibrate_from_score()

# ================== Anti-bias + ClassGuard ==================
def _lastk_freq(lastk:int)->Dict[int,float]:
    tail=_timeline_tail(lastk)
    if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
    tot=max(1,len(tail))
    return {c: tail.count(c)/tot for c in (1,2,3,4)}

def _entropy_mix(mix:Dict[int,float], push:float)->Dict[int,float]:
    uni={c:0.25 for c in (1,2,3,4)}
    return _norm({c: (1.0-push)*mix.get(c,0.0) + push*uni[c] for c in (1,2,3,4)})

def _antibias_adjust(mix:Dict[int,float])->Tuple[Dict[int,float],dict]:
    diag={}
    freq=_lastk_freq(DIV_LASTK)
    dom=max(freq, key=freq.get); share=freq[dom]
    same_reps=int(_kv_get("same_suggested_reps","0"))
    last=_kv_get("last_suggested","")

    adjusted=dict(mix)
    parts=[]

    # penaliza dominante recente de forma progressiva
    if share > DIV_MAX_SHARE:
        excess = min(0.30, max(0.0, share - DIV_MAX_SHARE))
        adjusted[dom] = max(0.0, adjusted.get(dom,0.0) * (1.0 - 0.9*excess))
        parts.append(f"share{dom}>{DIV_MAX_SHARE:.2f}")

    # corta mais se repetindo o mesmo topo
    if last.isdigit() and int(last)==dom and same_reps >= DIV_MAX_SAME:
        adjusted[dom] = max(0.0, adjusted.get(dom,0.0) * 0.55)
        parts.append(f"same_reps={same_reps}")

    # empurrão de entropia
    if ENTROPY_PUSH > 0:
        adjusted = _entropy_mix(adjusted, ENTROPY_PUSH)
        parts.append(f"H{ENTROPY_PUSH:.2f}")

    adjusted = _conf_floor(_norm(adjusted), 0.30, 0.95)
    diag.update({"dominant":dom,"dominant_share":round(share,4),
                 "same_reps":same_reps,"applied":" & ".join(parts) if parts else "none"})
    return adjusted, diag

def _class_guard_adjust(mix:Dict[int,float])->Tuple[Dict[int,float],dict]:
    """Guarda de classe: se performance geral está baixa ou streak de LOSS está alta,
    corta o topo (especialmente se é o último sugerido) para evitar grudar."""
    con=_con(); row=con.execute("SELECT green,loss,streak_loss FROM score WHERE id=1").fetchone(); con.close()
    g,l,sl=(int(row["green"]),int(row["loss"]),int(row["streak_loss"]))
    acc = g/max(1,(g+l))
    last = _kv_get("last_suggested","")
    adjusted=dict(mix); parts=[]
    # queda de acurácia: baixa o topo 20%
    if acc < CLASS_GUARD_MIN_HIT:
        top=max(adjusted,key=adjusted.get)
        adjusted[top] = adjusted[top]*0.80
        parts.append(f"acc<{CLASS_GUARD_MIN_HIT:.2f}")
    # streak de LOSS: corta ainda mais a última classe sugerida
    if sl >= CLASS_GUARD_HARDLOSS and last.isdigit():
        n=int(last)
        if n in adjusted:
            adjusted[n] = adjusted[n]*0.70
            parts.append(f"streak_loss>={CLASS_GUARD_HARDLOSS}")
    adjusted=_conf_floor(_norm(adjusted),0.30,0.95)
    return adjusted, {"acc":round(acc,4),"streak_loss":sl,"applied":" & ".join(parts) if parts else "none"}

# ================== Ban temporário ==================
def _ban_get()->dict:
    try:
        j=_kv_get("ban_classes","{}"); return json.loads(j) if j else {}
    except: return {}

def _ban_set(b:dict):
    _kv_set("ban_classes", json.dumps(b))

def _ban_apply(mix:dict)->dict:
    b=_ban_get(); adj=dict(mix)
    changed=False
    for k,v in list(b.items()):
        n=int(k); left=int(v)
        if left>0 and n in adj:
            adj[n]=0.0; changed=True
            b[k]=left-1
        if b.get(k,0)<=0:
            b.pop(k, None)
    if changed:
        _ban_set(b)
        s=sum(adj.values()) or 1.0
        adj={c: (p/s) for c,p in adj.items()}
    return adj

# ================== Controller FLOW ==================
def _ctl_load():
    con=_con(); row=con.execute("SELECT epsilon,min_conf,cool_after_losses,cool_secs,throttle_until,regime FROM neuroctl WHERE id=1").fetchone(); con.close()
    if not row: return 0.06, 0.58, 2, 180, 0, "neutral"
    return float(row["epsilon"]), float(row["min_conf"]), int(row["cool_after_losses"]), int(row["cool_secs"]), int(row["throttle_until"]), str(row["regime"])

def _ctl_force_flow():
    con=_con()
    con.execute("""INSERT OR REPLACE INTO neuroctl(id,epsilon,min_conf,cool_after_losses,cool_secs,throttle_until,regime)
                   VALUES(1,?,?,999999,1,0,'flow')""",(CTL_EPSILON, CTL_MIN_CONF))
    con.commit(); con.close()

def _ctl_decide_slate(ranked:List[Tuple[int,float]], gap01:float)->Tuple[bool,str,int]:
    _ctl_force_flow()
    eps, minc, *_ = _ctl_load()
    topn, topc = ranked[0]
    if topc < minc:
        return False, f"conf {topc:.2f} < min {minc:.2f} (flow)", -1
    # ε-explore quando gap pequeno
    if gap01 < MIN_ALT_GAP and len(ranked)>=2:
        # escolhe 2º em pequena fração das vezes
        if random.random() < max(0.02, min(0.50, eps*1.1)):
            return True, f"explore ε≈{eps:.2f} (gap={gap01:.3f})", int(ranked[1][0])
    return True, f"exploit (gap={gap01:.3f})", int(topn)

# ================== Decisão (ensemble + antibias + guards + slate) ==================
def _neural_decide_slate()->Tuple[List[Tuple[int,float]], int, Dict[int,float], float, str, dict, dict]:
    tail=_timeline_tail(400)
    # especialistas
    p_cur=_post_e2_short(tail)
    p_mid=_post_e3_mid(tail)
    p_lon=_post_e4_long(tail)
    p_ng =_post_e1_ngram(tail)
    p_fb =_e_fibo10(tail)
    p_t30=_e_t30()
    p_nn =_neural_probs(tail)
    # pesos
    _, wneu, _=_load_neural_params()
    rest=1.0 - wneu
    w_stat=0.55*rest     # (curto/médio/longo/ngram)
    w_f   =0.25*rest
    w_t   =0.20*rest
    mix={}
    for c in (1,2,3,4):
        stat = 0.40*p_cur.get(c,0)+0.35*p_mid.get(c,0)+0.15*p_lon.get(c,0)+0.10*p_ng.get(c,0)
        mix[c] = wneu*p_nn.get(c,0) + w_stat*stat + w_f*p_fb.get(c,0) + w_t*p_t30.get(c,0)
    mix=_conf_floor(_norm(mix), 0.30, 0.95)

    # anti-viés
    adj, ab_diag = _antibias_adjust(mix)
    # class guard
    adj, cg_diag = _class_guard_adjust(adj)
    # ban temporário
    adj = _ban_apply(adj)

    # ranking + slate
    ranked = sorted(adj.items(), key=lambda kv: kv[1], reverse=True)
    ranked = [(int(n), float(p)) for n,p in ranked[:max(1,min(SLATE_K,4))]]

    # class cooldown: evita grudar quando 2º está perto
    last = _kv_get("last_suggested","")
    reps = int(_kv_get("same_suggested_reps","0"))
    if len(ranked) >= 2 and last.isdigit() and int(last) == ranked[0][0] and reps >= CLASS_COOLDOWN_REPS:
        if (ranked[0][1] - ranked[1][1]) <= CLASS_COOLDOWN_DELTA:
            ranked[0], ranked[1] = ranked[1], ranked[0]  # promove o 2º

    gap01 = (ranked[0][1] - ranked[1][1]) if len(ranked)>=2 else ranked[0][1]
    temp, wneu_cur, _=_load_neural_params()
    reason=f"ProfoundSim+Multi(w={wneu_cur:.2f},T={temp:.2f})"
    return ranked, _timeline_size(), adj, gap01, reason, ab_diag, cg_diag

def _ngram_snapshot(suggested:int)->str:
    tail=_timeline_tail(400); post=_post_e1_ngram(tail)
    pct=lambda x:f"{x*100:.1f}%"
    p1,p2,p3,p4=pct(post[1]),pct(post[2]),pct(post[3]),pct(post[4])
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
    except Exception: pass

async def tg_send_return(chat_id: str, text: str, parse="HTML") -> Optional[int]:
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            r=await cli.post(f"{TELEGRAM_API}/sendMessage",
                             json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                   "disable_web_page_preview": True})
            data=r.json()
            if isinstance(data,dict) and data.get("ok") and data.get("result",{}).get("message_id"):
                return int(data["result"]["message_id"])
    except Exception: pass
    return None

async def tg_delete(chat_id: str, message_id: int):
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            await cli.post(f"{TELEGRAM_API}/deleteMessage",
                           json={"chat_id": chat_id, "message_id": int(message_id)})
    except Exception: pass

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
    nums=[]
    for m in RX_PAREN.finditer(text or ""):
        nums_in=[int(x) for x in RX_NUMS.findall(m.group(1))]
        if nums_in: nums.append(nums_in[-1])
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
    temp,wneu,bias=_load_neural_params()
    eps,minc,coolN,coolS,thr,reg=_ctl_load()
    freq=_lastk_freq(DIV_LASTK); dom=max(freq,key=freq.get); share=freq[dom]
    return {
        "MAX_GALE": MAX_GALE, "OBS_TIMEOUT_SEC": OBS_TIMEOUT_SEC, "DEDUP_WINDOW_SEC": DEDUP_WINDOW_SEC,
        "neural_temp": temp, "neural_weight": wneu, "neural_bias": bias, "bias_clamp": BIAS_CLAMP,
        "ctl_epsilon": eps, "ctl_min_conf": minc, "ctl_cool_after_losses": coolN,
        "ctl_cool_secs": coolS, "ctl_throttle_until": thr, "ctl_regime": reg,
        "antibias": {
            "lastk": DIV_LASTK, "max_share": DIV_MAX_SHARE, "max_same": DIV_MAX_SAME, "entropy_push": ENTROPY_PUSH,
            "dominant": dom, "dominant_share": share,
            "last_suggested": _kv_get("last_suggested",""),
            "same_suggested_reps": int(_kv_get("same_suggested_reps","0"))
        },
        "slate": {"k": SLATE_K, "gap_max": SLATE_GAP_MAX, "min_conf": SLATE_MINCONF, "min_alt_gap": MIN_ALT_GAP},
        "class_cooldown": {"reps": CLASS_COOLDOWN_REPS, "delta": CLASS_COOLDOWN_DELTA},
        "class_guard": {"lastN": CLASS_GUARD_LASTN, "min_hit": CLASS_GUARD_MIN_HIT, "hardloss": CLASS_GUARD_HARDLOSS},
        "ban_classes": _ban_get()
    }

# ----- Admin helpers -----
@app.get("/admin/status")
async def admin_status():
    pend=_pending_get()
    if not pend: return {"open": False}
    return {
        "open": True, "id": int(pend["id"]),
        "opened_at": int(pend["opened_at"] or 0),
        "age_sec": int(time.time()) - int(pend["opened_at"] or 0),
        "suggested": int(pend["suggested"] or 0),
        "seen": (pend["seen"] or "")
    }

@app.post("/admin/unlock")
async def admin_unlock():
    pend=_pending_get()
    if not pend: return {"ok": True, "message": "nenhuma pendência aberta"}
    suggested=int(pend["suggested"] or 0)
    msg_txt=_pending_close("X","LOSS","G0",suggested)
    return {"ok": True, "forced_close": True, "message": msg_txt}

@app.post("/admin/flush_bias")
async def admin_flush_bias():
    # zera bias, repetições e bans
    _save_neural_params(bias={"1":0.0,"2":0.0,"3":0.0,"4":0.0})
    _kv_set("same_suggested_reps","0")
    _kv_set("last_suggested","")
    _ban_set({})
    return {"ok": True, "message": "bias/reset aplicado"}

@app.post("/admin/retune")
async def admin_retune():
    _calibrate_from_score()
    t,w,b=_load_neural_params()
    return {"ok": True, "neural": {"temp":t, "weight":w, "bias":b}}

@app.post("/admin/ban/{cls}/{plays}")
async def admin_ban(cls: int, plays: int):
    cls=int(cls); plays=int(plays)
    if cls not in (1,2,3,4): return {"ok": False, "error":"classe inválida"}
    b=_ban_get(); b[str(cls)]=max(1,plays); _ban_set(b)
    return {"ok": True, "message": f"classe {cls} banida por {plays} jogadas", "ban": b}

# ================== Webhook ==================
@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data=await request.json()
    upd_id=str(data.get("update_id","")); _mark_processed(upd_id)

    msg = data.get("channel_post") or data.get("message") \
        or data.get("edited_channel_post") or data.get("edited_message") or {}
    chat=msg.get("chat") or {}
    chat_id=str(chat.get("id") or "")
    text=(msg.get("text") or msg.get("caption") or "").strip()

    # anti-trava watchdog
    try:
        watchdog=_pending_timeout_check()
        if watchdog and SHOW_DEBUG:
            if watchdog.get("msg"): await tg_send(TARGET_CHANNEL, f"DEBUG: Timeout — fechado automático.\n{watchdog['msg']}")
            else: await tg_send(TARGET_CHANNEL, "DEBUG: Timeout pendência — LOSS G0 X.")
    except Exception: pass

    # filtra fonte
    if SOURCE_CHANNEL and chat_id and chat_id != SOURCE_CHANNEL:
        if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, f"DEBUG: Ignorando chat {chat_id}. Esperado: {SOURCE_CHANNEL}")
        return {"ok": True, "skipped": "wrong_source"}

    # -------- ANALISANDO --------
    if RX_ANALISE.search(text):
        if _seen_recent("analise", _dedupe_key(text)):
            return {"ok": True, "skipped": "analise_dupe"}
        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)
        return {"ok": True, "analise_seq": len(seq)}

    # -------- FECHAMENTO --------
    if RX_FECHA.search(text):
        if _seen_recent("fechamento", _dedupe_key(text)):
            return {"ok": True, "skipped": "fechamento_dupe"}

        pend=_pending_get()
        if pend:
            suggested=int(pend["suggested"] or 0)
            obs=_parse_paren_last_one(text)
            if obs is not None: _pending_seen_set(str(obs))
            seen=(_pending_get()["seen"] or "").strip()
            outcome="LOSS"; stage_lbl="G0"
            if seen.isdigit() and int(seen)==suggested: outcome="GREEN"
            final_seen=seen if seen else "X"
            msg_txt=_pending_close(final_seen, outcome, stage_lbl, suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)
            return {"ok": True, "closed": outcome, "seen": final_seen}

        if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, "DEBUG: Fechamento reconhecido — sem pendência.")
        return {"ok": True, "noted_close": True}

    # -------- ENTRADA --------
    if RX_ENTRADA.search(text):
        if _seen_recent("entrada", _dedupe_key(text)):
            if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, "DEBUG: Entrada duplicada ignorada.")
            return {"ok": True, "skipped": "entrada_dupe"}

        # memória
        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)
        after=_parse_after(text)

        # fecha pendência esquecida como X
        pend=_pending_get()
        if pend:
            suggested=int(pend["suggested"] or 0)
            seen=(pend["seen"] or "").strip()
            final_seen=seen if seen else "X"
            outcome="LOSS"; stage_lbl="G0"
            if seen.isdigit() and int(seen)==suggested: outcome="GREEN"
            msg_txt=_pending_close(final_seen, outcome, stage_lbl, suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)

        # “analisando…”
        analyzing_id = await tg_send_return(TARGET_CHANNEL, "⏳ Analisando padrão, aguarde...")

        # decisão com slate
        ranked, samples, mix, gap01, reason, ab_diag, cg_diag = _neural_decide_slate()

        # filtra slate por thresholds
        slate = [(n,p) for (n,p) in ranked if p >= SLATE_MINCONF]
        if len(slate)==0: slate = [ranked[0]]

        play, why, chosen = _ctl_decide_slate(slate, gap01)
        if not play:
            if analyzing_id is not None: await tg_delete(TARGET_CHANNEL, analyzing_id)
            await tg_send(TARGET_CHANNEL, f"ℹ️ Sinal ignorado (conf baixa): {why}")
            return {"ok": True, "skipped_low_conf": True, "why": why}

        suggested = chosen if chosen>0 else slate[0][0]
        opened=_pending_open(suggested)
        if opened:
            aft_txt=f" após {after}" if after else ""
            alt_txt=""
            if len(slate)>=2 and gap01 <= SLATE_GAP_MAX:
                alts = ", ".join([f"{n}({p*100:.1f}%)" for n,p in slate[1:3]])
                alt_txt=f"\n🅱️ <b>Opções curto/médio:</b> {alts}"

            txt=(f"🤖 <b>IA SUGERE</b> — <b>{suggested}</b>\n"
                 f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
                 f"📊 <b>Conf:</b> {slate[0][1]*100:.2f}% | <b>Amostra≈</b>{samples} | <b>gap≈</b>{gap01*100:.1f}pp\n"
                 f"🧠 <b>Modo:</b> {reason} · flow\n"
                 f"♻️ <i>anti-viés:</i> dom={ab_diag['dominant']}({ab_diag['dominant_share']*100:.1f}%), rep={ab_diag['same_reps']} • {ab_diag['applied']}\n"
                 f"🛡️ <i>class-guard:</i> acc={cg_diag['acc']*100:.1f}% • {cg_diag['applied']}{alt_txt}\n"
                 f"{_ngram_snapshot(suggested)}")
            await tg_send(TARGET_CHANNEL, txt)
            if analyzing_id is not None: await tg_delete(TARGET_CHANNEL, analyzing_id)
            if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, f"DEBUG: Entrada aberta (FLOW slate).")
            return {"ok": True, "entry_opened": True, "best": suggested, "conf": slate[0][1], "gap": gap01}
        else:
            if analyzing_id is not None: await tg_delete(TARGET_CHANNEL, analyzing_id)
            if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, "DEBUG: pending já aberto; entrada ignorada.")
            return {"ok": True, "skipped": "pending_open"}

    # -------- não reconhecido --------
    if SHOW_DEBUG:
        await tg_send(TARGET_CHANNEL, "DEBUG: Mensagem não reconhecida como ENTRADA/FECHAMENTO/ANALISANDO.")
    return {"ok": True, "skipped": "unmatched"}