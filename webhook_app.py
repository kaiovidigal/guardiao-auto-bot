#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GuardiAo Auto Bot — webhook_app.py
v7.4.2 (G0-only; ProfoundSim + NeuroController + AntiBias + Diversidade + Timeout)
- Fecha G0 usando apenas o último número ENTRE PARÊNTESES (ex.: "GREEN (3)", "RED ❌ (4)")
- ProfoundSim (CPU) dominante + especialistas estatísticos como consultoras
- NeuroController: min_conf dinâmico, epsilon-explore, cooldown após loss-streak, regime
- AntiBias: desvia de classe superexplorada (ex.: #4) com janela móvel + caps
- Diversidade: penaliza repetir demais a mesma classe em janela curta
- Aprendizado leve por reforço (EMA) no bias de classes a cada GREEN/LOSS
- Anti-trava de pendência (OBS_TIMEOUT_SEC) e dedupe por conteúdo
- Admin: /admin/status, /admin/unlock, /debug_cfg
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
MAX_GALE         = int(os.getenv("MAX_GALE", "0"))  # G0 only
OBS_TIMEOUT_SEC  = int(os.getenv("OBS_TIMEOUT_SEC", "420"))
DEDUP_WINDOW_SEC = int(os.getenv("DEDUP_WINDOW_SEC", "40"))

# controller “preset equilibrado” (pode sobrescrever via DB depois)
DEFAULT_EPSILON          = float(os.getenv("CTL_EPSILON", "0.05"))
DEFAULT_MIN_CONF         = float(os.getenv("CTL_MIN_CONF", "0.57"))
DEFAULT_COOL_AFTER_LOSSES= int(os.getenv("CTL_COOL_AFTER", "2"))
DEFAULT_COOL_SECS        = int(os.getenv("CTL_COOL_SECS", "120"))

# antibias/diversidade presets (podem ser ajustados via DB)
AB_WINDOW    = int(os.getenv("AB_WINDOW", "240"))   # amostras p/ medir viés
AB_CAP       = float(os.getenv("AB_CAP", "0.36"))   # limiar de fração para capar
DIV_WINDOW   = int(os.getenv("DIV_WINDOW", "14"))   # evita repetir muito curto
DIV_PENALTY  = float(os.getenv("DIV_PENALTY", "0.05"))

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN or not TARGET_CHANNEL:
    raise RuntimeError("Faltam ENV: TG_BOT_TOKEN, WEBHOOK_TOKEN, TARGET_CHANNEL.")
TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
DB_PATH = "/opt/render/project/src/main.sqlite"

# ================== APP ==================
app = FastAPI(title="GuardiAo Auto Bot (webhook)", version="7.4.2")

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
        update_id TEXT PRIMARY KEY, seen_at INTEGER NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL, number INTEGER NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pending(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER, opened_at INTEGER,
        suggested INTEGER, seen TEXT DEFAULT '', open INTEGER DEFAULT 1)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS score(
        id INTEGER PRIMARY KEY CHECK(id=1),
        green INTEGER DEFAULT 0, loss INTEGER DEFAULT 0,
        streak_green INTEGER DEFAULT 0, streak_loss INTEGER DEFAULT 0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS dedupe(
        kind TEXT NOT NULL, dkey TEXT NOT NULL, ts INTEGER NOT NULL, PRIMARY KEY(kind,dkey))""")
    # Neural params
    cur.execute("""CREATE TABLE IF NOT EXISTS neural(
        id INTEGER PRIMARY KEY CHECK(id=1),
        temp REAL DEFAULT 0.85,
        bias_json TEXT DEFAULT '{"1":0.0,"2":0.0,"3":0.0,"4":0.0}',
        weight_neural REAL DEFAULT 0.72)""")
    # Controller params
    cur.execute("""CREATE TABLE IF NOT EXISTS neuroctl(
        id INTEGER PRIMARY KEY CHECK(id=1),
        epsilon REAL, min_conf REAL, cool_after_losses INTEGER, cool_secs INTEGER,
        throttle_until INTEGER DEFAULT 0, regime TEXT DEFAULT 'neutral')""")
    # AntiBias caps persistentes (opcional)
    cur.execute("""CREATE TABLE IF NOT EXISTS antibias(
        id INTEGER PRIMARY KEY CHECK(id=1),
        window INTEGER, cap REAL, div_window INTEGER, div_penalty REAL)""")

    if not con.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        con.execute("INSERT INTO score(id,green,loss,streak_green,streak_loss) VALUES(1,0,0,0,0)")
    if not con.execute("SELECT 1 FROM neural WHERE id=1").fetchone():
        con.execute("INSERT INTO neural(id,temp,bias_json,weight_neural) VALUES(1,0.85,'{\"1\":0.0,\"2\":0.0,\"3\":0.0,\"4\":0.0}',0.72)")
    if not con.execute("SELECT 1 FROM neuroctl WHERE id=1").fetchone():
        con.execute("""INSERT INTO neuroctl(id,epsilon,min_conf,cool_after_losses,cool_secs,throttle_until,regime)
                       VALUES(1,?,?,?,?,0,'neutral')""",
                    (DEFAULT_EPSILON, DEFAULT_MIN_CONF, DEFAULT_COOL_AFTER_LOSSES, DEFAULT_COOL_SECS))
    if not con.execute("SELECT 1 FROM antibias WHERE id=1").fetchone():
        con.execute("""INSERT INTO antibias(id,window,cap,div_window,div_penalty)
                       VALUES(1,?,?,?,?)""", (AB_WINDOW, AB_CAP, DIV_WINDOW, DIV_PENALTY))
    con.commit(); con.close()
db_init()

# ================== utils DB ==================
def _mark_processed(upd: str):
    try:
        con=_con()
        con.execute("INSERT OR IGNORE INTO processed(update_id,seen_at) VALUES(?,?)",(str(upd), int(time.time())))
        con.commit(); con.close()
    except: pass

def _timeline_tail(n:int=400)->List[int]:
    con=_con(); rows=con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?",(n,)).fetchall(); con.close()
    return [int(r["number"]) for r in rows][::-1]

def _append_seq(seq: List[int]):
    if not seq: return
    con=_con(); now=int(time.time())
    con.executemany("INSERT INTO timeline(created_at,number) VALUES(?,?)",[(now,int(x)) for x in seq])
    con.commit(); con.close()

def _timeline_size()->int:
    con=_con(); row=con.execute("SELECT COUNT(*) c FROM timeline").fetchone(); con.close()
    return int(row["c"] or 0)

def _score_add(outcome:str):
    con=_con()
    row=con.execute("SELECT green,loss,streak_green,streak_loss FROM score WHERE id=1").fetchone()
    g,l,sg,sl = (int(row["green"]), int(row["loss"]), int(row["streak_green"]), int(row["streak_loss"])) if row else (0,0,0,0)
    if outcome.upper()=="GREEN":
        g+=1; sg+=1; sl=0
    else:
        l+=1; sl+=1; sg=0
    con.execute("INSERT OR REPLACE INTO score(id,green,loss,streak_green,streak_loss) VALUES(1,?,?,?,?)",(g,l,sg,sl))
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
    con.execute("INSERT INTO pending(created_at,opened_at,suggested,seen,open) VALUES(?,?,?,?,1)",(now, now, int(suggested), ""))
    con.commit(); con.close(); return True

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
    snap=_ngram_snapshot(suggested)
    return (f"{'🟢' if outcome.upper()=='GREEN' else '🔴'} <b>{outcome.upper()}</b> — finalizado "
            f"(<b>{stage_lbl}</b>, nosso={our}, observados={final_seen}).\n"
            f"📊 Geral: {_score_text()}\n\n{snap}")

# anti-trava pendência
def _pending_timeout_check()->Optional[dict]:
    row=_pending_get()
    if not row: return None
    if int(time.time()) - int(row["opened_at"] or 0) >= OBS_TIMEOUT_SEC:
        suggested=int(row["suggested"] or 0)
        final_seen="X"; outcome="LOSS"; stage_lbl="G0"
        msg=_pending_close(final_seen, outcome, stage_lbl, suggested)
        return {"timeout_closed": True, "msg": msg}
    return None

# dedupe
def _dedupe_key(text: str) -> str:
    base=re.sub(r"\s+"," ",(text or "")).strip().lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _seen_recent(kind: str, dkey: str) -> bool:
    now=int(time.time()); con=_con()
    row=con.execute("SELECT ts FROM dedupe WHERE kind=? AND dkey=?", (kind,dkey)).fetchone()
    if row and now - int(row["ts"]) <= DEDUP_WINDOW_SEC:
        con.close(); return True
    con.execute("INSERT OR REPLACE INTO dedupe(kind,dkey,ts) VALUES(?,?,?)",(kind,dkey,now))
    con.commit(); con.close(); return False

# ================== Especialistas ==================
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

# ================== Neural (ProfoundSim CPU) ==================
def _load_neural_params():
    con=_con(); row=con.execute("SELECT temp,bias_json,weight_neural FROM neural WHERE id=1").fetchone(); con.close()
    temp=float(row["temp"]); wneu=float(row["weight_neural"])
    try: bias=json.loads(row["bias_json"]); bias={int(k):float(v) for k,v in bias.items()}
    except: bias={1:0.0,2:0.0,3:0.0,4:0.0}
    return temp, wneu, bias

def _save_neural_params(temp:float=None, wneu:float=None, bias:Dict[int,float]=None):
    con=_con(); row=con.execute("SELECT temp,bias_json,weight_neural FROM neural WHERE id=1").fetchone()
    curT=float(row["temp"]); curW=float(row["weight_neural"])
    try: curB=json.loads(row["bias_json"])
    except: curB={"1":0.0,"2":0.0,"3":0.0,"4":0.0}
    if temp is not None: curT=float(temp)
    if wneu is not None: curW=float(wneu)
    if bias is not None: curB={str(k):float(v) for k,v in bias.items()}
    con.execute("INSERT OR REPLACE INTO neural(id,temp,bias_json,weight_neural) VALUES(1,?,?,?)",(curT,json.dumps(curB),curW))
    con.commit(); con.close()

def _features_from_tail(tail:List[int])->List[float]:
    if not tail: return [0.25,0.25,0.25,0.25] + [0.0]*28
    L=len(tail)
    freq=[tail.count(c)/L for c in (1,2,3,4)]
    wins=[]
    for k in (8,13,21,34,55,89,144):
        win=tail[-k:] if L>=k else tail
        wins.extend([win.count(c)/max(1,len(win)) for c in (1,2,3,4)])
    trans=[[0]*4 for _ in range(4)]
    for a,b in zip(tail[:-1],tail[1:]): trans[a-1][b-1]+=1
    transn=[]
    for i in range(4):
        s=sum(trans[i]) or 1
        transn.extend([trans[i][j]/s for j in range(4)])
    streak=[0,0,0,0]; cur=tail[-1]; s=0
    for x in reversed(tail):
        if x==cur: s+=1
        else: break
    streak[cur-1]=s/max(1,min(20,L))
    mean=sum(tail)/L; var=sum((x-mean)**2 for x in tail)/L
    ent=0.0
    for p in freq:
        if p>0: ent -= p*math.log(p+1e-12)
    return freq+wins+transn+streak+[var/3.0, ent/1.4]

def _profoundsim_logits(feat:List[float], seed_base:int)->List[float]:
    H=512; h=[0.0]*H
    for i in range(H):
        s=0.0
        for j,f in enumerate(feat):
            seed=seed_base + i*131 + j*17
            w=math.sin(seed*0.000113)*math.cos(seed*0.000071)
            s+=f*w
        h[i]=math.tanh(1.2*s + 0.15*math.sin(s*3.0))
    logits=[0.0,0.0,0.0,0.0]
    for c in range(4):
        s=0.0
        for i,val in enumerate(h):
            seed=seed_base + (c+1)*997 + i*29
            w=math.sin(seed*0.000091)*math.cos(seed*0.000067)
            s+=val*w
        logits[c]=s
    return logits

def _softmax(x:List[float], temp:float)->List[float]:
    m=max(x); ex=[math.exp((xi-m)/max(0.15,temp)) for xi in x]; s=sum(ex) or 1e-9
    return [e/s for e in ex]

def _neural_probs(tail:List[int])->Dict[int,float]:
    temp, _, bias = _load_neural_params()
    feat=_features_from_tail(tail)
    seed_base=len(tail)*1009 + (sum(tail)%997)
    logits=_profoundsim_logits(feat, seed_base)
    for idx,c in enumerate((1,2,3,4)):
        logits[idx]+=float(bias.get(c,0.0))
    probs=_softmax(logits, temp)
    return {c: float(probs[c-1]) for c in (1,2,3,4)}

def _calibrate_from_score():
    con=_con(); row=con.execute("SELECT green,loss FROM score WHERE id=1").fetchone(); con.close()
    if not row: return
    g,l=int(row["green"]), int(row["loss"]); tot=g+l
    if tot<50: return
    acc=g/max(1,tot)
    temp,wneu,_=_load_neural_params()
    newT=max(0.55,min(1.10,1.00-0.35*(acc-0.50)))
    newW=max(0.58,min(0.86,0.64+0.44*(acc-0.50)))
    _save_neural_params(temp=newT,wneu=newW,bias=None)

def _update_neural_feedback(suggested:int, outcome:str):
    temp,wneu,bias=_load_neural_params()
    delta=0.10 if outcome.upper()=="GREEN" else -0.07
    ema=0.90
    cur=float(bias.get(suggested,0.0))
    bias[suggested]=ema*cur+(1-ema)*delta
    _save_neural_params(bias=bias)
    _calibrate_from_score()

# ================== AntiBias & Diversidade ==================
def _ab_load():
    con=_con(); row=con.execute("SELECT window,cap,div_window,div_penalty FROM antibias WHERE id=1").fetchone(); con.close()
    if not row: return AB_WINDOW, AB_CAP, DIV_WINDOW, DIV_PENALTY
    return int(row["window"]), float(row["cap"]), int(row["div_window"]), float(row["div_penalty"])

def _apply_antibias_diversity(mix:Dict[int,float])->Dict[int,float]:
    win, cap, dwin, dpen = _ab_load()
    tail=_timeline_tail(max(win,dwin))
    if tail:
        # AntiBias: se uma classe domina > cap na janela, reduz levemente sua prob
        # (ex.: 4 saindo demais)
        for c in (1,2,3,4):
            frac = tail[-win:].count(c)/max(1,min(len(tail),win))
            if frac > cap:
                mix[c] *= (1.0 - min(0.12, (frac-cap)*0.6))  # até ~12% de corte
        # Diversidade: evita repetir muitos iguais nos últimos dwin
        last = tail[-dwin:]
        if last:
            streak=1; cur=last[-1]
            for x in reversed(last[:-1]):
                if x==cur: streak+=1
                else: break
            if streak>=3:
                mix[cur]*=max(0.70, 1.0 - dpen*streak)  # penaliza 3+ reps
    return _norm(mix)

# ================== NeuroController ==================
def _ctl_load():
    con=_con(); row=con.execute("SELECT epsilon,min_conf,cool_after_losses,cool_secs,throttle_until,regime FROM neuroctl WHERE id=1").fetchone(); con.close()
    return float(row["epsilon"]), float(row["min_conf"]), int(row["cool_after_losses"]), int(row["cool_secs"]), int(row["throttle_until"]), str(row["regime"])

def _ctl_save(eps=None,minc=None,coolN=None,coolS=None,thr=None,reg=None):
    e,m,cn,cs,t,r=_ctl_load()
    if eps is not None: e=float(eps)
    if minc is not None: m=float(minc)
    if coolN is not None: cn=int(coolN)
    if coolS is not None: cs=int(coolS)
    if thr is not None: t=int(thr)
    if reg is not None: r=str(reg)
    con=_con()
    con.execute("""INSERT OR REPLACE INTO neuroctl(id,epsilon,min_conf,cool_after_losses,cool_secs,throttle_until,regime)
                   VALUES(1,?,?,?,?,?,?)""",(e,m,cn,cs,t,r))
    con.commit(); con.close()

def _ctl_perf():
    con=_con(); row=con.execute("SELECT green,loss,streak_green,streak_loss FROM score WHERE id=1").fetchone(); con.close()
    if not row: return 0,0,0,0
    return int(row["green"]), int(row["loss"]), int(row["streak_green"]), int(row["streak_loss"])

def _ctl_regime_update():
    tail=_timeline_tail(200)
    if len(tail)<50: _ctl_save(reg="neutral"); return "neutral"
    freq=[tail.count(c)/len(tail) for c in (1,2,3,4)]
    ent=0.0
    for p in freq:
        if p>0: ent-=p*math.log(p+1e-12)
    mean=sum(tail)/len(tail); var=sum((x-mean)**2 for x in tail)/len(tail)
    _,_,_,sl=_ctl_perf()
    if ent<1.15 or var>1.70 or sl>=2:
        _ctl_save(reg="volatile"); return "volatile"
    if ent>1.30 and var<1.40:
        _ctl_save(reg="stable"); return "stable"
    _ctl_save(reg="neutral"); return "neutral"

def _ctl_decide(best:int, conf:float, mix:Dict[int,float], gap:float)->Tuple[bool,str,int]:
    now=int(time.time())
    eps, minc, coolN, coolS, thr, reg=_ctl_load()
    g,l,sg,sl=_ctl_perf()
    if now<thr:
        return False, f"cooldown até {thr} (loss_streak={sl})", -1
    regime=_ctl_regime_update()
    dyn=minc
    if regime=="volatile": dyn=min(0.70, max(minc,0.62))
    elif regime=="stable": dyn=max(0.54, minc-0.02)
    if conf<dyn:
        return False, f"conf {conf:.2f} < min {dyn:.2f} (regime={regime})", -1
    # epsilon-explore
    sorted_mix=sorted(mix.items(), key=lambda kv: kv[1], reverse=True)
    second=sorted_mix[1][0]
    if gap<0.03 and random.random()<eps:
        return True, f"explore ε={eps:.2f} (gap={gap:.3f})", int(second)
    return True, f"exploit (gap={gap:.3f}, regime={regime})", int(best)

def _ctl_on_feedback(outcome:str):
    eps,minc,coolN,coolS,thr,reg=_ctl_load()
    g,l,sg,sl=_ctl_perf(); total=g+l
    if total>=50:
        acc=g/max(1,total)
        new_eps=max(0.01,min(0.12,0.10-0.15*(acc-0.50)))
        eps=0.9*eps+0.1*new_eps
    if sl>=2: minc=min(0.74,minc+0.01)
    elif sg>=3: minc=max(0.54,minc-0.01)
    now=int(time.time())
    if sl>=coolN: thr=now+coolS
    elif now>thr: thr=0
    _ctl_save(eps=eps,minc=minc,thr=thr)

# ================== Decisão final ==================
def _neural_decide()->Tuple[int,float,int,Dict[int,float],float,str]:
    tail=_timeline_tail(400)
    p1=_post_e1_ngram(tail); p2=_post_e2_short(tail); p3=_post_e3_long(tail); p4=_post_e4_llm(tail)
    pn=_neural_probs(tail)
    _, wneu, _=_load_neural_params()
    rest=1.0-wneu
    w={"neural":wneu,"e1":0.40*rest,"e2":0.30*rest,"e3":0.20*rest,"e4":0.10*rest}
    mix={c: w["neural"]*pn.get(c,0)+w["e1"]*p1.get(c,0)+w["e2"]*p2.get(c,0)+w["e3"]*p3.get(c,0)+w["e4"]*p4.get(c,0) for c in (1,2,3,4)}
    mix=_conf_floor(_apply_antibias_diversity(_norm(mix)), 0.30, 0.95)
    best=max(mix,key=mix.get); conf=float(mix[best])
    r=sorted(mix.items(), key=lambda kv: kv[1], reverse=True)
    gap=(r[0][1]-r[1][1]) if len(r)>=2 else r[0][1]
    temp, wneu_cur, _=_load_neural_params()
    reason=f"ProfoundSim(w={wneu_cur:.2f},T={temp:.2f})"
    return best, conf, _timeline_size(), mix, gap, reason

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
    nums=[]
    for m in RX_PAREN.finditer(text or ""):
        nums_in=[int(x) for x in RX_NUMS.findall(m.group(1))]
        if nums_in: nums.append(nums_in[-1])
    return nums[-1] if nums else None

# ================== Rotas ==================
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
    win,cap,dw,dp=_ab_load()
    return {
        "MAX_GALE": MAX_GALE, "OBS_TIMEOUT_SEC": OBS_TIMEOUT_SEC, "DEDUP_WINDOW_SEC": DEDUP_WINDOW_SEC,
        "neural_temp": temp, "neural_weight": wneu, "neural_bias": bias,
        "ctl_epsilon": eps, "ctl_min_conf": minc, "ctl_cool_after_losses": coolN,
        "ctl_cool_secs": coolS, "ctl_throttle_until": thr, "ctl_regime": reg,
        "antibias_window": win, "antibias_cap": cap, "div_window": dw, "div_penalty": dp,
        "timeline_samples": _timeline_size()
    }

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
    msg=_pending_close("X","LOSS","G0",suggested)
    return {"ok": True, "forced_close": True, "message": msg}

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
        watchdog=_pending_timeout_check()
        if watchdog and SHOW_DEBUG:
            await tg_send(TARGET_CHANNEL, f"DEBUG: Timeout pendência — fechado automático.\n{watchdog['msg']}")
    except Exception: pass

    if SOURCE_CHANNEL and chat_id and chat_id != SOURCE_CHANNEL:
        if SHOW_DEBUG:
            await tg_send(TARGET_CHANNEL, f"DEBUG: Ignorando chat {chat_id}. Fonte esperada: {SOURCE_CHANNEL}")
        return {"ok": True, "skipped": "wrong_source"}

    # ANALISANDO: só memoria
    if RX_ANALISE.search(text):
        if _seen_recent("analise", _dedupe_key(text)):
            return {"ok": True, "skipped": "analise_dupe"}
        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)
        if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, "DEBUG: Análise reconhecida ✅")
        return {"ok": True, "analise_seq": len(seq)}

    # FECHAMENTO (GREEN/RED/RESULTADO) -> fechar G0 por parênteses
    if RX_FECHA.search(text):
        if _seen_recent("fechamento", _dedupe_key(text)):
            return {"ok": True, "skipped": "fechamento_dupe"}
        pend=_pending_get()
        if pend:
            suggested=int(pend["suggested"] or 0)
            obs=_parse_paren_last_one(text)  # 1..4
            if obs is not None: _pending_seen_set(str(obs))
            seen=(_pending_get()["seen"] or "").strip()
            outcome="LOSS"; stage_lbl="G0"
            if seen.isdigit() and int(seen)==suggested:
                outcome="GREEN"
            final_seen = seen if seen else "X"
            msg_txt=_pending_close(final_seen,outcome,stage_lbl,suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)
            try: _ctl_on_feedback(outcome)
            except: pass
            return {"ok": True, "closed": outcome, "seen": final_seen}
        if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, "DEBUG: Fechamento reconhecido ✅ — sem pendência aberta")
        return {"ok": True, "noted_close": True}

    # ENTRADA
    if RX_ENTRADA.search(text):
        if _seen_recent("entrada", _dedupe_key(text)):
            if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, "DEBUG: Entrada duplicada ignorada.")
            return {"ok": True, "skipped": "entrada_dupe"}
        # memória
        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)
        after=_parse_after(text)

        # fecha pendência esquecida
        pend=_pending_get()
        if pend:
            seen=(pend["seen"] or "").strip()
            final_seen=seen if seen else "X"
            suggested=int(pend["suggested"] or 0)
            outcome="LOSS"; stage_lbl="G0"
            if seen.isdigit() and int(seen)==suggested:
                outcome="GREEN"
            msg_txt=_pending_close(final_seen,outcome,stage_lbl,suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)
            try: _ctl_on_feedback(outcome)
            except: pass

        analyzing_id = await tg_send_return(TARGET_CHANNEL, "⏳ Analisando padrão, aguarde...")

        # decisão neural + especialistas + anti-bias/diversidade
        best, conf, samples, mix, gap, reason = _neural_decide()

        # controller decide operar / escolher 2º em gap pequeno
        play, why, chosen = _ctl_decide(best, conf, mix, gap)
        if not play:
            if analyzing_id is not None: await tg_delete(TARGET_CHANNEL, analyzing_id)
            await tg_send(TARGET_CHANNEL, f"⛔ <b>NeuroController:</b> pulando entrada — {why}")
            return {"ok": True, "skipped_by_controller": True, "why": why}

        suggested = chosen if chosen>0 else best
        opened=_pending_open(suggested)
        if opened:
            aft_txt = f" após {after}" if after else ""
            label = "explore" if (chosen>0 and chosen!=best) else "exploit"
            txt=(f"🤖 <b>IA SUGERE</b> — <b>{suggested}</b>\n"
                 f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
                 f"📊 <b>Conf:</b> {conf*100:.2f}% | <b>Amostra≈</b>{samples} | <b>gap≈</b>{gap*100:.1f}pp\n"
                 f"🧠 <b>Modo:</b> {reason} · <i>{label}</i>\n"
                 f"{_ngram_snapshot(suggested)}")
            await tg_send(TARGET_CHANNEL, txt)
            if analyzing_id is not None: await tg_delete(TARGET_CHANNEL, analyzing_id)
            if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, f"DEBUG: Entrada aberta (controller: {why}).")
            return {"ok": True, "entry_opened": True, "best": suggested, "conf": conf, "why": why}
        else:
            if analyzing_id is not None: await tg_delete(TARGET_CHANNEL, analyzing_id)
            if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, "DEBUG: pending já aberto; entrada ignorada.")
            return {"ok": True, "skipped": "pending_open"}

    # não reconhecido
    if SHOW_DEBUG:
        await tg_send(TARGET_CHANNEL, "DEBUG: Mensagem não reconhecida como ENTRADA/FECHAMENTO/ANALISANDO.")
    return {"ok": True, "skipped": "unmatched"}