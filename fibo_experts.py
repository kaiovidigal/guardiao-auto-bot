# fibo_experts.py
# v2.0 — Especialistas Fibo (single/pair/triple) + IA reguladora embutida (EXP3/softmax)
# - tudo num arquivo só
# - persiste pesos/valência em SQLite (tabela mc_meta)
# - interface:
#     fibo_votes(tail, db_path, config) -> (probs, diag)
#     mc_feedback(db_path, outcome, suggested, expert_probs)

from typing import Dict, List, Tuple, Optional
import math, sqlite3, json, time, collections

CANDS = (1,2,3,4)

# =========================
#   META-CONTROLLER (IA)
# =========================
EXPERTS = ("single","pair","triple")
DEFAULTS = {
    "ema": 0.90,            # suavização da recompensa
    "lr":  0.12,            # passo de atualização de peso
    "temp": 0.85,           # temperatura do softmax (menor => mais confiante)
    "w_min": 0.15,          # piso p/ cada expert
    "w_max": 0.70,          # teto p/ cada expert
}

def _con(db_path:str):
    con = sqlite3.connect(db_path, check_same_thread=False, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

def _mc_init(db_path:str):
    con=_con(db_path); cur=con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS mc_meta(
        k TEXT PRIMARY KEY, v TEXT
    )""")
    if not cur.execute("SELECT 1 FROM mc_meta WHERE k='weights'").fetchone():
        cur.execute("INSERT INTO mc_meta(k,v) VALUES('weights','{\"single\":0.34,\"pair\":0.33,\"triple\":0.33}')")
    if not cur.execute("SELECT 1 FROM mc_meta WHERE k='valence'").fetchone():
        cur.execute("INSERT INTO mc_meta(k,v) VALUES('valence','{\"single\":0.5,\"pair\":0.5,\"triple\":0.5}')")
    con.commit(); con.close()

def _mc_get(db_path:str)->Tuple[Dict[str,float], Dict[str,float]]:
    _mc_init(db_path)
    con=_con(db_path)
    w=json.loads(con.execute("SELECT v FROM mc_meta WHERE k='weights'").fetchone()["v"])
    q=json.loads(con.execute("SELECT v FROM mc_meta WHERE k='valence'").fetchone()["v"])
    con.close()
    s=sum(w.values()) or 1.0
    w={k:max(1e-9,float(v)/s) for k,v in w.items()}
    return w,q

def _mc_save(db_path:str, w:Dict[str,float], q:Dict[str,float]):
    con=_con(db_path)
    con.execute("INSERT OR REPLACE INTO mc_meta(k,v) VALUES('weights',?)",[json.dumps(w)])
    con.execute("INSERT OR REPLACE INTO mc_meta(k,v) VALUES('valence',?)",[json.dumps(q)])
    con.commit(); con.close()

def _softmax(d:Dict[str,float], temp:float)->Dict[str,float]:
    if not d: return {}
    m=max(d.values())
    ex={k: math.exp((v-m)/max(0.15,temp)) for k,v in d.items()}
    s=sum(ex.values()) or 1.0
    return {k: ex[k]/s for k in d.keys()}

def mc_mix_weights(db_path:str, cfg:Dict=None)->Dict[str,float]:
    if cfg is None: cfg={}
    temp=float(cfg.get("temp", DEFAULTS["temp"]))
    lr  =float(cfg.get("lr",  DEFAULTS["lr"])) * 0.5
    w,q=_mc_get(db_path)
    ps=_softmax(q, temp)
    mixed={k: (1-lr)*w[k] + lr*ps[k] for k in EXPERTS}
    # clip + renorm
    wmin=float(cfg.get("w_min", DEFAULTS["w_min"]))
    wmax=float(cfg.get("w_max", DEFAULTS["w_max"]))
    mixed={k: min(wmax, max(wmin, mixed[k])) for k in EXPERTS}
    s=sum(mixed.values()) or 1.0
    mixed={k: mixed[k]/s for k in EXPERTS}
    return mixed

def mc_feedback(db_path:str,
                outcome:str,
                suggested:int,
                expert_probs:Dict[str,Dict[int,float]],
                cfg:Dict=None):
    if cfg is None: cfg={}
    ema=float(cfg.get("ema", DEFAULTS["ema"]))
    lr =float(cfg.get("lr",  DEFAULTS["lr"]))
    temp=float(cfg.get("temp", DEFAULTS["temp"]))
    wmin=float(cfg.get("w_min", DEFAULTS["w_min"]))
    wmax=float(cfg.get("w_max", DEFAULTS["w_max"]))
    reward = 1.0 if str(outcome).upper()=="GREEN" else 0.0
    w,q=_mc_get(db_path)

    # crédito proporcional à prob. dada AO NÚMERO SUGERIDO
    cred={k: float(expert_probs.get(k,{}).get(suggested, 0.25)) for k in EXPERTS}
    tot=sum(cred.values()) or 1.0
    for k in EXPERTS:
        share=cred[k]/tot
        q[k]= ema*q[k] + (1-ema)*(reward*share)

    ps=_softmax(q, temp)
    for k in EXPERTS:
        w[k]=(1-lr)*w[k] + lr*ps[k]
    w={k: min(wmax, max(wmin, w[k])) for k in EXPERTS}
    s=sum(w.values()) or 1.0
    w={k: w[k]/s for k in EXPERTS}
    _mc_save(db_path, w, q)

# =========================
#   UTILs
# =========================
def _norm_map(m:Dict[int,float])->Dict[int,float]:
    s=sum(m.values()) or 1e-9
    return {k: v/s for k,v in m.items()}

def _entropy_mix(p:Dict[int,float], push:float=0.0)->Dict[int,float]:
    if push<=0: return dict(p)
    uni={c: 1.0/len(CANDS) for c in CANDS}
    out={c: (1.0-push)*p.get(c,0.0) + push*uni[c] for c in CANDS}
    return _norm_map(out)

def _freq(vec:List[int], k:int)->Dict[int,float]:
    if not vec: return {c:1.0/len(CANDS) for c in CANDS}
    win=vec[-k:] if len(vec)>=k else vec
    tot=max(1,len(win))
    cnt=collections.Counter(win)
    return _norm_map({c: cnt.get(c,0)/tot for c in CANDS})

# =========================
#   ESPECIALISTAS FIBO
# =========================
def _e_single(tail:List[int])->Dict[int,float]:
    # “Fibo-like” janelas: 8, 13, 21, 34, 55
    mix={c:0.0 for c in CANDS}
    for k,w in ((8,0.22),(13,0.18),(21,0.20),(34,0.20),(55,0.20)):
        fk=_freq(tail, k)
        for c in CANDS: mix[c]+= w*fk[c]
    return _norm_map(mix)

def _e_pair(tail:List[int])->Dict[int,float]:
    # pares mais recentes (peso maior pros últimos). Converte para prob do próximo número
    if len(tail)<2: return {c:1/4 for c in CANDS}
    pair_counts={(a,b):0 for a in CANDS for b in CANDS}
    # janela crescente Fibo-like
    for k,w in ((21,0.30),(34,0.35),(55,0.35)):
        win=tail[-k:] if len(tail)>=k else tail
        for a,b in zip(win[:-1],win[1:]):
            pair_counts[(a,b)]+=w
    # prob do “b” virar “próximo c” — heurística Markov 1ª ordem
    # vamos somar evidências do último par observado
    last_pair=tuple(tail[-2:])
    scores={c:0.0 for c in CANDS}
    for c in CANDS:
        scores[c]+= pair_counts.get((last_pair[1], c), 0.0)
    # fallback leve com frequência simples
    base=_e_single(tail)
    for c in CANDS: scores[c]+=0.15*base[c]
    return _norm_map(scores)

def _e_triple(tail:List[int])->Dict[int,float]:
    # trincas mais frequentes em múltiplas janelas + suavização
    if len(tail)<3: return _e_pair(tail)
    tri_counts={(a,b,c):0 for a in CANDS for b in CANDS for c in CANDS}
    for k,w in ((34,0.35),(55,0.35),(89,0.30)):
        win=tail[-k:] if len(tail)>=k else tail
        for a,b,c in zip(win[:-2],win[1:-1],win[2:]):
            tri_counts[(a,b,c)]+=w
    last2=tuple(tail[-2:])
    scores={c:0.0 for c in CANDS}
    for c in CANDS:
        scores[c]+= tri_counts.get((last2[0],last2[1],c), 0.0)
    base=_e_single(tail)
    for c in CANDS: scores[c]+=0.20*base[c]
    return _norm_map(scores)

# =========================
#   BLOCO FIBO + IA
# =========================
def fibo_votes(tail:List[int],
               db_path:str,
               config:Dict=None) -> Tuple[Dict[int,float], Dict]:
    """
    Retorna:
      - fibo_probs: distribuição final (mistura single/pair/triple com IA reguladora)
      - diag: { 'single':{1..4}, 'pair':{1..4}, 'triple':{1..4}, 'mc_weights':{...} }
    """
    if config is None: config={}
    # especialistas “crus”
    p_single=_e_single(tail)
    p_pair  =_e_pair(tail)
    p_triple=_e_triple(tail)

    # pesos dinâmicos do controlador
    mcw=mc_mix_weights(db_path, cfg={
        "temp": float(config.get("MC_TEMP", 0.85)),
        "lr":   float(config.get("MC_LR",   0.12)),
        "w_min":float(config.get("MC_WMIN", 0.15)),
        "w_max":float(config.get("MC_WMAX", 0.70)),
    })

    # mistura
    out={c: mcw["single"]*p_single[c] + mcw["pair"]*p_pair[c] + mcw["triple"]*p_triple[c] for c in CANDS}

    # empurrão de entropia (evita colar num único número)
    push=float(config.get("ENTROPY_PUSH", 0.10))
    out=_entropy_mix(out, push=push)

    return _norm_map(out), {
        "single": p_single,
        "pair":   p_pair,
        "triple": p_triple,
        "mc_weights": mcw
    }