# fibo_experts.py
# Fibo (curto) + Combos + Fibo Adaptativa + Clock 30m (FanTan 1..4)
# 100% stateless: só lê a cauda (tail) e o relógio UTC (30 min)

from typing import Dict, List, Tuple
import datetime
import math

CANDS = (1, 2, 3, 4)

# --------- utils ---------
def _norm(d: Dict[int, float]) -> Dict[int, float]:
    s = sum(max(0.0, v) for v in d.values()) or 1e-9
    return {k: max(0.0, v) / s for k, v in d.items()}

def _freq(tail: List[int], k: int) -> Dict[int, float]:
    if not tail: return {c: 0.25 for c in CANDS}
    win = tail[-k:] if len(tail) >= k else tail
    tot = max(1, len(win))
    return _norm({c: win.count(c) / tot for c in CANDS})

# --------- Fibo curto ----------
def fibo10_base(tail: List[int]) -> Dict[int, float]:
    """
    Fibo curto: vota no mais frequente em janelas curtas (sem longo).
    """
    if not tail: return {c: 0.25 for c in CANDS}
    # janelas curtas estilo fibo (sem "longa"):
    wins = (10, 21, 34, 55)
    mix = {c: 0.0 for c in CANDS}
    for k in wins:
        win = tail[-k:] if len(tail) >= k else tail
        best = max(CANDS, key=lambda c: win.count(c))
        mix[best] += 1.0
    return _norm(mix)

# --------- Combos ----------
def combo2_probs(tail: List[int]) -> Dict[int, float]:
    """
    P(n | a,b) com a=penúltimo, b=último; Laplace +1 (curto, local).
    """
    if len(tail) < 2: return {c: 0.25 for c in CANDS}
    a, b = tail[-2], tail[-1]
    counts = {c: 1.0 for c in CANDS}
    for i in range(len(tail) - 2):
        if tail[i] == a and tail[i + 1] == b:
            nxt = tail[i + 2]
            if nxt in CANDS: counts[nxt] += 1.0
    return _norm(counts)

def combo3_probs(tail: List[int]) -> Dict[int, float]:
    """
    P(n | a,b,c) com (a,b,c)=últimos 3; Laplace +1 (curto, local).
    """
    if len(tail) < 3: return {c: 0.25 for c in CANDS}
    a, b, c = tail[-3], tail[-2], tail[-1]
    counts = {x: 1.0 for x in CANDS}
    for i in range(len(tail) - 3):
        if tail[i] == a and tail[i + 1] == b and tail[i + 2] == c:
            nxt = tail[i + 3]
            if nxt in CANDS: counts[nxt] += 1.0
    return _norm(counts)

# --------- Clock 30m ----------
def clock30_probs() -> Dict[int, float]:
    """
    Pseudo-estacional de 30 minutos (UTC), leve e estateless.
    """
    m = int(datetime.datetime.utcnow().timestamp() // 1800)  # bucket de 30m
    vals = [(m * (i + 1) * 17) % 100 for i in range(4)]
    s = sum(vals) or 1
    return {i + 1: vals[i] / s for i in range(4)}

# --------- Fibo Adaptativa (curta) ----------
def fibo_adaptativa(
    tail: List[int],
    alpha2: float = 0.55,
    alpha3: float = 0.75,
    damp_disagree: float = 0.75
) -> Dict[int, float]:
    """
    Parte da Fibo curto e reforça quando Combo2/3 CONFIRMAM.
    Reduz levemente quando discordam (anti-virada imediata).
    (curta, sem janelas longas)
    """
    fb = fibo10_base(tail)
    c2 = combo2_probs(tail)
    c3 = combo3_probs(tail)

    out = {}
    agree_top2 = max(c2, key=c2.get)
    agree_top3 = max(c3, key=c3.get)

    for c in CANDS:
        base = fb.get(c, 0.0)

        # reforços multiplicativos acima do uniforme
        boost = 1.0
        boost *= (1.0 + alpha2 * (c2.get(c, 0.0) - 0.25))
        boost *= (1.0 + alpha3 * (c3.get(c, 0.0) - 0.25))

        # se combos apontam outro dígito, amortecer um pouco
        if agree_top2 != c and agree_top3 != c:
            boost *= damp_disagree

        out[c] = base * max(0.10, boost)

    return _norm(out)

# --------- pacote curto (sem “longa”) ----------
def all_experts_short(tail: List[int]) -> Dict[str, Dict[int, float]]:
    """
    Retorna apenas especialistas curtos + clock de 30 minutos.
    (remove mid/long)
    """
    short20 = _freq(tail, 20)   # curto
    short40 = _freq(tail, 40)   # curto+
    ngram_proxy = _norm({
        c: 0.25 * _freq(tail, 8)[c]
         + 0.35 * _freq(tail, 21)[c]
         + 0.40 * _freq(tail, 55)[c]
        for c in CANDS
    })
    return {
        "fibo10": fibo10_base(tail),
        "combo2": combo2_probs(tail),
        "combo3": combo3_probs(tail),
        "fibo_adapt": fibo_adaptativa(tail),
        "short20": short20,
        "short40": short40,
        "ngram_proxy": ngram_proxy,
        "clock30": clock30_probs(),
    }

# --------- mistura pronta (curta + 30m) ----------
def fibo_short_suite(
    tail: List[int],
    w_fibo: float = 0.40,
    w_combo: float = 0.30,   # 2/3
    w_adapt: float = 0.15,
    w_ngram: float = 0.07,
    w_short: float = 0.03,   # short20/40
    w_clk30: float = 0.05
) -> Dict[int, float]:
    """
    Mistura pronta SÓ com curto + clock 30m.
    Sem mid/long. Valores default são conservadores.
    """
    ex = all_experts_short(tail)
    c2, c3 = ex["combo2"], ex["combo3"]
    combos = {c: 0.55 * c2[c] + 0.45 * c3[c] for c in CANDS}
    short_mix = {c: 0.5 * ex["short20"][c] + 0.5 * ex["short40"][c] for c in CANDS}

    mix = {}
    for c in CANDS:
        mix[c] = (
            w_fibo  * ex["fibo10"][c]    +
            w_combo * combos[c]          +
            w_adapt * ex["fibo_adapt"][c]+
            w_ngram * ex["ngram_proxy"][c]+
            w_short * short_mix[c]       +
            w_clk30 * ex["clock30"][c]
        )
    return _norm(mix)