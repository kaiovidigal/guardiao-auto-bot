# fibo_experts.py
# Fibo10 + Combos + Fibo Adaptativa (FanTan 1..4)
# 100% stateless, só lê a cauda (tail) de números

from typing import Dict, List
import math

CANDS = (1, 2, 3, 4)

def _norm(d: Dict[int, float]) -> Dict[int, float]:
    s = sum(max(0.0, v) for v in d.values()) or 1e-9
    return {k: max(0.0, v) / s for k, v in d.items()}

def _freq(tail: List[int], k: int) -> Dict[int, float]:
    if not tail: return {c: 0.25 for c in CANDS}
    win = tail[-k:] if len(tail) >= k else tail
    tot = max(1, len(win))
    return _norm({c: win.count(c) / tot for c in CANDS})

# ---------- Básicos ----------
def fibo10_base(tail: List[int]) -> Dict[int, float]:
    """Vota no mais frequente em janelas fibo curtas."""
    if not tail: return {c: 0.25 for c in CANDS}
    wins = (10, 21, 34, 55)
    mix = {c: 0.0 for c in CANDS}
    for k in wins:
        win = tail[-k:] if len(tail) >= k else tail
        best = max(CANDS, key=lambda c: win.count(c))
        mix[best] += 1.0
    return _norm(mix)

def combo2_probs(tail: List[int]) -> Dict[int, float]:
    """P(n | a,b) com a=penúltimo, b=último; Laplace +1."""
    if len(tail) < 2: return {c: 0.25 for c in CANDS}
    a, b = tail[-2], tail[-1]
    counts = {c: 1.0 for c in CANDS}
    for i in range(len(tail) - 2):
        if tail[i] == a and tail[i + 1] == b:
            nxt = tail[i + 2]
            if nxt in CANDS: counts[nxt] += 1.0
    return _norm(counts)

def combo3_probs(tail: List[int]) -> Dict[int, float]:
    """P(n | a,b,c) com (a,b,c)=últimos 3; Laplace +1."""
    if len(tail) < 3: return {c: 0.25 for c in CANDS}
    a, b, c = tail[-3], tail[-2], tail[-1]
    counts = {x: 1.0 for x in CANDS}
    for i in range(len(tail) - 3):
        if tail[i] == a and tail[i + 1] == b and tail[i + 2] == c:
            nxt = tail[i + 3]
            if nxt in CANDS: counts[nxt] += 1.0
    return _norm(counts)

# ---------- Fibo Adaptativa ----------
def fibo_adaptativa(tail: List[int],
                    alpha2: float = 0.55,
                    alpha3: float = 0.75,
                    damp_disagree: float = 0.75) -> Dict[int, float]:
    """
    Parte da Fibo10 e 'dobra' o peso quando Combo2/3 CONFIRMAM,
    e reduz levemente quando discordam (anti-virada imediata).
    - alpha3 > alpha2 porque triplo tem evidência mais forte.
    """
    fb = fibo10_base(tail)
    c2 = combo2_probs(tail)
    c3 = combo3_probs(tail)

    out = {}
    for c in CANDS:
        base = fb.get(c, 0.0)

        # reforços (ganho multiplicativo) se combos concordam
        boost = 1.0
        boost *= (1.0 + alpha2 * (c2.get(c, 0.0) - 0.25))  # acima de uniforme => reforça
        boost *= (1.0 + alpha3 * (c3.get(c, 0.0) - 0.25))

        # se ambos os combos concentram em OUTRO dígito, amortecer este
        agree_top2 = max(c2, key=c2.get)
        agree_top3 = max(c3, key=c3.get)
        if agree_top2 != c and agree_top3 != c:
            boost *= damp_disagree

        out[c] = base * max(0.10, boost)

    return _norm(out)

# ---------- pacote útil ----------
def all_experts(tail: List[int]) -> Dict[str, Dict[int, float]]:
    return {
        "fibo10": fibo10_base(tail),
        "combo2": combo2_probs(tail),
        "combo3": combo3_probs(tail),
        "fibo_adapt": fibo_adaptativa(tail),
        # extras simples para quem usa:
        "short40": _freq(tail, 40),
        "mid120": _freq(tail, 120),
        "long300": _freq(tail, 300),
        "ngram_proxy": _norm({
            c: 0.25 * _freq(tail, 8)[c]
             + 0.35 * _freq(tail, 21)[c]
             + 0.40 * _freq(tail, 55)[c]
            for c in CANDS
        })
    }