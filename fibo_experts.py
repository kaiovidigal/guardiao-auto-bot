# fibo_experts.py
# Conjunto de especialistas Fibonacci para sequência 1..4

from typing import Dict, List

def _norm(d: Dict[int, float]) -> Dict[int, float]:
    s = sum(d.values()) or 1e-9
    return {k: v / s for k, v in d.items()}

def _freq_window(tail: List[int], k: int) -> Dict[int, float]:
    if not tail:
        return {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
    win = tail[-k:] if len(tail) >= k else tail
    tot = max(1, len(win))
    return _norm({c: win.count(c) / tot for c in (1, 2, 3, 4)})

def _blend(ws: List[int], weights: List[float], tail: List[int]) -> Dict[int, float]:
    out = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    for k, w in zip(ws, weights):
        f = _freq_window(tail, k)
        for c in (1, 2, 3, 4):
            out[c] += w * f.get(c, 0.0)
    return _norm(out)

# 5 especialistas Fibo (diferentes horizontes/ênfases)
def fibo_A(tail: List[int]) -> Dict[int, float]:
    # curto agressivo
    return _blend([5, 8, 13], [0.45, 0.35, 0.20], tail)

def fibo_B(tail: List[int]) -> Dict[int, float]:
    # curto-médio
    return _blend([8, 13, 21], [0.35, 0.35, 0.30], tail)

def fibo_C(tail: List[int]) -> Dict[int, float]:
    # médio
    return _blend([13, 21, 34], [0.30, 0.35, 0.35], tail)

def fibo_D(tail: List[int]) -> Dict[int, float]:
    # médio-longo
    return _blend([21, 34, 55], [0.25, 0.35, 0.40], tail)

def fibo_E(tail: List[int]) -> Dict[int, float]:
    # longo
    return _blend([34, 55, 89], [0.20, 0.35, 0.45], tail)

def fibo_ensemble_probs(tail: List[int]) -> Dict[int, float]:
    """Ensemble dos 5 fibos com pesos levemente pró-médio (robustez)."""
    pA = fibo_A(tail); pB = fibo_B(tail); pC = fibo_C(tail); pD = fibo_D(tail); pE = fibo_E(tail)
    wA, wB, wC, wD, wE = 0.15, 0.20, 0.30, 0.20, 0.15
    mix = {c: wA*pA[c] + wB*pB[c] + wC*pC[c] + wD*pD[c] + wE*pE[c] for c in (1,2,3,4)}
    return _norm(mix)
