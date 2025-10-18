# fibo_experts.py
# -*- coding: utf-8 -*-
"""
Fibonacci Experts — 5 especialistas + mixer
Projetado para ser importado pelo webhook_app.py
Foco: estabilidade (sem travar), diversidade, e ajuste leve pró-acertividade.

APIs públicas:
- fibo_mix_probs(tail: List[int]) -> Dict[int, float]
  (retorna um dict normalizado {1..4: prob})

Notas:
- Não usa estado global, não trava em cauda pequena.
- Janela padrão: [8, 13, 21, 34, 55]
"""

from typing import List, Dict
import math

CANDS = (1, 2, 3, 4)
FIBO_WINDOWS = (8, 13, 21, 34, 55)

def _norm(d: Dict[int, float]) -> Dict[int, float]:
    s = float(sum(d.values())) or 1e-12
    return {k: float(d.get(k, 0.0))/s for k in CANDS}

def _freq_in_window(tail: List[int], k: int) -> Dict[int, float]:
    if not tail:
        return {c: 0.25 for c in CANDS}
    win = tail[-k:] if len(tail) >= k else tail
    tot = max(1, len(win))
    return {c: win.count(c) / tot for c in CANDS}

def _entropy(p: Dict[int, float]) -> float:
    e = 0.0
    for c in CANDS:
        pc = max(1e-12, float(p.get(c, 0.0)))
        e -= pc * math.log(pc)
    # log base natural; normaliza por log(4) para ~[0,1]
    return e / math.log(4.0)

# -------------------------
# Especialista 1 — Majority Fibo (maiorias por janela)
# -------------------------
def _e_fibo_majority(tail: List[int]) -> Dict[int, float]:
    if not tail:
        return {c: 0.25 for c in CANDS}
    scores = {c: 0.0 for c in CANDS}
    for k in FIBO_WINDOWS:
        freq = _freq_in_window(tail, k)
        # quem lidera nesta janela leva ponto
        leader = max(CANDS, key=lambda c: freq[c])
        scores[leader] += 1.0
    # transforma contagem em prob (com suavização)
    sm = sum(scores.values()) or 1.0
    base = {c: (scores[c] / sm) for c in CANDS}
    return _norm(base)

# -------------------------
# Especialista 2 — Weighted Fibo (média ponderada)
# -------------------------
def _e_fibo_weighted(tail: List[int]) -> Dict[int, float]:
    if not tail:
        return {c: 0.25 for c in CANDS}
    # pesos crescentes para janelas maiores (tendem a ser mais estáveis)
    weights = {8: 0.15, 13: 0.18, 21: 0.20, 34: 0.22, 55: 0.25}
    out = {c: 0.0 for c in CANDS}
    for k in FIBO_WINDOWS:
        fk = _freq_in_window(tail, k)
        w = weights[k]
        for c in CANDS:
            out[c] += w * fk[c]
    return _norm(out)

# -------------------------
# Especialista 3 — Turning-Point (busca reversões)
# Favorece classe diferente do último se dominância curta for alta
# -------------------------
def _e_fibo_turning(tail: List[int]) -> Dict[int, float]:
    if not tail:
        return {c: 0.25 for c in CANDS}
    last = tail[-1]
    short = _freq_in_window(tail, 8)
    # quanto mais dominante o último no curto prazo, maior o impulso de reversão
    dom = short[last]
    out = {c: 0.0 for c in CANDS}
    for c in CANDS:
        if c == last:
            out[c] = 0.20 * (1.0 - dom)  # segura o mesmo número
        else:
            out[c] = 0.20 + 0.80 * (dom / 3.0)  # espalha em outros
    return _norm(out)

# -------------------------
# Especialista 4 — Anti-Streak (penaliza sequência longa do mesmo número)
# -------------------------
def _e_fibo_antistreak(tail: List[int]) -> Dict[int, float]:
    if not tail:
        return {c: 0.25 for c in CANDS}
    # calcula streak do último número
    last = tail[-1]
    s = 0
    for x in reversed(tail):
        if x == last:
            s += 1
        else:
            break
    # frequência média para base
    base = _freq_in_window(tail, 21)
    # penalização leve proporcional ao streak
    pen = min(0.35, 0.05 * max(0, s - 1))
    out = {}
    for c in CANDS:
        if c == last:
            out[c] = max(0.0, base[c] * (1.0 - pen))
        else:
            out[c] = base[c] + (pen / 3.0)
    return _norm(out)

# -------------------------
# Especialista 5 — Entropy Guard (puxa p/ uniforme quando entropia baixa)
# -------------------------
def _e_fibo_entropy_guard(tail: List[int]) -> Dict[int, float]:
    if not tail:
        return {c: 0.25 for c in CANDS}
    ref = _freq_in_window(tail, 34)
    H = _entropy(ref)  # 0..1
    # quanto mais baixa a entropia, mais puxamos p/ uniforme
    # empurra 0.0..0.25 da massa, suavizando dominâncias
    push = max(0.0, 0.25 * (1.0 - H))
    uni = {c: 0.25 for c in CANDS}
    out = {c: (1.0 - push) * ref[c] + push * uni[c] for c in CANDS}
    return _norm(out)

# -------------------------
# Mixer — combinação dos 5 especialistas
# -------------------------
def fibo_mix_probs(tail: List[int]) -> Dict[int, float]:
    """
    Combina 5 especialistas de forma robusta.
    Pesos calibrados para não travar no mesmo número,
    mas sem destruir sinal quando houver padrão estável.
    """
    e1 = _e_fibo_majority(tail)
    e2 = _e_fibo_weighted(tail)
    e3 = _e_fibo_turning(tail)
    e4 = _e_fibo_antistreak(tail)
    e5 = _e_fibo_entropy_guard(tail)

    # pesos (soma 1.0)
    w1, w2, w3, w4, w5 = 0.24, 0.28, 0.16, 0.16, 0.16

    mix = {c: 0.0 for c in CANDS}
    for c in CANDS:
        mix[c] = (
            w1 * e1[c] +
            w2 * e2[c] +
            w3 * e3[c] +
            w4 * e4[c] +
            w5 * e5[c]
        )

    # piso/teto leve para evitar “grudar” ou “zeroar”
    mix = _conf_floor(mix, floor=0.28, cap=0.92)
    return mix

def _conf_floor(post: Dict[int, float], floor=0.30, cap=0.95) -> Dict[int, float]:
    post = _norm({c: float(post.get(c, 0.0)) for c in CANDS})
    b = max(CANDS, key=lambda c: post[c])
    mx = post[b]
    if mx < floor:
        others = [c for c in CANDS if c != b]
        s = sum(post[c] for c in others)
        take = min(floor - mx, s)
        if s > 0:
            scale = (s - take) / s
            for c in others:
                post[c] *= scale
        post[b] = min(cap, mx + take)
    if post[b] > cap:
        ex = post[b] - cap
        post[b] = cap
        add = ex / 3.0
        for c in CANDS:
            if c != b:
                post[c] += add
    return _norm(post)

# -------------------------
# Teste rápido (opcional)
# -------------------------
if __name__ == "__main__":
    demo = [1,2,3,4,4,4,2,1,3,2,1,4,4,3,2,1,2,2,3,4,1,1,1,2,3,4,2,3,1,4]
    p = fibo_mix_probs(demo)
    print("FIBO mix:", p)