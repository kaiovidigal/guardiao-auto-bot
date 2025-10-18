# fibo_experts.py
# v2.1 — Fibo + Combinações (2/3 dígitos) + Reversão de sequência + Anti-viés
# -----------------------------------------------------------------------------
# O que este módulo faz:
# - Frequência por janelas Fibo (10/21/34/55) do número isolado (1..4)
# - Transições de 1ª ordem (par: prev -> next)  => combinações de 2 dígitos
# - Transições de 2ª ordem (par duplo: prev2,prev1 -> next) => combinações de 3 dígitos
# - Blend ponderado entre os três modelos (single + pair + triple)
# - Anti-viés: corta dominância recente e empurra entropia
# - Reversão simples: se há streak longo do mesmo número, reduz prob. de repetir
#
# Como usar a partir do webhook:
#   from fibo_experts import fibo_votes
#   probs, diag = fibo_votes(tail, config={...})
#   # probs é {1: p1, 2: p2, 3: p3, 4: p4} somando ~1.0
#   # diag traz detalhes pra debug/log
#
# Parâmetros principais (config):
#   K_SINGLE      : janelas Fibo p/ modelo single (default (10,21,34,55))
#   K_PAIR        : janelas Fibo p/ pares (default (10,21,34,55))
#   K_TRIPLE      : janelas Fibo p/ triplas (default (10,21,34,55))
#   WEIGHTS       : pesos do blend {'single':0.4,'pair':0.35,'triple':0.25}
#   ENTROPY_PUSH  : empurra p/ uniforme (0..0.3 tipicamente)
#   MAX_SHARE_CAP : teto p/ dominância após anti-vies (ex: 0.95)
#   FLOOR_CAP     : piso mínimo p/ top-1 após floor (ex: 0.30)
#   STREAK_REV_N  : se o mesmo número repetir >= N, aplica reversão
#   STREAK_REV_CUT: fator de corte na prob. do número em streak (ex: 0.55)
#
# -----------------------------------------------------------------------------

from typing import Dict, List, Tuple
import math

CANDS = (1, 2, 3, 4)

# -------------------------
# Helpers básicos
# -------------------------
def _norm(d: Dict[int, float]) -> Dict[int, float]:
    s = float(sum(d.values())) or 1e-9
    return {k: float(v) / s for k, v in d.items()}

def _zeros() -> Dict[int, float]:
    return {c: 0.0 for c in CANDS}

def _cap_floor_mix(post: Dict[int, float], floor=0.30, cap=0.95) -> Dict[int, float]:
    post = _norm({c: float(post.get(c, 0.0)) for c in CANDS})
    b = max(post, key=post.get)
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
        add = ex / (len(CANDS) - 1)
        for c in CANDS:
            if c != b:
                post[c] += add
    return _norm(post)

def _entropy_push(post: Dict[int, float], alpha: float) -> Dict[int, float]:
    if alpha <= 0: 
        return _norm(post)
    uni = {c: 1.0 / len(CANDS) for c in CANDS}
    return _norm({c: (1 - alpha) * post.get(c, 0.0) + alpha * uni[c] for c in CANDS})

def _streak_of_last(tail: List[int]) -> int:
    if not tail:
        return 0
    last = tail[-1]
    s = 0
    for x in reversed(tail):
        if x == last:
            s += 1
        else:
            break
    return s

# -------------------------
# Modelo 1: SINGLE (frequência isolada)
# -------------------------
def _single_window_probs(win: List[int]) -> Dict[int, float]:
    out = _zeros()
    if not win:
        return {c: 1.0 / 4.0 for c in CANDS}
    L = len(win)
    for c in CANDS:
        out[c] = sum(1 for x in win if x == c) / float(L)
    return _norm(out)

def _single_fibo(tail: List[int], k_list=(10, 21, 34, 55), weights=(0.15, 0.25, 0.30, 0.30)) -> Dict[int, float]:
    mix = _zeros()
    for k, w in zip(k_list, weights):
        win = tail[-k:] if len(tail) >= k else tail
        p = _single_window_probs(win)
        for c in CANDS:
            mix[c] += w * p[c]
    return _norm(mix)

# -------------------------
# Modelo 2: PAIR (1ª ordem) — P(next | prev1)
# -------------------------
def _pair_window_probs(win: List[int], last: int) -> Dict[int, float]:
    # conta transições prev -> next dentro da janela
    out_counts = _zeros()
    total = 0
    for i in range(len(win) - 1):
        prev1 = win[i]
        nxt = win[i + 1]
        if prev1 == last:
            out_counts[nxt] += 1.0
            total += 1
    if total == 0:
        # sem dados do último estado -> fallback uniforme
        return {c: 1.0 / 4.0 for c in CANDS}
    return _norm(out_counts)

def _pair_fibo(tail: List[int], k_list=(10, 21, 34, 55), weights=(0.15, 0.25, 0.30, 0.30)) -> Dict[int, float]:
    if len(tail) < 1:
        return {c: 1.0 / 4.0 for c in CANDS}
    last = tail[-1]
    mix = _zeros()
    for k, w in zip(k_list, weights):
        win = tail[-k:] if len(tail) >= k else tail
        p = _pair_window_probs(win, last)
        for c in CANDS:
            mix[c] += w * p[c]
    return _norm(mix)

# -------------------------
# Modelo 3: TRIPLE (2ª ordem) — P(next | prev2, prev1)
# -------------------------
def _triple_window_probs(win: List[int], prev2: int, prev1: int) -> Dict[int, float]:
    out_counts = _zeros()
    total = 0
    for i in range(len(win) - 2):
        a, b, nxt = win[i], win[i + 1], win[i + 2]
        if a == prev2 and b == prev1:
            out_counts[nxt] += 1.0
            total += 1
    if total == 0:
        return {c: 1.0 / 4.0 for c in CANDS}
    return _norm(out_counts)

def _triple_fibo(tail: List[int], k_list=(10, 21, 34, 55), weights=(0.15, 0.25, 0.30, 0.30)) -> Dict[int, float]:
    if len(tail) < 2:
        return {c: 1.0 / 4.0 for c in CANDS}
    prev2, prev1 = tail[-2], tail[-1]
    mix = _zeros()
    for k, w in zip(k_list, weights):
        win = tail[-k:] if len(tail) >= k else tail
        p = _triple_window_probs(win, prev2, prev1)
        for c in CANDS:
            mix[c] += w * p[c]
    return _norm(mix)

# -------------------------
# Reversão de streak e anti-viés
# -------------------------
def _apply_streak_reversal(post: Dict[int, float],
                           tail: List[int],
                           streak_n: int = 3,
                           cut: float = 0.55) -> Dict[int, float]:
    """
    Se o mesmo número repetiu >= streak_n no histórico recente,
    reduz a probabilidade de repetir de novo (reversão básica).
    """
    if not tail:
        return _norm(post)
    s = _streak_of_last(tail)
    if s >= streak_n:
        last = tail[-1]
        adj = dict(post)
        adj[last] *= max(0.0, float(cut))
        return _norm(adj)
    return _norm(post)

def _anti_bias(post: Dict[int, float],
               dominant_cap: float = 0.95,
               entropy_push: float = 0.12) -> Dict[int, float]:
    """
    1) Satura o top em 'dominant_cap'
    2) Empurra leve pra uniforme (entropy_push)
    """
    post = _cap_floor_mix(post, floor=0.30, cap=dominant_cap)
    post = _entropy_push(post, entropy_push)
    return _norm(post)

# -------------------------
# Ensemble final (single + pair + triple)
# -------------------------
def fibo_votes(tail: List[int], config: dict = None) -> Tuple[Dict[int, float], dict]:
    """
    Retorna:
      probs: {1:...,2:...,3:...,4:...}
      diag : diagnóstico com componentes e escolhas
    """
    if config is None:
        config = {}

    K_SINGLE = tuple(config.get("K_SINGLE", (10, 21, 34, 55)))
    K_PAIR   = tuple(config.get("K_PAIR",   (10, 21, 34, 55)))
    K_TRIPLE = tuple(config.get("K_TRIPLE", (10, 21, 34, 55)))

    W_SINGLE = tuple(config.get("W_SINGLE", (0.15, 0.25, 0.30, 0.30)))
    W_PAIR   = tuple(config.get("W_PAIR",   (0.15, 0.25, 0.30, 0.30)))
    W_TRIPLE = tuple(config.get("W_TRIPLE", (0.15, 0.25, 0.30, 0.30)))

    WEIGHTS  = dict(config.get("WEIGHTS", {"single": 0.40, "pair": 0.35, "triple": 0.25}))
    ENTROPY_PUSH  = float(config.get("ENTROPY_PUSH", 0.12))
    MAX_SHARE_CAP = float(config.get("MAX_SHARE_CAP", 0.95))
    FLOOR_CAP     = float(config.get("FLOOR_CAP", 0.30))

    STREAK_REV_N  = int(config.get("STREAK_REV_N", 3))
    STREAK_REV_CUT= float(config.get("STREAK_REV_CUT", 0.55))

    # 1) scores por especialista
    p_single = _single_fibo(tail, K_SINGLE, W_SINGLE)
    p_pair   = _pair_fibo(tail,   K_PAIR,   W_PAIR)
    p_triple = _triple_fibo(tail, K_TRIPLE, W_TRIPLE)

    # 2) blend
    mix = _zeros()
    for c in CANDS:
        mix[c] = (
            WEIGHTS.get("single", 0.40) * p_single[c] +
            WEIGHTS.get("pair",   0.35) * p_pair[c]   +
            WEIGHTS.get("triple", 0.25) * p_triple[c]
        )
    mix = _norm(mix)

    # 3) reversão de streak (antipega repetição infinita)
    mix = _apply_streak_reversal(mix, tail, streak_n=STREAK_REV_N, cut=STREAK_REV_CUT)

    # 4) anti-viés + entropia
    mix = _cap_floor_mix(mix, floor=FLOOR_CAP, cap=MAX_SHARE_CAP)
    mix = _entropy_push(mix, ENTROPY_PUSH)
    mix = _norm(mix)

    # diagnóstico
    rank = sorted(mix.items(), key=lambda kv: kv[1], reverse=True)
    best = rank[0][0]
    gap  = (rank[0][1] - rank[1][1]) if len(rank) >= 2 else rank[0][1]

    diag = {
        "single": p_single,
        "pair":   p_pair,
        "triple": p_triple,
        "weights": WEIGHTS,
        "chosen": best,
        "gap": gap,
        "streak_len": _streak_of_last(tail),
        "params": {
            "K_SINGLE": K_SINGLE, "K_PAIR": K_PAIR, "K_TRIPLE": K_TRIPLE,
            "ENTROPY_PUSH": ENTROPY_PUSH, "MAX_SHARE_CAP": MAX_SHARE_CAP,
            "FLOOR_CAP": FLOOR_CAP, "STREAK_REV_N": STREAK_REV_N,
            "STREAK_REV_CUT": STREAK_REV_CUT
        }
    }
    return mix, diag