# fibo_experts.py
# Meta-experts baseados em Fibonacci + padrão/combinações, plug & play
# expõe:
#   - mc_predict_probs(tail: List[int]) -> Dict[int,float]
#   - mc_feedback(sugerido: int, observados: List[int], outcome: str) -> None
#
# Observações:
# - Não depende de nada externo.
# - Mantém estado leve em memória (bias) com EMA.
# - Usa 5 janelas Fibo (5, 8, 13, 21, 34) + pares e trincas (Markov 2/3) + regras de reversão e compostos.
# - Se o tail for curto, retorna distribuição quase-uniforme com leves heurísticas.

from typing import List, Dict
import math

# ======== Hiperparâmetros ========
FIBO_WINS = (5, 8, 13, 21, 34)    # 5 janelas
EMA_BIAS  = 0.90                  # suavização do feedback
DELTA_POS = 0.10                  # reforço por GREEN
DELTA_NEG = -0.07                 # punição por LOSS
REV_PUSH  = 0.07                  # empurrão de reversão
COMP_PUSH = 0.06                  # empurrão de padrão composto
PAIR_W    = 0.40                  # peso do Markov-2
TRI_W     = 0.35                  # peso do Markov-3
FIBO_W    = 0.25                  # peso do bloco fibo puro dentro do meta

# estado em memória do processo (bias nos 4 dígitos)
_bias = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

def _norm(d: Dict[int, float]) -> Dict[int, float]:
    s = sum(d.values()) or 1e-9
    return {k: v / s for k, v in d.items()}

def _freq(win: List[int]) -> Dict[int, float]:
    if not win:
        return {1:0.25,2:0.25,3:0.25,4:0.25}
    tot = len(win)
    return _norm({c: win.count(c)/tot for c in (1,2,3,4)})

def _fibo_block(tail: List[int]) -> Dict[int, float]:
    """Mistura de frequências em janelas Fibo."""
    if not tail:
        return {1:0.25,2:0.25,3:0.25,4:0.25}
    mix = {1:0.0,2:0.0,3:0.0,4:0.0}
    weights = [0.15, 0.20, 0.25, 0.20, 0.20]  # soma ~1
    for k, w in zip(FIBO_WINS, weights):
        win = tail[-k:] if len(tail) >= k else tail
        fk = _freq(win)
        for c in (1,2,3,4):
            mix[c] += w * fk[c]
    return _norm(mix)

def _markov2(tail: List[int]) -> Dict[int, float]:
    """Prob condicional do próximo dígito dado o último (pares)."""
    if len(tail) < 2:
        return {1:0.25,2:0.25,3:0.25,4:0.25}
    last = tail[-1]
    cnt = {1:0,2:0,3:0,4:0}
    for a,b in zip(tail[:-1], tail[1:]):
        if a == last:
            cnt[b] += 1
    tot = sum(cnt.values())
    if tot == 0:
        return {1:0.25,2:0.25,3:0.25,4:0.25}
    return _norm({c: cnt[c]/tot for c in (1,2,3,4)})

def _markov3(tail: List[int]) -> Dict[int, float]:
    """Prob condicional do próximo dígito dado os dois últimos (trincas)."""
    if len(tail) < 3:
        return {1:0.25,2:0.25,3:0.25,4:0.25}
    last2 = (tail[-2], tail[-1])
    cnt = {1:0,2:0,3:0,4:0}
    for i in range(len(tail)-2):
        a,b,c = tail[i], tail[i+1], tail[i+2]
        if (a,b) == last2:
            if i+2 < len(tail):
                # próximo após (a,b) é c
                cnt[c] += 1
    tot = sum(cnt.values())
    if tot == 0:
        return {1:0.25,2:0.25,3:0.25,4:0.25}
    return _norm({d: cnt[d]/tot for d in (1,2,3,4)})

def _reversal_push(tail: List[int]) -> Dict[int, float]:
    """Heurística de reversão: se há streak alto do último dígito, penaliza-o e
       distribui leve push para os demais."""
    if not tail:
        return {1:0.25,2:0.25,3:0.25,4:0.25}
    last = tail[-1]
    s = 0
    for x in reversed(tail):
        if x == last: s += 1
        else: break
    base = {1:0.25,2:0.25,3:0.25,4:0.25}
    if s >= 3:
        # tira um pouco do último e distribui
        drop = REV_PUSH
        base[last] = max(0.0, base[last] - drop)
        add = drop / 3.0
        for c in (1,2,3,4):
            if c != last:
                base[c] += add
    return _norm(base)

def _composite_rules(tail: List[int]) -> Dict[int, float]:
    """Padrões compostos simples: ex: 2-3 → 4 ; 1-4 → 2 ; 3-2-? → 1 etc.
       Regras fracas, só dão um empurrão."""
    boost = {1:0.25,2:0.25,3:0.25,4:0.25}
    if len(tail) >= 2:
        a,b = tail[-2], tail[-1]
        # regras 2-3 -> 4 ; 3-4 -> 1 ; 1-2 -> 3 (exemplos)
        if (a,b) == (2,3):
            boost[4] += COMP_PUSH
        if (a,b) == (3,4):
            boost[1] += COMP_PUSH
        if (a,b) == (1,2):
            boost[3] += COMP_PUSH
        # regra de alternância: 1-3 -> 2
        if (a,b) == (1,3):
            boost[2] += COMP_PUSH

    if len(tail) >= 3:
        a,b,c = tail[-3], tail[-2], tail[-1]
        # trinca 3-2-x costuma fechar em 1 (exemplo heurístico)
        if (a,b) == (3,2):
            boost[1] += COMP_PUSH
        # 4-4-? favorece 2
        if a == 4 and b == 4:
            boost[2] += COMP_PUSH

    return _norm(boost)

def _apply_bias(probs: Dict[int,float]) -> Dict[int,float]:
    out = {c: max(0.0, probs.get(c,0.0) + _bias.get(c,0.0)) for c in (1,2,3,4)}
    return _norm(out)

def mc_predict_probs(tail: List[int]) -> Dict[int, float]:
    """Combina FIBO puro + Markov (2/3) + reversão + compostos + bias (feedback)."""
    fibo_p = _fibo_block(tail)       # notas longas (5 janelas)
    mk2_p  = _markov2(tail)          # pares
    mk3_p  = _markov3(tail)          # trincas
    rev_p  = _reversal_push(tail)    # reversão
    comp_p = _composite_rules(tail)  # padrões compostos

    # mistura do meta-FIBO:
    # primeiro combina markov2/3 e compostos/reversão
    mk_mix = {c: PAIR_W*mk2_p[c] + TRI_W*mk3_p[c] for c in (1,2,3,4)}
    aux_mix = {c: 0.5*rev_p[c] + 0.5*comp_p[c] for c in (1,2,3,4)}
    # agora combina com o bloco FIBO puro
    base = {c: FIBO_W*fibo_p[c] + (1.0 - FIBO_W)*(0.60*mk_mix[c] + 0.40*aux_mix[c]) for c in (1,2,3,4)}

    # aplica bias aprendido
    out = _apply_bias(_norm(base))
    # pequeno piso/teto
    # (evita 0% e 100%)
    eps = 1e-6
    out = _norm({c: max(eps, min(1.0-3*eps, out[c])) for c in (1,2,3,4)})
    return out

def mc_feedback(sugerido: int, observados: List[int], outcome: str) -> None:
    """Atualiza BIAS com EMA conforme resultado consolidado."""
    if sugerido not in (1,2,3,4):
        return
    # decide se GREEN (sugerido == qualquer observado 'útil')
    green = any((x == sugerido) for x in observados) and outcome.upper()=="GREEN"
    delta = DELTA_POS if green else DELTA_NEG
    # EMA por classe sugerida
    cur = _bias.get(sugerido, 0.0)
    _bias[sugerido] = EMA_BIAS*cur + (1.0-EMA_BIAS)*delta
    # leve normalização para não explodir
    lim = 0.30
    for c in (1,2,3,4):
        _bias[c] = float(max(-lim, min(lim, _bias.get(c,0.0))))