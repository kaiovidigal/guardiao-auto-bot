def _early_hit_prior(tail: List[int], k: int = 24) -> dict[int, float]:
    """
    Bônus sutil para números que nos últimos 'k' resultados aparecem logo após um dado contexto curto.
    Implementação simples: usa frequência pura nos últimos k (proxy de 'cedo').
    Retorna multiplicadores para {1,2,3,4}.
    """
    if not tail:
        return {1:1.0, 2:1.0, 3:1.0, 4:1.0}
    last = tail[-k:] if len(tail) >= k else tail[:]
    # menos frequente recebe bônus, evitando “manada” tardia
    from collections import Counter
    c = Counter(last)
    # ordenar por frequência ASC (quem menos apareceu recebe bônus maior)
    least_first = sorted([1,2,3,4], key=lambda n: (c.get(n,0), n))
    boosts = {n:1.0 for n in [1,2,3,4]}
    if len(least_first) >= 1:
        boosts[least_first[0]] = EARLY_BONUS       # 1º menos frequente
    if len(least_first) >= 2:
        boosts[least_first[1]] = (EARLY_BONUS+1)/2 # 2º menos frequente (meio bônus)
    return boosts