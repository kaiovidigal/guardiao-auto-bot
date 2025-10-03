*** webhook_app.py.orig	2025-10-03
--- webhook_app.py	2025-10-03
***************
*** 197,202 ****
--- 197,226 ----
   migrate_db()
  
+ # --- (1) Prior do Hedge com pesos de Fibonacci [1,1,2,3] normalizados ---
+ def _seed_expert_w_fib():
+     """
+     Inicializa os pesos dos 4 especialistas com proporção Fibonacci (1,1,2,3),
+     **apenas** se ainda estiver no padrão plano ~1.0 para todos (não sobrescreve
+     pesos já aprendidos).
+     """
+     try:
+         w1, w2, w3, w4 = _get_expert_w()
+     except Exception:
+         # se _get_expert_w ainda não disponível por ordem de definição
+         return
+     # tolerância para "igual" a 1.0 após migração
+     def _is_flat(ws, tol=1e-6):
+         return all(abs(x-1.0) < tol for x in ws)
+     if _is_flat([w1, w2, w3, w4]):
+         base = [1, 1, 2, 3]
+         s = sum(base)
+         _set_expert_w(*(x/s for x in base))
+ 
+ # chama após migrate_db; se já houver pesos aprendidos, não altera
+ _seed_expert_w_fib()
+ 
***************
*** 413,418 ****
--- 437,496 ----
      return {k: v/s for k,v in d.items()}
  
+ # --- (2) Frequência multi-janela com janelas de Fibonacci ---
+ # Mantém especialista "longo" (K_LONG) como está; substitui o "curto" por blend fib.
+ FIB_WINS = [13, 21, 34, 55]  # janelas curtíssimas a curtas de Fibonacci
+ _FIB_WIN_WEIGHTS = [1, 1, 2, 3]
+ _FIB_WIN_WEIGHTS = [w/sum(_FIB_WIN_WEIGHTS) for w in _FIB_WIN_WEIGHTS]
+ 
+ def _post_freq_fib(tail: List[int]) -> Dict[int, float]:
+     """
+     Especialista curto 'fib': mistura frequências em janelas {13,21,34,55}
+     com pesos proporcionais a [1,1,2,3].
+     """
+     if not tail:
+         return {1:0.25, 2:0.25, 3:0.25, 4:0.25}
+     acc = {1:0.0, 2:0.0, 3:0.0, 4:0.0}
+     for k, wk in zip(FIB_WINS, _FIB_WIN_WEIGHTS):
+         pk = _post_freq_k(tail, k)
+         for c in (1,2,3,4):
+             acc[c] += wk * pk[c]
+     # normalização defensiva
+     tot = sum(acc.values()) or 1e-9
+     return {c: acc[c]/tot for c in (1,2,3,4)}
+ 
+ # --- (3) Anti-tilt com intensidade guiada por Fibonacci ---
+ def _fib(n: int) -> int:
+     """Fibonacci 1,1,2,3,5,... (n>=1)"""
+     if n <= 1:
+         return 1
+     a, b = 1, 1
+     for _ in range(n-1):
+         a, b = b, a + b
+     return a
+ 
  def _post_freq_k(tail: List[int], k: int) -> Dict[int,float]:
      if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
      win = tail[-k:] if len(tail) >= k else tail
      tot = max(1, len(win))
      return _norm_dict({c: win.count(c)/tot for c in [1,2,3,4]})
  
***************
*** 458,482 ****
  def _hedge_update4(true_c:int, p1:Dict[int,float], p2:Dict[int,float], p3:Dict[int,float], p4:Dict[int,float]):
      w1, w2, w3, w4 = _get_expert_w()
      l = lambda p: 1.0 - p.get(true_c, 0.0)
      from math import exp
      w1n = w1 * exp(-HEDGE_ETA * (1.0 - l(p1)))
      w2n = w2 * exp(-HEDGE_ETA * (1.0 - l(p2)))
      w3n = w3 * exp(-HEDGE_ETA * (1.0 - l(p3)))
      w4n = w4 * exp(-HEDGE_ETA * (1.0 - l(p4)))
      S = (w1n + w2n + w3n + w4n) or 1e-9
      _set_expert_w(w1n/S, w2n/S, w3n/S, w4n/S)
  
  # ========= (NOVO) Print do Crupiê (snapshot leve, SEM side effects) =========
*** 623,653 ****
  def _streak_adjust_choice(post:Dict[int,float], gap:float, ls:int) -> Tuple[int,str,Dict[int,float]]:
      reason = "IA"
      ranking = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
      best = ranking[0][0]
!     if ls >= 3:
!         comp = _norm_dict({c: max(1e-9, 1.0 - post[c]) for c in [1,2,3,4]})
!         post = _norm_dict({c: 0.7*post[c] + 0.3*comp[c] for c in [1,2,3,4]})
!         ranking = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
!         best = ranking[0][0]
!         reason = "IA_anti_tilt_mix"
      if ls >= 2:
          top2 = ranking[:2]
          if len(top2) == 2 and gap < 0.05:
              best = top2[1][0]
              reason = "IA_runnerup_ls2"
      return best, reason, post
--- 534,571 ----
  def _streak_adjust_choice(post:Dict[int,float], gap:float, ls:int) -> Tuple[int,str,Dict[int,float]]:
      reason = "IA"
      ranking = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
      best = ranking[0][0]
!     # Anti-tilt progressivo guiado por Fibonacci (cap 3) para ls >= 3
!     if ls >= 3:
!         k = min(3, _fib(ls))  # 1,1,2,3,... -> limitado a 3
!         # mistura com o complemento cresce suavemente: 0.20 + 0.05*k ∈ [0.25..0.35]
!         mix = 0.20 + 0.05 * k
!         comp = _norm_dict({c: max(1e-9, 1.0 - post[c]) for c in [1,2,3,4]})
!         post = _norm_dict({c: (1.0 - mix)*post[c] + mix*comp[c] for c in [1,2,3,4]})
!         ranking = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
!         best = ranking[0][0]
!         reason = f"IA_anti_tilt_fib{k}"
      if ls >= 2:
          top2 = ranking[:2]
          if len(top2) == 2 and gap < 0.05:
              best = top2[1][0]
              reason = "IA_runnerup_ls2"
      return best, reason, post
***************
*** 655,670 ****
  def choose_single_number(after: Optional[int]):
      tail = get_tail(400)
-     post_e1 = _post_from_tail(tail, after)         # n-grama + feedback
-     post_e2 = _post_freq_k(tail, K_SHORT)          # freq curta
-     post_e3 = _post_freq_k(tail, K_LONG)           # freq longa
+     post_e1 = _post_from_tail(tail, after)         # n-grama + feedback
+     post_e2 = _post_freq_fib(tail)                 # curto: blend multi-janela Fibonacci
+     post_e3 = _post_freq_k(tail, K_LONG)           # longo (300) como já é
      post_e4 = _llm_probs_from_tail(tail) or {1:0.25,2:0.25,3:0.25,4:0.25}  # IA local
      post, (w1,w2,w3,w4) = _hedge_blend4(post_e1, post_e2, post_e3, post_e4)
      ranking = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
      top2 = ranking[:2]
      gap = (top2[0][1] - top2[1][1]) if len(top2) >= 2 else ranking[0][1]
      base_best = ranking[0][0]
      conf = float(post[base_best])
      ls = _get_loss_streak()
      best, reason, post_adj = _streak_adjust_choice(post, gap, ls)
      conf = float(post_adj[best])
      r2 = sorted(post_adj.items(), key=lambda kv: kv[1], reverse=True)[:2]
      gap2 = (r2[0][1] - r2[1][1]) if len(r2) == 2 else r2[0][1]
      return best, conf, timeline_size(), post_adj, gap2, reason