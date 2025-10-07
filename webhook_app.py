    # 2) APOSTA ENCERRADA / GREEN / RED (com dedupe + FECHAMENTO via (x))
    if RX_FECHA.search(text) or RX_GREEN.search(text) or RX_RED.search(text):
        if _seen_recent("fechamento", _dedupe_key(text)):
            return {"ok": True, "skipped": "fechamento_dupe"}

        pend = _pending_get()
        if pend:
            suggested = int(pend["suggested"] or 0)
            need_obs  = max(1, min(2, MAX_GALE+1))

            # (A) PRIMEIRO: tenta ler números finais pelos parênteses
            paren_events = _extract_result_paren_events(text)
            if paren_events:
                # usa o ÚLTIMO evento (caso a mensagem tenha RED (...) e depois GREEN (...))
                label, final_num = paren_events[-1]
                _pending_seen_append([final_num], need=need_obs)

            else:
                # (B) FALLBACK: se não veio número entre parênteses, usa "Sequência: a | b"
                obs_pair = _parse_seq_pair(text, need=need_obs)
                if obs_pair:
                    _pending_seen_append(obs_pair, need=need_obs)

            # memória extra: se a msg tiver MAIS de um "(x)" (ex.: RED (4) ... GREEN (3)),
            # podemos alimentar a timeline com todos eles
            if paren_events:
                _append_seq([n for _, n in paren_events])

            # Reavalia pendência após update
            pend = _pending_get()
            seen = [s for s in (pend["seen"] or "").split("-") if s]

            # Lógica de fechamento (G0/G1) baseada APENAS no número observado
            outcome = "LOSS"; stage_lbl = "G1"

            # GREEN se sugerido bateu em G0
            if len(seen) >= 1 and seen[0].isdigit() and int(seen[0]) == suggested:
                outcome = "GREEN"; stage_lbl = "G0"
            # Senão, se há G1 permitido e o segundo observado bateu
            elif MAX_GALE >= 1 and len(seen) >= 2 and seen[1].isdigit() and int(seen[1]) == suggested:
                outcome = "GREEN"; stage_lbl = "G1"

            # Se ainda estamos em G0 e não bateu, e MAX_GALE>=1 -> aguarda G1 (não fecha)
            if MAX_GALE >= 1 and len(seen) == 1 and stage_lbl != "G0":
                if SHOW_DEBUG:
                    await tg_send(TARGET_CHANNEL, f"DEBUG: aguardando G1 (G0={seen[0]}, nosso={suggested}).")
                return {"ok": True, "waiting_g1": True, "seen": "-".join(seen)}

            # Fecha: GREEN G0, ou quando já temos o necessário (G0-only ou G1 observado)
            close_now = (stage_lbl == "G0") or (len(seen) >= need_obs)
            if close_now:
                final_seen = "-".join(seen[:need_obs]) if seen else "X"
                msg_txt = _pending_close(final_seen, outcome, stage_lbl, suggested)
                if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)
                return {"ok": True, "closed": outcome, "seen": final_seen}

            return {"ok": True, "waiting_more_obs": True, "seen": "-".join(seen)}

        return {"ok": True, "noted_close": True}