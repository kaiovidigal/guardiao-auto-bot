# 2) FECHAMENTO — fechar imediatamente pelo número no ÚLTIMO parêntese
if RX_FECHA.search(text) or RX_GREEN.search(text) or RX_RED.search(text):
    if _seen_recent("fechamento", _dedupe_key(text)):
        return {"ok": True, "skipped": "fechamento_dupe"}

    pend = _pending_get()
    if pend:
        suggested = int(pend["suggested"] or 0)

        # captura do canal-fonte: último dígito 1..4 no último ()
        obs = _parse_close_digit(text)

        # grava exatamente 1 observado (G0) — fechamento imediato
        if obs is not None:
            _pending_seen_append([obs], need=1)

        # lê o observado e decide
        seen_txt = (_pending_get()["seen"] or "").strip()
        final_seen = seen_txt if seen_txt else "X"
        outcome = "GREEN" if (seen_txt.isdigit() and int(seen_txt) == suggested) else "LOSS"
        stage_lbl = "G0"

        msg_txt = _pending_close(final_seen, outcome, stage_lbl, suggested)
        if msg_txt:
            await tg_send(TARGET_CHANNEL, msg_txt)

        if SHOW_DEBUG:
            await tg_send(
                TARGET_CHANNEL,
                f"DEBUG: fechamento(parênteses) -> {obs} | seen='{final_seen}' | nosso={suggested}"
            )
        return {"ok": True, "closed": outcome, "seen": final_seen}

    if SHOW_DEBUG:
        await tg_send(TARGET_CHANNEL, "DEBUG: Fechamento reconhecido — sem pendência.")
    return {"ok": True, "noted_close": True}