# -*- coding: utf-8 -*-
# ========= IMPORTS & APP =========
import os
import re
import asyncio
from fastapi import FastAPI, Request

# se suas funções já existem noutro módulo, ajuste os imports:
# from db import query_all, append_timeline, update_ngrams
# from telegram_helpers import tg_send_text

app = FastAPI()

# ========= CONFIG (fixa, com fallback seguro) =========
# Aceita ENV, mas tem default. strip() evita espaços; se vier vazio, vira "" e bloqueia envio (com log).
ALERT_CHANNEL = (os.getenv("ALERT_CHANNEL", "-1002796105884") or "").strip()

ALMOST_LIMIAR = 5        # alerta com 5 ou mais sem aparecer
ALMOST_SCAN_EVERY = 300  # 5 min

# ========= HELPERS =========
async def tg_alert(text: str):
    """
    Envia texto em HTML pro canal configurado. 
    Loga motivo quando não enviar (vazio, erro de API, etc).
    """
    if not ALERT_CHANNEL:
        print("[ALERT] ALERT_CHANNEL vazio — defina no Render/ENV ou ajuste a constante.")
        return

    # Alguns helpers caseiros exigem int; outros aceitam string. Tenta primeiro como string.
    try:
        await tg_send_text(ALERT_CHANNEL, text, "HTML")
        print(f"[ALERT] Enviado para {ALERT_CHANNEL!r}: {text[:80]!r}...")
        return
    except Exception as e1:
        print(f"[ALERT] Falha ao enviar como str ({ALERT_CHANNEL!r}): {e1}")

    # Tentativa como int (ex.: -1002796105884). Se falhar, loga e desiste.
    try:
        chat_id_int = int(ALERT_CHANNEL)
        await tg_send_text(chat_id_int, text, "HTML")
        print(f"[ALERT] Enviado para {chat_id_int} (int).")
    except Exception as e2:
        print(f"[ALERT] Falha também como int: {e2}")

# ========= CORE: ABSENÇAS =========
def get_absences() -> dict[int, int]:
    """
    Retorna quantas jogadas cada número (1..4) está sem aparecer,
    medido sobre a timeline em ordem DESC (mais recente primeiro).
    """
    rows = query_all("SELECT number FROM timeline ORDER BY id DESC LIMIT 5000")
    recent = [r["number"] for r in rows] if rows else []
    if not recent:
        return {1: 0, 2: 0, 3: 0, 4: 0}

    absences: dict[int, int] = {}
    for n in (1, 2, 3, 4):
        try:
            last_idx = recent.index(n)  # índice do último visto (0 = acabou de vir)
            absences[n] = last_idx
        except ValueError:
            absences[n] = len(recent)   # nunca visto no recorte
    return absences

_last_bullet_sent: dict[int, int] = {1: -1, 2: -1, 3: -1, 4: -1}  # anti-flood p/ 8/9

async def scan_and_alert_once():
    """
    Faz uma leitura e dispara:
    1) Alerta para todos n com ausência >= ALMOST_LIMIAR.
    2) Boletim de 'quase 10' (8/9), sem flood.
    """
    try:
        absences = get_absences()
    except Exception as e:
        print(f"[scan] erro em get_absences(): {e}")
        return

    if not absences:
        print("[scan] absences vazio")
        return

    # (1) Alerta principal
    hot = [(n, c) for n, c in absences.items() if c >= ALMOST_LIMIAR]
    if hot:
        hot.sort(key=lambda x: x[1], reverse=True)
        linhas = [f"⚠️ Número <b>{n}</b> está há <b>{c}</b> sem vir." for n, c in hot]
        rec = max(absences.values())
        msg = "⏰ <b>Alerta de ausência</b>\n" + "\n".join(linhas) + f"\n\n📈 Recorde atual observado: <b>{rec}</b>"
        await tg_alert(msg)

    # (2) Boletim quase 10
    for n, c in absences.items():
        if c in (8, 9) and _last_bullet_sent.get(n, -1) != c:
            _last_bullet_sent[n] = c
            await tg_alert(f"⏱️ Quase 10 sem vir: <b>{n}</b> (<b>{c}</b> sem vir)")

# ========= STARTUP LOOP =========
@app.on_event("startup")
async def _auto_loops_absence():
    async def _loop():
        print("[startup] loop de ausência iniciando…")
        # primeiro scan para "destravar"
        try:
            await asyncio.sleep(2)
            await scan_and_alert_once()
        except Exception as e:
            print(f"[startup_scan] error: {e}")

        # depois a cada X segundos
        while True:
            try:
                await scan_and_alert_once()
            except Exception as e:
                print(f"[loop_scan] error: {e}")
            await asyncio.sleep(ALMOST_SCAN_EVERY)

    # garante que não bloqueia o servidor
    asyncio.create_task(_loop())

# ========= ROTAS UTILITÁRIAS =========
@app.get("/test_alert")
async def test_alert():
    try:
        await tg_alert("🔧 Teste de alerta — canal OK.")
        return {"ok": True, "sent": True, "to": ALERT_CHANNEL}
    except Exception as e:
        return {"ok": False, "error": str(e), "to": ALERT_CHANNEL}

@app.post("/ingest_seq")
async def ingest_seq(request: Request):
    """
    Popular timeline rapidamente.
    Body JSON: {"seq": "1,2,3,2,4,1"}  (ordem cronológica)
    """
    try:
        data = await request.json()
        raw = (data.get("seq") or "").strip()
        parts = [int(x) for x in re.findall(r"[1-4]", raw)]
        if not parts:
            return {"ok": False, "error": "seq vazia"}

        for n in parts:
            append_timeline(n)

        update_ngrams()
        return {"ok": True, "added": len(parts)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/force_scan")
async def force_scan():
    try:
        await scan_and_alert_once()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}