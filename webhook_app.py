# -*- coding: utf-8 -*-
import os
import asyncio
import httpx
from fastapi import FastAPI

# ======== CONFIG ========
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
ALERT_CHANNEL = (os.getenv("ALERT_CHANNEL") or "").strip()
SCAN_EVERY = int(os.getenv("SCAN_EVERY", "2"))  # segundos

app = FastAPI()

# ======== DB (substitua pelos seus helpers reais) ========
# from db import query_all
def query_all(sql: str):
    # implemente com seu banco; deve retornar [{"number": 1}, {"number": 2}, ...]
    return []

# ======== TELEGRAM ========
TELEGRAM_API_BASE = "https://api.telegram.org"

async def tg_send_text(chat_id: int | str, text: str):
    if not TELEGRAM_TOKEN or not chat_id:
        return
    url = f"{TELEGRAM_API_BASE}/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": chat_id, "text": text}
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=15.0)) as client:
        await client.post(url, json=data)

# ======== LÓGICA ========
_last_alerted_at_9 = {1: False, 2: False, 3: False, 4: False}

def get_tail_absences_50():
    rows = query_all("SELECT number FROM timeline ORDER BY id DESC LIMIT 50")
    recent_desc = [r["number"] for r in rows] if rows else []
    # recent_desc[0] = mais recente; ausência = índice do número (ou len)
    absences = {}
    for n in (1, 2, 3, 4):
        try:
            c = recent_desc.index(n)
        except ValueError:
            c = len(recent_desc)
        absences[n] = c
    return absences

async def check_and_alert():
    absences = get_tail_absences_50()
    for n, c in absences.items():
        if c == 9 and not _last_alerted_at_9[n]:
            _last_alerted_at_9[n] = True
            await tg_send_text(ALERT_CHANNEL, f"Sinal: número {n} está 9 vezes sem vir")
        elif c == 0:
            _last_alerted_at_9[n] = False

# ======== LOOP ========
@app.on_event("startup")
async def _run():
    async def _loop():
        while True:
            try:
                await check_and_alert()
            finally:
                await asyncio.sleep(SCAN_EVERY)
    asyncio.create_task(_loop())