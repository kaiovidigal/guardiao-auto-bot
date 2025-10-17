#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from typing import List, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException

app = FastAPI(title="GuardiAo Auto Bot (webhook+mirror)", version="7.1")

# --------- ENV ----------
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "meusegredo123").strip()
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "").strip()  # ex: -1003052132833

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# --------- TELEGRAM ----------
async def tg_send(chat_id: str, text: str, parse_mode: Optional[str]="HTML"):
    if not TG_BOT_TOKEN or not chat_id:
        print("⚠️ Falta TG_BOT_TOKEN ou chat_id.")
        return
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            await cli.post(f"{TELEGRAM_API}/sendMessage",
                           json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode,
                                 "disable_web_page_preview": True})
    except Exception as e:
        print("Erro enviando ao Telegram:", e)

# --------- BASICS ----------
@app.get("/")
async def root():
    return {"ok": True, "service": "GuardiAo Auto Bot", "version": "7.1"}

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/debug_cfg")
async def debug_cfg():
    return {"WEBHOOK_TOKEN_set": bool(WEBHOOK_TOKEN),
            "TG_BOT_TOKEN_set": bool(TG_BOT_TOKEN),
            "TARGET_CHANNEL": TARGET_CHANNEL}

# --------- WEBHOOK PADRÃO (Telegram) ----------
@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    update = await request.json()
    # só loga pra não quebrar bot setWebhook
    print("📩 Update recebido:", update)
    return {"ok": True}

# --------- ESPELHO FAN TAN (POST) ----------
@app.post("/mirror/fantan/{token}")
async def mirror_fantan(token: str, request: Request):
    """
    Body esperado: {"numbers":[1,3,4]}
    """
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()
    seq: List[int] = data.get("numbers") or []
    seq = [int(x) for x in seq if int(x) in (1,2,3,4)]

    if not seq:
        return {"ok": False, "error": "no_numbers"}

    msg = f"📡 Espelho Fan Tan — sequência detectada: {seq}"
    print(msg)
    if TARGET_CHANNEL:
        await tg_send(TARGET_CHANNEL, msg)
    return {"ok": True, "mirrored": seq}

# --------- TESTE RÁPIDO (GET pelo navegador) ----------
@app.get("/mirror/test/{token}")
async def mirror_test(token: str, seq: str = "2-4-1"):
    """
    Use no navegador: /mirror/test/{token}?seq=2-4-1
    """
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        nums = [int(x) for x in seq.split("-") if int(x) in (1,2,3,4)]
    except Exception:
        nums = []
    if not nums:
        return {"ok": False, "error": "bad_seq"}
    msg = f"🧪 TESTE — sequência: {nums}"
    print(msg)
    if TARGET_CHANNEL:
        await tg_send(TARGET_CHANNEL, msg)
    return {"ok": True, "mirrored": nums}