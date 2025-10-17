#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
browser_worker.py
Captura números do Fan Tan (Pinbet) e espelha via POST /mirror/fantan/{token}.

ENVs obrigatórias:
  WEBHOOK_BASE   (ex: https://guardiao-auto-bot.onrender.com)
  WEBHOOK_TOKEN  (ex: meusegredo123)

ENVs opcionais:
  PINBET_URL     (default: https://pinbet.bet/live-casino/evolution/evo-oss-xs-fan-tan)
  POLL_SEC       (default: 5)          # intervalo de varredura (s)
  SHOW_DEBUG     (True|False; default False)

Como rodar local:
  pip install -r requirements_worker.txt
  python -m playwright install --with-deps chromium
  python browser_worker.py
"""

import os
import asyncio
import json
import re
from typing import List, Tuple, Optional

import httpx
from playwright.async_api import async_playwright, Page

WEBHOOK_BASE  = os.getenv("WEBHOOK_BASE", "").rstrip("/")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "").strip()
PINBET_URL    = os.getenv("PINBET_URL", "https://pinbet.bet/live-casino/evolution/evo-oss-xs-fan-tan")
POLL_SEC      = int(os.getenv("POLL_SEC", "5"))
SHOW_DEBUG    = os.getenv("SHOW_DEBUG", "False").lower() == "true"

if not WEBHOOK_BASE or not WEBHOOK_TOKEN:
    raise SystemExit("❌ Defina ENVs WEBHOOK_BASE e WEBHOOK_TOKEN.")

MIRROR_URL = f"{WEBHOOK_BASE}/mirror/fantan/{WEBHOOK_TOKEN}"

def log(*a):
    print(*a, flush=True)

async def extract_numbers(page: Page) -> List[int]:
    """
    Estratégia:
      1) Captura elementos cujo texto visível seja 1..4 (tape do topo costuma ser dígitos pequenos).
      2) Filtra por bounding-box (provavelmente na parte superior e com tamanho pequeno).
      3) Ordena por X para formar a sequência esquerda->direita.
      4) Se vazio, tenta fallback procurando containers de 'history/roadmap'.
    """
    numbers: List[int] = []
    try:
        handles = await page.locator(
            "xpath=//*[normalize-space(text())='1' or normalize-space(text())='2' or normalize-space(text())='3' or normalize-space(text())='4']"
        ).element_handles()

        candidates: List[Tuple[float, int]] = []
        for h in handles:
            box = await h.bounding_box()
            if not box:
                continue
            # Heurística: tape no topo, dígitos pequenos
            if box["y"] < 360 and box["height"] <= 42 and box["width"] <= 42:
                try:
                    txt = (await h.inner_text()).strip()
                    if txt in ("1", "2", "3", "4"):
                        candidates.append((box["x"], int(txt)))
                except Exception:
                    continue

        candidates.sort(key=lambda t: t[0])
        numbers = [n for _, n in candidates]

        if not numbers:
            # fallback por texto bruto em containers comuns de histórico
            txts = await page.locator(
                "css=div[class*='history'], div[class*='result'], div[class*='roadmap'], div[class*='recent']"
            ).all_inner_texts()
            blob = " ".join(txts)
            numbers = [int(ch) for ch in blob if ch in "1234"]

        # limita comprimento para evitar ruído gigante
        if len(numbers) > 80:
            numbers = numbers[-80:]

        if SHOW_DEBUG:
            log("🧪 detectados:", numbers)

    except Exception as e:
        log("⚠️ extract_numbers erro:", repr(e))
    return numbers

async def post_mirror(seq: List[int]) -> Optional[dict]:
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            r = await cli.post(MIRROR_URL, json={"numbers": seq})
            if r.status_code // 100 == 2:
                return r.json()
            log("⚠️ mirror falhou:", r.status_code, r.text[:200])
    except Exception as e:
        log("⚠️ mirror erro:", repr(e))
    return None

async def main():
    async with async_playwright() as p:
        # user-agent móvel (ajuda a pegar layout igual ao iPhone da sua captura)
        iphone_ua = (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6_2 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Mobile/15E148 Safari/604.1"
        )
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        ctx = await browser.new_context(user_agent=iphone_ua, locale="pt-BR")
        page = await ctx.new_page()

        log("🌐 Abrindo:", PINBET_URL)
        await page.goto(PINBET_URL, wait_until="networkidle", timeout=90000)

        # (Opcional) fechar overlays comuns
        for label in ("aceitar", "accept", "ok", "entendi", "audio", "áudio"):
            try:
                loc = page.get_by_text(re.compile(label, re.I))
                if await loc.count() > 0:
                    await loc.first.click(timeout=1200)
                    log("🫰 Fechei overlay:", label)
            except Exception:
                pass

        last_sent: Optional[int] = None  # último dígito espelhado
        log("✅ Worker iniciado. Aguardando números…")

        while True:
            try:
                seq = await extract_numbers(page)
                if seq:
                    last_digit = seq[-1]  # último da direita
                    if last_sent != last_digit:
                        last_sent = last_digit
                        payload = [last_digit]  # espelha só o último (pode trocar para seq[-2:] se quiser)
                        res = await post_mirror(payload)
                        log(f"📤 Espelho -> {payload} | resp: {res}")
                else:
                    log("ℹ️ Nenhum número detectado, tentando novamente…")

            except Exception as e:
                log("⚠️ loop erro:", repr(e))

            await asyncio.sleep(POLL_SEC)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("⏹️ Encerrado pelo usuário")