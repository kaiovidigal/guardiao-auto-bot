#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
browser_worker.py — Captura Fan Tan (Pinbet) e envia para /ingest/fantan/{token}

ENVs:
  WEBHOOK_BASE     = https://guardiao-auto-bot.onrender.com
  WEBHOOK_TOKEN    = meusegredo123
  PINBET_EMAIL     = seu@email
  PINBET_PASSWORD  = sua_senha
  FAN_TAN_URL      = https://pinbet.bet/live-casino/evolution/evo-oss-xs-fan-tan

Instalação local:
  pip install -r requirements_worker.txt
  playwright install

Execução:
  python browser_worker.py
"""

import os, time, json, asyncio
import requests

from playwright.async_api import async_playwright

WEBHOOK_BASE  = os.getenv("WEBHOOK_BASE", "https://guardiao-auto-bot.onrender.com").rstrip("/")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "meusegredo123")
INGEST_URL    = f"{WEBHOOK_BASE}/ingest/fantan/{WEBHOOK_TOKEN}"

PINBET_EMAIL    = os.getenv("PINBET_EMAIL", "")
PINBET_PASSWORD = os.getenv("PINBET_PASSWORD", "")
FAN_TAN_URL     = os.getenv("FAN_TAN_URL", "https://pinbet.bet/live-casino/evolution/evo-oss-xs-fan-tan")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "10"))

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context()
        page = await ctx.new_page()

        # TODO: ajustar seletor/fluxo de login conforme o site real
        await page.goto("https://pinbet.bet", wait_until="domcontentloaded")
        # Exemplo (placeholders):
        # await page.click("text=Entrar")
        # await page.fill("input[type=email]", PINBET_EMAIL)
        # await page.fill("input[type=password]", PINBET_PASSWORD)
        # await page.click("button:has-text('Entrar')")
        # await page.wait_for_load_state("networkidle")

        await page.goto(FAN_TAN_URL, wait_until="networkidle")

        last_sent = None

        while True:
            try:
                # TODO: substitua por seletores corretos da mesa Fan Tan
                # Ex.: resultados recentes em spans com classe .result-number
                elements = await page.query_selector_all(".result-number")
                nums = []
                for el in elements[:2]:
                    txt = (await el.inner_text()).strip()
                    if txt and txt[0] in "1234":
                        nums.append(int(txt[0]))

                if len(nums) >= 2 and nums != last_sent:
                    payload = {"numbers": nums[:2]}
                    print("→ Ingest:", payload)
                    r = requests.post(INGEST_URL, json=payload, timeout=15)
                    print("← Ingest resp:", r.status_code, r.text[:200])
                    if r.ok:
                        last_sent = nums[:2]

            except Exception as e:
                print("loop error:", e)

            await asyncio.sleep(POLL_SECONDS)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())