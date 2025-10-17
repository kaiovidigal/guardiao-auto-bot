#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
browser_worker.py
Capturador Playwright para Fan Tan (Pinbet) -> envia números ao GuardiAo Bot.

ENV obrigatórias:
- WEBHOOK_BASE   (ex: https://guardiao-auto-bot.onrender.com)
- WEBHOOK_TOKEN  (ex: meusegredo123)

ENV opcionais:
- PINBET_URL     (default: https://pinbet.bet/live-casino/evolution/evo-oss-xs-fan-tan)
- POLL_SEC       (default: 5)
- SHOW_DEBUG     (True|False; default False)

Como roda local:
  pip install -r requirements.txt
  python browser_worker.py

Como roda no Render:
  Procfile (adicione):
    worker: python browser_worker.py
"""

import os, re, asyncio, time, json
import httpx
from typing import List, Tuple, Optional
from playwright.async_api import async_playwright, Page

WEBHOOK_BASE  = os.getenv("WEBHOOK_BASE", "").rstrip("/")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "").strip()
PINBET_URL    = os.getenv("PINBET_URL", "https://pinbet.bet/live-casino/evolution/evo-oss-xs-fan-tan")
POLL_SEC      = int(os.getenv("POLL_SEC", "5"))
SHOW_DEBUG    = os.getenv("SHOW_DEBUG", "False").lower() == "true"

if not WEBHOOK_BASE or not WEBHOOK_TOKEN:
    raise SystemExit("❌ Set as ENVs: WEBHOOK_BASE e WEBHOOK_TOKEN")

INGEST_URL = f"{WEBHOOK_BASE}/ingest/fantan/{WEBHOOK_TOKEN}"

def _log(*a):
    print(*a, flush=True)

async def _safe_click(page: Page, text_regex: str):
    """tenta clicar em overlays (ex.: 'Toque para acionar o áudio')."""
    try:
        loc = page.get_by_text(re.compile(text_regex, re.I))
        if await loc.count() > 0:
            await loc.first.click(timeout=1000)
            _log("🫰 Cliquei overlay:", text_regex)
    except Exception:
        pass

async def extract_numbers(page: Page) -> List[int]:
    """
    Estratégia robusta:
    - procura TODOS os elementos cujo texto visível é 1,2,3,4
    - filtra por posição (parte superior da tela) e tamanho (pequenos = “bolinhas” do histórico)
    - ordena por X (esquerda -> direita) para recuperar a sequência
    """
    numbers: List[int] = []
    try:
        # pega *tudo* que é 1..4
        handles = await page.locator("xpath=//*[normalize-space(text())='1' or normalize-space(text())='2' or normalize-space(text())='3' or normalize-space(text())='4']").element_handles()
        cands: List[Tuple[float, int]] = []
        for h in handles:
            box = await h.bounding_box()
            if not box:  # elemento fora de viewport
                continue
            # heurística: o “tape” do histórico fica na parte de cima e números são pequenos
            if box["y"] < 350 and box["height"] <= 40 and box["width"] <= 40:
                try:
                    txt = (await h.inner_text()).strip()
                    if txt in ("1","2","3","4"):
                        cands.append((box["x"], int(txt)))
                except Exception:
                    continue

        # ordena da esquerda pra direita e extrai só os dígitos
        cands.sort(key=lambda t: t[0])
        numbers = [n for _, n in cands]

        # normaliza: remove buracos absurdos (ex.: se capturou 100 itens iguais)
        if len(numbers) > 80:
            numbers = numbers[-80:]

        if SHOW_DEBUG:
            _log("🧪 detectados (brutos):", numbers)

        # se não achou nada, tenta outro caminho lendo um container com “history”
        if not numbers:
            txts = await page.locator("css=div[class*='history'], div[class*='result'], div[class*='roadmap']").all_inner_texts()
            blob = " ".join(txts)
            numbers = [int(x) for x in blob if x in "1234"]
            if SHOW_DEBUG:
                _log("🧪 fallback history:", numbers)

    except Exception as e:
        _log("⚠️ extract_numbers erro:", repr(e))
    return numbers

async def push_to_webhook(seq: List[int]) -> Optional[dict]:
    try:
        async with httpx.AsyncClient(timeout=10) as cli:
            r = await cli.post(INGEST_URL, json={"numbers": seq})
            if r.status_code // 100 == 2:
                return r.json()
            _log("⚠️ ingest falhou:", r.status_code, r.text)
    except Exception as e:
        _log("⚠️ ingest erro:", repr(e))
    return None

async def main():
    async with async_playwright() as p:
        # agente de usuário móvel para ficar parecido com sua captura
        iphone_ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Mobile/15E148 Safari/604.1"
        browser = await p.chromium.launch(headless=True, args=[
            "--disable-dev-shm-usage",
            "--no-sandbox",
        ])
        context = await browser.new_context(user_agent=iphone_ua, locale="pt-BR")
        page = await context.new_page()

        _log("🌐 Abrindo:", PINBET_URL)
        await page.goto(PINBET_URL, wait_until="networkidle", timeout=90000)

        # Tenta fechar avisos/overlays
        await _safe_click(page, r"toque.*áudio|tap.*audio|clicar.*áudio")
        await _safe_click(page, r"aceitar|accept|ok|entendi")
        await asyncio.sleep(1.0)

        last_sent: Optional[int] = None   # último dígito enviado
        last_seq: List[int] = []          # último tape inteiro visto (p/ debug)

        _log("✅ Iniciado. Observando números…")
        while True:
            try:
                seq = await extract_numbers(page)

                # queremos o "último número novo" (direita do tape)
                if seq:
                    last_digit = seq[-1]
                    # só manda se mudou
                    if last_sent != last_digit:
                        last_sent = last_digit
                        last_seq = seq
                        # enviar só o último OU os dois últimos (opcional)
                        payload = [last_digit]
                        res = await push_to_webhook(payload)
                        _log(f"📤 Enviado ao webhook: {payload}  | resposta: {res}")
                else:
                    _log("ℹ️ nada detectado; tentando de novo…")

            except Exception as e:
                _log("⚠️ loop erro:", repr(e))

            await asyncio.sleep(POLL_SEC)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _log("⏹️ Encerrado pelo usuário")