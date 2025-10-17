import os, asyncio, json, time, re, httpx
from datetime import datetime

WEBHOOK_BASE   = os.getenv("MIRROR_BASE", "").rstrip("/")  # ex: https://guardiao-auto-bot.onrender.com
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "meusegredo123")
PINBET_URL     = os.getenv("PINBET_FANTAN_URL", "https://pinbet.bet/live-casino/evolution/evo-oss-xs-fan-tan")
POLL_SEC       = int(os.getenv("POLL_SEC", "15"))

# Trocar para Playwright real quando quiser (este arquivo funciona sem Playwright como "simulador")
USE_SIMULATOR  = os.getenv("USE_SIMULATOR", "true").lower() == "true"

async def post_numbers(nums):
    url = f"{WEBHOOK_BASE}/mirror/fantan/{WEBHOOK_TOKEN}"
    payload = {"numbers": nums}
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            r = await cli.post(url, json=payload)
            print("POST", url, payload, "->", r.status_code, r.text[:200])
    except Exception as e:
        print("POST error:", e)

async def simulator_loop():
    # Simula leitura: envia números aleatórios 1..4 só para provar pipeline
    import random
    seen = []
    while True:
        n = random.randint(1,4)
        seen.append(n)
        if len(seen) > 2:  # manda pares/duplas
            await post_numbers(seen[-2:])
        await asyncio.sleep(POLL_SEC)

async def playwright_loop():
    # Exemplo de esqueleto (quando for usar Playwright real)
    from playwright.async_api import async_playwright

    last_sig = None
    while True:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                ctx = await browser.new_context()
                page = await ctx.new_page()
                await page.goto(PINBET_URL, wait_until="domcontentloaded", timeout=60000)

                # TODO: se precisar login, faça aqui (page.fill / page.click)
                # TODO: identificar os seletores certos do histórico (depende do DOM da Pinbet)
                # Exemplo (ajuste de acordo com a página real):
                # await page.wait_for_selector(".history__list", timeout=60000)
                # html = await page.inner_html(".history__list")
                # nums = re.findall(r"[> ]([1-4])[ <]", html)
                # seq = [int(x) for x in nums[:2]]

                # enquanto não temos seletores confirmados, envia dummy para prova de vida:
                seq = [1, 3]
                if seq and seq != last_sig:
                    last_sig = seq
                    await post_numbers(seq)

                await browser.close()
        except Exception as e:
            print("playwright_loop error:", e)
        await asyncio.sleep(POLL_SEC)

async def main():
    if not WEBHOOK_BASE:
        print("Defina MIRROR_BASE (ex: https://guardiao-auto-bot.onrender.com)")
        return
    if USE_SIMULATOR:
        print("Rodando SIMULATOR (sem Playwright) …")
        await simulator_loop()
    else:
        print("Rodando Playwright …")
        await playwright_loop()

if __name__ == "__main__":
    asyncio.run(main())