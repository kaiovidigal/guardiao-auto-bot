import asyncio, httpx, time
from playwright.async_api import async_playwright

WEBHOOK_URL = "https://guardiao-auto-bot.onrender.com/ingest/fantan/meusegredo123"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://pinbet.bet/casino/fan-tan")
        print("✅ Acessou Fan Tan Pinbet")
        last_numbers = []

        while True:
            try:
                # Localiza a barra de resultados (classe depende do HTML atual)
                elements = await page.query_selector_all("div[class*='history']")
                if elements:
                    text = await elements[0].inner_text()
                    # Extrai apenas os números 1–4
                    numbers = [int(x) for x in text if x in "1234"]
                    if numbers and numbers != last_numbers:
                        last_numbers = numbers
                        print("📊 Sequência detectada:", numbers)
                        # Envia para o webhook
                        async with httpx.AsyncClient() as cli:
                            await cli.post(WEBHOOK_URL, json={"numbers": numbers[-2:]})
            except Exception as e:
                print("⚠️ Erro:", e)
            await asyncio.sleep(5)  # verifica a cada 5 segundos

asyncio.run(main())