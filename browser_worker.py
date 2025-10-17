import os, re, json, asyncio
from typing import List, Optional
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

LOGIN_URL = os.getenv("LOGIN_URL", "https://blaze.com/pt/login")
BET_USER = os.getenv("BET_USER", "")
BET_PASS = os.getenv("BET_PASS", "")
USER_SELECTOR = os.getenv("USER_SELECTOR", "input[name='username']")
PASS_SELECTOR = os.getenv("PASS_SELECTOR", "input[name='password']")
SUBMIT_SELECTOR = os.getenv("SUBMIT_SELECTOR", "button[type='submit']")
AFTER_LOGIN_SELECTOR = os.getenv("AFTER_LOGIN_SELECTOR", ".user-avatar")
RESULT_SELECTOR = os.getenv("RESULT_SELECTOR", ".roulette__history-item:first-child")
HEADLESS = os.getenv("HEADLESS", "true").lower() != "false"
STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/storage_state.json")
NAV_TIMEOUT_MS = int(os.getenv("NAV_TIMEOUT_MS", "30000"))
SEL_TIMEOUT_MS = int(os.getenv("SEL_TIMEOUT_MS", "15000"))

class ResultFetcher:
    def __init__(self):
        self._pw = None
        self._browser: Optional[Browser] = None
        self._ctx: Optional[BrowserContext] = None
        self._lock = asyncio.Lock()

    async def _ensure(self):
        if self._browser:
            return
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=HEADLESS, args=["--no-sandbox"])
        storage = STORAGE_PATH if os.path.exists(STORAGE_PATH) else None
        self._ctx = await self._browser.new_context(storage_state=storage)
        self._ctx.set_default_timeout(SEL_TIMEOUT_MS)

    async def _persist(self):
        if self._ctx:
            await self._ctx.storage_state(path=STORAGE_PATH)

    async def _login_if_needed(self):
        if not (LOGIN_URL and BET_USER and BET_PASS):
            return
        await self._ensure()
        page = await self._ctx.new_page()
        try:
            await page.goto(LOGIN_URL, timeout=NAV_TIMEOUT_MS)
            if AFTER_LOGIN_SELECTOR:
                try:
                    await page.wait_for_selector(AFTER_LOGIN_SELECTOR, timeout=3000)
                    await page.close()
                    return
                except Exception:
                    pass
            await page.fill(USER_SELECTOR, BET_USER)
            await page.fill(PASS_SELECTOR, BET_PASS)
            await page.click(SUBMIT_SELECTOR)
            if AFTER_LOGIN_SELECTOR:
                await page.wait_for_selector(AFTER_LOGIN_SELECTOR, timeout=SEL_TIMEOUT_MS)
            await self._persist()
        finally:
            await page.close()

    async def fetch_pair(self, url: str) -> List[int]:
        async with self._lock:
            await self._ensure()
            await self._login_if_needed()
            page: Page = await self._ctx.new_page()
            try:
                await page.goto(url, timeout=NAV_TIMEOUT_MS)
                await page.wait_for_load_state("networkidle", timeout=8000)
                text = ""
                try:
                    el = await page.query_selector(RESULT_SELECTOR)
                    if el:
                        text = (await el.text_content()) or ""
                except Exception:
                    pass
                if not text:
                    text = await page.content()
                nums = re.findall(r"[1-4]", text)
                return [int(x) for x in nums[:2]]
            finally:
                await page.close()

_fetcher: Optional[ResultFetcher] = None
async def get_fetcher() -> ResultFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = ResultFetcher()
    return _fetcher
