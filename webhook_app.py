from fastapi import FastAPI, Request, HTTPException
import httpx
import os

app = FastAPI(title="GuardiAo Auto Bot (Espelho Fan Tan)", version="1.0")

WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "meusegredo123")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "")

async def tg_send(chat_id: str, text: str):
    if not TG_BOT_TOKEN:
        print("⚠️ TG_BOT_TOKEN não configurado.")
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as cli:
        await cli.post(url, json={"chat_id": chat_id, "text": text})

@app.get("/")
async def root():
    return {"ok": True, "service": "GuardiAo Bot", "status": "online"}

@app.get("/health")
async def health():
    return {"ok": True, "status": "GuardiAo Bot ativo"}

@app.get("/mirror/ping")
async def ping():
    return {"ok": True, "ping": "pong"}

@app.get("/mirror/fantan/{token}")
async def info(token: str):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {
        "ok": True,
        "msg": "Use POST /mirror/fantan/{token} com JSON {'numbers':[1,2,3]}",
        "example": f"/mirror/fantan/{token}"
    }

@app.post("/mirror/fantan/{token}")
async def mirror_fantan(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()
    seq = data.get("numbers") or []
    if not seq:
        return {"ok": False, "error": "no_numbers"}

    msg = f"📡 Espelho Fan Tan — sequência detectada: {seq}"
    print(msg)

    if TARGET_CHANNEL:
        await tg_send(TARGET_CHANNEL, msg)

    return {"ok": True, "mirrored": seq}

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    update = await request.json()
    print("📩 Update recebido (Telegram):", update)
    return {"ok": True}