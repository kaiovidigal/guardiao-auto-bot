from fastapi import FastAPI, Request, HTTPException
import httpx, os, json

APP_VERSION = "espelho-1.0"

app = FastAPI(title="GuardiAo Espelho", version=APP_VERSION)

# 🔧 Variáveis de ambiente
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "meusegredo123")
TG_BOT_TOKEN  = os.getenv("TG_BOT_TOKEN", "8315698154:AAH38hr2RbR0DtfalMNuXdGsh4UghDeztK4")
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1003052132833")

# 🧠 Envio de mensagem pro Telegram
async def tg_send(chat_id: str, text: str):
    if not TG_BOT_TOKEN:
        print("⚠️ TG_BOT_TOKEN não configurado.")
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=15) as cli:
        r = await cli.post(url, json={"chat_id": chat_id, "text": text})
        print("↩️ Telegram resp:", r.status_code, r.text[:200])

# ✅ Rotas básicas
@app.get("/")
async def root():
    return {"ok": True, "service": "GuardiAo Espelho", "version": APP_VERSION}

@app.get("/health")
async def health():
    return {
        "ok": True,
        "status": "alive",
        "token_loaded": bool(WEBHOOK_TOKEN),
        "target_set": bool(TARGET_CHANNEL),
    }

@app.get("/admin/echo-token/{token}")
async def echo_token(token: str):
    return {"match": token == WEBHOOK_TOKEN, "received": token, "expected": WEBHOOK_TOKEN}

# 🧩 PING rápido
@app.get("/mirror/ping")
async def mirror_ping():
    return {
        "ok": True,
        "ping": "pong",
        "tip": "Use POST /mirror/fantan/{token} com JSON {'numbers':[2,4]}",
    }

# 📡 Modo espelho: envia números crus ao Telegram
@app.post("/mirror/fantan/{token}")
async def mirror_fantan(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        data = await request.json()
    except Exception:
        data = {}
    seq = data.get("numbers") or []
    if not isinstance(seq, list) or not seq:
        return {"ok": False, "error": "no_numbers", "hint": "Envie JSON: {'numbers':[3,1,4]}"}
    msg = f"📡 Espelho Fan Tan — sequência detectada: {seq}"
    print(msg)
    if TARGET_CHANNEL:
        await tg_send(TARGET_CHANNEL, msg)
    return {"ok": True, "mirrored": seq}

# 🪄 Webhook padrão do Telegram (mantido pra compatibilidade)
@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    update = await request.json()
    print("📩 Update recebido:", json.dumps(update)[:400])
    return {"ok": True}