from fastapi import FastAPI, Request, HTTPException
import httpx
import os

app = FastAPI(title="GuardiAo Auto Bot (espelho FanTan)", version="1.0")

# ===============================
# 🔐 Variáveis de ambiente
# ===============================
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "meusegredo123")
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "")
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "")

# ===============================
# 📤 Enviar mensagem pro Telegram
# ===============================
async def tg_send(chat_id: str, text: str):
    if not TG_BOT_TOKEN:
        print("⚠️ TG_BOT_TOKEN não configurado.")
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=15) as cli:
        await cli.post(url, json={"chat_id": chat_id, "text": text})

# ===============================
# 🩺 Health / Root
# ===============================
@app.get("/")
async def root():
    return {"ok": True, "service": "GuardiAo (espelho)", "hint": "/docs"}

@app.get("/health")
async def health():
    return {"ok": True, "status": "GuardiAo Bot ativo"}

# ===============================
# 📡 Modo Espelho (Fan Tan)
# ===============================
@app.get("/mirror/ping")
async def mirror_ping():
    return {"ok": True, "ping": "pong"}

@app.get("/mirror/fantan/{token}")
async def mirror_fantan_info(token: str):
    """Rota GET só pra não dar 404 e mostrar instruções."""
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {
        "ok": True,
        "how_to_use": "Envie um POST com JSON {\"numbers\":[2,4]} para esta mesma URL.",
        "example": f"POST /mirror/fantan/{token}  body: {{\"numbers\":[2,4]}}"
    }

@app.post("/mirror/fantan/{token}")
async def mirror_fantan(token: str, request: Request):
    """
    Recebe lista de números (ex: {"numbers":[2,4]})
    e apenas espelha no canal do Telegram definido em TARGET_CHANNEL.
    """
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_json")

    seq = data.get("numbers") or []
    if not isinstance(seq, list) or not all(str(x).isdigit() for x in seq):
        raise HTTPException(status_code=400, detail="numbers must be a list of ints (1..4)")

    msg = f"📡 Espelho Fan Tan — sequência detectada: {seq}"
    print(msg)

    if TARGET_CHANNEL:
        await tg_send(TARGET_CHANNEL, msg)

    return {"ok": True, "mirrored": seq}

# ===============================
# 🧩 Webhook Padrão (Telegram)
# ===============================
@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    update = await request.json()
    print("📩 Update recebido (Telegram):", update)
    return {"ok": True}