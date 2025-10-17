from fastapi import FastAPI, Request, HTTPException
import httpx
import os

app = FastAPI()

# Variáveis de ambiente
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "meusegredo123")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "")

# Função para enviar mensagem ao Telegram
async def tg_send(chat_id, text):
    if not TG_BOT_TOKEN:
        print("⚠️ TG_BOT_TOKEN não configurado.")
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as cli:
        await cli.post(url, json={"chat_id": chat_id, "text": text})

# ======== Healthcheck ========
@app.get("/health")
async def health():
    return {"ok": True, "status": "GuardiAo Bot ativo"}

# ========  Modo espelho: envia números crus ao canal do Telegram  ========
@app.post("/mirror/fantan/{token}")
async def mirror_fantan(token: str, request: Request):
    """
    Recebe lista de números (ex: {"numbers":[2,4]})
    e apenas espelha no canal do Telegram definido em TARGET_CHANNEL.
    """
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()
    seq = data.get("numbers") or []
    if not seq:
        return {"ok": False, "error": "no_numbers"}

    msg = f"📡 Espelho Fan Tan — sequência detectada: {seq}"
    print(msg)

    # Envia pro canal (se configurado)
    if TARGET_CHANNEL:
        await tg_send(TARGET_CHANNEL, msg)

    return {"ok": True, "mirrored": seq}

# ========  Webhook padrão (para compatibilidade)  ========
@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    """
    Rota padrão do Telegram (mantida para não quebrar setup).
    """
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    update = await request.json()
    print("📩 Update recebido:", update)
    return {"ok": True}