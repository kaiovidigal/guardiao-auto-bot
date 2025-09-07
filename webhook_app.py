# webhook_app.py
import os
import json
import logging
from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message

# -----------------------------
# Configuração e logs
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("guardiao-webhook")

TOKEN = os.getenv("TG_BOT_TOKEN")
PUBLIC_URL = (os.getenv("PUBLIC_URL") or "").rstrip("/")

if not TOKEN:
    raise RuntimeError("Faltando variável de ambiente TG_BOT_TOKEN")
if not PUBLIC_URL:
    log.warning("PUBLIC_URL não definido (defina no Render depois da 1ª implantação).")

WEBHOOK_PATH = f"/webhook/{TOKEN}"
WEBHOOK_URL = f"{PUBLIC_URL}{WEBHOOK_PATH}" if PUBLIC_URL else None

bot = Bot(token=TOKEN, parse_mode=types.ParseMode.HTML)
dp = Dispatcher(bot)
app = FastAPI()

# -----------------------------
# Handlers básicos do bot
# -----------------------------
@dp.message_handler(commands=["start"])
async def start_cmd(message: Message):
    await message.answer("🚀 Bot online! Webhook configurado com sucesso.")

@dp.message_handler(commands=["ping"])
async def ping_cmd(message: Message):
    await message.answer("🏓 pong")

@dp.message_handler(content_types=types.ContentTypes.TEXT)
async def echo(message: Message):
    await message.answer(f"Você disse: <b>{message.text}</b>")

# (Opcional) Se quiser testar recebendo posts de um canal:
# - coloque o bot como administrador do canal
# - defina CHANNEL_ID nas variáveis de ambiente
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))
@dp.channel_post_handler(content_types=types.ContentTypes.TEXT)
async def on_channel_post(msg: types.Message):
    if CHANNEL_ID and msg.chat.id == CHANNEL_ID:
        log.info("Recebi mensagem do canal: %s", msg.text)

# -----------------------------
# FastAPI (lifecycle + rotas)
# -----------------------------
@app.on_event("startup")
async def on_startup():
    # Remove webhooks antigos e define o atual
    await bot.delete_webhook(drop_pending_updates=True)
    if WEBHOOK_URL:
        await bot.set_webhook(WEBHOOK_URL)
        log.info("🌐 Webhook definido em %s", WEBHOOK_URL)
    else:
        log.warning("Webhook NÃO definido (PUBLIC_URL vazio).")

@app.on_event("shutdown")
async def on_shutdown():
    await bot.delete_webhook()

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.get("/")
async def root():
    return {"ok": True, "msg": "Guardião webhook ok"}

@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    # Telegram às vezes envia como bytes; tratamos ambos.
    body = await request.body()
    data = json.loads(body.decode("utf-8")) if body else await request.json()

    update = types.Update(**data)

    # 🔧 Correção: define o contexto atual para o aiogram v2
    Bot.set_current(bot)
    Dispatcher.set_current(dp)

    await dp.process_update(update)
    return {"ok": True}