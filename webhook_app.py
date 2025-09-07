import os
from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.utils.executor import start_webhook

# -----------------------------
# Configurações principais
# -----------------------------
TOKEN = os.getenv("TG_BOT_TOKEN")
WEBHOOK_HOST = os.getenv("PUBLIC_URL")   # URL do Render
WEBHOOK_PATH = f"/webhook/{TOKEN}"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

app = FastAPI()

# -----------------------------
# Handlers do bot
# -----------------------------
@dp.message_handler(commands=["start"])
async def start_cmd(message: Message):
    await message.answer("🚀 Bot funcionando! Você está conectado com sucesso.")

@dp.message_handler(commands=["ping"])
async def ping_cmd(message: Message):
    await message.answer("🏓 pong")

# Exemplo: ecoa qualquer mensagem de texto
@dp.message_handler(content_types=types.ContentTypes.TEXT)
async def echo(message: Message):
    await message.answer(f"Você disse: {message.text}")

# -----------------------------
# Rotas FastAPI
# -----------------------------
@app.on_event("startup")
async def on_startup():
    # Remove webhooks antigos e define o novo
    await bot.delete_webhook()
    await bot.set_webhook(WEBHOOK_URL)
    print(f"🌐 Webhook definido em {WEBHOOK_URL}")

@app.on_event("shutdown")
async def on_shutdown():
    await bot.delete_webhook()

@app.post(WEBHOOK_PATH)
async def webhook(request: Request):
    data = await request.json()
    update = types.Update(**data)
    await dp.process_update(update)
    return {"ok": True}

@app.get("/")
async def root():
    return {"ok": True}