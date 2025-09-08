# bot.py
import os, time, json
from telethon import TelegramClient, events
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor

API_ID = int(os.getenv("TG_API_ID"))
API_HASH = os.getenv("TG_API_HASH")
BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL")   # @seu_canal_de_sinais
TARGET_CHAT = os.getenv("TARGET_CHAT")         # @seu_canal_onde_publica
CONF_LIMIAR = float(os.getenv("CONF_LIMIAR", "0.99"))

# Persistência simples
STATE_FILE = "state.json"
state = {"cooldown_until": 0, "limiar": CONF_LIMIAR}
if os.path.exists(STATE_FILE):
    state.update(json.load(open(STATE_FILE)))

def save_state():
    json.dump(state, open(STATE_FILE, "w"))

# ===== Engine de risco (versão rápida) =====
from collections import deque
hist = deque(maxlen=300)  # (win=1/0) dos últimos 300 sinais
short = deque(maxlen=30)

def winrate(d):
    if not d: return 0.0
    return sum(d)/len(d)

def volatilidade(d):
    if len(d) < 10: return 0.0
    trocas = sum(1 for i in range(1,len(d)) if d[i]!=d[i-1])
    return trocas/(len(d)-1)

def streak_loss(d):
    s=0; mx=0
    for x in d:
        if x==0: s+=1; mx=max(mx,s)
        else: s=0
    return mx

def risco_conf(short_wr, long_wr, vol, max_reds):
    base = 0.6*short_wr + 0.3*long_wr + 0.1*(1.0 - vol)
    pena = 0.0
    if max_reds>=3: pena += 0.05*(max_reds-2)   # penaliza sequências de reds
    if vol>0.6:     pena += 0.05
    conf = max(0.0, min(1.0, base - pena))
    return conf

# ===== Telegram wiring =====
user_client = TelegramClient("session", API_ID, API_HASH)
bot = Bot(token=BOT_TOKEN, parse_mode="HTML")
dp = Dispatcher(bot)

@dp.message_handler(commands=["status"])
async def cmd_status(m: types.Message):
    short_wr = winrate(short)
    long_wr  = winrate(hist)
    vol = volatilidade(list(short))
    mxr = streak_loss(list(short))
    await m.answer(
        f"📊 Short WR(30): {short_wr*100:.1f}% | Long WR(300): {long_wr*100:.1f}%\n"
        f"🔀 Volatilidade: {vol:.2f} | 📉 Max Reds: {mxr}\n"
        f"🎚️ Limiar atual: {state['limiar']*100:.2f}%"
    )

@dp.message_handler(commands=["limiar"])
async def cmd_limiar(m: types.Message):
    try:
        val = float(m.get_args().strip())/100.0
        state["limiar"] = max(0.9, min(0.999, val))  # limita 90%–99.9%
        save_state()
        await m.answer(f"✅ Limiar ajustado para {state['limiar']*100:.2f}%")
    except:
        await m.answer("Use: /limiar 99.0   (em %)")

@dp.message_handler(commands=["pausa"])
async def cmd_pausa(m: types.Message):
    state["cooldown_until"] = time.time()+int(m.get_args() or "600")
    save_state()
    await m.answer("⏸️ Entradas pausadas temporariamente.")

@dp.message_handler(commands=["retomar"])
async def cmd_ret(m: types.Message):
    state["cooldown_until"] = 0; save_state()
    await m.answer("▶️ Retomado.")

# Utilitário para publicar
async def publicar(texto):
    await bot.send_message(TARGET_CHAT, texto)

# Parse de sinais do seu canal (exemplo genérico)
def parse_signal(msg_text: str):
    # Ajuste ao seu padrão (ex: "SINAL: Azul G0", "Vermelho", etc.)
    return "SINAL" in msg_text.upper()

# Quando chegar resultado (você marca win/lose no canal ou via comando)
def parse_result(msg_text: str):
    up = msg_text.upper()
    if "GREEN" in up or "WIN" in up: return 1
    if "RED" in up or "LOSS" in up: return 0
    return None

@user_client.on(events.NewMessage(chats=SOURCE_CHANNEL))
async def on_source(event):
    txt = event.raw_text.strip()

    # Atualiza histórico se mensagem for resultado
    r = parse_result(txt)
    if r is not None:
        hist.append(r); short.append(r)
        return

    # Se for um novo SINAL, calcula risco
    if parse_signal(txt):
        if time.time() < state["cooldown_until"]:
            await publicar("neutro")
            return

        short_wr = winrate(short)
        long_wr  = winrate(hist)
        vol = volatilidade(list(short))
        mxr = streak_loss(list(short))
        conf = risco_conf(short_wr, long_wr, vol, mxr)

        # (Opcional) estimativa de empate — substitua por sua métrica real
        prob_empate = 0.0  # preencha se tiver modelo/heurística
        empate_txt = f"\n⚠️ Empate: {prob_empate*100:.1f}%" if prob_empate >= 0.5 else ""

        if conf >= state["limiar"]:
            await publicar(
                f"🎯 Chance: {conf*100:.1f}%\n🛡️ Risco: BAIXO\n📍 Ação: ENTRAR{empate_txt}"
            )
        else:
            await publicar("neutro")

async def main():
    await user_client.start()
    # roda user_client (escuta canal) e bot (comandos) juntos
    await user_client.run_until_disconnected()

if __name__ == "__main__":
    # Rode o bot (aiogram) em paralelo (ex: outro processo) ou unifique com asyncio
    # Maneira simples: suba 2 processos: `python bot.py` (para aiogram) e um `listener.py` para o Telethon
    # Para fins de demo, deixamos aqui o listener principal:
    import asyncio
    loop = asyncio.get_event_loop()
    loop.create_task(dp.start_polling())
    loop.run_until_complete(main())