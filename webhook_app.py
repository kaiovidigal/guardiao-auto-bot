# -*- coding: utf-8 -*-
# webhook_app.py — Guardião Auto (DM-only)

import os, re, json, time, logging
from typing import List, Optional, Tuple

from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# ========= LOG =========
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("guardiao-auto-dm")

# ========= CONFIG (ENV) =========
BOT_TOKEN  = os.getenv("TG_BOT_TOKEN")
PUBLIC_URL = (os.getenv("PUBLIC_URL") or "").rstrip("/")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))  # Canal de sinais A

if not BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN nas variáveis de ambiente.")
if not PUBLIC_URL.startswith("http"):
    log.warning("PUBLIC_URL ausente/invalid. Ex.: https://guardiao-auto-bot.onrender.com")

# Odds fixas por número (Fan Tan)
ODDS_TOTAL = 3.85

# ========= TELEGRAM =========
bot = Bot(token=BOT_TOKEN, parse_mode=types.ParseMode.HTML)
dp  = Dispatcher(bot)

# ========= PERSISTÊNCIA =========
os.makedirs("data", exist_ok=True)
STATE_PATH = "data/state.json"
RISK_PATH  = "data/risk.json"

state = {
    "dm_user_id": 0,
    "seguir": True,
    "stake_base": 10.00,
    "gales_max": 1,
    "ciclo_max": 1,
    "multipliers": [1.0, 3.0],  # padrão G1 = 3x
    "ciclo_mult": 3.0,
    "cooldown_until": 0.0
}
risk = {
    "bankroll": 1000.00,
    "session_pnl": 0.0,
    "stop_win": 1000.00,
    "stop_loss": 1000.00,
    "prev_cycle_loss": 0.0,
    "cycle_left": 0,
    "open": None
}

def load(path, default):
    try:
        if os.path.exists(path):
            default.update(json.load(open(path, "r", encoding="utf-8")))
    except Exception as e:
        log.warning("Falha ao carregar %s: %s", path, e)
    return default

def save(path, obj):
    try:
        json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Falha ao salvar %s: %s", path, e)

load(STATE_PATH, state)
load(RISK_PATH, risk)
def save_state(): save(STATE_PATH, state)
def save_risk():  save(RISK_PATH,  risk)

# ========= PARSERS =========
re_sinal   = re.compile(r"\bENTRADA\s+CONFIRMADA\b", re.I)
re_apos    = re.compile(r"Entrar\s+ap[oó]s\s+o\s+([1-4])", re.I)
re_alvos   = re.compile(r"apostar\s+em\s+Ssh\s+([1-4])[\s\-\|]+([1-4])[\s\-\|]+([1-4])", re.I)

# só GREEN e RED
re_close   = re.compile(r"\bAPOSTA\s+ENCERRADA\b", re.I)
re_green   = re.compile(r"(GREEN|✅)", re.I)
re_red     = re.compile(r"(RED|❌)", re.I)

def eh_sinal(txt:str) -> bool:
    return bool(re_sinal.search(txt or ""))

def extrai_regra_sinal(txt:str) -> Optional[Tuple[int, List[int]]]:
    m1 = re_apos.search(txt or "")
    m2 = re_alvos.search(txt or "")
    if not m1 or not m2:
        return None
    apos = int(m1.group(1))
    alvos = [int(m2.group(1)), int(m2.group(2)), int(m2.group(3))]
    return apos, alvos

def eh_resultado(txt:str) -> Optional[int]:
    """
    1 = GREEN, 0 = RED, None = não é fechamento.
    """
    up = (txt or "").upper()
    if not re_close.search(up):
        return None
    if re_green.search(up): return 1
    if re_red.search(up):   return 0
    return None

# ========= CÁLCULOS =========
def lucro_liquido_no_acerto(por_num: float) -> float:
    return round((ODDS_TOTAL - 3.0) * por_num, 2)

def plano_por_tentativa(base_total: float, mult: float):
    stake_total = round(base_total * mult, 2)
    por_num = round(stake_total / 3.0, 2)
    lucro = lucro_liquido_no_acerto(por_num)
    return stake_total, por_num, lucro

def resumo_plano(mults: List[float], base_total: float) -> str:
    partes=[]; tot=0.0
    for m in mults:
        s, _, _ = plano_por_tentativa(base_total, m)
        partes.append(f"{s:.2f}")
        tot += s
    return f"{' → '.join(partes)} = <b>{tot:.2f}</b>"

def total_gasto_ate(mults: List[float], base_total: float, step_exclusive:int) -> float:
    tot=0.0
    for i in range(step_exclusive):
        s,_,_ = plano_por_tentativa(base_total, mults[i])
        tot += s
    return round(tot,2)

# ========= UI =========
def kb_painel():
    kb = InlineKeyboardMarkup(row_width=3)
    kb.row(
        InlineKeyboardButton("💼 Banca", callback_data="set_banca"),
        InlineKeyboardButton("🟢 Stop Win", callback_data="set_sw"),
        InlineKeyboardButton("🔴 Stop Loss", callback_data="set_sl"),
    )
    kb.row(
        InlineKeyboardButton("Stake -", callback_data="stake_-"),
        InlineKeyboardButton(f"Stake: R${state['stake_base']:.2f}", callback_data="noop"),
        InlineKeyboardButton("Stake +", callback_data="stake_+"),
    )
    kb.row(
        InlineKeyboardButton("Gales -", callback_data="gales_-"),
        InlineKeyboardButton(f"Gales: {state['gales_max']}", callback_data="noop"),
        InlineKeyboardButton("Gales +", callback_data="gales_+"),
    )
    kb.row(
        InlineKeyboardButton("Ciclo -", callback_data="ciclo_-"),
        InlineKeyboardButton(f"Ciclo: {state['ciclo_max']}", callback_data="noop"),
        InlineKeyboardButton("Ciclo +", callback_data="ciclo_+"),
    )
    seg = "🟢 Seguir: ON" if state["seguir"] else "⚪️ Seguir: OFF"
    kb.row(InlineKeyboardButton(seg, callback_data="toggle"))
    kb.row(InlineKeyboardButton("🔄 Atualizar", callback_data="refresh"))
    return kb

def painel_texto():
    mults = ", ".join(f"{x:.2f}" for x in state["multipliers"][:state["gales_max"]+1])
    return (
        "⚙️ <b>PAINEL</b>\n"
        f"💰 Base: <b>{state['stake_base']:.2f}</b> | ♻️ Gales: <b>{state['gales_max']}</b> | 🔁 Ciclo: <b>{state['ciclo_max']}</b>\n"
        f"✖️ Mults (G0..Gn): <b>{mults}</b>\n"
        f"🎯 Odds fixas: <b>{ODDS_TOTAL:.2f}x</b>\n"
        f"💼 Banca: <b>R${risk['bankroll']:.2f}</b> | Sessão: <b>{risk['session_pnl']:.2f}</b>\n"
        f"🧮 Plano (até G{state['gales_max']}): {resumo_plano(state['multipliers'][:state['gales_max']+1], state['stake_base'])}\n"
        f"Seguir: <b>{'ON' if state['seguir'] else 'OFF'}</b>"
    )

@dp.message_handler(commands=["start"])
async def cmd_start(m: types.Message):
    state["dm_user_id"] = m.chat.id
    save_state()
    await m.answer(
        "🤖 <b>Guardião Auto</b>\n"
        "• Lê <b>ENTRADA CONFIRMADA</b> no canal A e executa simulação aqui.\n"
        "• Só GREEN/RED alteram a banca.\n"
        f"• Canal A: <code>{CHANNEL_ID}</code>",
        disable_web_page_preview=True
    )
    await m.answer(painel_texto(), reply_markup=kb_painel())

@dp.message_handler(commands=["painel","config","status"])
async def cmd_painel(m: types.Message):
    await m.answer(painel_texto(), reply_markup=kb_painel())

# ========= CORE: abrir/fechar =========
def abrir_operacao(apos:int, alvos:List[int]):
    base = state["stake_base"]
    mults = state["multipliers"][:state["gales_max"]+1] or [1.0]
    op = {"apos": apos, "alvos": alvos, "base": base, "mult": mults, "step": 0, "closed": False}
    risk["open"] = op
    save_risk()
    return op

async def publicar_plano(op):
    s0, per0, _ = plano_por_tentativa(op["base"], op["mult"][0])
    plano_txt = resumo_plano(op["mult"], op["base"])
    txt = (
        "🟢 <b>CONFIRMAR</b>\n"
        f"🎯 Alvos: <b>{op['alvos'][0]}-{op['alvos'][1]}-{op['alvos'][2]}</b>\n"
        f"🧷 Após: <b>{op['apos']}</b>\n"
        f"💵 Tentativa 1: <b>R${s0:.2f}</b> (≈ {per0:.2f}/número)\n"
        f"🧮 Plano: {plano_txt}\n"
        f"💼 Banca: <b>{risk['bankroll']:.2f}</b> | Sessão: <b>{risk['session_pnl']:.2f}</b>"
    )
    if state["dm_user_id"]:
        await bot.send_message(state["dm_user_id"], txt)

async def fechar_com_green():
    op = risk.get("open")
    if not op or op["closed"]: return
    stake_total, _, lucro = plano_por_tentativa(op["base"], op["mult"][op["step"]])
    gastos_previos = total_gasto_ate(op["mult"], op["base"], op["step"])
    pnl = round(lucro - gastos_previos, 2)
    risk["session_pnl"] = round(risk["session_pnl"] + pnl, 2)
    risk["bankroll"]    = round(risk["bankroll"] + pnl, 2)
    op["closed"] = True
    risk["open"] = None
    save_risk()
    if state["dm_user_id"]:
        await bot.send_message(state["dm_user_id"],
            f"✅ <b>GREEN</b> | Lucro: R${pnl:.2f} | Sessão: {risk['session_pnl']:.2f} | Banca: {risk['bankroll']:.2f}")

async def avancar_depois_de_red():
    op = risk.get("open")
    if not op or op["closed"]: return
    preju = total_gasto_ate(op["mult"], op["base"], op["step"]+1)
    risk["session_pnl"] = round(risk["session_pnl"] - preju, 2)
    risk["bankroll"]    = round(risk["bankroll"] - preju, 2)
    op["closed"] = True
    risk["open"] = None
    save_risk()
    if state["dm_user_id"]:
        await bot.send_message(state["dm_user_id"],
            f"❌ <b>RED</b> | Perda: R${preju:.2f} | Sessão: {risk['session_pnl']:.2f} | Banca: {risk['bankroll']:.2f}")

# ========= HANDLER DO CANAL =========
async def _process_channel_text(msg: types.Message):
    if not CHANNEL_ID or msg.chat.id != CHANNEL_ID:
        return
    txt = (msg.text or "").strip()
    if not txt: return

    r = eh_resultado(txt)
    if r is not None:
        if risk.get("open") and not risk["open"]["closed"]:
            if r == 1: await fechar_com_green()
            else:      await avancar_depois_de_red()
        return

    if not eh_sinal(txt): return
    if not state["seguir"]: return
    if time.time() < state.get("cooldown_until", 0.0): return

    regra = extrai_regra_sinal(txt)
    if not regra: return
    apos, alvos = regra
    if len(alvos) != 3: return

    op = abrir_operacao(apos, alvos)
    await publicar_plano(op)
    state["cooldown_until"] = time.time() + 5
    save_state()

@dp.channel_post_handler(content_types=["text"])
async def on_channel_post(msg: types.Message):
    await _process_channel_text(msg)

@dp.edited_channel_post_handler(content_types=["text"])
async def on_channel_edit(msg: types.Message):
    await _process_channel_text(msg)

# ========= FASTAPI / WEBHOOK =========
app = FastAPI()

@app.get("/healthz")
def healthz(): 
    return {"ok": True}

@app.on_event("startup")
async def on_startup():
    if not PUBLIC_URL: return
    await bot.delete_webhook(drop_pending_updates=True)
    await bot.set_webhook(f"{PUBLIC_URL}/webhook/{BOT_TOKEN}")
    log.info("Webhook configurado em %s/webhook/<token>", PUBLIC_URL)

@app.post(f"/webhook/{BOT_TOKEN}")
async def tg_webhook(request: Request):
    data = await request.body()
    update = types.Update(**json.loads(data.decode("utf-8")))
    Bot.set_current(bot)
    Dispatcher.set_current(dp)
    await dp.process_update(update)
    return {"ok": True}