# -*- coding: utf-8 -*-
# webhook_app.py
import os, re, json, time, logging
from collections import deque
from typing import Optional, List, Tuple

from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# ================= LOG =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("guardiao-auto")

# =============== CONFIG =================
BOT_TOKEN   = os.getenv("TG_BOT_TOKEN")
PUBLIC_URL  = (os.getenv("PUBLIC_URL") or "").rstrip("/")
CHANNEL_A   = int(os.getenv("CHANNEL_A_ID", "0"))     # onde chegam os sinais
CHANNEL_B   = int(os.getenv("CHANNEL_B_ID", "0"))     # onde publica (opcional)

if not BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN nas variáveis de ambiente.")
if not PUBLIC_URL.startswith("http"):
    log.warning("PUBLIC_URL ausente ou inválida. Defina ex.: https://guardiao-auto-bot.onrender.com")

bot = Bot(token=BOT_TOKEN, parse_mode=types.ParseMode.HTML)
dp  = Dispatcher(bot)

# ============== PERSISTÊNCIA ============
os.makedirs("data", exist_ok=True)
STATE_FILE = "data/state.json"
RISK_FILE  = "data/risk.json"

state = {
    "seguir_sinal": True,
    "stake_base": 5.00,         # valor total da 1ª tentativa
    "gales_max": 1,             # quantos gales (total de tentativas = gales_max+1)
    "ciclo_max": 1,             # quantos ciclos seguidos tenta recuperar
    "gale_mult": 2.0,           # multiplicador rápido nos botões
    "multipliers": [1.0, 2.0],  # multiplicadores por tentativa
    "owner_id": None,           # preenchido no 1º /start
    "cooldown_until": 0.0,      # simples anti-spam
}
risk = {
    "bankroll": 100.00,
    "session_pnl": 0.0,
    "stop_win": 50.00,
    "stop_loss": 50.00,
    "odds_total": 3.85,         # payout por número
    "prev_cycle_loss": 0.0,     # prejuízo carregado entre ciclos
    "open": None,               # operação em aberto (simulada)
}

def _load_json(path, fallback):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                fallback.update(json.load(f))
    except Exception as e:
        log.warning("Falha ao carregar %s: %s", path, e)

def _save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Falha ao salvar %s: %s", path, e)

_load_json(STATE_FILE, state)
_load_json(RISK_FILE, risk)

def save_state(): _save_json(STATE_FILE, state)
def save_risk():  _save_json(RISK_FILE, risk)

# ======== MÉTRICAS simples (opcional) ========
hist_short = deque(maxlen=30)   # 0/1 para últimas entradas
ultimos_numeros = deque(maxlen=120)   # guarda sequência (se quiser usar depois)

# ============== PARSERS ==============
re_sinal   = re.compile(r"\bENTRADA\s+CONFIRMADA\b", re.I)
re_seq     = re.compile(r"Sequ[eê]ncia[:\s]*([^\n]+)", re.I)
re_apos    = re.compile(r"Entrar\s+ap[oó]s\s+o\s+([1-4])", re.I)
re_alvos   = re.compile(r"apostar\s+em\s+Ssh\s+([1-4])[\s\-\|]+([1-4])[\s\-\|]+([1-4])", re.I)
re_red     = re.compile(r"\b(RED|NEUTRO)\b", re.I)

def eh_sinal(txt:str) -> bool:
    return bool(re_sinal.search(txt or ""))

def extrai_regra_sinal(txt:str) -> Optional[Tuple[int,List[int]]]:
    """
    Esperado algo como:
    'Entrar após o 2 apostar em Ssh 4-3-2'
    Retorna (apos=2, alvos=[4,3,2]) ou None se inválido.
    """
    m1 = re_apos.search(txt or "")
    m2 = re_alvos.search(txt or "")
    if not m1 or not m2:
        return None
    apos = int(m1.group(1))
    alvos = [int(m2.group(1)), int(m2.group(2)), int(m2.group(3))]
    return apos, alvos

def eh_resultado(txt:str) -> Optional[int]:
    """
    Detecta resultado de fechamento:
      retorna 1 para GREEN, 0 para RED/NEUTRO, None caso não seja fechamento.
    """
    up = (txt or "").upper()
    if "APOSTA ENCERRADA" in up:
        if "GREEN" in up or "✅" in up:
            return 1
        if re_red.search(up):
            return 0
        # se não achou palavra, mas veio 'APOSTA ENCERRADA' sem GREEN -> considerar RED
        return 0
    return None

# ============== CÁLCULOS DE STAKE ==============
def lucro_liquido_no_acerto(por_num: float, odds_total: float) -> float:
    # ganha odds_total*por_num no nº vencedor e perde 2*por_num nos outros dois
    return round((odds_total - 3.0) * por_num, 2)

def plano_por_tentativa(base_total: float, mult: float, odds_total: float):
    stake_total = round(base_total * mult, 2)
    por_num = round(stake_total / 3.0, 2)
    lucro = lucro_liquido_no_acerto(por_num, odds_total)
    return stake_total, por_num, lucro

def resumo_plano_text(multipliers: List[float], base_total: float, odds_total: float):
    partes=[]; tot=0.0
    for m in multipliers:
        s, _, _ = plano_por_tentativa(base_total, m, odds_total)
        partes.append(f"{s:.2f}")
        tot += s
    return f"{' → '.join(partes)} = <b>{tot:.2f}</b>"

# ============== UI / PAINEL ==============
def kb_painel():
    seguir = "🟢 Seguir: ON" if state["seguir_sinal"] else "⚪️ Seguir: OFF"
    kb = InlineKeyboardMarkup(row_width=3)
    kb.row(InlineKeyboardButton("💰 -", callback_data="stake_-"),
           InlineKeyboardButton(f"Stake: R${state['stake_base']:.2f}", callback_data="noop"),
           InlineKeyboardButton("💰 +", callback_data="stake_+"))
    kb.row(InlineKeyboardButton("♻️ -", callback_data="gales_-"),
           InlineKeyboardButton(f"Gales: {state['gales_max']}", callback_data="noop"),
           InlineKeyboardButton("♻️ +", callback_data="gales_+"))
    kb.row(InlineKeyboardButton("🔁 -", callback_data="ciclo_-"),
           InlineKeyboardButton(f"Ciclo: {state['ciclo_max']}", callback_data="noop"),
           InlineKeyboardButton("🔁 +", callback_data="ciclo_+"))
    kb.row(InlineKeyboardButton("📈 -", callback_data="gmx_-"),
           InlineKeyboardButton(f"Mult x{state['gale_mult']:.2f}", callback_data="noop"),
           InlineKeyboardButton("📈 +", callback_data="gmx_+"))
    kb.row(
        InlineKeyboardButton("✖️ 1x", callback_data="preset_mult_1"),
        InlineKeyboardButton("✖️ 2x", callback_data="preset_mult_2"),
        InlineKeyboardButton("✖️ 3x", callback_data="preset_mult_3"),
        InlineKeyboardButton("✖️ 4x", callback_data="preset_mult_4"),
    )
    kb.row(InlineKeyboardButton(seguir, callback_data="toggle_seg"))
    kb.row(InlineKeyboardButton("🔄 Atualizar", callback_data="refresh"))
    return kb

def header_info() -> str:
    return (
        "🤖 <b>Guardião de Risco (AUTOMÁTICO)</b>\n"
        "• Lê apenas <b>ENTRADA CONFIRMADA</b> no canal A\n"
        "• Publica o plano no seu chat e (opcional) no canal B\n"
        "• /painel para stake/gales/ciclo/multiplicadores/odds/stops\n"
        f"• Canal A: <code>{CHANNEL_A}</code>\n"
        f"• Canal B: <code>{CHANNEL_B}</code>\n"
    )

@dp.message_handler(commands=["start"])
async def cmd_start(m: types.Message):
    if not state.get("owner_id"):
        state["owner_id"] = m.chat.id
        save_state()
    await m.answer(header_info(), disable_web_page_preview=True)
    await cmd_painel(m)

@dp.message_handler(commands=["painel","config"])
async def cmd_painel(m: types.Message):
    mults = ", ".join(f"{x:.2f}" for x in state["multipliers"])
    txt = (
        "⚙️ <b>PAINEL</b>\n"
        f"💰 Base: <b>{state['stake_base']:.2f}</b> | ♻️ Gales: <b>{state['gales_max']}</b> | 🔁 Ciclo: <b>{state['ciclo_max']}</b>\n"
        f"✖️ Mults: <b>{mults}</b> (use /mult 1,2,3,4)\n"
        f"📈 Mult rápido: x{state['gale_mult']:.2f}\n"
        f"🎯 Odds por número: <b>{risk['odds_total']:.2f}x</b>\n"
        f"💼 Banca: <b>R${risk['bankroll']:.2f}</b> | PnL Sessão: <b>{risk['session_pnl']:.2f}</b>\n"
        f"🧮 Plano: {resumo_plano_text(state['multipliers'][:state['gales_max']+1], state['stake_base'], risk['odds_total'])}\n"
        f"Seguir sinal: <b>{'ON' if state['seguir_sinal'] else 'OFF'}</b>"
    )
    await m.answer(txt, reply_markup=kb_painel())

@dp.callback_query_handler(lambda c: True)
async def on_cb(call: types.CallbackQuery):
    data = call.data; changed=False
    if data=="stake_+": state["stake_base"]=round(state["stake_base"]+1.0,2); changed=True
    elif data=="stake_-": state["stake_base"]=max(1.0, round(state["stake_base"]-1.0,2)); changed=True
    elif data=="gales_+": state["gales_max"]=min(3, state["gales_max"]+1); changed=True
    elif data=="gales_-": state["gales_max"]=max(0, state["gales_max"]-1); changed=True
    elif data=="ciclo_+": state["ciclo_max"]=min(10, state["ciclo_max"]+1); changed=True
    elif data=="ciclo_-": state["ciclo_max"]=max(1, state["ciclo_max"]-1); changed=True
    elif data=="gmx_+": state["gale_mult"]=round(min(4.0, state["gale_mult"]+0.5),2); changed=True
    elif data=="gmx_-": state["gale_mult"]=round(max(1.0, state["gale_mult"]-0.5),2); changed=True
    elif data=="toggle_seg": state["seguir_sinal"]=not state["seguir_sinal"]; changed=True
    elif data.startswith("preset_mult_"):
        k=int(data.split("_")[-1])  # 1..4
        lst=[1.0]; 
        for _ in range(3): lst.append(float(k))
        state["multipliers"]=lst[:state["gales_max"]+1]; changed=True

    if changed:
        save_state()
        try: await call.message.edit_reply_markup(kb_painel())
        except: pass
        await call.answer("Atualizado!")
    else:
        if data=="refresh":
            try: await call.message.edit_reply_markup(kb_painel())
            except: pass
            await call.answer("OK")
        else:
            await call.answer()

# ======== COMANDOS =========
@dp.message_handler(commands=["odds"])
async def cmd_odds(m: types.Message):
    try:
        v=float((m.get_args() or "").replace(",", "."))
        if v<3.0: raise ValueError()
        risk["odds_total"]=round(v,2); save_risk()
        await m.answer(f"✅ Odds por número ajustadas para <b>{risk['odds_total']:.2f}x</b>")
    except:
        await m.answer("Use: /odds 3.85")

@dp.message_handler(commands=["banca"])
async def cmd_banca(m: types.Message):
    try:
        v=float((m.get_args() or "").replace(",", "."))
        if v<=0: raise ValueError()
        risk["bankroll"]=round(v,2); save_risk()
        await m.answer(f"✅ Banca definida em <b>R${risk['bankroll']:.2f}</b>")
    except:
        await m.answer("Use: /banca 1000")

@dp.message_handler(commands=["stopwin"])
async def cmd_stopwin(m: types.Message):
    try:
        v=float((m.get_args() or "").replace(",", "."))
        if v<=0: raise ValueError()
        risk["stop_win"]=round(v,2); save_risk()
        await m.answer(f"✅ Stop Win: <b>R${risk['stop_win']:.2f}</b>")
    except:
        await m.answer("Use: /stopwin 50")

@dp.message_handler(commands=["stoploss"])
async def cmd_stoploss(m: types.Message):
    try:
        v=float((m.get_args() or "").replace(",", "."))
        if v<=0: raise ValueError()
        risk["stop_loss"]=round(v,2); save_risk()
        await m.answer(f"✅ Stop Loss: <b>R${risk['stop_loss']:.2f}</b>")
    except:
        await m.answer("Use: /stoploss 50")

@dp.message_handler(commands=["mult"])
async def cmd_mult(m: types.Message):
    try:
        raw=(m.get_args() or "").replace(" ", "")
        lst=[float(x.replace(",", ".")) for x in raw.split(",") if x]
        if not lst: raise ValueError()
        if len(lst)>4: lst=lst[:4]
        state["multipliers"]=lst[:state["gales_max"]+1]
        save_state()
        await m.answer(
            f"✅ Multiplicadores: <b>{', '.join(f'{x:.2f}' for x in state['multipliers'])}</b>"
        )
    except:
        await m.answer("Use: /mult 1,2,2.5,3 (até 4 valores)")

@dp.message_handler(commands=["saldo"])
async def cmd_saldo(m: types.Message):
    await m.answer(
        "💼 <b>Controle</b>\n"
        f"• Banca: <b>R${risk['bankroll']:.2f}</b>\n"
        f"• PnL Sessão: <b>{risk['session_pnl']:.2f}</b>\n"
        f"• Stop Win: <b>R${risk['stop_win']:.2f}</b> | Stop Loss: <b>R${risk['stop_loss']:.2f}</b>\n"
        f"• Odds (nº): <b>{risk['odds_total']:.2f}x</b>"
    )

# ============== STOP-WIN/LOSS ==========
async def check_stops_and_pause():
    if risk["session_pnl"] >= risk["stop_win"]:
        state["seguir_sinal"]=False; save_state(); save_risk()
        await publish_all("✅ <b>STOP WIN atingido</b>. Pausando entradas.")
        return True
    if risk["session_pnl"] <= -risk["stop_loss"]:
        state["seguir_sinal"]=False; save_state(); save_risk()
        await publish_all("⛔ <b>STOP LOSS atingido</b>. Pausando entradas.")
        return True
    return False

# ============== EXECUÇÃO SIMULADA =======
def abrir_operacao(apos:int, alvos:List[int], base_total:float, multipliers:List[float]):
    op = {
        "apos": apos,
        "alvos": alvos,
        "base": round(base_total,2),
        "mult": multipliers[:],
        "step": 0,
        "closed": False,
        "cycle_left": state["ciclo_max"],
    }
    risk["open"]=op
    save_risk()
    return op

def valor_tentativa(op):
    m = op["mult"][op["step"]]
    return plano_por_tentativa(op["base"], m, risk["odds_total"])

def total_gasto_ate(op, step_exclusive:int) -> float:
    """soma stake_total das tentativas já gastas (0..step-1)"""
    tot=0.0
    for i in range(step_exclusive):
        s,_,_ = plano_por_tentativa(op["base"], op["mult"][i], risk["odds_total"])
        tot += s
    return round(tot,2)

async def publicar_plano(op):
    s0, per0, _ = valor_tentativa(op)
    plano_txt = resumo_plano_text(op["mult"], op["base"], risk["odds_total"])
    msg_txt = (
        "🟢 <b>CONFIRMAR</b>\n"
        f"🎯 Alvos: <b>{op['alvos'][0]}-{op['alvos'][1]}-{op['alvos'][2]}</b>\n"
        f"🧷 Após: <b>{op['apos']}</b>\n"
        f"💵 Tentativa 1 (total): <b>R${s0:.2f}</b> (≈ <i>{per0:.2f} por número</i>)\n"
        f"🧮 Plano: {plano_txt}\n"
        f"📈 Odds por número: <b>{risk['odds_total']:.2f}x</b>\n"
        f"💼 Sessão: <b>{risk['session_pnl']:.2f}</b> | Banca: <b>{risk['bankroll']:.2f}</b>"
    )
    await publish_all(msg_txt)

async def publish_all(text:str):
    # sempre manda pro "owner" (1º /start). Se tiver CHANNEL_B, manda lá também.
    if state.get("owner_id"):
        try: await bot.send_message(state["owner_id"], text)
        except Exception as e: log.warning("DM owner falhou: %s", e)
    if CHANNEL_B:
        try: await bot.send_message(CHANNEL_B, text)
        except Exception as e: log.warning("Canal B falhou: %s", e)

async def avancar_depois_de_red():
    op=risk.get("open")
    if not op or op["closed"]: return
    op["step"] += 1
    if op["step"] >= len(op["mult"]):
        # fechou ciclo com prejuízo
        preju = total_gasto_ate(op, len(op["mult"]))
        risk["session_pnl"] = round(risk["session_pnl"] - preju, 2)
        risk["bankroll"]    = round(risk["bankroll"]    - preju, 2)
        op["closed"]=True
        risk["open"]=None
        save_risk()
        await publish_all(
            f"❌ <b>RED</b> | Perda: <b>R${preju:.2f}</b> | "
            f"Sessão: <b>{risk['session_pnl']:.2f}</b> | Banca: <b>{risk['bankroll']:.2f}</b>"
        )
        await check_stops_and_pause()
    else:
        save_risk()

async def fechar_com_green():
    op=risk.get("open")
    if not op or op["closed"]: return
    stake_total, por_num, lucro = valor_tentativa(op)
    gastos_previos = total_gasto_ate(op, op["step"])
    pnl = round(lucro - gastos_previos, 2)
    risk["session_pnl"] = round(risk["session_pnl"] + pnl, 2)
    risk["bankroll"]    = round(risk["bankroll"] + pnl, 2)
    op["closed"]=True
    risk["open"]=None
    save_risk()
    await publish_all(
        f"✅ <b>GREEN</b> (step {op['step']+1}) | Lucro: <b>R${pnl:.2f}</b> | "
        f"Sessão: <b>{risk['session_pnl']:.2f}</b> | Banca: <b>{risk['bankroll']:.2f}</b>"
    )
    await check_stops_and_pause()

# ============== HANDLER DO CANAL A ===========
@dp.channel_post_handler(content_types=["text"])
async def on_channel_post(msg: types.Message):
    if not CHANNEL_A or msg.chat.id != CHANNEL_A:
        return
    txt = (msg.text or "").strip()
    if not txt:
        return

    # resultado fecha operação
    r = eh_resultado(txt)
    if r is not None:
        hist_short.append(r)
        if risk.get("open") and not risk["open"]["closed"]:
            if r==1: await fechar_com_green()
            else:    await avancar_depois_de_red()
        return

    # entrada confirmada -> abre operação (sem confiança)
    if not eh_sinal(txt):
        return
    if not state["seguir_sinal"]:
        return
    if time.time() < state.get("cooldown_until", 0.0):
        return
    if await check_stops_and_pause():
        return

    regra = extrai_regra_sinal(txt)
    if not regra:
        log.info("Sinal sem padrão esperado: %s", txt)
        return
    apos_num, alvos = regra
    if len(alvos) != 3:
        return

    mults = state["multipliers"][:state["gales_max"]+1] or [1.0]
    base_total = state["stake_base"]

    op = abrir_operacao(apos_num, alvos, base_total, mults)
    await publicar_plano(op)

    # anti-spam simples
    state["cooldown_until"]=time.time()+5
    save_state()

# ============== FASTAPI / WEBHOOK =========
app = FastAPI()

@app.get("/healthz")
def healthz(): 
    return {"ok": True}

@app.on_event("startup")
async def on_startup():
    if not PUBLIC_URL:
        log.warning("PUBLIC_URL não definido; defina no Render.")
        return
    await bot.delete_webhook(drop_pending_updates=True)
    await bot.set_webhook(f"{PUBLIC_URL}/webhook/{BOT_TOKEN}")
    log.info("Webhook configurado: %s/webhook/<token>", PUBLIC_URL)

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.body()
    update = types.Update(**json.loads(data.decode("utf-8")))
    # Aiogram v2 precisa do contexto explícito
    Bot.set_current(bot)
    Dispatcher.set_current(dp)
    await dp.process_update(update)
    return {"ok": True}