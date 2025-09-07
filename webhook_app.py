# -*- coding: utf-8 -*-
# webhook_app.py — Guardião Auto (DM-only) com gatilho de G1

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
    log.warning("PUBLIC_URL ausente/invalid. Ex.: https://seu-app.onrender.com")

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
    "dm_user_id": 0,           # preenchido no /start
    "seguir": True,            # seguir sinais
    "stake_base": 10.00,       # valor total da tentativa 1 (3 números somados)
    "gales_max": 1,            # G0..G3  (tentativas = gales_max+1)
    "ciclo_max": 1,            # quantos sinais serão usados para recuperar
    "multipliers": [1.0, 3.0], # padrão G1=3x
    "ciclo_mult": 3.0,         # multiplica a perda do ciclo anterior
    "cooldown_until": 0.0
}
risk = {
    "bankroll": 1000.00,
    "session_pnl": 0.0,
    "stop_win": 1000.00,
    "stop_loss": 1000.00,
    "prev_cycle_loss": 0.0,    # perda carregada para recuperar
    "cycle_left": 0,           # quantos sinais restam no modo ciclo
    "open": None               # operação aberta
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
re_close   = re.compile(r"\bAPOSTA\s+ENCERRADA\b", re.I)
re_green   = re.compile(r"\bGREEN\b|✅", re.I)
re_red     = re.compile(r"\bRED\b|\bNEUTRO\b|❌", re.I)

# >>> NOVO: padrões para detectar mensagem de G1 no canal
re_g1_hint = re.compile(
    r"(?:estamos\s+no|vamos\s+para\s+|indo\s+para\s+)?(?:\b1\s*[ºo]?\s*gale\b|\bg1\b)",
    re.I
)

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
    1 = GREEN, 0 = RED/NEUTRO, None = não é fechamento.
    Aceita qualquer variação desde que contenha 'APOSTA ENCERRADA'.
    """
    up = (txt or "").upper()
    if not re_close.search(up):
        return None
    if re_green.search(up): return 1
    if re_red.search(up):  return 0
    return 0

# ========= CÁLCULOS =========
def lucro_liquido_no_acerto(por_num: float) -> float:
    # ganha ODDS_TOTAL*por_num no nº vencedor e perde 2*por_num nos outros dois
    return round((ODDS_TOTAL - 3.0) * por_num, 2)  # 0.85 * por_num

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
        InlineKeyboardButton("✖️ 1x", callback_data="preset_1"),
        InlineKeyboardButton("✖️ 2x", callback_data="preset_2"),
        InlineKeyboardButton("✖️ 3x", callback_data="preset_3"),
        InlineKeyboardButton("✖️ 4x", callback_data="preset_4"),
    )
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
    kb.row(
        InlineKeyboardButton("Ciclo Mult -", callback_data="cm_-"),
        InlineKeyboardButton(f"Ciclo Mult x{state['ciclo_mult']:.2f}", callback_data="noop"),
        InlineKeyboardButton("Ciclo Mult +", callback_data="cm_+"),
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
        f"📌 Ciclo Mult: <b>x{state['ciclo_mult']:.2f}</b>\n"
        f"🎯 Odds por nº (fixo): <b>{ODDS_TOTAL:.2f}x</b>\n"
        f"💼 Banca: <b>R${risk['bankroll']:.2f}</b> | PnL Sessão: <b>{risk['session_pnl']:.2f}</b>\n"
        f"🧮 Plano (até G1): {resumo_plano(state['multipliers'][:state['gales_max']+1], state['stake_base'])}\n"
        f"Seguir: <b>{'ON' if state['seguir'] else 'OFF'}</b>"
    )

@dp.message_handler(commands=["start"])
async def cmd_start(m: types.Message):
    state["dm_user_id"] = m.chat.id
    save_state()
    await m.answer(
        "🤖 <b>Guardião Auto (DM)</b>\n"
        "• Lê <b>ENTRADA CONFIRMADA</b> no canal A e executa simulação aqui.\n"
        "• Odds por número fixas (3.85x).\n"
        "• Ciclo de recuperação: perda × Ciclo Mult (padrão 3x).\n"
        f"• Canal A: <code>{CHANNEL_ID}</code>",
        disable_web_page_preview=True
    )
    await m.answer(painel_texto(), reply_markup=kb_painel())

@dp.message_handler(commands=["painel","config","status"])
async def cmd_painel(m: types.Message):
    await m.answer(painel_texto(), reply_markup=kb_painel())

# ====== entrada de números por prompt ======
AWAIT_NUMERIC = {}  # {user_id: "banca"|"sw"|"sl"}

@dp.callback_query_handler(lambda c: True)
async def on_cb(call: types.CallbackQuery):
    uid = call.from_user.id
    data = call.data
    changed=False

    if data=="stake_+": state["stake_base"]=round(state["stake_base"]+1.0,2); changed=True
    elif data=="stake_-": state["stake_base"]=max(1.0, round(state["stake_base"]-1.0,2)); changed=True
    elif data=="gales_+": state["gales_max"]=min(3, state["gales_max"]+1); changed=True
    elif data=="gales_-": state["gales_max"]=max(0, state["gales_max"]-1); changed=True
    elif data=="ciclo_+": state["ciclo_max"]=min(10, state["ciclo_max"]+1); changed=True
    elif data=="ciclo_-": state["ciclo_max"]=max(1, state["ciclo_max"]-1); changed=True
    elif data=="cm_+":   state["ciclo_mult"]=round(min(10.0, state["ciclo_mult"]+0.5),2); changed=True
    elif data=="cm_-":   state["ciclo_mult"]=round(max(1.0,  state["ciclo_mult"]-0.5),2); changed=True
    elif data=="toggle": state["seguir"]=not state["seguir"]; changed=True

    elif data=="preset_1":
        state["multipliers"]=[1.0]*(state["gales_max"]+1); changed=True
    elif data=="preset_2":
        state["multipliers"]=[1.0]+[2.0]*state["gales_max"]; changed=True
    elif data=="preset_3":
        state["multipliers"]=[1.0]+[3.0]*state["gales_max"]; changed=True
    elif data=="preset_4":
        state["multipliers"]=[1.0]+[4.0]*state["gales_max"]; changed=True

    elif data=="set_banca":
        AWAIT_NUMERIC[uid]="banca"
        await call.message.reply("💬 Digite a nova banca (ex: 1000):")
        await call.answer(); return
    elif data=="set_sw":
        AWAIT_NUMERIC[uid]="sw"
        await call.message.reply("💬 Digite o novo Stop Win (ex: 300):")
        await call.answer(); return
    elif data=="set_sl":
        AWAIT_NUMERIC[uid]="sl"
        await call.message.reply("💬 Digite o novo Stop Loss (ex: 300):")
        await call.answer(); return

    if changed:
        save_state()
        try:
            await call.message.edit_text(painel_texto(), reply_markup=kb_painel())
        except:
            pass
        await call.answer("Atualizado!")
    else:
        if data=="refresh":
            try: await call.message.edit_text(painel_texto(), reply_markup=kb_painel())
            except: pass
            await call.answer("OK")
        else:
            await call.answer()

@dp.message_handler(content_types=types.ContentTypes.TEXT)
async def on_numeric_reply(m: types.Message):
    uid = m.from_user.id
    if uid not in AWAIT_NUMERIC:
        return
    kind = AWAIT_NUMERIC.pop(uid)
    try:
        v = float((m.text or "").replace(",", "."))
        if v <= 0: raise ValueError()
        if kind=="banca":
            risk["bankroll"]=round(v,2)
            await m.reply(f"✅ Banca ajustada para <b>R${risk['bankroll']:.2f}</b>")
        elif kind=="sw":
            risk["stop_win"]=round(v,2)
            await m.reply(f"✅ Stop Win ajustado: <b>R${risk['stop_win']:.2f}</b>")
        elif kind=="sl":
            risk["stop_loss"]=round(v,2)
            await m.reply(f"✅ Stop Loss ajustado: <b>R${risk['stop_loss']:.2f}</b>")
        save_risk()
        await m.reply(painel_texto(), reply_markup=kb_painel())
    except:
        await m.reply("❗ Valor inválido.")

# ========= CORE: abrir/fechar =========
def abrir_operacao(apos:int, alvos:List[int]):
    # base padrão
    base = state["stake_base"]
    mults = state["multipliers"][:state["gales_max"]+1] or [1.0]

    # modo recuperação (ciclo ativo) usa perda × ciclo_mult
    if risk["cycle_left"] > 0 and risk["prev_cycle_loss"] > 0:
        base = max(base, round(risk["prev_cycle_loss"] * state["ciclo_mult"], 2))
        risk["cycle_left"] -= 1
        save_risk()

    op = {
        "apos": apos,
        "alvos": alvos,
        "base": base,
        "mult": mults,
        "step": 0,
        "closed": False
    }
    risk["open"] = op
    save_risk()
    return op

async def publicar_plano(op):
    s0, per0, _ = plano_por_tentativa(op["base"], op["mult"][0])
    plano_txt = resumo_plano(op["mult"], op["base"])
    txt = (
        "🟢 <b>CONFIRMAR</b>\n"
        f"🎯 Alvos: <b>{op['alvos'][0]}-{op['alvos'][1]}-{op['alvos'][2]}</b> (após {op['apos']})\n"
        f"💵 Tentativa 1 (total): <b>R${s0:.2f}</b> (≈ <i>{per0:.2f} por número</i>)\n"
        f"🧮 Plano: {plano_txt}\n"
        f"📈 Odds por nº (fixo): <b>{ODDS_TOTAL:.2f}x</b>\n"
        f"💼 Banca: <b>R${risk['bankroll']:.2f}</b> | Sessão: <b>{risk['session_pnl']:.2f}</b>"
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

    # GREEN zera recuperação
    risk["prev_cycle_loss"] = 0.0
    risk["cycle_left"] = 0
    save_risk()

    if state["dm_user_id"]:
        await bot.send_message(
            state["dm_user_id"],
            f"✅ <b>GREEN</b> (step {op['step']+1}) | Lucro: <b>R${pnl:.2f}</b> | "
            f"Sessão: <b>{risk['session_pnl']:.2f}</b> | Banca: <b>{risk['bankroll']:.2f}</b>"
        )
    await checar_stops()

async def avancar_depois_de_red():
    op = risk.get("open")
    if not op or op["closed"]: return
    op["step"] += 1
    if op["step"] >= len(op["mult"]):
        # ciclo perdido
        preju = total_gasto_ate(op["mult"], op["base"], len(op["mult"]))
        risk["session_pnl"] = round(risk["session_pnl"] - preju, 2)
        risk["bankroll"]    = round(risk["bankroll"] - preju, 2)
        risk["prev_cycle_loss"] = preju
        risk["cycle_left"] = state["ciclo_max"]  # ativar recuperação

        op["closed"] = True
        risk["open"] = None
        save_risk()

        if state["dm_user_id"]:
            await bot.send_message(
                state["dm_user_id"],
                f"❌ <b>RED</b> | Perda: <b>R${preju:.2f}</b> | "
                f"Sessão: <b>{risk['session_pnl']:.2f}</b> | Banca: <b>{risk['bankroll']:.2f}</b>"
            )
        await checar_stops()
    else:
        save_risk()

async def checar_stops():
    if risk["session_pnl"] >= risk["stop_win"]:
        state["seguir"]=False; save_state()
        if state["dm_user_id"]:
            await bot.send_message(state["dm_user_id"], "🟢 <b>STOP WIN atingido</b>. Pausando entradas.")
    if risk["session_pnl"] <= -risk["stop_loss"]:
        state["seguir"]=False; save_state()
        if state["dm_user_id"]:
            await bot.send_message(state["dm_user_id"], "🔴 <b>STOP LOSS atingido</b>. Pausando entradas.")

# ========= PROCESSADOR ÚNICO (post/edição) =========
async def _process_channel_text(msg: types.Message):
    if not CHANNEL_ID or msg.chat.id != CHANNEL_ID:
        return
    txt = (msg.text or "").strip()
    if not txt:
        return

    # 0) Gatilho: mensagem indicando G1 (ex.: "Estamos no 1° gale")
    if risk.get("open") and not risk["open"]["closed"] and re_g1_hint.search(txt):
        op = risk["open"]
        # só avança se estiver em G0 e há multiplicador para G1
        if op["step"] == 0 and len(op["mult"]) > 1:
            op["step"] = 1
            save_risk()
            s1, per1, _ = plano_por_tentativa(op["base"], op["mult"][1])
            if state["dm_user_id"]:
                await bot.send_message(
                    state["dm_user_id"],
                    f"🟠 <b>G1 ativado</b>\n"
                    f"💵 Tentativa 2 (G1) total: <b>R${s1:.2f}</b> (≈ <i>{per1:.2f} por número</i>)"
                )
        # não processa como novo sinal
        # (mas deixa seguir para fechar se a mesma msg tiver 'APOSTA ENCERRADA', o que é improvável)
        # -> não retorna aqui de propósito

    # 1) Resultado fecha/avança
    r = eh_resultado(txt)
    if r is not None:
        if risk.get("open") and not risk["open"]["closed"]:
            if r == 1: await fechar_com_green()
            else:      await avancar_depois_de_red()
        return

    # 2) Nova entrada confirmada -> abre operação (se permitido)
    if not eh_sinal(txt):
        return
    if not state["seguir"]:
        return
    if time.time() < state.get("cooldown_until", 0.0):
        return
    if (risk["session_pnl"] >= risk["stop_win"]) or (risk["session_pnl"] <= -risk["stop_loss"]):
        await checar_stops()
        return

    regra = extrai_regra_sinal(txt)
    if not regra:
        log.info("Sinal sem padrão esperado: %s", txt[:120])
        return
    apos, alvos = regra
    if len(alvos) != 3:
        return

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
    if not PUBLIC_URL:
        log.warning("PUBLIC_URL não definido; defina no Render.")
        return
    await bot.delete_webhook(drop_pending_updates=True)
    await bot.set_webhook(f"{PUBLIC_URL}/webhook/{BOT_TOKEN}")
    log.info("Webhook configurado em %s/webhook/<token>", PUBLIC_URL)

@app.post(f"/webhook/{BOT_TOKEN}")
async def tg_webhook(request: Request):
    data = await request.body()
    update = types.Update(**json.loads(data.decode("utf-8")))
    # contexto aiogram v2
    Bot.set_current(bot)
    Dispatcher.set_current(dp)
    await dp.process_update(update)
    return {"ok": True}