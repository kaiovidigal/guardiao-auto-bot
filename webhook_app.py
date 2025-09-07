# -*- coding: utf-8 -*-
import os, re, json, time, logging
from collections import deque
from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# ================= LOG =================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("guardiao-auto")

# =============== CONFIG =================
BOT_TOKEN   = os.getenv("TG_BOT_TOKEN")
PUBLIC_URL  = (os.getenv("PUBLIC_URL") or "").rstrip("/")
CHANNEL_IN  = int(os.getenv("CHANNEL_IN_ID", "-1002810508717"))   # Canal A (sinais)

if not BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN")
if not PUBLIC_URL:
    log.warning("PUBLIC_URL não definido; defina após a primeira implantação.")

# aiogram
bot = Bot(token=BOT_TOKEN, parse_mode=types.ParseMode.HTML)
dp  = Dispatcher(bot)

# ============== PERSISTÊNCIA ============
os.makedirs("data", exist_ok=True)
STATE_FILE   = "data/state.json"
RISK_FILE    = "data/risk.json"
CONTROL_FILE = "data/control.json"   # guarda o chat de controle (privado)

# estado operacional (ajustável via painel)
state = {
    "seguir_sinal": True,
    "cooldown_until": 0.0,
    "stake_base": 5.00,   # valor total da tentativa 1 (soma dos 3 números)
    "gales_max": 1,       # 0 = sem gale; 1 = G1; 2 = G2; até 3
    "ciclo_max": 1,       # quantidade de ciclos (encadeamentos) permitidos
    "gale_mult": 2.0,     # multiplicador rápido (+/- nos botões)
    "multipliers": [1.0, 2.0],  # multiplicadores por tentativa (T1, G1, G2, G3)
    "odds_num": 3.85,     # payout por número (Fantan Evolution ~3.85)
}
risk = {
    "bankroll": 100.00,    # banca simulada (editável com /banca 250)
    "session_pnl": 0.00,   # soma da sessão
    "stop_win": 50.00,     # stop win (editável /stopwin 50)
    "stop_loss": 50.00,    # stop loss (editável /stoploss 50)
    "open": None,          # operação aberta (se houver)
}
control = {"chat_id": None}  # privado onde o bot fala com você

def load_json(path, target):
    try:
        if os.path.exists(path):
            target.update(json.load(open(path, "r", encoding="utf-8")))
    except Exception as e:
        log.warning("Falha ao carregar %s: %s", path, e)

def save_json(path, obj):
    try:
        json.dump(obj, open(path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Falha ao salvar %s: %s", path, e)

load_json(STATE_FILE, state)
load_json(RISK_FILE, risk)
load_json(CONTROL_FILE, control)

# ======== HISTÓRICO SIMPLES =========
hist_short = deque(maxlen=30)   # 1 win / 0 loss
def winrate(d):
    d = list(d)
    return (sum(d)/len(d)) if d else 0.0

# ============== REGEX PARSER DO CANAL ==============
re_sinal   = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
re_seq     = re.compile(r"Sequ[eê]ncia[:\s]*([^\n]+)", re.I)
re_apos    = re.compile(r"Entrar\s+ap[oó]s\s+o\s+([1-4])", re.I)
re_apostar = re.compile(r"apostar\s+em\s+([A-Za-z]*\s*)?([1-4](?:[\s\-\|]*[1-4])*)", re.I)
re_red     = re.compile(r"\b(APOSTA\s+ENCERRADA|RED|NEUTRO)\b", re.I)
re_green   = re.compile(r"\b(GREEN|WIN|✅)\b", re.I)

def eh_sinal(txt): return bool(re_sinal.search(txt or ""))

def extrai_regra(txt):
    """ retorna (apos_num, alvos[3]) ou (None, []) """
    m1 = re_apos.search(txt or "")
    m2 = re_apostar.search(txt or "")
    apos = int(m1.group(1)) if m1 else None
    alvos = [int(x) for x in re.findall(r"[1-4]", (m2.group(2) if m2 else ""))]
    # força 3 alvos (ex.: 4-3-2)
    alvos = alvos[:3] if len(alvos) >= 3 else []
    return apos, alvos

def resultado_txt(txt):
    up = (txt or "").upper()
    if re_green.search(up): return 1
    if re_red.search(up):   return 0
    return None

# ============== CÁLCULO FINANCEIRO ==============
def lucro_liquido_no_acerto(stake_total, odds_num):
    """
    Aposta é feita em 3 números (mesmo valor cada).
    Retorno líquido no acerto = (odds_num-1)*por_num - (perdas nos 2 números) = (odds_num-3)*por_num
    """
    por_num = stake_total / 3.0
    return round((odds_num - 3.0) * por_num, 2)

def plano_por_step(base_total, mult):
    stake_total = round(base_total * mult, 2)
    por_num = round(stake_total / 3.0, 2)
    lucro = lucro_liquido_no_acerto(stake_total, state["odds_num"])
    return stake_total, por_num, lucro

# ============== UI (PAINEL) ==============
def kb_painel():
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
    kb.row(InlineKeyboardButton("✖️ 1x", callback_data="preset_1"),
           InlineKeyboardButton("✖️ 2x", callback_data="preset_2"),
           InlineKeyboardButton("✖️ 3x", callback_data="preset_3"),
           InlineKeyboardButton("✖️ 4x", callback_data="preset_4"))
    kb.row(InlineKeyboardButton("🟢 Seguir: ON" if state["seguir_sinal"] else "⚪️ Seguir: OFF",
                                callback_data="toggle"))
    kb.row(InlineKeyboardButton("🔄 Atualizar", callback_data="refresh"))
    return kb

def txt_painel():
    mults = ", ".join(f"{x:.2f}" for x in state["multipliers"])
    plano_tot = sum(plano_por_step(state["stake_base"], m)[0] for m in state["multipliers"][:state["gales_max"]+1])
    return (
        "⚙️ <b>PAINEL</b>\n"
        f"💵 Base: <b>{state['stake_base']:.2f}</b> | ♻️ Gales: <b>{state['gales_max']}</b> | 🔁 Ciclo: <b>{state['ciclo_max']}</b>\n"
        f"✖️ Mults: <b>{mults}</b> (use /mult 1,2,3,4)\n"
        f"📈 Mult rápido: x{state['gale_mult']:.2f}\n"
        f"🎯 Odds por número: <b>{state['odds_num']:.2f}x</b>\n"
        f"💼 Banca: <b>R${risk['bankroll']:.2f}</b> | PnL Sessão: <b>{risk['session_pnl']:.2f}</b>\n"
        f"🧮 Plano (até G{state['gales_max']}): <b>{plano_tot:.2f}</b>\n"
        f"Seguir sinal: <b>{'ON' if state['seguir_sinal'] else 'OFF'}</b>"
    )

# ============== COMANDOS PRIVADOS ==============
@dp.message_handler(commands=["start"])
async def cmd_start(m: types.Message):
    control["chat_id"] = m.chat.id
    save_json(CONTROL_FILE, control)
    await m.answer(
        "<b>🤖 Guardião de Risco (SIMULAÇÃO)</b>\n"
        "• Lê apenas <b>ENTRADA CONFIRMADA</b> do canal de sinais.\n"
        "• Publica tudo aqui no chat privado (sem canal B).\n"
        "• Comandos úteis:\n"
        "   /painel | /banca 250 | /stopwin 50 | /stoploss 50 | /odds 3.85 | /mult 1,2,2.5\n"
        f"• Canal de SINAIS (A): <code>{CHANNEL_IN}</code>\n"
        "• Confiança: DESLIGADA (executa sempre que houver sinal).",
        parse_mode="HTML"
    )
    await m.answer(txt_painel(), reply_markup=kb_painel(), parse_mode="HTML")

@dp.message_handler(commands=["painel","config"])
async def cmd_painel(m: types.Message):
    await m.answer(txt_painel(), reply_markup=kb_painel(), parse_mode="HTML")

@dp.callback_query_handler(lambda c: True)
async def on_cb(call: types.CallbackQuery):
    d = call.data; changed=False
    if   d=="stake_+": state["stake_base"]=round(state["stake_base"]+1.0,2); changed=True
    elif d=="stake_-": state["stake_base"]=max(1.0, round(state["stake_base"]-1.0,2)); changed=True
    elif d=="gales_+": state["gales_max"]=min(3, state["gales_max"]+1); changed=True
    elif d=="gales_-": state["gales_max"]=max(0, state["gales_max"]-1); changed=True
    elif d=="ciclo_+": state["ciclo_max"]=min(10,state["ciclo_max"]+1); changed=True
    elif d=="ciclo_-": state["ciclo_max"]=max(1, state["ciclo_max"]-1); changed=True
    elif d=="gmx_+":   state["gale_mult"]=round(min(4.0, state["gale_mult"]+0.5),2); changed=True
    elif d=="gmx_-":   state["gale_mult"]=round(max(1.0, state["gale_mult"]-0.5),2); changed=True
    elif d=="toggle":  state["seguir_sinal"]=not state["seguir_sinal"]; changed=True
    elif d.startswith("preset_"):
        k=int(d.split("_")[1])
        lst=[1.0]; 
        for _ in range(3): lst.append(float(k))
        state["multipliers"]=lst[:state["gales_max"]+1]; changed=True

    if changed:
        save_json(STATE_FILE, state)
        try: await call.message.edit_reply_markup(kb_painel())
        except: pass
        await call.answer("Atualizado!")
    else:
        if d=="refresh":
            try: await call.message.edit_text(txt_painel(), reply_markup=kb_painel(), parse_mode="HTML")
            except: pass
            await call.answer("OK")
        else:
            await call.answer()

# ======== Edição rápida por comandos ========
@dp.message_handler(commands=["banca"])
async def cmd_banca(m: types.Message):
    try:
        v=float((m.get_args() or "").replace(",", "."))
        if v<=0: raise ValueError()
        risk["bankroll"]=round(v,2); save_json(RISK_FILE,risk)
        await m.answer(f"✅ Banca atual: <b>R${risk['bankroll']:.2f}</b>", parse_mode="HTML")
    except:
        await m.answer("Use: /banca 250.00")

@dp.message_handler(commands=["stopwin"])
async def cmd_stopwin(m: types.Message):
    try:
        v=float((m.get_args() or "").replace(",", "."))
        if v<=0: raise ValueError()
        risk["stop_win"]=round(v,2); save_json(RISK_FILE,risk)
        await m.answer(f"✅ Stop Win: <b>R${risk['stop_win']:.2f}</b>", parse_mode="HTML")
    except:
        await m.answer("Use: /stopwin 50")

@dp.message_handler(commands=["stoploss"])
async def cmd_stoploss(m: types.Message):
    try:
        v=float((m.get_args() or "").replace(",", "."))
        if v<=0: raise ValueError()
        risk["stop_loss"]=round(v,2); save_json(RISK_FILE,risk)
        await m.answer(f"✅ Stop Loss: <b>R${risk['stop_loss']:.2f}</b>", parse_mode="HTML")
    except:
        await m.answer("Use: /stoploss 50")

@dp.message_handler(commands=["odds"])
async def cmd_odds(m: types.Message):
    try:
        v=float((m.get_args() or "").replace(",", "."))
        if v<3.0: raise ValueError()
        state["odds_num"]=round(v,2); save_json(STATE_FILE,state)
        await m.answer(f"✅ Odds por número: <b>{state['odds_num']:.2f}x</b>", parse_mode="HTML")
    except:
        await m.answer("Use: /odds 3.85")

@dp.message_handler(commands=["mult"])
async def cmd_mult(m: types.Message):
    try:
        raw=(m.get_args() or "").replace(" ", "")
        lst=[float(x.replace(",", ".")) for x in raw.split(",") if x]
        if not lst: raise ValueError()
        if len(lst)>4: lst=lst[:4]
        state["multipliers"]=lst[:state["gales_max"]+1]; save_json(STATE_FILE,state)
        await m.answer(
            f"✅ Multiplicadores: <b>{', '.join(f'{x:.2f}' for x in state['multipliers'])}</b>",
            parse_mode="HTML"
        )
    except:
        await m.answer("Use: /mult 1,2,2.5,3 (até 4 valores)")

@dp.message_handler(commands=["saldo"])
async def cmd_saldo(m: types.Message):
    await m.answer(
        f"💼 Banca: <b>R${risk['bankroll']:.2f}</b> | Sessão: <b>{risk['session_pnl']:.2f}</b>\n"
        f"🛑 StopWin: <b>{risk['stop_win']:.2f}</b> | StopLoss: <b>{risk['stop_loss']:.2f}</b>",
        parse_mode="HTML"
    )

# ============== CONTROLE DE CICLO / RESULTADOS ==============
def check_stops_and_pause():
    if risk["session_pnl"] >= risk["stop_win"]:
        state["seguir_sinal"]=False; save_json(STATE_FILE,state)
        bot.loop.create_task(bot.send_message(control["chat_id"], "✅ <b>STOP WIN atingido</b>. Pausando.", parse_mode="HTML"))
        return True
    if risk["session_pnl"] <= -risk["stop_loss"]:
        state["seguir_sinal"]=False; save_json(STATE_FILE,state)
        bot.loop.create_task(bot.send_message(control["chat_id"], "⛔ <b>STOP LOSS atingido</b>. Pausando.", parse_mode="HTML"))
        return True
    return False

def abrir_operacao(apos_num, alvos, base, mults):
    risk["open"] = {
        "apos": apos_num, "alvos": alvos, "base": round(base,2),
        "mults": mults[:], "step": 0, "closed": False
    }
    save_json(RISK_FILE, risk)
    return risk["open"]

def valor_step(op):
    s, pnum, lucro = plano_por_step(op["base"], op["mults"][op["step"]])
    return s, pnum, lucro

def fechar_win():
    op = risk.get("open"); ifnot = (not op) or op["closed"]
    if ifnot: return
    stake_total, _, lucro_liq = valor_step(op)
    perdas_prev = sum(plano_por_step(op["base"], op["mults"][i])[0] for i in range(op["step"]))
    pnl = round(lucro_liq - perdas_prev, 2)
    risk["session_pnl"] = round(risk["session_pnl"] + pnl, 2)
    risk["bankroll"]    = round(risk["bankroll"] + pnl, 2)
    op["closed"]=True; risk["open"]=None; save_json(RISK_FILE,risk)
    bot.loop.create_task(bot.send_message(
        control["chat_id"],
        f"✅ <b>GREEN</b> (step {op['step']}) | Lucro: <b>R${pnl:.2f}</b> | "
        f"Sessão: <b>{risk['session_pnl']:.2f}</b> | Banca: <b>{risk['bankroll']:.2f}</b>",
        parse_mode="HTML"
    ))
    check_stops_and_pause()

def fechar_loss_ou_avancar():
    op = risk.get("open")
    if not op or op["closed"]: return
    # registra perda do step atual
    perda_step = plano_por_step(op["base"], op["mults"][op["step"]])[0]
    risk["session_pnl"] = round(risk["session_pnl"] - perda_step, 2)
    risk["bankroll"]    = round(risk["bankroll"] - perda_step, 2)
    op["step"] += 1
    if op["step"] >= len(op["mults"]):
        op["closed"]=True; risk["open"]=None
        save_json(RISK_FILE,risk)
        bot.loop.create_task(bot.send_message(
            control["chat_id"],
            f"❌ <b>LOSS</b> (ciclo encerrado) | Sessão: <b>{risk['session_pnl']:.2f}</b> "
            f"| Banca: <b>{risk['bankroll']:.2f}</b>",
            parse_mode="HTML"
        ))
        check_stops_and_pause()
    else:
        save_json(RISK_FILE,risk)
        # informa que vai para próximo step (mas a entrada efetiva é textual/simulação)
        nxt = plano_por_step(op["base"], op["mults"][op["step"]])[0]
        bot.loop.create_task(bot.send_message(
            control["chat_id"],
            f"↪️ <b>LOSS no step anterior</b>. Próxima tentativa (step {op['step']}): "
            f"<b>R${nxt:.2f}</b> total (3 números).", parse_mode="HTML"
        ))

# ============== LEITURA DO CANAL (A) ==============
def publica_confirmacao(apos, alvos, base, mults):
    s0, por_num, _ = plano_por_step(base, mults[0])
    plano = " → ".join(f"{plano_por_step(base, m)[0]:.2f}" for m in mults)
    txt = (
        "🟢 <b>CONFIRMAR</b>\n"
        f"🎯 Alvos: <b>{alvos[0]}-{alvos[1]}-{alvos[2]}</b>\n"
        f"🧭 Após: <b>{apos}</b>\n"
        f"💵 Tentativa 1 (total): <b>R${s0:.2f}</b> (≈ <i>{por_num:.2f} por número</i>)\n"
        f"🧮 Plano: {plano}\n"
        f"📈 Odds por número: <b>{state['odds_num']:.2f}x</b>\n"
        f"💼 Sessão: <b>{risk['session_pnl']:.2f}</b> | Banca: <b>{risk['bankroll']:.2f}</b>"
    )
    if control["chat_id"]:
        bot.loop.create_task(bot.send_message(control["chat_id"], txt, parse_mode="HTML"))

@dp.channel_post_handler(content_types=["text"])
async def on_channel_post(msg: types.Message):
    if msg.chat.id != CHANNEL_IN:
        return
    txt = (msg.text or "").strip()
    if not txt:
        return

    # resultado (GREEN/LOSS)
    res = resultado_txt(txt)
    if res is not None:
        hist_short.append(res)
        if risk.get("open") and not risk["open"]["closed"]:
            if res == 1: fechar_win()
            else: fechar_loss_ou_avancar()
        return

    # apenas gatilho de entrada
    if not eh_sinal(txt): return
    if time.time() < state.get("cooldown_until", 0.0): return
    if not state["seguir_sinal"]: return
    if check_stops_and_pause(): return

    apos, alvos = extrai_regra(txt)
    if len(alvos) != 3 or not apos:
        log.info("Sinal sem regra clara: %s", txt)
        return

    mults = state["multipliers"][:state["gales_max"]+1] or [1.0]
    base  = state["stake_base"]

    abrir_operacao(apos, alvos, base, mults)
    publica_confirmacao(apos, alvos, base, mults)

    state["cooldown_until"] = time.time() + 10
    save_json(STATE_FILE, state)

# ============== FASTAPI / WEBHOOK =========
app = FastAPI()

@app.get("/healthz")
def healthz(): 
    return {"ok": True}

@app.on_event("startup")
async def on_startup():
    await bot.delete_webhook(drop_pending_updates=True)
    if PUBLIC_URL:
        await bot.set_webhook(f"{PUBLIC_URL}/webhook/{BOT_TOKEN}")
        log.info("Webhook configurado.")
    else:
        log.warning("PUBLIC_URL ausente (defina nas envs).")

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.body()
    update = types.Update(**json.loads(data.decode("utf-8")))
    # aiogram v2 precisa do contexto
    Bot.set_current(bot)
    Dispatcher.set_current(dp)
    await dp.process_update(update)
    return {"ok": True}