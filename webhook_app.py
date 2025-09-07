# webhook_app.py
import os, re, json, logging, time
from collections import deque
from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# ============== LOGGING ==============
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("guardiao-auto-dm")

# ============== ENV / CONFIG ==============
BOT_TOKEN  = os.getenv("TG_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Faltando TG_BOT_TOKEN")

CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))       # Canal de SINAIS A
PUBLIC_URL = os.getenv("PUBLIC_URL", "").rstrip("/") # URL do Render

# Payout fixo por número (Fan Tan) -> EMBUTIDO
ODDS_TOTAL = 3.85

# aiogram
bot = Bot(token=BOT_TOKEN, parse_mode=types.ParseMode.HTML)
dp  = Dispatcher(bot)

# ============== PERSISTÊNCIA ==============
os.makedirs("data", exist_ok=True)
STATE_PATH = "data/state.json"
RISK_PATH  = "data/risk.json"

state = {
    "dm_user_id": 0,           # preenchido no /start
    "stake_base": 5.00,        # soma dos 3 números na tentativa 1 (G0)
    "gales_max": 1,            # 0..3  (G1 padrão)
    "ciclo_max": 1,            # qtde de ciclos a acompanhar
    "multipliers": [1.0, 3.0], # G1 padrão com 3x (tentativas: [1x, 3x])
    "cycle_mult": 3.0          # multiplicador do CICLO: prev_loss * 3 (padrão)
}
risk = {
    "bankroll": 100.00,
    "session_pnl": 0.0,
    "stop_win": 50.00,
    "stop_loss": 50.00,
    "prev_cycle_loss": 0.0,    # prejuízo acumulado à recuperar no próximo ciclo
    "open": None               # operação em andamento
}

def _load(path, target):
    try:
        if os.path.exists(path):
            target.update(json.load(open(path, "r", encoding="utf-8")))
    except Exception as e:
        log.warning("Falha ao carregar %s: %s", path, e)
    return target

def _save(path, obj):
    try:
        json.dump(obj, open(path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Falha ao salvar %s: %s", path, e)

state = _load(STATE_PATH, state)
risk  = _load(RISK_PATH,  risk)
def save_state(): _save(STATE_PATH, state)
def save_risk():  _save(RISK_PATH,  risk)

# ============== PARSERS ==============
re_sinal   = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
re_seq     = re.compile(r"Sequ[eê]ncia[:\s]*([^\n]+)", re.I)
re_apos    = re.compile(r"Entrar\s+ap[oó]s\s+o\s+([1-4])", re.I)
re_apostar = re.compile(r"apostar\s+em\s+([A-Za-z]*\s*)?([1-4](?:[\s\-\|]*[1-4])*)", re.I)

def eh_sinal(txt:str) -> bool:
    return bool(re_sinal.search(txt or ""))

def extrai_regra_sinal(txt:str):
    m1=re_apos.search(txt or ""); m2=re_apostar.search(txt or "")
    apos = int(m1.group(1)) if m1 else None
    alvos = [int(x) for x in re.findall(r"[1-4]", (m2.group(2) if m2 else ""))]
    return apos, alvos

def eh_resultado(txt: str):
    """1 = GREEN, 0 = RED, None = ainda não é resultado.
       'APOSTA ENCERRADA' e 'NEUTRO' são ignorados (no seu canal vêm antes)."""
    up = (txt or "").upper()
    if "RED" in up or "❌" in up:   return 0
    if "GREEN" in up or "✅" in up: return 1
    if "NEUTRO" in up or "APOSTA ENCERRADA" in up: return None
    return None

# ============== STAKE & PNL ==============
def lucro_liquido_no_acerto(por_num: float) -> float:
    # acerta 1 número => ganha 3.85 * por_num e perde 2 * por_num
    return round((ODDS_TOTAL - 3.0) * por_num, 2)  # 0.85 * por_num

def plano_por_tentativa(base_total: float, mult: float):
    stake_total = round(base_total * mult, 2)      # soma nos 3 números
    por_num     = round(stake_total / 3.0, 2)
    lucro       = lucro_liquido_no_acerto(por_num) # líquido se acertar na tentativa
    return stake_total, por_num, lucro

def resumo_plano_text(multipliers, base_total):
    partes=[]; total=0.0
    for m in multipliers:
        s,_,_ = plano_por_tentativa(base_total, m)
        partes.append(f"{s:.2f}")
        total += s
    return f"{' → '.join(partes)} = <b>{total:.2f}</b>"

# ============== UI / PAINEL ==============
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
        InlineKeyboardButton(f"Ciclo Mult x{state['cycle_mult']:.2f}", callback_data="set_cycle_mult"),
        InlineKeyboardButton("🔄 Atualizar", callback_data="refresh"),
    )
    return kb

def painel_texto():
    mults = ", ".join(f"{x:.2f}" for x in state["multipliers"])
    plano = resumo_plano_text(state["multipliers"][:state["gales_max"]+1], state["stake_base"])
    return (
        "⚙️ <b>PAINEL</b>\n"
        f"💰 Base: <b>{state['stake_base']:.2f}</b> | ♻️ Gales: <b>{state['gales_max']}</b> | 🔁 Ciclo: <b>{state['ciclo_max']}</b>\n"
        f"✖️ Mults (G0..Gn): <b>{mults}</b>\n"
        f"📈 Ciclo Mult: <b>x{state['cycle_mult']:.2f}</b>\n"
        f"🎯 Odds por nº (fixo): <b>{ODDS_TOTAL:.2f}x</b>\n"
        f"💼 Banca: <b>R${risk['bankroll']:.2f}</b> | PnL Sessão: <b>{risk['session_pnl']:.2f}</b>\n"
        f"🧮 Plano (até G{state['gales_max']}): {plano}"
    )

@dp.message_handler(commands=["start"])
async def cmd_start(m: types.Message):
    state["dm_user_id"] = m.chat.id
    save_state()
    await m.answer(
        "<b>🤖 Guardião Auto (DM)</b>\n"
        "• Lê <b>ENTRADA CONFIRMADA</b> no canal A e executa a simulação aqui no privado.\n"
        "• Payout por número é fixo em <b>3.85x</b>.\n"
        "• G1 e Ciclo vêm <b>3x</b> por padrão (pode mudar com /mult e /cyclemult).\n\n"
        "<b>Comandos:</b>\n"
        "• /painel — ver/ajustar tudo\n"
        "• /mult 1,3 — muda multiplicadores de tentativas (ex.: G1 = [1,3])\n"
        "• /cyclemult 3 — muda multiplicador do ciclo (prejuízo * fator)\n",
        parse_mode="HTML"
    )
    await m.answer(painel_texto(), reply_markup=kb_painel(), parse_mode="HTML")

@dp.message_handler(commands=["painel","config","status","saldo"])
async def cmd_painel(m: types.Message):
    await m.answer(painel_texto(), reply_markup=kb_painel(), parse_mode="HTML")

# ====== fluxo de entrada de números (banca/stops/cycle_mult) ======
AWAIT_NUMERIC = {}  # {user_id: "banca"|"sw"|"sl"|"cycle_mult"}

@dp.callback_query_handler(lambda c: True)
async def on_cb(call: types.CallbackQuery):
    uid   = call.from_user.id
    data  = call.data
    changed = False

    if data=="stake_+":
        state["stake_base"]=round(state["stake_base"]+1.0,2); changed=True
    elif data=="stake_-":
        state["stake_base"]=max(1.0, round(state["stake_base"]-1.0,2)); changed=True
    elif data=="gales_+":
        state["gales_max"]=min(3, state["gales_max"]+1); changed=True
    elif data=="gales_-":
        state["gales_max"]=max(0, state["gales_max"]-1); changed=True
    elif data=="ciclo_+":
        state["ciclo_max"]=min(10, state["ciclo_max"]+1); changed=True
    elif data=="ciclo_-":
        state["ciclo_max"]=max(1, state["ciclo_max"]-1); changed=True

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
        await call.message.reply("💬 Digite a <b>banca</b> (ex: 1000):", parse_mode="HTML")
        await call.answer(); return
    elif data=="set_sw":
        AWAIT_NUMERIC[uid]="sw"
        await call.message.reply("💬 Digite o <b>Stop Win</b> (ex: 200):", parse_mode="HTML")
        await call.answer(); return
    elif data=="set_sl":
        AWAIT_NUMERIC[uid]="sl"
        await call.message.reply("💬 Digite o <b>Stop Loss</b> (ex: 150):", parse_mode="HTML")
        await call.answer(); return
    elif data=="set_cycle_mult":
        AWAIT_NUMERIC[uid]="cycle_mult"
        await call.message.reply("💬 Digite o <b>Multiplicador do Ciclo</b> (ex: 3):", parse_mode="HTML")
        await call.answer(); return

    if changed:
        save_state()
        try:
            await call.message.edit_text(painel_texto(), reply_markup=kb_painel(), parse_mode="HTML")
        except: pass
        await call.answer("Atualizado!")
    else:
        if data=="refresh":
            try:
                await call.message.edit_text(painel_texto(), reply_markup=kb_painel(), parse_mode="HTML")
            except: pass
            await call.answer("OK")
        else:
            await call.answer()

@dp.message_handler(commands=["mult"])
async def cmd_mult(m: types.Message):
    try:
        raw=(m.get_args() or "").replace(" ", "")
        arr=[float(x.replace(",", ".")) for x in raw.split(",") if x]
        if not arr: raise ValueError()
        if len(arr)>4: arr=arr[:4]
        state["multipliers"]=arr
        save_state()
        await m.answer(f"✅ Multiplicadores: <b>{', '.join(f'{x:.2f}' for x in arr)}</b>", parse_mode="HTML")
        await m.answer(painel_texto(), reply_markup=kb_painel(), parse_mode="HTML")
    except:
        await m.answer("Use: /mult 1,3  ou /mult 1,3,4 (até 4 valores)")

@dp.message_handler(commands=["cyclemult"])
async def cmd_cyclemult(m: types.Message):
    try:
        v=float((m.get_args() or "").replace(",", "."))
        if v<=0: raise ValueError()
        state["cycle_mult"]=round(v,2); save_state()
        await m.answer(f"✅ Ciclo Mult definido para <b>x{state['cycle_mult']:.2f}</b>", parse_mode="HTML")
        await m.answer(painel_texto(), reply_markup=kb_painel(), parse_mode="HTML")
    except:
        await m.answer("Use: /cyclemult 3   (fator aplicado sobre o prejuízo do ciclo)")

@dp.message_handler(content_types=types.ContentTypes.TEXT)
async def on_text_dm(m: types.Message):
    uid = m.from_user.id
    if uid not in AWAIT_NUMERIC:
        return
    kind = AWAIT_NUMERIC.pop(uid)
    txt = (m.text or "").strip().replace(",", ".")
    try:
        v=float(txt)
        if v<=0: raise ValueError()
        if kind=="banca":
            risk["bankroll"]=round(v,2)
            await m.reply(f"✅ Banca ajustada para <b>R${risk['bankroll']:.2f}</b>", parse_mode="HTML")
        elif kind=="sw":
            risk["stop_win"]=round(v,2)
            await m.reply(f"✅ Stop Win ajustado para <b>R${risk['stop_win']:.2f}</b>", parse_mode="HTML")
        elif kind=="sl":
            risk["stop_loss"]=round(v,2)
            await m.reply(f"✅ Stop Loss ajustado para <b>R${risk['stop_loss']:.2f}</b>", parse_mode="HTML")
        elif kind=="cycle_mult":
            state["cycle_mult"]=round(v,2); save_state()
            await m.reply(f"✅ Ciclo Mult agora <b>x{state['cycle_mult']:.2f}</b>", parse_mode="HTML")
        save_risk()
        await m.reply(painel_texto(), reply_markup=kb_painel(), parse_mode="HTML")
    except:
        await m.reply("❗ Valor inválido. Tente novamente.")

# ============== EXECUÇÃO (abrir/fechar) ==============
def abrir_operacao(apos_num, alvos, base_total, multipliers):
    op = {
        "apos": apos_num,
        "alvos": alvos,
        "base": round(base_total,2),
        "mult": multipliers[:],
        "step": 0,
        "closed": False,
        "cycle_left": state["ciclo_max"],
    }
    risk["open"]=op; save_risk()
    return op

def valor_tentativa(op):
    m = op["mult"][op["step"]]
    stake_total, por_num, lucro = plano_por_tentativa(op["base"], m)
    return stake_total, por_num, lucro

def fechar_com_green(dm_chat_id):
    op=risk.get("open")
    if not op or op["closed"]: return
    stake_total, por_num, lucro = valor_tentativa(op)

    perdas_prev = 0.0
    for i in range(op["step"]):
        s,_,_ = plano_por_tentativa(op["base"], op["mult"][i])
        perdas_prev += s

    pnl = round(lucro - perdas_prev, 2)
    risk["session_pnl"] = round(risk["session_pnl"] + pnl, 2)
    risk["bankroll"]    = round(risk["bankroll"] + pnl, 2)
    risk["prev_cycle_loss"] = 0.0

    op["closed"]=True
    risk["open"]=None
    save_risk()

    txt = (
        "✅ <b>GREEN</b>\n"
        f"• PnL desta mão: <b>R${pnl:.2f}</b>\n"
        f"• Banca: <b>R${risk['bankroll']:.2f}</b> | Sessão: <b>{risk['session_pnl']:.2f}</b>"
    )
    bot.loop.create_task(bot.send_message(dm_chat_id, txt, parse_mode="HTML"))

def avancar_depois_de_red(dm_chat_id):
    op=risk.get("open")
    if not op or op["closed"]: return
    op["step"] += 1

    if op["step"] >= len(op["mult"]):
        # ciclo perdido => soma tudo que foi apostado no ciclo
        preju = 0.0
        for mi in op["mult"]:
            s,_,_ = plano_por_tentativa(op["base"], mi)
            preju += s
        risk["session_pnl"] = round(risk["session_pnl"] - preju, 2)
        risk["bankroll"]    = round(risk["bankroll"] - preju, 2)

        # guarda prejuízo para recuperação no próximo sinal (base = prev_loss * cycle_mult)
        risk["prev_cycle_loss"] = round(preju, 2)

        op["cycle_left"] -= 1
        op["closed"] = True
        risk["open"] = None
        save_risk()

        txt = (
            "❌ <b>RED</b>\n"
            f"• Perda do ciclo: <b>R${preju:.2f}</b>\n"
            f"• Banca: <b>R${risk['bankroll']:.2f}</b> | Sessão: <b>{risk['session_pnl']:.2f}</b>"
        )
        bot.loop.create_task(bot.send_message(dm_chat_id, txt, parse_mode="HTML"))
    else:
        save_risk()

# ============== HANDLER DO CANAL (A) ==============
@dp.channel_post_handler(content_types=["text"])
async def on_channel_post(msg: types.Message):
    if msg.chat.id != CHANNEL_ID:
        return
    dm = state.get("dm_user_id", 0)
    if not dm:
        return

    txt = (msg.text or "").strip()
    if not txt:
        return

    # 1) Fechamento GREEN/RED
    r = eh_resultado(txt)
    if r is not None:
        if risk.get("open") and not risk["open"]["closed"]:
            if r==1: fechar_com_green(dm)
            elif r==0: avancar_depois_de_red(dm)
        return

    # 2) ENTRADA CONFIRMADA → executar SEM filtro de confiança
    if not eh_sinal(txt):
        return

    apos_num, alvos = extrai_regra_sinal(txt)
    if len(alvos) != 3:
        log.info("Sinal sem 3 alvos claros: %s", txt)
        return

    # base padrão
    base = state["stake_base"]

    # se houver prejuízo anterior, aplica multiplicador do CICLO (prev_loss * cycle_mult)
    if risk["prev_cycle_loss"] > 0:
        base = max(base, round(risk["prev_cycle_loss"] * state["cycle_mult"], 2))

    # multiplicadores de tentativas (G1 padrão = [1,3])
    mults = state["multipliers"][:state["gales_max"]+1] or [1.0]

    op = abrir_operacao(apos_num, alvos, base, mults)
    s0, per0, _ = plano_por_tentativa(op["base"], op["mult"][0])
    plano_txt = resumo_plano_text(mults, base)

    out = (
        "🟢 <b>CONFIRMAR</b>\n"
        f"🎯 Alvos: <b>{alvos[0]}-{alvos[1]}-{alvos[2]}</b> (após {apos_num})\n"
        f"💵 Tentativa 1 (total): <b>R${s0:.2f}</b> (≈ <i>{per0:.2f} por número</i>)\n"
        f"🧮 Plano: {plano_txt}\n"
        f"📈 Odds por nº (fixo): <b>{ODDS_TOTAL:.2f}x</b>\n"
        f"💼 Banca: <b>R${risk['bankroll']:.2f}</b> | Sessão: <b>{risk['session_pnl']:.2f}</b>"
    )
    await bot.send_message(dm, out, parse_mode="HTML")

# ============== FASTAPI / WEBHOOK ==============
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
    # fix de contexto (aiogram v2)
    Bot.set_current(bot)
    Dispatcher.set_current(dp)
    await dp.process_update(update)
    return {"ok": True}