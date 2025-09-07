# webhook_app.py
import os, re, json, time, logging
from collections import deque
from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# ================= LOG =================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("guardiao-webhook")

# =============== CONFIG =================
BOT_TOKEN   = os.getenv("TG_BOT_TOKEN")
PUBLIC_URL  = (os.getenv("PUBLIC_URL") or "").rstrip("/")
CHANNEL_IN  = int(os.getenv("CHANNEL_IN_ID", "0"))    # Canal de SINAIS (A)
CHANNEL_OUT = int(os.getenv("CHANNEL_OUT_ID", "0"))   # Canal de SAÍDA  (B)
COOLDOWN_S  = int(os.getenv("COOLDOWN_S", "8"))

if not BOT_TOKEN:
    raise RuntimeError("Faltando TG_BOT_TOKEN")
if not CHANNEL_IN or not CHANNEL_OUT:
    raise RuntimeError("Faltando CHANNEL_IN_ID ou CHANNEL_OUT_ID")

bot = Bot(token=BOT_TOKEN, parse_mode=types.ParseMode.HTML)
dp  = Dispatcher(bot)

# ============== PERSISTÊNCIA ============
os.makedirs("data", exist_ok=True)
STATE_FILE = "data/state.json"

state = {
    "seguir_sinal": True,
    "cooldown_until": 0.0,
    "stake_base": 5.00,     # valor base TOTAL da Tentativa 1
    "gales_max": 1,         # quantos gales (0..3)
    "ciclo_max": 1,         # quantos ciclos (apenas decorativo aqui)
    "gale_mult": 2.0,       # multiplicador rápido
    "multipliers": [1.0, 2.0],  # T1, T2, T3, T4 (até 4)
    "odds_num": 3.85,       # odds por número (usado no resumo)
}
def load_state():
    try:
        if os.path.exists(STATE_FILE):
            state.update(json.load(open(STATE_FILE, "r", encoding="utf-8")))
    except Exception as e:
        log.warning("Falha ao carregar state: %s", e)
def save_state():
    try:
        json.dump(state, open(STATE_FILE, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Falha ao salvar state: %s", e)
load_state()

# ============== PARSERS ==============
re_sinal   = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
re_seq     = re.compile(r"Sequ[eê]ncia[:\s]*([^\n]+)", re.I)
re_apos    = re.compile(r"Entrar\s+ap[oó]s\s+o\s+([1-4])", re.I)
re_apostar = re.compile(r"apostar\s+em\s+(?:Ssh\s*)?([1-4](?:[\s\-\|]*[1-4])*)", re.I)

def extrai_regra_sinal(txt: str):
    """
    Tenta extrair:
      - número de referência 'Entrar após o X'
      - a lista de alvos 'apostar em Ssh 4-3-2' (3 números)
    Retorna (apos_num, [a,b,c]) ou (None, []) se não achar.
    """
    apos = None
    alvos = []
    m1 = re_apos.search(txt or "")
    if m1:
        try: apos = int(m1.group(1))
        except: pass
    m2 = re_apostar.search(txt or "")
    if m2:
        alvos = [int(x) for x in re.findall(r"[1-4]", m2.group(1))][:3]
    return apos, alvos

# ============== CÁLCULO DO PLANO =========
def plano_por_tentativa(base_total: float, mult: float):
    """
    Considera 3 números. Divide o total igualmente pelos 3 (por_num).
    """
    stake_total = round(base_total * mult, 2)
    por_num = round(stake_total / 3.0, 2)
    # lucro líquido aproximado: (odds - 3)*por_num (ganha 1 num, perde 2)
    lucro = round((state["odds_num"] - 3.0) * por_num, 2)
    return stake_total, por_num, lucro

def resumo_plano_text(multipliers, base_total):
    partes=[]; tot=0.0
    for m in multipliers:
        s, _, _ = plano_por_tentativa(base_total, m)
        partes.append(f"{s:.2f}")
        tot += s
    return f"{' → '.join(partes)} = <b>{tot:.2f}</b>"

# ============== UI / PAINEL ==============
def kb_painel():
    seguir = "🟢 Seguir: ON" if state["seguir_sinal"] else "⚪️ Seguir: OFF"
    presets_row = [
        InlineKeyboardButton("✖️ 1x", callback_data="preset_mult_1"),
        InlineKeyboardButton("✖️ 2x", callback_data="preset_mult_2"),
        InlineKeyboardButton("✖️ 3x", callback_data="preset_mult_3"),
        InlineKeyboardButton("✖️ 4x", callback_data="preset_mult_4"),
    ]
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
    kb.row(*presets_row)
    kb.row(InlineKeyboardButton(seguir, callback_data="toggle_seg"))
    kb.row(InlineKeyboardButton("🔄 Atualizar", callback_data="refresh"))
    return kb

@dp.message_handler(commands=["start"])
async def cmd_start(m: types.Message):
    await m.answer(
        "<b>🤖 Guardião de Risco (AUTOMÁTICO)</b>\n"
        "• Lê apenas <b>ENTRADA CONFIRMADA</b> no canal A\n"
        "• Publica o plano no canal B (SEM filtro de confiança)\n"
        "• /painel para configurar stake/gales/ciclo/multiplicadores/odds\n"
        f"• Canal de SINAIS (A): <code>{CHANNEL_IN}</code>\n"
        f"• Canal de SAÍDA (B): <code>{CHANNEL_OUT}</code>\n"
        f"• Confiança desligada: <b>SIM</b>\n",
        parse_mode="HTML"
    )

@dp.message_handler(commands=["painel","config"])
async def cmd_painel(m: types.Message):
    mults = ", ".join(f"{x:.2f}" for x in state["multipliers"])
    await m.answer(
        "⚙️ <b>PAINEL</b>\n"
        f"💰 Base: <b>{state['stake_base']:.2f}</b> | ♻️ Gales: <b>{state['gales_max']}</b> | 🔁 Ciclo: <b>{state['ciclo_max']}</b>\n"
        f"✖️ Mults: <b>{mults}</b> (use /mult 1,2,3,4)\n"
        f"📈 Mult rápido: x{state['gale_mult']:.2f}\n"
        f"🎯 Odds por número: <b>{state['odds_num']:.2f}x</b>\n"
        f"🧮 Plano: {resumo_plano_text(state['multipliers'][:state['gales_max']+1], state['stake_base'])}\n"
        f"Seguir sinal: <b>{'ON' if state['seguir_sinal'] else 'OFF'}</b>",
        reply_markup=kb_painel(), parse_mode="HTML"
    )

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

# ======== COMANDOS AUXILIARES =========
@dp.message_handler(commands=["odds"])
async def cmd_odds(m: types.Message):
    try:
        v=float((m.get_args() or "").replace(",", "."))
        if v<3.0: raise ValueError()
        state["odds_num"]=round(v,2); save_state()
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
        state["multipliers"]=lst[:state["gales_max"]+1]
        save_state()
        await m.answer(
            f"✅ Multiplicadores: <b>{', '.join(f'{x:.2f}' for x in state['multipliers'])}</b>",
            parse_mode="HTML"
        )
    except:
        await m.answer("Use: /mult 1,2,2.5,3 (até 4 valores)")

@dp.message_handler(commands=["seguir"])
async def cmd_seguir(m: types.Message):
    arg=(m.get_args() or "").strip().lower()
    if arg in ("on","off"):
        state["seguir_sinal"]=(arg=="on"); save_state()
    await m.answer(f"Seguir sinal: <b>{'ON' if state['seguir_sinal'] else 'OFF'}</b>", parse_mode="HTML")

# ============== PUBLICAÇÃO (SEM CONFIANÇA) =======
def build_plan_text(apos_num, alvos):
    mults = state["multipliers"][:state["gales_max"]+1] or [1.0]
    base_total = state["stake_base"]
    s0, per0, _ = plano_por_tentativa(base_total, mults[0])
    plano_txt = resumo_plano_text(mults, base_total)
    alvos_txt = "-".join(str(x) for x in (alvos or []))
    return (
        "🟢 <b>CONFIRMAR</b>\n"
        f"🎯 Alvos: <b>{alvos_txt}</b>\n"
        f"⏩ Após: <b>{apos_num}</b>\n"
        f"💵 Tentativa 1 (total): <b>R${s0:.2f}</b> (≈ <i>{per0:.2f} por número</i>)\n"
        f"🧮 Plano: {plano_txt}\n"
        f"📈 Odds por número: <b>{state['odds_num']:.2f}x</b>"
    )

@dp.channel_post_handler(content_types=["text"])
async def on_channel_post(msg: types.Message):
    # Lê APENAS do Canal A
    if msg.chat.id != CHANNEL_IN:
        return
    txt = (msg.text or "").strip()
    if not txt:
        return
    if not re_sinal.search(txt):
        return

    # Anti-duplicação / cooldown
    now = time.time()
    if now < state.get("cooldown_until", 0):
        return
    if not state["seguir_sinal"]:
        return

    # Extrai regras do texto e publica SEM checar confiança
    apos_num, alvos = extrai_regra_sinal(txt)
    if len(alvos) != 3:
        log.info("Sinal sem 3 alvos claros (ignorado): %s", txt)
        return

    msg_txt = build_plan_text(apos_num, alvos)

    try:
        await bot.send_message(CHANNEL_OUT, msg_txt, parse_mode="HTML")
        log.info("Publicado no canal B: %s", CHANNEL_OUT)
    except Exception as e:
        log.error("Falha ao publicar no canal B: %s", e)

    state["cooldown_until"] = now + COOLDOWN_S
    save_state()

# ============== FASTAPI / WEBHOOK =========
app = FastAPI()

@app.get("/healthz")
def healthz(): 
    return {"ok": True}

@app.on_event("startup")
async def on_startup():
    if not PUBLIC_URL:
        log.warning("PUBLIC_URL não definido; defina depois da 1ª implantação.")
        return
    await bot.delete_webhook(drop_pending_updates=True)
    await bot.set_webhook(f"{PUBLIC_URL}/webhook/{BOT_TOKEN}")
    log.info("Webhook configurado: %s/webhook/<token>", PUBLIC_URL)

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.body()
    update = types.Update(**json.loads(data.decode("utf-8")))
    # >>> Fix de contexto (aiogram v2 em FastAPI)
    Bot.set_current(bot)
    Dispatcher.set_current(dp)
    # ---------------------------------------
    await dp.process_update(update)
    return {"ok": True}