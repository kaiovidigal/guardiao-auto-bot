# -*- coding: utf-8 -*-
# webhook_app.py — Guardião Auto (DM-only) | G1 imediato (texto exato), ciclo no próximo,
# stake editável, e override embutido para priorizar nº1 + aviso "não vai respeitar"

import os, re, json, time, logging
from collections import deque
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
    "dm_user_id": 0,            # preenchido no /start
    "seguir": True,             # seguir sinais
    "stake_base": 10.00,        # valor total da 1ª tentativa (3 números somados)
    "gales_max": 1,             # G0..G3  (tentativas = gales_max+1)
    "ciclo_max": 1,             # quantos sinais serão usados para recuperar
    "multipliers": [1.0, 3.0],  # padrão: G1=3x imediato
    "ciclo_mult": 3.0,          # multiplica a perda do ciclo anterior (só no PRÓXIMO sinal)
    "cooldown_until": 0.0
}
risk = {
    "bankroll": 1000.00,
    "session_pnl": 0.0,
    "stop_win": 1000.00,
    "stop_loss": 1000.00,
    "prev_cycle_loss": 0.0,     # perda carregada p/ recuperar
    "cycle_left": 0,            # quantos sinais restam no modo ciclo
    "open": None                # operação aberta
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

# ========= BUFFERS (para priorizar nº1) =========
ultimos_numeros = deque(maxlen=20)

# ========= PARSERS =========
re_sinal   = re.compile(r"\bENTRADA\s+CONFIRMADA\b", re.I)
re_apos    = re.compile(r"Entrar\s+ap[oó]s\s+o\s+([1-4])", re.I)
re_alvos   = re.compile(r"apostar\s+em\s+Ssh\s+([1-4])[\s\-\|]+([1-4])[\s\-\|]+([1-4])", re.I)

# Sequência para aprender últimos resultados Fantan (para a lógica do nº1)
re_seq     = re.compile(r"Sequ[eê]ncia[:\s]*([^\n]+)", re.I)

# Resultado: **apenas GREEN e RED** (NEUTRO removido)
re_close   = re.compile(r"\bAPOSTA\s+ENCERRADA\b", re.I)
re_green   = re.compile(r"(GREEN|✅)", re.I)
re_red     = re.compile(r"(RED|❌)", re.I)

# G1: exatamente este texto (com emoji e símbolo de grau U+00B0)
G1_TEXTO_EXATO = "🔁 Estamos no 1° gale"

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

def extrai_sequencia(txt: str) -> List[int]:
    m = re_seq.search(txt or "")
    if not m:
        return []
    return [int(x) for x in re.findall(r"[1-4]", m.group(1))]

def eh_resultado(txt:str) -> Optional[int]:
    """
    1 = GREEN, 0 = RED, None = não é fechamento.
    Somente considera GREEN/RED; se encerrar sem essas palavras, ignora.
    """
    up = (txt or "").upper()
    if not re_close.search(up):
        return None
    if re_green.search(up): return 1
    if re_red.search(up):   return 0
    return None  # sem NEUTRO, sem forçar RED

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
        InlineKeyboardButton("✏️ Stake", callback_data="set_stake"),
        InlineKeyboardButton("💼 Banca", callback_data="set_banca"),
    )
    kb.row(
        InlineKeyboardButton("🟢 Stop Win", callback_data="set_sw"),
        InlineKeyboardButton("🔴 Stop Loss", callback_data="set_sl"),
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
        f"💰 Stake: <b>{state['stake_base']:.2f}</b> | ♻️ Gales: <b>{state['gales_max']}</b> | 🔁 Ciclo: <b>{state['ciclo_max']}</b>\n"
        f"✖️ Mults (G0..Gn): <b>{mults}</b>\n"
        f"📌 Ciclo Mult (próx. sinal): <b>x{state['ciclo_mult']:.2f}</b>\n"
        f"🎯 Odds por nº (fixo): <b>{ODDS_TOTAL:.2f}x</b>\n"
        f"💼 Banca: <b>R${risk['bankroll']:.2f}</b> | PnL Sessão: <b>{risk['session_pnl']:.2f}</b>\n"
        f"🧮 Plano (até G{state['gales_max']}): {resumo_plano(state['multipliers'][:state['gales_max']+1], state['stake_base'])}\n"
        f"Seguir: <b>{'ON' if state['seguir'] else 'OFF'}</b>"
    )

@dp.message_handler(commands=["start"])
async def cmd_start(m: types.Message):
    state["dm_user_id"] = m.chat.id
    save_state()
    await m.answer(
        "🤖 <b>Guardião Auto (DM)</b>\n"
        "• Lê <b>ENTRADA CONFIRMADA</b> no canal A e executa simulação aqui.\n"
        "• Odds por número fixas (3.85x). Somente GREEN/RED contam.\n"
        "• G1 é imediato no mesmo sinal; <b>ciclo</b> só no <b>próximo sinal</b>.\n"
        f"• Canal A: <code>{CHANNEL_ID}</code>",
        disable_web_page_preview=True
    )
    await m.answer(painel_texto(), reply_markup=kb_painel())

@dp.message_handler(commands=["painel","config","status"])
async def cmd_painel(m: types.Message):
    await m.answer(painel_texto(), reply_markup=kb_painel())

# ====== entrada de números por prompt ======
AWAIT_NUMERIC = {}  # {user_id: "stake"|"banca"|"sw"|"sl"}

@dp.callback_query_handler(lambda c: True)
async def on_cb(call: types.CallbackQuery):
    uid = call.from_user.id
    data = call.data
    changed=False

    if data=="gales_+": state["gales_max"]=min(3, state["gales_max"]+1); changed=True
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

    elif data=="set_stake":
        AWAIT_NUMERIC[uid]="stake"
        await call.message.reply("💬 Digite o novo <b>stake total</b> da 1ª tentativa (ex: 12.50):", parse_mode="HTML")
        await call.answer(); return
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
        if kind=="stake":
            state["stake_base"]=round(v,2)
            save_state()
            await m.reply(f"✅ Stake ajustado para <b>R${state['stake_base']:.2f}</b>", parse_mode="HTML")
        elif kind=="banca":
            risk["bankroll"]=round(v,2)
            save_risk()
            await m.reply(f"✅ Banca ajustada para <b>R${risk['bankroll']:.2f}</b>", parse_mode="HTML")
        elif kind=="sw":
            risk["stop_win"]=round(v,2)
            save_risk()
            await m.reply(f"✅ Stop Win ajustado: <b>R${risk['stop_win']:.2f}</b>", parse_mode="HTML")
        elif kind=="sl":
            risk["stop_loss"]=round(v,2)
            save_risk()
            await m.reply(f"✅ Stop Loss ajustado: <b>R${risk['stop_loss']:.2f}</b>", parse_mode="HTML")
        await m.reply(painel_texto(), reply_markup=kb_painel())
    except:
        await m.reply("❗ Valor inválido.")

# ========= AJUSTE: priorizar nº1 quando fizer sentido =========
def ajusta_alvos_priorizar_1(apos: Optional[int], alvos: List[int]) -> Tuple[List[int], Optional[str]]:
    """
    Regras de override para priorizar o nº1:
    - Se 'apos' == 1 e 1 NÃO estiver nos alvos -> usar [1,2,3].
    - OU se os dois últimos observados forem 1-1 e 1 NÃO estiver nos alvos -> usar [1,2,3].
    Retorna (alvos_modificados, motivo_textual_ou_None).
    """
    # Checa últimos 2
    ult2 = list(ultimos_numeros)[-2:]
    dois_ultimos_sao_11 = (len(ult2) == 2 and ult2[0] == 1 and ult2[1] == 1)

    if alvos and 1 not in alvos and ((apos == 1) or dois_ultimos_sao_11):
        return [1, 2, 3], ("apos=1" if apos == 1 else "sequência 1-1")
    return alvos, None

# ========= CORE: abrir/fechar =========
def abrir_operacao(apos:int, alvos:List[int]):
    """
    Abre nova operação.
    - G1 (multiplier step 1) é IMEDIATO quando vier o texto exato '🔁 Estamos no 1° gale'.
    - Ciclo só será aplicado NO PRÓXIMO SINAL, ajustando a base.
    """
    base = state["stake_base"]
    mults = state["multipliers"][:state["gales_max"]+1] or [1.0]