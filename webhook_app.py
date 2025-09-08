# -*- coding: utf-8 -*-
# webhook_app.py — Bot Telegram + Executor interno (FastAPI + Playwright)
# - Painel com Mult G1 (G0=1.00), Modo REAL, Sync Banca (Blaze)
# - Executor interno (Playwright) para Blaze (Fan Tan + Double básicos)
# ATENÇÃO: Automatização pode violar ToS da casa. Use por sua conta e risco.

import os, re, json, time, logging, asyncio
from typing import List, Optional, Tuple, Literal

from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from pydantic import BaseModel, conlist, confloat

# ========= LOG =========
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("guardiao-webhook")

# ========= CONFIG (ENV) =========
BOT_TOKEN  = os.getenv("TG_BOT_TOKEN")
PUBLIC_URL = (os.getenv("PUBLIC_URL") or "").rstrip("/")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))

if not BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN nas variáveis de ambiente.")
if not PUBLIC_URL.startswith("http"):
    log.warning("PUBLIC_URL ausente/invalid. Ex.: https://seu-app.onrender.com")

# Blaze / Executor
BLAZE_EMAIL = os.getenv("BLAZE_EMAIL", "").strip()
BLAZE_PASS  = os.getenv("BLAZE_PASS", "").strip()
HEADLESS    = os.getenv("HEADLESS", "1") == "1"
BASE_URL    = os.getenv("BASE_URL", "https://blaze.com").rstrip("/")
STORAGE     = os.getenv("STORAGE", "storage_state.json")

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
    "stake_base": 10.00,        # stake total da tentativa 1 (3 números somados)
    "gales_max": 1,             # G0..G3 (tentativas = gales_max+1)
    "ciclo_max": 1,             # quantos sinais serão usados para recuperar
    "multipliers": [1.0, 3.0],  # G0=1.0; G1 default=3.0 (G2/G3 livres)
    "ciclo_mult": 3.0,          # multiplica a perda do ciclo anterior (no PRÓXIMO sinal)
    "cooldown_until": 0.0,
    "modo_real": False,         # 🧪/💸 toggle
    "house": "blaze"            # casa alvo (apenas blaze neste arquivo)
}
risk = {
    "bankroll": 1000.00,        # banca mostrada no painel (pode ser sincronizada com a Blaze)
    "session_pnl": 0.0,
    "stop_win": 1000.00,
    "stop_loss": 1000.00,
    "prev_cycle_loss": 0.0,
    "cycle_left": 0,
    "open": None
}

def _load(path, default):
    try:
        if os.path.exists(path):
            default.update(json.load(open(path, "r", encoding="utf-8")))
    except Exception as e:
        log.warning("Falha ao carregar %s: %s", path, e)
    return default

def _save(path, obj):
    try:
        json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Falha ao salvar %s: %s", path, e)

_load(STATE_PATH, state)
_load(RISK_PATH, risk)
def save_state(): _save(STATE_PATH, state)
def save_risk():  _save(RISK_PATH,  risk)

# ========= PARSERS =========
re_sinal   = re.compile(r"\bENTRADA\s+CONFIRMADA\b", re.I)
re_apos    = re.compile(r"Entrar\s+ap[oó]s\s+o\s+([1-4])", re.I)
re_alvos   = re.compile(r"apostar\s+em\s+Ssh\s+([1-4])[\s\-\|]+([1-4])[\s\-\|]+([1-4])", re.I)

re_close   = re.compile(r"\bAPOSTA\s+ENCERRADA\b", re.I)
re_green   = re.compile(r"(GREEN|✅)", re.I)
re_red     = re.compile(r"(RED|❌)", re.I)

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

def eh_resultado(txt:str) -> Optional[int]:
    up = (txt or "").upper()
    if not re_close.search(up):
        return None
    if re_green.search(up): return 1
    if re_red.search(up):   return 0
    return None

# ========= CÁLCULOS =========
def lucro_liquido_no_acerto(por_num: float) -> float:
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

# ========= EXECUTOR INTERNO (Playwright) =========
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

_pw = None
_browser: Browser | None = None
_ctx: BrowserContext | None = None
_page: Page | None = None
_exec_lock = asyncio.Lock()  # evita corrida de múltiplas apostas

async def _login_if_needed(page: Page):
    try:
        await page.goto(f"{BASE_URL}/pt", wait_until="domcontentloaded")
        if await page.locator('button:has-text("Entrar")').count() == 0:
            return
    except:
        pass
    if not BLAZE_EMAIL or not BLAZE_PASS:
        raise RuntimeError("BLAZE_EMAIL/PASS não definidos para login.")
    await page.goto(f"{BASE_URL}/pt/?modal=auth&tab=login", wait_until="domcontentloaded")
    await page.fill('input[type="email"]', BLAZE_EMAIL)
    await page.fill('input[type="password"]', BLAZE_PASS)
    await page.click('button:has-text("Entrar")')
    await page.wait_for_timeout(3000)

async def ensure_session() -> Page:
    global _pw, _browser, _ctx, _page
    if _page:
        return _page
    _pw = await async_playwright().start()
    _browser = await _pw.chromium.launch(headless=HEADLESS)
    storage_state = STORAGE if os.path.exists(STORAGE) else None
    _ctx = await _browser.new_context(storage_state=storage_state)
    _page = await _ctx.new_page()
    await _page.goto(f"{BASE_URL}/pt", wait_until="domcontentloaded")
    await _login_if_needed(_page)
    try: await _ctx.storage_state(path=STORAGE)
    except: pass
    return _page

async def bet_double(page: Page, color: Literal["black","red","white"], amount: float):
    await page.goto(f"{BASE_URL}/pt/games/double", wait_until="domcontentloaded")
    if await page.locator('input[type="number"]').count():
        await page.fill('input[type="number"]', f"{amount:.2f}")
    elif await page.locator('input').count():
        await page.fill('input', f"{amount:.2f}")
    color_selector_map = {
        "red":   'button:has-text("Vermelho"), button[aria-label="Apostar no Vermelho"]',
        "black": 'button:has-text("Preto"), button[aria-label="Apostar no Preto"]',
        "white": 'button:has-text("Branco"), button[aria-label="Apostar no Branco"]',
    }
    await page.click(color_selector_map[color])

async def bet_fantan(page: Page, targets: List[int], amount: float):
    await page.goto(f"{BASE_URL}/pt/games/fantan", wait_until="domcontentloaded")
    # valor global
    if await page.locator('input[type="number"]').count():
        await page.fill('input[type="number"]', f"{amount:.2f}")
    elif await page.locator('input').count():
        await page.fill('input', f"{amount:.2f}")
    # cliques 1..4 com fallback simples (ajuste se necessário)
    for n in targets:
        if n not in (1,2,3,4):
            raise ValueError(f"Target inválido: {n}")
        tried = [
            f'[data-testid="fantan-cell-{n}"]',
            f'button[aria-label="Apostar no {n}"]',
            f'button:has-text("{n}")'
        ]
        done = False
        for sel in tried:
            loc = page.locator(sel)
            if await loc.count():
                try:
                    await loc.first.scroll_into_view_if_needed()
                    await loc.first.click()
                    done = True
                    break
                except:
                    pass
        if not done:
            raise RuntimeError(f"Não consegui clicar no alvo {n} — ajuste seletores.")

# ========== Leitura de SALDO Blaze ==========
def _parse_brl_to_float(txt: str) -> float:
    if not txt: return 0.0
    t = txt.replace("R$", "").strip()
    t = re.sub(r"\.", "", t)     # remove milhar
    t = t.replace(",", ".")
    t = re.sub(r"[^0-9.]", "", t)
    try: return float(t)
    except: return 0.0

async def get_blaze_balance(page: Page) -> float:
    await page.goto(f"{BASE_URL}/pt", wait_until="domcontentloaded")
    candidates = [
        '[data-testid="header-balance"]',
        '[data-testid="balance"]',
        'header >> text=/R\\$/',
        'text=/R\\$\\s*[0-9\\.,]+/'
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel)
            if await loc.count():
                txt = await loc.first.inner_text()
                val = _parse_brl_to_float(txt)
                if val >= 0:
                    return round(val, 2)
        except:
            pass
    try:
        content = await page.content()
        m = re.search(r"R\$\s*[0-9\.\,]+", content or "")
        if m:
            return round(_parse_brl_to_float(m.group(0)), 2)
    except:
        pass
    return 0.0

async def sync_bankroll_from_blaze() -> float:
    async with _exec_lock:
        page = await ensure_session()
        bal = await get_blaze_balance(page)
        if bal >= 0:
            risk["bankroll"] = round(bal, 2)
            save_risk()
        return bal

# ========== Funções de aposta (wrapper) ==========
async def executor_place_bet(game:str, targets:List[int], stake_per_number:float) -> bool:
    if stake_per_number <= 0:
        return False
    async with _exec_lock:
        page = await ensure_session()
        if game == "double":
            mapa = {1:"black", 2:"red", 3:"white"}
            color = mapa.get(targets[0], "red")
            await bet_double(page, color, stake_per_number)
        elif game == "fantan":
            await bet_fantan(page, targets, stake_per_number)
        else:
            raise RuntimeError("Jogo não suportado.")
        # Atualiza sessão e tenta ler saldo após apostar
        try:
            await _ctx.storage_state(path=STORAGE)
            bal = await get_blaze_balance(page)
            if bal >= 0:
                risk["bankroll"] = round(bal, 2)
                save_risk()
        except:
            pass
    return True

# ========= UI (PAINEL) =========
def kb_painel():
    kb = InlineKeyboardMarkup(row_width=3)
    kb.row(
        InlineKeyboardButton("✖️ 1x", callback_data="preset_1"),
        InlineKeyboardButton("✖️ 2x", callback_data="preset_2"),
        InlineKeyboardButton("✖️ 3x", callback_data="preset_3"),
        InlineKeyboardButton("✖️ 4x", callback_data="preset_4"),
    )
    kb.row(InlineKeyboardButton("✏️ Mult G1", callback_data="set_mult_g1"))
    kb.row(
        InlineKeyboardButton("✏️ Stake", callback_data="set_stake"),
        InlineKeyboardButton("💼 Banca (local)", callback_data="set_banca"),
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
    modo = "💸 Modo: REAL" if state.get("modo_real") else "🧪 Modo: SIMULAÇÃO"
    kb.row(InlineKeyboardButton(modo, callback_data="toggle_mode"))
    # Sync da banca (Blaze)
    kb.row(InlineKeyboardButton("🔃 Sync Banca (Blaze)", callback_data="sync_bankroll"))
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
        f"💼 Banca (local): <b>R${risk['bankroll']:.2f}</b> | Sessão: <b>{risk['session_pnl']:.2f}</b>\n"
        f"🔁 Modo: <b>{'REAL' if state.get('modo_real') else 'SIMULAÇÃO'}</b> | 🏦 Casa: <b>{state.get('house','blaze')}</b>\n"
        f"🧮 Plano (até G{state['gales_max']}): {resumo_plano(state['multipliers'][:state['gales_max']+1], state['stake_base'])}\n"
        f"Seguir: <b>{'ON' if state['seguir'] else 'OFF'}</b>"
    )

# ========= COMANDOS =========
@dp.message_handler(commands=["start"])
async def cmd_start(m: types.Message):
    state["dm_user_id"] = m.chat.id; save_state()
    await m.answer(
        "🤖 <b>Guardião Auto</b>\n"
        "• Lê <b>ENTRADA CONFIRMADA</b> no canal A e simula aqui.\n"
        "• G1 é <b>imediato</b> com texto exato.\n"
        "• <b>Modo REAL</b>: executor Playwright interno para Blaze.\n"
        "• Use <b>🔃 Sync Banca (Blaze)</b> para ler o saldo real.",
        disable_web_page_preview=True
    )
    await m.answer(painel_texto(), reply_markup=kb_painel())

@dp.message_handler(commands=["painel","config","status"])
async def cmd_painel(m: types.Message):
    await m.answer(painel_texto(), reply_markup=kb_painel())

# ========= ENTRADA POR PROMPT =========
AWAIT_NUMERIC = {}  # {user_id: "stake"|"banca"|"sw"|"sl"|"mult_g1"}

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
    elif data=="toggle_mode": state["modo_real"]=not state.get("modo_real", False); changed=True

    elif data=="preset_1": state["multipliers"]=[1.0]*(state["gales_max"]+1); changed=True
    elif data=="preset_2": state["multipliers"]=[1.0]+[2.0]*state["gales_max"]; changed=True
    elif data=="preset_3": state["multipliers"]=[1.0]+[3.0]*state["gales_max"]; changed=True
    elif data=="preset_4": state["multipliers"]=[1.0]+[4.0]*state["gales_max"]; changed=True

    elif data=="set_stake":
        AWAIT_NUMERIC[uid]="stake"
        await call.message.reply("💬 Digite o novo stake total da 1ª tentativa (ex: 12.50):")
        await call.answer(); return
    elif data=="set_banca":
        AWAIT_NUMERIC[uid]="banca"
        await call.message.reply("💬 Digite a banca (local) desejada (ex: 1000):")
        await call.answer(); return
    elif data=="set_sw":
        AWAIT_NUMERIC[uid]="sw"
        await call.message.reply("💬 Digite o novo Stop Win (ex: 300):")
        await call.answer(); return
    elif data=="set_sl":
        AWAIT_NUMERIC[uid]="sl"
        await call.message.reply("💬 Digite o novo Stop Loss (ex: 300):")
        await call.answer(); return
    elif data=="set_mult_g1":
        AWAIT_NUMERIC[uid]="mult_g1"
        await call.message.reply("💬 Digite o multiplicador do G1 (máx. 4.00). Ex.: 3 ou 2,5")
        await call.answer(); return

    elif data == "sync_bankroll":
        await call.answer("Sincronizando...")
        try:
            bal = await sync_bankroll_from_blaze()
            if bal >= 0:
                await call.message.reply(f"💼 Banca (Blaze) sincronizada: <b>R${bal:.2f}</b>", parse_mode="HTML")
            else:
                await call.message.reply("⚠️ Não consegui ler o saldo na Blaze agora.")
        except Exception as e:
            await call.message.reply(f"⚠️ Falha ao sincronizar banca: {e}")
        # Atualiza painel depois da sync
        try:
            await call.message.edit_text(painel_texto(), reply_markup=kb_painel())
        except:
            pass
        return

    if changed:
        if state["multipliers"]:
            state["multipliers"][0] = 1.0  # G0 sempre 1.0
        save_state()
        try: await call.message.edit_text(painel_texto(), reply_markup=kb_painel())
        except: pass
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
    txt  = m.text or ""
    try:
        if kind in ("stake","banca","sw","sl"):
            v = float(txt.replace(",", "."))
            if v <= 0: raise ValueError()
            if kind=="stake":
                state["stake_base"]=round(v,2); save_state()
                await m.reply(f"✅ Stake ajustado para R${state['stake_base']:.2f}", parse_mode="HTML")
            elif kind=="banca":
                risk["bankroll"]=round(v,2); save_risk()
                await m.reply(f"✅ Banca (local) ajustada para R${risk['bankroll']:.2f}", parse_mode="HTML")
            elif kind=="sw":
                risk["stop_win"]=round(v,2); save_risk()
                await m.reply(f"✅ Stop Win ajustado: R${risk['stop_win']:.2f}", parse_mode="HTML")
            elif kind=="sl":
                risk["stop_loss"]=round(v,2); save_risk()
                await m.reply(f"✅ Stop Loss ajustado: R${risk['stop_loss']:.2f}", parse_mode="HTML")
        elif kind=="mult_g1":
            v = float(txt.replace(",", "."))
            if v <= 0 or v > 4.0: raise ValueError()
            if not state.get("multipliers"): state["multipliers"]=[1.0, v]
            else:
                if len(state["multipliers"]) == 1: state["multipliers"].append(v)
                else: state["multipliers"][1] = v
                state["multipliers"][0] = 1.0
            state["multipliers"] = [round(x,2) for x in state["multipliers"]]
            save_state()
            await m.reply(f"✅ Multiplicador do G1 ajustado para x{state['multipliers'][1]:.2f} (G0=x1.00)", parse_mode="HTML")
        await m.reply(painel_texto(), reply_markup=kb_painel())
    except:
        await m.reply("❗ Valor inválido.")

# ========= CORE: abrir/fechar =========
def abrir_operacao(apos:int, alvos:List[int]):
    base = state["stake_base"]
    mults = state["multipliers"][:state["gales_max"]+1] or [1.0]
    if mults: mults[0] = 1.0
    if risk["cycle_left"] > 0 and risk["prev_cycle_loss"] > 0:
        base = max(base, round(risk["prev_cycle_loss"] * state["ciclo_mult"], 2))
        risk["cycle_left"] -= 1; save_risk()
    op = {"apos": apos, "alvos": alvos, "base": base, "mult": mults, "step": 0, "closed": False}
    risk["open"] = op; save_risk()
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
        f"💼 Banca (local): <b>R${risk['bankroll']:.2f}</b> | Sessão: <b>{risk['session_pnl']:.2f}</b>"
    )
    if state["dm_user_id"]:
        await bot.send_message(state["dm_user_id"], txt)

def valor_tentativa(op):
    m = op["mult"][op["step"]]
    return plano_por_tentativa(op["base"], m)

async def fechar_com_green():
    op = risk.get("open")
    if not op or op["closed"]: return
    _, _, lucro = valor_tentativa(op)
    gastos_previos = total_gasto_ate(op["mult"], op["base"], op["step"])
    pnl = round(lucro - gastos_previos, 2)
    risk["session_pnl"] = round(risk["session_pnl"] + pnl, 2)
    risk["bankroll"]    = round(risk["bankroll"] + pnl, 2)  # banca local (a real é lida na sync)
    op["closed"] = True; risk["open"] = None
    risk["prev_cycle_loss"] = 0.0; risk["cycle_left"] = 0; save_risk()
    if state["dm_user_id"]:
        await bot.send_message(state["dm_user_id"],
            f"✅ <b>GREEN</b> (step {op['step']+1}) | Lucro: <b>R${pnl:.2f}</b> | "
            f"Sessão: <b>{risk['session_pnl']:.2f}</b> | Banca (local): <b>R${risk['bankroll']:.2f}</b>")
    await checar_stops()

async def avancar_depois_de_red():
    op = risk.get("open")
    if not op or op["closed"]: return
    op["step"] += 1
    if op["step"] >= len(op["mult"]):
        preju = total_gasto_ate(op["mult"], op["base"], len(op["mult"]))
        risk["session_pnl"] = round(risk["session_pnl"] - preju, 2)
        risk["bankroll"]    = round(risk["bankroll"] - preju, 2)
        risk["prev_cycle_loss"] = preju
        risk["cycle_left"] = state["ciclo_max"]
        op["closed"] = True; risk["open"] = None; save_risk()
        if state["dm_user_id"]:
            await bot.send_message(state["dm_user_id"],
                f"❌ <b>RED</b> | Perda: <b>R${preju:.2f}</b> | "
                f"Sessão: <b>{risk['session_pnl']:.2f}</b> | Banca (local): <b>{risk['bankroll']:.2f}</b>")
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

# ========= PROCESSADOR DO CANAL =========
async def _process_channel_text(msg: types.Message):
    if not CHANNEL_ID or msg.chat.id != CHANNEL_ID:
        return
    txt = (msg.text or "").strip()
    if not txt:
        return

    # (A) G1 imediato no mesmo sinal
    if risk.get("open") and not risk["open"]["closed"] and txt == G1_TEXTO_EXATO:
        op = risk["open"]
        if op["step"] == 0 and len(op["mult"]) > 1:
            op["step"] = 1; save_risk()
            s1, per1, _ = plano_por_tentativa(op["base"], op["mult"][1])
            if state["modo_real"]:
                try:
                    await executor_place_bet("fantan", op["alvos"], per1)
                    if state["dm_user_id"]:
                        await bot.send_message(state["dm_user_id"],
                            f"💸 Apostando G1 (REAL): {op['alvos']} | ~R${per1:.2f} por número (total R${s1:.2f})")
                except Exception as e:
                    if state["dm_user_id"]:
                        await bot.send_message(state["dm_user_id"], f"⚠️ Falha ao apostar G1: {e}")
            if state["dm_user_id"]:
                await bot.send_message(
                    state["dm_user_id"],
                    f"🟠 <b>G1 ativado</b> (mesmo sinal)\n"
                    f"💵 Tentativa 2 (G1) total: <b>R${s1:.2f}</b> (≈ <i>{per1:.2f} por número</i>)"
                )

    # (B) GREEN/RED fecha/avança
    r = eh_resultado(txt)
    if r is not None:
        if risk.get("open") and not risk["open"]["closed"]:
            if r == 1: await fechar_com_green()
            elif r == 0: await avancar_depois_de_red()
        return

    # (C) Nova entrada
    if not eh_sinal(txt): return
    if not state["seguir"]: return
    if time.time() < state.get("cooldown_until", 0.0): return
    if (risk["session_pnl"] >= risk["stop_win"]) or (risk["session_pnl"] <= -risk["stop_loss"]):
        await checar_stops(); return

    regra = extrai_regra_sinal(txt)
    if not regra:
        log.info("Sinal sem padrão esperado: %s", txt[:160])
        return
    apos, alvos = regra
    if len(alvos) != 3:
        return

    op = abrir_operacao(apos, alvos)
    await publicar_plano(op)

    # Aposta G0 se Modo REAL
    if state["modo_real"]:
        s0, per0, _ = plano_por_tentativa(op["base"], op["mult"][0])
        try:
            await executor_place_bet("fantan", op["alvos"], per0)
            if state["dm_user_id"]:
                await bot.send_message(state["dm_user_id"],
                    f"💸 Apostando (REAL): {op['alvos']} | ~R${per0:.2f} por número (total R${s0:.2f})")
        except Exception as e:
            if state["dm_user_id"]:
                await bot.send_message(state["dm_user_id"], f"⚠️ Falha ao apostar G0: {e}")

    state["cooldown_until"] = time.time() + 5; save_state()

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
    # Inicia webhook do Telegram
    if not PUBLIC_URL:
        log.warning("PUBLIC_URL não definido; defina no Render.")
        return
    await bot.delete_webhook(drop_pending_updates=True)
    await bot.set_webhook(f"{PUBLIC_URL}/webhook/{BOT_TOKEN}")
    log.info("Webhook configurado em %s/webhook/<token>", PUBLIC_URL)

    # Prepara Playwright (executor interno)
    try:
        await ensure_session()
        # (Opcional) sincroniza banca ao iniciar
        try:
            bal = await sync_bankroll_from_blaze()
            log.info("Banca (Blaze) inicial: R$ %.2f", bal)
        except Exception as e:
            log.warning("Não consegui sincronizar banca no startup: %s", e)
        log.info("Executor Playwright pronto.")
    except Exception as e:
        log.warning("Executor não inicializado agora (lazy). Motivo: %s", e)

@app.post(f"/webhook/{BOT_TOKEN}")
async def tg_webhook(request: Request):
    data = await request.body()
    update = types.Update(**json.loads(data.decode("utf-8")))
    Bot.set_current(bot); Dispatcher.set_current(dp)
    await dp.process_update(update)
    return {"ok": True}