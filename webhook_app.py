# webhook_app.py
import os, re, json, time, logging
from collections import deque
from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# ============== LOGGING ==============
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("guardiao-risco-auto")

# ============== CONFIG BÁSICA ==============
BOT_TOKEN   = os.getenv("TG_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Faltando TG_BOT_TOKEN")

CHANNEL_ID  = int(os.getenv("CHANNEL_ID", "-1002810508717"))  # seu canal de sinais
PUBLIC_URL  = os.getenv("PUBLIC_URL", "").rstrip("/")
CONF_LIMIAR = float(os.getenv("CONF_LIMIAR", "0.92"))         # conf mínima para CONFIRMAR
COOLDOWN_S  = int(os.getenv("COOLDOWN_S", "10"))              # anti-flood

bot = Bot(token=BOT_TOKEN, parse_mode=types.ParseMode.HTML)
dp  = Dispatcher(bot)

# ============== PERSISTÊNCIA ==============
os.makedirs("data", exist_ok=True)
STATE_FILE = "data/state.json"
RISK_FILE  = "data/risk.json"

# estado do bot (painel/config)
state = {
    "seguir_sinal": True,        # só age quando vier ENTRADA CONFIRMADA
    "cooldown_until": 0.0,
    "limiar": CONF_LIMIAR,

    # gestão de stake/gale/ciclo
    "stake_base": 5.00,          # valor base (soma dos 3 números na tentativa 0)
    "gales_max": 1,              # 0..3
    "ciclo_max": 1,              # nº de ciclos a rodar antes de pausar (1 = ciclo único)
    "gale_mult": 2.0,            # multiplicador único (para botões rápidos)
    "multipliers": [1.0, 2.0],   # até 4; pode configurar via /mult 1,2,3,4

    # modo exibição
    "modo_real": False,          # SIMULADO sempre (não usa navegador)
}
def load_state():
    try:
        if os.path.exists(STATE_FILE):
            state.update(json.load(open(STATE_FILE, "r", encoding="utf-8")))
    except Exception as e:
        log.warning("Falha ao carregar state: %s", e)
def save_state():
    try:
        json.dump(state, open(STATE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Falha ao salvar state: %s", e)
load_state()

# risco/banca/stops e trade aberto
risk = {
    "bankroll": 100.00,     # banca virtual
    "session_pnl": 0.0,     # PnL da sessão
    "stop_win": 50.00,
    "stop_loss": 50.00,
    "odds_total": 3.85,     # retorno total por número no acerto (Fan Tan ~3.85x)
    "prev_cycle_loss": 0.0, # prejuízo do ciclo anterior (para recuperação)
    "open": None            # operação aberta (vincula aos próximos RED/GREEN)
}
def load_risk():
    try:
        if os.path.exists(RISK_FILE):
            risk.update(json.load(open(RISK_FILE, "r", encoding="utf-8")))
    except Exception as e:
        log.warning("Falha ao carregar risk: %s", e)
def save_risk():
    try:
        json.dump(risk, open(RISK_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Falha ao salvar risk: %s", e)
load_risk()

# ============== MÉTRICAS DE APRENDIZADO (para CONFIRMAR) ==============
hist_long  = deque(maxlen=300)   # 0/1
hist_short = deque(maxlen=30)    # 0/1
ultimos_numeros = deque(maxlen=120)   # 1..4
contagem_num = [0, 0, 0, 0, 0]
transicoes   = [[0]*5 for _ in range(5)]

def atualiza_estat_num(seq_nums):
    for n in seq_nums:
        if 1 <= n <= 4:
            if ultimos_numeros:
                prev = ultimos_numeros[-1]
                if 1 <= prev <= 4:
                    transicoes[prev][n] += 1
            ultimos_numeros.append(n)
            contagem_num[n] += 1

def winrate(d): d=list(d); return (sum(d)/len(d)) if d else 0.0
def volatilidade(d):
    d=list(d)
    if len(d)<10: return 0.0
    trocas=sum(1 for i in range(1,len(d)) if d[i]!=d[i-1])
    return trocas/(len(d)-1)
def streak_loss(d):
    d=list(d); s=0; mx=0
    for x in d:
        if x==0: s+=1; mx=max(mx,s)
        else: s=0
    return mx

def probs_depois(depois_de):
    alpha=1.0
    def dist_global():
        tot = sum(contagem_num[1:5]) + 4*alpha
        return [0] + [(contagem_num[i] + alpha)/tot for i in range(1,5)]
    if not (isinstance(depois_de,int) and 1<=depois_de<=4):
        return dist_global()
    total = sum(transicoes[depois_de][1:5])
    if total < 8:
        return dist_global()
    tot = total + 4*alpha
    return [0] + [(transicoes[depois_de][i] + alpha)/tot for i in range(1,5)]

def risco_por_numeros(apos_num, alvos):
    if not alvos: return 0.5
    ultimo_ref = apos_num if apos_num else (ultimos_numeros[-1] if ultimos_numeros else None)
    probs = probs_depois(ultimo_ref)
    p_hit = sum(probs[a] for a in alvos if 1<=a<=4)
    return max(0.0, min(1.0, 1.0 - p_hit))

def conf_final(short_wr, long_wr, vol, max_reds, risco_num):
    base = 0.55*short_wr + 0.30*long_wr + 0.10*(1.0 - vol) + 0.05*(1.0 - risco_num)
    pena = 0.0
    if max_reds >= 3: pena += 0.05*(max_reds-2)
    if vol > 0.6:     pena += 0.05
    return max(0.0, min(1.0, base - pena))

# ============== PARSERS DE TEXTO DO CANAL ==============
re_sinal   = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
re_seq     = re.compile(r"Sequ[eê]ncia[:\s]*([^\n]+)", re.I)
re_apos    = re.compile(r"Entrar\s+ap[oó]s\s+o\s+([1-4])", re.I)
re_apostar = re.compile(r"apostar\s+em\s+([A-Za-z]*\s*)?([1-4](?:[\s\-\|]*[1-4])*)", re.I)
re_red     = re.compile(r"\bRED\b", re.I)
re_close   = re.compile(r"APOSTA\s+ENCERRADA", re.I)
def eh_sinal(txt): return bool(re_sinal.search(txt or ""))
def extrai_sequencia(txt):
    m=re_seq.search(txt or ""); 
    if not m: return []
    return [int(x) for x in re.findall(r"[1-4]", m.group(1))]
def extrai_regra_sinal(txt):
    m1=re_apos.search(txt or ""); m2=re_apostar.search(txt or "")
    apos = int(m1.group(1)) if m1 else None
    alvos = [int(x) for x in re.findall(r"[1-4]", (m2.group(2) if m2 else ""))]
    return (apos, alvos)
def eh_resultado(txt):
    up=(txt or "").upper()
    if re_red.search(up) or re_close.search(up): return 0
    if "GREEN" in up or "WIN" in up or "✅" in up: return 1
    return None

# ============== AJUDA DE CÁLCULO (cobre 3 números) ==============
def lucro_liquido_no_acerto(por_num, odds_total):
    # ganha odds_total*por_num no número vencedor e perde 2*por_num
    return round((odds_total - 3.0) * por_num, 2)

def plano_por_tentativa(base_total, mult):
    """
    base_total: valor total da tentativa (soma dos 3 números) na ENTRADA (mult=1).
    mult: multiplicador para a tentativa específica.
    retorna (stake_total, stake_por_numero, lucro_liquido_se_acertar_essa)
    """
    stake_total = round(base_total * mult, 2)
    por_num = round(stake_total / 3.0, 2)
    lucro = lucro_liquido_no_acerto(por_num, risk["odds_total"])
    return stake_total, por_num, lucro

def required_base_for_recovery(prev_loss, want_profit, multipliers):
    """
    Queremos que, MESMO acertando no último gale, o lucro total do ciclo >= (prev_loss + want_profit).
    Lucro total ao acertar na tentativa j:
      PNL = - base_total*sum_{i<j} mult_i  +  lucro_liquido_no_acerto(base_total*mult_j/3)
          = base_total * [ (odds-3)*(mult_j/3) - sum_{i<j} mult_i ]
    Usamos j = último (pior caso). Se denominador <=0, não há recuperação com esses mult/odds.
    """
    j = len(multipliers) - 1
    sum_prev = sum(multipliers[:j])
    denom = (risk["odds_total"] - 3.0) * (multipliers[j] / 3.0) - sum_prev
    target = prev_loss + want_profit
    if denom <= 0:
        return None  # impossível recuperar
    base_total = target / denom
    return round(max(0.01, base_total), 2)

# ============== UI / PAINEL ==============
def kb_painel():
    modo = "🧪 SIMULADO"
    seguir = "🟢 Seguir: ON" if state["seguir_sinal"] else "⚪️ Seguir: OFF"
    # linha de presets de multiplicadores rápidos (até 4x)
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
    kb.row(InlineKeyboardButton(modo, callback_data="noop"),
           InlineKeyboardButton("🔄 Atualizar", callback_data="refresh"))
    return kb

def resumo_plano_text(multipliers, base_total):
    partes=[]
    tot=0.0
    for m in multipliers:
        s, per, _ = plano_por_tentativa(base_total, m)
        partes.append(f"{s:.2f}")
        tot += s
    return f"{' → '.join(partes)} = <b>{tot:.2f}</b>"

@dp.message_handler(commands=["start"])
async def cmd_start(m: types.Message):
    await m.answer(
        "<b>🤖 Guardião de Risco (AUTOMÁTICO)</b>\n"
        "• Lê apenas <b>ENTRADA CONFIRMADA</b>\n"
        "• Não usa navegador (SIMULAÇÃO)\n"
        "• /painel para configurar stake/gales/ciclo/multiplicadores/odds/stops\n"
        f"• Canal: <code>{CHANNEL_ID}</code>\n"
        f"• Limiar: <b>{state['limiar']:.2f}</b>\n",
        parse_mode="HTML"
    )

@dp.message_handler(commands=["painel","config"])
async def cmd_painel(m: types.Message):
    mults = ", ".join(f"{x:.2f}" for x in state["multipliers"])
    await m.answer(
        "⚙️ <b>PAINEL</b>\n"
        f"💰 Base: <b>{state['stake_base']:.2f}</b> | ♻️ Gales: <b>{state['gales_max']}</b> | 🔁 Ciclo: <b>{state['ciclo_max']}</b>\n"
        f"✖️ Mults: <b>{mults}</b> (use /mult 1,2,3,4 para definir)\n"
        f"📈 Mult rápido (botões): x{state['gale_mult']:.2f}\n"
        f"🎯 Odds por número: <b>{risk['odds_total']:.2f}x</b>\n"
        f"💼 Banca: <b>R${risk['bankroll']:.2f}</b> | PnL Sessão: <b>{risk['session_pnl']:.2f}</b>\n"
        f"🧮 Plano atual: {resumo_plano_text(state['multipliers'][:state['gales_max']+1], state['stake_base'])}\n"
        f"Seguir sinal: <b>{'ON' if state['seguir_sinal'] else 'OFF'}</b>",
        reply_markup=kb_painel(), parse_mode="HTML"
    )

@dp.callback_query_handler(lambda c: True)
async def on_cb(call: types.CallbackQuery):
    data = call.data
    changed=False
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
        # gera lista até 4 passos: [1, k, k, k] limitado a gales+1
        lst=[1.0]; 
        for _ in range(3): lst.append(float(k))
        state["multipliers"]=lst[:state["gales_max"]+1]
        changed=True

    if changed:
        save_state()
        try:
            await call.message.edit_reply_markup(kb_painel())
        except: pass
        await call.answer("Atualizado!")
    else:
        if data=="refresh": 
            try: await call.message.edit_reply_markup(kb_painel())
            except: pass
            await call.answer("OK")
        else:
            await call.answer()

# ======== COMANDOS DE CONTROLE (odds, stops, banca, mult) ========
@dp.message_handler(commands=["odds"])
async def cmd_odds(m: types.Message):
    try:
        v=float((m.get_args() or "").replace(",", "."))
        if v<3.0: raise ValueError()
        risk["odds_total"]=round(v,2); save_risk()
        await m.answer(f"✅ Odds por número: <b>{risk['odds_total']:.2f}x</b>", parse_mode="HTML")
    except:
        await m.answer("Use: /odds 3.85  (retorno total no acerto de UM número)")

@dp.message_handler(commands=["banca"])
async def cmd_banca(m: types.Message):
    try:
        v=float((m.get_args() or "").replace(",", "."))
        if v<=0: raise ValueError()
        risk["bankroll"]=round(v,2); save_risk()
        await m.answer(f"✅ Banca virtual: <b>R${risk['bankroll']:.2f}</b>", parse_mode="HTML")
    except:
        await m.answer("Use: /banca 250.00")

@dp.message_handler(commands=["stopwin"])
async def cmd_stopwin(m: types.Message):
    try:
        v=float((m.get_args() or "").replace(",", "."))
        if v<=0: raise ValueError()
        risk["stop_win"]=round(v,2); save_risk()
        await m.answer(f"✅ Stop Win: <b>R${risk['stop_win']:.2f}</b>", parse_mode="HTML")
    except:
        await m.answer("Use: /stopwin 50")

@dp.message_handler(commands=["stoploss"])
async def cmd_stoploss(m: types.Message):
    try:
        v=float((m.get_args() or "").replace(",", "."))
        if v<=0: raise ValueError()
        risk["stop_loss"]=round(v,2); save_risk()
        await m.answer(f"✅ Stop Loss: <b>R${risk['stop_loss']:.2f}</b>", parse_mode="HTML")
    except:
        await m.answer("Use: /stoploss 50")

@dp.message_handler(commands=["saldo"])
async def cmd_saldo(m: types.Message):
    await m.answer(
        "💼 <b>Controle</b>\n"
        f"• Banca: <b>R${risk['bankroll']:.2f}</b>\n"
        f"• PnL Sessão: <b>{risk['session_pnl']:.2f}</b>\n"
        f"• Stop Win: <b>R${risk['stop_win']:.2f}</b>\n"
        f"• Stop Loss: <b>R${risk['stop_loss']:.2f}</b>\n"
        f"• Odds (nº): <b>{risk['odds_total']:.2f}x</b>",
        parse_mode="HTML"
    )

@dp.message_handler(commands=["resetpnl"])
async def cmd_resetpnl(m: types.Message):
    risk["session_pnl"]=0.0
    risk["prev_cycle_loss"]=0.0
    save_risk()
    await m.answer("✅ PnL e prejuízo de ciclo zerados.")

@dp.message_handler(commands=["mult"])
async def cmd_mult(m: types.Message):
    try:
        raw=(m.get_args() or "").replace(" ", "")
        lst=[float(x.replace(",", ".")) for x in raw.split(",") if x]
        if not lst: raise ValueError()
        if len(lst)>4: lst=lst[:4]
        state["multipliers"]=lst[:state["gales_max"]+1]
        save_state()
        await m.answer(f"✅ Multiplicadores: <b>{', '.join(f'{x:.2f}' for x in state['multipliers'])}</b>", parse_mode="HTML")
    except:
        await m.answer("Use: /mult 1,2,2.5,3  (até 4 valores)")

@dp.message_handler(commands=["status"])
async def cmd_status(m: types.Message):
    short_wr=winrate(hist_short); long_wr=winrate(hist_long)
    vol=volatilidade(hist_short); reds=streak_loss(hist_short)
    ultimo=ultimos_numeros[-1] if ultimos_numeros else None
    await m.answer(
        "📊 <b>Status</b>\n"
        f"WR30: <b>{short_wr*100:.1f}%</b> | WR300: <b>{long_wr*100:.1f}%</b>\n"
        f"Volatilidade: <b>{vol:.2f}</b> | Max REDs: <b>{reds}</b>\n"
        f"Último número: <b>{ultimo}</b>\n"
        f"Limiar: <b>{state['limiar']:.2f}</b> | Seguir: <b>{'ON' if state['seguir_sinal'] else 'OFF'}</b>",
        parse_mode="HTML"
    )

# ============== STOP-WIN/LOSS ==============
def check_stops_and_pause():
    if risk["session_pnl"] >= risk["stop_win"]:
        state["seguir_sinal"]=False; save_state(); save_risk()
        bot.loop.create_task(bot.send_message(CHANNEL_ID, "✅ <b>STOP WIN atingido</b>. Pausando entradas.", parse_mode="HTML"))
        return True
    if risk["session_pnl"] <= -risk["stop_loss"]:
        state["seguir_sinal"]=False; save_state(); save_risk()
        bot.loop.create_task(bot.send_message(CHANNEL_ID, "⛔ <b>STOP LOSS atingido</b>. Pausando entradas.", parse_mode="HTML"))
        return True
    return False

# ============== EXECUÇÃO SIMULADA ==============
def abrir_operacao(apos_num, alvos, base_total, multipliers):
    op = {
        "apos": apos_num,
        "alvos": alvos,
        "base": round(base_total,2),
        "mult": multipliers[:],
        "step": 0,                    # idx da tentativa atual
        "closed": False,
        "cycle_left": state["ciclo_max"],   # ciclos restantes (contagem regressiva)
        "carry_target": risk["prev_cycle_loss"]   # quanto recuperar do ciclo anterior
    }
    risk["open"] = op
    save_risk()
    log.info("OPEN %s", op)
    return op

def valor_tentativa(op):
    m = op["mult"][op["step"]]
    stake_total, por_num, lucro = plano_por_tentativa(op["base"], m)
    return stake_total, por_num, lucro

def avancar_depois_de_red():
    """após RED, avança step; se acabou o ciclo (sem bater), acumula prejuízo para o próximo ciclo."""
    op=risk.get("open")
    if not op or op["closed"]: return
    op["step"] += 1
    if op["step"] >= len(op["mult"]):
        # ciclo perdido → acumula prejuízo (soma de todas as tentativas desse ciclo)
        preju = 0.0
        for i,mi in enumerate(op["mult"]):
            s,_,_ = plano_por_tentativa(op["base"], mi)
            preju += s
        risk["prev_cycle_loss"] = round(risk["prev_cycle_loss"] + preju, 2)
        op["cycle_left"] -= 1
        op["closed"] = True
        risk["open"]=None
        save_risk()
        bot.loop.create_task(bot.send_message(
            CHANNEL_ID,
            f"❌ Ciclo perdido. Prejuízo acumulado: <b>R${risk['prev_cycle_loss']:.2f}</b>",
            parse_mode="HTML"
        ))
    else:
        save_risk()

def fechar_com_green():
    op=risk.get("open")
    if not op or op["closed"]: return
    # lucro líquido nessa tentativa
    stake_total, por_num, lucro = valor_tentativa(op)
    # perdas das anteriores
    perdas = 0.0
    for i in range(op["step"]):
        s,_,_ = plano_por_tentativa(op["base"], op["mult"][i])
        perdas += s
    pnl = round(lucro - perdas, 2)

    # atualiza PnL sessão e bankroll virtuais
    risk["session_pnl"] = round(risk["session_pnl"] + pnl, 2)
    risk["bankroll"]    = round(risk["bankroll"] + pnl, 2)

    # ao ganhar, zera carry do ciclo anterior
    risk["prev_cycle_loss"] = 0.0

    op["closed"]=True
    risk["open"]=None
    save_risk()

    bot.loop.create_task(bot.send_message(
        CHANNEL_ID,
        f"✅ GREEN (step {op['step']}) | PnL: <b>R${pnl:.2f}</b> | Sessão: <b>{risk['session_pnl']:.2f}</b>",
        parse_mode="HTML"
    ))
    check_stops_and_pause()

# ============== HANDLERS DO CANAL ==============
@dp.channel_post_handler(content_types=["text"])
async def on_channel_post(msg: types.Message):
    if msg.chat.id != CHANNEL_ID:
        return
    txt = (msg.text or "").strip()
    if not txt: return

    # aprendizado
    seq=extrai_sequencia(txt)
    if seq: atualiza_estat_num(seq)
    r = eh_resultado(txt)
    if r is not None:
        hist_long.append(r); hist_short.append(r)
        # fecha/avança operação aberta
        if risk.get("open") and not risk["open"]["closed"]:
            if r==1: fechar_com_green()
            else:    avancar_depois_de_red()
        return

    # só responde a ENTRADA CONFIRMADA
    if not eh_sinal(txt):
        return

    now=time.time()
    if now < state.get("cooldown_until", 0):
        return
    if not state["seguir_sinal"]:
        return
    if check_stops_and_pause():
        return

    # regra do sinal
    apos_num, alvos = extrai_regra_sinal(txt)
    # sanity: se não vierem 3 números, aborta
    if len(alvos) != 3:
        log.info("Sinal sem 3 alvos claros: %s", txt)
        return

    # decisão (CONFIRMAR?)
    short_wr=winrate(hist_short); long_wr=winrate(hist_long)
    vol=volatilidade(hist_short); mx_reds=streak_loss(hist_short)
    risco_num=risco_por_numeros(apos_num, alvos)
    conf=conf_final(short_wr, long_wr, vol, mx_reds, risco_num)

    if conf < state["limiar"]:
        return  # silencioso

    # preparar base_total com RECUPERAÇÃO do ciclo anterior (se houver)
    mults = state["multipliers"][:state["gales_max"]+1]
    if not mults: mults=[1.0]
    base_total = state["stake_base"]

    if risk["prev_cycle_loss"] > 0:
        # queremos recuperar prev_loss; lucro alvo mínimo = 0 (ou coloque +x se quiser)
        rec = required_base_for_recovery(risk["prev_cycle_loss"], 0.0, mults)
        if rec is not None:
            base_total = max(base_total, rec)

    # abre operação e mostra plano
    op = abrir_operacao(apos_num, alvos, base_total, mults)
    s0, per0, lucro0 = valor_tentativa(op)
    plano_txt = resumo_plano_text(mults, base_total)

    msg_txt = (
        "🟢 <b>CONFIRMAR</b>\n"
        f"🎯 Chance: <b>{conf*100:.1f}%</b>\n"
        f"🎯 Alvos: <b>{alvos[0]}-{alvos[1]}-{alvos[2]}</b>\n"
        f"💵 Tentativa 1 (total): <b>R${s0:.2f}</b> (≈ <i>{per0:.2f} por número</i>)\n"
        f"🧮 Plano: {plano_txt}\n"
        f"📈 Odds por número: <b>{risk['odds_total']:.2f}x</b>\n"
        f"💼 Sessão: <b>{risk['session_pnl']:.2f}</b> | Banca: <b>{risk['bankroll']:.2f}</b>"
    )
    await bot.send_message(CHANNEL_ID, msg_txt, parse_mode="HTML")

    state["cooldown_until"]=now+COOLDOWN_S
    save_state()

# ============== FASTAPI / WEBHOOK ==============
app = FastAPI()

@app.get("/healthz")
def healthz(): return {"ok": True}

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
    await dp.process_update(update)
    return {"ok": True}
