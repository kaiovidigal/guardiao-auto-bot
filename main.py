# main.py
import os
import sys
import time
import asyncio
import signal as os_signal
import logging
from dataclasses import dataclass
from typing import Optional, AsyncGenerator

# --- env / setup -------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bot")

# --- módulos internos (do patch que te enviei) -------------------------------
from safety import STATE, pause, resume, heartbeat, boot_watchdog
from fib import FibonacciStake
from decision import allowed_to_enter

# --- config adicionais -------------------------------------------------------
BANKROLL = float(os.getenv("BANKROLL", "0")) or None  # opcional (proteção 20%)
MAX_CONSECUTIVE_LOSS = int(os.getenv("MAX_CONSECUTIVE_LOSS", "3"))
SHOW_NOT_COUNTED = os.getenv("SHOW_NOT_COUNTED", "0") == "1"

# --- modelo de dado de sinal -------------------------------------------------
@dataclass
class Sinal:
    id: str
    payload: dict         # tudo que precisar pra executar a aposta
    prob: float           # 0.0–1.0 (ex.: 0.62 para 62%)
    sample: int           # tamanho da amostra (ex.: 2919)
    ts: float             # epoch seconds

# --- integrações (adapte estas duas funções ao seu projeto) ------------------
async def stream_sinais() -> AsyncGenerator[Sinal, None]:
    """
    STREAM DE SINAIS (ADAPTE AQUI)
    ------------------------------------------------------------------
    Substitua pelo seu conector real (ex.: Telethon/pyrogram lendo o
    canal do Telegram; fila Redis; websocket; etc.)

    ESTE MOCK gera nada até que você integre.
    Pra testar, você pode criar um gerador de exemplos.
    """
    while True:
        await asyncio.sleep(1.0)  # mantenha o loop vivo
        # yield Sinal(id="exemplo-1", payload={...}, prob=0.66, sample=300, ts=time.time())

async def executar_aposta(sinal: Sinal, stake: int) -> str:
    """
    EXECUÇÃO DA APOSTA (ADAPTE AQUI)
    ------------------------------------------------------------------
    Implemente a ação real: abrir navegador/req, clicar, confirmar, etc.
    Deve retornar "GREEN" ou "LOSS" (padrão do teu robô).
    """
    # TODO: implementar integração real.
    # Enquanto não implementa, apenas loga e "não contabiliza".
    log.info(f"(DRY-RUN) Apostaria stake={stake} no sinal {sinal.id} payload={sinal.payload}")
    # Retornar "not_counted" evita mexer no Fibonacci enquanto testa
    return "NOT_COUNTED"

# --- núcleo de decisão + execução -------------------------------------------
class BotCore:
    def __init__(self):
        self.fib = FibonacciStake()
        boot_watchdog()

    async def handle_sinal(self, s: Sinal) -> str:
        # heartbeat e atualização de estado
        heartbeat()
        STATE["last_signal_mono"] = time.monotonic()
        STATE["confidence"] = s.prob
        STATE["sample"] = s.sample

        # se estiver pausado, tentar retomar (respeita critérios do safety)
        if STATE.get("paused", False) and not resume():
            log.warning("Em pausa de proteção (aguardando critérios de retomada)")
            return "PAUSED"

        # gate de probabilidade + amostra
        ok, reason = allowed_to_enter(s.prob, s.sample)
        if not ok:
            msg = f"Sem entrada: {reason}"
            if SHOW_NOT_COUNTED:
                log.info(msg)
            else:
                log.debug(msg)
            return "SKIP"

        # valor da aposta via Fibonacci (bounded por CAP e / ou 20% da banca)
        stake = self.fib.next_bet(bankroll=BANKROLL)

        # executar aposta
        result = await executar_aposta(s, stake)

        # pós-processamento
        if result == "GREEN":
            STATE["loss_streak"] = 0
            self.fib.on_result("GREEN")
            log.info(f"GREEN | stake={stake} | prob={s.prob:.2%} | sample={s.sample}")
        elif result == "LOSS":
            STATE["loss_streak"] = STATE.get("loss_streak", 0) + 1
            self.fib.on_result("LOSS")
            log.warning(f"LOSS  | stake={stake} | prob={s.prob:.2%} | sample={s.sample} | streak={STATE['loss_streak']}")
            if STATE["loss_streak"] >= MAX_CONSECUTIVE_LOSS:
                pause()
                self.fib.on_pause()
                log.error("Pausa de proteção ativada (limite de perdas atingido)")
        else:
            # NOT_COUNTED ou qualquer outro rótulo não altera estado
            if SHOW_NOT_COUNTED:
                log.info(f"Resultado não contabilizado ({result}) | stake={stake}")
            else:
                log.debug(f"Resultado não contabilizado ({result})")

        return result

# --- laço principal ----------------------------------------------------------
async def main():
    log.info("Inicializando bot...")
    core = BotCore()

    # sinais de encerramento gracioso
    stop_event = asyncio.Event()

    def _graceful_shutdown(*_):
        log.warning("Encerrando (signal recebido)...")
        stop_event.set()

    for sig in (os_signal.SIGINT, os_signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _graceful_shutdown)
        except NotImplementedError:
            # Windows / ambientes que não suportam
            pass

    # loop de consumo de sinais
    try:
        async for s in stream_sinais():
            if stop_event.is_set():
                break
            try:
                await core.handle_sinal(s)
            except Exception as e:
                log.exception(f"Erro ao processar sinal {getattr(s, 'id', '?')}: {e}")
    except Exception as e:
        log.exception(f"Falha no stream de sinais: {e}")

    log.info("Bot finalizado.")

# --- entrypoint --------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
        log.warning("Interrompido pelo usuário.")
        sys.exit(130)