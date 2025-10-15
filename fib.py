# fib.py
import os
from decimal import Decimal, ROUND_HALF_UP

def _d(v):  # arredondamento estilo "banco"
    return Decimal(v).quantize(Decimal("1."), rounding=ROUND_HALF_UP)

class FibonacciStake:
    def __init__(self):
        self.enabled = os.getenv("FIB_ENABLED", "1") == "1"
        self.base = Decimal(os.getenv("FIB_BASE_STAKE", "150"))
        self.max_step = int(os.getenv("FIB_MAX_STEP", "6"))
        self.cap = Decimal(os.getenv("FIB_CAP", "2000"))
        self.reset_on_win = os.getenv("FIB_RESET_ON_WIN", "1") == "1"
        self.reset_on_pause = os.getenv("FIB_RESET_ON_PAUSE", "1") == "1"
        self.seq = [1,1,2,3,5,8,13,21,34,55]  # pode ampliar se quiser
        if self.max_step + 1 > len(self.seq):
            raise ValueError("Amplie self.seq ou reduza FIB_MAX_STEP")
        self.step = 0

    def next_bet(self, bankroll=None):
        if not self.enabled:
            return int(self.base)
        mult = self.seq[min(self.step, self.max_step)]
        stake = self.base * mult
        stake = min(stake, self.cap)
        if bankroll is not None:
            soft_cap = max(Decimal(1), _d(Decimal(str(bankroll)) * Decimal("0.20")))
            stake = min(stake, soft_cap)
        return int(_d(stake))

    def on_result(self, result: str):
        if str(result).upper().startswith("GREEN"):
            self.step = 0 if self.reset_on_win else max(0, self.step - 1)
        else:
            self.step = min(self.max_step, self.step + 1)

    def on_pause(self):
        if self.reset_on_pause:
            self.step = 0