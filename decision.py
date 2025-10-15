# decision.py
import os
MIN_PROB = float(os.getenv("MIN_PROBABILITY", "0.50"))
MIN_SAMPLE = int(os.getenv("MIN_SAMPLE", "50"))

def allowed_to_enter(probability: float, sample_size: int):
    if sample_size < MIN_SAMPLE:
        return False, f"Sample insuficiente ({sample_size}<{MIN_SAMPLE})"
    if probability < MIN_PROB:
        return False, f"Probabilidade {probability:.2%}<{MIN_PROB:.0%}"
    return True, "OK"