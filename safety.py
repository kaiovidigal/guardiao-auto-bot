# safety.py
import time, json, os, threading

CFG = {
    "resume_after_min": int(os.getenv("RESUME_AFTER_MINUTES", 5)),
    "resume_conf": float(os.getenv("RESUME_REQUIRE_CONFIDENCE", 0.55)),
    "resume_min_sample": int(os.getenv("RESUME_MIN_SAMPLE", 50)),
    "max_loss": int(os.getenv("MAX_CONSECUTIVE_LOSS", 3)),
    "pause_lock": os.getenv("PAUSE_LOCK_PATH", "./pause.lock"),
    "state_path": os.getenv("STATE_PATH", "./state.json"),
}

STATE = {
    "paused": False,
    "loss_streak": 0,
    "confidence": 0.0,
    "sample": 0,
    "last_signal_mono": time.monotonic(),
    "heartbeat_mono": time.monotonic()
}

def _save():
    try:
        os.makedirs(os.path.dirname(CFG["state_path"]), exist_ok=True)
        with open(CFG["state_path"], "w") as f:
            json.dump(STATE, f)
    except Exception:
        pass

def _load():
    try:
        with open(CFG["state_path"]) as f:
            STATE.update(json.load(f))
    except Exception:
        pass

def pause():
    STATE["paused"] = True
    os.makedirs(os.path.dirname(CFG["pause_lock"]), exist_ok=True)
    open(CFG["pause_lock"], "w").close()
    _save()

def resume(force=False):
    if os.path.exists(CFG["pause_lock"]):
        try: os.remove(CFG["pause_lock"])
        except Exception: pass
    if force or (
        STATE["confidence"] >= CFG["resume_conf"] and
        STATE["sample"] >= CFG["resume_min_sample"]
    ):
        STATE["paused"] = False
        STATE["loss_streak"] = 0
        _save()
        return True
    return False

def heartbeat():
    STATE["heartbeat_mono"] = time.monotonic()

def watchdog():
    _load()
    while True:
        now = time.monotonic()
        if STATE["paused"]:
            cooldown = (now - STATE["last_signal_mono"]) >= CFG["resume_after_min"] * 60
            criteria = (STATE["confidence"] >= CFG["resume_conf"] and
                        STATE["sample"] >= CFG["resume_min_sample"])
            stale = (now - STATE["heartbeat_mono"]) > 180
            if cooldown and (criteria or stale):
                resume(force=stale)
        else:
            if STATE["loss_streak"] >= CFG["max_loss"]:
                pause()
        time.sleep(5)

def boot_watchdog():
    threading.Thread(target=watchdog, daemon=True).start()