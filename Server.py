# server.py (adicione no final)
from fastapi import HTTPException
import json, os

STATE_PATH = os.getenv("STATE_PATH", "/var/app/data/state.json")
PAUSE_LOCK = os.getenv("PAUSE_LOCK_PATH", "/var/app/data/pause.lock")

@app.post("/admin/force-resume")
async def force_resume(x_webhook_secret: str = Header(default="")):
    if not WEBHOOK_SECRET or x_webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # remove lock
    try:
        if os.path.exists(PAUSE_LOCK):
            os.remove(PAUSE_LOCK)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"lock: {e}"}, status_code=500)

    # zera pausa/streak
    try:
        st = {"paused": False, "loss_streak": 0}
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH) as f:
                cur = json.load(f)
            cur.update(st)
            st = cur
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(st, f)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"state: {e}"}, status_code=500)

    return {"ok": True}
