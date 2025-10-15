# server.py
import os
import json
import time
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from webhook_queue import get_queue, make_sinal_from_dict

app = FastAPI(title="Bot Webhook")

# --- CONFIG ------------------------------------------------------------
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
STATE_PATH = os.getenv("STATE_PATH", "/var/app/data/state.json")
PAUSE_LOCK = os.getenv("PAUSE_LOCK_PATH", "/var/app/data/pause.lock")

# -----------------------------------------------------------------------
@app.get("/health")
async def health():
    """Teste rápido para ver se o servidor está online"""
    return {"ok": True, "time": time.strftime("%Y-%m-%d %H:%M:%S")}

# -----------------------------------------------------------------------
@app.post("/webhook/sinal")
async def webhook_sinal(
    request: Request,
    x_webhook_secret: str = Header(default="")
):
    """Recebe sinais externos e coloca na fila"""
    if not WEBHOOK_SECRET or x_webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    items = data if isinstance(data, list) else [data]
    q = get_queue()
    accepted = 0

    for item in items:
        try:
            s = make_sinal_from_dict(item)
            await q.put(s)
            accepted += 1
        except Exception:
            pass

    return JSONResponse({"status": "ok", "accepted": accepted})

# -----------------------------------------------------------------------
@app.post("/admin/force-resume")
async def force_resume(x_webhook_secret: str = Header(default="")):
    """Força retomada manual mesmo que critérios não estejam batendo"""
    if not WEBHOOK_SECRET or x_webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # remove lock
        if os.path.exists(PAUSE_LOCK):
            os.remove(PAUSE_LOCK)
        # zera estado
        st = {"paused": False, "loss_streak": 0}
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH) as f:
                cur = json.load(f)
            cur.update(st)
            st = cur
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True