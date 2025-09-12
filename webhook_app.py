# -*- coding: utf-8 -*-
# Guardião Híbrido (Réplica + IA) — versão gatilho imediato

import os, re, time, sqlite3, asyncio
from typing import List, Optional, Tuple, Dict
from collections import Counter
import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone

# ====== ENV ======
DB_PATH = os.getenv("DB_PATH", "/var/data/main.sqlite").strip()
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "meusegredo123").strip()
PUBLIC_CHANNEL = os.getenv("PUBLIC_CHANNEL", "").strip()  # se setado, só replica desse chat_id
REPL_CHANNEL   = os.getenv("REPL_CHANNEL", "").strip()
SELF_LABEL_IA  = os.getenv("SELF_LABEL_IA", "Tiro seco por IA").strip()
INTEL_ANALYZE_INTERVAL = float(os.getenv("INTEL_ANALYZE_INTERVAL", "2"))

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

if not TG_BOT_TOKEN:
    print("⚠️ Defina TG_BOT_TOKEN.")
if not REPL_CHANNEL:
    print("⚠️ Defina REPL_CHANNEL.")

# ====== DB helpers (tolerante a ausência de tabelas) ======
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    return con

def _safe_exec(sql: str, params: tuple = ()):
    try:
        con = _connect()
        con.execute(sql, params)
        con.commit()
    except Exception as e:
        # não quebra o fluxo de réplica
        print("[DB] ignore:", e)
    finally:
        try:
            con.close()
        except: pass

def _safe_query_one(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    try:
        con = _connect()
        r = con.execute(sql, params).fetchone()
        con.close()
        return r
    except Exception as e:
        print("[DB] ignore:", e)
        return None

def _safe_query_all(sql: str, params: tuple = ()) -> list:
    try:
        con = _connect()
        r = con.execute(sql, params).fetchall()
        con.close()
        return r
    except Exception as e:
        print("[DB] ignore:", e)
        return []

def append_timeline(n: int):
    _safe_exec("INSERT INTO timeline (created_at, number) VALUES (?,?)", (int(time.time()), int(n)))

# ====== Telegram ======
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    if not TG_BOT_TOKEN or not chat_id:
        return
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": parse, "disable_web_page_preview": True})

# ====== IA simples (tolerante) ======
WINDOW = 400
W4, W3, W2, W1 = 0.38, 0.30, 0.20, 0.12
ALPHA = 1.05
MIN_SAMPLES = 1000
GAP_MIN = 0.08

def _recent_tail(window: int = WINDOW) -> List[int]:
    rows = _safe_query_all("SELECT number FROM timeline ORDER BY id DESC LIMIT ?", (window,))
    return [r["number"] for r in rows][::-1]

def _prob(ctx: List[int], cand: int) -> float:
    n = len(ctx) + 1
    if n < 2 or n > 5: return 0.0
    ctx_key = ",".join(map(str, ctx))
    row_tot = _safe_query_one("SELECT SUM(weight) AS w FROM ngram_stats WHERE n=? AND ctx=?", (n, ctx_key))
    tot = (row_tot["w"] or 0.0) if row_tot else 0.0
    if tot <= 0: return 0.0
    row_c = _safe_query_one("SELECT weight FROM ngram_stats WHERE n=? AND ctx=? AND next=?",
                            (n, ctx_key, cand))
    w = (row_c["weight"] or 0.0) if row_c else 0.0
    return w / tot

def _score(tail: List[int], cand: int) -> float:
    if not tail: return 0.0
    ctx4 = tail[-4:] if len(tail)>=4 else []
    ctx3 = tail[-3:] if len(tail)>=3 else []
    ctx2 = tail[-2:] if len(tail)>=2 else []
    ctx1 = tail[-1:] if len(tail)>=1 else []
    parts=[]
    if len(ctx4)==4: parts.append((W4, _prob(ctx4[:-1], cand)))
    if len(ctx3)==3: parts.append((W3, _prob(ctx3[:-1], cand)))
    if len(ctx2)==2: parts.append((W2, _prob(ctx2[:-1], cand)))
    if len(ctx1)==1: parts.append((W1, _prob(ctx1[:-1], cand)))
    return sum(w*p for w,p in parts)

def suggest_number() -> Tuple[Optional[int], float, int, Dict[int, float]]:
    base=[1,2,3,4]
    tail = _recent_tail(WINDOW)
    scores={}
    for c in base:
        sc = (1.0/len(base)) * ((_score(tail, c) or 1e-6) ** ALPHA)
        scores[c]=sc
    tot = sum(scores.values()) or 1e-9
    post = {k:v/tot for k,v in scores.items()}
    a = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    if not a: return None,0.0,len(tail),post
    gap = a[0][1] - (a[1][1] if len(a)>1 else 0.0)
    number = a[0][0] if gap >= GAP_MIN else None
    row_s = _safe_query_one("SELECT SUM(weight) AS s FROM ngram_stats")
    samples = int((row_s["s"] or 0) if row_s else 0)
    if samples < MIN_SAMPLES: return None, 0.0, samples, post
    return number, post.get(number,0.0), samples, post

# ====== FastAPI ======
app = FastAPI(title="Guardião Híbrido", version="1.2.0")

class Update(BaseModel):
    update_id: int
    channel_post: Optional[dict] = None
    message: Optional[dict] = None
    edited_channel_post: Optional[dict] = None
    edited_message: Optional[dict] = None

@app.get("/")
async def root():
    return {"ok": True, "detail": "Webhook ativo"}

def _authorized_path(token_path: str) -> bool:
    # permite WEBHOOK_TOKEN ou o próprio TG_BOT_TOKEN (modo simples)
    return token_path == WEBHOOK_TOKEN or token_path == TG_BOT_TOKEN

@app.post("/webhook/{token_path}")
async def webhook(token_path: str, request: Request):
    if not _authorized_path(token_path):
        raise HTTPException(status_code=403, detail="Forbidden")
    data = await request.json()
    upd = Update(**data)
    msg = upd.channel_post or upd.message or upd.edited_channel_post or upd.edited_message
    if not msg:
        return {"ok": True, "skip": "no message"}

    chat = msg.get("chat") or {}
    chat_id = str(chat.get("id", ""))

    # se PUBLIC_CHANNEL estiver setado, só aceita se bater
    if PUBLIC_CHANNEL and chat_id != str(PUBLIC_CHANNEL):
        return {"ok": True, "skip": "different chat"}

    text = (msg.get("text") or msg.get("caption") or "").strip()
    if not text:
        return {"ok": True, "skip": "no text"}

    # ===== Réplica imediata (SEM TRAVA) =====
    # Sempre replica o texto bruto (se quiser filtrar, comente a linha abaixo).
    await tg_send_text(REPL_CHANNEL, f"📡 Cópia direta:\n{text}")

    # ===== Atualizações no banco (se houver GREEN/RED) =====
    upper = text.upper()
    if "GREEN" in upper or "RED" in upper:
        nums = re.findall(r"[1-4]", text)
        if nums:
            append_timeline(int(nums[0]))

    # ===== Gatilho IA quando detectar "ENTRADA CONFIRMADA" =====
    if "ENTRADA CONFIRMADA" in upper:
        num, conf, samples, _ = suggest_number()
        if num:
            confp = f"{conf*100:.2f}%"
            txt = (f"🤖 <b>{SELF_LABEL_IA} [FIRE]</b>\n"
                   f"🎯 Número seco (G0): <b>{num}</b>\n"
                   f"📈 Conf: <b>{confp}</b> | Amostra≈<b>{samples}</b>")
            await tg_send_text(REPL_CHANNEL, txt)

    return {"ok": True, "replicated": True}

# opcional: loop de IA contínuo (desligado para evitar spam). Ative se quiser.
# @app.on_event("startup")
# async def _boot():
#     async def _loop():
#         while True:
#             try:
#                 num, conf, samples, _ = suggest_number()
#                 # exemplo: só envia se muito confiante (você ajusta as regras)
#                 # if num and conf >= 0.75: ...
#             except Exception as e:
#                 print("[IA] erro:", e)
#             await asyncio.sleep(max(0.2, INTEL_ANALYZE_INTERVAL))
#     asyncio.create_task(_loop())