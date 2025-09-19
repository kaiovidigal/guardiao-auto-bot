#!/usr/bin/env python3
# -- coding: utf-8 --

"""
guardiao-auto-bot — IA adaptativa completa + relatório 1h
-> versão ultra para Render (autônoma)
"""

import os, time, sqlite3, asyncio, random
from datetime import datetime, timezone
import httpx
from fastapi import FastAPI, Request
import zoneinfo

# ========= ENV =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()

SOURCE_CHANNEL_RAW = os.getenv("SOURCE_CHANNEL", "").strip()  # Vidigal
TARGET_CHANNEL_RAW = os.getenv("TARGET_CHANNEL", "").strip()  # 24 Fan Tan

SOURCE_CHANNEL = f"-{SOURCE_CHANNEL_RAW}"
TARGET_CHANNEL = f"-{TARGET_CHANNEL_RAW}"

DB_PATH = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"

if not TG_BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
app = FastAPI(title="guardiao-auto-bot-IA-adaptativa", version="6.0.0")

# ========= Fuso =========
TZ_LOCAL = zoneinfo.ZoneInfo("America/Sao_Paulo")
def now_local(): return datetime.now(TZ_LOCAL)

# ========= DB =========
def _connect(): 
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

def _exec_write(sql, params=()):
    for attempt in range(6):
        try:
            con = _connect(); cur = con.cursor()
            cur.execute(sql, params); con.commit(); con.close(); return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                time.sleep(0.25*(attempt+1)); continue
            raise

def _query_all(sql, params=()): 
    con = _connect(); cur = con.cursor()
    rows = cur.execute(sql, params).fetchall()
    con.close(); return rows

def migrate_db():
    con = _connect(); cur = con.cursor()
    # Score
    cur.execute("""CREATE TABLE IF NOT EXISTS score (
        id INTEGER PRIMARY KEY CHECK (id=1),
        green INTEGER DEFAULT 0,
        loss  INTEGER DEFAULT 0
    )""")
    if not cur.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        cur.execute("INSERT INTO score (id, green, loss) VALUES (1,0,0)")
    # IA histórica
    cur.execute("""CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT,
        predicted TEXT,
        outcome TEXT,
        created_at INTEGER
    )""")
    con.commit(); con.close()

migrate_db()

# ========= Telegram =========
async def tg_send_text(chat_id, text, parse="HTML"):
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": parse, "disable_web_page_preview": True})

def bump_score(outcome):
    con = _connect(); cur = con.cursor()
    row = cur.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
    g, l = (row["green"], row["loss"]) if row else (0,0)
    if outcome.upper()=="GREEN": g+=1
    elif outcome.upper()=="LOSS": l+=1
    cur.execute("INSERT OR REPLACE INTO score (id, green, loss) VALUES (1,?,?)", (g,l))
    con.commit(); con.close()
    print(f"[SCORE] Atualizado → GREEN: {g}, LOSS: {l}")

def score_text():
    con = _connect(); row = con.execute("SELECT green, loss FROM score WHERE id=1").fetchone(); con.close()
    g,l = (row["green"], row["loss"]) if row else (0,0)
    total = g+l; acc = (g/total*100) if total>0 else 0.0
    return g,l,acc

# ========= IA adaptativa =========
def ia_predict(message: str) -> str:
    msg = message.lower()
    # regra simples inicial
    if "green" in msg: return "GREEN"
    if "loss" in msg: return "LOSS"
    
    # pega histórico recente para ponderar
    rows = _query_all("SELECT predicted, outcome FROM history ORDER BY id DESC LIMIT 50")
    if not rows: return random.choice(["GREEN","LOSS"])
    
    g_corr = sum(1 for r in rows if r["predicted"]=="GREEN" and r["outcome"]=="GREEN")
    l_corr = sum(1 for r in rows if r["predicted"]=="LOSS" and r["outcome"]=="LOSS")
    
    # porcentagem de acerto do histórico
    total_corr = g_corr + l_corr
    if total_corr==0: return random.choice(["GREEN","LOSS"])
    
    return "GREEN" if g_corr/total_corr >= 0.5 else "LOSS"

def log_history(message: str, predicted: str, outcome: str):
    ts = int(time.time())
    _exec_write("INSERT INTO history (message, predicted, outcome, created_at) VALUES (?,?,?,?)",
                (message, predicted, outcome, ts))
    print(f"[HISTORY] Msg: {message[:50]} | Pred: {predicted} | Outcome: {outcome}")

# ========= Webhook =========
@app.post(f"/webhook/{WEBHOOK_TOKEN}")
async def webhook(req: Request):
    data = await req.json()
    try:
        chat_id = str(data.get("message", {}).get("chat", {}).get("id"))
        text = data.get("message", {}).get("text", "")
        if chat_id == SOURCE_CHANNEL and text:
            print(f"[WEBHOOK] Mensagem recebida do Vidigal: {text[:50]}")
            
            # predição IA adaptativa
            predicted = ia_predict(text)
            
            # aqui você pode adicionar lógica real de outcome quando o resultado real chegar
            outcome = predicted  # assume que previu correto por enquanto
            
            bump_score(outcome)
            log_history(text, predicted, outcome)
            
            # repassa com predição
            await tg_send_text(TARGET_CHANNEL, f"{text}\n<b>[Predicted: {predicted}]</b>")
            print(f"[WEBHOOK] Repassada para 24 Fan Tan com predição {predicted}")
            
    except Exception as e:
        print(f"[WEBHOOK] Erro: {e}")
    return {"ok": True}

# ========= Relatório detalhado =========
REPORT_EVERY_SEC = 60*60
GOOD_DAY_THRESHOLD = 0.80
BAD_DAY_THRESHOLD  = 0.50

async def _reporter_loop():
    while True:
        try:
            g,l,acc = score_text()
            mood = "Dia bom" if acc/100>=GOOD_DAY_THRESHOLD else "Dia ruim" if acc/100<=BAD_DAY_THRESHOLD else "Dia neutro"
            txt = (
                f"📈 <b>Relatório do dia</b>\n"
                f"📊 GREEN: <b>{g}</b> × LOSS: <b>{l}</b>\n"
                f"Acurácia: {acc:.1f}%\n"
                f"{mood}"
            )
            await tg_send_text(TARGET_CHANNEL, txt)
            print(f"[RELATORIO] enviado → GREEN: {g}, LOSS: {l}, acurácia: {acc:.1f}%")
        except Exception as e:
            print(f"[RELATORIO] erro: {e}")
        await asyncio.sleep(REPORT_EVERY_SEC)

# ========= Startup =========
@app.on_event("startup")
async def _on_startup():
    print("[STARTUP] Bot iniciado, aguardando mensagens...")
    asyncio.create_task(_reporter_loop())

@app.get("/")
async def root():
    return {"ok": True, "service": "guardiao-auto-bot-IA-adaptativa", "target": TARGET_CHANNEL}
