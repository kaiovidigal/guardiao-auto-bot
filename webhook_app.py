import os
import sqlite3
import asyncio
from fastapi import FastAPI, Request
from datetime import datetime
import httpx

# ===================== CONFIGURAÇÃO =====================
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "1003081474331").strip()  # Sinal 24 Fan Tan
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "1002810508717").strip()  # Vidigal
DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"

# ===================== BANCO =====================
def _connect():
    return sqlite3.connect(DB_PATH)

def migrate_db():
    con = _connect()
    cur = con.cursor()
    # Garante que as colunas predicted e outcome existam
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pending (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER,
            message TEXT,
            predicted TEXT DEFAULT '',
            outcome TEXT DEFAULT ''
        )
    """)
    con.commit()
    con.close()

migrate_db()

# ===================== BOT =====================
app = FastAPI()

async def send_telegram_message(chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, data={"chat_id": chat_id, "text": text})

def save_message(message_text: str):
    con = _connect()
    cur = con.cursor()
    cur.execute("INSERT INTO pending (created_at, message) VALUES (?, ?)",
                (int(datetime.now().timestamp()), message_text))
    con.commit()
    con.close()

def generate_signal(message_text: str) -> str:
    """
    Simples IA placeholder: pode ser substituída pela sua lógica real.
    Por enquanto retorna 'GREEN' ou 'LOSS' aleatoriamente.
    """
    import random
    return random.choice(["GREEN", "LOSS"])

@app.post(f"/webhook/{WEBHOOK_TOKEN}")
async def webhook(request: Request):
    data = await request.json()
    # ====== Recebe mensagem ======
    message_text = data.get("message", {}).get("text", "")
    chat_id = str(data.get("message", {}).get("chat", {}).get("id", ""))
    print(f"[DEBUG] Recebido: {message_text} do chat {chat_id}")

    # ====== Filtra só o canal correto ======
    if chat_id != SOURCE_CHANNEL:
        return {"status": "ignored"}

    # ====== Salva no banco ======
    save_message(message_text)

    # ====== Gera e envia sinal ======
    signal = generate_signal(message_text)
    await send_telegram_message(TARGET_CHANNEL, f"Sinal automático: {signal}")

    return {"status": "ok"}

# ===================== RELATÓRIO =====================
async def send_daily_report():
    while True:
        await asyncio.sleep(24*60*60)  # 24 horas
        con = _connect()
        cur = con.cursor()
        cur.execute("SELECT predicted, outcome FROM pending")
        rows = cur.fetchall()
        green = sum(1 for r in rows if r[1] == "GREEN")
        loss = sum(1 for r in rows if r[1] == "LOSS")
        acuracia = green / max(1, (green + loss)) * 100
        report = f"📈 Relatório do dia\n📊 GREEN: {green} × LOSS: {loss}\nAcurácia: {acuracia:.1f}%"
        await send_telegram_message(TARGET_CHANNEL, report)
        con.close()

@app.on_event("startup")
async def startup_event():
    print("[STARTUP] Bot iniciado, aguardando mensagens...")
    asyncio.create_task(send_daily_report())
