import os
import sqlite3
import asyncio
from fastapi import FastAPI, Request
from datetime import datetime
import httpx
import random

# ===================== CONFIGURAÇÃO =====================
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "1003081474331").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "1002810508717").strip()
DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"

# ===================== BANCO =====================
def _connect():
    return sqlite3.connect(DB_PATH)

def migrate_db():
    con = _connect()
    cur = con.cursor()
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

def save_message(message_text: str, predicted: str = "", outcome: str = ""):
    con = _connect()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO pending (created_at, message, predicted, outcome) VALUES (?, ?, ?, ?)",
        (int(datetime.now().timestamp()), message_text, predicted, outcome)
    )
    con.commit()
    con.close()

def calculate_dynamic_chance():
    """Calcula chance de GREEN com base nos últimos 50 resultados"""
    con = _connect()
    cur = con.cursor()
    cur.execute("SELECT outcome FROM pending ORDER BY id DESC LIMIT 50")
    rows = cur.fetchall()
    con.close()
    if not rows:
        return 70  # chance inicial
    green_count = sum(1 for r in rows if r[0] == "GREEN")
    total = len(rows)
    # Ajuste dinâmico: aumenta ou diminui chance conforme histórico recente
    base_chance = 50
    adjustment = int((green_count - (total - green_count)) * 0.5)
    chance = min(max(30, base_chance + adjustment), 90)
    return chance

def generate_signal(message_text: str = "") -> tuple[str, str]:
    green_chance = calculate_dynamic_chance()
    outcome = "GREEN" if random.randint(1, 100) <= green_chance else "LOSS"
    emoji = "✅" if outcome == "GREEN" else "❌"
    return f"{emoji} {outcome}", outcome

@app.post(f"/webhook/{WEBHOOK_TOKEN}")
async def webhook(request: Request):
    data = await request.json()
    message_text = data.get("message", {}).get("text", "")
    chat_id = str(data.get("message", {}).get("chat", {}).get("id", ""))
    print(f"[DEBUG] Recebido: {message_text} do chat {chat_id}")

    if chat_id != SOURCE_CHANNEL:
        return {"status": "ignored"}

    # Gera sinal ultra evolutivo
    signal_text, signal_outcome = generate_signal(message_text)
    save_message(message_text, predicted=signal_outcome)
    await send_telegram_message(TARGET_CHANNEL, f"🚀 Sinal automático: {signal_text}")

    return {"status": "ok", "signal": signal_text}

# ===================== RELATÓRIO DIÁRIO =====================
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

# ===================== SINAL INICIAL =====================
async def send_initial_signal():
    await asyncio.sleep(2)  # espera o bot subir
    signal_text, signal_outcome = generate_signal("Sinal inicial de teste")
    save_message("Sinal inicial de teste", predicted=signal_outcome)
    await send_telegram_message(TARGET_CHANNEL, f"🚀 Sinal inicial: {signal_text}")
    print(f"[DEBUG] Sinal inicial enviado: {signal_text}")

@app.on_event("startup")
async def startup_event():
    print("[STARTUP] Bot iniciado, aguardando mensagens...")
    # Inicia tarefas contínuas
    asyncio.create_task(send_daily_report())
    asyncio.create_task(send_initial_signal())
