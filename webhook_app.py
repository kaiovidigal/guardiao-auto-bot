import os
import sqlite3
import asyncio
from fastapi import FastAPI, Request
from datetime import datetime
import httpx
import random

TG_BOT_TOKEN=os.getenv("TG_BOT_TOKEN","").strip()
WEBHOOK_TOKEN=os.getenv("WEBHOOK_TOKEN","").strip()
TARGET_CHANNEL=os.getenv("TARGET_CHANNEL","-1003081474331").strip()
SOURCE_CHANNEL=os.getenv("SOURCE_CHANNEL","-1002810508717").strip()
DB_PATH=os.getenv("DB_PATH","./data.db").strip() or "./data.db"

def _connect(): return sqlite3.connect(DB_PATH)
def migrate_db():
    con=_connect()
    cur=con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS pending (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER, message TEXT DEFAULT '', predicted TEXT DEFAULT '', outcome TEXT DEFAULT '', stage TEXT NOT NULL DEFAULT 'initial', suggested TEXT DEFAULT '')""")
    con.commit()
    con.close()
migrate_db()

app=FastAPI()
async def send_telegram_message(chat_id:str,text:str):
    url=f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client: await client.post(url,data={"chat_id":chat_id,"text":text})
def save_message(message_text:str,predicted="",outcome="",stage="initial",suggested=""):
    con=_connect()
    cur=con.cursor()
    cur.execute("INSERT INTO pending (created_at,message,predicted,outcome,stage,suggested) VALUES (?,?,?,?,?,?)",(int(datetime.now().timestamp()),message_text,predicted,outcome,stage,suggested))
    con.commit()
    con.close()

def generate_real_signal(message_text:str)->str:
    keywords_loss=["perdeu","erro","cancel"]
    if any(word in message_text.lower() for word in keywords_loss): return "❌ LOSS"
    return "✅ GREEN"

@app.post(f"/webhook/{WEBHOOK_TOKEN}")
async def webhook(request:Request):
    data=await request.json()
    message_text=data.get("message",{}).get("text","")
    chat_id=str(data.get("message",{}).get("chat",{}).get("id",""))
    print(f"[DEBUG] Recebido: {message_text} do chat {chat_id}")
    if chat_id!=SOURCE_CHANNEL: return {"status":"ignored"}
    save_message(message_text,stage="received")
    signal=generate_real_signal(message_text)
    save_message(f"Sinal gerado: {signal}",predicted=signal,outcome="",stage="sent")
    await send_telegram_message(TARGET_CHANNEL,f"🎯 Sinal oficial: {signal}")
    return {"status":"ok"}

async def send_initial_signal():
    await asyncio.sleep(5)
    signal_outcome=generate_real_signal("Sinal inicial")
    save_message("Sinal inicial oficial",predicted=signal_outcome,outcome=signal_outcome,stage="initial")
    await send_telegram_message(TARGET_CHANNEL,f"🚀 Bot oficial online! Primeiro sinal: {signal_outcome}")

async def send_daily_report():
    while True:
        await asyncio.sleep(24*60*60)
        con=_connect()
        cur=con.cursor()
        cur.execute("SELECT predicted,outcome FROM pending")
        rows=cur.fetchall()
        green=sum(1 for r in rows if "GREEN" in r[1])
        loss=sum(1 for r in rows if "LOSS" in r[1])
        acuracia=green/max(1,(green+loss))*100
        report=f"📈 Relatório do dia\n📊 ✅ GREEN: {green} × ❌ LOSS: {loss}\n🎯 Acurácia: {acuracia:.1f}%"
        await send_telegram_message(TARGET_CHANNEL,report)
        con.close()

@app.on_event("startup")
async def startup_event():
    print("[STARTUP] Bot oficial iniciado, aguardando mensagens...")
    asyncio.create_task(send_initial_signal())
    asyncio.create_task(send_daily_report())

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=int(os.getenv("PORT",10000)))
