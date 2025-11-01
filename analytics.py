import os
import json
from datetime import datetime
from openai import OpenAI
import httpx

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CANAL_DESTINO_ID = os.getenv("CANAL_DESTINO_ID", "-1002796105884")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
SEND_MESSAGE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

async def enviar_telegram(mensagem: str):
    async with httpx.AsyncClient() as client_http:
        await client_http.post(SEND_MESSAGE_URL, json={"chat_id": CANAL_DESTINO_ID, "text": mensagem, "parse_mode": "Markdown"})

def carregar_historico():
    eventos = []
    if not os.path.exists("historico.json"):
        return eventos
    with open("historico.json") as f:
        for linha in f:
            try:
                eventos.append(json.loads(linha))
            except:
                pass
    return eventos

def gerar_resumo(eventos):
    analise = {}
    for e in eventos:
        hora = e["hora"][11:13]
        if e["tipo"] == "resultado" and e.get("resultado") == "GREEN":
            analise[hora] = analise.get(hora, 0) + 1
    return json.dumps(analise, indent=2)

async def analisar_com_chatgpt():
    eventos = carregar_historico()
    if not eventos:
        await enviar_telegram("⚠️ Nenhum dado registrado ainda.")
        return
    resumo = gerar_resumo(eventos)
    prompt = f"""
    Aqui estão os dados de acertos por hora:
    {resumo}

    Gere uma análise:
    - Horários com mais greens
    - Padrões percebidos
    - Melhores janelas para sinais futuros
    """
    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um analista de performance de sinais de cassino."},
            {"role": "user", "content": prompt},
        ]
    )
    analise = resposta.choices[0].message.content
    await enviar_telegram("📊 *Análise automática:*\n\n" + analise)

# Exemplo de execução manual (pode ser agendado via cron)
# asyncio.run(analisar_com_chatgpt())
