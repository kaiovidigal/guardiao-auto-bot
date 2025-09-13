# ==== CONFIG BÁSICA PARA ALERTA ====
ALERT_CHANNEL = os.getenv("ALERT_CHANNEL", "").strip()  # ex: -1002796105884
ALMOST_LIMIAR = int(os.getenv("ALMOST_LIMIAR", "7"))    # dispara em >=7 sem vir
ALMOST_SCAN_EVERY = 300                                  # 5 min

# ==== HELPERS DE TELEGRAM (reusa seu tg_send_text/tg_broadcast se já tiver) ====
async def tg_alert(text: str):
    if not ALERT_CHANNEL:
        print("[ALERT] ALERT_CHANNEL vazio — defina no Render.")
        return
    await tg_send_text(ALERT_CHANNEL, text, "HTML")

# ==== MÉTRICAS DE AUSÊNCIA (usa sua tabela timeline) ====
def get_absences() -> dict[int, int]:
    """
    Retorna {numero: ausencias} onde 'ausencias' é quantas jogadas desde a última
    vez que o número apareceu. Se a timeline estiver vazia, volta 0 para todos.
    """
    rows = query_all("SELECT number FROM timeline ORDER BY id DESC LIMIT 5000")
    recent = [r["number"] for r in rows]
    if not recent:
        return {1: 0, 2: 0, 3: 0, 4: 0}

    # procura última posição de cada número
    absences = {}
    for n in (1, 2, 3, 4):
        try:
            last_idx = recent.index(n)   # primeira ocorrência na ordem DESC
            # como a lista está DESC, 'last_idx' é quantos eventos para trás
            absences[n] = last_idx
        except ValueError:
            # não apareceu em nenhuma das últimas N amostras
            absences[n] = len(recent)
    return absences

_last_bullet_sent: dict[int, int] = {1:-1,2:-1,3:-1,4:-1}  # evita repetir

async def scan_and_alert_once():
    """
    Scaneia ausências e dispara:
      - Alerta quando algum número >= ALMOST_LIMIAR (ex: 7,8,9).
      - Boletim 'quase 10' se algum chegou em 8 ou 9 (sem repetir).
    """
    absences = get_absences()
    if not absences: 
        return

    # 1) ALERTA principal: qualquer número >= limiar
    hot = [(n, c) for n, c in absences.items() if c >= ALMOST_LIMIAR]
    if hot:
        hot.sort(key=lambda x: x[1], reverse=True)
        linhas = [f"⚠️ Número <b>{n}</b> está há <b>{c}</b> sem vir." for n, c in hot]
        rec = max(absences.values())
        msg = "⏰ <b>Alerta de ausência</b>\n" + "\n".join(linhas) + f"\n\n📈 Recorde atual observado: <b>{rec}</b>"
        await tg_alert(msg)

    # 2) Boletim “quase 10”: 8 ou 9, sem flood
    for n, c in absences.items():
        if c in (8, 9) and _last_bullet_sent.get(n, -1) != c:
            _last_bullet_sent[n] = c
            await tg_alert(f"⏱️ Quase 10 sem vir: <b>{n}</b> (<b>{c}</b> sem vir)")

# ==== LOOPS DE FUNDO ====
@app.on_event("startup")
async def _auto_loops_absence():
    async def _loop():
        while True:
            try:
                await scan_and_alert_once()
            except Exception as e:
                print(f"[loop_scan] error: {e}")
            await asyncio.sleep(ALMOST_SCAN_EVERY)  # a cada 5 min
    asyncio.create_task(_loop())

# ==== ROTAS DE TESTE/INGESTÃO ====
@app.get("/test_alert")
async def test_alert():
    try:
        await tg_alert("🔧 Teste de alerta — canal OK.")
        return {"ok": True, "to": ALERT_CHANNEL}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/ingest_seq")
async def ingest_seq(request: Request):
    """
    Envie uma sequência para popular a timeline rapidamente.
    Body JSON: {"seq": "1,2,3,2,4,1"}  (ordem cronológica)
    """
    try:
        data = await request.json()
        raw = (data.get("seq") or "").strip()
        parts = [int(x) for x in re.findall(r"[1-4]", raw)]
        if not parts:
            return {"ok": False, "error": "seq vazia"}
        for n in parts:
            append_timeline(n)
        update_ngrams()
        return {"ok": True, "added": len(parts)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/force_scan")
async def force_scan():
    try:
        await scan_and_alert_once()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}