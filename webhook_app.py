# ==== CONFIG FIXA (sem depender de ENV) ====
ALERT_CHANNEL = os.getenv("ALERT_CHANNEL", "-1002796105884").strip()  # saída padrão (ajuste se quiser)
ALMOST_LIMIAR = 5        # 🔔 ALERTA já em 5 sem vir (fixo)
ALMOST_SCAN_EVERY = 300  # boletim a cada 5 min

# ==== helper para enviar alerta ====
async def tg_alert(text: str):
    if not ALERT_CHANNEL:
        print("[ALERT] ALERT_CHANNEL vazio — defina no Render ou ajuste acima.")
        return
    await tg_send_text(ALERT_CHANNEL, text, "HTML")

# ---- ABSENÇAS (reuso das suas funções/tabelas) ----
def get_absences() -> dict[int, int]:
    rows = query_all("SELECT number FROM timeline ORDER BY id DESC LIMIT 5000")
    recent = [r["number"] for r in rows]
    if not recent:
        return {1: 0, 2: 0, 3: 0, 4: 0}
    absences = {}
    for n in (1,2,3,4):
        try:
            last_idx = recent.index(n)   # posição na ordem DESC
            absences[n] = last_idx
        except ValueError:
            absences[n] = len(recent)
    return absences

_last_bullet_sent: dict[int, int] = {1:-1,2:-1,3:-1,4:-1}  # evita repetição 8/9

async def scan_and_alert_once():
    absences = get_absences()
    if not absences:
        return

    # 1) ALERTA principal: qualquer número >= 5
    hot = [(n, c) for n, c in absences.items() if c >= ALMOST_LIMIAR]
    if hot:
        hot.sort(key=lambda x: x[1], reverse=True)
        linhas = [f"⚠️ Número <b>{n}</b> está há <b>{c}</b> sem vir." for n, c in hot]
        rec = max(absences.values())
        msg = "⏰ <b>Alerta de ausência</b>\n" + "\n".join(linhas) + f"\n\n📈 Recorde atual observado: <b>{rec}</b>"
        await tg_alert(msg)

    # 2) Boletim “quase 10”: se quiser manter (8/9) sem flood
    for n, c in absences.items():
        if c in (8, 9) and _last_bullet_sent.get(n, -1) != c:
            _last_bullet_sent[n] = c
            await tg_alert(f"⏱️ Quase 10 sem vir: <b>{n}</b> (<b>{c}</b> sem vir)")

# ---- STARTUP: faz 1º scan logo e inicia loop ----
@app.on_event("startup")
async def _auto_loops_absence():
    async def _loop():
        # primeiro scan rápido para destravar
        try:
            await asyncio.sleep(2)
            await scan_and_alert_once()
        except Exception as e:
            print(f"[startup_scan] error: {e}")
        # depois a cada 5 min
        while True:
            try:
                await scan_and_alert_once()
            except Exception as e:
                print(f"[loop_scan] error: {e}")
            await asyncio.sleep(ALMOST_SCAN_EVERY)
    asyncio.create_task(_loop())

# ---- ROTAS DE TESTE/INGESTÃO ----
@app.get("/test_alert")
async def test_alert():
    try:
        await tg_alert("🔧 Teste de alerta — canal OK.")
        return {"ok": True, "sent": True, "to": ALERT_CHANNEL}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/ingest_seq")
async def ingest_seq(request: Request):
    """
    Popular timeline rapidamente.
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