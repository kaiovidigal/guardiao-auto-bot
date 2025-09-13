# =========================
# AUSÊNCIAS (alertas 7 e 10) — robusto
# =========================
ALERT_CHANNEL = os.getenv("ALERT_CHANNEL", "").strip()
WATCH_CHANNEL = os.getenv("WATCH_CHANNEL", "").strip()
ABS_ALERT_LEVELS = (7, 10)          # >>> agora avisa no 7 e no 10
ABS_LOOP_SECONDS = 300              # checagem periódica

def abs_init_db():
    con = _connect(); cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS absence_state (
            id INTEGER PRIMARY KEY CHECK (id=1),
            a1 INTEGER NOT NULL DEFAULT 0,
            a2 INTEGER NOT NULL DEFAULT 0,
            a3 INTEGER NOT NULL DEFAULT 0,
            a4 INTEGER NOT NULL DEFAULT 0,
            r1 INTEGER NOT NULL DEFAULT 0,
            r2 INTEGER NOT NULL DEFAULT 0,
            r3 INTEGER NOT NULL DEFAULT 0,
            r4 INTEGER NOT NULL DEFAULT 0,
            last_ts INTEGER NOT NULL DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS absence_alerts (
            number INTEGER PRIMARY KEY,
            last_level INTEGER NOT NULL DEFAULT 0
        )
    """)
    cur.execute("INSERT OR IGNORE INTO absence_state (id) VALUES (1)")
    con.commit(); con.close()

abs_init_db()

def _abs_get_state():
    row = query_one("SELECT a1,a2,a3,a4,r1,r2,r3,r4,last_ts FROM absence_state WHERE id=1")
    a = [0, row["a1"], row["a2"], row["a3"], row["a4"]] if row else [0,0,0,0,0]
    r = [0, row["r1"], row["r2"], row["r3"], row["r4"]] if row else [0,0,0,0,0]
    return {"a": a, "r": r, "ts": (row["last_ts"] if row else 0)}

def _abs_set_state(a, r):
    exec_write("""
        UPDATE absence_state SET
            a1=?, a2=?, a3=?, a4=?,
            r1=?, r2=?, r3=?, r4=?,
            last_ts=?
        WHERE id=1
    """, (a[1],a[2],a[3],a[4], r[1],r[2],r[3],r[4], now_ts()))

def _abs_current_level(cur:int) -> int:
    if cur >= 10: return 10
    if cur >= 7:  return 7
    return 0

def _abs_get_last_level(n:int) -> int:
    row = query_one("SELECT last_level FROM absence_alerts WHERE number=?", (n,))
    return int(row["last_level"]) if row else 0

def _abs_set_last_level(n:int, level:int):
    exec_write("""
        INSERT INTO absence_alerts (number,last_level)
        VALUES (?,?)
        ON CONFLICT(number) DO UPDATE SET last_level=excluded.last_level
    """, (n, level))

def _abs_reset_last_level(n:int):
    exec_write("""
        INSERT INTO absence_alerts (number,last_level)
        VALUES (?,0)
        ON CONFLICT(number) DO UPDATE SET last_level=0
    """, (n,))

async def _abs_maybe_alert_one(n:int, cur:int, rec:int):
    level = _abs_current_level(cur)
    if level == 0: 
        return
    last = _abs_get_last_level(n)
    if level > last:
        if ALERT_CHANNEL:
            txt = (
                f"⏱️ <b>Quase {level} sem vir:</b> {n} ({cur} sem vir)\n"
                f"📊 <i>Recorde histórico:</i> {rec} vezes"
            )
            await tg_send_text(ALERT_CHANNEL, txt)
        _abs_set_last_level(n, level)

def abs_register_number(n_seen:int):
    """
    Chame SEMPRE que chegar um número real 1..4.
    Atualiza ausências + recorde. Reseta dedupe do número que acabou de vir.
    Também dispara checagem imediata (assíncrona) para alertar sem esperar o loop.
    """
    st = _abs_get_state()
    a, r = st["a"], st["r"]

    # atualiza contadores
    for k in (1,2,3,4):
        if k == n_seen:
            a[k] = 0
            _abs_reset_last_level(k)  # permite alertar de novo na próxima sequência
        else:
            a[k] += 1
            if a[k] > r[k]:
                r[k] = a[k]

    _abs_set_state(a, r)

    # checa imediatamente quem atingiu níveis (sem esperar o loop)
    rec = max(r[1], r[2], r[3], r[4])
    try:
        loop = asyncio.get_event_loop()
        for k in (1,2,3,4):
            if k != n_seen and _abs_current_level(a[k]) > 0:
                loop.create_task(_abs_maybe_alert_one(k, a[k], rec))
    except Exception as e:
        print(f"[ABS] immediate check error: {e}")

async def abs_check_and_alert():
    """
    Loop periódico de segurança (a cada ABS_LOOP_SECONDS).
    Garante que o alerta seja emitido mesmo sem eventos recentes.
    """
    st = _abs_get_state()
    a, r = st["a"], st["r"]
    rec = max(r[1], r[2], r[3], r[4])
    for n in (1,2,3,4):
        await _abs_maybe_alert_one(n, a[n], rec)

# startup do loop
@app.on_event("startup")
async def _start_abs_loop():
    async def _loop_abs():
        while True:
            try:
                await abs_check_and_alert()
            except Exception as e:
                print(f"[ABS] loop error: {e}")
            await asyncio.sleep(ABS_LOOP_SECONDS)
    try:
        asyncio.create_task(_loop_abs())
    except Exception as e:
        print(f"[ABS] startup error: {e}")