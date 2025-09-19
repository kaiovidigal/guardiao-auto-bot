[05:56, 9/18/2025] Cartão Sam's Club: #!/usr/bin/env python3
# -- coding: utf-8 --

"""
Webhook (GEN híbrido + estratégia + timeout 45s + relatório 1h + reset diário America/Sao_Paulo)
— versão enxuta e estável, com os IDs atualizados
"""

import os, re, time, json, sqlite3, asyncio
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import FastAPI, Request, HTTPException
import zoneinfo

# ===================== ENV =====================
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()

# >>> ATUALIZADOS <<<
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1003081474331").strip()  # destino (Sinal 24 fan tan)
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "-100281050…
[06:31, 9/18/2025] Cartão Sam's Club: #!/usr/bin/env python3
# -- coding: utf-8 --

"""
guardiao-auto-bot — GEN híbrido + relatório 1h
-> versão enxuta e estável para Render/railway (polling interno + FastAPI viva)
"""

import os, re, time, json, sqlite3, asyncio
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone, timedelta
import httpx
from fastapi import FastAPI
import zoneinfo

# ========= ENV =========
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()  # pode ficar vazio se não usar webhook
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "-1003804744331").strip()  # << NOVO GRUPO
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", os.getenv("PUBLIC_CHANNEL", "")).strip()  # opcional
DB_PATH        = os.getenv("DB_PATH", "/var/data/data.db").strip() or "/var/data/data.db"

if not TG_BOT_TOKEN:
    raise RuntimeError("Defina TG_BOT_TOKEN no ambiente.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
app = FastAPI(title="guardiao-auto-bot (relatório 1h)", version="3.2.0")

# ========= Fuso local =========
TZ_LOCAL = zoneinfo.ZoneInfo("America/Sao_Paulo")
def now_local(): return datetime.now(TZ_LOCAL)
def day_key(dt=None):
    dt = dt or now_local(); return dt.strftime("%Y%m%d")
def hour_key(dt=None):
    dt = dt or now_local(); return dt.strftime("%Y%m%d%H")

# ========= Parâmetros de decisão (mantidos) =========
SHORT_WINDOW    = 80
LONG_WINDOW     = 800
CONF_SHORT_MIN  = 0.85
CONF_LONG_MIN   = 0.90
GAP_MIN         = 0.050
FINAL_TIMEOUT   = 45
MIN_SAMPLES_SHORT = 120
MIN_SAMPLES_LONG  = 400
MAX_LOSS_STREAK   = 2
COOLDOWN_SECONDS  = 120
QUIET_HOURS       = [(0, 5)]
_last_cooldown_until = 0

# ========= Relatórios =========
REPORT_EVERY_SEC   = 60 * 60
GOOD_DAY_THRESHOLD = 0.80
BAD_DAY_THRESHOLD  = 0.50

# ========= Utils =========
def now_ts() -> int: return int(time.time())

# ========= DB =========
def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

def _exec_write(sql: str, params: tuple=()):
    for attempt in range(6):
        try:
            con = _connect(); cur = con.cursor()
            cur.execute(sql, params)
            con.commit(); con.close(); return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                time.sleep(0.25*(attempt+1)); continue
            raise

def _query_all(sql: str, params: tuple=()) -> List[sqlite3.Row]:
    con = _connect(); cur = con.cursor()
    rows = cur.execute(sql, params).fetchall()
    con.close()
    return rows

def migrate_db():
    con = _connect(); cur = con.cursor()
    # tabelas usadas por pending/score (mantém compat)
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pending (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER,
        suggested INTEGER,
        open INTEGER DEFAULT 1,
        seen TEXT,
        opened_at INTEGER,
        last_post_short TEXT,
        last_post_long  TEXT,
        last_conf_short REAL,
        last_conf_long  REAL,
        d_final INTEGER,
        base TEXT,
        pattern_key TEXT,
        closed_at INTEGER,
        outcome TEXT,
        stage TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS score (
        id INTEGER PRIMARY KEY CHECK (id=1),
        green INTEGER DEFAULT 0,
        loss  INTEGER DEFAULT 0
    )""")
    if not cur.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        cur.execute("INSERT INTO score (id, green, loss) VALUES (1,0,0)")

    # colunas que podem faltar
    for col, ddl in [
        ("d_final","ALTER TABLE pending ADD COLUMN d_final INTEGER"),
        ("last_post_short","ALTER TABLE pending ADD COLUMN last_post_short TEXT"),
        ("last_post_long","ALTER TABLE pending ADD COLUMN last_post_long TEXT"),
        ("last_conf_short","ALTER TABLE pending ADD COLUMN last_conf_short REAL"),
        ("last_conf_long","ALTER TABLE pending ADD COLUMN last_conf_long REAL"),
        ("base","ALTER TABLE pending ADD COLUMN base TEXT"),
        ("pattern_key","ALTER TABLE pending ADD COLUMN pattern_key TEXT"),
        ("closed_at","ALTER TABLE pending ADD COLUMN closed_at INTEGER"),
        ("outcome","ALTER TABLE pending ADD COLUMN outcome TEXT"),
        ("stage","ALTER TABLE pending ADD COLUMN stage TEXT"),
    ]:
        try: cur.execute(f"SELECT {col} FROM pending LIMIT 1")
        except sqlite3.OperationalError:
            try: cur.execute(ddl)
            except sqlite3.OperationalError: pass

    # patches de relatório/controle por chat
    cur.execute("""
    CREATE TABLE IF NOT EXISTS relatorio_controle (
        chat_id TEXT NOT NULL,
        hour_key TEXT NOT NULL,
        sent_at INTEGER NOT NULL,
        PRIMARY KEY (chat_id, hour_key)
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS relatorio_dia (
        chat_id TEXT PRIMARY KEY,
        day_key TEXT NOT NULL
    );
    """)

    con.commit(); con.close()

migrate_db()

# ========= Telegram =========
async def tg_send_text(chat_id: str, text: str, parse: str="HTML"):
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(f"{TELEGRAM_API}/sendMessage",
                          json={"chat_id": chat_id, "text": text,
                                "parse_mode": parse, "disable_web_page_preview": True})

# ========= Score =========
def bump_score(outcome: str):
    con = _connect(); cur = con.cursor()
    row = cur.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
    g, l = (row["green"], row["loss"]) if row else (0, 0)
    if outcome.upper() == "GREEN": g += 1
    elif outcome.upper() == "LOSS": l += 1
    cur.execute("INSERT OR REPLACE INTO score (id, green, loss) VALUES (1,?,?)", (g, l))
    con.commit(); con.close()

def score_text() -> str:
    con = _connect()
    row = con.execute("SELECT green, loss FROM score WHERE id=1").fetchone()
    con.close()
    if not row: return "0 GREEN × 0 LOSS — 0.0%"
    g, l = int(row["green"]), int(row["loss"])
    total = g + l
    acc = (g/total*100.0) if total>0 else 0.0
    return f"{g} GREEN × {l} LOSS — {acc:.1f}%"

# ========= Relatório / snapshot do dia =========
def _report_snapshot_day(chat_id: str) -> Dict[str,int]:
    start = now_local().replace(hour=0, minute=0, second=0, microsecond=0)
    start_ts = int(start.astimezone(timezone.utc).timestamp())

    rows = _query_all("""
        SELECT outcome, stage FROM pending
        WHERE closed_at IS NOT NULL AND closed_at >= ?
    """, (start_ts,))

    g0=g1=g2=0; l0=l1=l2=0
    for r in rows:
        oc = (r["outcome"] or "").upper()
        st = (r["stage"] or "").upper()
        if oc == "GREEN":
            if st == "G0": g0 += 1
            elif st == "G1": g1 += 1
            else: g2 += 1
        elif oc == "LOSS":
            if st == "G0": l0 += 1
            elif st == "G1": l1 += 1
            else: l2 += 1

    row = _query_all("SELECT green, loss FROM score WHERE id=1")
    g = int(row[0]["green"] if row else 0)
    l = int(row[0]["loss"] if row else 0)
    total = g + l
    acc = (g/total) if total>0 else 0.0
    return {"g0":g0,"g1":g1,"g2":g2,"l0":l0,"l1":l1,"l2":l2,"day_green":g,"day_loss":l,"day_acc":acc}

def _day_mood(acc: float) -> str:
    if acc >= GOOD_DAY_THRESHOLD: return "Dia bom"
    if acc <= BAD_DAY_THRESHOLD:  return "Dia ruim"
    return "Dia neutro"

# ========= Controle por grupo =========
def gua_reset_if_new_day(chat_id: str):
    dk = day_key()
    rows = _query_all("SELECT day_key FROM relatorio_dia WHERE chat_id=?", (chat_id,))
    if not rows:
        _exec_write("INSERT OR REPLACE INTO relatorio_dia (chat_id, day_key) VALUES (?,?)", (chat_id, dk))
        return
    if rows[0]["day_key"] != dk:
        _exec_write("UPDATE relatorio_dia SET day_key=? WHERE chat_id=?", (dk, chat_id))

def gua_try_reserve_hour(chat_id: str) -> bool:
    hk = hour_key(); ts = int(time.time())
    try:
        _exec_write(
            "INSERT OR IGNORE INTO relatorio_controle(chat_id, hour_key, sent_at) VALUES (?,?,?)",
            (chat_id, hk, ts)
        )
        rows = _query_all(
            "SELECT sent_at FROM relatorio_controle WHERE chat_id=? AND hour_key=?",
            (chat_id, hk)
        )
        return bool(rows) and int(rows[0]["sent_at"]) == ts
    except Exception:
        return False

# ========= Loop de relatório (1x por hora) =========
async def _reporter_loop():
    while True:
        try:
            chat_id = TARGET_CHANNEL
            gua_reset_if_new_day(str(chat_id))
            if gua_try_reserve_hour(str(chat_id)):
                snap = _report_snapshot_day(str(chat_id))
                txt = (
                    "📈 <b>Relatório do dia</b>\n"
                    f"G0: <b>{snap['g0']}</b> GREEN / <b>{snap['l0']}</b> LOSS\n"
                    f"G1: <b>{snap['g1']}</b> GREEN / <b>{snap['l1']}</b> LOSS\n"
                    f"G2: <b>{snap['g2']}</b> GREEN / <b>{snap['l2']}</b> LOSS\n"
                    "—\n"
                    f"📊 <b>Dia</b>: <b>{snap['day_green']}</b> GREEN × <b>{snap['day_loss']}</b> LOSS — "
                    f"{snap['day_acc']*100:.1f}%\n"
                    f"{_day_mood(snap['day_acc'])}"
                )
                await tg_send_text(TARGET_CHANNEL, txt)
        except Exception as e:
            print(f"[RELATORIO] erro: {e}")
        await asyncio.sleep(REPORT_EVERY_SEC)  # 1h

# ========= Vida da app (Render) =========
@app.on_event("startup")
async def _on_startup():
    asyncio.create_task(_reporter_loop())

@app.get("/")
async def root():
    return {"ok": True, "service": "guardiao-auto-bot", "target": TARGET_CHANNEL}
