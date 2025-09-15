def init_db():
    con = _connect()
    cur = con.cursor()

    # Tabela de pendências (apenas estado; não influencia a IA)
    cur.execute("""CREATE TABLE IF NOT EXISTS pending_outcome (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        strategy TEXT DEFAULT '',
        suggested INTEGER NOT NULL,
        stage INTEGER NOT NULL,
        open INTEGER NOT NULL,
        window_left INTEGER NOT NULL,
        seen_numbers TEXT DEFAULT '',
        announced INTEGER NOT NULL DEFAULT 0,
        source TEXT NOT NULL DEFAULT 'CHAN'
    )""")

    # Migra suavemente se o banco já existir sem alguma coluna
    for col, ddl in [
        ("strategy",     "ALTER TABLE pending_outcome ADD COLUMN strategy TEXT DEFAULT ''"),
        ("seen_numbers", "ALTER TABLE pending_outcome ADD COLUMN seen_numbers TEXT DEFAULT ''"),
        ("announced",    "ALTER TABLE pending_outcome ADD COLUMN announced INTEGER NOT NULL DEFAULT 0"),
        ("source",       "ALTER TABLE pending_outcome ADD COLUMN source TEXT NOT NULL DEFAULT 'CHAN'"),
    ]:
        try:
            cur.execute(f"SELECT {col} FROM pending_outcome LIMIT 1")
        except sqlite3.OperationalError:
            try:
                cur.execute(ddl)
            except sqlite3.OperationalError:
                pass

    # Índices leves para desempenho (opcional, não afeta lógica)
    try: cur.execute("CREATE INDEX IF NOT EXISTS idx_pending_open ON pending_outcome(open)")
    except sqlite3.OperationalError: pass
    try: cur.execute("CREATE INDEX IF NOT EXISTS idx_pending_stage ON pending_outcome(stage)")
    except sqlite3.OperationalError: pass

    con.commit()
    con.close()