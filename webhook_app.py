# -*- coding: utf-8 -*-
# Fan Tan — Guardião (G0 + Recuperação G1/G2) — sem IA autonoma
# Ajustado para salvar o banco em /var/data/data.db/main.sqlite

import os, re, json, time, sqlite3, asyncio, shutil
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone, timedelta
from collections import Counter

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# =========================
# ENV / CONFIG
# =========================
DB_PATH = os.getenv("DB_PATH", "/var/data/data.db/main.sqlite").strip()
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
PUBLIC_CHANNEL = os.getenv("PUBLIC_CHANNEL", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
FLUSH_KEY      = os.getenv("FLUSH_KEY", "meusegredo123").strip()

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN:
    print("⚠️ Defina TG_BOT_TOKEN e WEBHOOK_TOKEN.")

TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
REPL_ENABLED, REPL_CHANNEL = True, os.getenv("REPL_CHANNEL", "-1003052132833").strip() or "-1003052132833"

app = FastAPI(title="Fantan Guardião — Canal-only (G0 + Recuperação)", version="4.2.1")

# =========================
# DB helpers
# =========================
OLD_DB_CANDIDATES = [
    "/var/data/data.db",
    "/opt/render/project/src/data.db",
    "/opt/render/project/src/data/data.db",
    "/data/data.db",
]

def _ensure_db_dir():
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    except Exception as e:
        print(f"[DB] mkdir: {e}")

def _migrate_old_db_if_needed():
    if os.path.exists(DB_PATH):
        return
    for src in OLD_DB_CANDIDATES:
        if os.path.exists(src):
            try:
                _ensure_db_dir()
                shutil.copy2(src, DB_PATH)
                print(f"[DB] Migrado {src} -> {DB_PATH}")
                return
            except Exception as e:
                print(f"[DB] Migração falhou {src}: {e}")

_ensure_db_dir()
_migrate_old_db_if_needed()

def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con