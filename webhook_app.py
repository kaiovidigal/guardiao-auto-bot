#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GuardiAo Auto Bot — webhook_app.py
v7.9.5 (Final: Otimização, Persistência e Reset Diário do Placar)

ENV obrigatórias (Render -> Environment):
- TG_BOT_TOKEN
- WEBHOOK_TOKEN
- SOURCE_CHANNEL
- TARGET_CHANNEL
- GEMINI_API_KEY
- MAX_GALE (DEVE SER DEFINIDO COMO '0' PARA FOCO G0)
...
"""

import os, re, json, time, math, sqlite3, datetime, hashlib
from typing import List, Dict, Optional, Tuple, Any

import httpx
from fastapi import FastAPI, Request, HTTPException

# Imports para FUSO HORÁRIO
import pytz ### MODIFICAÇÃO DIÁRIA ###

# >>> IMPORTS DO GEMINI <<<
from google import genai
from google.genai import types

# ------------------------------------------------------
# ENV & constantes
# ------------------------------------------------------
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "").strip()
# NOVO: Chave do Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

SHOW_DEBUG       = os.getenv("SHOW_DEBUG", "False").strip().lower() == "true"
# IMPORTANTE: MAX_GALE deve ser configurado como '0' no Render para foco G0
MAX_GALE         = int(os.getenv("MAX_GALE", "0")) 
OBS_TIMEOUT_SEC  = int(os.getenv("OBS_TIMEOUT_SEC", "420"))
DEDUP_WINDOW_SEC = int(os.getenv("DEDUP_WINDOW_SEC", "40"))
# Janela de dados para otimização (100-200 é um bom equilíbrio)
OPTIMIZATION_WINDOW = 150 

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN or not TARGET_CHANNEL:
    raise RuntimeError("Faltam ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN, TARGET_CHANNEL.")
TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

# >>> AJUSTE CRUCIAL: Apontando para o disco persistente do Render (/var/data) <<<
DB_PATH = "/var/data/main.sqlite"

# FUSO HORÁRIO DE BRASÍLIA
TZ = pytz.timezone('America/Sao_Paulo') ### MODIFICAÇÃO DIÁRIA ###

# ------------------------------------------------------
# Cliente Gemini (Inicialização)
# ------------------------------------------------------
GEMINI_CLIENT = None
GEMINI_MODEL  = "gemini-2.5-flash" 

if GEMINI_API_KEY:
    try:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini Client inicializado com sucesso.")
    except Exception as e:
        print(f"AVISO: Falha ao inicializar Gemini Client: {e}. E4 usará 25% fixo.")

# ------------------------------------------------------
# App
# ------------------------------------------------------
app = FastAPI(title="GuardiAo Auto Bot (webhook)", version="7.9.5")

# ------------------------------------------------------
# DB helpers
# ------------------------------------------------------
def _con():
    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

def db_init():
    con = _con(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS processed(
        update_id TEXT PRIMARY KEY,
        seen_at   INTEGER NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER NOT NULL,
        number INTEGER NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pending(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER,
        opened_at  INTEGER,
        suggested  INTEGER,
        stage      INTEGER DEFAULT 0,
        seen       TEXT     DEFAULT '',
        open       INTEGER  DEFAULT 1
    )""")
    # TABELA SCORE: ADICIONADO last_reset_day
    cur.execute("""CREATE TABLE IF NOT EXISTS score(
        id INTEGER PRIMARY KEY CHECK(id=1),
        green INTEGER DEFAULT 0,
        loss  INTEGER DEFAULT 0,
        last_reset_day TEXT DEFAULT '1970-01-01' ### MODIFICAÇÃO DIÁRIA ###
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS dedupe(
        kind TEXT NOT NULL,
        dkey TEXT NOT NULL,
        ts   INTEGER NOT NULL,
        PRIMARY KEY (kind, dkey)
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS config(
        id INTEGER PRIMARY KEY CHECK(id=1),
        w1 REAL DEFAULT 0.30, 
        w2 REAL DEFAULT 0.20, 
        w3 REAL DEFAULT 0.20, 
        w4 REAL DEFAULT 0.30,
        last_opt_ts INTEGER DEFAULT 0
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS detailed_history(
        id INTEGER PRIMARY KEY,
        opened_at INTEGER NOT NULL,
        suggested INTEGER NOT NULL,
        result    INTEGER,
        p1_conf   REAL, p2_conf   REAL, 
        p3_conf   REAL, p4_conf   REAL
    )""")

    if not con.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        # Define a data de reset inicial como a data de hoje para evitar reset imediato
        today_str = datetime.datetime.now(TZ).strftime('%Y-%m-%d') ### MODIFICAÇÃO DIÁRIA ###
        con.execute("INSERT INTO score(id,green,loss,last_reset_day) VALUES(1,0,0,?)",(today_str,)) ### MODIFICAÇÃO DIÁRIA ###
        
    if not con.execute("SELECT 1 FROM config WHERE id=1").fetchone():
        con.execute("INSERT INTO config(id,w1,w2,w3,w4) VALUES(1,0.30,0.20,0.20,0.30)")
        
    con.commit(); con.close()

db_init()


def _db_get_weights()->Tuple[float,float,float,float]:
    con=_con(); row=con.execute("SELECT w1,w2,w3,w4 FROM config WHERE id=1").fetchone(); con.close()
    return (row["w1"], row["w2"], row["w3"], row["w4"]) if row else (0.30, 0.20, 0.20, 0.30)

def _db_save_weights(w:Dict[str,float]):
    con=_con(); now=int(time.time())
    con.execute("UPDATE config SET w1=?,w2=?,w3=?,w4=?,last_opt_ts=? WHERE id=1",
                (w["E1"], w["E2"], w["E3"], w["E4"], now)) 
    con.commit(); con.close()

def _db_get_detailed_history(n:int=OPTIMIZATION_WINDOW):
    con=_con()
    rows = con.execute("SELECT * FROM detailed_history WHERE result IS NOT NULL ORDER BY id DESC LIMIT ?", (n,)).fetchall()
    con.close()
    return rows


def _mark_processed(upd: str):
    try:
        con = _con()
        con.execute("INSERT OR IGNORE INTO processed(update_id,seen_at) VALUES(?,?)",(str(upd), int(time.time())))
        con.commit(); con.close()
    except Exception:
        pass

def _timeline_tail(n:int=400)->List[int]:
    con = _con()
    rows = con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?",(n,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def _append_seq(seq: List[int]):
    if not seq: return
    con = _con()
    now = int(time.time())
    con.executemany("INSERT INTO timeline(created_at,number) VALUES(?,?)",[(now,int(x)) for x in seq])
    con.commit(); con.close()

def _timeline_size()->int:
    con=_con(); row=con.execute("SELECT COUNT(*) c FROM timeline").fetchone(); con.close()
    return int(row["c"] or 0)

def _score_add(outcome:str):
    con = _con()
    # A consulta agora inclui o last_reset_day, mas só usamos green/loss
    row = con.execute("SELECT green,loss FROM score WHERE id=1").fetchone()
    g,l = (int(row["green"]), int(row["loss"])) if row else (0,0)
    if outcome.upper()=="GREEN": g+=1
    elif outcome.upper()=="LOSS": l+=1
    
    con.execute("UPDATE score SET green=?,loss=? WHERE id=1",(g,l))
    con.commit(); con.close()

def _score_reset_if_new_day(): ### MODIFICAÇÃO DIÁRIA ###
    """Verifica e zera o placar se um novo dia (00:00 GMT-3) começou."""
    con = _con()
    row = con.execute("SELECT green, loss, last_reset_day FROM score WHERE id=1").fetchone()
    
    if not row:
        con.close()
        return

    # Data atual em GMT-3 (Brasília)
    now_br = datetime.datetime.now(TZ)
    today_str = now_br.strftime('%Y-%m-%d')
    last_reset_day = row["last_reset_day"]
    
    if today_str != last_reset_day:
        # É um novo dia (meia-noite passou). ZERA O PLACAR.
        
        # O reset deve ser feito aqui no DB.
        con.execute("UPDATE score SET green=0, loss=0, last_reset_day=? WHERE id=1", (today_str,))
        con.commit()

    con.close() ### MODIFICAÇÃO DIÁRIA ###

def _score_text()->str: ### MODIFICAÇÃO DIÁRIA ###
    # 1. Tenta resetar o placar se for um novo dia
    _score_reset_if_new_day() 
    
    # 2. Lê e retorna o placar atualizado
    con = _con(); row = con.execute("SELECT green,loss FROM score WHERE id=1").fetchone(); con.close()
    if not row: return "0 GREEN × 0 LOSS — 0.0%"
    g,l = int(row["green"]), int(row["loss"])
    tot = g+l; acc = (g/tot*100.0) if tot>0 else 0.0
    
    # Adiciona a data atual ao placar para referência
    now_br = datetime.datetime.now(TZ).strftime('%d/%m')
    return f"{g} GREEN × {l} LOSS — {acc:.1f}% ({now_br})" ### MODIFICAÇÃO DIÁRIA ###

def _pending_get()->Optional[sqlite3.Row]:
    con = _con(); row = con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone(); con.close()
    return row

def _pending_open(suggested:int):
    if _pending_get(): return False
    con = _con()
    now = int(time.time())
    con.execute("INSERT INTO pending(created_at,opened_at,suggested,stage,seen,open) VALUES(?,?,?,?,?,1)",
                (now, now, int(suggested), 0, ""))
    con.commit(); con.close()
    return True

def _pending_set_stage(stage:int):
    con = _con()
    con.execute("UPDATE pending SET stage=? WHERE open=1",(int(stage),))
    con.commit(); con.close()

def _pending_seen_append(nums: List[int], need:int=2):
    row = _pending_get()
    if not row: return
    seen = (row["seen"] or "").strip()
    arr = [s for s in seen.split("-") if s]
    for n in nums:
        if len(arr) >= need: break
        arr.append(str(int(n)))
    txt = "-".join(arr[:need])
    con = _con(); con.execute("UPDATE pending SET seen=? WHERE id=?", (txt, int(row["id"]))); con.commit(); con.close()

def _pending_close(final_seen: str, outcome: str, stage_lbl: str, suggested:int)->str:
    row = _pending_get()
    if not row: return ""
    con = _con()
    con.execute("UPDATE pending SET open=0, seen=? WHERE id=? AND open=1", (final_seen, int(row["id"])))
    con.commit(); con.close()
    _score_add(outcome)
    # alimentar timeline com observados
    obs = [int(x) for x in final_seen.split("-") if x.isdigit()]
    _append_seq(obs)
    
    # AQUI FOI CORRIGIDO para exibir o palpite do bot
    our = suggested
    msg = (f"{'🟢' if outcome.upper()=='GREEN' else '🔴'} <b>{outcome.upper()}</b> — finalizado "
           f"(<b>{stage_lbl}</b>, nosso={our}, observados={final_seen}).\n"
           f"📊 Geral: {_score_text()}")
    return msg

# ------------------ DEDUPE por conteúdo ------------------
def _dedupe_key(text: str) -> str:
    base = re.sub(r"\s+", " ", (text or "")).strip().lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _seen_recent(kind: str, dkey: str) -> bool:
    now = int(time.time())
    con = _con()
    row = con.execute("SELECT ts FROM dedupe WHERE kind=? AND dkey=?", (kind, dkey)).fetchone()
    if row and now - int(row["ts"]) <= DEDUP_WINDOW_SEC:
        con.close()
        return True
    con.execute("INSERT OR REPLACE INTO dedupe(kind, dkey, ts) VALUES (?,?,?)", (kind, dkey, now))
    con.commit(); con.close()
    return False

# ------------------------------------------------------
# IA “12 camadas” compacta (4 especialistas + ajustes)
# ------------------------------------------------------

def _norm(d: Dict[int,float])->Dict[int,float]:
    s=sum(d.values()) or 1e-9
    return {k:v/s for k,v in d.items()}

def _post_freq(tail:List[int], k:int)->Dict[int,float]:
    if not tail: return {1:0.25,2:0.25,3:0.25,4:0.25}
    win = tail[-k:] if len(tail)>=k else tail
    tot=max(1,len(win))
    return _norm({c:win.count(c)/tot for c in (1,2,3,4)})

def _post_e1_ngram(tail:List[int])->Dict[int,float]:
    # proxy simples: mistura de janelas (Fibonacci curto embutido)
    mix={c:0.0 for c in (1,2,3,4)}
    for k,w in ((8,0.25),(21,0.35),(55,0.40)):
        pk=_post_freq(tail,k)
        for c in (1,2,3,4): mix[c]+=w*pk[c]
    return _norm(mix)

def _post_e2_short(tail):  return _post_freq(tail, 60)
def _post_e3_long(tail):   return _post_freq(tail, 300)

# >>> ESPECIALISTA E4 - GEMINI (COM FOCO G0/ALTO RISCO) <<<
async def _post_e4_llm(tail:List[int])->Dict[int,float]:
    # Variável para armazenar o resultado do E4 para o snapshot
    _post_e4_llm.last_run = {1:0.25,2:0.25,3:0.25,4:0.25}

    if not GEMINI_CLIENT:
        return _post_e4_llm.last_run

    # Prepara o histórico (usaremos os últimos 40 resultados)
    history_str = ",".join(map(str, tail[-40:])) 
    
    prompt = f"""
    Sua única tarefa é atuar como um especialista de alto risco/alta recompensa.
    A sequência histórica recente de resultados (os mais recentes à direita) é: {history_str}.
    Os resultados possíveis são SOMENTE os números 1, 2, 3 e 4.
    
    Analise esta sequência para identificar o **único resultado** que tem a MAIOR PROBABILIDADE de sair na próxima rodada, ignorando a necessidade de cobertura (Gale).
    
    Sua resposta deve ser *exclusivamente* um objeto JSON formatado como o schema. Você deve atribuir **pelo menos 40%** de chance ao resultado mais provável, e o restante deve ser distribuído entre os outros três, somando 1.0.
    """

    try:
        # Chamada assíncrona
        response = await GEMINI_CLIENT.models.generate_content_async(
            model=GEMINI_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                response_schema={
                    "type": "object", 
                    "properties": {
                        "1": {"type": "number"}, "2": {"type": "number"}, 
                        "3": {"type": "number"}, "4": {"type": "number"}
                    },
                    "required": ["1", "2", "3", "4"]
                }
            )
        )
        
        data = json.loads(response.text)
        post = {int(k): float(v) for k, v in data.items() if k.isdigit() and 1<=int(k)<=4}
        
        if len(post) == 4 and all(v >= 0 for v in post.values()):
            _post_e4_llm.last_run = _norm(post) # Armazena para snapshot
            return _post_e4_llm.last_run
        
        raise ValueError("Resposta JSON do Gemini inválida ou incompleta.")

    except Exception as e:
        if SHOW_DEBUG: print(f"DEBUG Gemini Error (E4): {e}")
        return _post_e4_llm.last_run # Retorna
