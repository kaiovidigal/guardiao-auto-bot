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
import pytz 

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
TZ = pytz.timezone('America/Sao_Paulo')

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
        last_reset_day TEXT DEFAULT '1970-01-01'
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
        today_str = datetime.datetime.now(TZ).strftime('%Y-%m-%d')
        con.execute("INSERT INTO score(id,green,loss,last_reset_day) VALUES(1,0,0,?)",(today_str,))
        
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
    row = con.execute("SELECT green,loss FROM score WHERE id=1").fetchone()
    g,l = (int(row["green"]), int(row["loss"])) if row else (0,0)
    if outcome.upper()=="GREEN": g+=1
    elif outcome.upper()=="LOSS": l+=1
    
    con.execute("UPDATE score SET green=?,loss=? WHERE id=1",(g,l))
    con.commit(); con.close()

def _score_reset_if_new_day():
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
        g, l = row["green"], row["loss"]
        
        # Opcional: Notificação de fechamento do dia anterior (executado pelo processo que chamou _score_text)
        # O reset deve ser feito aqui no DB.
        con.execute("UPDATE score SET green=0, loss=0, last_reset_day=? WHERE id=1", (today_str,))
        con.commit()

    con.close()

def _score_text()->str:
    # 1. Tenta resetar o placar se for um novo dia
    _score_reset_if_new_day() 
    
    # 2. Lê e retorna o placar atualizado
    con = _con(); row = con.execute("SELECT green,loss FROM score WHERE id=1").fetchone(); con.close()
    if not row: return "0 GREEN × 0 LOSS — 0.0%"
    g,l = int(row["green"]), int(row["loss"])
    tot = g+l; acc = (g/tot*100.0) if tot>0 else 0.0
    
    # Adiciona a data atual ao placar para referência
    now_br = datetime.datetime.now(TZ).strftime('%d/%m')
    return f"{g} GREEN × {l} LOSS — {acc:.1f}% ({now_br})"

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
        return _post_e4_llm.last_run # Retorna 25% fixo em caso de erro

# LÊ OS PESOS DINAMICAMENTE
def _hedge(p1,p2,p3,p4):
    w1, w2, w3, w4 = _db_get_weights() 
    w = (w1, w2, w3, w4) 
    cands=(1,2,3,4)
    out={c: w[0]*p1.get(c,0)+w[1]*p2.get(c,0)+w[2]*p3.get(c,0)+w[3]*p4.get(c,0) for c in cands}
    return _norm(out)

def _runnerup_ls2(post:Dict[int,float], loss_streak:int)->Tuple[int,Dict[int,float],str]:
    rank=sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    best=rank[0][0]
    
    if loss_streak>=2 and len(rank)>=2 and (rank[0][1]-rank[1][1])<0.05:
        return rank[1][0], post, "IA_runnerup_ls2"
        
    return best, post, "IA"

def _conf_floor(post:Dict[int,float], floor=0.35, cap=0.95):
    # Aumentando o piso de confiança para 35% (0.35)
    post=_norm({c:float(post.get(c,0)) for c in (1,2,3,4)})
    b=max(post,key=post.get); mx=post[b]
    if mx<floor:
        others=[c for c in (1,2,3,4) if c!=b]
        s=sum(post[c] for c in others)
        take=min(floor-mx, s)
        if s>0:
            scale=(s-take)/s
            for c in others: post[c]*=scale
        post[b]=min(cap, mx+take)
    if post[b]>cap:
        ex=post[b]-cap; post[b]=cap
        add=ex/3.0
        for c in (1,2,3,4):
            if c!=b: post[c]+=add
    return _norm(post)

def _get_ls()->int:
    # A complexidade de cálculo de loss streak é alta, e não é crucial para o modo G0
    # Retorna 0 no modo G0 para simplificar a lógica
    return 0

# >>> FUNÇÃO PRINCIPAL DE ESCOLHA (ASSÍNCRONA) <<<
async def _choose_number()->Tuple[int,float,int,Dict[int,float],float,str,Dict[int,float],Dict[int,float],Dict[int,float]]:
    tail=_timeline_tail(400)
    p1=_post_e1_ngram(tail)
    p2=_post_e2_short(tail)
    p3=_post_e3_long(tail)
    
    # CHAMADA ASSÍNCRONA PARA O GEMINI
    p4=await _post_e4_llm(tail) 
    
    base=_hedge(p1,p2,p3,p4)
    best, post_raw, reason = _runnerup_ls2(base, loss_streak=_get_ls())
    
    post_final=_conf_floor(post_raw, 0.35, 0.95) # O resultado final (após o floor)
    
    best=max(post_final,key=post_final.get); conf=float(post_final[best])
    r=sorted(post_final.items(), key=lambda kv: kv[1], reverse=True)
    gap=(r[0][1]-r[1][1]) if len(r)>=2 else r[0][1]
    
    # Retorna o post processado e agora também os posts individuais (p1, p2, p4) para o Passo 2
    return best, conf, _timeline_size(), post_final, gap, reason, p1, p2, p4

# ------------------------------------------------------
# PASSO 3: Lógica de Otimização (Meta-IA)
# ------------------------------------------------------
def _analyze_performance(rows: List[sqlite3.Row])->Dict[str, Any]:
    """Calcula a performance de cada especialista na janela de dados."""
    if not rows: return {"count": 0, "stats": {}}
    
    N = len(rows)
    # Inicializa contadores de acertos por especialista
    hits = {"E1": 0, "E2": 0, "E3": 0, "E4": 0}
    # Inicializa acerto/erro do Palpite Final (aqui 'suggested' é o palpite final)
    overall_hits = 0 
    
    # Colunas de confiança
    conf_cols = {"E1": "p1_conf", "E2": "p2_conf", "E3": "p3_conf", "E4": "p4_conf"}

    for row in rows:
        suggested = int(row["suggested"])
        result = int(row["result"])
        
        # 1. Acerto/Erro do Palpite Final (Importante para contexto)
        if suggested == result: overall_hits += 1

        # 2. Avaliação da performance de cada especialista (E1, E2, E3, E4)
        for e_key, conf_col in conf_cols.items():
            # Métrica: O especialista acertou se o número sugerido (que ele apoiou) foi o resultado.
            if suggested == result and row[conf_col] > 0.30: 
                 hits[e_key] += 1
            elif suggested != result and row[conf_col] <= 0.30: 
                 hits[e_key] += 0.5 # Bônus neutro por não apoiar o perdedor


    # Calcula as taxas de acerto
    stats = {}
    for e_key, h_count in hits.items():
        accuracy = (h_count / N) if N > 0 else 0.0
        stats[e_key] = {"hits": h_count, "accuracy": accuracy}

    overall_accuracy = (overall_hits / N) if N > 0 else 0.0

    return {
        "count": N,
        "overall_accuracy": overall_accuracy,
        "current_weights": dict(zip(["E1", "E2", "E3", "E4"], _db_get_weights())),
        "expert_performance": stats
    }


# Função principal de otimização (Passo 3)
async def _optimize_weights(optimization_key:str)->Dict[str,Any]:
    """Chama o Gemini para sugerir novos pesos com base na performance."""
    if optimization_key != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden: Optimization key invalid.")
        
    if not GEMINI_CLIENT:
        return {"status": "error", "message": "Gemini Client não está inicializado (API Key faltando)."}
        
    rows = _db_get_detailed_history(n=OPTIMIZATION_WINDOW)
    analysis = _analyze_performance(rows)
    N = analysis["count"]
    
    if N < 30:
        return {"status": "skipped", "message": f"Amostra insuficiente. Necessário >30 sinais, encontrado {N}.", "details": analysis}

    # Gera o prompt para o Gemini
    current_weights_str = json.dumps(analysis["current_weights"])
    performance_str = json.dumps(analysis["expert_performance"], indent=2)
    
    prompt = f"""
    Sua tarefa é atuar como um otimizador de peso de Meta-IA para um sistema de apostas com quatro especialistas (E1, E2, E3, E4).
    A aposta é baseada em uma média ponderada da confiança de cada especialista.
    
    Análise de Performance Recente (Últimos {N} Sinais):
    -----------------------------------------------------
    {performance_str}
    
    Pesos Atuais: {current_weights_str}
    Acurácia Geral do Sistema: {analysis['overall_accuracy']:.2f}
    
    Instruções para Otimização:
    1. Aumente o peso (W) dos especialistas (E1 a E4) que tiveram a maior 'accuracy' (taxa de acerto) nos últimos sinais.
    2. Diminua o peso dos especialistas com 'accuracy' baixa.
    3. Os pesos (W1, W2, W3, W4) devem somar **EXATAMENTE 1.0** (ou seja, 100%).
    4. Cada peso deve ser no mínimo 0.05 (5%) e no máximo 0.40 (40%) para garantir que nenhum especialista seja totalmente ignorado ou domine.
    
    Sua resposta deve ser *exclusivamente* um objeto JSON formatado como o schema.
    """
    
    # Schema de resposta
    response_schema = {
        "type": "object", 
        "properties": {
            "E1": {"type": "number", "description": "Novo peso W1"}, 
            "E2": {"type": "number", "description": "Novo peso W2"}, 
            "E3": {"type": "number", "description": "Novo peso W3"}, 
            "E4": {"type": "number", "description": "Novo peso W4 (Gemini)"}
        },
        "required": ["E1", "E2", "E3", "E4"]
    }

    try:
        response = await GEMINI_CLIENT.models.generate_content_async(
            model=GEMINI_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                response_schema=response_schema
            )
        )
        
        new_weights = json.loads(response.text)
        
        # Validação final e normalização
        if not isinstance(new_weights, dict) or len(new_weights) != 4:
             raise ValueError("Resposta JSON do Gemini inválida ou incompleta.")

        # Garante que os pesos somem 1.0 após o Gemini (ajuste fino)
        total_sum = sum(new_weights.values())
        if total_sum > 0:
            new_weights = {k: round(v / total_sum, 4) for k, v in new_weights.items()}

        # Salva os novos pesos
        _db_save_weights(new_weights)
        
        return {"status": "success", "message": "Otimização de pesos concluída.", "new_weights": new_weights, "details": analysis}

    except Exception as e:
        error_msg = f"DEBUG Gemini Error (Otimização): {e}"
        if SHOW_DEBUG: print(error_msg)
        return {"status": "error", "message": "Falha na otimização do Gemini.", "error": str(e), "details": analysis}


# ------------------------------------------------------
# Telegram helpers (send + delete)
# ------------------------------------------------------
async def tg_send(chat_id: str, text: str, parse="HTML"):
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            await cli.post(f"{TELEGRAM_API}/sendMessage",
                           json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                 "disable_web_page_preview": True})
    except Exception:
        pass

async def tg_send_return(chat_id: str, text: str, parse="HTML") -> Optional[int]:
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            r = await cli.post(f"{TELEGRAM_API}/sendMessage",
                               json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                     "disable_web_page_preview": True})
            data = r.json()
            if isinstance(data, dict) and data.get("ok") and data.get("result", {}).get("message_id"):
                return int(data["result"]["message_id"])
    except Exception:
        pass
    return None

async def tg_delete(chat_id: str, message_id: int):
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            await cli.post(f"{TELEGRAM_API}/deleteMessage",
                           json={"chat_id": chat_id, "message_id": int(message_id)})
    except Exception:
        pass


# ------------------------------------------------------
# Parser do canal-fonte
# ------------------------------------------------------
RX_ENTRADA = re.compile(r"ENTRADA\s+CONFIRMADA", re.I)
RX_ANALISE = re.compile(r"\bANALISANDO\b", re.I)
RX_FECHA   = re.compile(r"APOSTA\s+ENCERRADA", re.I)

RX_SEQ     = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
RX_NUMS    = re.compile(r"[1-4]")
RX_AFTER   = re.compile(r"ap[oó]s\s+o\s+o?\s*([1-4])", re.I)

RX_GREEN   = re.compile(r"GREEN|✅", re.I)
RX_RED     = re.compile(r"RED|❌", re.I)
RX_PAREN   = re.compile(r"\(([^\)]*)\)\s*$")

# >>> AJUSTE CRÍTICO AQUI: Regex para o formato "(1)" ou "(4)" no final da mensagem <<<
# Busca (1) ou (2) ou (3) ou (4) no final da string.
RX_CLOSING_NUM = re.compile(r"\(([1-4])\)\s*$", re.I) 

def _parse_seq_list(text:str)->List[int]:
    m=RX_SEQ.search(text or "")
    if not m: return []
    return [int(x) for x in RX_NUMS.findall(m.group(1))]

def _parse_seq_pair(text:str, need:int=2)->List[int]:
    arr=_parse_seq_list(text)
    return arr[:need]

def _parse_after(text:str)->Optional[int]:
    m=RX_AFTER.search(text or "")
    if not m: return None
    try: return int(m.group(1))
    except: return None

def _parse_paren_pair(text:str, need:int=2)->List[int]:
    m=RX_PAREN.search(text or "")
    if not m: return []
    nums=[int(x) for x in re.findall(r"[1-4]", m.group(1))]
    return nums[:need]

def _parse_closing_number(text:str)->Optional[int]:
    m=RX_CLOSING_NUM.search(text or "")
    if not m: return None
    try: 
        return int(m.group(1)) 
    except: return None
# ------------------------------------------------------
# Rotas básicas
# ------------------------------------------------------
@app.get("/")
async def root():
    return {"ok": True, "service": "GuardiAo Auto Bot", "time": datetime.datetime.utcnow().isoformat()+"Z"}

@app.get("/health")
async def health():
    return {"ok": True, "db_exists": os.path.exists(DB_PATH), "db_path": DB_PATH}

@app.get("/debug_cfg")
async def debug_cfg():
    w1, w2, w3, w4 = _db_get_weights()
    history_count = len(_db_get_detailed_history(n=10000))
    
    return {
        "MAX_GALE": MAX_GALE, 
        "OBS_TIMEOUT_SEC": OBS_TIMEOUT_SEC, 
        "DEDUP_WINDOW_SEC": DEDUP_WINDOW_SEC,
        "CURRENT_WEIGHTS": {"E1": w1, "E2": w2, "E3": w3, "E4": w4},
        "OPTIMIZATION_WINDOW": OPTIMIZATION_WINDOW,
        "CLOSED_SIGNALS_IN_DB": history_count
    }

# ROTA PARA O PASSO 3: OTIMIZAÇÃO MANUAL OU VIA CRON JOB
@app.post("/optimize/{optimization_key}")
async def optimize_endpoint(optimization_key: str):
    result = await _optimize_weights(optimization_key)
    
    # Envia notificação via Telegram se a otimização foi bem-sucedida
    if result.get("status") == "success":
        new_w = result["new_weights"]
        msg = (f"🔄 **OTIMIZAÇÃO DE PESOS CONCLUÍDA!**\n"
               f"📊 Amostra Analisada: {result['details']['count']} sinais\n"
               f"**Novos Pesos:**\n"
               f"• E1 (N-gram): **{new_w['E1']*100:.1f}%**\n"
               f"• E2 (Short): **{new_w['E2']*100:.1f}%**\n"
               f"• E3 (Long): **{new_w['E3']*100:.1f}%**\n"
               f"• E4 (Gemini): **{new_w['E4']*100:.1f}%**")
        await tg_send(TARGET_CHANNEL, msg)
    
    return result

# ------------------------------------------------------
# Webhook principal
# ------------------------------------------------------
@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()
    upd_id = str(data.get("update_id", "")); _mark_processed(upd_id)

    msg = data.get("channel_post") or data.get("message") \
        or data.get("edited_channel_post") or data.get("edited_message") or {}
    chat = msg.get("chat") or {}
    chat_id = str(chat.get("id") or "")
    text = (msg.get("text") or msg.get("caption") or "").strip()

    # filtra fonte se configurado
    if SOURCE_CHANNEL and chat_id and chat_id != SOURCE_CHANNEL:
        return {"ok": True, "skipped": "wrong_source"}

    # 1) ANALISANDO: apenas alimenta memória (com dedupe)
    if RX_ANALISE.search(text):
        if _seen_recent("analise", _dedupe_key(text)):
            return {"ok": True, "skipped": "analise_dupe"}
        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)
        return {"ok": True, "analise_seq": len(seq)}

    # 2) APOSTA ENCERRADA / GREEN / RED (com dedupe + FECHAMENTO)
    if RX_FECHA.search(text) or RX_GREEN.search(text) or RX_RED.search(text):
        if _seen_recent("fechamento", _dedupe_key(text)):
            return {"ok": True, "skipped": "fechamento_dupe"}

        pend=_pending_get()
        if pend:
            suggested=int(pend["suggested"] or 0)
            
            # TENTA EXTRAIR O NÚMERO FINAL DO PARÊNTESE (Corrigido para o formato "(1)")
            observed_result = _parse_closing_number(text)
            
            # (a) Observados do PLACAR: (Mantido para alimentar 'seen' e timeline, caso haja a linha de Sequência)
            need_obs = min(2, MAX_GALE + 1)
            obs_pair = _parse_seq_pair(text, need=need_obs)
            if obs_pair:
                _pending_seen_append(obs_pair, need=need_obs)

            # (b) Memória extra: números finais entre parênteses "(x | y)" -> só para timeline
            extra_tail = _parse_paren_pair(text, need=2)
            if extra_tail:
                _append_seq(extra_tail)
            
            # --- LÓGICA CRÍTICA DE FECHAMENTO PARA MAX_GALE=0 ---
            if MAX_GALE == 0:
                if observed_result is not None:
                    # Se encontramos o número final no parêntese, usamos ele como resultado G0
                    outcome="LOSS"; stage_lbl="G0"
                    if observed_result == suggested: outcome="GREEN"
                    
                    final_seen=str(observed_result)
                    
                    # PASSO 2: ATUALIZA O RESULTADO FINAL NO detailed_history
                    con=_con()
                    pend_id = int(pend["id"])
                    con.execute("UPDATE detailed_history SET result=? WHERE id=?", (observed_result, pend_id))
                    con.commit(); con.close()
                    
                    # Fechamento: 
                    # A chamada a _pending_close é onde o _score_text() é chamado para atualizar o placar.
                    msg_txt=_pending_close(final_seen, outcome, stage_lbl, suggested)
                    if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)
                    return {"ok": True, "closed": outcome, "seen": final_seen}
                else:
                    # AVISO CRÍTICO: Fechamento G0 ignorado se o número final estiver faltando.
                    await tg_send(TARGET_CHANNEL, f"⚠️ **ERRO/AVISO CRÍTICO:** Fechamento G0 IGNORADO. Não encontrou o número final da aposta (1-4) no parêntese. Verifique o formato da mensagem de origem. Texto: '{text}'")
                    return {"ok": True, "waiting_obs_g0_miss": True}

            # LÓGICA COMPLETA
            # Esta seção conteria a lógica para MAX_GALE > 0 (G1, G2, etc.)
            
    # 3) ENTRADA CONFIRMADA: abre o sinal
    if RX_ENTRADA.search(text):
        if _seen_recent("entrada", _dedupe_key(text)):
            return {"ok": True, "skipped": "entrada_dupe"}

        # Se já tiver um sinal pendente, ignora nova entrada
        if _pending_get():
            return {"ok": True, "skipped": "already_open"}

        # --- GERA O PALPITE G0 ---
        number, conf, timeline_size, post_final, gap, reason, p1, p2, p4 = await _choose_number()

        if conf < 0.35: # Mantém o piso de confiança
            return {"ok": True, "skipped": "low_confidence", "conf": conf, "reason": reason}

        # Armazena o sinal no DB (status: 0)
        _pending_open(suggested=number)

        # Salva o detalhe do sinal para otimização futura (Passo 2)
        con=_con(); pend_id = con.execute("SELECT id FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone()["id"]; con.close()
        
        con=_con()
        con.execute("""INSERT INTO detailed_history(id, opened_at, suggested, p1_conf, p2_conf, p3_conf, p4_conf) 
                       VALUES(?,?,?,?,?,?,?)""",
                    (pend_id, int(time.time()), number, post_final.get(1,0), post_final.get(2,0), post_final.get(3,0), post_final.get(4,0)))
        con.commit(); con.close()


        # Mensagem G0
        msg_txt = (f"⚽️ **ENTRADA CONFIRMADA (G0)**\n"
                   f"🎯 Aposta: **{number}**\n"
                   f"📈 Confiança IA: {conf*100:.1f}% (Gap: {gap*100:.1f}%) "
                   f"[{reason}]\n"
                   f"📊 Geral: {_score_text()}")

        # Envia e armazena ID da mensagem para possível edição/exclusão (opcional)
        await tg_send(TARGET_CHANNEL, msg_txt)

        return {"ok": True, "opened": number, "conf": conf, "size": timeline_size}

    # 4) Observação G1 / G2 (caso MAX_GALE > 0)
    if MAX_GALE > 0:
        pend=_pending_get()
        if pend and int(pend["stage"]) < MAX_GALE:
            # Tenta extrair 1 ou 2 números da sequência ou do parêntese
            need_obs = min(2, MAX_GALE + 1)
            obs_pair = _parse_seq_pair(text, need=need_obs) or _parse_paren_pair(text, need=need_obs)
            
            if obs_pair:
                current_seen = pend["seen"].split("-")
                
                # Verifica se a nova observação move o sinal para o próximo Gale
                # A lógica de Gale é complexa e exige saber qual número de fato saiu
                
                # Por simplicidade no G0/foco, não implementaremos a lógica completa de Gale aqui. 
                # Apenas alimentamos o "visto" no DB.
                _pending_seen_append(obs_pair, need=need_obs)

    return {"ok": True, "processed": False}
