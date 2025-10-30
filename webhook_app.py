#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GuardiAo Auto Bot — webhook_app.py
v7.8 (INÍCIO OTIMIZAÇÃO - PASSO 2: REGISTRO DETALHADO)

ENV obrigatórias (Render -> Environment):
- TG_BOT_TOKEN
- WEBHOOK_TOKEN
- SOURCE_CHANNEL           ex: -1002810508717
- TARGET_CHANNEL           ex: -1003052132833
- GEMINI_API_KEY           <<< CHAVE OBRIGATÓRIA PARA O GEMINI
- MAX_GALE                 <<< DEVE SER DEFINIDO COMO '0' PARA FOCO G0
...
"""

import os, re, json, time, math, sqlite3, datetime, hashlib
from typing import List, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException

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

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN or not TARGET_CHANNEL:
    raise RuntimeError("Faltam ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN, TARGET_CHANNEL.")
TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

DB_PATH = "/opt/render/project/src/main.sqlite"

# ------------------------------------------------------
# Cliente Gemini (Inicialização)
# ------------------------------------------------------
GEMINI_CLIENT = None
GEMINI_MODEL  = "gemini-2.5-flash" # Modelo rápido para análise de padrões

if GEMINI_API_KEY:
    try:
        # A chamada assíncrona é usada nas funções de predição
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini Client inicializado com sucesso.")
    except Exception as e:
        print(f"AVISO: Falha ao inicializar Gemini Client: {e}. E4 usará 25% fixo.")

# ------------------------------------------------------
# App
# ------------------------------------------------------
app = FastAPI(title="GuardiAo Auto Bot (webhook)", version="7.8")

# ------------------------------------------------------
# DB helpers
# ------------------------------------------------------
def _con():
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
    cur.execute("""CREATE TABLE IF NOT EXISTS score(
        id INTEGER PRIMARY KEY CHECK(id=1),
        green INTEGER DEFAULT 0,
        loss  INTEGER DEFAULT 0
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS dedupe(
        kind TEXT NOT NULL,
        dkey TEXT NOT NULL,
        ts   INTEGER NOT NULL,
        PRIMARY KEY (kind, dkey)
    )""")
    
    # PASSO 1: Tabela para armazenar a configuração dinâmica de pesos
    cur.execute("""CREATE TABLE IF NOT EXISTS config(
        id INTEGER PRIMARY KEY CHECK(id=1),
        w1 REAL DEFAULT 0.30, 
        w2 REAL DEFAULT 0.20, 
        w3 REAL DEFAULT 0.20, 
        w4 REAL DEFAULT 0.30,
        last_opt_ts INTEGER DEFAULT 0
    )""")

    # PASSO 2: Tabela para o histórico detalhado dos palpites (FEEDBACK)
    cur.execute("""CREATE TABLE IF NOT EXISTS detailed_history(
        id INTEGER PRIMARY KEY,         -- Usa o ID da tabela 'pending'
        opened_at INTEGER NOT NULL,
        suggested INTEGER NOT NULL,     -- O número final sugerido
        result    INTEGER,              -- O resultado final (1, 2, 3 ou 4)
        p1_conf   REAL, p2_conf   REAL, -- Confiança do especialista 1 e 2
        p3_conf   REAL, p4_conf   REAL  -- Confiança do especialista 3 e 4 (Gemini)
    )""")

    if not con.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        con.execute("INSERT INTO score(id,green,loss) VALUES(1,0,0)")
        
    # PASSO 1: Insere os pesos iniciais se a tabela estiver vazia
    if not con.execute("SELECT 1 FROM config WHERE id=1").fetchone():
        con.execute("INSERT INTO config(id,w1,w2,w3,w4) VALUES(1,0.30,0.20,0.20,0.30)")
        
    con.commit(); con.close()

db_init()

# PASSO 1: Função para ler os pesos atuais
def _db_get_weights()->Tuple[float,float,float,float]:
    con=_con(); row=con.execute("SELECT w1,w2,w3,w4 FROM config WHERE id=1").fetchone(); con.close()
    return (row["w1"], row["w2"], row["w3"], row["w4"]) if row else (0.30, 0.20, 0.20, 0.30)

# PASSO 1: Função para salvar novos pesos e atualizar timestamp
def _db_save_weights(w:Dict[str,float]):
    con=_con(); now=int(time.time())
    con.execute("UPDATE config SET w1=?,w2=?,w3=?,w4=?,last_opt_ts=? WHERE id=1",
                (w["E1"], w["E2"], w["E3"], w["E4"], now)) 
    con.commit(); con.close()

# PASSO 2: Função para registrar um novo histórico detalhado na abertura da aposta
def _db_save_detailed_entry(entry_id: int, suggested: int, post_raw: Dict[int,float]):
    con=_con(); now=int(time.time())
    # O objeto 'post_raw' é a distribuição de probabilidade final da IA (antes do _conf_floor)
    
    # Extrai a confiança do número sugerido (best) para cada especialista
    # Usamos o _post_eX_llm.last_run (Gemini) e os outros posts (p1, p2, p3)
    p4_post = getattr(_post_e4_llm, 'last_run', {1:0.25,2:0.25,3:0.25,4:0.25})
    
    # ATENÇÃO: Os posts (p1, p2, p3) são calculados na _choose_number. 
    # Para simplificar o Passo 2, vamos usar o post_raw (o resultado da combinação) como proxy
    # para a confiança de todos (p1_conf = post_raw[suggested], etc.). 
    # Para ser mais preciso, o _choose_number precisaria retornar p1, p2, p3, p4 também.
    
    # Por hora, registraremos a confiança do número sugerido (best) na distribuição final 'post_raw'
    # Esta não é a confiança ideal de cada especialista, mas é o suficiente para o Passo 3.
    conf_sug = post_raw.get(suggested, 0.25)
    
    con.execute("INSERT OR IGNORE INTO detailed_history(id,opened_at,suggested,p1_conf,p2_conf,p3_conf,p4_conf) VALUES(?,?,?,?,?,?,?)",
                (entry_id, now, suggested, conf_sug, conf_sug, conf_sug, conf_sug))
    con.commit(); con.close()

# PASSO 2: Adicionamos esta função para ser mais preciso, mas a chamaremos na _choose_number()
def _db_get_detailed_history(n:int=100):
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
    con.execute("INSERT OR REPLACE INTO score(id,green,loss) VALUES(1,0,0)") # Reset temporário
    con.execute("UPDATE score SET green=?,loss=? WHERE id=1",(g,l))
    con.commit(); con.close()

def _score_text()->str:
    con = _con(); row = con.execute("SELECT green,loss FROM score WHERE id=1").fetchone(); con.close()
    if not row: return "0 GREEN × 0 LOSS — 0.0%"
    g,l = int(row["green"]), int(row["loss"])
    tot = g+l; acc = (g/tot*100.0) if tot>0 else 0.0
    return f"{g} GREEN × {l} LOSS — {acc:.1f}%"

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
    our = suggested if outcome.upper()=="GREEN" else "X"
    
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
        if SHOW_DEBUG: print(f"DEBUG Gemini Error: {e}")
        return _post_e4_llm.last_run # Retorna 25% fixo em caso de erro

# ALTERADO: Removemos o argumento 'w' fixo e lemos do DB (PASSO 1)
def _hedge(p1,p2,p3,p4):
    w1, w2, w3, w4 = _db_get_weights() # LÊ OS PESOS DINAMICAMENTE
    w = (w1, w2, w3, w4) 
    cands=(1,2,3,4)
    # A fórmula agora usa os pesos lidos: w[0]=w1, w[1]=w2, etc.
    out={c: w[0]*p1.get(c,0)+w[1]*p2.get(c,0)+w[2]*p3.get(c,0)+w[3]*p4.get(c,0) for c in cands}
    return _norm(out)

def _runnerup_ls2(post:Dict[int,float], loss_streak:int)->Tuple[int,Dict[int,float],str]:
    # Lógica mantida, mas como MAX_GALE=0, a intenção de G1 é ignorada
    rank=sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    best=rank[0][0]
    
    # Mantemos a lógica IA_runnerup_ls2 por segurança, mas ela terá pouco efeito com GALE=0
    if loss_streak>=2 and len(rank)>=2 and (rank[0][1]-rank[1][1])<0.05:
        return rank[1][0], post, "IA_runnerup_ls2"
        
    return best, post, "IA"

def _conf_floor(post:Dict[int,float], floor=0.30, cap=0.95):
    # Lógica mantida para garantir uma confiança mínima/máxima
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
    best, post_raw, reason = _runnerup_ls2(base, loss_streak=_get_ls()) # Renomeado post para post_raw
    
    # ALTERADO: Aumentando o piso de confiança para 35% (0.35)
    post_final=_conf_floor(post_raw, 0.35, 0.95) # O resultado final (após o floor)
    
    best=max(post_final,key=post_final.get); conf=float(post_final[best])
    r=sorted(post_final.items(), key=lambda kv: kv[1], reverse=True)
    gap=(r[0][1]-r[1][1]) if len(r)>=2 else r[0][1]
    
    # Retorna o post processado e agora também os posts individuais (p1, p2, p3, p4) para o Passo 2
    return best, conf, _timeline_size(), post_final, gap, reason, p1, p2, p4

# >>> FUNÇÃO DE SNAPSHOT (Mantida para consistência, mas não usada na mensagem principal) <<<
def _ia_snapshot(suggested:int, post:Dict[int,float], reason:str)->str:
    pct=lambda x:f"{x*100:.1f}%"
    p1_final,p2_final,p3_final,p4_final = post[1], post[2], post[3], post[4]
    conf=pct(post.get(int(suggested),0.0))
    
    # Tenta obter o post cru do E4 (Gemini) para exibir no snapshot
    e4_post = getattr(_post_e4_llm, 'last_run', {1:0.25,2:0.25,3:0.25,4:0.25})
    e4_sug_pct = pct(e4_post.get(int(suggested), 0.0))
    
    return (f"📈 Amostra: {_timeline_size()} • Conf. Final: {conf}\n"
            f"🔎 E1(n-gram): {pct(p1_final)} | E2(short): {pct(p2_final)}\n"
            f"🧠 **E4(Gemini) Sugeriu:** {e4_sug_pct}\n"
            f"💡 **Modo:** {reason}")
    

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
RX_AFTER   = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)

RX_GREEN   = re.compile(r"GREEN|✅", re.I)
RX_RED     = re.compile(r"RED|❌", re.I)
RX_PAREN   = re.compile(r"\(([^\)]*)\)\s*$")

# >>> PARSER CORRIGIDO para ler (X | Y) ou (X) no final da mensagem <<<
# Ele busca o primeiro número (1-4) dentro do último parêntese.
RX_CLOSING_NUM = re.compile(r"\(([^\)]*?)([1-4]).*?\)\s*$", re.I)

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
        # m.group(2) é o primeiro número [1-4] capturado dentro do parêntese.
        return int(m.group(2)) 
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
    return {"MAX_GALE": MAX_GALE, "OBS_TIMEOUT_SEC": OBS_TIMEOUT_SEC, "DEDUP_WINDOW_SEC": DEDUP_WINDOW_SEC}

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
        if SHOW_DEBUG:
            await tg_send(TARGET_CHANNEL, f"DEBUG: Ignorando chat {chat_id}. Fonte esperada: {SOURCE_CHANNEL}")
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
            
            # NOVO: TENTA EXTRAIR O NÚMERO FINAL DO PARÊNTESE (Ex: GREEN!!! (3 | 4))
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
                    
                    # Usa o resultado observado como final_seen
                    final_seen=str(observed_result)
                    
                    # PASSO 2: ATUALIZA O RESULTADO FINAL NO detailed_history
                    con=_con()
                    con.execute("UPDATE detailed_history SET result=? WHERE id=?", (observed_result, int(pend["id"])))
                    con.commit(); con.close()
                    
                    # Fechamento: (já alimenta o timeline com final_seen e adiciona o score)
                    msg_txt=_pending_close(final_seen, outcome, stage_lbl, suggested)
                    if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)
                    return {"ok": True, "closed": outcome, "seen": final_seen}
                else:
                    # AVISO CRÍTICO: Fechamento G0 ignorado se o número final estiver faltando.
                    await tg_send(TARGET_CHANNEL, f"⚠️ **ERRO/AVISO CRÍTICO:** Fechamento G0 IGNORADO. Não encontrou o número final da aposta (1-4) no parêntese. Verifique o formato da mensagem de origem. Texto: '{text}'")
                    return {"ok": True, "waiting_obs_g0_miss": True}

            # LÓGICA COMPLETA PARA MAX_GALE > 0 (caso você mude a ENV) - SEM ALTERAÇÕES AQUI
            else:
                # Lógica completa de GALE > 0 é mantida aqui (porém ignorada se MAX_GALE=0)
                
                pend=_pending_get()
                seen = [s for s in (pend["seen"] or "").split("-") if s]
                
                if len(seen)==1 and seen[0].isdigit() and int(seen[0]) != suggested and MAX_GALE>=1:
                    if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, f"DEBUG: aguardando G1 (visto G0={seen[0]}, nosso={suggested}).")
                    return {"ok": True, "waiting_g1": True, "seen": "-".join(seen)}

                outcome="LOSS"; stage_lbl="G1"
                if len(seen)>=1 and seen[0].isdigit() and int(seen[0])==suggested:
                    outcome="GREEN"; stage_lbl="G0"
                elif len(seen)>=2 and seen[1].isdigit() and int(seen[1])==suggested and MAX_GALE>=1:
                    outcome="GREEN"; stage_lbl="G1"

                if stage_lbl=="G0" or len(seen)>=min(2, MAX_GALE+1):
                    final_seen="-".join(seen[:min(2, MAX_GALE+1)]) if seen else "X"
                    
                    # Aqui, a lógica de GALE > 0 não sabe o resultado exato (1, 2, 3 ou 4) do fechamento
                    # O Passo 2 no GALE > 0 é mais complexo, por enquanto, apenas feche a aposta 'pending'
                    # e use o 'observed_result' (se existir) para o detailed_history
                    if observed_result is not None:
                        con=_con()
                        con.execute("UPDATE detailed_history SET result=? WHERE id=?", (observed_result, int(pend["id"])))
                        con.commit(); con.close()
                    
                    msg_txt=_pending_close(final_seen, outcome, stage_lbl, suggested)
                    if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)
                    return {"ok": True, "closed": outcome, "seen": final_seen}
                else:
                    return {"ok": True, "waiting_more_obs": True, "seen": "-".join(seen)}


        return {"ok": True, "noted_close": True}

    # 3) ENTRADA CONFIRMADA (com dedupe + “Analisando...” auto-delete)
    if RX_ENTRADA.search(text):
        if _seen_recent("entrada", _dedupe_key(text)):
            # Debug de entrada duplicada removido
            return {"ok": True, "skipped": "entrada_dupe"}

        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)               # memória
        after = _parse_after(text)             # usado apenas para exibir "após X"

        # fecha pendência anterior se esquecida (com X)
        pend=_pending_get()
        if pend:
            # Lógica de fechamento forçado aqui é a mesma do FECHA, mas garantindo que feche com X
            if MAX_GALE == 0:
                final_seen="X"; suggested=int(pend["suggested"] or 0); outcome="LOSS"; stage_lbl="G0"
            else:
                seen=[s for s in (pend["seen"] or "").split("-") if s]
                while len(seen)<min(2,MAX_GALE+1): seen.append("X")
                final_seen="-".join(seen[:min(2,MAX_GALE+1)]); suggested=int(pend["suggested"] or 0)
                outcome="LOSS"; stage_lbl="G1"
                if len(seen)>=1 and seen[0].isdigit() and int(seen[0])==suggested: outcome="GREEN"; stage_lbl="G0"
                elif len(seen)>=2 and seen[1].isdigit() and int(seen[1])==suggested and MAX_GALE>=1: outcome="GREEN"; stage_lbl="G1"
            
            # Quando aposta é fechada à força, o resultado no detailed_history fica como NULL
            msg_txt=_pending_close(final_seen, outcome, stage_lbl, suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)

        # “Analisando...” (apaga depois)
        analyzing_id = await tg_send_return(TARGET_CHANNEL, "⏳ Analisando padrão (com Gemini), aguarde...")

        # escolhe nova sugestão (ASSÍNCRONO!)
        # ALTERADO: A função agora retorna os posts individuais (p1, p2, p4) para o Passo 2
        best, conf, samples, post_final, gap, reason, p1, p2, p4 = await _choose_number()
        
        opened=_pending_open(best)
        if opened:
            aft_txt = f"{after}" if after else "X"
            
            # NOVO: Captura a ID da aposta aberta para o Passo 2
            row = _pending_get()
            entry_id = int(row["id"]) if row else None 
            
            # PASSO 2: Registra o palpite detalhado ANTES de enviar a mensagem
            if entry_id:
                # O post_raw (base) foi renomeado para 'base' na _choose_number. 
                # Precisamos da confiança de CADA especialista (p1, p2, p3, p4) para o número sugerido ('best')
                
                # Para maior precisão no Passo 2, vamos registrar a confiança de 'best' em cada especialista:
                conf_p1 = p1.get(best, 0.25)
                conf_p2 = p2.get(best, 0.25)
                conf_p3 = _post_e3_long(_timeline_tail(400)).get(best, 0.25) # Re-calculando p3 para simplificar
                conf_p4 = p4.get(best, 0.25) # p4 já está limpo
                
                con=_con(); now=int(time.time())
                con.execute("INSERT OR IGNORE INTO detailed_history(id,opened_at,suggested,p1_conf,p2_conf,p3_conf,p4_conf) VALUES(?,?,?,?,?,?,?)",
                            (entry_id, now, best, conf_p1, conf_p2, conf_p3, conf_p4))
                con.commit(); con.close()


            # --- NOVO MODELO DE MENSAGEM (Modelo 2: Foco em Ação e Risco) ---
            
            pct=lambda x:f"{x*100:.1f}"
            
            # Tenta obter o post cru do E4 (Gemini) para exibir.
            e4_post = getattr(_post_e4_llm, 'last_run', {1:0.25,2:0.25,3:0.25,4:0.25})
            e4_sug_pct = pct(e4_post.get(int(best), 0.0))
            
            # A mensagem reformulada (Modelo 2)
            txt=(f"🚀 **ENTRADA IMEDIATA:** Número **{best}** (G0)\n"
                 f"🌟 **CONFIANÇA** — **{conf*100:.1f}%**\n"
                 f"🧠 **IA Leader:** Gemini ({e4_sug_pct}%)\n"
                 f"📊 **Base:** Padrão {aft_txt} | Amostra ≈{samples}\n"
                 f"💡 **Modo:** {reason} (Gap {gap*100:.1f}pp)")
            
            # --- FIM DO NOVO MODELO DE MENSAGEM ---
            
            await tg_send(TARGET_CHANNEL, txt)

            if analyzing_id is not None:
                await tg_delete(TARGET_CHANNEL, analyzing_id)

            return {"ok": True, "entry_opened": True, "best": best, "conf": conf}
        else:
            if analyzing_id is not None:
                await tg_delete(TARGET_CHANNEL, analyzing_id)
            if SHOW_DEBUG:
                await tg_send(TARGET_CHANNEL, "DEBUG: pending já aberto; entrada ignorada.")
            return {"ok": True, "skipped": "pending_open"}

    # Não reconhecido
    if SHOW_DEBUG:
        await tg_send(TARGET_CHANNEL, "DEBUG: Mensagem não reconhecida como ENTRADA/FECHAMENTO/ANALISANDO.")
    return {"ok": True, "skipped": "unmatched"}
