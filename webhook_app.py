#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GuardiAo Auto Bot — webhook_app.py
v7.0-prob-combos (G0-only destravado + prob no texto + fechamento estrito por ())
+ Fibo10, Combo2, Combo3 e Fibo50 (cópia)

- Fechamento IMEDIATO (G0) lendo o ÚLTIMO número 1..4 dentro do ÚLTIMO par "()" no FINAL da mensagem do canal-fonte.
  Exemplos que FECHAM: "✅ GREEN (3)" | "RED ❌ (4)"  (e nada depois do ")").
- “Destravado”: fluxo rápido (MAX_GALE=0, SEND_ANALYZING=False recomendados via ENV).
- Sugestão inclui as probabilidades por dígito (1..4).
- Ensemble leve: n-gram curto, janelas (60/300), Fibo10, Combo2, Combo3 e Fibo50.

ENV sugeridas p/ destravar (no painel):
  SHOW_DEBUG=True
  MAX_GALE=0
  OBS_TIMEOUT_SEC=180
  DEDUP_WINDOW_SEC=10
  SEND_ANALYZING=False

Start:
  uvicorn webhook_app:app --host 0.0.0.0 --port $PORT
"""

import os, re, json, time, math, sqlite3, datetime, hashlib
from typing import List, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException

# ================= ENV =================
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "").strip()

SHOW_DEBUG       = os.getenv("SHOW_DEBUG", "False").strip().lower() == "true"
MAX_GALE         = int(os.getenv("MAX_GALE", "0"))           # destravado: G0-only por padrão
OBS_TIMEOUT_SEC  = int(os.getenv("OBS_TIMEOUT_SEC", "180"))  # curto p/ não travar
DEDUP_WINDOW_SEC = int(os.getenv("DEDUP_WINDOW_SEC", "10"))  # curto p/ não travar
SEND_ANALYZING   = os.getenv("SEND_ANALYZING", "False").strip().lower() == "true"

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN or not TARGET_CHANNEL:
    raise RuntimeError("Faltam ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN, TARGET_CHANNEL.")
TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"
DB_PATH = "/opt/render/project/src/main.sqlite"

# ================= APP =================
app = FastAPI(title="GuardiAo Auto Bot (webhook)", version="7.0-prob-combos")

# ================= DB =================
def _con():
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con

def db_init():
    con=_con(); cur=con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS processed(
        update_id TEXT PRIMARY KEY, seen_at INTEGER NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS timeline(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL, number INTEGER NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pending(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at INTEGER, opened_at INTEGER,
        suggested INTEGER, stage INTEGER DEFAULT 0,
        seen TEXT DEFAULT '', open INTEGER DEFAULT 1)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS score(
        id INTEGER PRIMARY KEY CHECK(id=1),
        green INTEGER DEFAULT 0, loss INTEGER DEFAULT 0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS dedupe(
        kind TEXT NOT NULL, dkey TEXT NOT NULL, ts INTEGER NOT NULL,
        PRIMARY KEY (kind,dkey))""")
    if not con.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        con.execute("INSERT INTO score(id,green,loss) VALUES(1,0,0)")
    con.commit(); con.close()
db_init()

def _mark_processed(upd: str):
    try:
        con=_con()
        con.execute("INSERT OR IGNORE INTO processed(update_id,seen_at) VALUES(?,?)",(str(upd),int(time.time())))
        con.commit(); con.close()
    except Exception: pass

# ===== timeline =====
def _timeline_tail(n:int=400)->List[int]:
    con=_con()
    rows=con.execute("SELECT number FROM timeline ORDER BY id DESC LIMIT ?",(n,)).fetchall()
    con.close()
    return [int(r["number"]) for r in rows][::-1]

def _append_seq(seq: List[int]):
    if not seq: return
    con=_con(); now=int(time.time())
    con.executemany("INSERT INTO timeline(created_at,number) VALUES(?,?)",[(now,int(x)) for x in seq])
    con.commit(); con.close()

def _timeline_size()->int:
    con=_con(); row=con.execute("SELECT COUNT(*) c FROM timeline").fetchone(); con.close()
    return int(row["c"] or 0)

# ===== score =====
def _score_add(outcome:str):
    con=_con()
    row=con.execute("SELECT green,loss FROM score WHERE id=1").fetchone()
    g,l=(int(row["green"]),int(row["loss"])) if row else (0,0)
    if outcome.upper()=="GREEN": g+=1
    elif outcome.upper()=="LOSS": l+=1
    con.execute("INSERT OR REPLACE INTO score(id,green,loss) VALUES(1,?,?)",(g,l))
    con.commit(); con.close()

def _score_text()->str:
    con=_con(); row=con.execute("SELECT green,loss FROM score WHERE id=1").fetchone(); con.close()
    g,l=(int(row["green"]),int(row["loss"])) if row else (0,0)
    tot=g+l; acc=(g/tot*100.0) if tot>0 else 0.0
    return f"{g} GREEN × {l} LOSS — {acc:.1f}%"

# ===== pending =====
def _pending_get()->Optional[sqlite3.Row]:
    con=_con(); row=con.execute("SELECT * FROM pending WHERE open=1 ORDER BY id DESC LIMIT 1").fetchone(); con.close()
    return row

def _pending_open(suggested:int):
    if _pending_get(): return False
    con=_con(); now=int(time.time())
    con.execute("INSERT INTO pending(created_at,opened_at,suggested,stage,seen,open) VALUES(?,?,?,?,?,1)",
                (now, now, int(suggested), 0, ""))
    con.commit(); con.close(); return True

def _pending_seen_set(v:str):
    row=_pending_get()
    if not row: return
    con=_con(); con.execute("UPDATE pending SET seen=? WHERE id=?", (v, int(row["id"]))); con.commit(); con.close()

def _pending_seen_append(nums: List[int], need:int=1):
    row=_pending_get()
    if not row: return
    seen=(row["seen"] or "").strip()
    arr=[s for s in seen.split("-") if s]
    for n in nums:
        if len(arr) >= need: break
        arr.append(str(int(n)))
    txt="-".join(arr[:need])
    con=_con(); con.execute("UPDATE pending SET seen=? WHERE id=?", (txt, int(row["id"]))); con.commit(); con.close()

def _pending_close(final_seen: str, outcome: str, stage_lbl: str, suggested:int)->str:
    row=_pending_get()
    if not row: return ""
    con=_con()
    con.execute("UPDATE pending SET open=0, seen=? WHERE id=?", (final_seen, int(row["id"])))
    con.commit(); con.close()
    _score_add(outcome)
    obs=[int(x) for x in final_seen.split("-") if x.isdigit()]
    _append_seq(obs)
    our=suggested if outcome.upper()=="GREEN" else "X"
    return (f"{'🟢' if outcome.upper()=='GREEN' else '🔴'} <b>{outcome.upper()}</b> — finalizado "
            f"(<b>{stage_lbl}</b>, nosso={our}, observados={final_seen}).\n"
            f"📊 Geral: {_score_text()}\n\n{_ngram_snapshot(suggested)}")

# ===== dedupe =====
def _dedupe_key(text:str)->str:
    base=re.sub(r"\s+"," ",(text or "")).strip().lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _seen_recent(kind:str,dkey:str)->bool:
    now=int(time.time()); con=_con()
    row=con.execute("SELECT ts FROM dedupe WHERE kind=? AND dkey=?", (kind,dkey)).fetchone()
    if row and now-int(row["ts"])<= DEDUP_WINDOW_SEC:
        con.close(); return True
    con.execute("INSERT OR REPLACE INTO dedupe(kind,dkey,ts) VALUES (?,?,?)",(kind,dkey,now))
    con.commit(); con.close(); return False

# ================= IA compacta (com prob + Fibo/Combos) =================
CANDS=(1,2,3,4)

def _norm(d: Dict[int,float])->Dict[int,float]:
    s=sum(d.values()) or 1e-9
    return {k:max(0.0,v)/s for k,v in d.items()}

def _freq(tail:List[int], k:int)->Dict[int,float]:
    if not tail: return {c:0.25 for c in CANDS}
    win = tail[-k:] if len(tail)>=k else tail
    tot=max(1,len(win))
    return _norm({c:win.count(c)/tot for c in CANDS})

def _post_e1_ngram(tail:List[int])->Dict[int,float]:
    mix={c:0.0 for c in CANDS}
    for k,w in ((8,0.25),(21,0.35),(55,0.40)):
        pk=_freq(tail,k)
        for c in CANDS: mix[c]+=w*pk[c]
    return _norm(mix)

def _post_e2_short(tail):  return _freq(tail, 60)
def _post_e3_long(tail):   return _freq(tail, 300)

# ---------- Fibo10 ----------
def _fibo10(tail: List[int]) -> Dict[int, float]:
    if not tail: return {c:0.25 for c in CANDS}
    wins=(10,21,34,55)
    mix={c:0.0 for c in CANDS}
    for k in wins:
        win=tail[-k:] if len(tail)>=k else tail
        best=max(CANDS, key=lambda c: win.count(c))
        mix[best]+=1.0
    return _norm(mix)

# ---------- Combinação 2 dígitos ----------
def _combo2(tail: List[int]) -> Dict[int, float]:
    if len(tail) < 2: return {c:0.25 for c in CANDS}
    a,b=tail[-2],tail[-1]
    counts={c:1.0 for c in CANDS}   # Laplace +1
    for i in range(len(tail)-2):
        if tail[i]==a and tail[i+1]==b:
            nxt=tail[i+2]
            if nxt in CANDS: counts[nxt]+=1.0
    return _norm(counts)

# ---------- Combinação 3 dígitos ----------
def _combo3(tail: List[int]) -> Dict[int, float]:
    if len(tail) < 3: return {c:0.25 for c in CANDS}
    a,b,c=tail[-3],tail[-2],tail[-1]
    counts={x:1.0 for x in CANDS}   # Laplace +1
    for i in range(len(tail)-3):
        if tail[i]==a and tail[i+1]==b and tail[i+2]==c:
            nxt=tail[i+3]
            if nxt in CANDS: counts[nxt]+=1.0
    return _norm(counts)

# ---------- Fibo 50 (cópia/leitura por “copie”) ----------
def _fibo50(tail: List[int]) -> Dict[int, float]:
    # leitura simples da frequência na janela 50 (estável e barata)
    return _freq(tail, 50)

def _conf_floor(post:Dict[int,float], floor=0.30, cap=0.95):
    post=_norm({c:float(post.get(c,0)) for c in CANDS})
    b=max(post,key=post.get); mx=post[b]
    if mx<floor:
        others=[c for c in CANDS if c!=b]
        s=sum(post[c] for c in others); take=min(floor-mx,s)
        if s>0:
            scale=(s-take)/s
            for c in others: post[c]*=scale
        post[b]=min(cap,mx+take)
    if post[b]>cap:
        ex=post[b]-cap; post[b]=cap
        add=ex/3.0
        for c in CANDS:
            if c!=b: post[c]+=add
    return _norm(post)

def _choose_number()->Tuple[int,float,int,Dict[int,float],float,str]:
    tail=_timeline_tail(400)

    p_ng  = _post_e1_ngram(tail)
    p_s60 = _post_e2_short(tail)
    p_l300= _post_e3_long(tail)
    p_fb10= _fibo10(tail)
    p_c2  = _combo2(tail)
    p_c3  = _combo3(tail)
    p_f50 = _fibo50(tail)

    # pesos leve/balanceados (somam 1):
    W_NG   = 0.18
    W_S60  = 0.18
    W_L300 = 0.09
    W_FB10 = 0.20
    W_C2   = 0.18
    W_C3   = 0.12
    W_F50  = 0.05

    post={}
    for c in CANDS:
        post[c] = (W_NG  * p_ng.get(c,0)   +
                   W_S60 * p_s60.get(c,0)  +
                   W_L300* p_l300.get(c,0) +
                   W_FB10* p_fb10.get(c,0) +
                   W_C2  * p_c2.get(c,0)   +
                   W_C3  * p_c3.get(c,0)   +
                   W_F50 * p_f50.get(c,0))
    post=_conf_floor(_norm(post), 0.30, 0.95)

    best=max(post,key=post.get); conf=float(post[best])
    r=sorted(post.items(), key=lambda kv: kv[1], reverse=True)
    gap=(r[0][1]-r[1][1]) if len(r)>=2 else r[0][1]
    reason="ngram+win(60/300)+fibo10+combo2/3+fibo50"
    return best, conf, _timeline_size(), post, gap, reason

def _fmt_probs(post:Dict[int,float])->str:
    pct=lambda p: f"{p*100:.1f}%"
    return f"1 {pct(post.get(1,0))} | 2 {pct(post.get(2,0))} | 3 {pct(post.get(3,0))} | 4 {pct(post.get(4,0))}"

def _ngram_snapshot(suggested:int)->str:
    tail=_timeline_tail(400)
    post=_post_e1_ngram(tail)
    pct=lambda x:f"{x*100:.1f}%"
    p1,p2,p3,p4 = pct(post[1]), pct(post[2]), pct(post[3]), pct(post[4])
    conf=pct(post.get(int(suggested),0.0))
    return (f"📈 Amostra: {_timeline_size()} • Conf(n-gram sobre {suggested}): {conf}\n"
            f"🔎 n-gram: 1 {p1} | 2 {p2} | 3 {p3} | 4 {p4}")

# =============== Telegram helpers ===============
async def tg_send(chat_id: str, text: str, parse="HTML"):
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            await cli.post(f"{TELEGRAM_API}/sendMessage",
                           json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                 "disable_web_page_preview": True})
    except Exception: pass

async def tg_send_return(chat_id: str, text: str, parse="HTML") -> Optional[int]:
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            r=await cli.post(f"{TELEGRAM_API}/sendMessage",
                             json={"chat_id": chat_id, "text": text, "parse_mode": parse,
                                   "disable_web_page_preview": True})
            data=r.json()
            if isinstance(data,dict) and data.get("ok") and data.get("result",{}).get("message_id"):
                return int(data["result"]["message_id"])
    except Exception: pass
    return None

async def tg_delete(chat_id: str, message_id: int):
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            await cli.post(f"{TELEGRAM_API}/deleteMessage",
                           json={"chat_id": chat_id, "message_id": int(message_id)})
    except Exception: pass

# =============== Parser canal-fonte ===============
RX_ENTRADA = re.compile(r"ENTRADA\s+CONFIRMADA|ENTRADA\s*OK", re.I)
RX_ANALISE = re.compile(r"\bANALIS(A|Á)NDO\b|ANALISANDO|🧩", re.I)
RX_FECHA   = re.compile(r"APOSTA\s+ENCERRADA|RESULTADO|GREEN|RED|✅|❌|🟢|🔴", re.I)

RX_SEQ     = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
RX_NUMS    = re.compile(r"[1-4]")
RX_AFTER   = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)

# FECHAMENTO ESTRITO: pega o ÚLTIMO "( ... )" no FINAL da mensagem
RX_LAST_PAREN_STRICT = re.compile(
    r"\(\s*([^)]*?)\s*\)\s*(?:✅|❌|GREEN|RED|🟢|🔴)?\s*$",
    re.I | re.M
)

def _strip_noise(s: str) -> str:
    return (s or "").replace("\u200d","").replace("\u200c","").replace("\ufe0f","").strip()

def _extract_last_digit_1_4(chunk: str) -> Optional[int]:
    nums = re.findall(r"[1-4]", chunk or "")
    return int(nums[-1]) if nums else None

def _parse_seq_list(text:str)->List[int]:
    m=RX_SEQ.search(text or "")
    if not m: return []
    return [int(x) for x in RX_NUMS.findall(m.group(1))]

def _parse_after(text:str)->Optional[int]:
    m=RX_AFTER.search(text or "")
    if not m: return None
    try: return int(m.group(1))
    except: return None

def _parse_close_digit_strict(text: str) -> Optional[int]:
    t=_strip_noise(text)
    m=RX_LAST_PAREN_STRICT.search(t)
    if not m: return None
    return _extract_last_digit_1_4(m.group(1))

# ================= Rotas =================
@app.get("/")
async def root():
    return {"ok": True, "service": "GuardiAo Auto Bot", "time": datetime.datetime.utcnow().isoformat()+"Z"}

@app.get("/health")
async def health():
    return {"ok": True, "db_exists": os.path.exists(DB_PATH), "db_path": DB_PATH}

@app.get("/debug_cfg")
async def debug_cfg():
    return {"MAX_GALE": MAX_GALE, "OBS_TIMEOUT_SEC": OBS_TIMEOUT_SEC, "DEDUP_WINDOW_SEC": DEDUP_WINDOW_SEC}

# ================= Webhook =================
@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if token != WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data=await request.json()
    upd_id = str(data.get("update_id","")); _mark_processed(upd_id)

    msg = data.get("channel_post") or data.get("message") \
        or data.get("edited_channel_post") or data.get("edited_message") or {}
    chat=msg.get("chat") or {}
    chat_id = str(chat.get("id") or "")
    text = (msg.get("text") or msg.get("caption") or "").strip()

    # filtra fonte
    if SOURCE_CHANNEL and chat_id and chat_id != SOURCE_CHANNEL:
        if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, f"DEBUG: Ignorando chat {chat_id}. Fonte esperada: {SOURCE_CHANNEL}")
        return {"ok": True, "skipped": "wrong_source"}

    # ANALISANDO → só memória
    if RX_ANALISE.search(text):
        if _seen_recent("analise", _dedupe_key(text)):  # sem flood
            return {"ok": True, "skipped": "analise_dupe"}
        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)
        return {"ok": True, "analise_seq": len(seq)}

    # FECHAMENTO → fecha imediato por () estrito
    if RX_FECHA.search(text):
        if _seen_recent("fechamento", _dedupe_key(text)):
            return {"ok": True, "skipped": "fechamento_dupe"}

        pend=_pending_get()
        if pend:
            suggested=int(pend["suggested"] or 0)

            obs=_parse_close_digit_strict(text)
            if obs is not None:
                _pending_seen_append([obs], need=1)

            seen_txt = (_pending_get()["seen"] or "").strip()
            final_seen = seen_txt if seen_txt else "X"
            outcome = "GREEN" if (seen_txt.isdigit() and int(seen_txt)==suggested) else "LOSS"
            stage_lbl = "G0"

            msg_txt=_pending_close(final_seen, outcome, stage_lbl, suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)

            if SHOW_DEBUG:
                await tg_send(TARGET_CHANNEL, f"DEBUG: fechamento(() estrito) -> {obs} | seen='{final_seen}' | nosso={suggested}")

            return {"ok": True, "closed": outcome, "seen": final_seen}

        if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, "DEBUG: Fechamento reconhecido — sem pendência.")
        return {"ok": True, "noted_close": True}

    # ENTRADA CONFIRMADA
    if RX_ENTRADA.search(text):
        if _seen_recent("entrada", _dedupe_key(text)):
            if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, "DEBUG: entrada duplicada ignorada.")
            return {"ok": True, "skipped": "entrada_dupe"}

        # memória auxiliar (se vier “Sequência:”)
        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)
        after=_parse_after(text)

        # fecha pendência esquecida (X)
        pend=_pending_get()
        if pend:
            seen=(pend["seen"] or "").strip()
            final_seen = seen if seen else "X"
            suggested=int(pend["suggested"] or 0)
            outcome="GREEN" if (seen.isdigit() and int(seen)==suggested) else "LOSS"
            msg_txt=_pending_close(final_seen, outcome, "G0", suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)

        # “Analisando…” opcional
        analyzing_id=None
        if SEND_ANALYZING:
            analyzing_id=await tg_send_return(TARGET_CHANNEL, "⏳ Analisando padrão, aguarde...")

        # decide
        best, conf, samples, post, gap, reason = _choose_number()
        opened=_pending_open(best)
        if opened:
            aft_txt=f" após {after}" if after else ""
            probs_txt=_fmt_probs(post)
            txt=(f"🤖 <b>IA SUGERE</b> — <b>{best}</b>\n"
                 f"📊 <b>Prob:</b> {probs_txt}\n"
                 f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
                 f"📈 <b>Amostra≈</b>{samples} | <b>gap≈</b>{gap*100:.1f}pp\n"
                 f"🧠 <b>Modo:</b> {reason}\n"
                 f"{_ngram_snapshot(best)}")
            await tg_send(TARGET_CHANNEL, txt)

            if analyzing_id is not None:
                await tg_delete(TARGET_CHANNEL, analyzing_id)

            return {"ok": True, "entry_opened": True, "best": best, "conf": conf}
        else:
            if analyzing_id is not None:
                await tg_delete(TARGET_CHANNEL, analyzing_id)
            if SHOW_DEBUG: await tg_send(TARGET_CHANNEL, "DEBUG: pending já aberto; entrada ignorada.")
            return {"ok": True, "skipped": "pending_open"}

    # não reconhecido
    if SHOW_DEBUG:
        await tg_send(TARGET_CHANNEL, "DEBUG: Mensagem não reconhecida como ENTRADA/FECHAMENTO/ANALISANDO.")
    return {"ok": True, "skipped": "unmatched"}