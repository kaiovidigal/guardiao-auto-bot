#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GuardiAo Auto Bot — webhook_app.py
v8.0 (Entrada única, sem G0/G1, fechamento estrito, NEUTRO por filtros,
      IA FiboMix dinâmica, dedupe por conteúdo, “Analisando...” auto-delete, DB SQLite)

ENV obrigatórias (Render -> Environment):
- TG_BOT_TOKEN
- WEBHOOK_TOKEN
- SOURCE_CHANNEL           ex: -1002810508717
- TARGET_CHANNEL           ex: -1003052132833

ENV opcionais:
- SHOW_DEBUG           (default False)
- OBS_TIMEOUT_SEC      (default 420)       # janela de espera pelo "fecha" (informativo)
- DEDUP_WINDOW_SEC     (default 40)
- FIBO_WINDOWS         (default "8,13,21,34,55")
- SHOW_TOP3            (default "True")
- MIN_CONF             (default "0.52")
- MIN_GAP              (default "0.06")    # gap = p(top1) - p(top2)
- ENTROPY_CUTOFF       (default "1.35")    # H alto -> cenário uniforme -> NEUTRO
- ABSTAIN_ON_TIMEOUT   (default "True")    # se não vier "fecha" e surgir nova ENTRADA: descarta sem pontuar

Start command:
  uvicorn webhook_app:app --host 0.0.0.0 --port $PORT
"""

import os, re, json, time, math, sqlite3, datetime, hashlib
from typing import List, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException

# ------------------------------------------------------
# ENV & constantes
# ------------------------------------------------------
TG_BOT_TOKEN   = os.getenv("TG_BOT_TOKEN", "").strip()
WEBHOOK_TOKEN  = os.getenv("WEBHOOK_TOKEN", "").strip()
SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "").strip()

SHOW_DEBUG         = os.getenv("SHOW_DEBUG", "False").strip().lower() == "true"
OBS_TIMEOUT_SEC    = int(os.getenv("OBS_TIMEOUT_SEC", "420"))
DEDUP_WINDOW_SEC   = int(os.getenv("DEDUP_WINDOW_SEC", "40"))

def _parse_fibo_env(txt: str) -> List[int]:
    out=[]
    for p in (txt or "").split(","):
        p=p.strip()
        if not p: continue
        try:
            k=int(p)
            if k>0: out.append(k)
        except:
            pass
    return out or [8,13,21,34,55]

FIBO_WINDOWS       = _parse_fibo_env(os.getenv("FIBO_WINDOWS", "8,13,21,34,55"))
SHOW_TOP3          = os.getenv("SHOW_TOP3", "True").strip().lower() == "true"
MIN_CONF           = float(os.getenv("MIN_CONF", "0.52"))
MIN_GAP            = float(os.getenv("MIN_GAP", "0.06"))
ENTROPY_CUTOFF     = float(os.getenv("ENTROPY_CUTOFF", "1.35"))
ABSTAIN_ON_TIMEOUT = os.getenv("ABSTAIN_ON_TIMEOUT", "True").strip().lower() == "true"

# Entrada única (sem gale)
NEED_OBS = 1

if not TG_BOT_TOKEN or not WEBHOOK_TOKEN or not TARGET_CHANNEL:
    raise RuntimeError("Faltam ENV obrigatórias: TG_BOT_TOKEN, WEBHOOK_TOKEN, TARGET_CHANNEL.")
TELEGRAM_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

DB_PATH = "/opt/render/project/src/main.sqlite"

# ------------------------------------------------------
# App
# ------------------------------------------------------
app = FastAPI(title="GuardiAo Auto Bot (webhook)", version="8.0")

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
    if not con.execute("SELECT 1 FROM score WHERE id=1").fetchone():
        con.execute("INSERT INTO score(id,green,loss) VALUES(1,0,0)")
    con.commit(); con.close()

db_init()

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
    try:
        con = _con()
        now = int(time.time())
        con.executemany("INSERT INTO timeline(created_at,number) VALUES(?,?)",[(now,int(x)) for x in seq])
        con.commit(); con.close()
    except Exception:
        pass

def _timeline_size()->int:
    con=_con(); row=con.execute("SELECT COUNT(*) c FROM timeline").fetchone(); con.close()
    return int(row["c"] or 0)

def _score_add(outcome:str):
    try:
        con = _con()
        row = con.execute("SELECT green,loss FROM score WHERE id=1").fetchone()
        g,l = (int(row["green"]), int(row["loss"])) if row else (0,0)
        if outcome.upper()=="GREEN": g+=1
        elif outcome.upper()=="LOSS": l+=1
        con.execute("INSERT OR REPLACE INTO score(id,green,loss) VALUES(1,?,?)",(g,l))
        con.commit(); con.close()
    except Exception:
        pass

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
    con.execute("INSERT INTO pending(created_at,opened_at,suggested,seen,open) VALUES(?,?,?,?,1)",
                (now, now, int(suggested), ""))
    con.commit(); con.close()
    return True

def _pending_seen_append(nums: List[int], need:int=NEED_OBS):
    row = _pending_get()
    if not row: return
    seen = (row["seen"] or "").strip()
    arr = [s for s in seen.split("-") if s]
    for n in nums:
        if len(arr) >= need: break
        arr.append(str(int(n)))
    txt = "-".join(arr[:need])
    con = _con(); con.execute("UPDATE pending SET seen=? WHERE id=?", (txt, int(row["id"]))); con.commit(); con.close()

def _pending_close(final_seen: str, outcome: str, suggested:int)->str:
    row = _pending_get()
    if not row: return ""
    con = _con()
    con.execute("UPDATE pending SET open=0, seen=? WHERE id=?", (final_seen, int(row["id"])))
    con.commit(); con.close()
    _score_add(outcome)
    # alimentar timeline com observados
    obs = [int(x) for x in str(final_seen).split("-") if str(x).isdigit()]
    _append_seq(obs)
    our = suggested if outcome.upper()=="GREEN" else "X"
    snap = _ngram_snapshot(suggested)
    msg = (f"{'🟢' if outcome.upper()=='GREEN' else '🔴'} <b>{outcome.upper()}</b> — finalizado "
           f"(nosso={our}, observado={final_seen}).\n"
           f"📊 Geral: {_score_text()}\n\n{snap}")
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
# IA — FiboMix + Short/Long + filtros (entrada única)
# ------------------------------------------------------
def _norm(d: Dict[int,float])->Dict[int,float]:
    s=sum(d.values()) or 1e-9
    return {k:(v/s) for k,v in d.items()}

def _post_freq(tail:List[int], k:int, alpha:float=1.0)->Dict[int,float]:
    # Dirichlet smoothing (alpha) para estabilidade
    counts={c:alpha for c in (1,2,3,4)}
    if tail:
        win = tail[-k:] if len(tail)>=k else tail
        for x in win:
            if x in counts: counts[x]+=1
    return _norm(counts)

def _weights_for_fibo(ws: List[int])->Dict[int,float]:
    # peso = 1/sqrt(k) (suaviza preferência por janelas curtas sem ignorar as longas)
    raw={k:1.0/math.sqrt(k) for k in ws}
    s=sum(raw.values()) or 1e-9
    return {k:v/s for k,v in raw.items()}

def _post_e1_fibo_mix(tail: List[int], windows: List[int]) -> Dict[int,float]:
    mix={c:0.0 for c in (1,2,3,4)}
    wts=_weights_for_fibo(windows)
    for k, wk in wts.items():
        pk=_post_freq(tail,k,alpha=1.0)
        for c in (1,2,3,4):
            mix[c]+=wk*pk[c]
    return _norm(mix)

def _post_e2_short(tail):  return _post_freq(tail, 60, alpha=1.0)
def _post_e3_long(tail):   return _post_freq(tail, 300, alpha=1.0)
def _post_e4_llm(tail):    return {1:0.25,2:0.25,3:0.25,4:0.25}  # placeholder offline

def _hedge(p1,p2,p3,p4, w=(0.45,0.25,0.20,0.10)):
    cands=(1,2,3,4)
    out={c: w[0]*p1.get(c,0)+w[1]*p2.get(c,0)+w[2]*p3.get(c,0)+w[3]*p4.get(c,0) for c in cands}
    return _norm(out)

def _entropy(p: Dict[int,float])->float:
    eps=1e-12
    return -sum(max(p.get(c,0.0),eps)*math.log(max(p.get(c,0.0),eps)) for c in (1,2,3,4))

def _top_k(post: Dict[int,float], k:int=3)->List[Tuple[int,float]]:
    return sorted(post.items(), key=lambda kv: kv[1], reverse=True)[:k]

def _choose_number()->Tuple[Optional[int],float,int,Dict[int,float],float,str,List[Tuple[int,float]],Dict[int,float],float]:
    """
    Retorna:
      best (ou None se NEUTRO), conf, samples, post, gap, reason, top3, e1mix, entropia
    """
    tail=_timeline_tail(400)
    p1=_post_e1_fibo_mix(tail, FIBO_WINDOWS)
    p2=_post_e2_short(tail)
    p3=_post_e3_long(tail)
    p4=_post_e4_llm(tail)
    base=_hedge(p1,p2,p3,p4)
    # filtros
    rank=sorted(base.items(), key=lambda kv: kv[1], reverse=True)
    best=rank[0][0]; conf=float(rank[0][1])
    gap=float(rank[0][1]-rank[1][1]) if len(rank)>=2 else float(rank[0][1])
    ent=_entropy(base)

    # Regra de NEUTRO (não abre entrada)
    if ent>ENTROPY_CUTOFF or conf<MIN_CONF or gap<MIN_GAP:
        return None, conf, _timeline_size(), base, gap, "NEUTRO", _top_k(base,3), p1, ent

    top3=_top_k(base, 3)
    return best, conf, _timeline_size(), base, gap, "IA", top3, p1, ent

def _snapshot_e1(suggested:Optional[int])->str:
    tail=_timeline_tail(400)
    post=_post_e1_fibo_mix(tail, FIBO_WINDOWS)
    pct=lambda x:f"{x*100:.1f}%"
    p1,p2,p3,p4 = pct(post[1]), pct(post[2]), pct(post[3]), pct(post[4])
    wtxt=",".join(str(k) for k in FIBO_WINDOWS)
    conf=pct(post.get(int(suggested),0.0)) if suggested else "—"
    return (f"📈 Amostra: {_timeline_size()} • Conf(E1): {conf}\n"
            f"🌀 E1(FiboMix {wtxt}): 1 {p1} | 2 {p2} | 3 {p3} | 4 {p4}")

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

# compat (não fecham sozinhos)
RX_FECHA   = re.compile(r"APOSTA\s+ENCERRADA", re.I)
RX_GREEN   = re.compile(r"GREEN|✅", re.I)
RX_RED     = re.compile(r"RED|❌", re.I)

RX_SEQ     = re.compile(r"Sequ[eê]ncia:\s*([^\n\r]+)", re.I)
RX_NUMS    = re.compile(r"[1-4]")
RX_AFTER   = re.compile(r"ap[oó]s\s+o\s+([1-4])", re.I)
RX_PAREN   = re.compile(r"\(([^\)]*)\)\s*$")

# FECHAMENTO ESTRITO: "GREEN/RED (...) fecha"
RX_CLOSE_STRICT = re.compile(
    r"(?:✅\s*)?\b(GREEN|RED)\b.*?\(([1-4])\).*?(?:✅\s*)?fecha\b",
    re.I | re.S
)

def _parse_seq_list(text:str)->List[int]:
    m=RX_SEQ.search(text or "")
    if not m: return []
    return [int(x) for x in RX_NUMS.findall(m.group(1))]

def _parse_after(text:str)->Optional[int]:
    m=RX_AFTER.search(text or "")
    if not m: return None
    try: return int(m.group(1))
    except: return None

def _parse_paren_pair(text:str, need:int=NEED_OBS)->List[int]:
    m=RX_PAREN.search(text or "")
    if not m: return []
    nums=[int(x) for x in re.findall(r"[1-4]", m.group(1))]
    return nums[:need]

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
    return {
        "OBS_TIMEOUT_SEC": OBS_TIMEOUT_SEC,
        "DEDUP_WINDOW_SEC": DEDUP_WINDOW_SEC,
        "FIBO_WINDOWS": FIBO_WINDOWS,
        "SHOW_TOP3": SHOW_TOP3,
        "MIN_CONF": MIN_CONF,
        "MIN_GAP": MIN_GAP,
        "ENTROPY_CUTOFF": ENTROPY_CUTOFF,
        "ABSTAIN_ON_TIMEOUT": ABSTAIN_ON_TIMEOUT
    }

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

    # 1) ANALISANDO: alimenta memória (com dedupe)
    if RX_ANALISE.search(text):
        if _seen_recent("analise", _dedupe_key(text)):
            return {"ok": True, "skipped": "analise_dupe"}
        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)
        return {"ok": True, "analise_seq": len(seq)}

    # 2) FECHAMENTO ESTRITO: somente GREEN/RED (...) fecha
    m_close = RX_CLOSE_STRICT.search(text)
    if m_close:
        if _seen_recent("fechamento", _dedupe_key(text)):
            return {"ok": True, "skipped": "fechamento_dupe"}

        pend=_pending_get()
        if pend:
            suggested=int(pend["suggested"] or 0)

            # observado/hit dentro de parênteses
            hit = None
            try:
                hit = int(m_close.group(2))
            except:
                pass

            # memória extra: números nos parênteses -> timeline
            extra_tail = _parse_paren_pair(text, need=NEED_OBS)
            if extra_tail:
                _append_seq(extra_tail)

            # atualizar 'seen' com o hit (1 observado basta)
            if hit is not None:
                _pending_seen_append([hit], need=NEED_OBS)

            pend=_pending_get()
            seen = [s for s in (pend["seen"] or "").split("-") if s]

            # outcome do próprio texto
            raw_outcome = (m_close.group(1) or "").strip().upper()
            outcome = "GREEN" if raw_outcome == "GREEN" else "LOSS"

            final_seen = seen[0] if seen else (str(hit) if hit is not None else "X")
            msg_txt=_pending_close(str(final_seen), outcome, suggested)
            if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)
            return {"ok": True, "closed": outcome, "seen": str(final_seen)}

        return {"ok": True, "noted_close": True}

    # 3) ENTRADA CONFIRMADA (com dedupe + “Analisando...” auto-delete)
    if RX_ENTRADA.search(text):
        if _seen_recent("entrada", _dedupe_key(text)):
            if SHOW_DEBUG:
                await tg_send(TARGET_CHANNEL, "DEBUG: entrada duplicada ignorada (conteúdo repetido).")
            return {"ok": True, "skipped": "entrada_dupe"}

        seq=_parse_seq_list(text)
        if seq: _append_seq(seq)               # memória
        after = _parse_after(text)             # usado apenas para exibir "após X"

        # fecha pendência anterior esquecida
        pend=_pending_get()
        if pend:
            seen=[s for s in (pend["seen"] or "").split("-") if s]

            # se não houve "fecha" e vamos iniciar outra ENTRADA:
            if ABSTAIN_ON_TIMEOUT and not seen:
                # descarta sem pontuar
                con = _con()
                con.execute("UPDATE pending SET open=0 WHERE id=?", (int(pend["id"]),))
                con.commit(); con.close()
            else:
                # houve observado, fecha normalmente com GREEN/LOSS conforme correspondente
                suggested=int(pend["suggested"] or 0)
                outcome="LOSS"
                if seen and seen[0].isdigit() and int(seen[0])==suggested:
                    outcome="GREEN"
                final_seen = seen[0] if seen else "X"
                msg_txt=_pending_close(str(final_seen), outcome, suggested)
                if msg_txt: await tg_send(TARGET_CHANNEL, msg_txt)

        # “Analisando...” (apaga depois)
        analyzing_id = await tg_send_return(TARGET_CHANNEL, "⏳ Analisando padrão, aguarde...")

        # escolhe nova sugestão (com filtros NEUTRO)
        best, conf, samples, post, gap, reason, top3, e1mix, ent = _choose_number()

        if best is None:
            # NEUTRO — não abre pendência
            def pct(x): return f"{x*100:.2f}%"
            wtxt=",".join(str(k) for k in FIBO_WINDOWS)
            txt=(f"🤖 <b>NEUTRO</b>\n"
                 f"🧩 <b>Padrão:</b> GEN{(' após '+str(after)) if after else ''}\n"
                 f"📊 Conf={pct(conf)} | gap≈{gap*100:.1f}pp | H={ent:.2f} (corte={ENTROPY_CUTOFF})\n"
                 f"🧠 Critérios: conf≥{int(MIN_CONF*100)}% • gap≥{int(MIN_GAP*100)}pp • entropia≤{ENTROPY_CUTOFF}\n")
            if SHOW_TOP3 and top3:
                top3_txt = " | ".join(f"{n} ({pct(p)})" for n,p in top3)
                txt += f"🔬 <b>Estudo 3 números:</b> {top3_txt}\n"
            txt += _snapshot_e1(None)
            await tg_send(TARGET_CHANNEL, txt)

            if analyzing_id is not None:
                await tg_delete(TARGET_CHANNEL, analyzing_id)

            return {"ok": True, "neutral": True, "conf": conf, "gap": gap, "entropy": ent}

        # caso válido: abrir pendência
        opened=_pending_open(best)
        if opened:
            def pct(x): return f"{x*100:.2f}%"
            wtxt=",".join(str(k) for k in FIBO_WINDOWS)
            aft_txt = f" após {after}" if after else ""
            top3_txt = " | ".join(f"{n} ({pct(p)})" for n,p in (top3 or []))
            txt=(f"🤖 <b>IA SUGERE</b> — <b>{best}</b>\n"
                 f"🧩 <b>Padrão:</b> GEN{aft_txt}\n"
                 f"📊 <b>Conf:</b> {pct(conf)} | <b>Amostra≈</b>{samples} | <b>gap≈</b>{gap*100:.1f}pp | H={ent:.2f}\n"
                 f"🧠 <b>Modelo:</b> FiboMix[{wtxt}] + Short/Long\n")
            if SHOW_TOP3 and top3:
                txt += f"🔬 <b>Estudo 3 números:</b> {top3_txt}\n"
            txt += _snapshot_e1(best)
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