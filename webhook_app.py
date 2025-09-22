#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treina **um modelo por padrão** (GEN, KWOK, SSH, SEQ, ODD, EVEN) a partir do SQLite do bot.
Saídas: model_<PADRAO>.pkl (ex.: model_GEN.pkl) em MODEL_DIR (default: /var/data)
Se um padrão tiver poucos exemplos, ele é pulado e o script registra isso no registry.json.
Compatível com o extrator de features do runtime (mesmas chaves e ordem de features).
"""
import os
import json
import argparse
import sqlite3
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# ========= Defaults =========
DEFAULT_DB_PATH   = os.getenv("DB_PATH", "/var/data/data.db")
DEFAULT_MODEL_DIR = os.getenv("MODEL_DIR", "/var/data")
MIN_SAMPLES_PER_PATTERN = int(os.getenv("MIN_SAMPLES_PER_PATTERN", "120"))  # ajuste se necessário

PATTERNS = ["GEN","KWOK","SSH","SEQ","ODD","EVEN"]

# ========= Feature helpers (espelham o runtime) =========
def build_base_feats(after: int, base: List[int], pattern_key: str,
                     conf_s: float, conf_l: float,
                     gap_s: float, gap_l: float,
                     samples_s: int) -> Dict[str, float]:
    pk = (pattern_key or "GEN").upper()
    return {
        "after": float(after or 0),
        "has_after": 1.0 if after else 0.0,
        "conf_short": float(conf_s or 0.0),
        "conf_long":  float(conf_l or 0.0),
        "gap_short":  float(max(0.0, gap_s or 0.0)),
        "gap_long":   float(max(0.0, gap_l or 0.0)),
        "samples_short": float(samples_s or 0),
        "base_len": float(len(base or [])),
        # one-hots por padrão
        "pat_GEN": 1.0 if pk.startswith("GEN") else 0.0,
        "pat_KWOK":1.0 if pk.startswith("KWOK") else 0.0,
        "pat_SSH": 1.0 if pk.startswith("SSH") else 0.0,
        "pat_SEQ": 1.0 if pk.startswith("SEQ") else 0.0,
        "pat_ODD": 1.0 if pk == "ODD" else 0.0,
        "pat_EVEN":1.0 if pk == "EVEN" else 0.0,
    }

def expand_candidate_feats(base_feats: Dict[str,float], cand: int,
                           post_s: Dict[int,float], post_l: Dict[int,float]) -> Dict[str,float]:
    x = dict(base_feats)
    x.update({
        "cand": float(cand),
        "post_s": float(post_s.get(cand, 0.0)),
        "post_l": float(post_l.get(cand, 0.0)),
    })
    return x

def to_feature_vector(feats: Dict[str,float]) -> Tuple[List[str], List[float]]:
    keys = sorted(feats.keys())  # mesma ordem do runtime
    return keys, [float(feats[k]) for k in keys]

# ========= Data loading =========
def load_rows(db_path: str):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT
            m.pending_id, m.after, m.base, m.pattern_key,
            m.conf_short, m.conf_long, m.gap_short, m.gap_long, m.samples_short,
            m.chosen, m.label, m.stage, m.outcome,
            p.last_post_short, p.last_post_long, p.suggested, p.outcome AS p_outcome, p.seen
        FROM ml_log m
        LEFT JOIN pending p ON p.id = m.pending_id
        WHERE m.label IS NOT NULL
    """).fetchall()
    con.close()
    return rows

def rows_to_dataset(rows, pattern_filter: str):
    X, y = [], []
    feat_names = None
    for r in rows:
        pk = (r["pattern_key"] or "GEN").upper()
        # aplica filtro do padrão atual
        if pattern_filter and not pk.startswith(pattern_filter):
            continue

        try:
            base = json.loads(r["base"]) if r["base"] else []
            post_s = json.loads(r["last_post_short"]) if r["last_post_short"] else {}
            post_l = json.loads(r["last_post_long"]) if r["last_post_long"] else {}
        except Exception:
            base, post_s, post_l = [], {}, {}

        cand = int(r["chosen"]) if r["chosen"] is not None else None
        if cand is None:
            continue

        feats0 = build_base_feats(
            after=int(r["after"] or 0),
            base=base,
            pattern_key=pk,
            conf_s=float(r["conf_short"] or 0.0),
            conf_l=float(r["conf_long"] or 0.0),
            gap_s=float(r["gap_short"] or 0.0),
            gap_l=float(r["gap_long"] or 0.0),
            samples_s=int(r["samples_short"] or 0),
        )
        feats = expand_candidate_feats(feats0, cand, post_s, post_l)
        keys, vec = to_feature_vector(feats)
        if feat_names is None:
            feat_names = keys
        X.append(vec)
        y.append(int(r["label"]))
    return np.array(X, dtype=float), np.array(y, dtype=int), feat_names

# ========= Train one =========
def train_one(X, y, random_state=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    yproba = clf.predict_proba(Xte)[:,1]
    metrics = {}
    report = classification_report(yte, ypred, digits=3, output_dict=True)
    metrics["report"] = report
    try:
        metrics["roc_auc"] = float(roc_auc_score(yte, yproba))
    except Exception:
        metrics["roc_auc"] = None
    cm = confusion_matrix(yte, ypred).tolist()
    metrics["confusion_matrix"] = cm
    return clf, metrics

# ========= Main =========
def main():
    parser = argparse.ArgumentParser(description="Treina um modelo por padrão.")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help=f"Caminho para o SQLite (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--outdir", default=DEFAULT_MODEL_DIR, help=f"Diretório de saída dos modelos (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_PATTERN, help="Mínimo de exemplos por padrão para treinar")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = load_rows(args.db)
    registry = {"models": {}, "skipped": {}}

    for pat in PATTERNS:
        X, y, feat_names = rows_to_dataset(rows, pat)
        total = len(X)
        if total < args.min_samples or len(set(y)) < 2:
            registry["skipped"][pat] = {"reason": "insufficient_data", "samples": int(total)}
            print(f"[{pat}] pulado: dados insuficientes (samples={total}).")
            continue

        print(f"
=== Treinando [{pat}] ===")
        print(f"Exemplos: {len(X)} | Dimensão: {X.shape[1]}")
        clf, metrics = train_one(X, y)
        out_path = os.path.join(args.outdir, f"model_{pat}.pkl")
        joblib.dump(clf, out_path)
        print(f"[{pat}] ✅ salvo em: {out_path}")
        registry["models"][pat] = {
            "path": out_path,
            "samples": int(total),
            "metrics": metrics,
            "feature_names_sorted": feat_names,
        }

    # global fallback (todos os padrões juntos), caso queira
    Xg, yg, feat_names_g = rows_to_dataset(rows, pattern_filter="")
    if len(Xg) >= max(200, args.min_samples) and len(set(yg)) >= 2:
        print(f"
=== Treinando [GLOBAL] ===")
        clf_g, metrics_g = train_one(Xg, yg)
        out_path_g = os.path.join(args.outdir, "model_GLOBAL.pkl")
        joblib.dump(clf_g, out_path_g)
        registry["models"]["GLOBAL"] = {
            "path": out_path_g,
            "samples": int(len(Xg)),
            "metrics": metrics_g,
            "feature_names_sorted": feat_names_g,
        }
        print(f"[GLOBAL] ✅ salvo em: {out_path_g}")
    else:
        registry["skipped"]["GLOBAL"] = {"reason": "insufficient_data", "samples": int(len(Xg))}
        print("[GLOBAL] pulado: dados insuficientes.")

    reg_path = os.path.join(args.outdir, "registry.json")
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"
📘 Registry salvo em: {reg_path}")
    print("Pronto!")

if __name__ == "__main__":
    main()
