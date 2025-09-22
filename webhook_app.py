#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treina um modelo (model.pkl) a partir do SQLite do bot.
Compatível com o extrator de features do runtime (mesmos nomes de colunas).
Usa apenas os casos em que houve fechamento (label preenchido).
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
DEFAULT_MODEL_OUT = os.getenv("MODEL_PATH", "/var/data/model.pkl")

# ========= Feature helpers (espelham o runtime) =========
PATTERN_KEYS = ["GEN","KWOK","SSH","SEQ","ODD","EVEN"]

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
    # A ordem no runtime é 'sorted(keys)'; seguimos igual para compatibilidade
    keys = sorted(feats.keys())
    return keys, [float(feats[k]) for k in keys]

# ========= Data loading =========
def load_data(db_path: str):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # ml_log tem: pending_id, after, base(json), pattern_key, conf_short, conf_long, gap_short, gap_long, samples_short,
    # proba_json (opcional), chosen (cand), label (1 green, 0 loss), stage, outcome
    # pending tem: last_post_short (json dict cand->prob), last_post_long (json), suggested, outcome, seen
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

    X, y = [], []
    feature_names = None

    for r in rows:
        try:
            base = json.loads(r["base"]) if r["base"] else []
            post_s = json.loads(r["last_post_short"]) if r["last_post_short"] else {}
            post_l = json.loads(r["last_post_long"]) if r["last_post_long"] else {}
        except Exception:
            base, post_s, post_l = [], {}, {}

        # usamos APENAS o candidato escolhido (chosen) para o treinamento supervisionado
        cand = int(r["chosen"]) if r["chosen"] is not None else None
        if cand is None:
            continue

        feats0 = build_base_feats(
            after=int(r["after"] or 0),
            base=base,
            pattern_key=r["pattern_key"] or "GEN",
            conf_s=float(r["conf_short"] or 0.0),
            conf_l=float(r["conf_long"] or 0.0),
            gap_s=float(r["gap_short"] or 0.0),
            gap_l=float(r["gap_long"] or 0.0),
            samples_s=int(r["samples_short"] or 0),
        )
        feats = expand_candidate_feats(feats0, cand, post_s, post_l)
        keys, vec = to_feature_vector(feats)
        if feature_names is None:
            feature_names = keys
        X.append(vec)
        y.append(int(r["label"]))

    if not X:
        raise RuntimeError("Nenhum dado rotulado encontrado em ml_log/pending. Deixe o bot operar para gerar rótulos.")

    return np.array(X, dtype=float), np.array(y, dtype=int), feature_names

# ========= Train =========
def train_model(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    # split estratificado
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

    # avaliação
    ypred = clf.predict(Xte)
    yproba = clf.predict_proba(Xte)[:,1]

    print("\n=== Avaliação (holdout) ===")
    print(classification_report(yte, ypred, digits=3))
    try:
        auc = roc_auc_score(yte, yproba)
        print(f"ROC-AUC: {auc:.3f}")
    except Exception:
        pass
    print("Matriz de confusão:")
    print(confusion_matrix(yte, ypred))

    return clf

# ========= Main =========
def main():
    parser = argparse.ArgumentParser(description="Treina model.pkl a partir do SQLite do bot.")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help=f"Caminho para o SQLite (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--out", default=DEFAULT_MODEL_OUT, help=f"Saída do modelo (default: {DEFAULT_MODEL_OUT})")
    args = parser.parse_args()

    print(f"Lendo dados de: {args.db}")
    X, y, feat_names = load_data(args.db)
    print(f"Exemplos: {len(X)} | Dimensão: {X.shape[1]}")

    print("Treinando modelo...")
    model = train_model(X, y)

    # Salvamos também os nomes das features e uma pequena estrutura para checagem
    bundle = {
        "model": model,
        "feature_names_sorted": feat_names,  # para debug (o runtime ordena por chave, garantimos compat)
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(bundle["model"], args.out)
    print(f"✅ Modelo salvo em: {args.out}")

if __name__ == "__main__":
    main()
