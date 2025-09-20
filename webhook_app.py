# train_ml.py
import os, json, sqlite3, pickle
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

DB_PATH = os.getenv("DB_PATH", "/var/data/data.db")
OUT = os.getenv("MODEL_PATH", "/var/data/model.pkl")

con = sqlite3.connect(DB_PATH); con.row_factory = sqlite3.Row
rows = con.execute("""
    SELECT created_at, after, base, pattern_key, conf_short, conf_long,
           gap_short, gap_long, samples_short, proba_json, chosen, label
      FROM ml_log
     WHERE label IS NOT NULL
     ORDER BY created_at ASC
""").fetchall()
con.close()

X, y = [], []
for r in rows:
    base = json.loads(r["base"] or "[]")
    proba = json.loads(r["proba_json"] or "{}")
    chosen = int(r["chosen"] or 0)
    label  = int(r["label"] or 0)

    # mesmas features do build_features_for_candidates + por candidato escolhido
    feats = {
        "after": float(r["after"] or 0), "has_after": 1.0 if (r["after"] or 0) else 0.0,
        "conf_short": float(r["conf_short"] or 0.0), "conf_long": float(r["conf_long"] or 0.0),
        "gap_short": float(r["gap_short"] or 0.0), "gap_long": float(r["gap_long"] or 0.0),
        "samples_short": float(r["samples_short"] or 0.0),
        "base_len": float(len(base)),
        "pat_GEN": 1.0 if (r["pattern_key"] or "").upper().startswith("GEN") else 0.0,
        "pat_KWOK": 1.0 if (r["pattern_key"] or "").upper().startswith("KWOK") else 0.0,
        "pat_SSH":  1.0 if (r["pattern_key"] or "").upper().startswith("SSH") else 0.0,
        "pat_SEQ":  1.0 if (r["pattern_key"] or "").upper().startswith("SEQ") else 0.0,
        "pat_ODD":  1.0 if (r["pattern_key"] or "").upper()=="ODD" else 0.0,
        "pat_EVEN": 1.0 if (r["pattern_key"] or "").upper()=="EVEN" else 0.0,
        "cand": float(chosen),
        "post_s": float(proba.get(str(chosen), proba.get(chosen, 0.0))),  # compat int/str key
        "post_l": float(0.0),  # pode enriquecer depois
    }
    keys = sorted(feats.keys())
    X.append([feats[k] for k in keys])
    y.append(label)

X = np.array(X); y = np.array(y)
if len(y) < 200:
    raise SystemExit("Poucos exemplos rotulados. Colete mais decisões/fechamentos antes de treinar.")

tscv = TimeSeriesSplit(n_splits=5)
best_auc, best_model = -1, None
for n_estimators in (100, 200, 300):
    model = GradientBoostingClassifier(random_state=17, n_estimators=n_estimators, max_depth=3)
    aucs = []
    for tr, va in tscv.split(X):
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[va])[:,1]
        aucs.append(roc_auc_score(y[va], p))
    avg = float(np.mean(aucs))
    if avg > best_auc:
        best_auc, best_model = avg, model

best_model.fit(X, y)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "wb") as f:
    pickle.dump(best_model, f)
print(f"Modelo salvo em {OUT} | AUC_cv≈{best_auc:.3f}")