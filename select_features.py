"""Reproduce the notebook's feature selection for the toolkit (paragraph_replication.ipynb, Phase 2-3).

E1 feature list  = notebook's file-grouped RFECV survivors ('final4g_surv', from notebook_cache/brown_phase2_v2.pkl).
E3 feature list  = notebook's procedure run on (77-feature Kazmi pool + 7 temporal): per seed, 70/30 file-grouped split,
                   standardise, RFECV (L1 logistic regression, StratifiedGroupKFold inner CV); features kept in more than
                   half of the 5 seeds.  (The notebook does not store this list, so it is recomputed here.)
Writes notebook_features.json.   Usage: python3 select_features.py
"""
import json, os, pickle, sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
import train as T

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]


def run_rfecv_grouped(df, y, groups, cols, name):      # same as the notebook's run_rfecv_grouped
    accs, surv = [], Counter()
    for seed in SEEDS:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed).split(df, y, groups))
        X = df[cols].values
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        est = LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=seed)
        n_splits = min(10, pd.Series(groups[tr]).nunique())
        cv = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(Xtr, y[tr], groups[tr]))
        sel = RFECV(est, cv=cv, scoring="accuracy", min_features_to_select=1, step=max(1, len(cols) // 20)).fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], sel.predict(Xte)))
        surv.update(c for c, k in zip(cols, sel.support_) if k)
    stable = [c for c in cols if surv[c] > len(SEEDS) / 2]
    print(f"{name:10s}: {len(cols)} -> {len(stable)} stable  acc={np.mean(accs):.3f}+/-{np.std(accs):.3f}  (file-grouped)")
    return stable


def main():
    nbk = pickle.load(open(os.path.join(HERE, "notebook_cache", "brown_phase2_v2.pkl"), "rb"))
    pool, e1 = nbk["combined_cols_4fam"], nbk["final4g_surv"]
    paras = T.load_paragraphs()
    parses = pickle.load(open(os.path.join(T.MODELS, "brown_parses.pkl"), "rb"))
    rel_v, big_v = set(), set()
    for ps in parses:
        _, rel, _, big, _ = F.kazmi_dict(ps); rel_v |= set(rel); big_v |= set(big)
    rows = []
    for ps in parses:
        d = F.kazmi_dict(ps, sorted(rel_v), sorted(big_v)); d.update(F.temporal_features(ps)); rows.append(d)
    df = pd.DataFrame(rows)
    y = np.array([1 if p["label"] == "fiction" else 0 for p in paras])
    groups = np.array([p["file"] for p in paras])
    print("E1 (notebook list, re-scored on toolkit features):")
    run_rfecv_grouped(df, y, groups, e1, "E1 list")      # sanity: ~0.887 expected
    print("E1 pool + temporal -> E3:")
    e3 = run_rfecv_grouped(df, y, groups, pool + F.TEMPORAL_COLS, "E3")
    print("temporal kept:", [c for c in e3 if c in F.TEMPORAL_COLS])
    json.dump({"e1": e1, "e3": e3, "pool": pool}, open(os.path.join(HERE, "notebook_features.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
