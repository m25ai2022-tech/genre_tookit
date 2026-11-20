"""Train the toolkit's two models on Brown paragraphs (5-6 sentences).

  E1 = Kazmi families, restricted to the features the notebook's RFECV keeps (notebook_features.json)
  E3 = E1 pool + temporal features, again the notebook's RFECV survivors

File-grouped split: whole files go to train or test, so no paragraph of a test
file is ever seen in training. Writes models/toolkit.joblib.
Usage:  python3 train.py            (parses Brown once, cached in models/)
"""
import json, os, pickle, re, sys, time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
BROWN_CSV = os.path.join(HERE, "..", "brown_corpus", "brown.csv")
MODELS = os.path.join(HERE, "models"); os.makedirs(MODELS, exist_ok=True)
FICTION = {"fiction", "mystery", "science_fiction", "romance", "adventure"}
NONFICTION = {"news", "hobbies", "government", "reviews", "learned"}
SEEDS = [0, 1, 2, 3, 4]
DEMO_SEED = 0


def detok(sents):
    """Brown token lists -> natural-looking text (so demo samples go through the same raw-text path as user input)."""
    out = []
    for s in sents:
        t = " ".join(s).replace("`` ", '"').replace(" ''", '"').replace("``", '"').replace("''", '"')
        t = re.sub(r" ([,.;:?!)\]])", r"\1", t)
        t = re.sub(r"([(\[]) ", r"\1", t)
        t = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", t)
        out.append(t)
    return " ".join(out)


def load_paragraphs():
    df = pd.read_csv(BROWN_CSV)
    df = df[df["label"].isin(FICTION | NONFICTION)].copy()
    df["cls"] = df["label"].apply(lambda g: "fiction" if g in FICTION else "non-fiction")
    n = df.groupby(["filename", "para_id"]).size().reset_index(name="n")
    keep = n[n["n"].isin((5, 6))][["filename", "para_id"]]
    dfk = df.merge(keep, on=["filename", "para_id"])
    out = []
    for (fn, pid), g in dfk.sort_values("sent_id").groupby(["filename", "para_id"]):
        out.append({"id": f"{fn}_{pid}", "file": fn, "genre": g["label"].iloc[0],
                    "label": g["cls"].iloc[0],
                    "tokens": [str(t).split() for t in g["tokenized_text"]]})
    return out


def file_split(paras, seed, test_frac=0.30):
    """Stratified (by class) split of FILES."""
    rng = np.random.RandomState(seed)
    files = {}
    for p in paras:
        files.setdefault(p["file"], p["label"])
    test = set()
    for lab in ("fiction", "non-fiction"):
        fl = sorted(f for f, l in files.items() if l == lab)
        rng.shuffle(fl)
        test |= set(fl[:int(round(len(fl) * test_frac))])
    return [i for i, p in enumerate(paras) if p["file"] not in test], \
           [i for i, p in enumerate(paras) if p["file"] in test]


def fit(X, y, seed):
    """Same final classifier as the notebook (paragraph_replication.ipynb): standardise + L1 logistic regression."""
    return make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=seed))


def notebook_features():
    """Feature lists from the notebook's RFECV selection (see select_features.py)."""
    nf = json.load(open(os.path.join(HERE, "notebook_features.json")))
    e1 = nf["e1"]
    # toolkit only: all 7 temporal features are kept (the notebook's selection keeps 3 of them on Brown)
    e3 = [c for c in nf["e3"] if c not in F.TEMPORAL_COLS] + list(F.TEMPORAL_COLS)
    return e1, e3



def main():
    t0 = time.time()
    paras = load_paragraphs()
    print(len(paras), "paragraphs", Counter(p["label"] for p in paras))
    cache_p = os.path.join(MODELS, "brown_parses.pkl")
    if os.path.exists(cache_p):
        parses = pickle.load(open(cache_p, "rb"))
    else:
        parses = F.parse_batch([p["tokens"] for p in paras])
        pickle.dump(parses, open(cache_p, "wb"))
    print(f"parsed ({time.time()-t0:.0f}s)")

    # vocab fit on the DEMO training split only (no test-file leakage)
    tr0, te0 = file_split(paras, DEMO_SEED)
    rel_v, big_v = set(), set()
    for i in tr0:
        _, rel, _, big, _ = F.kazmi_dict(parses[i])
        rel_v |= set(rel); big_v |= set(big)
    rel_v, big_v = sorted(rel_v), sorted(big_v)

    rows = []
    for i, ps in enumerate(parses):
        d = F.kazmi_dict(ps, rel_v, big_v)
        d.update(F.temporal_features(ps))
        rows.append(d)
    df = pd.DataFrame(rows)
    print("features", df.shape, f"({time.time()-t0:.0f}s)")
    y = np.array([1 if p["label"] == "fiction" else 0 for p in paras])
    e1_cols, e3_cols = notebook_features()
    print("E1 features:", len(e1_cols), " E3 features:", len(e3_cols), "(temporal:", [c for c in e3_cols if c in F.TEMPORAL_COLS], ")")

    res = {"E1": [], "E3": []}
    for s in SEEDS:
        tr, te = file_split(paras, s)
        for name, cols in (("E1", e1_cols), ("E3", e3_cols)):
            m = fit(None, None, s).fit(df.loc[tr, cols].values, y[tr])
            res[name].append(float((m.predict(df.loc[te, cols].values) == y[te]).mean()))
    print({k: (np.mean(v), np.std(v)) for k, v in res.items()})

    models = {}
    held = {}
    for name, cols in (("E1", e1_cols), ("E3", e3_cols)):
        m = fit(None, None, DEMO_SEED).fit(df.loc[tr0, cols].values, y[tr0])
        models[name] = {"model": m, "cols": cols}
        held[name] = float((m.predict(df.loc[te0, cols].values) == y[te0]).mean())
    print("demo-split held-out", held)

    samples = [{"id": paras[i]["id"], "genre": paras[i]["genre"], "label": paras[i]["label"],
                "text": detok(paras[i]["tokens"])} for i in te0]
    meta = {"rel_vocab": rel_v, "big_vocab": big_v, "cv": {k: [float(np.mean(v)), float(np.std(v))] for k, v in res.items()},
            "seeds": SEEDS, "e1_cols": e1_cols, "e3_cols": e3_cols, "demo_heldout": held, "n_train": len(tr0), "n_test": len(te0),
            "n_test_files": len({paras[i]["file"] for i in te0}), "n_paragraphs": len(paras),
            "class_balance_test": float(y[te0].mean()),
            "temporal_means": {lab: {c: float(df.loc[[i for i in tr0 if paras[i]["label"] == lab], c].mean())
                                     for c in F.TEMPORAL_COLS} for lab in ("fiction", "non-fiction")}}
    joblib.dump({"models": models, "meta": meta, "samples": samples}, os.path.join(MODELS, "toolkit.joblib"))
    print(f"saved ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
