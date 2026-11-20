"""Base-paper model: train on WHOLE Brown documents, test on the same unseen files as the other stages.
Also evaluates the same model on short (5-6 sentence) test paragraphs: the drop the thesis is about.
Usage: python3 train_basepaper.py   -> models/basepaper.joblib"""
import os, sys, time
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F, basepaper as B, train as T

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    t0 = time.time()
    df = pd.read_csv(T.BROWN_CSV)
    df = df[df["label"].isin(T.FICTION | T.NONFICTION)].copy()
    df["cls"] = df["label"].apply(lambda g: "fiction" if g in T.FICTION else "non-fiction")
    docs = {}
    for fn, g in df.sort_values(["para_id", "sent_id"]).groupby("filename"):
        docs[fn] = {"label": g["cls"].iloc[0], "sents": [str(t).split() for t in g["tokenized_text"]]}
    print(len(docs), "documents", flush=True)
    paras = T.load_paragraphs()
    tr0, te0 = T.file_split(paras, T.DEMO_SEED)
    test_files = {paras[i]["file"] for i in te0}

    rows = {}
    for k, (fn, d) in enumerate(docs.items()):
        rows[fn] = B.features(F.xpos_tags(d["sents"]))
        if k % 50 == 0:
            print(f"  tagged {k}/{len(docs)} ({time.time()-t0:.0f}s)", flush=True)
    X = pd.DataFrame(rows).T[B.COLS]
    y = pd.Series({fn: int(d["label"] == "fiction") for fn, d in docs.items()})
    tr_f = [f for f in docs if f not in test_files]; te_f = [f for f in docs if f in test_files]
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=2000, random_state=42)
    clf.fit(X.loc[tr_f].values, y.loc[tr_f].values)
    acc_doc = float((clf.predict(X.loc[te_f].values) == y.loc[te_f].values).mean())
    print(f"whole documents: train {len(tr_f)}  test {len(te_f)}  accuracy {acc_doc:.4f}", flush=True)

    Xp = np.array([[B.features(F.xpos_tags(paras[i]["tokens"]))[c] for c in B.COLS] for i in te0])
    yp = np.array([int(paras[i]["label"] == "fiction") for i in te0])
    acc_par = float((clf.predict(Xp) == yp).mean())
    print(f"same model on 5-6 sentence test paragraphs: {len(te0)}  accuracy {acc_par:.4f}", flush=True)
    joblib.dump({"clf": clf, "cols": B.COLS, "acc_doc": acc_doc, "acc_par": acc_par, "n_test_docs": len(te_f),
                 "n_train_docs": len(tr_f), "n_test_par": len(te0),
                 "class_means": {lab: {c: float(X.loc[[f for f in tr_f if docs[f]["label"] == lab], c].mean()) for c in B.COLS}
                                 for lab in ("fiction", "non-fiction")}},
                os.path.join(HERE, "models", "basepaper.joblib"))
    print("saved", f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
