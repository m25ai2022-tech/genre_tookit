"""Train the toolkit's neural models: static (no time) and time-aware (Time2Vec, ABEF, addition).
Same test files as train.py (seed-0 file split). Saves models/neural.pt.   Usage: python3 train_neural.py"""
import os, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import neural as N
import train as T

HERE = os.path.dirname(os.path.abspath(__file__))
GLOVE = os.path.expanduser("~/Downloads/glove.6B.100d.txt")
LR, TIME_LR, BATCH, MAX_EPOCHS, PATIENCE, P_AUG, MIN_FREQ = 5e-5, 1e-3, 32, 30, 4, 0.10, 3
torch.set_num_threads(os.cpu_count() or 4)


def load_glove(words):
    out, want = {}, set(words)
    with open(GLOVE, encoding="utf8") as f:
        for line in f:
            w, _, rest = line.partition(" ")
            if w in want:
                out[w] = np.asarray(rest.split(), np.float32)
    return out


def build_syn_pool(itos, stoi):
    from nltk.corpus import wordnet as wn, stopwords
    stop = set(stopwords.words("english")); pool = {}
    for i, w in enumerate(itos):
        if i < 2 or not w.isalpha() or len(w) < 3 or w in stop:
            continue
        c = {stoi[l.name().lower()] for s in wn.synsets(w) for l in s.lemmas()
             if l.name().lower() != w and l.name().isalpha() and l.name().lower() in stoi}
        if c:
            pool[i] = np.array(sorted(c))
    return pool


def augment(ids, pool, rng):
    out = ids.copy()
    has = np.zeros(max(int(ids.max()), max(pool)) + 1, bool); has[list(pool)] = True
    sel = (ids > 0) & has[ids] & (rng.random(ids.shape) < P_AUG)
    for r, c in zip(*np.nonzero(sel)):
        cand = pool[ids[r, c]]; out[r, c] = cand[rng.integers(len(cand))]
    return out


def batch(Tn, idx):
    ids, tau, ln = Tn[0][idx], Tn[1][idx], Tn[2][idx]
    Lb = int(ln.max()); ids, tau = ids[:, :Lb], tau[:, :Lb]
    mask = torch.arange(Lb).unsqueeze(0) < ln.unsqueeze(1)
    return (ids, tau, mask) + ((Tn[3][idx],) if len(Tn) > 3 else ())


@torch.no_grad()
def probs(model, Tn):
    model.eval(); out = []
    for s in range(0, Tn[0].size(0), 256):
        ids, tau, mask = batch(Tn, torch.arange(s, min(s + 256, Tn[0].size(0))))[:3]
        out.append(torch.softmax(model(ids, tau, mask)[0], -1)[:, 1].numpy())
    return np.concatenate(out)


def train_branch(kind, time_aware, emb, Ttr, Tva, seed):
    torch.manual_seed(seed * 100 + (0 if kind == "cnn" else 1))
    m = N.build(kind, emb, time_aware)
    tnames = {n for n, _ in m.named_parameters() if n.startswith("temb.") and not n.startswith("temb.emb.")}
    groups = [{"params": [p for n, p in m.named_parameters() if n not in tnames], "lr": LR}]
    if tnames:
        groups.append({"params": [p for n, p in m.named_parameters() if n in tnames], "lr": TIME_LR})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, best_state, bad = 1e9, None, 0
    n = Ttr[0].size(0)
    for ep in range(1, MAX_EPOCHS + 1):
        m.train(); perm = torch.randperm(n)
        for s in range(0, n, BATCH):
            ids, tau, mask, yb = batch(Ttr, perm[s:s + BATCH])
            loss = F.cross_entropy(m(ids, tau, mask)[0], yb)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); vl = 0.0
        with torch.no_grad():
            for s in range(0, Tva[0].size(0), 256):
                ids, tau, mask, yb = batch(Tva, torch.arange(s, min(s + 256, Tva[0].size(0))))
                vl += F.cross_entropy(m(ids, tau, mask)[0], yb, reduction="sum").item()
        vl /= Tva[0].size(0); sched.step(vl)
        if vl < best - 1e-4:
            best, bad = vl, 0; best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        print(f"    {kind} time={time_aware} ep{ep} val_loss {vl:.4f}", flush=True)
    m.load_state_dict(best_state)
    return m


def main():
    t0 = time.time()
    paras = T.load_paragraphs()
    toks = [[w.lower() for s in p["tokens"] for w in s] for p in paras]
    cnt = Counter(w for t in toks for w in t)
    glove = load_glove(cnt)
    words = sorted(w for w, c in cnt.items() if (w in glove) or c >= MIN_FREQ)
    itos = ["<pad>", "<unk>"] + words; stoi = {w: i for i, w in enumerate(itos)}
    rng = np.random.default_rng(123)
    emb = rng.normal(0, 0.3, size=(len(itos), N.EMB_DIM)).astype(np.float32); emb[0] = 0
    for i, w in enumerate(itos[2:], 2):
        if w in glove:
            emb[i] = glove[w]
    print(f"vocab {len(itos)-2}, GloVe-initialised {sum(w in glove for w in itos[2:])}  ({time.time()-t0:.0f}s)", flush=True)

    Np = len(paras)
    IDS = np.zeros((Np, N.MAX_LEN), np.int64); TAU = np.zeros((Np, N.MAX_LEN), np.float32); LEN = np.zeros(Np, np.int64)
    for k, t in enumerate(toks):
        I, Tt, L = N.encode(t, stoi); IDS[k] = I[0]; TAU[k] = Tt[0]; LEN[k] = L
    y = np.array([int(p["label"] == "fiction") for p in paras]); files = np.array([p["file"] for p in paras])

    tr0, te0 = T.file_split(paras, T.DEMO_SEED)             # identical test files to the feature models
    tr0 = np.array(tr0); te0 = np.array(te0)
    tr2, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0).split(np.zeros(len(tr0)), y[tr0], files[tr0]))
    tr, va = tr0[tr2], tr0[va]
    pool = build_syn_pool(itos, stoi)
    ids_aug = augment(IDS[tr], pool, np.random.default_rng(0))
    def dev(i, t, l, yy=None):
        o = [torch.as_tensor(i), torch.as_tensor(t), torch.as_tensor(l)]
        return o + ([torch.as_tensor(yy)] if yy is not None else [])
    Ttr = dev(np.concatenate([IDS[tr], ids_aug]), np.concatenate([TAU[tr]] * 2), np.concatenate([LEN[tr]] * 2), np.concatenate([y[tr]] * 2))
    Tva = dev(IDS[va], TAU[va], LEN[va], y[va]); Tte = dev(IDS[te0], TAU[te0], LEN[te0])
    print(f"train {len(tr)} (x2 aug)  val {len(va)}  test {len(te0)}  ({time.time()-t0:.0f}s)", flush=True)

    out = {"itos": itos, "emb_shape": emb.shape, "models": {}, "acc": {}}
    for name, ta in (("static", False), ("time", True)):
        ms, pr = {}, []
        for kind in ("cnn", "bilstm"):
            m = train_branch(kind, ta, emb, Ttr, Tva, seed=0)
            ms[kind] = m; pr.append(probs(m, Tte))
            print(f"  {name} {kind}: test acc {((pr[-1] >= .5) == y[te0]).mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)
        ens = np.mean(pr, 0)
        out["acc"][name] = float(((ens >= .5) == y[te0]).mean())
        out["models"][name] = {k: v.state_dict() for k, v in ms.items()}
        print(f"{name} ensemble test accuracy {out['acc'][name]:.4f}", flush=True)
    out["n_test"] = len(te0)
    torch.save(out, os.path.join(HERE, "models", "neural.pt"))
    print("saved", f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
