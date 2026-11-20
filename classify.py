"""Inference: raw text -> genre prediction with and without temporal features."""
import os, sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLE = None

NICE = {"temp_past_ratio": "past-tense share", "temp_fut_ratio": "future (will/shall) share",
        "temp_prog_ratio": "progressive (-ing) share", "temp_perf_ratio": "perfect (have + pp) share",
        "temp_transition_rate": "tense-switch rate", "temp_tense_deviation_ratio": "off-dominant-tense share",
        "temp_first_last_shift": "first≠last sentence tense"}
N_SHUFFLE = 20
MIN_SENTENCES = 3
CHUNK = 6


def pretty(c):
    if c in NICE: return NICE[c]
    if c.startswith("deprel__"): return "dependency: " + c[8:]
    if c.startswith("depbig__"): return "dep. pair: " + c[8:].replace("_", " ")
    return {"cd_ttr": "char diversity (TTR)", "cd_maas": "char diversity (Maas)", "cd_msttr": "char diversity (MSTTR)",
            "cd_mattr": "char diversity (MATTR)", "cd_mtld": "char diversity (MTLD)", "cd_mtld_ma": "char diversity (MTLD-MA)",
            "cd_vocd": "char diversity (vocd-D)", "cd_yulek": "char diversity (Yule K)", "lex_density": "lexical density",
            "avg_sent_len": "avg sentence length", "std_sent_len": "sentence-length spread"}.get(c, c.replace("_", " "))


_BASE = _NEURAL = None


def basebundle():
    global _BASE
    if _BASE is None:
        _BASE = joblib.load(os.path.join(HERE, "models", "basepaper.joblib"))
    return _BASE


def neuralbundle():
    """Narrative-time neural models (static and time-aware), loaded once."""
    global _NEURAL
    if _NEURAL is None:
        import torch, neural as N
        d = torch.load(os.path.join(HERE, "models", "neural.pt"), map_location="cpu", weights_only=False)
        emb = np.zeros(d["emb_shape"], np.float32)
        models = {}
        for name, ta in (("static", False), ("time", True)):
            models[name] = {}
            for kind in ("cnn", "bilstm"):
                m = N.build(kind, emb, ta); m.load_state_dict(d["models"][name][kind]); m.eval()
                models[name][kind] = m
        _NEURAL = {"models": models, "stoi": {w: i for i, w in enumerate(d["itos"])}, "acc": d["acc"], "n_test": d["n_test"]}
    return _NEURAL


def bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(os.path.join(HERE, "models", "toolkit.joblib"))
    return _BUNDLE


def chunk_sentences(sents):
    """Paragraph-size windows of 6 sentences (a short tail is merged into the previous window)."""
    n = len(sents)
    if n <= CHUNK + 2:
        return [list(range(n))]
    idx, chunks, i = list(range(n)), [], 0
    while i < n:
        chunks.append(idx[i:i + CHUNK]); i += CHUNK
    if len(chunks) > 1 and len(chunks[-1]) < MIN_SENTENCES:
        chunks[-2] += chunks.pop()
    return chunks


def _feature_row(parsed, meta):
    d = F.kazmi_dict(parsed, meta["rel_vocab"], meta["big_vocab"])
    d.update(F.temporal_features(parsed))
    return d


def _contrib(model, cols, row):
    """Per-feature share of the logit (positive = fiction) = z-score x learned weight. Also returns z and weights."""
    sc, lr = model.steps[0][1], model.steps[1][1]
    x = np.array([row[c] for c in cols], float)
    z = (x - sc.mean_) / sc.scale_
    return z * lr.coef_[0], z, lr.coef_[0]


def _calc_text(key, d):
    """Human-readable arithmetic behind one temporal feature value, using this window's counts."""
    def frac(n, m, what):
        return f"{n} {what[0]} ÷ {m} {what[1]}" if m else f"no {what[1]} found → 0"
    if key == "temp_past_ratio": return frac(*d[key], ("past-tense verbs", "tensed verbs"))
    if key == "temp_fut_ratio": return frac(*d[key], ("will/shall verbs", "tensed verbs"))
    if key == "temp_prog_ratio": return frac(*d[key], ("-ing participles", "tensed verbs"))
    if key == "temp_perf_ratio": return frac(*d[key], ("have + participle verbs", "tensed verbs"))
    if key == "temp_transition_rate": return frac(*d[key], ("tense switches", "sentence pairs"))
    if key == "temp_tense_deviation_ratio":
        n, m = d[key]
        return (f"{n} sentences off the dominant tense ({d['dominant']}) ÷ {m} tensed sentences" if m
                else "no tensed sentences → 0")
    first, last = d["first_last"]
    if first is None: return "no tensed sentences → 0"
    return f"first = {first}, last = {last} → {'different = 1' if first != last else 'same = 0'}"


def analyze(text):
    import basepaper as BP, neural as N
    b = bundle(); meta = b["meta"]; bb = basebundle(); nb = neuralbundle()
    token_sents = F.split_sentences(text)
    if len(token_sents) < MIN_SENTENCES:
        raise ValueError(f"Need at least {MIN_SENTENCES} sentences (found {len(token_sents)}). "
                         f"The models were trained on 5-6 sentence paragraphs.")
    # --- stage 1: base paper, applied to the whole text (it was designed for whole documents)
    brow = BP.features(F.xpos_tags(token_sents))
    p_base = float(bb["clf"].predict_proba(np.array([[brow[c] for c in BP.COLS]]))[0, 1])
    note = None
    if brow["_n_pronoun"] == 0:
        note = ("No pronouns found: the base paper's adjective/pronoun and noun/pronoun ratios are undefined "
                "(set to 0), which pushes it towards 'fiction'. This is its weakness on short text.")
    base = {"p_fiction": p_base, "note": note,
            "features": [{"name": BP.NICE[c], "value": brow[c], "fiction_mean": bb["class_means"]["fiction"][c],
                          "nonfiction_mean": bb["class_means"]["non-fiction"][c]} for c in BP.COLS]}
    chunks = chunk_sentences(token_sents)
    out_chunks = []
    for ch in chunks:
        parsed = F.parse_sentences([token_sents[i] for i in ch])
        row = _feature_row(parsed, meta)
        res = {}
        for name in ("E1", "E3"):
            m, cols = b["models"][name]["model"], b["models"][name]["cols"]
            res[name] = float(m.predict_proba(np.array([[row[c] for c in cols]]))[0, 1])  # P(fiction)
        # neural stages (paragraph = this window's tokens; time = position inside the window)
        toks = [t for i in ch for t in token_sents[i]]
        I, T, L = N.encode(toks, nb["stoi"])
        pn = {}; att = None
        for name in ("static", "time"):
            pn[name], a_ = N.predict_one(nb["models"][name], I, T, L)
            if name == "time":
                att = a_
        # shuffle test (as in the notebook): permute the time values t among the words, N_SHUFFLE times
        rng = np.random.default_rng(0); sh = []; example = None
        for k in range(N_SHUFFLE):
            Ts = T.copy(); Ts[0, :L] = T[0, :L][rng.permutation(L)]
            sh.append(N.predict_one(nb["models"]["time"], I, Ts, L)[0])
            if k == 0:   # one concrete shuffle, shown on screen
                example = [{"token": t, "t_real": float(T[0, j]), "t_shuf": float(Ts[0, j])} for j, t in enumerate(toks[:8])]
        real = pn["time"]
        time_check = {"p_real": real, "p_shuffled_mean": float(np.mean(sh)), "p_shuffled_min": float(np.min(sh)),
                      "p_shuffled_max": float(np.max(sh)), "n": N_SHUFFLE, "example": example,
                      "n_flips": int(sum((p >= .5) != (real >= .5) for p in sh))}
        lo = [t.lower() for t in toks][:N.MAX_LEN]
        w = (att / att.max()) if att is not None and att.max() > 0 else np.zeros(len(lo))
        m3 = b["models"]["E3"]["model"]
        cols3 = b["models"]["E3"]["cols"]
        con, zz, ww = _contrib(m3, cols3, row)
        cmap = dict(zip(cols3, con)); zmap = dict(zip(cols3, zz)); wmap = dict(zip(cols3, ww))
        icpt = float(m3.steps[1][1].intercept_[0])
        t_sum = float(sum(cmap.get(k, 0.0) for k in F.TEMPORAL_COLS))
        s_sum = float(sum(v for k, v in cmap.items() if k not in F.TEMPORAL_COLS))
        decision = {"intercept": icpt, "short_text": s_sum, "temporal": t_sum, "n_short": len(cols3) - len([k for k in F.TEMPORAL_COLS if k in cmap]),
                    "total": icpt + s_sum + t_sum, "p_fiction": res["E3"]}
        states = F.sentence_states(parsed)
        tdet = F.temporal_detail(parsed)
        out_chunks.append({
            "sentences": [{"text": " ".join(token_sents[i]), "state": s} for i, s in zip(ch, states)],
            "n_sentences": len(ch),
            "p_fiction": {**res, "neural_static": pn["static"], "neural_time": pn["time"]},
            "attention": [{"token": t, "w": float(x)} for t, x in zip(toks[:N.MAX_LEN], w)],
            "temporal": [{"name": NICE[c], "key": c, "value": row[c], "used": c in cmap, "push": float(cmap[c]) if c in cmap else None,
                          "z": float(zmap[c]) if c in cmap else None, "weight": float(wmap[c]) if c in cmap else None,
                          "calc": _calc_text(c, tdet),
                          "fiction_mean": meta["temporal_means"]["fiction"][c],
                          "nonfiction_mean": meta["temporal_means"]["non-fiction"][c]} for c in F.TEMPORAL_COLS],
            "decision": decision, "time_check": time_check,
        })
    keys = ("E1", "E3", "neural_static", "neural_time")
    avg = {n: float(np.mean([c["p_fiction"][n] for c in out_chunks])) for n in keys}
    avg["base"] = p_base
    warn = None
    if len(token_sents) < 5:
        warn = "Fewer than 5 sentences: tense-switch features are less reliable."
    return {"n_sentences": len(token_sents), "n_chunks": len(out_chunks), "p_fiction": avg, "base": base,
            "chunks": out_chunks, "warning": warn}


def verdict(p):
    return "fiction" if p >= 0.5 else "non-fiction"
