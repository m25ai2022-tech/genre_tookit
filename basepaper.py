"""Base paper stage: Qureshi et al. 2019 (ACL W19-3409), 'A Simple Approach to Classify Fictional and Non-Fictional Genres'.

The paper's six selected features, computed over a whole text with Penn-style POS tags, fed to an L1 logistic regression:
  avg sentence length, avg word length, hyphen/quote ratio, adverb/adjective, adjective/pronoun, noun/pronoun.
"""
import numpy as np

PENN = {"adjective": {"JJ", "JJR", "JJS"}, "adverb": {"RB", "RBR", "RBS", "WRB"},
        "noun": {"NN", "NNS", "NNP", "NNPS"}, "pronoun": {"PRP", "WP"}}
HYPHEN = {"--", "-", "—", "–"}
QUOTE = {"``", "''", '"', "“", "”"}
COLS = ["avg_sent_len", "avg_word_len", "hyphen_quote", "adverb_adj", "adj_pronoun", "noun_pronoun"]
NICE = {"avg_sent_len": "average sentence length", "avg_word_len": "average word length",
        "hyphen_quote": "hyphen / quote ratio", "adverb_adj": "adverb / adjective ratio",
        "adj_pronoun": "adjective / pronoun ratio", "noun_pronoun": "noun / pronoun ratio"}


def _div(a, b):
    return 0.0 if b == 0 else a / b


def features(tagged_sents):
    """tagged_sents: list of [(word, penn_tag)] -> dict of the six paper features."""
    sl = [sum(1 for w, _ in s if w.isalpha()) for s in tagged_sents]
    sl = [x for x in sl if x > 0]
    aw = [w for s in tagged_sents for w, _ in s if w.isalpha()]
    toks = [w for s in tagged_sents for w, _ in s]
    c = {k: 0 for k in PENN}
    for s in tagged_sents:
        for w, t in s:
            for k, ts in PENN.items():
                if t in ts:
                    c[k] += 1; break
    hy = sum(1 for w in toks if w in HYPHEN); qu = sum(1 for w in toks if w in QUOTE)
    return {"avg_sent_len": float(np.mean(sl)) if sl else 0.0,
            "avg_word_len": float(np.mean([len(w) for w in aw])) if aw else 0.0,
            "hyphen_quote": _div(hy, qu),
            "adverb_adj": _div(c["adverb"], c["adjective"]),
            "adj_pronoun": _div(c["adjective"], c["pronoun"]),
            "noun_pronoun": _div(c["noun"], c["pronoun"]),
            "_n_pronoun": c["pronoun"]}
