"""Feature extraction for the genre toolkit.

Same five feature families as the thesis notebooks (Kazmi et al. 2022 families
+ the 7 hand-crafted temporal features), computed from a single Stanza parse so
that training text (Brown) and user text go through exactly the same code.

Parse format (one paragraph): list of sentences, each a list of
(upos, feats, lemma, head, deprel, word_id, text) tuples.
"""
import math
import random
from collections import Counter, defaultdict

import numpy as np

# ------------------------------------------------------------------ parsing
_NLP_TOK = None
_NLP_PARSE = None


def _pipelines():
    global _NLP_TOK, _NLP_PARSE
    if _NLP_PARSE is None:
        import stanza
        try:      # models are cached after the first run; download only if missing
            _NLP_TOK = stanza.Pipeline("en", processors="tokenize", verbose=False, download_method=None)
            _NLP_PARSE = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse",
                                         tokenize_pretokenized=True, verbose=False, download_method=None)
        except Exception:
            stanza.download("en", processors="tokenize,pos,lemma,depparse", verbose=False)
            _NLP_TOK = stanza.Pipeline("en", processors="tokenize", verbose=False, download_method=None)
            _NLP_PARSE = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse",
                                         tokenize_pretokenized=True, verbose=False, download_method=None)
    return _NLP_TOK, _NLP_PARSE


_NLP_POS = None


def xpos_tags(token_sents):
    """Penn-style POS tags (Stanza xpos) for pre-tokenised sentences -> list of [(word, tag)]; tagging only (fast)."""
    global _NLP_POS
    if _NLP_POS is None:
        import stanza
        _NLP_POS = stanza.Pipeline("en", processors="tokenize,pos", tokenize_pretokenized=True,
                                   verbose=False, download_method=None)
    doc = _NLP_POS(token_sents)
    return [[(w.text, w.xpos or "") for w in s.words] for s in doc.sentences]


def split_sentences(text):
    """Raw text -> list of token lists (Stanza tokenizer / sentence splitter)."""
    tok, _ = _pipelines()
    doc = tok(text)
    return [[t.text for t in s.tokens] for s in doc.sentences]


def parse_sentences(token_sents):
    """list of token lists -> parsed paragraph (list of sentences of tuples)."""
    _, nlp = _pipelines()
    doc = nlp(token_sents)
    return [[(w.upos, w.feats, w.lemma, w.head, w.deprel, w.id, w.text) for w in s.words]
            for s in doc.sentences]


def parse_batch(paragraph_token_sents, batch=100):
    """Parse many paragraphs at once (training); returns list aligned with input."""
    _, nlp = _pipelines()
    out = []
    for i in range(0, len(paragraph_token_sents), batch):
        chunk = paragraph_token_sents[i:i + batch]
        flat = [s for p in chunk for s in p]
        doc = nlp(flat)
        k = 0
        for p in chunk:
            sents = []
            for _ in p:
                s = doc.sentences[k]; k += 1
                sents.append([(w.upos, w.feats, w.lemma, w.head, w.deprel, w.id, w.text)
                              for w in s.words])
            out.append(sents)
    return out


# ------------------------------------------------------------------ helpers
def safe_div(a, b, eps=1e-9):
    return 0.0 if abs(b) <= eps else a / b


def _alpha_words(parsed):
    return [w[6] for s in parsed for w in s if w[6].isalpha()]


# ------------------------------------------------------------ family 1: raw
def raw_features(parsed):
    sl = [sum(1 for w in s if w[6].isalpha()) for s in parsed]
    sl = [x for x in sl if x > 0]
    return {"avg_sent_len": float(np.mean(sl)) if sl else 0.0,
            "std_sent_len": float(np.std(sl)) if sl else 0.0}


# ------------------------------------------------- family 2: lexical diversity
def _chars(parsed):
    return list("".join(w.lower() for w in _alpha_words(parsed)))


def _ttr(c):
    return 0.0 if not c else len(set(c)) / len(c)


def _maas(c):
    N, V = len(c), len(set(c))
    if N <= 1 or V <= 0:
        return 0.0
    return (math.log(N) - math.log(V)) / (math.log(N) ** 2)


def _msttr(c, w=50):
    if len(c) < w:
        return _ttr(c)
    segs = [c[i:i + w] for i in range(0, len(c), w)]
    segs = [s for s in segs if len(s) == w]
    return float(np.mean([_ttr(s) for s in segs])) if segs else _ttr(c)


def _mattr(c, w=50):
    if len(c) <= w:
        return _ttr(c)
    return float(np.mean([_ttr(c[i:i + w]) for i in range(0, len(c) - w + 1)]))


def _mtld_fwd(c, th=0.72):
    if not c:
        return 0.0
    factors, types, n = 0, set(), 0
    for ch in c:
        types.add(ch); n += 1
        if len(types) / n <= th:
            factors += 1; types, n = set(), 0
    if n > 0:
        factors += (1 - len(types) / n) / (1 - th)
    return safe_div(len(c), factors) if factors > 0 else float(len(c))


def _mtld(c):
    if len(c) < 2:
        return 0.0
    return float(np.mean([_mtld_fwd(c), _mtld_fwd(list(reversed(c)))]))


def _mtld_ma(c, th=0.72):
    n = len(c)
    if n < 2:
        return 0.0
    lens = []
    for st in range(n):
        types, k = set(), 0
        for ch in c[st:]:
            types.add(ch); k += 1
            if len(types) / k <= th:
                lens.append(k); break
        else:
            if k > 0:
                lens.append(k)
    return float(np.mean(lens)) if lens else float(n)


def _vocd(c, sizes=range(35, 51), n_samples=100, seed=42):
    if len(c) < max(sizes):
        return _ttr(c) * len(c)
    rng = random.Random(seed)
    m = [np.mean([len(set(rng.sample(c, n))) / n for _ in range(n_samples)]) for n in sizes]
    s, t = np.array(list(sizes), float), np.array(m, float)
    best_d, best_e = None, float("inf")
    for d in np.linspace(1, 200, 400):
        e = np.sum(((d / s) * (np.sqrt(1 + 2 * s / d) - 1) - t) ** 2)
        if e < best_e:
            best_e, best_d = e, d
    return float(best_d)


def _yulek(c):
    N = len(c)
    if N == 0:
        return 0.0
    return 10000.0 * safe_div(sum(f * f for f in Counter(c).values()) - N, N * N)


# UPOS -> coarse categories (mirrors the Brown tag groups used in the notebooks:
# nouns incl. proper nouns; "verb" = lexical verbs only, as be/have/do/modals
# have their own tags in Brown).
_CAT = {"ADJ": "adjective", "ADV": "adverb", "NOUN": "noun", "PROPN": "noun",
        "VERB": "verb", "PRON": "pronoun"}


def _pos_counts(parsed):
    c = defaultdict(int)
    for s in parsed:
        for w in s:
            if w[6].isalpha() or w[0] == "PRON":
                cat = _CAT.get(w[0])
                if cat:
                    c[cat] += 1
    return c


def lexical_features(parsed):
    c = _chars(parsed)
    f = {"cd_ttr": _ttr(c), "cd_maas": _maas(c), "cd_msttr": _msttr(c),
         "cd_mattr": _mattr(c), "cd_mtld": _mtld(c), "cd_mtld_ma": _mtld_ma(c),
         "cd_vocd": _vocd(c), "cd_yulek": _yulek(c)}
    pc = _pos_counts(parsed)
    content = pc.get("adjective", 0) + pc.get("adverb", 0) + pc.get("noun", 0) + pc.get("verb", 0)
    total = len(_alpha_words(parsed))
    f["lex_density"] = safe_div(content, max(total - content, 0))
    return f


def pos_ratio_features(parsed):
    pc = _pos_counts(parsed)
    adj, adv, noun, verb, pron = (pc.get(k, 0) for k in
                                  ("adjective", "adverb", "noun", "verb", "pronoun"))
    return {"adverb_noun": safe_div(adv, noun), "adverb_pronoun": safe_div(adv, pron),
            "adj_verb": safe_div(adj, verb), "noun_verb": safe_div(noun, verb),
            "verb_pronoun": safe_div(verb, pron), "adverb_adj": safe_div(adv, adj),
            "adj_pronoun": safe_div(adj, pron), "noun_pronoun": safe_div(noun, pron)}


# ---------------------------------------------------- family 4/5: syntactic
def dep_counts(parsed):
    rel, big, n1, n2 = Counter(), Counter(), 0, 0
    for s in parsed:
        upos_by_id = {w[5]: w[0] for w in s}
        for upos, feats, lemma, head, deprel, wid, _ in s:
            rel[deprel] += 1; n1 += 1
            if head == 0:
                continue
            hu = upos_by_id.get(head)
            if hu is None:
                continue
            big[(hu, upos, "before" if wid < head else "after")] += 1; n2 += 1
    return rel, n1, big, n2


# ------------------------------------------------------------ temporal (7)
FUTURE_MODALS = {"will", "shall"}
TEMPORAL_COLS = ["temp_past_ratio", "temp_fut_ratio", "temp_prog_ratio", "temp_perf_ratio",
                 "temp_transition_rate", "temp_tense_deviation_ratio", "temp_first_last_shift"]


def sentence_states(parsed):
    """Tense state of each sentence = state of its first tense-bearing token."""
    out = []
    for s in parsed:
        state = None
        for upos, feats, lemma, head, deprel, wid, _ in s:
            f = feats or ""
            if upos == "AUX" and lemma in FUTURE_MODALS and deprel == "aux":
                state = "Fut"
            elif "Tense=Past" in f:
                state = "Past"
            elif "Tense=Pres" in f:
                state = "Pres"
            if state is not None:
                break
        out.append(state)
    return out


def temporal_detail(parsed):
    """The 7 temporal features with the counts behind them: key -> (numerator, denominator),
    plus 'first_last' -> (first state, last state)."""
    n_past = n_fut = n_fin = n_prog = n_perf = 0
    for s in parsed:
        children = defaultdict(list)
        for upos, feats, lemma, head, deprel, wid, _ in s:
            children[head].append((lemma, deprel))
        for upos, feats, lemma, head, deprel, wid, _ in s:
            f = feats or ""
            fut = upos == "AUX" and lemma in FUTURE_MODALS and deprel == "aux"
            tense = "Past" if "Tense=Past" in f else ("Pres" if "Tense=Pres" in f else None)
            if tense or fut:
                n_fin += 1
                if fut:
                    n_fut += 1
                elif tense == "Past":
                    n_past += 1
            if "VerbForm=Part" in f:
                if "Tense=Pres" in f:
                    n_prog += 1
                elif "Tense=Past" in f and "Voice=Pass" not in f:
                    if any(l == "have" for l, d in children[wid] if d in ("aux", "aux:pass")):
                        n_perf += 1
    st = sentence_states(parsed)
    pairs = [(a, b) for a, b in zip(st, st[1:]) if a is not None and b is not None]
    trans = sum(1 for a, b in pairs if a != b)
    known = [x for x in st if x is not None]
    if known:
        modal = Counter(known).most_common(1)[0][0]
        off = sum(1 for x in known if x != modal)
        first, last = known[0], known[-1]
    else:
        modal, off, first, last = None, 0, None, None
    return {"temp_past_ratio": (n_past, n_fin), "temp_fut_ratio": (n_fut, n_fin),
            "temp_prog_ratio": (n_prog, n_fin), "temp_perf_ratio": (n_perf, n_fin),
            "temp_transition_rate": (trans, len(pairs)),
            "temp_tense_deviation_ratio": (off, len(known)), "dominant": modal,
            "first_last": (first, last)}


def temporal_features(parsed):
    d = temporal_detail(parsed)
    out = {k: safe_div(*d[k]) for k in ("temp_past_ratio", "temp_fut_ratio", "temp_prog_ratio", "temp_perf_ratio",
                                        "temp_transition_rate", "temp_tense_deviation_ratio")}
    first, last = d["first_last"]
    out["temp_first_last_shift"] = 1.0 if (first is not None and first != last) else 0.0
    return out


# ------------------------------------------------------------- full vector
def kazmi_dict(parsed, deprel_vocab=None, depbig_vocab=None):
    """Raw + lexical + POS-ratio + dependency features (the 4 Kazmi families).
    With vocabularies given, returns the fixed column set used by the models."""
    f = {}
    f.update(raw_features(parsed))
    f.update(lexical_features(parsed))
    f.update(pos_ratio_features(parsed))
    rel, n1, big, n2 = dep_counts(parsed)
    if deprel_vocab is None:
        return f, rel, n1, big, n2
    for k in deprel_vocab:
        f[f"deprel__{k}"] = safe_div(rel.get(k, 0), n1)
    for k in depbig_vocab:
        f["depbig__" + "_".join(k)] = safe_div(big.get(k, 0), n2)
    return f
