"""Command-line fallback:  python3 cli.py            (reads text from stdin)
                          python3 cli.py -f file.txt"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify

ap = argparse.ArgumentParser(); ap.add_argument("-f", "--file")
a = ap.parse_args()
text = open(a.file, encoding="utf-8").read() if a.file else sys.stdin.read()
r = classify.analyze(text)
if r["warning"]:
    print("!", r["warning"])
print(f"{r['n_sentences']} sentences, {r['n_chunks']} paragraph window(s)")
for n, lab in (("base", "Base paper (Qureshi 2019)        "), ("E1", "Short-text features (Kazmi 2022) "),
               ("E3", "+ Handcrafted temporal features  "), ("neural_static", "Neural network, no time         "),
               ("neural_time", "Narrative-time neural (Time2Vec)")):
    p = r["p_fiction"][n]
    print(f"  {lab}: {classify.verdict(p):12s} P(fiction)={p:.2f}")
for i, c in enumerate(r["chunks"], 1):
    print(f"\nwindow {i}: tense per sentence =", [s["state"] or "-" for s in c["sentences"]])
    for t in c["temporal"]:
        print(f"   {t['name']:28s} {t['value']:.2f}   (fiction avg {t['fiction_mean']:.2f}, non-fiction avg {t['nonfiction_mean']:.2f})")
