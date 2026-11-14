"""Heads on frozen embeddings, trained on Jev's labels, judged on the 500 blind labels.

emb_<name>.npy: first 500 rows are the gold sample, the rest the labelled pool
(either the teach rows in order, or the whole pool in pool order).
Usage: TEACH=teach_v2.jsonl heads.py <name>...
"""
import os as _os
# The checkout that holds data/ and models/: this one, unless NYAN_ROOT names another.
ROOT = _os.environ.get("NYAN_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, os, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/scripts")
from eval_jev import Jev, MODEL, gold_labels, in_split, is_right, read_jsonl
from nyan.jev import STATE_NOTE, load_questions

D = Path(f"{ROOT}/data/distill"); E = Path(f"{ROOT}/data/jev_eval")
CATS = ["not_news", "war", "politics", "economy", "incident", "other", "tech", "sports", "entertainment", "science"]
sample = read_jsonl(E / "sample.jsonl"); gold = gold_labels(E)
teach = [r for f in os.environ.get("TEACH", "teach_v2.jsonl").split(",") for r in read_jsonl(D / f)]
import random; random.Random(1).shuffle(teach)  # mixed teachers, so any prefix is a fair sample
y_hard = np.array([CATS.index(r["answers"]["category"]["choice"]) for r in teach])
y_soft = np.array([[r["answers"]["category"]["probabilities"].get(c, 0.0) for c in CATS] for r in teach])
y_soft = y_soft / y_soft.sum(1, keepdims=True)

q = load_questions(f"{ROOT}/configs/jev_questions.json")
cache = {r["key"]: r for r in read_jsonl(E / "cache.jsonl")}
jev_gold = []
for p in sample:
    r = cache.get(Jev.key({"model": MODEL, "state": {"note": STATE_NOTE, "post_text": p["text"][:1500]}, "questions": q}))
    jev_gold.append(r["answers"]["category"]["choice"] if r else None)
test = np.array([in_split(p["url"], "test") for p in sample])

def acc(pred, mask=None):
    idx = [i for i in range(len(sample)) if (mask is None or mask[i]) and pred[i] is not None]
    return (100 * np.mean([is_right(pred[i], gold[sample[i]["url"]]) for i in idx]) if idx else float("nan")), len(idx)

def fit(x, n, soft, C):
    X = x[:n]
    if soft:  # each post once per class, weighted by the teacher's probability
        k = len(CATS); rows = np.repeat(np.arange(n), k); ys = np.tile(np.arange(k), n); w = y_soft[:n].ravel()
        keep = w > 1e-3
        return LogisticRegression(C=C, max_iter=3000).fit(X[rows[keep]], ys[keep], sample_weight=w[keep])
    return LogisticRegression(C=C, max_iter=3000).fit(X, y_hard[:n])

def judge(name, proba):
    pred = [CATS[i] for i in proba.argmax(1)]
    print(f"  {name:26s} all {acc(pred)[0]:.1f}%  test {acc(pred, test)[0]:.1f}%")
    return pred

def cascade(pred, proba):
    conf = proba.max(1)
    for t in (0.5, 0.6, 0.7, 0.8):
        mixed = [pred[i] if conf[i] >= t or jev_gold[i] is None else jev_gold[i] for i in range(500)]
        local = conf >= t
        a_loc = 100 * np.mean([is_right(pred[i], gold[sample[i]["url"]]) for i in range(500) if local[i]])
        print(f"  cascade t={t}: local {100*local.mean():.0f}% at {a_loc:.1f}%, overall all {acc(mixed)[0]:.1f}% test {acc(mixed, test)[0]:.1f}%")
    news = [i for i in range(500) if "not_news" not in (gold[sample[i]["url"]]["category"], gold[sample[i]["url"]].get("also_ok"))]
    junk = [i for i in range(500) if gold[sample[i]["url"]]["category"] == "not_news"]
    print(f"  not_news: caught {100*np.mean([pred[i]=='not_news' for i in junk]):.1f}% of {len(junk)}, news dropped {sum(pred[i]=='not_news' for i in news)}/{len(news)}")

def report(name):
    emb = np.load(D / f"emb_{name}.npy").astype(np.float32)
    g, x = emb[:500], emb[500:]
    if len(x) != len(teach):  # the whole pool, in pool order
        x = x[[r["i"] for r in teach]]
    else:
        raise SystemExit(f"{name}: rows follow one teach file; embed the pool to mix files")
    print(f"\n== {name} dim={emb.shape[1]}")
    for n in [n for n in (1000, 2000, 5000, 10000) if n < len(teach)]:
        clf = fit(x, n, False, 10.0); p = np.zeros((500, len(CATS))); p[:, clf.classes_] = clf.predict_proba(g)
        judge(f"n={n} hard C=10", p)
    best = None
    for soft, C in ((False, 10.0), (False, 100.0), (True, 10.0), (True, 100.0)):
        clf = fit(x, len(teach), soft, C); p = np.zeros((500, len(CATS))); p[:, clf.classes_] = clf.predict_proba(g)
        pred = judge(f"n={len(teach)} {'soft' if soft else 'hard'} C={C:g}", p)
        dev_acc = acc(pred, ~test)[0]  # choose on dev, report test
        if best is None or dev_acc > best[0]:
            best = (dev_acc, pred, p, f"{'soft' if soft else 'hard'} C={C:g}")
    print(f"  best by dev: {best[3]}")
    cascade(best[1], best[2])

if __name__ == "__main__":
    print("jev:", "all %.1f%%" % acc(jev_gold)[0], "test %.1f%%" % acc(jev_gold, test)[0])
    prod = [p["prod_category"] for p in sample]
    print("prod head:", "all %.1f%%" % acc(prod)[0], "test %.1f%%" % acc(prod, test)[0])
    for name in sys.argv[1:]:
        report(name)
