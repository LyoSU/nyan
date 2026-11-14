"""Posts for the owner to decide: where the teachers disagree on the category.

From the blind 500 (luna, grok, Jev and the blind label) and from the pool posts
Jev and luna both labelled. Writes review/posts.json, one row per post.
"""
import os as _os
# The checkout that holds data/ and models/: this one, unless NYAN_ROOT names another.
ROOT = _os.environ.get("NYAN_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, random, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/scripts")
from eval_jev import gold_labels, read_jsonl

D = Path(f"{ROOT}/data/distill"); E = Path(f"{ROOT}/data/jev_eval")
rng = random.Random(11)
rows = []
sample = read_jsonl(E / "sample.jsonl"); gold = gold_labels(E)
luna = {r["url"]: r.get("category") for r in read_jsonl(D / "gold_multi_gpt-5.6-luna.jsonl")}
grok = {r["url"]: r.get("category") for r in read_jsonl(D / "gold_grok-4.7.jsonl")}
jev = {r["url"]: r.get("category") for r in read_jsonl(E / "replay.jsonl")}
for p in sample:
    votes = {"luna": luna.get(p["url"]), "grok": grok.get(p["url"]), "Jev": jev.get(p["url"]), "сліпа розмітка": gold[p["url"]]["category"]}
    if len({v for v in votes.values() if v}) > 1:
        rows.append({"url": p["url"], "channel": p.get("channel_id"), "text": p["text"][:2000], "models": {k: v for k, v in votes.items() if v}, "why": "еталон і моделі розійшлися", "src": "gold"})
jev_pool = {r["i"]: r["answers"]["category"]["choice"] for r in read_jsonl(D / "teach_v2.jsonl")}
luna_pool = {r["i"]: r for r in read_jsonl(D / "teach_multi_gpt-5.6-luna.jsonl")}
pool_rows = []
for i, j in jev_pool.items():
    l = luna_pool.get(i)
    if l and l["labels"]["category"] != j:
        pool_rows.append({"url": l["url"], "channel": l["url"].split("/")[3], "text": l["text"][:2000],
                          "models": {"luna": l["labels"]["category"], "Jev": j}, "why": "luna і Jev розійшлися", "src": "pool",
                          "pair": tuple(sorted((l["labels"]["category"], j)))})
print("gold disagreements", len(rows), "pool disagreements", len(pool_rows), "of", len(jev_pool))
print(Counter(r["pair"] for r in pool_rows).most_common(12))
# spread the pool picks across kinds of disagreement, not just the commonest one
by = {}
for r in pool_rows: by.setdefault(r["pair"], []).append(r)
need = 150 - len(rows); picked = []
while len(picked) < need and any(by.values()):
    for k in sorted(by, key=lambda k: -len(by[k])):
        if by[k] and len(picked) < need:
            picked.append(by[k].pop(rng.randrange(len(by[k]))))
allr = rows + picked
rng.shuffle(allr)
for n, r in enumerate(allr):
    r["n"] = n; r.pop("pair", None)
(D / "review").mkdir(exist_ok=True)
json.dump(allr, open(D / "review/posts.json", "w"), ensure_ascii=False, indent=0)
print("picked", len(allr))
