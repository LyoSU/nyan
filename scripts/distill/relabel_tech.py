"""Relabel only the posts the tech rule can move, and merge them into a copy of the v3b labels.

The v3 rule "a company's decisions in any industry — economy" pulled what tech
companies launch or open out of tech (an AI lab, a mobile network); cat_v3.json
now says where those go. Relabelling all 40k for that would cost the full run,
so this asks the teacher again only about posts either prompt tied to tech — by
category, second guess or topic — and keeps every other post's v3b answer.

The teacher judges a post against the others in its batch: twenty tech posts
together came back with far fewer tech answers than the same posts among the
pool's usual mix (a run of gadget reviews read as ads). So each batch is MIX
tied posts among random others, as in the original labelling, and only the tied
posts' answers are kept.

Usage: PROMPT=v3 TAG=_v3c OPENLUX_KEY=... relabel_tech.py
Writes teach_multi_<model>_v3c.jsonl next to the v3b file.
"""
import json, os, random, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
assert os.environ.get("PROMPT") == "v3" and os.environ.get("TAG"), "run with PROMPT=v3 and a new TAG"
import multi_label as ml
from eval_jev import read_jsonl, write_jsonl

D = ml.D
v1 = {r["url"]: r["labels"] for r in read_jsonl(D / f"teach_multi_{ml.MODEL}.jsonl")}
v3b = read_jsonl(D / f"teach_multi_{ml.MODEL}_v3b.jsonl")

def tied_to_tech(a: dict | None) -> bool:
    if not a:
        return False
    return "tech" in (a.get("category"), a.get("second")) or str(a.get("topic", "")).startswith("tech.")

picked = [i for i, r in enumerate(v3b) if tied_to_tech(r["labels"]) or tied_to_tech(v1.get(r["url"]))]
print(f"relabelling {len(picked)} of {len(v3b)} posts", flush=True)
MIX = int(os.environ.get("MIX", 4))
tied = set(picked)
others = [i for i in range(len(v3b)) if i not in tied]
rng = random.Random(0)
texts, slots = [], []
for k in range(0, len(picked), MIX):
    batch = [(i, True) for i in picked[k:k + MIX]] + [(i, False) for i in rng.sample(others, ml.BATCH - MIX)]
    rng.shuffle(batch)
    texts += [v3b[i]["text"] for i, _ in batch]; slots += batch
print(f"{len(texts)} posts asked in {len(texts) // ml.BATCH} mixed batches", flush=True)
raw = ml._label(texts, int(os.environ.get("WORKERS", 16)))
# Batches are shuffled, so answers are matched back by post, not by position.
answers = {i: a for a, (i, is_tied) in zip(raw, slots) if is_tied}
assert len(raw) == len(slots) and set(answers) == tied

moves = Counter(); kept = 0
out = [dict(r) for r in v3b]
for i in picked:
    a = answers[i]
    if not a:
        kept += 1
        continue
    moves[(out[i]["labels"]["category"], a["category"])] += 1
    out[i]["labels"] = a
write_jsonl(D / f"teach_multi_{ml.MODEL}{ml.TAG}.jsonl", out)
tech = lambda rows: sum(r["labels"]["category"] == "tech" for r in rows)
print(f"tech {tech(v3b)} -> {tech(out)}; kept the v3b answer for {kept} the teacher refused; ${ml.cost():.3f}")
for (a, b), n in moves.most_common(15):
    if a != b:
        print(f"  {a} -> {b}: {n}")
