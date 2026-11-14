"""A stronger teacher on the posts the cheap one splits on: is it worth escalating to?"""
import os as _os
# The checkout that holds data/ and models/: this one, unless NYAN_ROOT names another.
ROOT = _os.environ.get("NYAN_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, os, sys
from pathlib import Path
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, str(Path(__file__).parent))
import multi_label as m
from eval_jev import gold_labels, read_jsonl, is_right
E = Path(f"{ROOT}/data/jev_eval")
gold = gold_labels(E); texts = {p["url"]: p["text"] for p in read_jsonl(E / "sample.jsonl")}
urls = json.load(open(m.D / "hard_urls.json"))
ans = m.label([texts[u] for u in urls], workers=4)
ok = [a is not None and is_right(a["category"], gold[u]) for u, a in zip(urls, ans)]
print(f"{m.MODEL}: {100*sum(ok)/len(ok):.0f}% on {len(ok)} hard posts, ${m.cost():.4f} at luna prices")
