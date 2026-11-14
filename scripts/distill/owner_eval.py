"""Score teacher prompts against the owner's own answers on the review page."""
import os as _os
# The checkout that holds data/ and models/: this one, unless NYAN_ROOT names another.
ROOT = _os.environ.get("NYAN_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import glob, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
R = Path(f"{ROOT}/data/distill/review")
owner = {Path(f).stem: json.load(open(f)) for f in glob.glob(str(R / "answers_dump/answers/*.json"))}
posts = {pid: json.load(open(R / f"seed/{pid}.json")) for pid in owner}
# The two answers the IPTC reading overrules, which the owner accepted with the new rules.
RULED = {"p017": "not_news", "p018": "economy"}
import multi_label as m
ids = sorted(owner)
ans = m.label([posts[p]["text"] for p in ids], workers=4)
def right(pid, cat, ruled):
    o = owner[pid]; want = {RULED[pid]} if ruled and pid in RULED else {o["category"], *o.get("also_ok", [])}
    return cat in want
for ruled in (False, True):
    hits = sum(a is not None and right(p, a["category"], ruled) for p, a in zip(ids, ans))
    print(f"{os.environ.get('PROMPT','v1')}: {hits}/{len(ids)} owner answers{' (with the two rulings)' if ruled else ''}")
for p, a in zip(ids, ans):
    if a and not right(p, a["category"], True):
        print(f"  {p} owner {RULED.get(p, owner[p]['category'])} model {a['category']} | {posts[p]['text'][:90]!r}")
