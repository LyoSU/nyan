"""Which of the rank shadow's records a different student would still make.

Reads the records dumped from production with their posts' texts, rates every
post with each model, and redoes `nyan/rank_shadow.py`'s cluster vote: one
vote per owner, significance the median of the votes, urgency the share of
votes above 0.5. Only the reasons are redone, not the ranking: a record whose
reasons all vanish would not have been made, one that keeps some still would.

Usage: shadow_rescore.py <records.json> <major> <minor> [urgent per-post cut]
  thresholds for the new models; v1 keeps 3.8 / 1.8 / 0.5.
"""
import json, statistics, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from students import D, Model, SCALE

records = json.load(open(sys.argv[1]))
MAJOR, MINOR = float(sys.argv[2]), float(sys.argv[3])
CUT = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
texts = sorted({d["text"] for r in records for d in r["docs"].values()})
at = {t: i for i, t in enumerate(texts)}

scores = {}
v1 = Model("e5_multi"); Za = v1.embed(texts)
h3c = torch.load(D / "model_e5_multi_h3c.pt", map_location="cpu")["heads"]
scores["v1"] = (v1.head("significance", Za) @ SCALE, v1.head("urgent", Za))
scores["h3c"] = (v1.head("significance", Za, h3c) @ SCALE, v1.head("urgent", Za, h3c))
del v1.enc; (torch.backends.mps.is_available() and torch.mps.empty_cache())
v3c = Model("e5_multi_v3c"); Zb = v3c.embed(texts)
scores["v3c"] = (v3c.head("significance", Zb) @ SCALE, v3c.head("urgent", Zb))

def reasons(r, model, major, minor, cut):
    sig, urg = scores[model]
    by_owner = {}
    for d in r["docs"].values():
        by_owner.setdefault(d.get("master") or d["channel"], d["text"])
    if len(by_owner) < 2:
        return None
    s = statistics.median(sig[at[t]] for t in by_owner.values())
    u = sum(urg[at[t]] > cut for t in by_owner.values()) / len(by_owner)
    out = ["major"] if s >= major else ["minor"] if s <= minor else []
    # The shadow also needs the story to be young: its age when recorded.
    if u >= 0.5 and r["age"] < 30 * 60:
        out.append("urgent")
    return out, round(s, 2), round(u, 2)

kept = Counter()
for r in records:
    row = [f"{r['change']:7} {','.join(r['reasons']) or '-':13}"]
    for m, args in (("v1", (3.8, 1.8, 0.5)), ("h3c", (MAJOR, MINOR, CUT)), ("v3c", (MAJOR, MINOR, CUT))):
        got = reasons(r, m, *args)
        if got is None:
            row.append(f"{m} -"); continue
        rs, s, u = got
        kept[m, bool(rs)] += 1
        row.append(f"{m} {','.join(rs) or '—':13} {s:.2f}/{u:.2f}")
    print(" | ".join(row), "|", r["title"][:60])
print({f"{m} {'kept' if k else 'gone'}": n for (m, k), n in sorted(kept.items())})
