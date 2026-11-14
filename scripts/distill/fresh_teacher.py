"""Teacher labels (prompt v1, what the student learned from) for the first N fresh posts,
scored against the student's answers in fresh_student.jsonl.

Usage: OPENLUX_KEY=... TAG=_fresh fresh_teacher.py <n>
"""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import multi_label as ml
n = int(sys.argv[1])
fresh = ml.read_jsonl(ml.D / "fresh.jsonl")[:n]
ans = ml.label([p["text"] for p in fresh])
ml.write_jsonl(ml.D / "fresh_teacher.jsonl", ({"url": p["url"], **(a or {})} for p, a in zip(fresh, ans)))
print(f"labelled {sum(a is not None for a in ans)}/{n}, filtered {ml.filtered[0]}, ${ml.cost():.4f}")
student = {r["url"]: r for r in ml.read_jsonl(ml.D / "fresh_student.jsonl")}
pairs = [(a, student[p["url"]], p) for p, a in zip(fresh, ans) if a and p["url"] in student]
ok = sum(a["category"] == s["category"] for a, s, _ in pairs)
prod = sum(a["category"] == p.get("category") for a, _, p in pairs)
print(f"category vs teacher: student {100*ok/len(pairs):.1f}%, prod head {100*prod/len(pairs):.1f}%  (n={len(pairs)})")
news = [(a, s) for a, s, _ in pairs if a["category"] != "not_news"]
print(f"news dropped by student {sum(s['category'] == 'not_news' for _, s in news)}/{len(news)}")
for f in ["topic", "scope", "region", "role"]:
    xs = [(a.get(f), s[f]) for a, s, _ in pairs if a.get(f) is not None]
    print(f"  {f}: {100*sum(x == y for x, y in xs)/len(xs):.0f}%")
print("disagreements:", Counter((a["category"], s["category"]) for a, s, _ in pairs if a["category"] != s["category"]).most_common(10))
