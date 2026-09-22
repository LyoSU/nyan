"""What the student-aware ranker would have done to the feed, from its shadow.

`nyan/rank_shadow.py` records every new story it would have published or held
where the real ranker did not. This joins those records with what was actually
posted, and answers three questions per kind of change:

- publish, later posted anyway: how much sooner it would have been out;
- publish, never posted: stories the feed would have gained — read the titles,
  because this is where noise would come in;
- hold, posted anyway: stories the feed would have lost — read these too.

Reads production Mongo and writes nothing.

    python scripts/rank_shadow.py [days] [mongo_config]
"""

import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any

from nyan.mongo import get_clusters_collection, get_rank_shadow_collection


def posted(mongo_config: str, urls: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Posted clusters by the url of every document in them, for the urls asked."""
    by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    collection = get_clusters_collection(mongo_config)
    for start in range(0, len(urls), 500):
        chunk = urls[start : start + 500]
        fields = {"docs.url": 1, "messages.issue": 1, "create_time": 1}
        for cluster in collection.find({"docs.url": {"$in": chunk}}, fields):
            for doc in cluster.get("docs", []):
                if doc.get("url") in chunk:
                    by_url[doc["url"]].append(cluster)
    return by_url


def main(days: float = 3, mongo_config: str = "configs/mongo_config.json") -> None:
    since = int(time.time() - days * 86400)
    records = list(get_rank_shadow_collection(mongo_config).find({"ts": {"$gte": since}}))
    if not records:
        print(f"no shadow records in the last {days:g} days")
        return
    by_url = posted(mongo_config, sorted({r["first_url"] for r in records}))

    groups: dict[str, list[tuple[dict[str, Any], int | None]]] = defaultdict(list)
    for r in records:
        issues = [
            (c.get("create_time"), m.get("issue"))
            for c in by_url.get(r["first_url"], [])
            for m in c.get("messages", [])
        ]
        times = [t for t, issue in issues if issue == r["issue"] and t]
        when = min(times) if times else None
        outcome = "posted" if when else "never posted"
        groups[f"{r['change']}, {outcome}"].append((r, when))

    print(f"{len(records)} shadow records over {days:g} days")
    print("by reason: " + ", ".join(f"{k} {n}" for k, n in Counter(
        f"{r['change']}:{'+'.join(r['reasons']) or '-'}" for r in records).most_common()))
    for name in ["publish, posted", "publish, never posted", "hold, posted", "hold, never posted"]:
        rows = groups.get(name, [])
        print(f"\n== {name}: {len(rows)}")
        if name == "publish, posted" and rows:
            leads = [(when - r["ts"]) / 60 for r, when in rows if when and when > r["ts"]]
            if leads:
                print(f"   sooner by median {statistics.median(leads):.0f} min, "
                      f"max {max(leads):.0f} min ({len(leads)} of {len(rows)} came out later)")
        if name in ("publish, never posted", "hold, posted"):
            for r, _ in sorted(rows, key=lambda x: -x[0]["ts"])[:20]:
                s = r.get("signals") or {}
                print(f"   [{r['issue']:8} {'+'.join(r['reasons']):12} sig {s.get('significance', 0):.1f}"
                      f" urg {s.get('urgent', 0):.2f} src {r['sources']:.1f}/{r['min_channels']}] {r['title'][:90]}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(float(args[0]) if args else 3, *(args[1:2]))
