"""What do the rubric patterns take out of the feed, and what routine is still in it?

An editorial rule such as «a power-outage schedule is not news» ends up in three
places: the category rules the teacher labels by (`cat_en.json`), a pattern in
`configs/annotator_config.json` for the wording that is clear enough to match,
and, at the next retraining, the model itself. This script is for the middle
step, run on a dump of production posts:

  taken     posts the patterns catch that the model called news, by pattern,
            with examples: read them before merging a new pattern.
  repeated  headings one channel posts again and again that still pass as news.
            The same story told by twenty channels is news; the same heading
            from one channel five times a week is a rubric, and that is where
            the next rule usually is.

The dump is the jsonl `drift.py` and the audits read: url, channel_id,
patched_text, category.

Usage: routine_candidates.py <posts.jsonl> [taken|repeated] [--min 5]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("NYAN_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
from nyan.rubrics import RubricDetector  # noqa: E402


def heading(text: str) -> str:
    """The first line with what changes from day to day taken out of it."""
    line = re.sub(r"[^\w\s]", "", re.sub(r"\d+", "N", text.split("\n")[0].lower()))
    return " ".join(line.split()[:5])


def show(text: str) -> str:
    return text[:140].replace("\n", " / ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("posts")
    parser.add_argument("mode", nargs="?", default="taken", choices=["taken", "repeated"])
    parser.add_argument("--min", type=int, default=5, help="repeats that make a heading a rubric")
    parser.add_argument("--config", default=str(ROOT / "configs/annotator_config.json"))
    args = parser.parse_args()

    detector = RubricDetector(json.load(open(args.config))["rubric_detector"])
    rows = [json.loads(line) for line in open(args.posts)]
    news = [r for r in rows if r.get("patched_text") and r.get("category") != "not_news"]
    print(f"{len(rows)} posts, {len(news)} the model called news")

    if args.mode == "taken":
        taken: dict[str, list[dict]] = defaultdict(list)
        for row in news:
            pattern = detector.explain(row["patched_text"])
            if pattern:
                taken[pattern].append(row)
        print(f"the patterns take {sum(map(len, taken.values()))} of them")
        for pattern, found in sorted(taken.items(), key=lambda item: -len(item[1])):
            print(f"\n{len(found)}  {pattern[:110]}")
            for heading_, count in Counter(heading(r["patched_text"]) for r in found).most_common(6):
                print(f"    {count:4}  {heading_}")
        return

    left = [r for r in news if not detector(r["patched_text"])]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in left:
        key = heading(row["patched_text"])
        if len(key) >= 10:
            groups[(row["channel_id"], key)].append(row)
    repeated = sorted((g for g in groups.values() if len(g) >= args.min), key=len, reverse=True)
    print(f"{len(repeated)} headings repeated {args.min}+ times by one channel, "
          f"{sum(map(len, repeated))} posts")
    for found in repeated:
        top = Counter(r["category"] for r in found).most_common(1)[0][0]
        print(f"{len(found):4}  {found[0]['channel_id']:24} {top:13} {show(found[-1]['patched_text'])}")


if __name__ == "__main__":
    main()
