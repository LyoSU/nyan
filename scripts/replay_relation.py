"""Ask the relation judge again about a week of published stories, two ways.

Every story the feed published is replayed at the moment it was published: the
candidates are the posts already out in the same issue that `nearest_clusters`
would have offered, and `judge_relation` is asked what the story is to them —
once with the candidates as the judge saw them (one channel's text), once as
the reader saw them (the post's headline and prose). Stories folded in as
"same" never became posts of their own, so they are not here: what this can
show is which published stories each way would have folded in, and whether
those were duplicates or news.

Read-only against production; the judge is asked through `nyan.openai` with
whatever LLM_* the environment gives, and every answer is kept in a cache
file, so a second run is free.

Usage: LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=... \
       replay_relation.py [--days 7] [--others 60] [--out data/relation_replay.jsonl]
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import nyan.openai
import nyan.relation as relation
from nyan.clusters import Cluster
from nyan.mongo import get_clusters_collection
from nyan.util import get_current_ts, ts_to_dt


ap = argparse.ArgumentParser()
ap.add_argument("--days", type=float, default=7)
ap.add_argument("--others", type=int, default=60, help="stories sampled beyond the likely duplicates")
ap.add_argument("--out", default="data/relation_replay.jsonl")
ap.add_argument("--mongo", default="configs/mongo_config.json")
ap.add_argument("--dry", action="store_true", help="only count what would be asked")
ap.add_argument("--clids", type=int, nargs="*", default=[], help="stories to ask about whatever else is sampled")
ap.add_argument("--repeat", type=int, default=1, help="ask each story this many times: the judge is not deterministic")
args = ap.parse_args()

usage: Counter[str] = Counter()
client = nyan.openai.get_client()
create = client.chat.completions.create


def counted_create(**kwargs: Any) -> Any:
    response = create(**kwargs)
    if response.usage:
        with lock:
            usage["in"] += response.usage.prompt_tokens
            usage["out"] += response.usage.completion_tokens
    with lock:
        usage["calls"] += 1
    return response


client.chat.completions.create = counted_create  # type: ignore[method-assign]

since = get_current_ts() - int(args.days * 86400)
stored = get_clusters_collection(args.mongo).find(
    {"create_time": {"$gte": since - 86400}, "messages.0": {"$exists": True}}
)
published = sorted((Cluster.fromdict(d) for d in stored), key=lambda c: c.create_time or 0)
print(f"{len(published)} published stories", flush=True)


def words(text: str | None) -> set[str]:
    return set(re.findall(r"\w{4,}", (text or "").lower()))


def overlap(a: Cluster, b: Cluster) -> float:
    x, y = words(a.stored_headline), words(b.stored_headline)
    return len(x & y) / max(1, len(x | y))


cases = []
for index, story in enumerate(published):
    if (story.create_time or 0) < since:
        continue
    issues = {m.issue for m in story.messages}
    # What was out when this story went out, in the issue it went to.
    before = [
        c for c in published[:index]
        if {m.issue for m in c.messages} & issues
        and abs(c.pub_time - (story.create_time or 0)) <= 24 * 3600
    ]
    candidates = relation.nearest_clusters(story, before)
    if candidates:
        cases.append((story, candidates))

suspect = [case for case in cases if overlap(case[0], case[1][0]) >= 0.3]
rest = [case for case in cases if overlap(case[0], case[1][0]) < 0.3]
random.Random(0).shuffle(rest)
chosen = suspect + rest[: args.others]
chosen += [case for case in cases if case[0].clid in args.clids and case not in chosen]
print(f"{len(cases)} stories had candidates; asking about {len(suspect)} with a close headline and {min(args.others, len(rest))} others", flush=True)

# Part of the cache key: an answer to another version of the rules is not an answer.
with open(os.path.join(os.path.dirname(relation.__file__), "prompts", "relation.txt"), "rb") as rules:
    PROMPT = hashlib.sha256(rules.read()).hexdigest()[:12]
cache: dict[str, dict[str, Any]] = {}
lock = threading.Lock()
if os.path.exists(args.out):
    with open(args.out) as r:
        for line in r:
            row = json.loads(line)
            cache[row["key"]] = row


def ask(way: str, story: Cluster, candidates: list[Cluster], rep: int = 0) -> dict[str, Any]:
    key = hashlib.sha256(
        json.dumps([way, story.clid, [c.clid for c in candidates], os.getenv("LLM_MODEL"), PROMPT, rep]).encode()
    ).hexdigest()[:20]
    if key in cache:
        return cache[key]
    verdict = relation.judge_relation(story, candidates)
    row = {
        "key": key, "way": way, "clid": story.clid, "rep": rep, "prompt": PROMPT,
        "verdict": verdict.verdict,
        "match": verdict.cluster.clid if verdict.cluster else None,
    }
    # Kept as it comes: a run that dies halfway has still paid for these.
    with lock, open(args.out, "a") as w:
        w.write(json.dumps(row, ensure_ascii=False) + "\n")
        cache[key] = row
    return row


if args.dry:
    sys.exit(0)
answers = {}
original = relation._as_published
for way in ("old", "pub"):
    # "old" shows the candidates as the judge saw them before `_as_published`:
    # the one channel's text, the same material the story itself gets. Swapped
    # for the whole pass, so the calls inside it can run side by side.
    if way == "old":
        relation._as_published = relation._as_material
    try:
        with ThreadPoolExecutor(8) as pool:
            asked = [(case, rep) for rep in range(args.repeat) for case in chosen]
            answers[way] = list(pool.map(lambda job, way=way: ask(way, job[0][0], job[0][1], job[1]), asked))[: len(chosen)]
    finally:
        relation._as_published = original

by_clid = {c.clid: c for c in published}
flips: Counter[tuple[str, str]] = Counter()
print()
for (story, candidates), old, new in zip(chosen, answers["old"], answers["pub"]):
    was = "follow_up" if story.reply_to_headline else "unrelated"
    flips[(old["verdict"], new["verdict"])] += 1
    if old["verdict"] == new["verdict"] and old["verdict"] == was:
        continue
    target = by_clid.get(new["match"] or old["match"]) or candidates[0]
    print(
        f"[{story.clid}] then {was:9} old {old['verdict']:9} pub {new['verdict']:9} | "
        f"{ts_to_dt(story.create_time or 0):%d.%m %H:%M} {story.stored_headline}\n"
        f"{'':>10} vs [{target.clid}] {ts_to_dt(target.create_time or 0):%d.%m %H:%M} {target.stored_headline}"
    )
print("\n(old, pub) verdicts:", dict(flips))
if args.repeat > 1:
    print("\nhow often each story was judged the same event, of", args.repeat)
    for story, _ in chosen:
        same = {way: sum(r["verdict"] == "same" for r in cache.values()
                         if r["clid"] == story.clid and r["way"] == way and r.get("prompt") == PROMPT)
                for way in ("old", "pub")}
        print(f"  [{story.clid}] old {same['old']} pub {same['pub']} | {story.stored_headline}")
print(f"{usage['calls']} calls, {usage['in']} in / {usage['out']} out tokens")
