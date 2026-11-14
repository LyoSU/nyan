"""Replay a recorded stretch of the feed through the publish boundary, old rules against new.

Every story the feed published is put back at the moment it went out: the
posts already out are the ones created before it, and the story is asked what
it is to them exactly as the daemon would ask now. What production actually did
is the "old" side — it is on record — so the LLM is only asked for the new
side, and only about stories where the new rules change what the judge sees.

Read-only against production: the clusters are read once (`--pull`) into a
local dump and everything after that works from the dump. The LLM is reached
through a local CLI, one process at a time with stdin closed, and every answer
is cached on disk by the exact prompt, so a second run costs nothing.

Usage:
    feed_replay.py --pull --mongo configs/mongo_config.json --dump week.jsonl \
        --since 2026-11-04
    feed_replay.py --dump week.jsonl --from 2026-11-07 --to 2026-11-14 \
        --cache answers.jsonl --ask 80
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import nyan.relation as relation
from nyan.clusters import Cluster, Clusters
from nyan.util import ts_to_dt

# What the prod model is, as the CLI names it.
CLI_MODEL = "opencode-go/gpt-5.6-luna"


def pull(mongo_config: str, since: int, path: str) -> None:
    from nyan.mongo import get_clusters_collection

    collection = get_clusters_collection(mongo_config)
    count = 0
    with open(path, "w") as w:
        for record in collection.find({"create_time": {"$gte": since}}):
            record.pop("_id", None)
            # Refusals are long lists nothing here reads.
            record.pop("refused_docs", None)
            record.pop("refused_texts", None)
            w.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            count += 1
    print(f"{count} clusters written to {path}")


def load(path: str) -> list[Cluster]:
    clusters = []
    with open(path) as r:
        for line in r:
            clusters.append(Cluster.fromdict(json.loads(line)))
    return sorted(clusters, key=lambda c: c.create_time or 0)


class Tokens:
    """Counts the way the provider bills, when tiktoken is there to say so."""

    def __init__(self) -> None:
        try:
            import tiktoken

            self.encoding: Any = tiktoken.get_encoding("o200k_base")
        except ImportError:
            self.encoding = None

    def __call__(self, text: str) -> int:
        if self.encoding is None:
            return len(text) // 3
        return len(self.encoding.encode(text))


def reuse_answers(
    prompts_dir: str, cache: dict[str, str], reask: Any
) -> dict[str, str]:
    """Answers from a run whose prompts were kept, by the user half of the prompt.

    A change to the rules changes every prompt, and asking every story again
    to see what one rule moved costs a full replay. Stories whose old answer
    `reask` says the rule could move are asked again; the rest keep their
    answer as long as they are shown exactly as before. A story whose context
    changed because an earlier answer changed is asked too.
    """
    kept: dict[str, str] = {}
    for name in os.listdir(prompts_dir):
        key = name.removesuffix(".json")
        if key not in cache:
            continue
        with open(os.path.join(prompts_dir, name)) as r:
            messages = json.load(r)
        if not reask(cache[key]):
            kept[str(messages[-1]["content"])] = cache[key]
    return kept


class CliLLM:
    """`openai_completion` over a local CLI: sequential, stdin closed, cached."""

    def __init__(self, cache_path: str, budget: int) -> None:
        self.cache_path = cache_path
        self.budget = budget
        self.asked = 0
        self.cache: dict[str, str] = {}
        self.usage: Counter[str] = Counter()
        self.tokens = Tokens()
        # Answers to carry over from a run under earlier rules, by the user half
        # of the prompt; see `reuse_answers`.
        self.reuse: dict[str, str] = {}
        self.reused = 0
        if os.path.exists(cache_path):
            with open(cache_path) as r:
                for line in r:
                    row = json.loads(line)
                    self.cache[row["key"]] = row["answer"]

    def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        prompt = "\n\n".join(str(m["content"]) for m in messages)
        purpose = kwargs.get("purpose") or kwargs.get("prompt_cache_key") or "-"
        self.usage[f"{purpose}:calls"] += 1
        self.usage[f"{purpose}:prompt"] += self.tokens(prompt)
        key = hashlib.sha256(f"{CLI_MODEL}\n{prompt}".encode()).hexdigest()[:24]
        # To re-ask a few answers under a changed prompt without replaying
        # everything: REPLAY_PROMPTS=dir keeps every prompt under its key.
        if keep := os.environ.get("REPLAY_PROMPTS"):
            with open(os.path.join(keep, f"{key}.json"), "w") as w:
                json.dump(messages, w, ensure_ascii=False)
        user = str(messages[-1]["content"])
        if key in self.cache:
            answer = self.cache[key]
        elif user in self.reuse:
            # The same story shown the same way under an earlier version of the
            # rules, and its answer was not one the change is about.
            answer = self.reuse[user]
            self.reused += 1
        else:
            if self.asked >= self.budget:
                raise RuntimeError("replay budget spent")
            self.asked += 1
            answer = self._ask(prompt)
            with open(self.cache_path, "a") as w:
                w.write(json.dumps({"key": key, "answer": answer}, ensure_ascii=False) + "\n")
            self.cache[key] = answer
        self.usage[f"{purpose}:completion"] += self.tokens(answer)
        return answer

    def _ask(self, prompt: str) -> str:
        for attempt in range(3):
            done = subprocess.run(
                ["opencode", "run", "--standalone", "-m", CLI_MODEL, prompt],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=300,
            )
            out = done.stdout
            # The CLI prints a banner line before the answer.
            if "{" in out:
                return out[out.find("{") : out.rfind("}") + 1]
            time.sleep(2 + attempt * 5)
        return out


def old_candidates(story: Cluster, published: Sequence[Cluster], now: int) -> list[Cluster]:
    """What `nearest_clusters` offered before this branch, as of `now`.

    A day counted from each post's first source, a post whose sources looked
    newer than the story's dropped, and no notion of threads.
    """
    vector = _unit(story.embedding)
    if vector is None:
        return []
    story_time = _percentile_at(story, now)
    scored = []
    for index, candidate in enumerate(published):
        if abs(candidate.pub_time - now) > 24 * 3600:
            continue
        if _percentile_at(candidate, now) > story_time:
            continue
        other = _unit(candidate.embedding)
        if other is None or other.shape != vector.shape:
            continue
        similarity = float(vector @ other)
        if similarity >= relation.CANDIDATE_SIMILARITY:
            scored.append((-similarity, index, candidate))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [candidate for _, _, candidate in scored[: relation.MAX_CANDIDATES]]


def new_candidates(story: Cluster, before: Clusters, issue: str, now: int) -> list[Cluster]:
    return relation.nearest_clusters(
        story,
        before.get_embedded_clusters(now, issue, relation.CANDIDATE_WINDOW),
        now=now,
    )


def _percentile_at(cluster: Cluster, now: int) -> int:
    """The story's 20th-percentile source time, counting only sources out by `now`."""
    times = sorted(d.pub_time for d in cluster.docs if d.pub_time <= now) or sorted(
        d.pub_time for d in cluster.docs
    )
    return times[len(times) // 5]


def _unit(embedding: Sequence[float] | None) -> Any:
    if not embedding:
        return None
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else None


def similarity(a: Cluster, b: Cluster) -> float:
    x, y = _unit(a.embedding), _unit(b.embedding)
    if x is None or y is None or x.shape != y.shape:
        return 0.0
    return float(x @ y)


def label(cluster: Cluster) -> str:
    when = ts_to_dt(cluster.create_time or cluster.pub_time).strftime("%d.%m %H:%M")
    title = cluster.stored_headline or cluster.annotation_doc.patched_text.split("\n")[0]
    return f"[{cluster.clid}] {when} {title[:90]}"


def day(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%d").timestamp())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--mongo", default="configs/mongo_config.json")
    ap.add_argument("--since", default="2026-11-04")
    ap.add_argument("--from", dest="start", default="2026-11-07")
    ap.add_argument("--to", dest="end", default="2026-11-14")
    ap.add_argument("--cache", default="feed_replay_answers.jsonl")
    ap.add_argument("--ask", type=int, default=0, help="most new LLM calls to make")
    ap.add_argument("--min-new", type=float, default=0.95, help="ask only when a candidate this close is new")
    ap.add_argument("--clids", type=int, nargs="*", default=[], help="ask only about these stories")
    args = ap.parse_args()

    if args.pull:
        pull(args.mongo, day(args.since), args.dump)
        return

    clusters = [c for c in load(args.dump) if c.messages and c.create_time]
    start, end = day(args.start), day(args.end)
    llm = CliLLM(args.cache, args.ask)
    relation.openai_completion = llm  # type: ignore[assignment]

    counts: Counter[str] = Counter()
    changed: list[tuple[Cluster, list[Cluster], list[Cluster]]] = []
    for index, story in enumerate(clusters):
        now = story.create_time or 0
        if not start <= now < end:
            continue
        issue = story.messages[0].issue
        out = [c for c in clusters[:index] if any(m.issue == issue for m in c.messages)]
        before = Clusters()
        for cluster in out:
            before.add(cluster)
        old = old_candidates(story, out, now)
        new = new_candidates(story, before, issue, now)
        counts["posts"] += 1
        counts["old_candidates"] += len(old)
        counts["new_candidates"] += len(new)
        counts["posts_old_empty"] += not old
        counts["posts_new_empty"] += not new
        fresh = [c for c in new if c not in old]
        if fresh:
            counts["posts_with_a_new_candidate"] += 1
        if args.clids:
            if story.clid in args.clids:
                changed.append((story, old, new))
        elif any(similarity(story, c) >= args.min_new for c in fresh):
            changed.append((story, old, new))

    print("candidates:", dict(counts))
    print(f"{len(changed)} posts gained a candidate at >= {args.min_new}")

    verdicts: Counter[str] = Counter()
    for story, _old, new in changed:
        then = "follow_up" if story.reply_to_clid else "standalone"
        # A spent budget surfaces as FAILED: the judge catches everything.
        judged = relation.judge_relation(story, new)
        verdicts[f"{then} -> {judged.verdict}"] += 1
        if judged.verdict in (relation.UNRELATED, relation.FAILED):
            if args.clids:
                print(f"\n{judged.verdict.upper():9} {label(story)}")
            continue
        target = judged.cluster
        assert target is not None
        was = next((c for c in clusters if c.clid == story.reply_to_clid), None)
        if was is target and not args.clids:
            continue
        print(f"\n{judged.verdict.upper():9} {label(story)}")
        print(f"   now under {label(target)}  cos {similarity(story, target):.3f}")
        print(f"   then      {label(was) if was else 'standalone'}")

    print("\nverdicts (production then -> judge with new candidates):", dict(verdicts))
    print(f"LLM: {llm.asked} new calls; usage incl. cached answers: {dict(llm.usage)}")


if __name__ == "__main__":
    main()
