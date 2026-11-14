"""Replay a stretch of the feed through the editor, against what production published.

The documents of the stretch are fed to the real clusterer every `--step`
minutes, as the crawler would have delivered them, and the daemon — with the
editor switched on — is run over the result exactly as in production: the
same gate, the same memo of decisions, the same actions. Telegram and the
writer are faked; the editor is asked through a local CLI, one process at a
time with stdin closed, and every answer is cached by its prompt so a rerun is
free. The writer is never called: its prompt is rendered and counted, so the
cost of what would have been written is in the totals.

The channel's history before the stretch is production's own posts, so the
editor starts the stretch knowing what readers had already been told.

Then the same measures are taken over the posts production published in the
stretch and over the posts the replay published: posts per day, series of
posts on one story inside 90 minutes, posts under three minutes apart,
suspected duplicates, time from the first source and from the fourth
newsroom to the post, and LLM calls and tokens per day. Every story that
changed sides is printed so it can be read.

Usage:
    editor_replay.py --docs docs.jsonl --emb docs_emb.npy --dump dump.jsonl \
        --from "2026-11-11 06:00" --hours 24 --cache answers.jsonl --ask 150
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import pairwise
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import nyan.clusters
import nyan.daemon as daemon_module
import nyan.editor as editor
import nyan.relation as relation
from nyan.client import MessageId
from nyan.clusterer import Clusterer
from nyan.clusters import Cluster, Clusters, render_prompt
from nyan.daemon import Daemon
from nyan.document import Document
from nyan.rank_shadow import signals
from nyan.ranker import Ranker, independent_sources
from nyan.util import normalize_url, ts_to_dt

from feed_replay import CliLLM, Tokens, load  # type: ignore[import-not-found]

# How long the crawler takes to deliver a post, roughly: channels are read
# every five to fifteen minutes.
CRAWL_LAG = 5 * 60

# The ranker's pool: nothing older than this is ranked in any issue.
POOL = 6 * 3600

# Two posts this close in meaning, half a day apart at most, and not one a
# reply to the other, are counted as a suspected duplicate. The same measure
# for production and for the replay; the pairs are printed to be read.
DUPLICATE_SIMILARITY = 0.95
DUPLICATE_WINDOW = 12 * 3600

# Posts on one story: this close, inside this window.
SERIES_SIMILARITY = 0.93
SERIES_WINDOW = 90 * 60


def ts(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%d %H:%M").timestamp())


def load_docs(path: str, emb_path: str, start: int, end: int) -> list[Document]:
    vectors = np.load(emb_path, mmap_mode="r")
    docs = []
    with open(path) as r:
        for index, line in enumerate(r):
            record = json.loads(line)
            pub_time = int(record.get("pub_time") or 0)
            if not start - POOL <= pub_time < end:
                continue
            record.pop("embedding_key", None)
            known = set(Document.__dataclass_fields__)
            doc = Document(**{k: v for k, v in record.items() if k in known})
            doc.embedding = [float(x) for x in vectors[index]]
            docs.append(doc)
    return sorted(docs, key=lambda d: d.pub_time)


class FakeRenderer:
    """Counts what the writer would have been asked, without asking it."""

    def __init__(self, tokens: Tokens) -> None:
        self.tokens = tokens
        self.calls = 0
        self.rewrites = 0
        self.prompt_tokens = 0

    def render_cluster(
        self, cluster: Cluster, issue_name: str, post_format: str | None = None
    ) -> str:
        if cluster.messages:
            # A published post is rewritten only when the coverage ladder or
            # the editor says so; otherwise the edit reuses its text.
            saved = cluster.saved_analysis or {}
            if saved.get("generation") == cluster.generation and not cluster.force_rewrite:
                return "post"
            self.rewrites += 1
        docs = cluster.prompt_docs
        messages = render_prompt(
            "summary" if len(cluster.channels) > 1 else "headline",
            docs=docs,
            annotation_doc=cluster.annotation_doc,
            today="",
            reply_to_headline=cluster.reply_to_headline,
            reply_to_text=cluster.reply_to_text,
            reply_to_thread=cluster.reply_to_thread,
            previous_post="",
        )
        self.calls += 1
        self.prompt_tokens += sum(self.tokens(m["content"]) for m in messages)
        # What a reader would see as the headline: the editor is shown these.
        saved = cluster.saved_analysis or {}
        cluster.saved_analysis = {
            **saved,
            "headline": saved.get("headline")
            or cluster.annotation_doc.patched_text.split("\n")[0][:120],
            "generation": cluster.generation,
        }
        cluster.force_rewrite = False
        return "post"


class FakeClient:
    def __init__(self) -> None:
        self.next_id = 900000
        self.edits = 0

    def send_post(self, post: Any, issue_name: str, reply_to: int | None = None) -> MessageId:
        self.next_id += 1
        return MessageId(message_id=self.next_id, issue=issue_name)

    def update_post(self, message: MessageId, post: Any) -> bool:
        self.edits += 1
        return True

    def update_discussion_mapping(self, issue_name: str) -> None:
        pass

    def has_issue(self, issue_name: str) -> bool:
        return issue_name == "main"


def build_daemon(tokens: Tokens) -> Daemon:
    daemon = object.__new__(Daemon)
    with open("configs/daemon_config.json") as r:
        daemon.config = json.load(r)
    daemon.config["editor"] = {"enabled": True}
    daemon.ranker = Ranker("configs/ranker_config.json")
    daemon.renderer = FakeRenderer(tokens)  # type: ignore[assignment]
    daemon.client = FakeClient()  # type: ignore[assignment]
    return daemon


def unit(cluster: Cluster) -> Any:
    vector = np.asarray(cluster.embedding or [], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else None


def fourth_newsroom(cluster: Cluster) -> int | None:
    """When the story reached four newsrooms, counted as the ranker counts."""
    seen = Cluster()
    for doc in sorted(cluster.docs, key=lambda d: d.pub_time):
        seen.add(doc)
        if independent_sources(seen) >= 4:
            return doc.pub_time
    return None


def measure(posts: list[Cluster], days: float, name: str) -> dict[str, Any]:
    posts = sorted(posts, key=lambda c: c.create_time or 0)
    vectors = [unit(post) for post in posts]
    first = [(p.create_time or 0) - p.pub_time for p in posts]
    fourth = [
        (p.create_time or 0) - t for p in posts if (t := fourth_newsroom(p)) is not None
    ]
    close = sum(
        1
        for a, b in pairwise(posts)
        if (b.create_time or 0) - (a.create_time or 0) < 180
    )
    duplicates = []
    series_members: set[int] = set()
    for i, a in enumerate(posts):
        for j in range(i + 1, len(posts)):
            b = posts[j]
            delta = (b.create_time or 0) - (a.create_time or 0)
            if delta > DUPLICATE_WINDOW:
                break
            if vectors[i] is None or vectors[j] is None:
                continue
            similarity = float(vectors[i] @ vectors[j])
            linked = b.reply_to_clid == a.clid
            if similarity >= DUPLICATE_SIMILARITY and not linked:
                duplicates.append((similarity, a, b))
            if similarity >= SERIES_SIMILARITY and delta <= SERIES_WINDOW:
                series_members.update((i, j))
    # A series is three or more posts on one story inside 90 minutes.
    series = 0
    by_window: dict[int, list[int]] = defaultdict(list)
    for i in sorted(series_members):
        by_window[(posts[i].create_time or 0) // SERIES_WINDOW].append(i)
    series = sum(1 for members in by_window.values() if len(members) >= 3)

    def q(values: list[int], p: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        return values[min(len(values) - 1, int(len(values) * p))] / 60

    return {
        "name": name,
        "posts": len(posts),
        "posts_per_day": round(len(posts) / days, 1),
        "replies": sum(1 for p in posts if p.reply_to_clid),
        "under_3_min_apart": close,
        "series_3plus_in_90min": series,
        "suspected_duplicates": len(duplicates),
        "first_source_to_post_min": (round(q(first, 0.5)), round(q(first, 0.9))),
        "fourth_newsroom_to_post_min": (round(q(fourth, 0.5)), round(q(fourth, 0.9))),
        "_duplicates": duplicates,
    }


def label(cluster: Cluster) -> str:
    when = ts_to_dt(cluster.create_time or cluster.pub_time).strftime("%d.%m %H:%M")
    title = cluster.stored_headline or cluster.annotation_doc.patched_text.split("\n")[0]
    return f"{when} {title[:100]}"


def trace(needle: str, now: int, clusters: list[Cluster], asked: list[Cluster], daemon: Daemon) -> None:
    """Where one story stops: size, newsrooms, the student's reading, the gates."""
    chosen = {id(c) for c in asked}
    for cluster in clusters:
        if not any(needle in (doc.text or "") for doc in cluster.docs):
            continue
        found = signals(cluster)
        print(
            f"TRACE {ts_to_dt(now):%H:%M} docs={len(cluster.docs)} "
            f"sources={independent_sources(cluster):.1f} age={cluster.age // 60}m "
            f"signals={found and (round(found.significance, 1), round(found.urgent, 2))} "
            f"candidate={id(cluster) in chosen}/{len(asked)} issues={cluster.issues} "
            f"lang={sorted({d.language for d in cluster.docs})} "
            f"worth={id(cluster) in chosen and daemon.worth_asking(cluster)} "
            f"| {cluster.cropped_title[:60]}",
            file=sys.stderr,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--step", type=int, default=10, help="minutes between iterations")
    ap.add_argument("--cache", default="answers.jsonl")
    ap.add_argument("--ask", type=int, default=0)
    ap.add_argument("--min-sources", type=float, default=3.0)
    ap.add_argument("--dry", action="store_true", help="answer every question 'digest' without asking")
    ap.add_argument("--trace", help="print each iteration's gates for stories whose text has this")
    args = ap.parse_args()

    start = ts(args.start)
    end = start + int(args.hours * 3600)
    days = args.hours / 24

    production = [c for c in load(args.dump) if c.messages and c.create_time]
    history = [c for c in production if start - 3 * 86400 <= (c.create_time or 0) < start]
    actual = [c for c in production if start <= (c.create_time or 0) < end]
    actual = [c for c in actual if any(m.issue == "main" for m in c.messages)]

    docs = load_docs(args.docs, args.emb, start, end)
    print(f"{len(docs)} documents, {len(history)} posts of history, {len(actual)} published then")

    tokens = Tokens()
    llm = CliLLM(args.cache, args.ask)
    if args.dry:

        def answer(**kwargs: Any) -> str:
            llm.usage["editor:calls"] += 1
            llm.usage["editor:prompt"] += sum(tokens(m["content"]) for m in kwargs["messages"])
            return '{"relation": "new", "post": null, "decision": "digest"}'

        editor.openai_completion = answer  # type: ignore[assignment]
    else:
        editor.openai_completion = llm  # type: ignore[assignment]

    # The late-documents question still goes to the old judge in production; it
    # is counted here and answered "unrelated", which folds the documents in.
    def judge(story: Cluster, candidates: Any, reasoning_effort: Any = None) -> Any:
        llm.usage["relation:calls"] += 1
        return relation.Relation(relation.UNRELATED)

    daemon_module.judge_relation = judge  # type: ignore[assignment]
    daemon_module.notify_published = lambda clids: None  # type: ignore[assignment]

    daemon = build_daemon(tokens)
    daemon.config["editor"]["min_sources"] = args.min_sources
    clusterer = Clusterer("configs/clusterer_config.json")
    posted = Clusters()
    for post in history:
        posted.add(post)
    decisions: list[tuple[int, Cluster, editor.Decision]] = []
    seen: set[int] = set()
    original_ask = daemon.ask_editor

    def recording_ask(cluster: Cluster, posted_clusters: Clusters, issue_name: str) -> Any:
        decision = original_ask(cluster, posted_clusters, issue_name)
        if decision is not None and not decision.failed and id(decision) not in seen:
            seen.add(id(decision))
            decisions.append((now, cluster, decision))
        return decision

    daemon.ask_editor = recording_ask  # type: ignore[method-assign]

    now = start
    while now < end:
        clock = lambda moment=now: moment
        daemon_module.get_current_ts = clock  # type: ignore[assignment]
        nyan.clusters.get_current_ts = clock  # type: ignore[assignment]
        pool = [d for d in docs if now - POOL <= d.pub_time <= now - CRAWL_LAG]
        for doc in pool:
            doc.fetch_time = now
        if len(pool) > 1:
            clusters = clusterer(pool, posted.published_documents())
            candidates = daemon.ranker.candidates(clusters, args.min_sources)
            if args.trace:
                trace(args.trace, now, clusters, candidates.get("main", []), daemon)
            daemon.held_now.clear()
            for cluster in candidates.get("main", []):
                daemon.send_cluster(cluster, "main", posted, None, None)
        now += args.step * 60

    replayed = [
        c for c in posted.clid2cluster.values() if c.messages and start <= (c.create_time or 0) < end
    ]
    before = measure(actual, days, "production")
    after = measure(replayed, days, "editor")
    print()
    for key in before:
        if key.startswith("_") or key == "name":
            continue
        print(f"{key:30} {before[key]!s:>14} {after[key]!s:>14}")

    usage = dict(llm.usage)
    writer = daemon.renderer  # type: ignore[assignment]
    print(
        f"\nLLM per day (replay): editor {usage.get('editor:calls', 0) / days:.0f} calls, "
        f"{usage.get('editor:prompt', 0) / days / 1000:.0f}k prompt tokens, "
        f"{usage.get('editor:completion', 0) / days / 1000:.1f}k completion; "
        f"writer {writer.calls / days:.0f} calls, {writer.prompt_tokens / days / 1000:.0f}k prompt tokens "  # type: ignore[attr-defined]
        f"(counted, not asked); late-docs judge {usage.get('relation:calls', 0) / days:.0f} calls"
    )
    revisions = sum(len((c.saved_analysis or {}).get("revisions") or []) for c in actual)
    print(
        f"LLM per day (production, from storage): writer {len(actual) / days:.0f} first writes "
        f"+ {revisions / days:.0f} rewrites; the judge is not stored"
    )
    print(f"new CLI calls this run: {llm.asked}")

    print("\nDecisions:", Counter(f"{d.relation}/{d.decision}" for _, _, d in decisions))
    per_story = Counter(normalize_url(c.first_doc.url) for _, c, _ in decisions)
    print(f"Stories asked about: {len(per_story)}, asks per story: {Counter(per_story.values())}")
    print("Levels of published:", Counter(c.level for c in replayed))

    published_urls = {normalize_url(u) for c in replayed for u in c.urls}
    actual_urls = {normalize_url(u) for c in actual for u in c.urls}
    print("\n== Published then, not by the editor (what it decided) ==")
    last: dict[str, tuple[int, editor.Decision]] = {}
    for when, cluster, decision in decisions:
        for url in cluster.urls:
            last[normalize_url(url)] = (when, decision)
    for post in actual:
        urls = {normalize_url(u) for u in post.urls}
        if urls & published_urls:
            continue
        found = [last[u] for u in urls if u in last]
        said = (
            f"{found[-1][1].relation}/{found[-1][1].decision}: {found[-1][1].why}" if found else "never asked"
        )
        print(f"  {label(post)}\n      -> {said}")

    print("\n== Published by the editor, not then ==")
    for post in sorted(replayed, key=lambda c: c.create_time or 0):
        if {normalize_url(u) for u in post.urls} & actual_urls:
            continue
        print(f"  {label(post)}  [{post.level}{', ' + post.storyline if post.storyline else ''}]")

    print("\n== Folded into a post as new numbers or nothing new ==")
    for when, cluster, decision in decisions:
        if decision.relation in (editor.SAME, editor.UPDATE) and decision.post is not None:
            print(
                f"  {ts_to_dt(when):%d.%m %H:%M} {decision.relation:6} "
                f"{cluster.cropped_title[:80]}\n         into {label(decision.post)}"
            )

    print("\n== Storylines ==")
    lines: dict[str, list[Cluster]] = defaultdict(list)
    for post in posted.clid2cluster.values():
        if post.storyline:
            lines[post.storyline].append(post)
    for name, members in sorted(lines.items(), key=lambda item: -len(item[1])):
        if len(members) < 2:
            continue
        print(f"  {name} ({len(members)})")
        for post in sorted(members, key=lambda c: c.create_time or 0):
            print(f"      {label(post)}")

    for result in (before, after):
        print(f"\n== Suspected duplicates, {result['name']} ==")
        for similarity, a, b in result["_duplicates"]:
            print(f"  {similarity:.3f} {label(a)}\n        {label(b)}")


if __name__ == "__main__":
    main()
