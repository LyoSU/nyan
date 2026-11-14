"""One editorial decision per story: what it is to the channel, and whether to say it.

Before this, two mechanisms decided between them. The ranker let a story
through on counts — four newsrooms, fifteen minutes, a views border measured
against whatever else was in the pool — and the relation judge then asked
whether it was something the channel had told already. Over 7-14 November that
published a Lada in London, snow in Volyn and four GTA 6 posts while a strike
on Pavlohrad with three wounded stayed out, and it put every change in a
casualty count out as a post of its own: one strike, five posts in ninety
minutes. Neither mechanism could see what a reader of the channel sees, which
is the last few days of the channel itself.

So the question is asked once, of a model that is shown exactly that: the
channel's recent headlines with their times, the closest posts in full, and
the story's sources. The cheap mechanics stay where they are good — the
cosine picks which posts to show in full, the source count decides which
stories are worth asking about at all — and make no call of their own.

The answer has two halves. `relation` is what the story is to the channel:
the same event (nothing new, or new numbers), a consequence of a post, another
event on the same storyline, or nothing the channel has touched. `decision` is
whether to tell the readers now, later, in the digest, or not at all. A
storyline label comes with it at no extra cost, for the "more on this" line
and for the writer.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from nyan.clusters import Cluster, clean_boundary, judged_text_key, render_prompt
from nyan.document import Document
from nyan.openai import openai_completion
from nyan.ranker import independent_sources
from nyan.rank_shadow import signals
from nyan.relation import (
    CANDIDATE_SIMILARITY,
    OLDER_SIMILARITY,
    _unit,
    gap,
    nearest_clusters,
)
from nyan.util import ts_to_dt


SAME = "same"
UPDATE = "update"
FOLLOW_UP = "follow_up"
STORYLINE = "storyline"
NEW = "new"
RELATIONS = (SAME, UPDATE, FOLLOW_UP, STORYLINE, NEW)

PUBLISH = "publish"
WAIT = "wait"
DIGEST = "digest"
SKIP = "skip"
DECISIONS = (PUBLISH, WAIT, DIGEST, SKIP)

LEVELS = ("breaking", "normal", "minor")

# Not an answer the model gives: the call failed or what came back was not an
# answer. The story is held, exactly as for the relation judge.
FAILED = "failed"

# The channel's own recent headlines, all of them: what a reader scrolling up
# would see. A day of the feed is some fifty lines.
HISTORY_WINDOW = 24 * 3600
MAX_HISTORY = 60

# Older posts are listed too, but only the ones close to this story: three days
# of every headline would be most of the prompt, and almost none of it is
# about this story.
OLDER_WINDOW = 3 * 24 * 3600
MAX_OLDER = 10

# Posts shown with their text, not only the headline: the ones this story is
# most likely the same as.
MAX_NEAREST = 3

# The story's sources as the editor reads them. One per channel and one per
# text — channels copy each other word for word — the earliest first, each cut
# to a length that still carries the facts.
MAX_STORY_DOCS = 6
MAX_DOC_CHARS = 700


@dataclass(frozen=True)
class Decision:
    relation: str
    decision: str = SKIP
    post: Cluster | None = None
    storyline: str | None = None
    level: str = "normal"
    why: str = ""

    @property
    def failed(self) -> bool:
        return self.relation == FAILED


def decide(
    story: Cluster,
    published: Sequence[Cluster],
    now: int,
    reasoning_effort: str | None = None,
) -> Decision:
    """The editor's call on one story. Never raises; a failure is FAILED."""
    history, nearest = context(story, published, now)
    by_number = dict(enumerate(history, start=1))
    numbers = {id(post): number for number, post in by_number.items()}

    messages = render_prompt(
        "editor",
        history=[_history_line(post, number, now) for number, post in by_number.items()],
        nearest=[_nearest_entry(post, numbers[id(post)], story) for post in nearest],
        story=_story_entry(story, now),
    )
    try:
        content = openai_completion(
            messages=messages,
            response_format={"type": "json_object"},
            reasoning_effort=reasoning_effort,
            prompt_cache_key="editor",
        )
        parsed = json.loads(content[content.find("{") : content.rfind("}") + 1])
    except Exception:
        logging.exception("Editor could not decide on '%s'", story.cropped_title)
        return Decision(FAILED)
    return parse(parsed, by_number, story)


def parse(parsed: Any, by_number: dict[int, Cluster], story: Cluster) -> Decision:
    """The model's JSON as a decision, or FAILED when it is not one."""
    if not isinstance(parsed, dict):
        return Decision(FAILED)
    relation = parsed.get("relation")
    decision = parsed.get("decision")
    if relation not in RELATIONS or decision not in DECISIONS:
        logging.warning("Not a decision for '%s': %s", story.cropped_title, parsed)
        return Decision(FAILED)
    post = by_number.get(parsed["post"]) if isinstance(parsed.get("post"), int) else None
    if relation in (SAME, UPDATE, FOLLOW_UP) and post is None:
        # A link to nothing is no answer: acting on it would publish a repeat.
        logging.warning("'%s' names no post: %s", relation, parsed)
        return Decision(FAILED)
    storyline = parsed.get("storyline")
    if not isinstance(storyline, str) or not storyline.strip():
        storyline = post.storyline if post is not None and relation != NEW else None
    level = str(parsed["level"]) if parsed.get("level") in LEVELS else "normal"
    why = str(parsed.get("why") or "")
    return Decision(
        relation=relation,
        decision=decision,
        post=post if relation != NEW else None,
        storyline=storyline.strip() if storyline else None,
        level=level,
        why=why,
    )


def context(
    story: Cluster, published: Sequence[Cluster], now: int
) -> tuple[list[Cluster], list[Cluster]]:
    """What the editor is shown: the channel's recent posts, and the closest of them in full.

    The history is newest first, the day's posts all of them and the older
    ones only when close to this story. The posts in full come from the same
    list, picked by `nearest_clusters` — one per thread, a near-copy only
    past a day.
    """
    out = [
        post
        for post in published
        if post.messages and post.create_time and 0 <= now - post.create_time <= OLDER_WINDOW
    ]
    recent = [post for post in out if now - (post.create_time or 0) <= HISTORY_WINDOW]
    recent.sort(key=lambda post: post.create_time or 0, reverse=True)
    recent = recent[:MAX_HISTORY]

    vector = _unit(story.embedding) if story.embedding else None
    older_scored = []
    for post in out:
        if now - (post.create_time or 0) <= HISTORY_WINDOW or vector is None:
            continue
        other = _unit(post.embedding) if post.embedding else None
        if other is None or other.shape != vector.shape:
            continue
        similarity = float(vector @ other)
        if similarity >= OLDER_SIMILARITY:
            older_scored.append((similarity, post))
    older_scored.sort(key=lambda item: -item[0])
    older = [post for _, post in older_scored[:MAX_OLDER]]
    older.sort(key=lambda post: post.create_time or 0, reverse=True)

    history = recent + older
    nearest = nearest_clusters(
        story, history, limit=MAX_NEAREST, threshold=CANDIDATE_SIMILARITY, now=now
    )
    return history, nearest


def _history_line(post: Cluster, number: int, now: int) -> dict[str, Any]:
    published_at = post.create_time or post.pub_time
    return {
        "number": number,
        "time": ts_to_dt(published_at).strftime("%d.%m %H:%M"),
        "gap": _ago(now - published_at),
        "headline": post.stored_headline or _first_line(post),
        "storyline": post.storyline,
    }


def _nearest_entry(post: Cluster, number: int, story: Cluster) -> dict[str, Any]:
    prose = post.stored_summary.as_text() or clean_boundary(post.annotation_doc.patched_text)
    return {
        "number": number,
        "time": ts_to_dt(post.pub_time_percentile).strftime("%d.%m %H:%M"),
        "gap": gap(post.pub_time_percentile, story.pub_time_percentile),
        "text": "\n".join(part for part in (post.stored_headline, prose) if part),
    }


def _story_entry(story: Cluster, now: int) -> dict[str, Any]:
    sources = independent_sources(story)
    found = signals(story)
    described = ""
    if found is not None:
        described = f"важливість {found.significance:.1f} з 5"
        if found.urgent >= 0.5:
            described += ", термінове"
    return {
        "sources": f"{sources:.1f} незалежних редакцій ({len(story.channels)} каналів)",
        "first": ts_to_dt(story.pub_time).strftime("%d.%m %H:%M"),
        "now": ts_to_dt(now).strftime("%d.%m %H:%M"),
        "signals": described,
        "docs": [
            {
                "channel": doc.channel_title or doc.channel_id,
                "time": ts_to_dt(doc.pub_time).strftime("%H:%M"),
                "text": _crop(clean_boundary(doc.patched_text)),
            }
            for doc in story_docs(story)
        ],
    }


def story_docs(story: Cluster, limit: int = MAX_STORY_DOCS) -> list[Document]:
    """One source per channel and per text, earliest first.

    Channels copy each other word for word, and a story on thirty channels
    would otherwise be thirty copies of three texts.
    """
    seen_channels: set[str] = set()
    seen_texts: set[str] = set()
    chosen: list[Document] = []
    for doc in sorted(story.docs, key=lambda d: d.pub_time):
        if doc.monitor_only:
            continue
        owner = doc.master or doc.channel_id
        key = judged_text_key(doc)
        if owner in seen_channels or key is None or key in seen_texts:
            continue
        seen_channels.add(owner)
        seen_texts.add(key)
        chosen.append(doc)
        if len(chosen) >= limit:
            break
    return chosen


def _crop(text: str) -> str:
    if len(text) <= MAX_DOC_CHARS:
        return text
    cut = text[:MAX_DOC_CHARS]
    return cut[: cut.rfind(" ")] + "…"


def _first_line(post: Cluster) -> str:
    return clean_boundary(post.annotation_doc.patched_text).split("\n")[0][:150]


def _ago(seconds: int) -> str:
    minutes = max(seconds, 0) // 60
    if minutes < 60:
        return f"{minutes} хв тому"
    hours = round(minutes / 60)
    if hours < 48:
        return f"{hours} год тому"
    return f"{round(hours / 24)} дн тому"
