"""Whether a story about to be published is one the feed has already told.

The clusterer answers this by cosine distance between text embeddings, and on
production data it cannot: over one week, pairs of separately published posts
that are the same event and pairs that merely share a template overlap almost
entirely, giving an AUC of 0.70 and a best achievable accuracy of 74% for any
single threshold. The most similar pair of the whole week, at 0.984, was two
different nights of explosions over Kyiv.

So the cosine is used for what it is good at — narrowing hundreds of published
posts down to a couple of plausible ones — and the decision itself is read out
of the text by a model, once per post, on the boundary where a new message would
otherwise be sent.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nyan.clusters import Cluster, clean_boundary, render_prompt
from nyan.openai import openai_completion
from nyan.util import ts_to_dt


# The story is the same event as the candidate: an update, a correction, a
# detail added. The reader has already been told this.
SAME = "same"

# A different event that only makes sense next to the candidate: the next day of
# a protest, the response to a strike, the sentence after the arrest.
FOLLOW_UP = "follow_up"

# Nothing to do with any candidate.
UNRELATED = "unrelated"

VERDICTS = (SAME, FOLLOW_UP, UNRELATED)

# Not an answer the model gives: the question could not be asked, or what came
# back was not an answer. Callers hold the story and ask again later. Answering
# UNRELATED instead, as this did, publishes exactly the stories the judge exists
# to stop — the same outage that failed this call failed the summary too, and a
# week of production carried Moldova giving up Russian gas twice, 0.995 apart.
FAILED = "failed"

# How close a published post has to be before it is worth asking about. Set for
# recall, not precision: the model makes the call, so a candidate only has to be
# plausible. Measured over a week of production, this leaves 1.7 candidates per
# post on average and none at all for 45% of them.
CANDIDATE_SIMILARITY = 0.86

# At most this many go into one prompt. The distribution has a long tail — the
# 95th percentile is seven — and a story that is genuinely a development of
# something is a development of the closest thing, not the seventh closest.
MAX_CANDIDATES = 3

# How far back a story's thread is looked for, counted from when each post went
# out. Counted from its first channel instead, "the Kraków shooter died in
# hospital" no longer saw the shooting's own posts and went out standalone.
#
# A day and not more, until the judge can tell a thread from a storyline. A
# replay of 7-14 November with three days found the Kraków and Chornomorsk
# threads, but also hung "protests in France" under protests two days earlier
# and the daily blackout posts under each other: the judge has no answer for
# "the same storyline" but "follow_up". `candidate_window` in the daemon config
# widens it.
CANDIDATE_WINDOW = 24 * 3600

# Past this age a post has to be much closer to be offered. Nearly every pair of
# posts a day apart clears 0.86 — "вибухи в Києві" is written the same way every
# night — and three days of them crowd the day's own thread out of the three
# slots. The Kraków posts sat at 0.96-0.97 to each other across the day.
RECENT_WINDOW = 24 * 3600
OLDER_SIMILARITY = 0.94


@dataclass(frozen=True)
class Relation:
    verdict: str
    cluster: Cluster | None = None


def nearest_clusters(
    cluster: Cluster,
    published: Sequence[Cluster],
    limit: int = MAX_CANDIDATES,
    threshold: float = CANDIDATE_SIMILARITY,
    now: int | None = None,
) -> list[Cluster]:
    """The published posts this story might be about, closest first.

    One per thread. A story that ran as a chain of replies is one story to the
    reader, and its posts are each other's nearest neighbours: offered one by
    one they filled every slot, and nothing else was ever asked about. The
    thread is represented by its member closest to this story.

    No order in time is required. A post that went out is out, whenever its
    sources were written; dropping candidates whose sources were newer than
    this story's let "free travel in Kyiv" and a car burning after a drone
    strike go out twice, 13 and 4 minutes apart: the second cluster held an
    earlier channel's post, so it looked older than the post it repeated. Over
    the week that rule hid a closer candidate from 127 of 368 posts.
    """
    pivot = cluster.embedding
    if not pivot:
        return []
    vector = _unit(pivot)
    if vector is None:
        return []
    by_clid = {c.clid: c for c in published if c.clid is not None}

    best: dict[int, tuple[float, int, Cluster]] = {}
    for index, candidate in enumerate(published):
        if not candidate.messages:
            continue
        embedding = candidate.embedding
        if not embedding or len(embedding) != len(pivot):
            continue
        other = _unit(embedding)
        if other is None:
            continue
        similarity = float(vector @ other)
        floor = threshold
        if now is not None and now - (candidate.create_time or now) > RECENT_WINDOW:
            floor = max(threshold, OLDER_SIMILARITY)
        if similarity < floor:
            continue
        thread = thread_root(candidate, by_clid)
        # The index breaks ties without reaching for the cluster itself, which
        # has no ordering, and keeps the result stable across runs.
        scored = (-similarity, index, candidate)
        if thread not in best or scored[:2] < best[thread][:2]:
            best[thread] = scored

    ranked = sorted(best.values(), key=lambda item: (item[0], item[1]))
    return [candidate for _, _, candidate in ranked[:limit]]


def thread_root(cluster: Cluster, by_clid: dict[int, Cluster]) -> int:
    """The id of the first post of the thread this one stands in.

    Followed only as far as the posts at hand go: a parent that has left the
    window is where the thread starts as far as this pass is concerned.
    """
    seen: set[int] = set()
    current = cluster
    while (
        current.reply_to_clid is not None
        and current.reply_to_clid in by_clid
        and current.reply_to_clid not in seen
    ):
        seen.add(current.reply_to_clid)
        current = by_clid[current.reply_to_clid]
    return current.clid if current.clid is not None else -id(current)


def judge_relation(
    cluster: Cluster,
    candidates: Sequence[Cluster],
    reasoning_effort: str | None = None,
) -> Relation:
    """What this story is to the closest thing already published.

    Never raises: the daemon is a synchronous loop. A failure comes back as
    FAILED, which is not a verdict — there is no safe guess between "already
    told" and "new", so the caller holds the story for the next iteration.
    """
    if not candidates:
        return Relation(UNRELATED)

    messages = render_prompt(
        "relation",
        story=_as_material(cluster),
        candidates=[
            _as_published(candidate, number, cluster)
            for number, candidate in enumerate(candidates, start=1)
        ],
    )
    try:
        content = openai_completion(
            messages=messages,
            response_format={"type": "json_object"},
            reasoning_effort=reasoning_effort,
            prompt_cache_key="relation",
        )
        parsed = json.loads(content[content.find("{") : content.rfind("}") + 1])
    except Exception:
        logging.exception("Could not judge what '%s' is", cluster.cropped_title)
        return Relation(FAILED)

    verdict = parsed.get("verdict") if isinstance(parsed, dict) else None
    match = parsed.get("match") if isinstance(parsed, dict) else None
    if verdict == UNRELATED:
        return Relation(UNRELATED)
    if verdict not in VERDICTS:
        logging.warning("Not a verdict for '%s': %s", cluster.cropped_title, parsed)
        return Relation(FAILED)
    if not isinstance(match, int) or not 1 <= match <= len(candidates):
        logging.warning("Verdict '%s' names no candidate: %s", verdict, match)
        return Relation(FAILED)

    chosen = candidates[match - 1]
    logging.info(
        "'%s' is %s of '%s'", cluster.cropped_title, verdict, chosen.cropped_title
    )
    return Relation(verdict, chosen)


def _as_material(cluster: Cluster, number: int = 0) -> dict[str, str]:
    """One story as the prompt sees it: when it happened, and what it says.

    The time is spelled out because it carries what the text cannot. Two nights
    of explosions over Kyiv are written in the same words, and the timestamps
    are the only thing that tells them apart.
    """
    return {
        "number": str(number),
        "time": ts_to_dt(cluster.pub_time_percentile).strftime("%d.%m.%Y, %H:%M"),
        "text": clean_boundary(cluster.annotation_doc.patched_text),
    }


def gap(earlier: int, later: int) -> str:
    """How long before the new story a post went out, in the words a reader uses.

    Spelled out because a timestamp alone was not read. "07.11.2026, 00:52"
    against "07.11.2026, 23:59" is the same date, and the judge called two drone
    hits on Kharkiv — two nights, 23 hours apart — one event, which would have
    folded the second night into the first night's post where nobody sees it.
    """
    minutes = (later - earlier) // 60
    if minutes < 30:
        return "приблизно тоді ж"
    if minutes < 90:
        return f"за {minutes} хв до нової"
    hours = round(minutes / 60)
    if hours < 48:
        return f"за {hours} год до нової"
    return f"за {round(hours / 24)} дн до нової"


def _as_published(cluster: Cluster, number: int, story: Cluster | None = None) -> dict[str, str]:
    """A post already out, as its readers saw it: the headline and the prose.

    One channel's text stood in for it before, and that channel's text can
    miss exactly what the post led with. A meeting with the British prime
    minister went out under "Zelensky: Russia is preparing a new massive
    attack", but the judge was shown the one channel that wrote only about the
    meeting — and the next story, the same statement from another channel, was
    judged a different event and published a second time.

    Stored, never generated: asking a published post for its headline would pay
    to rewrite it. A post left in its channel's own words has no prose of its
    own, and that text is what its readers saw.
    """
    material = _as_material(cluster, number)
    if story is not None:
        material["gap"] = gap(cluster.pub_time_percentile, story.pub_time_percentile)
    prose = cluster.stored_summary.as_text() or material["text"]
    material["text"] = "\n".join(part for part in (cluster.stored_headline, prose) if part)
    return material


def _unit(embedding: Sequence[float]) -> np.ndarray | None:  # type: ignore[type-arg]
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return None
    return vector / norm
