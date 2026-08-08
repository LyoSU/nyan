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

# How close a published post has to be before it is worth asking about. Set for
# recall, not precision: the model makes the call, so a candidate only has to be
# plausible. Measured over a week of production, this leaves 1.7 candidates per
# post on average and none at all for 45% of them.
CANDIDATE_SIMILARITY = 0.86

# At most this many go into one prompt. The distribution has a long tail — the
# 95th percentile is seven — and a story that is genuinely a development of
# something is a development of the closest thing, not the seventh closest.
MAX_CANDIDATES = 3


@dataclass(frozen=True)
class Relation:
    verdict: str
    cluster: Cluster | None = None


def nearest_clusters(
    cluster: Cluster,
    published: Sequence[Cluster],
    limit: int = MAX_CANDIDATES,
    threshold: float = CANDIDATE_SIMILARITY,
) -> list[Cluster]:
    """The published posts this story might be about, closest first."""
    pivot = cluster.embedding
    if not pivot:
        return []
    vector = _unit(pivot)
    if vector is None:
        return []

    scored: list[tuple[float, int, Cluster]] = []
    for index, candidate in enumerate(published):
        if not candidate.messages:
            continue
        # A post published after this story cannot be what it develops.
        if candidate.pub_time_percentile > cluster.pub_time_percentile:
            continue
        embedding = candidate.embedding
        if not embedding or len(embedding) != len(pivot):
            continue
        other = _unit(embedding)
        if other is None:
            continue
        similarity = float(vector @ other)
        if similarity < threshold:
            continue
        # The index breaks ties without reaching for the cluster itself, which
        # has no ordering, and keeps the result stable across runs.
        scored.append((-similarity, index, candidate))

    scored.sort(key=lambda item: (item[0], item[1]))
    return [candidate for _, _, candidate in scored[:limit]]


def judge_relation(
    cluster: Cluster,
    candidates: Sequence[Cluster],
    reasoning_effort: str | None = None,
) -> Relation:
    """What this story is to the closest thing already published.

    Never raises. The daemon is a synchronous loop, and every failure here has
    the same safe answer: publish the post on its own, which is what happened
    before this existed.
    """
    if not candidates:
        return Relation(UNRELATED)

    messages = render_prompt(
        "relation",
        story=_as_material(cluster),
        candidates=[
            _as_material(candidate, number)
            for number, candidate in enumerate(candidates, start=1)
        ],
    )
    try:
        content = openai_completion(
            messages=messages,
            response_format={"type": "json_object"},
            reasoning_effort=reasoning_effort,
        )
        parsed = json.loads(content[content.find("{") : content.rfind("}") + 1])
    except Exception:
        logging.exception("Could not judge what '%s' is", cluster.cropped_title)
        return Relation(UNRELATED)

    verdict = parsed.get("verdict")
    match = parsed.get("match")
    if verdict not in VERDICTS or verdict == UNRELATED:
        return Relation(UNRELATED)
    if not isinstance(match, int) or not 1 <= match <= len(candidates):
        logging.warning("Verdict '%s' names no candidate: %s", verdict, match)
        return Relation(UNRELATED)

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


def _unit(embedding: Sequence[float]) -> np.ndarray | None:  # type: ignore[type-arg]
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return None
    return vector / norm
