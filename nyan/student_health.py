"""Is the student still labelling production, and how sure of itself is it?

Two things can go wrong with `nyan/student.py` without anything failing. The
model can be missing from the models volume after a move or a redeploy: the
annotator then logs one warning and goes back to the older category head, which
is right on 70% of posts where the student is right on 94%. And the news can
move away from the week the student was trained on, which shows first as more
posts it is unsure about.

Both can be read from what is already in Mongo, with no teacher and no cost:
which posts carry a `student` record, and how high their best category score
is. On the blind 500 the model was right on 99% of posts it gave 0.99 or more
and on 82% of those between 0.5 and 0.7, so the share of unsure posts moves
before accuracy does. It is a smoke detector, not a measurement:
`scripts/distill/drift.py` is what asks the teacher and gives a number.

Usage: python -m nyan.student_health [--mongo-config configs/mongo_config.json] [--hours 24]
Env: ALERT_BOT_TOKEN and ALERT_CHAT_ID, to be told in Telegram when a check fails.
Exits 1 when a check fails, so a scheduler shows the run as failed too.
"""

import argparse
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx

from nyan.mongo import get_annotated_documents_collection

#: Below this a category is a guess: the threshold the feed itself uses before
#: it lets the model drop a post as not news.
UNSURE_BELOW = 0.7
#: 8.5% of posts were unsure in the week the model was trained on, and 11-14% a
#: day seven weeks later, with the teacher still agreeing. A fifth is past
#: anything seen while the model was known to be fine.
MAX_UNSURE_SHARE = 0.20
#: A handful of posts without a record is a restart mid-batch; a tenth is a
#: host annotating without the model.
MAX_UNLABELLED_SHARE = 0.10
#: Fewer posts than this in the window says the crawler is the problem, and the
#: shares below would be noise.
MIN_POSTS = 200


@dataclass
class Health:
    posts: int = 0
    unlabelled: int = 0
    unsure: int = 0
    confidence_sum: float = 0.0
    models: Counter[str] = field(default_factory=Counter)

    @property
    def labelled(self) -> int:
        return self.posts - self.unlabelled

    @property
    def unlabelled_share(self) -> float:
        return self.unlabelled / self.posts if self.posts else 0.0

    @property
    def unsure_share(self) -> float:
        return self.unsure / self.labelled if self.labelled else 0.0

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.labelled if self.labelled else 0.0


def summarize(docs: Iterable[dict[str, Any]]) -> Health:
    """Counts over annotated documents that have text; the others get no category at all."""
    health = Health()
    for doc in docs:
        if not doc.get("patched_text"):
            continue
        health.posts += 1
        record = doc.get("student") or {}
        scores = doc.get("category_scores") or {}
        if not record or not scores:
            health.unlabelled += 1
            continue
        health.models[str(record.get("model"))] += 1
        best = max(float(score) for score in scores.values())
        health.confidence_sum += best
        if best < UNSURE_BELOW:
            health.unsure += 1
    return health


def problems(health: Health) -> list[str]:
    if health.posts < MIN_POSTS:
        return [f"only {health.posts} posts with text in the window: nothing to judge the model on"]
    found = []
    if health.unlabelled_share > MAX_UNLABELLED_SHARE:
        found.append(
            f"{health.unlabelled_share:.0%} of posts have no student record: "
            "the model is not loaded and the older category head is answering"
        )
    if health.unsure_share > MAX_UNSURE_SHARE:
        found.append(
            f"the student is unsure of {health.unsure_share:.0%} of posts "
            f"(limit {MAX_UNSURE_SHARE:.0%}): run scripts/distill/drift.py"
        )
    if len(health.models) > 1:
        found.append(f"posts labelled by different models: {dict(health.models)}")
    return found


def report(health: Health) -> str:
    return (
        f"{health.posts} posts, {health.unlabelled_share:.1%} without a student record, "
        f"{health.unsure_share:.1%} unsure (best score < {UNSURE_BELOW}), "
        f"mean best score {health.mean_confidence:.3f}, models {dict(health.models)}"
    )


def alert(text: str, *, transport: httpx.BaseTransport | None = None) -> bool:
    """Tell the owner in Telegram, when a bot and a chat are configured for it."""
    token, chat = os.getenv("ALERT_BOT_TOKEN"), os.getenv("ALERT_CHAT_ID")
    if not token or not chat:
        return False
    try:
        with httpx.Client(transport=transport, timeout=10) as client:
            response = client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": text},
            )
        return response.status_code == 200
    except httpx.HTTPError:
        logging.exception("Could not send the student health alert")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mongo-config", default="configs/mongo_config.json")
    parser.add_argument("--hours", type=float, default=24)
    args = parser.parse_args()

    collection = get_annotated_documents_collection(args.mongo_config)
    docs = collection.find(
        {"pub_time": {"$gte": int(time.time() - args.hours * 3600)}},
        {"_id": 0, "patched_text": 1, "category_scores": 1, "student.model": 1},
    )
    health = summarize(docs)
    print(report(health))
    found = problems(health)
    for problem in found:
        print(f"!! {problem}")
    if found:
        alert("nyan student: " + "; ".join(found) + "\n" + report(health))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
