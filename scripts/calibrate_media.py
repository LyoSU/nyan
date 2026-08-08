"""Measure how the duplicate thresholds behave on the pictures actually posted.

Both numbers that decide what a post shows — the SigLIP similarity above which
two pictures are one, and the Hamming distance below which two hashes are one —
are judgement calls, and a judgement call that nobody re-checks against real
data drifts as the feed changes. This prints the evidence for both.

The interesting cell is `hash says same, embedding says different`: those are
the reposts a channel stamped, which the embedding threshold alone lets through
into the slideshow twice.

Usage:
    python -m scripts.calibrate_media --mongo-config configs/mongo_config.json
"""

import argparse
import itertools
import json
import logging
from collections import Counter
from collections.abc import Iterator
from typing import Any

import numpy as np
import requests
from PIL import Image
from io import BytesIO

from nyan.clusters import DUPLICATE_IMAGE_SIMILARITY
from nyan.mongo import get_clusters_collection
from nyan.picture import (
    DEFAULT_MAX_DISTANCE,
    hamming_distance,
    is_same_picture,
    perceptual_hashes,
)


def fetch(url: str) -> Image.Image | None:
    try:
        response = requests.get(url, timeout=20)
    except Exception:
        return None
    if response.status_code != 200:
        return None
    try:
        image = Image.open(BytesIO(response.content))
        image.load()
    except Exception:
        return None
    return image


def clustered_documents(
    collection: Any, min_docs: int
) -> Iterator[list[dict[str, Any]]]:
    """Documents grouped by the cluster they were published in, newest first.

    Grouped by cluster because a duplicate only matters inside one: two
    channels showing the same stock photo of a courtroom in two unrelated
    stories is not the problem being measured. The clusters collection already
    stores its documents nested, so the grouping is free — and the pictures are
    recent, which matters because a Telegram CDN url stops resolving once it is
    a few days old.
    """
    for cluster in collection.find(
        {f"docs.{min_docs - 1}": {"$exists": True}}, sort=[("create_time", -1)]
    ):
        docs = [doc for doc in cluster["docs"] if doc.get("embedded_images")]
        if len(docs) >= min_docs:
            yield docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mongo-config", default="configs/mongo_config.json")
    parser.add_argument("--clusters", type=int, default=30)
    parser.add_argument("--min-docs", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    collection = get_clusters_collection(args.mongo_config)

    verdicts: Counter[str] = Counter()
    disagreements: list[dict[str, Any]] = []
    fetched_total = 0

    groups = itertools.islice(
        clustered_documents(collection, args.min_docs), args.clusters
    )
    for index, docs in enumerate(groups):
        pictures: list[dict[str, Any]] = []
        for doc in docs:
            for embedded in doc.get("embedded_images") or []:
                image = fetch(embedded["url"])
                if image is None:
                    continue
                pictures.append(
                    {
                        "channel": doc["channel_id"],
                        "url": embedded["url"],
                        "embedding": np.asarray(embedded["embedding"], dtype=np.float32),
                        "hashes": perceptual_hashes(image),
                    }
                )
        fetched_total += len(pictures)
        logging.info("cluster %d: %d pictures fetched", index, len(pictures))

        for one, other in itertools.combinations(pictures, 2):
            if one["channel"] == other["channel"]:
                continue
            first = one["embedding"] / np.linalg.norm(one["embedding"])
            second = other["embedding"] / np.linalg.norm(other["embedding"])
            similarity = float(first @ second)
            distance = min(
                hamming_distance(a, b)
                for a, b in zip(one["hashes"], other["hashes"], strict=True)
            )
            by_embedding = similarity >= DUPLICATE_IMAGE_SIMILARITY
            by_hash = is_same_picture(one["hashes"], other["hashes"])
            verdicts[f"embedding={by_embedding} hash={by_hash}"] += 1
            if by_hash != by_embedding:
                disagreements.append(
                    {
                        "similarity": round(similarity, 4),
                        "distance": distance,
                        "urls": [one["url"], other["url"]],
                        "channels": [one["channel"], other["channel"]],
                    }
                )

    logging.info("\nFetched %d pictures", fetched_total)
    logging.info("Thresholds: cosine >= %s, hamming <= %s", DUPLICATE_IMAGE_SIMILARITY, DEFAULT_MAX_DISTANCE)
    for verdict, count in verdicts.most_common():
        logging.info("  %-34s %d", verdict, count)

    caught = [d for d in disagreements if d["distance"] <= DEFAULT_MAX_DISTANCE]
    logging.info("\nStamped reposts the embedding alone misses: %d", len(caught))
    for case in sorted(caught, key=lambda d: d["similarity"])[:15]:
        logging.info(
            "  cos=%.3f hamming=%2d  %s vs %s",
            case["similarity"],
            case["distance"],
            case["channels"][0],
            case["channels"][1],
        )
        logging.info("      %s", case["urls"][0])
        logging.info("      %s", case["urls"][1])

    with open("media_calibration.json", "w") as w:
        json.dump(disagreements, w, ensure_ascii=False, indent=2)
    logging.info("\nAll disagreements written to media_calibration.json")


if __name__ == "__main__":
    main()
