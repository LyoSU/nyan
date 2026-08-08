"""Compare what a post used to show with what the new selection would show.

Run before changing the thresholds in `nyan.media`: the rules that decide a
post's attachments are cheap to reason about and expensive to be wrong about,
because the failure is silent — a published post simply carries the wrong
picture, or none, and nothing logs it.

Reports, over a sample of stored clusters: how many posts gain, lose or keep
their media, and what the lead picture becomes.

Usage:
    python -m scripts.preview_media --clusters 60
"""

import argparse
import logging
from collections import Counter
from io import BytesIO
from typing import Any

import numpy as np
import requests
from PIL import Image

from nyan.media import (
    MEDIA_PHOTO,
    MEDIA_VIDEO,
    MIN_LEAD_CHANNELS,
    MediaCandidate,
    group_media,
    select_media,
)
from nyan.channels import group_authority
from nyan.mongo import get_clusters_collection
from nyan.picture import perceptual_hashes, picture_quality, thumbnail_signature

MAX_CLUSTER_MEDIA = 4


def read_picture(url: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """What the annotator would have recorded, computed here for the preview."""
    if url in cache:
        return cache[url]
    read: dict[str, Any] = {"hashes": (), "quality": 0.0, "signature": ()}
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.load()
            read = {
                "hashes": perceptual_hashes(image),
                "quality": picture_quality(image).score,
                "signature": thumbnail_signature(image),
            }
    except Exception:
        pass
    cache[url] = read
    return read


def old_selection(docs: list[dict[str, Any]]) -> list[str]:
    """What `Cluster.media` chose before: one item per channel, cosine only.

    Reimplemented here rather than imported, because the point is to compare
    against behaviour that no longer exists in the tree.
    """
    doc_count = len(docs)
    if doc_count == 0:
        return []
    with_media = sum(bool(d.get("images") or d.get("videos")) for d in docs)
    if with_media / doc_count < 0.4 and with_media < 3:
        return []

    kept: list[str] = []
    kept_vectors: list[np.ndarray] = []
    seen_channels: set[str] = set()
    seen_urls: set[str] = set()
    for doc in sorted(docs, key=lambda d: d["pub_time"]):
        if doc["channel_id"] in seen_channels:
            continue
        options: list[tuple[str, Any]] = []
        if doc.get("videos"):
            video_url = doc["videos"][0]
            embedding = next(
                (
                    v.get("embedding")
                    for v in doc.get("embedded_videos") or []
                    if v.get("url") == video_url
                ),
                None,
            )
            options.append((video_url, embedding))
        for image in doc.get("embedded_images") or []:
            if image.get("url"):
                options.append((image["url"], image.get("embedding")))
                break
        for url, embedding in options:
            if url in seen_urls:
                continue
            vector = None
            if embedding:
                vector = np.asarray(embedding, dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                vector = vector / norm if norm else None
            if vector is not None and kept_vectors:
                comparable = [v for v in kept_vectors if v.shape == vector.shape]
                if comparable and float(np.max(np.stack(comparable) @ vector)) >= 0.92:
                    continue
            seen_urls.add(url)
            kept.append(url)
            if vector is not None:
                kept_vectors.append(vector)
            seen_channels.add(doc["channel_id"])
            break
        if len(kept) >= MAX_CLUSTER_MEDIA:
            break
    return kept


def build_candidates(
    docs: list[dict[str, Any]], cache: dict[str, dict[str, Any]]
) -> list[MediaCandidate]:
    candidates: list[MediaCandidate] = []
    for doc in docs:
        authority = group_authority(doc.get("groups", {}).get(doc.get("issue") or "main", ""))
        embedded_videos = {
            str(v.get("url")): v for v in doc.get("embedded_videos") or [] if v.get("url")
        }
        for url in doc.get("videos") or []:
            video = embedded_videos.get(url, {})
            candidates.append(
                MediaCandidate(
                    type=MEDIA_VIDEO,
                    url=url,
                    channel_id=doc["channel_id"],
                    pub_time=doc["pub_time"],
                    embedding=tuple(video.get("embedding") or ()) or None,
                    authority=authority,
                    **(
                        read_picture(str(video["thumb"]), cache)
                        if video.get("thumb")
                        else {}
                    ),
                )
            )
        for image in doc.get("embedded_images") or []:
            if not image.get("url"):
                continue
            candidates.append(
                MediaCandidate(
                    type=MEDIA_PHOTO,
                    url=str(image["url"]),
                    channel_id=doc["channel_id"],
                    pub_time=doc["pub_time"],
                    embedding=tuple(image.get("embedding") or ()) or None,
                    authority=authority,
                    **read_picture(str(image["url"]), cache),
                )
            )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mongo-config", default="configs/mongo_config.json")
    parser.add_argument("--clusters", type=int, default=60)
    parser.add_argument("--min-docs", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    collection = get_clusters_collection(args.mongo_config)

    cache: dict[str, dict[str, Any]] = {}
    outcomes: Counter[str] = Counter()
    lead_changes = 0
    examples: list[dict[str, Any]] = []
    sizes_before: list[int] = []
    sizes_after: list[int] = []

    query = {f"docs.{args.min_docs - 1}": {"$exists": True}}
    for cluster in collection.find(query, sort=[("create_time", -1)]).limit(args.clusters):
        docs = cluster["docs"]
        if not any(d.get("images") or d.get("videos") for d in docs):
            continue
        before = old_selection(docs)
        candidates = build_candidates(docs, cache)
        after = [item.url for item in select_media(candidates, MAX_CLUSTER_MEDIA)]

        sizes_before.append(len(before))
        sizes_after.append(len(after))
        if before and not after:
            outcomes["lost all media"] += 1
        elif not before and after:
            outcomes["gained media"] += 1
        elif len(after) < len(before):
            outcomes["fewer items"] += 1
        elif len(after) > len(before):
            outcomes["more items"] += 1
        else:
            outcomes["same count"] += 1
        if before and after and before[0] != after[0]:
            lead_changes += 1

        groups = group_media(candidates)
        examples.append(
            {
                "headline": (cluster.get("headline") or "")[:70],
                "channels": len({d["channel_id"] for d in docs}),
                "before": len(before),
                "after": len(after),
                "groups": sorted((g.consensus for g in groups), reverse=True),
            }
        )

    total = sum(outcomes.values())
    logging.info("\nClusters with media: %d", total)
    for outcome, count in outcomes.most_common():
        logging.info("  %-16s %3d  (%.0f%%)", outcome, count, 100 * count / max(total, 1))
    logging.info("\nMedia per post: %.2f before -> %.2f after",
                 np.mean(sizes_before or [0]), np.mean(sizes_after or [0]))
    logging.info("Lead picture changed in %d of %d posts that kept media", lead_changes, total)
    logging.info("Posts with no group reaching %d channels: %d",
                 MIN_LEAD_CHANNELS,
                 sum(1 for e in examples if not e["groups"] or e["groups"][0] < MIN_LEAD_CHANNELS))

    logging.info("\nPer-cluster detail (group sizes = channels behind each picture):")
    for example in examples[:40]:
        logging.info(
            "  ch=%2d  %d->%d  groups=%s  %s",
            example["channels"],
            example["before"],
            example["after"],
            example["groups"][:6],
            example["headline"],
        )


if __name__ == "__main__":
    main()
