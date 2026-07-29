"""Which channels are the same voice, measured from what they published.

## The question this answers

A story page says «56 джерел». Read as fifty-six independent confirmations, that
is social proof standing in for verification, and it is often wrong: seven of the
fifty-six are clones of one newsroom posting the same text within a minute of
each other, and thirty more reposted a single anonymous channel. The site
already warns about this in prose. Prose is not a number, and the number is what
gets read.

Three of the four ways two channels can turn out to be one voice are visible in a
single story — a declared `master`, a `forward_from`, a verbatim copy — and the
site computes those where it renders. The fourth is not visible in one story at
all: two channels that always publish the same events within seconds of each
other are behaving as one source whether or not any single post proves it. That
is what this job measures, over months rather than one cluster.

## What it computes, and why not Agency2Vec

The whitepaper's Agency2Vec trains Word2Vec over stories-as-sentences and
channels-as-words. `scripts/agency2vec.py` is that prototype. Two problems ruled
it out here. It is stochastic, so two runs over identical data disagree about how
close two channels are, and a number that moves on its own cannot be a stable
part of a published figure. Worse, its output is a coordinate in a latent space,
and «cosine 0.87» is not something a channel's editor can dispute.

So the measure is deliberately elementary:

    overlap(a, b) = clusters carrying both / clusters carrying the rarer of the two
    lag(a, b)     = median seconds between their first posts in a shared cluster

Both are checkable claims about the archive. «When times_ukraina_news publishes,
times_ukraina publishes the same event 96% of the time, typically 12 seconds
apart» is a sentence a reader can argue with, which is the property that matters
for a number the site is going to subtract sources on the strength of.

Coordinates for the media map are a separate concern and *are* a projection —
MDS over cosine distance between co-occurrence vectors. They position dots on a
picture, and nothing is subtracted on their authority.

## What none of this establishes

Why two channels behave as one. A pair publishing identical text within seconds
could be one operator running both, one channel lifting from the other, or both
carrying the same placement. Those are different things and this measure cannot
tell them apart, so nothing here — and nothing the site renders from it — may say
which. What it establishes is narrower and enough: neither channel checked the
other, so counting them as two confirmations overstates the story.

The distinction is not pedantry, it is the difference between a checkable claim
and an accusation. `same_voice` is therefore a statement about function — these do
not constitute two independent voices — and the wording downstream stays on the
observable act: «той самий текст, на 17 с пізніше», never «клон» or «замовне».
Same rule as `ChannelBadge` in the site's types: a claim about a channel carries
the name of whoever made it, and we are not making this one.

## Running it

    python -m scripts.build_channel_graph --mongo-config-path configs/mongo_config.json

Cheap enough for a nightly cron: the co-occurrence pass is linear in documents,
and the pairwise step is over a few hundred channels.

## What it actually finds today, which is less than it sounds

Measured over the whole archive — 3205 multi-channel stories, 218 channels — this
returns **three** pairs, not thirty:

    uanova           ~ uaonlii          100% of 11 stories, 17s apart
    insiderukr       ~ ukrinformator    100% of  9 stories, 66s apart
    kyivcityofficial ~ vitaliy_klitschko 98% of 111 stories, 83s apart

All three were checked by hand and all three are real: the last is the city
council and the mayor, and one of the shared stories in the second pair is
word-for-word identical in both channels. None of them is declared in
channels.json, so each is a fact the registry did not have.

The declared `times_*` clone network, by contrast, is *not* recovered — those
channels co-occur two or three times in seven months, because they barely reach
the clusters at all. Which sets the honest expectation for this measure: at
thirty stories a day it is a small source of new facts, not the main one. The
three cheap signals that live in a single story — a declared master, a
`forward_from`, a verbatim copy — do most of the work, and this adds the cases
none of them can see. It gets stronger on its own as the archive grows, with no
change here.
"""

import argparse
import logging
from collections import defaultdict
from statistics import median
from typing import Any

import numpy as np
from pymongo import UpdateOne
from sklearn.manifold import MDS

from nyan.mongo import get_channel_graph_collection, get_clusters_collection
from nyan.util import get_current_ts


# How far back to read. Long enough that a pair's overlap is not an accident of
# one busy week, short enough that a channel which changed hands a year ago is
# not still judged on what it used to republish.
DEFAULT_WINDOW_DAYS = 60

# Below this many shared stories a pair says nothing. Two channels that appeared
# together three times, always within a minute, look identical to this measure
# and are probably two channels that both posted three big events.
#
# Eight, from the archive rather than from taste. At 12 the measure loses two of
# the three pairs it finds — uanova/uaonlii and insiderukr/ukrinformator, both
# checked by hand and both real — and at 5 it admits nothing further. The archive
# is around thirty stories a day, so a pair sharing eight of them while the rarer
# of the two publishes almost nothing alone is already a strong statement.
MIN_TOGETHER = 8

# What counts as one voice. The overlap bar is high on purpose: at 0.9, the rarer
# of the two channels is almost never publishing anything the other one misses,
# which is the behaviour of a mirror rather than of a newsroom that happens to
# share a beat.
SAME_VOICE_OVERLAP = 0.9

# And it has to be simultaneous. `MIN_MEANINGFUL_GAP` on the site says a gap
# under two minutes is not a finding — the same threshold, for the same reason:
# below it, "who was first" is crawl jitter. Two channels reliably inside that
# window are not reacting to each other, they are being posted by one hand.
SAME_VOICE_LAG = 120

# Neighbours kept per channel for display. The list answers «with whom does this
# channel travel», and the tail of it is noise.
MAX_NEIGHBOURS = 8

# Categories that are not news. Same exclusion the site applies everywhere, so
# the graph is built over the stories a reader can actually see.
INVISIBLE = ("not_news", "unknown")


def read_appearances(
    mongo_config_path: str, window_days: int
) -> list[dict[str, int]]:
    """Every story as {channel_id: when that channel first carried it}.

    One entry per story rather than per post: a channel that posts four follow-ups
    to the same event is still one voice on it, and counting the posts would let a
    chatty channel look like broad coverage.

    Stories carried by a single channel are kept, even though they can never
    produce a pair. They are the denominator. Dropping them — which this did at
    first — measures overlap against only the stories a channel shared, so a
    channel publishing a hundred things, ten of them alongside `a` and ninety on
    its own, scored 10/10 = «always travels with a». The honest figure is 10/100.
    """
    collection = get_clusters_collection(mongo_config_path)
    cutoff = get_current_ts() - window_days * 24 * 3600

    stories = []
    cursor = collection.find(
        {
            "create_time": {"$gte": cutoff},
            "annotation_doc.category": {"$nin": INVISIBLE},
        },
        # Project hard. `annotation_doc` carries a 768-float embedding and one per
        # attached image, and this reads tens of thousands of clusters.
        {"_id": 0, "docs.channel_id": 1, "docs.pub_time": 1},
    )
    for cluster in cursor:
        first_seen: dict[str, int] = {}
        for doc in cluster.get("docs", []):
            channel_id = str(doc.get("channel_id", "")).lower()
            pub_time = doc.get("pub_time")
            if not channel_id or not pub_time:
                continue
            if channel_id not in first_seen or pub_time < first_seen[channel_id]:
                first_seen[channel_id] = int(pub_time)
        if first_seen:
            stories.append(first_seen)
    return stories


def count_cooccurrence(
    stories: list[dict[str, int]],
) -> tuple[dict[str, int], dict[tuple[str, str], int], dict[tuple[str, str], list[int]]]:
    """How often each channel published, each pair coincided, and how far apart.

    Pair keys are sorted, so `(a, b)` and `(b, a)` are one entry. Lags are
    absolute: the direction of a two-second gap is noise, and the direction of a
    real one belongs to the per-story timeline, not to a months-long median.
    """
    totals: dict[str, int] = defaultdict(int)
    together: dict[tuple[str, str], int] = defaultdict(int)
    lags: dict[tuple[str, str], list[int]] = defaultdict(list)

    for first_seen in stories:
        channels = sorted(first_seen)
        for channel_id in channels:
            totals[channel_id] += 1
        for i, left in enumerate(channels):
            for right in channels[i + 1 :]:
                together[(left, right)] += 1
                lags[(left, right)].append(abs(first_seen[left] - first_seen[right]))
    return dict(totals), dict(together), dict(lags)


def build_pairs(
    totals: dict[str, int],
    together: dict[tuple[str, str], int],
    lags: dict[tuple[str, str], list[int]],
) -> list[dict[str, Any]]:
    """Every pair with enough shared stories to mean something, scored."""
    pairs = []
    for (left, right), count in together.items():
        if count < MIN_TOGETHER:
            continue
        rarer = min(totals[left], totals[right])
        if rarer == 0:
            continue
        pair_lags = lags[(left, right)]
        pairs.append(
            {
                "a": left,
                "b": right,
                "together": count,
                "overlap": count / rarer,
                "median_lag": int(median(pair_lags)),
            }
        )
    return pairs


def project(totals: dict[str, int], together: dict[tuple[str, str], int]) -> dict[str, tuple[float, float]]:
    """Channels as points, close where they cover the same events.

    Cosine over co-occurrence vectors, then MDS down to two dimensions. Unlike
    the overlap figure this *is* a projection with no checkable meaning per pair,
    which is why nothing is subtracted on its authority — it places dots on a
    map and that is all.

    `random_state` and `n_init` are both fixed, and the second matters as much as
    the first: MDS runs the layout several times from different starts and keeps
    the best, so a change to how many times it tries moves every point even with
    the seed held. scikit-learn is about to change that default from 4 to 1,
    which would have silently reshuffled the whole map on an upgrade — and a
    reader who learned where a channel sits should find it there tomorrow.

    `init` is left alone deliberately: its default changes in scikit-learn 1.10,
    and requirements.txt caps the library below that for unrelated reasons. Naming
    it here would break the 1.4 floor, which does not accept it.
    """
    channels = sorted(totals)
    index = {channel_id: i for i, channel_id in enumerate(channels)}
    size = len(channels)
    if size < 3:
        return {}

    matrix = np.zeros((size, size), dtype=np.float64)
    for (left, right), count in together.items():
        matrix[index[left], index[right]] = count
        matrix[index[right], index[left]] = count
    for channel_id, total in totals.items():
        matrix[index[channel_id], index[channel_id]] = total

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A channel with no co-occurrences at all would divide by zero and poison
    # every distance in the matrix, not only its own row.
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    distance = np.clip(1.0 - normalized @ normalized.T, 0.0, None)
    np.fill_diagonal(distance, 0.0)

    model = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=20260729,
        n_init=4,
        normalized_stress="auto",
    )
    coordinates = model.fit_transform(distance)
    return {
        channel_id: (float(coordinates[i, 0]), float(coordinates[i, 1]))
        for channel_id, i in index.items()
    }


def build_records(
    totals: dict[str, int],
    pairs: list[dict[str, Any]],
    coordinates: dict[str, tuple[float, float]],
    window_days: int,
) -> list[dict[str, Any]]:
    """One document per channel: where it sits, and who it travels with."""
    neighbours: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        for channel_id, other in ((pair["a"], pair["b"]), (pair["b"], pair["a"])):
            neighbours[channel_id].append(
                {
                    "channel_id": other,
                    "together": pair["together"],
                    "overlap": round(pair["overlap"], 3),
                    "median_lag": pair["median_lag"],
                    # Precomputed rather than re-derived by every reader, so the
                    # thresholds live in exactly one place.
                    "same_voice": (
                        pair["overlap"] >= SAME_VOICE_OVERLAP
                        and pair["median_lag"] < SAME_VOICE_LAG
                    ),
                }
            )

    generated_at = get_current_ts()
    records = []
    for channel_id, total in totals.items():
        ranked = sorted(
            neighbours.get(channel_id, []),
            key=lambda row: (row["overlap"], row["together"]),
            reverse=True,
        )
        x, y = coordinates.get(channel_id, (0.0, 0.0))
        records.append(
            {
                "channel_id": channel_id,
                "generated_at": generated_at,
                "window_days": window_days,
                "stories": total,
                "x": x,
                "y": y,
                "neighbours": ranked[:MAX_NEIGHBOURS],
                # The flat list the story page needs. Not truncated with the
                # neighbours: a clone network of ten is exactly the case where
                # dropping the ninth would overstate independence.
                "same_voice": sorted(
                    row["channel_id"] for row in ranked if row["same_voice"]
                ),
            }
        )
    return records


def write_records(mongo_config_path: str, records: list[dict[str, Any]]) -> None:
    collection = get_channel_graph_collection(mongo_config_path)
    collection.create_index([("channel_id", 1)], unique=True, name="channel")
    collection.bulk_write(
        [
            UpdateOne({"channel_id": r["channel_id"]}, {"$set": r}, upsert=True)
            for r in records
        ],
        ordered=False,
    )


def main(mongo_config_path: str, window_days: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    stories = read_appearances(mongo_config_path, window_days)
    logging.info("Read %d stories", len(stories))
    if not stories:
        logging.warning("Nothing to build a graph from; leaving it as it was")
        return

    totals, together, lags = count_cooccurrence(stories)
    logging.info("%d channels, %d co-occurring pairs", len(totals), len(together))

    pairs = build_pairs(totals, together, lags)
    same_voice = [p for p in pairs if p["overlap"] >= SAME_VOICE_OVERLAP and p["median_lag"] < SAME_VOICE_LAG]
    logging.info("%d pairs above the noise floor, %d of them one voice", len(pairs), len(same_voice))
    for pair in sorted(same_voice, key=lambda p: -p["overlap"])[:20]:
        logging.info(
            "  %s ~ %s: %.0f%% of %d stories, %ds apart",
            pair["a"],
            pair["b"],
            pair["overlap"] * 100,
            pair["together"],
            pair["median_lag"],
        )

    coordinates = project(totals, together)
    records = build_records(totals, pairs, coordinates, window_days)
    write_records(mongo_config_path, records)
    logging.info("Wrote %d channel records", len(records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mongo-config-path", type=str, default="configs/mongo_config.json")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    args = parser.parse_args()
    main(**vars(args))
