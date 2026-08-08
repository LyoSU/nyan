"""Unit tests for the distance matrix, which the snapshot test only checks end to end.

The penalties are pure numpy now, so they can be verified directly instead of
through the clustering result.
"""

import json
from typing import Any

import numpy as np
import pytest

from nyan.clusterer import Clusterer
from nyan.document import Document


DIM = 4


def make_doc(
    channel_id: str,
    pub_time: int,
    issue: str = "main",
    embedding: list[float] | None = None,
    images: int = 0,
) -> Document:
    return Document(
        url=f"https://t.me/{channel_id}/{pub_time}",
        channel_id=channel_id,
        post_id=pub_time,
        views=10,
        pub_time=pub_time,
        issue=issue,
        embedding=embedding or [1.0, 0.0, 0.0, 0.0],
        embedded_images=tuple(
            {"url": f"i{i}", "embedding": [0.0, 1.0, 0.0, 0.0]} for i in range(images)
        ),
    )


def make_clusterer(tmp_path: Any, **distances: Any) -> Clusterer:
    config = {
        "clustering": {
            "n_clusters": None,
            "metric": "precomputed",
            "linkage": "average",
            "distance_threshold": 0.1,
        },
        "distances": distances,
    }
    path = tmp_path / "clusterer.json"
    path.write_text(json.dumps(config))
    return Clusterer(str(path))


def test_similar_texts_from_one_channel_are_pushed_apart(tmp_path: Any) -> None:
    clusterer = make_clusterer(tmp_path, same_channels_penalty=5.0)
    # Similar but not identical: the penalty is a multiplier, so it has no
    # effect on a pair whose distance is already zero.
    similar = [1.0, 0.2, 0.0, 0.0]
    docs = [
        make_doc("a", 1000, embedding=similar),
        make_doc("a", 2000),
        make_doc("b", 1000),
    ]

    distances = clusterer.calc_distances(docs)

    # Same channel: the penalty applies. Different channels: it does not.
    assert distances[0, 1] > distances[0, 2]
    assert np.allclose(np.diag(distances), 0.0)


def test_the_penalty_never_pushes_a_distance_above_one(tmp_path: Any) -> None:
    clusterer = make_clusterer(tmp_path, same_channels_penalty=100.0)
    docs = [
        make_doc("a", 1000, embedding=[1.0, 0.0, 0.0, 0.0]),
        make_doc("a", 2000, embedding=[0.0, 1.0, 0.0, 0.0]),
    ]

    distances = clusterer.calc_distances(docs)

    assert distances.max() <= 1.0


def test_posts_hours_apart_are_penalized(tmp_path: Any) -> None:
    clusterer = make_clusterer(
        tmp_path, time_penalty_modifier=4.0, time_shift_hours=6
    )
    close = [make_doc("a", 0, embedding=[1.0, 0.5, 0.0, 0.0]), make_doc("b", 600)]
    apart = [
        make_doc("a", 0, embedding=[1.0, 0.5, 0.0, 0.0]),
        make_doc("b", 48 * 3600),
    ]

    assert clusterer.calc_distances(apart)[0, 1] > clusterer.calc_distances(close)[0, 1]


def test_issues_exempt_from_the_time_penalty(tmp_path: Any) -> None:
    """Tech and economy stories stay relevant for days, so they are exempt.

    The exemption needs both documents to belong to such an issue: a tech post
    and a news post are still compared with the penalty.
    """
    clusterer = make_clusterer(
        tmp_path,
        time_penalty_modifier=4.0,
        time_shift_hours=6,
        no_time_penalty_issues=["tech"],
    )
    embedding = [1.0, 0.5, 0.0, 0.0]
    both_tech = [
        make_doc("a", 0, issue="tech", embedding=embedding),
        make_doc("b", 48 * 3600, issue="tech"),
    ]
    one_tech = [
        make_doc("a", 0, issue="tech", embedding=embedding),
        make_doc("b", 48 * 3600, issue="main"),
    ]

    assert (
        clusterer.calc_distances(both_tech)[0, 1]
        < clusterer.calc_distances(one_tech)[0, 1]
    )


def test_image_bonus_is_off_when_configured_to_zero(tmp_path: Any) -> None:
    """A zero bonus used to raise NameError instead of disabling the rule."""
    clusterer = make_clusterer(tmp_path, image_bonus=0.0)
    docs = [make_doc("a", 0, images=1), make_doc("b", 100, images=1)]

    distances = clusterer.calc_distances(docs)

    assert distances.shape == (2, 2)


def test_matching_images_pull_documents_together(tmp_path: Any) -> None:
    with_bonus = make_clusterer(tmp_path, image_bonus=0.5)
    without_bonus = make_clusterer(tmp_path)
    docs = [
        make_doc("a", 0, embedding=[1.0, 0.5, 0.0, 0.0], images=1),
        make_doc("b", 100, images=1),
    ]

    assert (
        with_bonus.calc_distances(docs)[0, 1] < without_bonus.calc_distances(docs)[0, 1]
    )


def test_image_embeddings_of_different_widths_do_not_crash(tmp_path: Any) -> None:
    """A swap of the image encoder leaves both widths in one batch.

    Annotations are cached in Mongo, so on the first iteration after the swap
    the documents already stored carry the old model's vectors while freshly
    annotated ones carry the new model's. Comparing them is meaningless, but it
    must not take the whole sender down with it.
    """
    clusterer = make_clusterer(tmp_path, image_bonus=0.5)
    old = make_doc("a", 0, images=1)
    new = make_doc("b", 100)
    new.embedded_images = ({"url": "i0", "embedding": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]},)

    distances = clusterer.calc_distances([old, new])

    assert distances.shape == (2, 2)


def test_a_single_document_still_clusters(tmp_path: Any) -> None:
    clusterer = make_clusterer(tmp_path, same_channels_penalty=5.0)

    clusters = clusterer([make_doc("a", 0)])

    assert len(clusters) == 1
    assert len(clusters[0].docs) == 1


def test_clusterer_refuses_an_empty_input(tmp_path: Any) -> None:
    clusterer = make_clusterer(tmp_path)

    with pytest.raises(AssertionError):
        clusterer([])


def test_documents_already_published_together_stay_together(tmp_path: Any) -> None:
    """A published post must not be re-cut by a later run.

    Measured on production: re-clustering the same 24-hour window put 47% of
    already published clusters into more than one piece, four of them into four
    to eight pieces. A piece dominated by documents the post never carried
    shares too few URLs with it to be recognized, and is published a second
    time as its own story.
    """
    clusterer = make_clusterer(tmp_path)
    apart = [0.0, 1.0, 0.0, 0.0]
    docs = [
        make_doc("a", 1000),
        make_doc("b", 1000),
        # Far enough from the other two that the clusterer would split it off.
        make_doc("c", 1000, embedding=apart),
    ]
    published = {doc.url: 42 for doc in docs}

    clusters = clusterer(docs, published)

    assert len(clusters) == 1
    assert {doc.channel_id for doc in clusters[0].docs} == {"a", "b", "c"}


def test_an_unpublished_document_is_not_dragged_along(tmp_path: Any) -> None:
    """Only what a post already carries is held in place.

    Documents are moved into their post's piece rather than the pieces being
    merged, so a story that happens to share a piece with a published one is
    not swallowed by it.
    """
    clusterer = make_clusterer(tmp_path)
    apart = [0.0, 1.0, 0.0, 0.0]
    docs = [
        make_doc("a", 1000),
        make_doc("b", 1000, embedding=apart),
        make_doc("c", 1000, embedding=apart),
    ]
    published = {docs[0].url: 42, docs[1].url: 42}

    clusters = clusterer(docs, published)

    by_channel = {
        doc.channel_id: index
        for index, cluster in enumerate(clusters)
        for doc in cluster.docs
    }
    assert by_channel["a"] == by_channel["b"]
    assert by_channel["c"] != by_channel["a"]


def test_clustering_without_a_published_map_is_unchanged(tmp_path: Any) -> None:
    clusterer = make_clusterer(tmp_path)
    apart = [0.0, 1.0, 0.0, 0.0]
    docs = [make_doc("a", 1000), make_doc("b", 1000, embedding=apart)]

    assert len(clusterer(docs)) == 2
