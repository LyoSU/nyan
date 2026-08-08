"""Tests for turning a cluster's documents into candidates for its post.

The choosing itself lives in `nyan.media` and is tested in
`test_media_selection.py`, against plain candidates. What is left here is the
adapter: that every attachment of every document is offered, that what the
annotator recorded about each picture survives the trip, and that a cluster
read back from Mongo offers exactly what it offered before it was stored.

That last one is not a formality. The daemon re-reads posted clusters on every
iteration, and a stored document is written short — so anything the selection
needs and `asdict(is_short=True)` drops makes a published post's media change
under the reader.
"""

import math

from nyan.clusters import MAX_CLUSTER_MEDIA, Cluster
from nyan.document import Document


def unit(angle: float) -> list[float]:
    """A point on the unit circle, so cosine similarity is a known quantity."""
    return [math.cos(angle), math.sin(angle)]


# Two vectors this far apart have cosine similarity ~0.995: the same photo.
NEAR = 0.1
# And these ~0.54: different photos.
FAR = 1.0

# Hashes that agree with nothing else, so grouping is decided by the embedding
# alone unless a test says otherwise.
def hashes(seed: str) -> list[str]:
    return [seed * 16, seed * 16]


def make_doc(
    channel_id: str,
    images: list[dict[str, object]],
    pub_time: int = 100,
    videos: tuple[str, ...] = (),
    embedded_videos: list[dict[str, object]] | None = None,
    group: str = "blue",
) -> Document:
    return Document(
        url=f"https://t.me/{channel_id}/1",
        channel_id=channel_id,
        post_id=1,
        views=1,
        pub_time=pub_time,
        images=tuple(str(image["url"]) for image in images),
        embedded_images=images,
        videos=videos,
        embedded_videos=embedded_videos or [],
        # An accountable tier by default: an anonymous channel's uncorroborated
        # picture is not shown at all, which is the selection's business and
        # not this file's.
        groups={"main": group},
        issue="main",
    )


def make_cluster(*docs: Document) -> Cluster:
    cluster = Cluster()
    for doc in docs:
        cluster.add(doc)
    cluster.saved_annotation_doc = docs[0]
    return cluster


def test_media_is_gathered_from_every_channel_not_just_the_chosen_one() -> None:
    """The channel that writes best is often not the one that was there."""
    first = make_doc("writer", [], pub_time=100)
    second = make_doc("witness", [{"url": "b.jpg", "embedding": unit(0.0)}], pub_time=200)
    cluster = make_cluster(first, second)

    assert cluster.images == ("b.jpg",)


def test_every_attachment_of_a_document_is_offered() -> None:
    """Not just the first: which of them earns a slot is decided later.

    A channel used to offer one candidate — its video, or its first photo —
    which meant a picture three other channels also posted was never even
    considered if that channel happened to list it second.
    """
    shared = {"url": "wire.jpg", "embedding": unit(FAR)}
    cluster = make_cluster(
        make_doc("a", [{"url": "own.jpg", "embedding": unit(0.0)}, shared]),
        make_doc("b", [{"url": "wire-copy.jpg", "embedding": unit(FAR + 0.01)}]),
    )

    assert cluster.images[0] == "wire.jpg"


def test_photos_and_videos_are_offered_together() -> None:
    """One slideshow carries both, so they compete for the same slots."""
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}]),
        make_doc("b", [], videos=("b.mp4",)),
    )

    assert {(item.type, item.url) for item in cluster.media} == {
        ("photo", "a.jpg"),
        ("video", "b.mp4"),
    }


def test_a_videos_still_is_what_compares_it() -> None:
    """The clip is never downloaded and its url differs per channel.

    `embedded_videos` is keyed by the video's own url rather than the still's,
    because that is what the cluster has in hand.
    """
    cluster = make_cluster(
        make_doc(
            "a",
            [],
            videos=("a.mp4",),
            embedded_videos=[{"url": "a.mp4", "embedding": unit(0.0)}],
        ),
        make_doc(
            "b",
            [],
            videos=("b.mp4",),
            embedded_videos=[{"url": "b.mp4", "embedding": unit(NEAR)}],
        ),
    )

    assert cluster.videos == ("a.mp4",)


def test_the_number_of_attachments_is_capped() -> None:
    cluster = make_cluster(
        *[
            make_doc(f"c{i}", [{"url": f"{i}.jpg", "embedding": unit(i * FAR)}])
            for i in range(10)
        ]
    )

    assert len(cluster.media) == MAX_CLUSTER_MEDIA


def test_a_picture_only_an_anonymous_channel_has_is_not_shown() -> None:
    """Nothing confirms it and nobody is answerable for it."""
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}], group="grey"),
        make_doc("b", [], group="grey"),
    )

    assert cluster.media == ()


def test_what_the_annotator_recorded_reaches_the_selection() -> None:
    """The hashes, quality and signature decide between copies of one picture.

    They are read at annotation time, from the picture itself, and they are the
    only thing that can tell a clean copy from the same photograph inside
    somebody's generated card.
    """
    clean = {
        "url": "clean.jpg",
        "embedding": unit(0.0),
        "hashes": hashes("a"),
        "quality": 0.9,
        "signature": [100] * 256,
    }
    carded = {
        "url": "carded.jpg",
        "embedding": unit(NEAR),
        "hashes": hashes("a"),
        "quality": 0.2,
        "signature": [100] * 256,
    }
    cluster = make_cluster(make_doc("a", [carded]), make_doc("b", [clean]))

    assert cluster.images == ("clean.jpg",)


def test_media_survives_a_trip_through_storage() -> None:
    """The daemon re-reads posted clusters from storage on every iteration.

    A stored document is written short, and anything the selection needs that
    `asdict(is_short=True)` drops makes a published post's media change on its
    own — which is what happened when `embedded_images` was dropped and
    `images` was kept.
    """
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0), "quality": 0.5}]),
        make_doc("b", [{"url": "b.jpg", "embedding": unit(FAR), "quality": 0.5}]),
    )

    restored = Cluster.deserialize(cluster.serialize())

    assert restored.media == cluster.media


def test_relevance_to_the_story_survives_storage() -> None:
    """It is computed from embeddings a stored cluster no longer carries.

    So it is recorded on the document while they are still in hand. Without
    that, the same cluster judges the same attachments differently before and
    after being stored.
    """
    near = make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}])
    near.embedding = unit(0.0)
    far = make_doc("b", [{"url": "b.jpg", "embedding": unit(FAR)}])
    far.embedding = unit(FAR)
    cluster = make_cluster(near, far)
    # Selecting is what records it; nothing else in the cluster asks for it.
    assert cluster.media

    restored = Cluster.deserialize(cluster.serialize())

    assert [doc.story_relevance for doc in restored.docs] == [
        doc.story_relevance for doc in cluster.docs
    ]


def test_a_photo_without_an_embedding_is_still_used() -> None:
    """Older documents have none, and a picture beats no picture."""
    cluster = make_cluster(
        make_doc("a", [{"url": "same.jpg"}]),
        make_doc("b", [{"url": "same.jpg"}]),
    )

    assert cluster.images == ("same.jpg",)


def test_a_zero_embedding_does_not_break_comparison() -> None:
    """It carries no direction, so it cannot be compared — keep the picture."""
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": [0.0, 0.0]}]),
        make_doc("b", [{"url": "b.jpg", "embedding": [0.0, 0.0]}]),
    )

    assert len(cluster.media) == 2


def test_embeddings_of_different_widths_do_not_break_comparison() -> None:
    """An encoder swap leaves both widths in the annotation cache at once."""
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}]),
        make_doc("b", [{"url": "b.jpg", "embedding": [1.0, 0.0, 0.0]}]),
    )

    assert len(cluster.media) == 2
