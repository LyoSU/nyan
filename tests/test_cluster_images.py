"""Tests for choosing a cluster's photos.

Photos are gathered from every channel that posted one, which is the only way a
story covered by a channel that writes well but was not there gets a picture at
all. The cost is that the same wire photo arrives once per channel, under a
different Telegram CDN URL each time, so URL comparison cannot see the
duplication and CLIP embeddings have to.
"""

import math

from nyan.clusters import MAX_CLUSTER_IMAGES, Cluster
from nyan.document import Document


def unit(angle: float) -> list[float]:
    """A point on the unit circle, so cosine similarity is a known quantity."""
    return [math.cos(angle), math.sin(angle)]


# Two vectors this far apart have cosine similarity ~0.995: the same photo.
NEAR = 0.1
# And these ~0.54: different photos.
FAR = 1.0


def make_doc(
    channel_id: str,
    images: list[dict[str, object]],
    pub_time: int = 100,
) -> Document:
    return Document(
        url=f"https://t.me/{channel_id}/1",
        channel_id=channel_id,
        post_id=1,
        views=1,
        pub_time=pub_time,
        images=tuple(str(image["url"]) for image in images),
        embedded_images=images,
    )


def make_cluster(*docs: Document) -> Cluster:
    cluster = Cluster()
    for doc in docs:
        cluster.add(doc)
    cluster.saved_annotation_doc = docs[0]
    return cluster


def test_photos_come_from_every_channel_that_has_one() -> None:
    """The channel that writes best is often not the one that was there."""
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}]),
        make_doc("b", [{"url": "b.jpg", "embedding": unit(FAR)}]),
        make_doc("c", [{"url": "c.jpg", "embedding": unit(2 * FAR)}]),
    )

    assert cluster.images == ("a.jpg", "b.jpg", "c.jpg")


def test_the_same_photo_from_several_channels_is_shown_once() -> None:
    """Different URLs, one picture: only the embedding can tell."""
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}]),
        make_doc("b", [{"url": "b.jpg", "embedding": unit(NEAR)}]),
        make_doc("c", [{"url": "c.jpg", "embedding": unit(2 * NEAR)}]),
        make_doc("d", [{"url": "d.jpg", "embedding": unit(FAR)}]),
    )

    assert cluster.images == ("a.jpg", "d.jpg")


def test_one_photo_per_channel() -> None:
    """A channel posting six shots of one scene must not fill the slideshow."""
    cluster = make_cluster(
        make_doc(
            "a",
            [
                {"url": "a1.jpg", "embedding": unit(0.0)},
                {"url": "a2.jpg", "embedding": unit(FAR)},
                {"url": "a3.jpg", "embedding": unit(2 * FAR)},
            ],
        ),
        make_doc("b", [{"url": "b.jpg", "embedding": unit(3 * FAR)}]),
        make_doc("c", [{"url": "c.jpg", "embedding": unit(4 * FAR)}]),
    )

    assert cluster.images == ("a1.jpg", "b.jpg", "c.jpg")


def test_the_number_of_photos_is_capped() -> None:
    cluster = make_cluster(
        *[
            make_doc(f"c{i}", [{"url": f"{i}.jpg", "embedding": unit(i * FAR)}])
            for i in range(10)
        ]
    )

    assert len(cluster.images) == MAX_CLUSTER_IMAGES


def test_the_chosen_channels_photo_comes_first() -> None:
    """The post's text and its picture should come from the same report."""
    first = make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}], pub_time=100)
    second = make_doc("b", [{"url": "b.jpg", "embedding": unit(FAR)}], pub_time=200)
    third = make_doc("c", [{"url": "c.jpg", "embedding": unit(2 * FAR)}], pub_time=300)
    cluster = make_cluster(first, second, third)
    cluster.saved_annotation_doc = third

    assert cluster.images[0] == "c.jpg"


def test_a_photo_without_an_embedding_is_still_used() -> None:
    """Older documents have none; a possible duplicate beats no picture."""
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg"}]),
        make_doc("b", [{"url": "b.jpg"}]),
        make_doc("c", [{"url": "c.jpg"}]),
    )

    assert cluster.images == ("a.jpg", "b.jpg", "c.jpg")


def test_the_same_url_twice_is_shown_once() -> None:
    cluster = make_cluster(
        make_doc("a", [{"url": "same.jpg"}]),
        make_doc("b", [{"url": "same.jpg"}]),
        make_doc("c", [{"url": "other.jpg"}]),
    )

    assert cluster.images == ("same.jpg", "other.jpg")


def test_a_picture_only_one_channel_has_is_not_shown() -> None:
    """A lone image in a well-covered story is usually the channel's branding."""
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}]),
        Document(
            url="https://t.me/b/1", channel_id="b", post_id=1, views=1, pub_time=200
        ),
        Document(
            url="https://t.me/c/1", channel_id="c", post_id=1, views=1, pub_time=300
        ),
        Document(
            url="https://t.me/d/1", channel_id="d", post_id=1, views=1, pub_time=400
        ),
    )

    assert cluster.images == ()


def test_a_single_source_story_still_shows_its_photo() -> None:
    """One channel out of one having a picture is not a minority."""
    cluster = make_cluster(make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}]))

    assert cluster.images == ("a.jpg",)


def test_embeddings_of_different_widths_do_not_break_comparison() -> None:
    """An encoder swap leaves both widths in the annotation cache at once.

    They cannot be compared with each other, so the picture of the odd width is
    kept — but the pair that does share a width is still deduplicated.
    """
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}]),
        make_doc("b", [{"url": "b.jpg", "embedding": [1.0, 0.0, 0.0]}]),
        make_doc("c", [{"url": "c.jpg", "embedding": unit(NEAR)}]),
    )

    assert cluster.images == ("a.jpg", "b.jpg")


def test_a_zero_embedding_does_not_break_comparison() -> None:
    """It carries no direction, so it cannot be compared — keep the photo."""
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": [0.0, 0.0]}]),
        make_doc("b", [{"url": "b.jpg", "embedding": [0.0, 0.0]}]),
        make_doc("c", [{"url": "c.jpg", "embedding": unit(0.0)}]),
    )

    assert cluster.images == ("a.jpg", "b.jpg", "c.jpg")
