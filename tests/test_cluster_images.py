"""Tests for choosing a cluster's photos.

Photos are gathered from every channel that posted one, which is the only way a
story covered by a channel that writes well but was not there gets a picture at
all. The cost is that the same wire photo arrives once per channel, under a
different Telegram CDN URL each time, so URL comparison cannot see the
duplication and CLIP embeddings have to.
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


def make_doc(
    channel_id: str,
    images: list[dict[str, object]],
    pub_time: int = 100,
    videos: tuple[str, ...] = (),
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

    assert len(cluster.images) == MAX_CLUSTER_MEDIA


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


def test_photos_survive_a_trip_through_storage() -> None:
    """The daemon re-reads posted clusters from storage on every iteration.

    A stored document was written short, which dropped `embedded_images` and
    kept `images` — so the ratio gate still passed while the candidate list came
    back empty, and the next update rewrote a published post without any of its
    photos. Photos left a live post and came back on their own, depending only
    on whether the sources happened to be re-crawled that iteration.
    """
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}]),
        make_doc("b", [{"url": "b.jpg", "embedding": unit(FAR)}]),
    )

    restored = Cluster.deserialize(cluster.serialize())

    assert restored.images == cluster.images == ("a.jpg", "b.jpg")


def test_stored_photos_are_still_deduplicated_by_content() -> None:
    """Without the embeddings, dedup falls back to URLs — which are per channel.

    So the same wire photo from four channels filled the whole slideshow with
    one picture, but only for posts that had been through storage.
    """
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}]),
        make_doc("b", [{"url": "b.jpg", "embedding": unit(NEAR)}]),
    )

    restored = Cluster.deserialize(cluster.serialize())

    assert restored.images == ("a.jpg",)


def test_a_zero_embedding_does_not_break_comparison() -> None:
    """It carries no direction, so it cannot be compared — keep the photo."""
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": [0.0, 0.0]}]),
        make_doc("b", [{"url": "b.jpg", "embedding": [0.0, 0.0]}]),
        make_doc("c", [{"url": "c.jpg", "embedding": unit(0.0)}]),
    )

    assert cluster.images == ("a.jpg", "b.jpg", "c.jpg")


# --------------------------------------------------------------- video and mix


def test_a_video_is_taken_from_any_channel_that_posted_one() -> None:
    """Not only from the channel whose text was chosen.

    Videos used to be read off `annotation_doc` alone, so a story that ten
    channels filmed had no video whenever the eleventh wrote it best.
    """
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}]),
        make_doc("b", [], videos=("b.mp4",)),
        make_doc("c", [{"url": "c.jpg", "embedding": unit(FAR)}]),
    )

    assert cluster.videos == ("b.mp4",)


def test_a_video_only_one_channel_has_is_not_shown() -> None:
    """The same corroboration rule photos have: one source is not the story.

    A video used to need no confirmation at all, while a photo needed 40% of the
    sources — so the weaker evidence had the lower bar.
    """
    cluster = make_cluster(
        make_doc("a", [], videos=("a.mp4",)),
        make_doc("b", []),
        make_doc("c", []),
        make_doc("d", []),
    )

    assert cluster.videos == ()
    assert cluster.media == ()


def test_photos_and_videos_are_chosen_together() -> None:
    """One slideshow carries both, so they compete for the same four slots."""
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}]),
        make_doc("b", [], videos=("b.mp4",)),
    )

    assert [(item.type, item.url) for item in cluster.media] == [
        ("photo", "a.jpg"),
        ("video", "b.mp4"),
    ]


def test_one_media_item_per_channel_across_types() -> None:
    """A channel that posted both is still one source, so it gets one slot."""
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}], videos=("a.mp4",)),
        make_doc("b", [{"url": "b.jpg", "embedding": unit(FAR)}]),
    )

    assert len(cluster.media) == 2


def test_the_same_video_url_from_two_channels_is_shown_once() -> None:
    cluster = make_cluster(
        make_doc("a", [], videos=("same.mp4",)),
        make_doc("b", [], videos=("same.mp4",)),
        make_doc("c", [], videos=("other.mp4",)),
    )

    assert cluster.videos == ("same.mp4", "other.mp4")


def test_media_survives_a_trip_through_storage() -> None:
    cluster = make_cluster(
        make_doc("a", [{"url": "a.jpg", "embedding": unit(0.0)}]),
        make_doc("b", [], videos=("b.mp4",)),
    )

    restored = Cluster.deserialize(cluster.serialize())

    assert restored.media == cluster.media


def make_video_doc(
    channel_id: str,
    videos: tuple[str, ...],
    embedded_videos: list[dict[str, object]],
    pub_time: int = 100,
) -> Document:
    return Document(
        url=f"https://t.me/{channel_id}/1",
        channel_id=channel_id,
        post_id=1,
        views=1,
        pub_time=pub_time,
        videos=videos,
        embedded_videos=embedded_videos,
    )


def test_the_same_footage_from_two_channels_is_shown_once() -> None:
    """Different urls, one clip — and only the preview can tell.

    Two channels posting the same video get a different CDN url from each, so the
    url check let both into the slideshow. What is comparable is the still
    Telegram renders for a video, embedded like any other picture.
    """
    cluster = make_cluster(
        make_video_doc("a", ("a.mp4",), [{"url": "a.mp4", "embedding": unit(0.0)}]),
        make_video_doc("b", ("b.mp4",), [{"url": "b.mp4", "embedding": unit(NEAR)}]),
        make_video_doc("c", ("c.mp4",), [{"url": "c.mp4", "embedding": unit(FAR)}]),
    )

    assert cluster.videos == ("a.mp4", "c.mp4")


def test_a_video_whose_preview_matches_a_photo_wins_the_slot() -> None:
    """A channel's clip and another's still of the same frame are one thing.

    Both are kept only because nothing could compare them before. The video is
    the stronger material, and it is offered first, so it takes the slot.
    """
    cluster = make_cluster(
        make_video_doc("a", ("a.mp4",), [{"url": "a.mp4", "embedding": unit(0.0)}]),
        make_doc("b", [{"url": "b.jpg", "embedding": unit(NEAR)}]),
    )

    assert cluster.videos == ("a.mp4",)
    assert cluster.images == ()


def test_a_video_without_a_preview_is_still_shown() -> None:
    """Same rule as a photo with no embedding: a possible repeat beats nothing."""
    cluster = make_cluster(
        make_video_doc("a", ("a.mp4",), []),
        make_video_doc("b", ("b.mp4",), []),
    )

    assert cluster.videos == ("a.mp4", "b.mp4")


def test_a_videos_preview_vector_survives_storage() -> None:
    """Same lesson as the photos: the stored form has to keep what picks media.

    Otherwise a published post's clips are compared by url again from the next
    iteration on, and the duplicate footage comes back.
    """
    cluster = make_cluster(
        make_video_doc("a", ("a.mp4",), [{"url": "a.mp4", "embedding": unit(0.0)}]),
        make_video_doc("b", ("b.mp4",), [{"url": "b.mp4", "embedding": unit(NEAR)}]),
    )

    restored = Cluster.deserialize(cluster.serialize())

    assert restored.videos == ("a.mp4",)
