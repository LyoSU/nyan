"""Tests for deciding what a post shows and in what order.

Three questions, in this order: which pictures are the same picture, which of
them the coverage actually agrees on, and how to fill the remaining slots
without repeating what the first one already showed.

The order is the whole answer to "which is the main one". The carousel shows
the first item, and so does the site, so nothing needs to be flagged as the
lead — it only has to be sorted there.
"""

import math

from nyan.picture import SIGNATURE_SIDE
from nyan.media import (
    AUTHORITY_ANONYMOUS,
    AUTHORITY_MEDIA,
    AUTHORITY_OFFICIAL,
    MEDIA_PHOTO,
    MEDIA_VIDEO,
    MediaCandidate,
    group_media,
    select_media,
)


def unit(angle: float) -> tuple[float, ...]:
    """A point on the unit circle, so cosine similarity is a known quantity."""
    return (math.cos(angle), math.sin(angle))


# Two vectors this far apart have cosine similarity ~0.995: one picture.
NEAR = 0.1
# And these ~0.54: different pictures.
FAR = 1.0

# A hash pair for a picture nothing else here resembles. Hashes are compared by
# Hamming distance, so hex digits that differ everywhere are what "unrelated"
# looks like.
UNIQUE_HASHES = ("0000000000000000", "0000000000000000")
OTHER_HASHES = ("ffffffffffffffff", "ffffffffffffffff")


def candidate(
    channel: str,
    url: str,
    angle: float = 0.0,
    pub_time: int = 100,
    hashes: tuple[str, ...] = UNIQUE_HASHES,
    relevance: float = 1.0,
    media_type: str = MEDIA_PHOTO,
    authority: int = AUTHORITY_ANONYMOUS,
    quality: float = 0.0,
    signature: tuple[int, ...] = (),
    duration: int = 0,
) -> MediaCandidate:
    return MediaCandidate(
        type=media_type,
        url=url,
        channel_id=channel,
        pub_time=pub_time,
        embedding=unit(angle),
        hashes=hashes,
        relevance=relevance,
        authority=authority,
        quality=quality,
        signature=signature,
        duration=duration,
    )


def urls(items: object) -> list[str]:
    return [item.url for item in items]  # type: ignore[attr-defined]


def test_the_picture_most_channels_posted_comes_first() -> None:
    """Consensus is the strongest relevance signal a cluster has.

    Three channels independently choosing one photograph for one story is a
    stronger statement about that photograph than any similarity score. The
    single channel's own picture is not wrong, but it goes behind.
    """
    shared = [
        candidate("a", "wire.jpg", angle=0.0, pub_time=100),
        candidate("b", "wire-copy.jpg", angle=NEAR, pub_time=200),
        candidate("c", "wire-again.jpg", angle=2 * NEAR, pub_time=300),
    ]
    lone = candidate("d", "own.jpg", angle=FAR, pub_time=50, hashes=OTHER_HASHES)

    selected = select_media([lone, *shared], limit=4)

    assert urls(selected)[0] == "wire.jpg"


def test_a_post_whose_pictures_nobody_else_posted_shows_none() -> None:
    """One channel's picture of a story is usually its own branding.

    Not a photograph of the event: a logo, a stock illustration, a card the
    channel generates for every post. Nothing confirms it, so nothing shows it.
    """
    selected = select_media(
        [
            candidate("a", "one.jpg", angle=0.0, hashes=UNIQUE_HASHES),
            candidate("b", "two.jpg", angle=FAR, hashes=OTHER_HASHES),
        ],
        limit=4,
    )

    assert urls(selected) == []


def test_an_accountable_source_can_lead_a_post_on_its_own() -> None:
    """Nobody corroborated it, but the source is answerable for it.

    A ministry's photograph of its own announcement will never have a second
    channel behind it — everyone else is quoting the ministry. Requiring
    consensus of a source that is itself the origin leaves such a post with no
    picture at all, which is worse than showing the one the source published.
    """
    selected = select_media(
        [
            candidate("ministry", "official.jpg", angle=0.0, authority=AUTHORITY_OFFICIAL),
            candidate("aggregator", "own.jpg", angle=FAR, hashes=OTHER_HASHES),
        ],
        limit=4,
    )

    assert urls(selected)[0] == "official.jpg"


def test_an_anonymous_channel_alone_leads_nothing() -> None:
    """Nothing confirms it and nobody is answerable for it.

    This is where the unconfirmed picture is most often a channel's own
    branding — a logo, a generated card — and least often a photograph of the
    event.
    """
    selected = select_media(
        [
            candidate("anon", "one.jpg", angle=0.0, authority=AUTHORITY_ANONYMOUS),
            candidate("anon2", "two.jpg", angle=FAR, hashes=OTHER_HASHES),
        ],
        limit=4,
    )

    assert urls(selected) == []


def test_between_two_equally_carried_pictures_the_better_source_leads() -> None:
    """Consensus first, and accountability only where consensus is tied."""
    from_media = [
        candidate("m1", "media.jpg", angle=0.0, pub_time=100, authority=AUTHORITY_MEDIA),
        candidate(
            "m2", "media-2.jpg", angle=NEAR, pub_time=200, authority=AUTHORITY_MEDIA
        ),
    ]
    from_anon = [
        candidate("a1", "anon.jpg", angle=2.0, pub_time=50, hashes=OTHER_HASHES),
        candidate(
            "a2",
            "anon-2.jpg",
            angle=2.0 + NEAR,
            pub_time=60,
            hashes=("ffffffff0000ffff",) * 2,
        ),
    ]

    selected = select_media([*from_anon, *from_media], limit=4)

    assert urls(selected)[0] == "media.jpg"


def test_a_stamped_repost_counts_towards_the_same_picture() -> None:
    """What the embedding threshold alone cannot see.

    Measured on production: a photograph reposted under another channel's
    watermark scores ~0.86 by SigLIP — below the duplicate threshold, so it
    used to take a second slot — while its perceptual hash is 6 bits away.
    Counting it as the same picture is what gives the group its third channel.
    """
    stamped = ("00000000000000ff", "00000000000000ff")
    selected = select_media(
        [
            candidate("a", "wire.jpg", angle=0.0, pub_time=100),
            candidate("b", "wire-stamped.jpg", angle=FAR, pub_time=200, hashes=stamped),
            candidate("c", "unrelated.jpg", angle=2 * FAR, pub_time=300, hashes=OTHER_HASHES),
        ],
        limit=4,
    )

    assert urls(selected)[0] == "wire.jpg"
    assert "wire-stamped.jpg" not in urls(selected)


def test_one_picture_per_group_however_many_channels_carried_it() -> None:
    shared = [
        candidate("a", "wire.jpg", angle=0.0, pub_time=100),
        candidate("b", "wire-copy.jpg", angle=NEAR, pub_time=200),
        candidate("c", "wire-again.jpg", angle=2 * NEAR, pub_time=300),
    ]

    assert urls(select_media(shared, limit=4)) == ["wire.jpg"]


def test_the_cleanest_copy_represents_the_group() -> None:
    """The copies differ in ways a reader sees, even when the hash cannot.

    One channel posts the photograph; another posts it as a third of its own
    generated card, under a bar, re-encoded. Same picture, different thing to
    look at — and the earlier of the two is not reliably the cleaner.
    """
    selected = select_media(
        [
            candidate("carded", "in-a-card.jpg", angle=0.0, pub_time=100),
            candidate("clean", "photograph.jpg", angle=NEAR, pub_time=200),
            candidate("also", "another-card.jpg", angle=2 * NEAR, pub_time=300),
        ],
        limit=4,
    )

    assert urls(selected) == ["in-a-card.jpg"]

    better = select_media(
        [
            candidate("carded", "in-a-card.jpg", angle=0.0, pub_time=100),
            candidate("clean", "photograph.jpg", angle=NEAR, pub_time=200, quality=0.9),
            candidate("also", "another-card.jpg", angle=2 * NEAR, pub_time=300),
        ],
        limit=4,
    )

    assert urls(better) == ["photograph.jpg"]


def test_the_copy_that_departs_from_the_others_is_not_shown() -> None:
    """A watermark is a difference only one copy has.

    Three channels posted the photograph and one stamped it. Nothing here knows
    what a watermark looks like: the stamped copy is simply the one furthest
    from what the copies agree the picture is, pixel by pixel.
    """
    plain = (100,) * (SIGNATURE_SIDE * SIGNATURE_SIDE)
    stamped = (100,) * (SIGNATURE_SIDE * SIGNATURE_SIDE // 2) + (255,) * (
        SIGNATURE_SIDE * SIGNATURE_SIDE // 2
    )
    selected = select_media(
        [
            candidate("stamper", "stamped.jpg", angle=0.0, pub_time=100, signature=stamped),
            candidate("clean", "clean.jpg", angle=NEAR, pub_time=200, signature=plain),
            candidate("also", "also-clean.jpg", angle=2 * NEAR, pub_time=300, signature=plain),
        ],
        limit=4,
    )

    assert urls(selected) == ["clean.jpg"]


def test_cleanliness_and_quality_are_weighed_against_each_other() -> None:
    """Neither one alone picks the copy a reader should get.

    Ranking on the watermark measure first makes it absolute: a barely marked
    copy beats an unmarked one whatever the two look like, so a clean 240px
    thumbnail wins over a lightly stamped full-size photograph. Ranking on
    quality first does the reverse and puts the watermark back on screen.
    """
    plain = (100,) * (SIGNATURE_SIDE * SIGNATURE_SIDE)
    barely_marked = (100,) * (SIGNATURE_SIDE * SIGNATURE_SIDE - 8) + (130,) * 8
    selected = select_media(
        [
            # Unmarked, and a poor rendition of the photograph.
            candidate("thumb", "tiny.jpg", angle=0.0, signature=plain, quality=0.05),
            # A faint mark, and the picture itself in full.
            candidate(
                "full", "full.jpg", angle=NEAR, signature=barely_marked, quality=0.95
            ),
            candidate("third", "third.jpg", angle=2 * NEAR, signature=plain, quality=0.1),
        ],
        limit=4,
    )

    assert urls(selected) == ["full.jpg"]


def test_a_heavy_watermark_still_loses_to_a_worse_rendition() -> None:
    """The balance must not tip so far that a stamped copy wins on size."""
    plain = (100,) * (SIGNATURE_SIDE * SIGNATURE_SIDE)
    stamped = (100,) * (SIGNATURE_SIDE * SIGNATURE_SIDE // 2) + (255,) * (
        SIGNATURE_SIDE * SIGNATURE_SIDE // 2
    )
    selected = select_media(
        [
            candidate("clean", "clean.jpg", angle=0.0, signature=plain, quality=0.35),
            candidate("big", "stamped.jpg", angle=NEAR, signature=stamped, quality=1.0),
            candidate("third", "third.jpg", angle=2 * NEAR, signature=plain, quality=0.3),
        ],
        limit=4,
    )

    assert urls(selected) == ["clean.jpg"]


def test_the_earliest_channels_copy_represents_the_group() -> None:
    """Whoever posted first is the source; later copies carry other marks."""
    selected = select_media(
        [
            candidate("late", "late.jpg", angle=NEAR, pub_time=500),
            candidate("early", "early.jpg", angle=0.0, pub_time=100),
            candidate("later", "later.jpg", angle=2 * NEAR, pub_time=900),
        ],
        limit=4,
    )

    assert urls(selected) == ["early.jpg"]


def test_a_second_slot_avoids_what_the_first_already_showed() -> None:
    """The slideshow is filled for variety, not for the ranking alone.

    A near-duplicate the hash did not catch — another frame of the same scene —
    is confirmed by its own channels and would otherwise win the second slot on
    consensus. It is pushed behind the picture that shows something else.
    """
    first = [
        candidate("a", "scene.jpg", angle=0.0, pub_time=100),
        candidate("b", "scene-b.jpg", angle=NEAR, pub_time=200),
        candidate("c", "scene-c.jpg", angle=2 * NEAR, pub_time=300),
    ]
    # Close to the first group, but far enough from every one of its members
    # not to be merged into it: sameness is transitive, so a candidate that
    # matches any member joins the whole group.
    almost = [
        candidate("d", "angle.jpg", angle=0.8, pub_time=400, hashes=("00000000ffffffff",) * 2),
        candidate(
            "e",
            "angle-e.jpg",
            angle=0.8 + NEAR,
            pub_time=500,
            hashes=("00000000ffff0fff",) * 2,
        ),
    ]
    different = [
        candidate("f", "elsewhere.jpg", angle=2.5, pub_time=600, hashes=OTHER_HASHES),
        candidate(
            "g",
            "elsewhere-g.jpg",
            angle=2.5 + NEAR,
            pub_time=700,
            hashes=("ffffffff0000ffff",) * 2,
        ),
    ]

    selected = select_media([*first, *almost, *different], limit=2)

    assert urls(selected) == ["scene.jpg", "elsewhere.jpg"]


def test_one_channel_fills_at_most_one_unconfirmed_slot() -> None:
    """A channel posting six shots of one scene must not fill the slideshow.

    The limit is on unconfirmed pictures only. Two pictures that five channels
    each carried are two things the coverage agreed on, and both belong in the
    post — but six pictures from one channel are one source's gallery.
    """
    selected = select_media(
        [
            candidate("a", "a1.jpg", angle=0.0, authority=AUTHORITY_MEDIA),
            candidate(
                "a", "a2.jpg", angle=FAR, hashes=OTHER_HASHES, authority=AUTHORITY_MEDIA
            ),
            candidate(
                "a",
                "a3.jpg",
                angle=2 * FAR,
                hashes=("ffffffff0000ffff",) * 2,
                authority=AUTHORITY_MEDIA,
            ),
        ],
        limit=4,
    )

    assert len(selected) == 1


def test_two_confirmed_pictures_from_the_same_channels_both_show() -> None:
    """Agreement on two pictures is agreement on two pictures."""
    selected = select_media(
        [
            candidate("a", "first.jpg", angle=0.0, pub_time=100),
            candidate("b", "first-b.jpg", angle=NEAR, pub_time=200),
            candidate("a", "second.jpg", angle=2.0, pub_time=100, hashes=OTHER_HASHES),
            candidate(
                "b",
                "second-b.jpg",
                angle=2.0 + NEAR,
                pub_time=200,
                hashes=("ffffffff0000ffff",) * 2,
            ),
        ],
        limit=4,
    )

    assert len(selected) == 2


def test_a_channels_video_outranks_its_own_photo() -> None:
    """Footage from the scene is stronger material than a still of it."""
    selected = select_media(
        [
            candidate(
                "a",
                "clip.mp4",
                angle=0.0,
                media_type=MEDIA_VIDEO,
                authority=AUTHORITY_MEDIA,
            ),
            candidate(
                "a", "photo.jpg", angle=FAR, hashes=OTHER_HASHES, authority=AUTHORITY_MEDIA
            ),
        ],
        limit=4,
    )

    assert urls(selected) == ["clip.mp4"]


def test_a_lone_picture_from_a_document_far_from_the_story_is_dropped() -> None:
    """The neighbouring story the clustering pulled in at its threshold.

    Its picture is of something else, and nothing in the cluster confirms it,
    so a slot it fills is a slot showing the wrong event.
    """
    confirmed = [
        candidate("a", "wire.jpg", angle=0.0, pub_time=100),
        candidate("b", "wire-copy.jpg", angle=NEAR, pub_time=200),
    ]
    stray = candidate(
        "c", "other-story.jpg", angle=FAR, pub_time=300, hashes=OTHER_HASHES, relevance=0.5
    )

    selected = select_media([*confirmed, stray], limit=4, min_relevance=0.9)

    assert urls(selected) == ["wire.jpg"]


def test_accountability_does_not_make_an_unrelated_picture_relevant() -> None:
    """A trustworthy source can vouch for a picture, not change its subject.

    This is the failure mode of a reply whose own post has no media: the older
    context post is in the same cluster and has a legitimate photograph, but
    that photograph still belongs to the older event.  Authority is considered
    only after relevance has admitted a picture to the current story.
    """
    selected = select_media(
        [
            candidate(
                "ministry",
                "previous-event.jpg",
                authority=AUTHORITY_OFFICIAL,
                relevance=0.2,
            )
        ],
        limit=4,
        min_relevance=0.9,
    )

    assert urls(selected) == []


def test_a_video_can_be_the_lead_when_it_is_the_confirmed_one() -> None:
    """Footage from the scene is stronger material than a wire photo."""
    video = [
        candidate("a", "clip.mp4", angle=0.0, pub_time=100, media_type=MEDIA_VIDEO),
        candidate("b", "clip-b.mp4", angle=NEAR, pub_time=200, media_type=MEDIA_VIDEO),
        candidate("c", "clip-c.mp4", angle=2 * NEAR, pub_time=300, media_type=MEDIA_VIDEO),
    ]
    photo = [
        candidate("d", "photo.jpg", angle=FAR, pub_time=400, hashes=OTHER_HASHES),
        candidate(
            "e",
            "photo-e.jpg",
            angle=FAR + NEAR,
            pub_time=500,
            hashes=("ffffffff0000ffff",) * 2,
        ),
    ]

    selected = select_media([*photo, *video], limit=4)

    assert urls(selected)[0] == "clip.mp4"


def test_two_clips_of_different_lengths_are_not_one_clip() -> None:
    """Posters can agree where the footage does not.

    Telegram re-encodes what it is given, so one channel's poster of a clip can
    be another channel's poster of a different clip from the same scene — same
    place, same light, same framing. The length is what tells them apart, and
    it costs nothing: it is printed on the player.
    """
    groups = group_media(
        [
            candidate("a", "short.mp4", angle=0.0, media_type=MEDIA_VIDEO, duration=15),
            candidate("b", "long.mp4", angle=NEAR, media_type=MEDIA_VIDEO, duration=94),
        ]
    )

    assert len(groups) == 2


def test_two_clips_of_the_same_length_still_group() -> None:
    """A re-encode shifts the reported length by a second at most."""
    groups = group_media(
        [
            candidate("a", "one.mp4", angle=0.0, media_type=MEDIA_VIDEO, duration=42),
            candidate("b", "two.mp4", angle=NEAR, media_type=MEDIA_VIDEO, duration=43),
        ]
    )

    assert len(groups) == 1


def test_a_clip_with_no_known_length_is_compared_as_before() -> None:
    """Older documents carry none, and no length is not a mismatch."""
    groups = group_media(
        [
            candidate("a", "one.mp4", angle=0.0, media_type=MEDIA_VIDEO, duration=42),
            candidate("b", "two.mp4", angle=NEAR, media_type=MEDIA_VIDEO, duration=0),
        ]
    )

    assert len(groups) == 1


def test_groups_merge_transitively() -> None:
    """A matches B by hash, B matches C by embedding: all three are one picture.

    Comparing every pair against every other would leave the group split into
    two, and the same photograph would take two of the four slots.
    """
    groups = group_media(
        [
            candidate("a", "a.jpg", angle=0.0, hashes=("0000000000000000",) * 2),
            candidate("b", "b.jpg", angle=FAR, hashes=("0000000000000003",) * 2),
            candidate("c", "c.jpg", angle=FAR + NEAR, hashes=OTHER_HASHES),
        ]
    )

    assert len(groups) == 1
    assert groups[0].channels == {"a", "b", "c"}


def test_a_channel_posting_the_same_picture_twice_counts_once() -> None:
    """Consensus counts channels, not attachments: a channel is one source."""
    groups = group_media(
        [
            candidate("a", "one.jpg", angle=0.0),
            candidate("a", "one-again.jpg", angle=NEAR),
        ]
    )

    assert groups[0].channels == {"a"}


def test_two_copies_cannot_reveal_a_watermark() -> None:
    """The median of two values is their mean, so both depart equally.

    With one copy and one stamped copy there is nothing to say which is which:
    the consensus they form is exactly halfway between them. Ranking then falls
    to quality, which is the honest answer — pretending otherwise would pick a
    copy at random and call it clean.
    """
    plain = (100,) * (SIGNATURE_SIDE * SIGNATURE_SIDE)
    stamped = (100,) * (SIGNATURE_SIDE * SIGNATURE_SIDE // 2) + (255,) * (
        SIGNATURE_SIDE * SIGNATURE_SIDE // 2
    )
    selected = select_media(
        [
            candidate("a", "stamped.jpg", angle=0.0, signature=stamped, quality=0.2),
            candidate("b", "clean.jpg", angle=NEAR, signature=plain, quality=0.8),
        ],
        limit=4,
    )

    assert urls(selected) == ["clean.jpg"]
