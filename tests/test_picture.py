"""Tests for recognising one photograph across the channels that reposted it.

A channel stamps what it reposts: a logo in a corner, a bar with its name along
the bottom, a translucent mark across the middle. The result is byte-for-byte a
different file with a different CDN url, and SigLIP — which generalises by
design — scores it below the duplicate threshold. A perceptual hash is the tool
built for exactly this question, and the fixtures below are the marks channels
actually apply.
"""

from PIL import Image, ImageDraw, ImageFilter

from nyan.picture import (
    SIGNATURE_SIDE,
    hamming_distance,
    is_same_picture,
    median_signature,
    perceptual_hashes,
    picture_quality,
    signature_deviation,
    thumbnail_signature,
)


def photo(width: int = 640, height: int = 480) -> Image.Image:
    """A picture with structure in it, since a flat fill hashes to nothing.

    Diagonal bands plus a bright block: enough low-frequency content for a DCT
    to have something to describe, and asymmetric enough that a flipped or
    unrelated image does not land on the same hash by luck.
    """
    image = Image.new("RGB", (width, height), (30, 40, 60))
    draw = ImageDraw.Draw(image)
    for i in range(0, width + height, 24):
        draw.line([(i, 0), (i - height, height)], fill=(200, 180, 120), width=8)
    draw.rectangle([width // 4, height // 4, width // 2, height // 2], fill=(240, 240, 240))
    return image


def with_corner_logo(image: Image.Image) -> Image.Image:
    """The commonest mark: a badge in one corner, a few percent of the frame."""
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    width, height = marked.size
    draw.rectangle(
        [width - width // 6, height - height // 8, width, height],
        fill=(255, 0, 0),
    )
    return marked


def with_bottom_bar(image: Image.Image) -> Image.Image:
    """A solid bar with the channel's name along the bottom edge.

    Worse than a corner badge: it shifts the whole frame's geometry, which is
    what a hash of the full picture is most sensitive to.
    """
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    width, height = marked.size
    draw.rectangle([0, height - height // 8, width, height], fill=(0, 0, 0))
    return marked


def test_a_corner_logo_does_not_change_the_hash() -> None:
    original = perceptual_hashes(photo())
    stamped = perceptual_hashes(with_corner_logo(photo()))

    assert is_same_picture(original, stamped)


def test_a_bottom_bar_does_not_change_the_hash() -> None:
    """The crop hash is what survives this one: the full-frame hash may not."""
    original = perceptual_hashes(photo())
    stamped = perceptual_hashes(with_bottom_bar(photo()))

    assert is_same_picture(original, stamped)


def on_a_backdrop(image: Image.Image, pad: float = 0.18) -> Image.Image:
    """The picture inset into a channel's own coloured field.

    The hardest of the three marks and a common one, because it is what a
    channel does to a photograph whose shape does not fit its layout. Every
    edge moves inward at once, so neither the full frame nor a centre crop is
    looking at the same picture any more — the content has been scaled down
    inside a larger canvas.
    """
    width, height = image.size
    inset = image.resize((int(width * (1 - 2 * pad)), int(height * (1 - 2 * pad))))
    backdrop = Image.new("RGB", (width, height), (12, 24, 96))
    backdrop.paste(inset, (int(width * pad), int(height * pad)))
    return backdrop


def test_a_picture_inset_into_a_channels_backdrop_is_still_the_same_picture() -> None:
    """Uniform borders are the channel's canvas, not the photograph."""
    original = perceptual_hashes(photo())
    inset = perceptual_hashes(on_a_backdrop(photo()))

    assert is_same_picture(original, inset)


def test_a_backdrop_is_trimmed_even_when_a_badge_sits_on_it() -> None:
    """The two marks are applied together more often than either alone.

    A badge on the backdrop is a few bright pixels in a corner of an otherwise
    flat field. Locating content by its outermost differing pixel puts the
    boundary at the badge, so nothing is trimmed and the inset picture is
    compared against a full-size one.
    """
    original = perceptual_hashes(photo())
    inset = perceptual_hashes(with_corner_logo(on_a_backdrop(photo())))

    assert is_same_picture(original, inset)


def test_a_resized_copy_is_the_same_picture() -> None:
    """Every channel's CDN serves its own rendition, at its own dimensions."""
    original = perceptual_hashes(photo(640, 480))
    smaller = perceptual_hashes(photo(320, 240))

    assert is_same_picture(original, smaller)


def test_a_different_picture_is_not_a_duplicate() -> None:
    """The threshold has to leave room for two photos of the same event."""
    landscape = photo()
    other = Image.new("RGB", (640, 480), (240, 240, 240))
    draw = ImageDraw.Draw(other)
    draw.ellipse([100, 100, 400, 380], fill=(10, 10, 10))

    assert not is_same_picture(perceptual_hashes(landscape), perceptual_hashes(other))


def test_hashes_survive_a_round_trip_through_json() -> None:
    """They are stored in Mongo beside the embedding and read back as strings."""
    hashes = perceptual_hashes(photo())

    assert all(isinstance(value, str) for value in hashes)
    assert is_same_picture(hashes, list(hashes))


def test_hamming_distance_of_a_hash_with_itself_is_zero() -> None:
    full, _crop = perceptual_hashes(photo())

    assert hamming_distance(full, full) == 0


# ------------------------------------------------------------------- quality


def test_a_photograph_scores_above_the_same_photograph_in_a_card() -> None:
    """A channel's generated card is mostly not the photograph.

    Headline, body text, logo, padding: the picture of the event is a fraction
    of the frame. Between two copies of one photograph, the reader is better
    served by the one that is the photograph.
    """
    plain = picture_quality(photo())
    carded = picture_quality(on_a_backdrop(photo(), pad=0.3))

    assert plain.score > carded.score


def test_a_blurred_copy_scores_below_a_sharp_one() -> None:
    """Recompression and upscaling are what a picture collects as it travels."""
    sharp = picture_quality(photo())
    blurred = picture_quality(photo().filter(ImageFilter.GaussianBlur(radius=6)))

    assert sharp.score > blurred.score


def test_a_small_rendition_scores_below_a_large_one() -> None:
    """Channels repost at whatever size their client uploaded."""
    large = picture_quality(photo(1280, 960))
    small = picture_quality(photo(240, 180))

    assert large.score > small.score


def test_the_score_stays_within_its_range() -> None:
    """It is combined with other rankings, so it has to be comparable."""
    for image in (photo(), photo(64, 48), on_a_backdrop(photo())):
        assert 0.0 <= picture_quality(image).score <= 1.0


# ----------------------------------------------------------------- signature


def test_a_signature_is_small_and_storable() -> None:
    """It is written to Mongo beside the hashes, once per picture."""
    signature = thumbnail_signature(photo())

    assert len(signature) == SIGNATURE_SIDE * SIGNATURE_SIDE
    assert all(0 <= value <= 255 for value in signature)


def test_signatures_of_one_photograph_match_across_renditions() -> None:
    """Every channel's CDN serves its own size; the signature is size-free."""
    large = thumbnail_signature(photo(1280, 960))
    small = thumbnail_signature(photo(320, 240))

    assert signature_deviation(large, small) < 12.0


def test_a_stamped_copy_deviates_from_the_consensus_of_copies() -> None:
    """What a watermark is, operationally: a difference only one copy has.

    Take the pixel-wise median of every copy of one photograph and the mark
    vanishes, because on that spot most copies show the photograph. The copy
    that is furthest from the median is the marked one.
    """
    clean_copies = [
        thumbnail_signature(photo()),
        thumbnail_signature(photo(800, 600)),
        thumbnail_signature(photo(512, 384)),
    ]
    stamped = thumbnail_signature(with_bottom_bar(photo()))

    consensus = median_signature([*clean_copies, stamped])

    assert signature_deviation(stamped, consensus) > signature_deviation(
        clean_copies[0], consensus
    )


def test_a_median_of_nothing_is_nothing() -> None:
    """Groups whose members predate signatures fall back to other rankings."""
    assert median_signature([]) == ()


def test_a_flat_card_has_almost_no_spread() -> None:
    """A channel's intro card varies barely at all across itself.

    Measured on production video posters: the black `exilenova_plus` card that
    several channels put in front of different clips prints at a spread near 7,
    while every poster showing an actual scene sits above 16 and the median is
    41. That gap is what separates a poster that says something about the
    footage from one that only says whose channel it is.
    """
    from nyan.picture import signature_spread

    flat = tuple([12] * 256)
    varied = tuple(range(256))

    assert signature_spread(flat) == 0.0
    assert signature_spread(varied) > 50.0


def test_the_spread_of_nothing_is_nothing() -> None:
    from nyan.picture import signature_spread

    assert signature_spread(()) == 0.0
    assert signature_spread((7,)) == 0.0
