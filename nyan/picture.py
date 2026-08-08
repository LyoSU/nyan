"""Recognising one photograph across the channels that reposted it.

Deliberately free of any ML import: this is arithmetic over bits, and pulling
`nyan.vision` in would make every duplicate test load SigLIP — a minute of
startup, the weights on disk, and a failure whenever the torch stack is out of
step, all to compare two integers.

Why a perceptual hash at all, when the embeddings are already there: SigLIP
generalises by design, so its score drifts continuously and has no clean line
where "the same photograph" ends. A stamped repost lands below the duplicate
threshold, and lowering that threshold merges two genuinely different photos of
one event instead. A perceptual hash answers the narrow question — is this the
same picture, re-encoded and marked up — which is the one channels keep posing.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import imagehash
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageChops

# 64 bits, the standard size: enough to separate unrelated photos, small enough
# to store beside every image in Mongo without thinking about it.
HASH_SIZE = 8

# How much of the frame the second hash keeps, measured from the centre. A
# channel's bar along the bottom edge is the mark a full-frame hash handles
# worst — the DCT sees a different picture, not a marked one — and it typically
# covers under a tenth of the height. Cropping a fifth off each dimension
# removes it with room to spare, while keeping enough of the picture that two
# different photos still hash apart.
CROP_RATIO = 0.8

# How far a pixel may sit from the corner colour and still count as backdrop
# rather than picture. Not zero, because JPEG leaves a flat fill slightly
# uneven, and an exact comparison would find content in the noise and trim
# nothing at all.
BORDER_TOLERANCE = 12

# What share of a row must differ from the backdrop for the row to be picture.
# Set well above nothing so that a badge lying on the flat field — a few
# percent of its row — does not hold the boundary out at the frame's edge, and
# well below a half so that a picture's own thin top edge still counts.
BORDER_CONTENT_SHARE = 0.1

# The smallest share of the frame that trimming may leave. A backstop against
# the pathological case — a picture flat enough that its own subject reads as
# backdrop, and trimming would keep only a highlight — not a limit on how much
# furniture a channel may add. Cards where the photograph is under a third of
# the frame are ordinary, so a stricter bound here silently switched trimming
# off for exactly the pictures that need it most.
MIN_CONTENT_RATIO = 0.15

# Bits that may differ before two hashes stop being the same picture. The value
# where the near-duplicate literature settles for 64-bit hashes, and the number
# most worth re-checking against a live sample: see scripts/calibrate_media.py.
DEFAULT_MAX_DISTANCE = 10


def _trim_uniform_border(image: Image.Image) -> Image.Image:
    """The picture without the flat field a channel inset it into.

    A channel whose layout wants one shape and whose photograph has another
    pads it onto a coloured canvas. That moves every edge inward at once, so
    neither hash is looking at the same picture any more — and unlike a badge
    or a bar, no fixed crop can find it, because how much was added varies with
    the photograph.

    What does not vary is that the field is flat. So the content is located
    rather than assumed: the frame is compared against its own corner colour,
    and everything that differs from it is the picture.

    A row counts as picture when enough of it differs, not when any of it does.
    The badge and the backdrop are applied together more often than either
    alone, and a badge is a handful of bright pixels in a corner of the flat
    field — under the outermost-differing-pixel rule it puts the boundary at
    the frame's edge and nothing is trimmed at all.

    Trimming is skipped when it would claim most of the frame, since a
    photograph that really is dark at the edges — a night shot, a studio
    portrait — would otherwise be cropped down to its highlights.
    """
    backdrop = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    difference = ImageChops.difference(image, backdrop).convert("L")
    content = np.asarray(difference) > BORDER_TOLERANCE
    rows = _dense_span(content.mean(axis=1))
    columns = _dense_span(content.mean(axis=0))
    if rows is None or columns is None:
        return image
    width, height = image.size
    if (columns[1] - columns[0]) * (rows[1] - rows[0]) < width * height * MIN_CONTENT_RATIO:
        return image
    return image.crop((columns[0], rows[0], columns[1], rows[1]))


def _dense_span(shares: NDArray[np.float64]) -> tuple[int, int] | None:
    """The longest unbroken run of positions that are picture rather than field.

    The longest run, not the first and last: a badge lying on the flat field
    makes its own short run of dense rows, separated from the photograph by the
    field between them. Taking the outermost dense positions would span both
    and trim nothing, while the longest run is the photograph whatever size the
    badge is — which is what makes this independent of a threshold tuned to one
    channel's logo.
    """
    dense = shares > BORDER_CONTENT_SHARE
    if not dense.any():
        return None
    # Where the run structure changes, padded so a run touching either end is
    # closed off rather than running past the edge of the array.
    edges = np.flatnonzero(np.diff(np.concatenate(([False], dense, [False]))))
    starts, ends = edges[::2], edges[1::2]
    longest = int(np.argmax(ends - starts))
    return int(starts[longest]), int(ends[longest])


def _centre_crop(image: Image.Image, ratio: float) -> Image.Image:
    width, height = image.size
    kept_width, kept_height = int(width * ratio), int(height * ratio)
    left, top = (width - kept_width) // 2, (height - kept_height) // 2
    return image.crop((left, top, left + kept_width, top + kept_height))


def _hash(image: Image.Image) -> str:
    return str(imagehash.phash(image, hash_size=HASH_SIZE))


def perceptual_hashes(image: Image.Image) -> tuple[str, str]:
    """Two hashes of one picture: the whole frame, and its centre.

    Both are needed because the two common marks fail differently. A badge in a
    corner leaves the full-frame hash intact, and survives the crop only if it
    happened to sit outside it. A bar along an edge is the opposite: it shifts
    the full frame's geometry, and the centre is the only part left untouched.
    Keeping both means either mark can be seen through, and a match on either
    one is a match — see `is_same_picture`.

    Hex strings rather than `ImageHash` objects, because these are stored in
    Mongo alongside the embedding and read back by whatever runs next.
    """
    prepared = _trim_uniform_border(image.convert("RGB"))
    return _hash(prepared), _hash(_centre_crop(prepared, CROP_RATIO))


def hamming_distance(first: str, second: str) -> int:
    """How many bits two hex-encoded hashes disagree on."""
    return (int(first, 16) ^ int(second, 16)).bit_count()


def is_same_picture(
    first: Sequence[str],
    second: Sequence[str],
    max_distance: int = DEFAULT_MAX_DISTANCE,
) -> bool:
    """Whether two hash pairs describe one photograph.

    A match on either hash is a match, and the pairs are compared position by
    position: a full frame is only ever compared with a full frame, since a
    crop's hash is a hash of a different picture and would be near-random
    against it.
    """
    return any(
        hamming_distance(one, other) <= max_distance
        for one, other in zip(first, second, strict=True)
    )


# A picture at least this many pixels is as large as the ranking cares about.
# Roughly a 1280x960 photograph: past that, more resolution changes nothing for
# a reader on a phone, and rewarding it would promote an upscaled copy over the
# original.
REFERENCE_PIXELS = 1_200_000

# The Laplacian variance of a photograph that is in focus and has not been
# through several rounds of recompression. Measured on the fixtures rather than
# derived: what matters is the ordering it produces, not the absolute value.
REFERENCE_SHARPNESS = 400.0

# How the three readings are weighted against each other. Content dominates
# because it answers a different question from the other two: not how good the
# picture is, but how much of the frame is the picture at all. A sharp,
# high-resolution card whose photograph occupies a third of it is still mostly
# not a photograph.
CONTENT_WEIGHT = 0.5
SHARPNESS_WEIGHT = 0.3
SIZE_WEIGHT = 0.2


@dataclass(frozen=True)
class PictureQuality:
    """What can be told about a picture without asking a model.

    Used to choose between copies of one photograph. Every channel that
    reposted it serves its own rendition — its own size, its own recompression,
    its own furniture around the edges — and the reader should get the cleanest
    of them, not whichever happened to be crawled first.

    `content` is the share of the frame left after the channel's backdrop is
    trimmed away, so it reads low for a generated card and 1.0 for a
    photograph posted as itself. `sharpness` is the variance of the Laplacian,
    the standard focus measure, which falls with every re-encode and with
    upscaling. `pixels` is the size of the picture proper, after trimming.
    """

    content: float
    sharpness: float
    pixels: int

    @property
    def score(self) -> float:
        """The three readings as one number in [0, 1], for ranking."""
        return (
            CONTENT_WEIGHT * self.content
            + SHARPNESS_WEIGHT * min(self.sharpness / REFERENCE_SHARPNESS, 1.0)
            + SIZE_WEIGHT * min(self.pixels / REFERENCE_PIXELS, 1.0)
        )


def _sharpness(image: Image.Image) -> float:
    """Variance of the Laplacian: how much fine detail the picture still has.

    Written out rather than convolved, because a four-neighbour Laplacian is
    four array slices and adding a convolution dependency for it would be the
    more surprising choice.
    """
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    if min(grey.shape) < 3:
        return 0.0
    laplacian = (
        grey[:-2, 1:-1]
        + grey[2:, 1:-1]
        + grey[1:-1, :-2]
        + grey[1:-1, 2:]
        - 4 * grey[1:-1, 1:-1]
    )
    return float(laplacian.var())


def picture_quality(image: Image.Image) -> PictureQuality:
    """How clean and how large this rendition of a picture is."""
    prepared = image.convert("RGB")
    trimmed = _trim_uniform_border(prepared)
    full_pixels = prepared.width * prepared.height
    trimmed_pixels = trimmed.width * trimmed.height
    return PictureQuality(
        content=trimmed_pixels / full_pixels if full_pixels else 0.0,
        sharpness=_sharpness(trimmed),
        pixels=trimmed_pixels,
    )


# The side of the stored thumbnail, in pixels. Small enough that 346 channels'
# worth of pictures cost nothing in Mongo — 256 bytes each, beside embeddings
# that are twenty times that — and large enough to see the marks channels
# actually apply, which are bars and badges covering a tenth of the frame
# rather than single pixels.
SIGNATURE_SIDE = 16


def thumbnail_signature(image: Image.Image) -> tuple[int, ...]:
    """A tiny greyscale print of the picture, for comparing copies pixel by pixel.

    Stored rather than recomputed, because the comparison happens when a post
    is assembled and the pictures are long since crawled: fetching every copy
    again would put the network inside `Cluster.media`, which the daemon calls
    for every cluster on every iteration.

    Trimmed and squared first, so that two channels' renditions of one
    photograph — different sizes, different amounts of furniture — line up
    against each other. Aspect ratio is deliberately discarded: a copy cropped
    to a different shape still has to be comparable, and the alternative is
    having no comparison at all for exactly the copies that were altered most.
    """
    trimmed = _trim_uniform_border(image.convert("RGB"))
    thumbnail = trimmed.convert("L").resize(
        (SIGNATURE_SIDE, SIGNATURE_SIDE), Image.Resampling.BILINEAR
    )
    return tuple(int(value) for value in np.asarray(thumbnail).reshape(-1))


def median_signature(signatures: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """The pixel-wise median of several copies of one photograph.

    This is what a watermark is measured against. A mark exists in one copy and
    not in the others, so at its pixels the median still holds the photograph —
    and the copy that departs from the median furthest is the marked one. It
    needs no detector, no model and no idea of what a watermark looks like,
    only more copies than marks, which is the ordinary case for a picture
    several channels reposted.
    """
    usable = [s for s in signatures if len(s) == SIGNATURE_SIDE * SIGNATURE_SIDE]
    if not usable:
        return ()
    stacked = np.asarray(usable, dtype=np.float32)
    return tuple(int(value) for value in np.median(stacked, axis=0))


def signature_deviation(
    signature: Sequence[int], consensus: Sequence[int]
) -> float:
    """Mean absolute difference per pixel, or infinity when incomparable."""
    if (
        len(signature) != SIGNATURE_SIDE * SIGNATURE_SIDE
        or len(consensus) != SIGNATURE_SIDE * SIGNATURE_SIDE
    ):
        return float("inf")
    first = np.asarray(signature, dtype=np.float32)
    second = np.asarray(consensus, dtype=np.float32)
    return float(np.abs(first - second).mean())
