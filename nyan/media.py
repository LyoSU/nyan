"""What a post can attach, and what Telegram gave back for it.

Two types, deliberately separate. `MediaItem` is a candidate: a URL from the
crawl, plus whatever the annotator knows about its content, chosen by
`Cluster.media`. `SentMedia` is a fact: what Telegram actually stored when the
post went out, which is the only handle that stays valid.

They live in their own module because the three places that need them —
selection in `clusters`, block building in `rich`, transport in `client` — form
a chain of imports that a shared type in any one of them would close into a
cycle.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nyan.picture import (
    DEFAULT_MAX_DISTANCE,
    is_same_picture,
    median_signature,
    signature_deviation,
)
from nyan.util import Serializable


# The kinds of attachment a post can carry. The strings are Bot API media types,
# used verbatim in sendMediaGroup and in rich blocks.
MEDIA_PHOTO = "photo"
MEDIA_VIDEO = "video"
MEDIA_ANIMATION = "animation"

MEDIA_TYPES = (MEDIA_PHOTO, MEDIA_VIDEO, MEDIA_ANIMATION)

# Above this cosine two pictures are one. Kept where it has always been — the
# same encoder, the same meaning — but no longer the only thing deciding, since
# a stamped repost scores well below it; see `_is_duplicate`.
DUPLICATE_IMAGE_SIMILARITY = 0.92

# How many channels must have carried a picture before it can lead a post. One
# channel's picture is, more often than not, its own branding rather than the
# event: a logo, a stock illustration, a card it generates for every post.
# Nothing else in the cluster confirms it, so nothing shows it.
MIN_LEAD_CHANNELS = 2

# How answerable the source is for what it published, as the position of its
# tier in `nyan.channels.GROUP_ORDER`. Lower is more accountable: an official
# body, then media and named authors, then anonymous channels.
AUTHORITY_OFFICIAL = 0
AUTHORITY_MEDIA = 1
AUTHORITY_ANONYMOUS = 2

# The least accountable source whose picture may lead a post with nothing
# corroborating it. A body publishing its own announcement will never have a
# second channel behind its photograph — everyone else is quoting it — so
# demanding consensus there leaves the post with no picture at all. An
# anonymous channel is the opposite case: nothing confirms the picture and
# nobody is answerable for it, which is exactly where an unconfirmed picture is
# most often the channel's own branding.
MAX_LONE_AUTHORITY = AUTHORITY_MEDIA

# How much of the ranking is consensus and how much is variety, when filling
# the slots after the lead. At 1.0 the slideshow is ordered purely by how many
# channels agreed, which fills it with several angles of one scene; at 0.0 it
# is ordered purely by being unlike what is already there, which promotes
# whatever nobody else posted. See `_rank_by_relevance_and_variety`.
VARIETY_LAMBDA = 0.7

# How far two clips' reported lengths may differ and still be one clip. A
# re-encode shifts the figure by a second at most; anything beyond that is
# different footage, whatever the posters look like.
DURATION_TOLERANCE = 1

# How far a copy may sit from what the copies agree the picture is before it
# counts as fully marked. In grey levels averaged over the thumbnail: a badge
# in a corner moves a couple of levels, a bar across the frame tens of them.
FULLY_MARKED_DEPARTURE = 40.0

# How the two readings that pick between copies are weighed. Both matter and
# neither may win outright: ranking on the mark first makes a clean 240px
# thumbnail beat a lightly stamped full-size photograph, and ranking on quality
# first puts the watermark back on screen. The mark weighs more because it is
# somebody else's brand on our post, while a softer rendition is only softer.
CLEANLINESS_WEIGHT = 0.6
QUALITY_WEIGHT = 0.4

# How close to the story a document must be for its unconfirmed picture to be
# shown. Documents enter a cluster at a cosine of about 0.9, so this only
# excludes the ones that arrived at the very edge of the threshold — the
# neighbouring story, whose picture is of something else.
MIN_DOCUMENT_RELEVANCE = 0.9


@dataclass(frozen=True)
class MediaItem:
    """One attachment a cluster can show, with what is known about its content.

    `embedding` is the vector of what the reader will see: the photo itself, or
    for a video the still Telegram renders for it. Both come out of the same
    encoder, so they are comparable with each other — a clip and another
    channel's frame of the same moment are one thing, and only one of them takes
    a slot. Absent when the annotator could not fetch the image, and on documents
    stored before any of this; see `nyan.clusters._deduplicate_media`.
    """

    type: str
    url: str
    embedding: tuple[float, ...] | None = None
    #: Which channel's copy this is, and the post it came from. A carousel
    #: mixes channels, so a single credit for the whole post cannot say which
    #: frame belongs to whom — the byline has to travel with the frame.
    channel_title: str = ""
    source_url: str = ""


@dataclass(frozen=True)
class MediaCandidate:
    """One attachment a channel offered, with what decides whether it is shown.

    Every attachment of every document becomes a candidate, not one per channel
    as before. Which of a channel's pictures takes its slot cannot be settled
    per channel, because it depends on what the rest of the cluster carried —
    and the count of channels behind a picture, which is the strongest signal
    here, is only visible once they are all in one pile.

    `relevance` is how close this candidate's document sits to the story, by
    the text embedding that put it in the cluster. It is what separates a
    picture nobody else posted because they had nothing, from one nobody else
    posted because it belongs to the neighbouring story the clustering pulled
    in at its threshold.
    """

    type: str
    url: str
    channel_id: str
    pub_time: int
    #: How the channel is named to a reader, and the post this copy is in. Both
    #: ride along so the chosen copy can be credited where it is shown.
    channel_title: str = ""
    source_url: str = ""
    embedding: tuple[float, ...] | None = None
    hashes: tuple[str, ...] = ()
    relevance: float = 1.0
    authority: int = AUTHORITY_ANONYMOUS
    #: How clean this rendition is, from `nyan.picture.picture_quality`. Only
    #: ever compared between copies of one picture, where it answers which of
    #: them the reader should get: the photograph itself, or the same
    #: photograph as a third of somebody's generated card.
    quality: float = 0.0
    #: A tiny greyscale print of the picture, from
    #: `nyan.picture.thumbnail_signature`. Compared only within a group, where
    #: the copies agree on what the photograph is and disagree exactly where
    #: one of them was stamped.
    signature: tuple[int, ...] = ()
    #: How long the clip runs, in seconds, or 0 when unknown. Videos only, and
    #: only ever used to tell two clips apart — see `_is_duplicate`.
    duration: int = 0


@dataclass
class MediaGroup:
    """Every copy of one picture, and the channels that posted it."""

    members: list[MediaCandidate] = field(default_factory=list)

    @property
    def channels(self) -> set[str]:
        return {member.channel_id for member in self.members}

    @property
    def consensus(self) -> int:
        """How many channels independently chose this picture for this story.

        Channels rather than copies: a channel posting the same photograph in
        two of its own posts is one source saying one thing twice.
        """
        return len(self.channels)

    @property
    def representative(self) -> MediaCandidate:
        """The copy to actually show: unmarked and a good rendition, at once.

        The copies of one photograph differ in ways a reader sees. One channel
        posts the picture; another posts it under its own bar, as a third of a
        generated card, re-encoded and upscaled. They are one picture to the
        hash and two different things to look at.

        Time breaks an exact tie, and does so meaningfully: whoever posted
        first is where the picture came from, so its copy has not yet collected
        anybody else's furniture. It only breaks ties, because a source that
        posts fast is not thereby the one that posts cleanly — but on documents
        with nothing recorded at all, which is every document stored before
        this, it is the only thing left to sort by.
        """
        consensus = median_signature([m.signature for m in self.members if m.signature])
        return max(
            self.members,
            # One score rather than a tuple: cleanliness and quality have to be
            # weighed against each other, and a tuple compares
            # lexicographically — the first element decides and the second is
            # never reached in practice. `max` keeps the first of equal
            # candidates, so the order is the cluster's own document order and
            # survives a trip through storage; sorting by url instead would
            # pick whichever copy happens to sort last, which is a property of
            # Telegram's CDN filenames and of nothing else.
            key=lambda member: (
                self._copy_score(member, consensus),
                -member.pub_time,
            ),
        )

    def _copy_score(self, member: MediaCandidate, consensus: Sequence[int]) -> float:
        """How well a copy serves a reader: unmarked, and a good rendition.

        Both at once, because either alone picks badly. The cleanest copy of a
        photograph is sometimes a 240px thumbnail somebody re-uploaded, and the
        largest copy is often the one with a channel's bar across it.
        """
        departure = self._departure(member, consensus)
        cleanliness = 1.0 - min(departure / FULLY_MARKED_DEPARTURE, 1.0)
        return CLEANLINESS_WEIGHT * cleanliness + QUALITY_WEIGHT * member.quality

    @staticmethod
    def _departure(member: MediaCandidate, consensus: Sequence[int]) -> float:
        """How far this copy strays from what the copies agree the picture is.

        This is the watermark measure, and it needs no idea of what a watermark
        looks like. A mark is in one copy and not in the others, so at its
        pixels the median still holds the photograph and the marked copy is the
        one that departs. Copies with nothing to compare — one lone member, or
        documents stored before signatures existed — depart by zero, which
        leaves the ranking to quality and time.
        """
        if not consensus or not member.signature:
            return 0.0
        return signature_deviation(member.signature, consensus)

    @property
    def relevance(self) -> float:
        return max(member.relevance for member in self.members)

    @property
    def authority(self) -> int:
        """The most accountable source behind this picture.

        The best of them, not the average: one official body publishing a
        photograph is what makes it accountable, however many anonymous
        channels reposted it afterwards.
        """
        return min(member.authority for member in self.members)

    @property
    def can_lead(self) -> bool:
        """Whether this picture may open a post.

        Either the coverage agrees on it, or somebody answerable published it.
        """
        return (
            self.consensus >= MIN_LEAD_CHANNELS or self.authority <= MAX_LONE_AUTHORITY
        )


@dataclass
class SentMedia(Serializable):
    """An attachment as it exists inside Telegram, after the post was sent.

    `file_id` is what makes this worth storing: it is valid for our bot
    indefinitely, while the CDN URL the crawler found rots — which is what
    `nyan.rich.fix_media_url` has been papering over. Once a file_id is known,
    every later edit references the file Telegram already has instead of asking
    it to fetch a URL again.

    `message_id` is the attachment's own message. A media group is not one
    message but one per attachment, so editing the third photo of a slideshow
    needs the third message's id, not the group's first.
    """

    type: str
    file_id: str
    url: str = ""
    message_id: int = 0


def _unit_vector(embedding: Sequence[float] | None) -> NDArray[np.float32] | None:
    if not embedding:
        return None
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return None
    return vector / norm


def _similarity(first: MediaCandidate, second: MediaCandidate) -> float:
    """How alike two candidates look, or 0 when that cannot be known.

    Zero rather than a guess: a missing embedding — an older document, a video
    whose still never loaded, a fetch that failed — means the pair is only
    comparable by hash, and pretending otherwise would either merge two
    different pictures or push a good one out of the slideshow.
    """
    one, other = _unit_vector(first.embedding), _unit_vector(second.embedding)
    if one is None or other is None or one.shape != other.shape:
        return 0.0
    return float(one @ other)


def _is_duplicate(
    first: MediaCandidate,
    second: MediaCandidate,
    max_distance: int = DEFAULT_MAX_DISTANCE,
) -> bool:
    """Whether two candidates are the same picture, by any of three readings.

    The url settles the trivial case. The perceptual hash settles the case the
    embedding cannot: a repost under another channel's watermark, or a
    photograph inset into a channel's own card, both of which SigLIP scores
    around 0.86 — comfortably below the duplicate threshold, which is how they
    used to take a second slot. The embedding settles what the hash cannot: a
    re-crop or a re-colouring severe enough to move the hash, which is still
    plainly the same photograph.
    """
    if first.url == second.url:
        return True
    # Length settles video before anything else looks at the poster. Telegram
    # re-encodes what it is given, so one channel's poster of a clip can be
    # another channel's poster of different footage from the same scene — same
    # place, same light, same framing, and a hash that cannot tell them apart.
    # Only when both lengths are known: older documents carry none, and no
    # length is not a mismatch.
    if (
        first.type == MEDIA_VIDEO
        and second.type == MEDIA_VIDEO
        and first.duration
        and second.duration
        and abs(first.duration - second.duration) > DURATION_TOLERANCE
    ):
        return False
    if (
        first.hashes
        and second.hashes
        and len(first.hashes) == len(second.hashes)
        and is_same_picture(first.hashes, second.hashes, max_distance)
    ):
        return True
    return _similarity(first, second) >= DUPLICATE_IMAGE_SIMILARITY


def group_media(
    candidates: Sequence[MediaCandidate],
    max_distance: int = DEFAULT_MAX_DISTANCE,
) -> list[MediaGroup]:
    """Candidates gathered into one group per distinct picture.

    Union-find rather than pairwise filtering, because sameness here is not
    transitive on its own: A can match B by hash and B match C by embedding
    without A and C matching directly. Comparing every pair and keeping the
    first of each match would leave that group split in two, and one
    photograph would take two of the four slots.

    Groups, and not just a deduplicated list, because the size of a group is
    the most valuable thing the cluster knows about a picture: how many
    channels independently chose it for this story. The old code computed that
    number as a side effect of skipping duplicates and threw it away.
    """
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, first in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            if find(i) == find(j):
                continue
            if _is_duplicate(first, candidates[j], max_distance):
                parent[find(j)] = find(i)

    grouped: dict[int, MediaGroup] = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(find(index), MediaGroup()).members.append(candidate)
    return list(grouped.values())


def _rank_by_relevance_and_variety(
    groups: Sequence[MediaGroup], limit: int
) -> list[MediaGroup]:
    """Groups ordered so that each one adds something the previous did not.

    Maximal marginal relevance: every slot after the first goes to the group
    that scores best on consensus *minus* how much it resembles what is already
    chosen. Ranking on consensus alone fills a slideshow with four angles of
    one scene, because a well-covered event is well covered from every angle,
    and each of those angles has its own channels behind it.

    Deliberately not a second duplicate threshold. Two frames of one scene at a
    cosine of 0.85 are not the same picture — merging them would be wrong — but
    they are alike enough that showing both wastes a slot. A threshold has to
    call that either same or different; a penalty just puts the second one
    behind anything that shows something else.
    """
    remaining = list(groups)
    strongest = max((group.consensus for group in remaining), default=1)
    chosen: list[MediaGroup] = []
    # Channels that have already had an unconfirmed picture shown. A channel
    # posting six shots of one scene is one source's gallery, not six things
    # the story is about, and without this it fills every slot. The limit is on
    # unconfirmed pictures only: two pictures that five channels each carried
    # are two things the coverage agreed on, and both belong in the post.
    spent: set[str] = set()
    while remaining and len(chosen) < limit:
        allowed = [
            group
            for group in remaining
            if group.consensus >= MIN_LEAD_CHANNELS or not (group.channels & spent)
        ]
        if not allowed:
            break
        best = max(
            allowed,
            key=lambda group: (
                VARIETY_LAMBDA * (group.consensus / strongest)
                - (1 - VARIETY_LAMBDA)
                * max(
                    (
                        _similarity(group.representative, picked.representative)
                        for picked in chosen
                    ),
                    default=0.0,
                ),
                group.consensus,
                -group.authority,
                # Footage from the scene is stronger material than a still of
                # it, and a reader tells them apart at a glance. Only a
                # tie-breaker: a photograph five channels carried still leads
                # over a clip that one did.
                group.representative.type == MEDIA_VIDEO,
                -group.representative.pub_time,
            ),
        )
        remaining.remove(best)
        chosen.append(best)
        if best.consensus < MIN_LEAD_CHANNELS:
            spent |= best.channels
    return chosen


def select_media(
    candidates: Sequence[MediaCandidate],
    limit: int,
    min_relevance: float = MIN_DOCUMENT_RELEVANCE,
    max_distance: int = DEFAULT_MAX_DISTANCE,
) -> tuple[MediaItem, ...]:
    """What the post shows, best first.

    First is not a flag but a position: the carousel opens on it and the site
    uses it as the post's cover, so leading with the right picture is a matter
    of sorting rather than of marking one.

    A post shows nothing at all unless some picture was carried by more than
    one channel. That is a stronger rule than the old ratio gate, which asked
    only whether enough documents had *any* attachment and then showed whatever
    they happened to be. Agreement on a particular photograph is what makes it
    the story's; agreement that photographs exist is not.
    """
    groups = group_media(candidates, max_distance)
    leadable = [group for group in groups if group.can_lead]
    if not leadable:
        return tuple()
    showable = leadable + [
        group
        for group in groups
        if not group.can_lead and group.relevance >= min_relevance
    ]
    ranked = _rank_by_relevance_and_variety(showable, limit)
    # The lead has to be a picture that may lead even when another outranks it
    # on variety, since variety is meaningless for the slot nothing precedes.
    if ranked and not ranked[0].can_lead:
        lead = max(
            leadable, key=lambda group: (group.consensus, -group.authority)
        )
        ranked = [lead, *[group for group in ranked if group is not lead][: limit - 1]]
    return tuple(
        MediaItem(
            type=group.representative.type,
            url=group.representative.url,
            embedding=group.representative.embedding,
            channel_title=group.representative.channel_title
            or group.representative.channel_id,
            source_url=group.representative.source_url,
        )
        for group in ranked
    )


def largest_file_id(payload: Any) -> str:
    """The file_id of the biggest rendition Telegram reports.

    A photo comes back as a list of sizes, ascending, and the largest is the one
    worth keeping: it is what a reader opens, and a thumbnail's file_id would
    quietly downgrade the post on the next edit. Anything else — video,
    animation — is a single object.
    """
    if isinstance(payload, list):
        if not payload:
            return ""
        largest = max(
            payload,
            key=lambda size: (
                size.get("file_size") or 0,
                (size.get("width") or 0) * (size.get("height") or 0),
            ),
        )
        return str(largest.get("file_id", ""))
    if isinstance(payload, dict):
        return str(payload.get("file_id", ""))
    return ""


def _walk_media(node: Any) -> list[tuple[str, Any]]:
    """Every media payload inside a Telegram message, in the order found.

    A walk rather than a fixed set of paths: the same extraction has to work for
    sendPhoto, sendVideo, sendMediaGroup and for a rich message, whose media sit
    inside a block tree. Only the keys that name a media type are followed, so a
    video's own `thumbnail` is not mistaken for a second attachment.
    """
    found: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in MEDIA_TYPES:
                found.append((key, value))
                continue
            found.extend(_walk_media(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_walk_media(value))
    return found


def extract_sent_media(result: Any) -> list[SentMedia]:
    """What Telegram says it stored, from any send or edit response.

    sendMediaGroup answers with one message per attachment and everything else
    with a single message, so both shapes are normalized to a list first.
    """
    messages = result if isinstance(result, list) else [result]
    media: list[SentMedia] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        message_id = int(message.get("message_id", 0) or 0)
        for media_type, payload in _walk_media(message):
            file_id = largest_file_id(payload)
            if not file_id:
                continue
            media.append(
                SentMedia(type=media_type, file_id=file_id, message_id=message_id)
            )
    return media


def attach_urls(media: list[SentMedia], urls: list[str]) -> list[SentMedia]:
    """Pair what was asked for with what came back, by position.

    Telegram answers in the order the attachments were sent, so position is the
    only link between a URL we chose and the file_id it became. A length
    mismatch — a rendition we failed to read, an attachment Telegram dropped —
    leaves the extra entries without a URL rather than shifting the rest onto the
    wrong file.
    """
    for item, url in zip(media, urls, strict=False):
        item.url = url
    return media
