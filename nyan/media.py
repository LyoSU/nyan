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

from dataclasses import dataclass
from typing import Any

from nyan.util import Serializable


# The kinds of attachment a post can carry. The strings are Bot API media types,
# used verbatim in sendMediaGroup and in rich blocks.
MEDIA_PHOTO = "photo"
MEDIA_VIDEO = "video"
MEDIA_ANIMATION = "animation"

MEDIA_TYPES = (MEDIA_PHOTO, MEDIA_VIDEO, MEDIA_ANIMATION)


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
