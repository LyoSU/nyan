"""Builders for Telegram rich messages (Bot API 10.1+).

A rich message is a tree of typed blocks rather than a markup string, which
removes the two failure modes of the HTML/Markdown path: nothing here has to
escape channel titles or news text, and nothing depends on where newlines
land, because layout is structural (`divider()` is an object, not a blank
line). The only serialization step is `json.dumps` at the transport boundary.

Reference: https://core.telegram.org/bots/api#rich-messages
"""

from dataclasses import dataclass, field
from typing import Any, Union
from collections.abc import Sequence

from nyan.media import MediaItem

# A RichText is a plain string, one of the inline objects built below, or a
# list mixing both. The API accepts all three wherever RichText is expected.
RichText = Union[str, dict[str, Any], list[Any]]  # noqa: UP007
Block = dict[str, Any]


# Telegram limits, kept here so callers can budget against them.
# https://core.telegram.org/bots/api#rich-message-limits
MAX_TEXT_LENGTH = 32768
MAX_BLOCKS = 500
MAX_MEDIA = 50


# Telegram's newer telesco.pe media host is not reachable from the Bot API
# fetcher, while the legacy CDN domain serves the same files.
# See issue #31 for the proper long-term fix.
_LEGACY_MEDIA_HOST = ("telesco.pe", "cdn-telegram.org")


def fix_media_url(url: str) -> str:
    """Rewrite media URLs Telegram itself refuses to fetch."""
    old_host, new_host = _LEGACY_MEDIA_HOST
    if old_host in url:
        return url.replace(old_host, new_host)
    return url


# ---------------------------------------------------------------- inline text


def bold(text: RichText) -> RichText:
    return {"type": "bold", "text": text}


def italic(text: RichText) -> RichText:
    return {"type": "italic", "text": text}


def marked(text: RichText) -> RichText:
    return {"type": "marked", "text": text}


def link(text: RichText, url: str) -> RichText:
    return {"type": "url", "text": text, "url": url}


def date_time(text: str, unix_time: int, fmt: str = "t") -> RichText:
    """A timestamp rendered in the reader's own locale and timezone.

    `text` is the server-side rendering, shown verbatim by clients too old to
    know the entity, so it must always be a sensible standalone value.
    Formats: "t" short time, "T" long time, "d"/"D" date, "w" weekday,
    "r" relative. "r" is deliberately unused here: a channel post stays in
    the archive, where "a week ago" is worse than a clock time.
    """
    return {
        "type": "date_time",
        "text": text,
        "unix_time": unix_time,
        "date_time_format": fmt,
    }


def join(parts: Sequence[RichText | None], separator: RichText = " · ") -> RichText:
    """Concatenate inline parts, dropping empties so separators never double up."""
    result: list[Any] = []
    for part in parts:
        if part is None or part == "" or part == []:
            continue
        if result:
            result.append(separator)
        result.append(part)
    return result


# --------------------------------------------------------------------- blocks


def heading(text: RichText, size: int = 3) -> Block:
    """A section heading. `size` is 1-6 where 1 is the largest."""
    assert 1 <= size <= 6, f"Heading size must be 1-6, got {size}"
    return {"type": "heading", "text": text, "size": size}


def paragraph(text: RichText) -> Block:
    return {"type": "paragraph", "text": text}


def footer(text: RichText) -> Block:
    """Trailing attribution line; clients render it smaller and dimmer."""
    return {"type": "footer", "text": text}


def divider() -> Block:
    return {"type": "divider"}


def blockquote(*blocks: Block, credit: RichText | None = None) -> Block:
    """A quotation. `credit` maps to <cite>, i.e. who said it."""
    block: Block = {"type": "blockquote", "blocks": list(blocks)}
    if credit:
        block["credit"] = credit
    return block


def details(summary: RichText, *blocks: Block, is_open: bool = False) -> Block:
    """A collapsible block. `summary` stays visible, so make it informative."""
    block: Block = {"type": "details", "summary": summary, "blocks": list(blocks)}
    if is_open:
        block["is_open"] = True
    return block


def bullet_list(*items: Sequence[Block]) -> Block:
    """An unordered list; every item is itself a sequence of blocks."""
    return {"type": "list", "items": [{"blocks": list(item)} for item in items]}


def _caption(
    text: RichText | None = None, credit: RichText | None = None
) -> dict[str, Any] | None:
    if not text and not credit:
        return None
    caption: dict[str, Any] = {"text": text if text else ""}
    if credit:
        caption["credit"] = credit
    return caption


def _media_block(
    block_type: str, media_type: str, url: str, caption: dict[str, Any] | None
) -> Block:
    block: Block = {
        "type": block_type,
        block_type: {"type": media_type, "media": fix_media_url(url)},
    }
    if caption:
        block["caption"] = caption
    return block


def photo(
    url: str, text: RichText | None = None, credit: RichText | None = None
) -> Block:
    return _media_block("photo", "photo", url, _caption(text, credit))


def video(
    url: str, text: RichText | None = None, credit: RichText | None = None
) -> Block:
    return _media_block("video", "video", url, _caption(text, credit))


def animation(
    url: str, text: RichText | None = None, credit: RichText | None = None
) -> Block:
    return _media_block("animation", "animation", url, _caption(text, credit))


def slideshow(
    *blocks: Block, text: RichText | None = None, credit: RichText | None = None
) -> Block:
    """Swipeable media group. Preferred over a collage for news: nothing is
    cropped into a grid and the post stays short regardless of photo count."""
    block: Block = {"type": "slideshow", "blocks": list(blocks)}
    caption = _caption(text, credit)
    if caption:
        block["caption"] = caption
    return block


def media_payloads(blocks: Sequence[Block]) -> list[dict[str, Any]]:
    """Every media payload in a block tree, in the order it will be sent.

    The payloads themselves, not copies: callers rewrite them in place to swap a
    URL for a file_id. Order is what pairs them with Telegram's answer, which
    comes back in the order the attachments went out.
    """
    found: list[dict[str, Any]] = []
    for block in blocks:
        for value in block.values():
            if isinstance(value, dict) and isinstance(value.get("media"), str):
                found.append(value)
            elif isinstance(value, list):
                found.extend(
                    media_payloads([item for item in value if isinstance(item, dict)])
                )
            elif isinstance(value, dict):
                found.extend(media_payloads([value]))
    return found


@dataclass
class RenderedPost:
    """A post ready to send, in whichever format the renderer produced.

    Carrying both shapes in one object keeps the format choice out of the
    daemon: it renders and sends, and only the client cares which API method
    a post needs.

    `media` is one ordered list rather than a field per type. Photos and videos
    used to be separate, and every consumer then had to decide which wins — the
    rich renderer preferred video, the legacy client preferred photos, so one
    cluster showed different attachments depending on a config value.
    """

    blocks: list[Block] | None = None
    text: str | None = None
    media: Sequence[MediaItem] = field(default_factory=tuple)

    @property
    def is_rich(self) -> bool:
        return self.blocks is not None

    @property
    def has_media(self) -> bool:
        return bool(self.media)


def count_blocks(blocks: Sequence[Block]) -> int:
    """Total blocks including nested ones, for checking against MAX_BLOCKS."""
    return sum(
        1
        + count_blocks(block.get("blocks", ()))
        + sum(count_blocks(item.get("blocks", ())) for item in block.get("items", ()))
        for block in blocks
    )
