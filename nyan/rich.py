"""Builders for Telegram rich messages (Bot API 10.1+).

A rich message is a tree of typed blocks rather than a markup string, which
removes the two failure modes of the HTML/Markdown path: nothing here has to
escape channel titles or news text, and nothing depends on where newlines
land, because layout is structural (`divider()` is an object, not a blank
line). The only serialization step is `json.dumps` at the transport boundary.

Reference: https://core.telegram.org/bots/api#rich-messages
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

# A RichText is a plain string, one of the inline objects built below, or a
# list mixing both. The API accepts all three wherever RichText is expected.
RichText = Union[str, Dict[str, Any], List[Any]]
Block = Dict[str, Any]


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


def join(parts: Sequence[Optional[RichText]], separator: RichText = " · ") -> RichText:
    """Concatenate inline parts, dropping empties so separators never double up."""
    result: List[Any] = []
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
    assert 1 <= size <= 6, "Heading size must be 1-6, got {}".format(size)
    return {"type": "heading", "text": text, "size": size}


def paragraph(text: RichText) -> Block:
    return {"type": "paragraph", "text": text}


def footer(text: RichText) -> Block:
    """Trailing attribution line; clients render it smaller and dimmer."""
    return {"type": "footer", "text": text}


def divider() -> Block:
    return {"type": "divider"}


def blockquote(*blocks: Block, credit: Optional[RichText] = None) -> Block:
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
    text: Optional[RichText] = None, credit: Optional[RichText] = None
) -> Optional[Dict[str, Any]]:
    if not text and not credit:
        return None
    caption: Dict[str, Any] = {"text": text if text else ""}
    if credit:
        caption["credit"] = credit
    return caption


def _media_block(
    block_type: str, media_type: str, url: str, caption: Optional[Dict[str, Any]]
) -> Block:
    block: Block = {
        "type": block_type,
        block_type: {"type": media_type, "media": fix_media_url(url)},
    }
    if caption:
        block["caption"] = caption
    return block


def photo(
    url: str, text: Optional[RichText] = None, credit: Optional[RichText] = None
) -> Block:
    return _media_block("photo", "photo", url, _caption(text, credit))


def video(
    url: str, text: Optional[RichText] = None, credit: Optional[RichText] = None
) -> Block:
    return _media_block("video", "video", url, _caption(text, credit))


def animation(
    url: str, text: Optional[RichText] = None, credit: Optional[RichText] = None
) -> Block:
    return _media_block("animation", "animation", url, _caption(text, credit))


def slideshow(
    *blocks: Block, text: Optional[RichText] = None, credit: Optional[RichText] = None
) -> Block:
    """Swipeable media group. Preferred over a collage for news: nothing is
    cropped into a grid and the post stays short regardless of photo count."""
    block: Block = {"type": "slideshow", "blocks": list(blocks)}
    caption = _caption(text, credit)
    if caption:
        block["caption"] = caption
    return block


@dataclass
class RenderedPost:
    """A post ready to send, in whichever format the renderer produced.

    Carrying both shapes in one object keeps the format choice out of the
    daemon: it renders and sends, and only the client cares which API method
    a post needs.
    """

    blocks: Optional[List[Block]] = None
    text: Optional[str] = None
    photos: Sequence[str] = field(default_factory=tuple)
    videos: Sequence[str] = field(default_factory=tuple)
    animations: Sequence[str] = field(default_factory=tuple)

    @property
    def is_rich(self) -> bool:
        return self.blocks is not None

    @property
    def has_media(self) -> bool:
        return bool(self.photos or self.videos or self.animations)


def count_blocks(blocks: Sequence[Block]) -> int:
    """Total blocks including nested ones, for checking against MAX_BLOCKS."""
    return sum(
        1
        + count_blocks(block.get("blocks", ()))
        + sum(count_blocks(item.get("blocks", ())) for item in block.get("items", ()))
        for block in blocks
    )
