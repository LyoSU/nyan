"""The model's post, parsed into something the renderer can trust.

This module is the trust boundary. Everything on the far side of it is text a
language model produced: block types that do not exist, a `list` whose items are
dicts, a quote with no author, forty blocks instead of five. None of that may
reach `nyan.rich`, because a malformed payload is a 400 from Telegram and a
missing post rather than a visible defect.

The shape of a post is the model's decision — news does not come in one shape,
and a fixed template turns a one-line story into a padded one and a complex one
into a wall. So the checks here are deliberately about honesty rather than form:
a quotation must name who said it, a post must open with prose rather than
bullets, and nothing may be so long that it stops being a post. Anything the
model chooses within those bounds is passed through.

Parsing drops rather than raises. Every unusable piece is discarded and logged,
and what survives is guaranteed renderable: the caller gets a `Summary` whose
blocks are known types with non-empty text, or a falsy `Summary` that makes the
renderer fall back to quoting one channel directly.
"""

import logging
from dataclasses import dataclass, field
from typing import Any


TEXT = "text"
LIST = "list"
QUOTE = "quote"
HIDDEN = "hidden"
DISPUTED = "disputed"
SUBHEADING = "subheading"

# Runaway protection, not editorial guidance: the prompt asks for at most six
# blocks, and these caps only stop a derailed answer from becoming a wall.
MAX_BLOCKS = 10
MAX_LIST_ITEMS = 6
MAX_TEXT_LENGTH = 1000
MAX_SUMMARY_LENGTH = 100
MAX_AUTHOR_LENGTH = 120
MAX_HEADLINE_LENGTH = 200

# Per-type limits for the kinds that stop working when repeated. Four quote
# cards in a row invert the post's hierarchy — the thing we moved away from —
# and two "sources disagree" notes mean the disagreement was not summarized.
# Types absent from here are limited only by MAX_BLOCKS.
_TYPE_LIMITS = {QUOTE: 2, HIDDEN: 2, SUBHEADING: 2, DISPUTED: 1}

# A post has to start by saying what happened. A quote lead is legitimate
# journalism; opening with bullets, a subheading or a caveat is not.
_OPENING_TYPES = (TEXT, QUOTE)


@dataclass
class SummaryBlock:
    """One renderable piece of the post.

    A single class with optional fields rather than one per kind: the shapes
    differ by a field or two, and the renderer switches on the kind anyway.
    """

    type: str
    text: str = ""
    items: list[str] = field(default_factory=list)
    author: str = ""
    summary: str = ""

    def asdict(self) -> dict[str, Any]:
        record: dict[str, Any] = {"type": self.type}
        for key in ("text", "items", "author", "summary"):
            value = getattr(self, key)
            if value:
                record[key] = value
        return record

    @classmethod
    def fromdict(cls, record: dict[str, Any]) -> "SummaryBlock":
        return cls(
            type=str(record.get("type", "")),
            text=str(record.get("text", "")),
            items=[str(item) for item in record.get("items", [])],
            author=str(record.get("author", "")),
            summary=str(record.get("summary", "")),
        )


@dataclass
class Summary:
    """A whole post as the model wrote it, after sanitation."""

    headline: str = ""
    blocks: list[SummaryBlock] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Truthy only when there is a story to render.

        A headline alone is not enough — it would leave a title with nothing
        under it — so the renderer treats a falsy summary as "fall back".
        """
        return bool(self.blocks)

    def asdict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "blocks": [block.asdict() for block in self.blocks],
        }

    @classmethod
    def fromdict(cls, record: dict[str, Any]) -> "Summary":
        """Rebuild a stored summary. Storage is ours, so no sanitation."""
        return cls(
            headline=str(record.get("headline", "")),
            blocks=[SummaryBlock.fromdict(b) for b in record.get("blocks", [])],
        )


def parse_summary(raw: Any, context: str = "") -> Summary:
    """A `Summary` built from whatever the model returned.

    Never raises: unusable input yields an empty summary, and unusable parts of
    usable input are dropped.
    """
    if not isinstance(raw, dict):
        logging.warning("Summary for '%s' is not an object: %r", context, type(raw))
        return Summary()

    raw_blocks = raw.get("blocks")
    if not isinstance(raw_blocks, list):
        logging.warning("Summary for '%s' carries no block list", context)
        raw_blocks = []

    blocks: list[SummaryBlock] = []
    counts: dict[str, int] = {}
    for raw_block in raw_blocks[:MAX_BLOCKS]:
        block = _parse_block(raw_block, context)
        if block is None:
            continue
        counts[block.type] = counts.get(block.type, 0) + 1
        limit = _TYPE_LIMITS.get(block.type)
        if limit is not None and counts[block.type] > limit:
            logging.info(
                "Dropping a %s block beyond the %d allowed for '%s'",
                block.type,
                limit,
                context,
            )
            continue
        blocks.append(block)

    # Earlier fixed-shape versions of the prompt returned this as its own
    # field. Accepting both shapes costs one branch and means a model that
    # answers the old way still produces a complete post.
    disputed = _clean(raw.get(DISPUTED), MAX_TEXT_LENGTH)
    if disputed and not any(block.type == DISPUTED for block in blocks):
        blocks.append(SummaryBlock(type=DISPUTED, text=disputed))

    while blocks and blocks[0].type not in _OPENING_TYPES:
        logging.info(
            "Dropping a leading %s block for '%s': a post has to open with prose",
            blocks[0].type,
            context,
        )
        blocks.pop(0)

    return Summary(
        headline=_clean(raw.get("headline"), MAX_HEADLINE_LENGTH),
        blocks=blocks,
    )


def _parse_block(raw: Any, context: str) -> SummaryBlock | None:
    if not isinstance(raw, dict):
        logging.info("Skipping a non-object block for '%s': %r", context, type(raw))
        return None

    block_type = raw.get("type")

    if block_type in (TEXT, DISPUTED, SUBHEADING):
        text = _clean(raw.get("text"), MAX_TEXT_LENGTH)
        return SummaryBlock(type=block_type, text=text) if text else None

    if block_type == LIST:
        raw_items = raw.get("items")
        if not isinstance(raw_items, list):
            return None
        items = [_clean(item, MAX_TEXT_LENGTH) for item in raw_items[:MAX_LIST_ITEMS]]
        items = [item for item in items if item]
        # A one-item list is a paragraph wearing a bullet.
        if len(items) < 2:
            return SummaryBlock(type=TEXT, text=items[0]) if items else None
        return SummaryBlock(type=LIST, items=items)

    if block_type == QUOTE:
        text = _clean(raw.get("text"), MAX_TEXT_LENGTH)
        author = _clean(raw.get("author"), MAX_AUTHOR_LENGTH)
        # An unattributed quotation is the exact failure this feature has to
        # avoid: words in quote marks that nobody is on record saying.
        if not text or not author:
            logging.info("Dropping an unattributed quote for '%s'", context)
            return None
        return SummaryBlock(type=QUOTE, text=text, author=author)

    if block_type == HIDDEN:
        text = _clean(raw.get("text"), MAX_TEXT_LENGTH)
        summary = _clean(raw.get("summary"), MAX_SUMMARY_LENGTH)
        if not text:
            return None
        # Without a summary the reader is asked to tap on nothing in
        # particular, so the content is better shown than hidden.
        if not summary:
            return SummaryBlock(type=TEXT, text=text)
        return SummaryBlock(type=HIDDEN, text=text, summary=summary)

    logging.info("Skipping an unknown block type %r for '%s'", block_type, context)
    return None


def _clean(value: Any, max_length: int) -> str:
    """A trimmed single-spaced string, or "" for anything unusable."""
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    return text[:max_length].strip()
