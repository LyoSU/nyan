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
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from nyan.markup import drop_links, find_links, strip_markup
from nyan.util import normalize_channel_id


TEXT = "text"
LIST = "list"
LINKS = "links"
QUOTE = "quote"
HIDDEN = "hidden"
DISPUTED = "disputed"
ATTRIBUTED = "attributed"
SUBHEADING = "subheading"

MAX_LIST_ITEMS = 6
MAX_LINKS = 12
MAX_CLAIMS = 4
# Three names is where a credit line stops being a credit line and becomes a
# second source list. The upstream diff renderer cut at three for the same reason.
MAX_CLAIM_CHANNELS = 3
MAX_TEXT_LENGTH = 1000
MAX_SUMMARY_LENGTH = 100
MAX_AUTHOR_LENGTH = 120
MAX_HEADLINE_LENGTH = 200


@dataclass(frozen=True)
class Limits:
    """How much of each thing a post of this kind may contain.

    Runaway protection, not editorial guidance: the prompts ask for far less.
    These only stop a derailed answer from becoming a wall of text, and they
    differ by kind because a digest is a genuinely different sort of post — it
    covers a whole shift under several headings, where a single story never
    needs more than one.
    """

    max_blocks: int
    # Types absent from here are limited only by max_blocks.
    per_type: dict[str, int]
    # What the post may open with. A story has to start by saying what happened;
    # a digest is a list by nature, so it may open with the list itself.
    opening: tuple[str, ...]

    def limit_for(self, block_type: str) -> int | None:
        return self.per_type.get(block_type)


# A story: four quote cards in a row invert the post's hierarchy — the thing
# this design moved away from — and two "sources disagree" notes mean the
# disagreement was never summarized. A quote lead is legitimate journalism;
# opening with bullets, a subheading or a caveat is not.
POST_LIMITS = Limits(
    max_blocks=10,
    per_type={QUOTE: 2, HIDDEN: 2, SUBHEADING: 2, DISPUTED: 1, ATTRIBUTED: 1},
    opening=(TEXT, QUOTE),
)

# A digest: one heading per topic, so headings are the structure rather than an
# exception, and there is no single story for sources to disagree about. On a
# quiet period the whole digest is one list, so it may open with one.
#
# The two attributed kinds stay in this table as runaway protection rather than
# as an offer: a digest is built from other posts of ours, so it is never given
# channel ids, and `_parse_claims` therefore drops every claim it is handed. The
# digest prompt asks for neither, and a caller that wanted them would have to
# pass `allowed_channels` for the whole period first.
DIGEST_LIMITS = Limits(
    max_blocks=24,
    per_type={QUOTE: 2, HIDDEN: 3, DISPUTED: 1, ATTRIBUTED: 1},
    opening=(TEXT, QUOTE, LINKS, SUBHEADING),
)


@dataclass
class SummaryBlock:
    """One renderable piece of the post.

    A single class with optional fields rather than one per kind: the shapes
    differ by a field or two, and the renderer switches on the kind anyway.
    """

    type: str
    text: str = ""
    items: list[str] = field(default_factory=list)
    links: list[dict[str, str]] = field(default_factory=list)
    # For `attributed` and `disputed`: a statement plus the channels that made
    # it. Whose claim it is *is* the content of these two blocks — a lone
    # channel's version is a different fact from the same version in six.
    claims: list[dict[str, Any]] = field(default_factory=list)
    author: str = ""
    summary: str = ""

    def asdict(self) -> dict[str, Any]:
        record: dict[str, Any] = {"type": self.type}
        for key in ("text", "items", "links", "claims", "author", "summary"):
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
            links=[
                {"text": str(link.get("text", "")), "url": str(link.get("url", ""))}
                for link in record.get("links", [])
            ],
            claims=[
                {
                    "text": str(claim.get("text", "")),
                    "channels": [str(name) for name in claim.get("channels", [])],
                }
                for claim in record.get("claims", [])
            ],
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

    def as_text(self) -> str:
        """The prose of the post, for a prompt that needs to read it.

        Markup, links and disclosure titles are left out: a model reading this
        needs the facts, not the typography. Leaving the delimiters in would also
        teach the next model to write more of them, in places where they mean
        something else — in a digest headline a `**` span picks the link anchor.
        """
        parts: list[str] = []
        for block in self.blocks:
            if block.type in (TEXT, SUBHEADING, HIDDEN):
                parts.append(block.text)
                parts.extend(link["text"] for link in block.links)
            elif block.type == LIST:
                parts.extend(block.items)
            elif block.type == LINKS:
                parts.extend(link["text"] for link in block.links)
            elif block.type in (ATTRIBUTED, DISPUTED):
                # Whichever shape arrived: `text` is the old unattributed line.
                parts.append(block.text)
                parts.extend(claim["text"] for claim in block.claims)
            elif block.type == QUOTE:
                parts.append(f"{block.author}: {block.text}")
        return strip_markup(" ".join(part for part in parts if part))

    @classmethod
    def fromdict(cls, record: dict[str, Any]) -> "Summary":
        """Rebuild a stored summary. Storage is ours, so no sanitation."""
        return cls(
            headline=str(record.get("headline", "")),
            blocks=[SummaryBlock.fromdict(b) for b in record.get("blocks", [])],
        )


def parse_summary(
    raw: Any,
    context: str = "",
    allowed_urls: Collection[str] = (),
    limits: Limits = POST_LIMITS,
    allowed_channels: Collection[str] = (),
) -> Summary:
    """A `Summary` built from whatever the model returned.

    `allowed_urls` is the set of links the model was given; anything else it
    puts in a `links` block is a URL it made up, and a made-up link in a digest
    sends the reader to a post that does not exist. Empty by default, which
    means a `links` block cannot survive unless the caller is a digest.

    `allowed_channels` is the same guard for attribution: the channel ids the
    model was shown. An invented id would credit a claim to a channel that never
    made it, which is worse than leaving the claim unattributed — so it is
    dropped. Empty means nothing can be attributed at all.

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

    known_urls = frozenset(allowed_urls)
    # Normalized id to the form the documents use, so a claim is stored under
    # the id the renderer and the site look it up by. The model copies these out
    # of the prompt and gets the copy slightly wrong — "@Suspilne" for
    # "suspilne" — often enough that matching them verbatim would drop real
    # attributions, and storing what it wrote would credit nobody findable.
    known_channels = {
        normalize_channel_id(channel): channel for channel in allowed_channels
    }
    blocks: list[SummaryBlock] = []
    counts: dict[str, int] = {}
    for raw_block in raw_blocks[: limits.max_blocks]:
        block = _parse_block(raw_block, context, known_urls, known_channels)
        if block is None:
            continue
        counts[block.type] = counts.get(block.type, 0) + 1
        limit = limits.limit_for(block.type)
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

    while blocks and blocks[0].type not in limits.opening:
        logging.info(
            "Dropping a leading %s block for '%s': not something a post opens with",
            blocks[0].type,
            context,
        )
        blocks.pop(0)

    return Summary(
        headline=_clean(raw.get("headline"), MAX_HEADLINE_LENGTH),
        blocks=blocks,
    )


def _parse_block(
    raw: Any,
    context: str,
    known_urls: frozenset[str],
    known_channels: Mapping[str, str],
) -> SummaryBlock | None:
    if not isinstance(raw, dict):
        logging.info("Skipping a non-object block for '%s': %r", context, type(raw))
        return None

    block_type = raw.get("type")

    if block_type == LINKS:
        return _parse_links(raw, context, known_urls)

    if block_type in (DISPUTED, ATTRIBUTED):
        return _parse_claims(raw, block_type, context, known_channels)

    if block_type == SUBHEADING:
        text = _clean(raw.get("text"), MAX_TEXT_LENGTH)
        return SummaryBlock(type=block_type, text=text) if text else None

    if block_type == TEXT:
        # A paragraph may link the posts it draws on — the digest lede does —
        # under the same guard as a list: a link the model was never given is
        # reduced to its words. For a story post nothing is allowed, so its
        # prose can carry no links at all. The links kept are also recorded on
        # the block, so a digest can tell which posts its lede already covers.
        written = _clean(raw.get("text"), MAX_TEXT_LENGTH)
        text = drop_links(written, known_urls)
        if not text:
            return None
        links = find_links(text)
        for invented in find_links(written):
            if invented not in links:
                logging.info(
                    "Dropping an invented link %r from prose for '%s'",
                    invented["url"],
                    context,
                )
        return SummaryBlock(type=TEXT, text=text, links=links)

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
        # A hidden block may fold away headlines as well as prose: in a digest
        # that is how the tail of a busy shift stays in the post without
        # stretching it over three screens. The same URL guard applies, since a
        # folded invented link is still an invented link.
        folded = _parse_links(raw, context, known_urls)
        links = folded.links if folded else []
        if not text and not links:
            return None
        # Without a summary the reader is asked to tap on nothing in
        # particular, so the content is better shown than hidden.
        if not summary:
            if text:
                return SummaryBlock(type=TEXT, text=text)
            return SummaryBlock(type=LINKS, links=links)
        return SummaryBlock(type=HIDDEN, text=text, summary=summary, links=links)

    logging.info("Skipping an unknown block type %r for '%s'", block_type, context)
    return None


def _parse_claims(
    raw: dict[str, Any],
    block_type: str,
    context: str,
    known_channels: Mapping[str, str],
) -> SummaryBlock | None:
    """Statements with the channels behind them.

    Both kinds hold the same shape because both are deviations from the prose
    above them, and they differ only in what kind: `attributed` adds something
    the other sources do not carry, `disputed` contradicts what the post says.
    Neither ever restates the consensus — that is what the prose is — so a
    single claim is a complete block. One channel against everyone else is the
    most informative case there is, not a malformed dispute.

    An unattributed claim is dropped: without a name the reader cannot tell a
    lone channel's addition from what everyone reported, which is the whole
    reason these blocks exist.
    """
    raw_claims = raw.get("claims")
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    if isinstance(raw_claims, list):
        for raw_claim in raw_claims[:MAX_CLAIMS]:
            if not isinstance(raw_claim, dict):
                continue
            text = _clean(raw_claim.get("text"), MAX_TEXT_LENGTH)
            if not text:
                continue
            raw_names = raw_claim.get("channels")
            if not isinstance(raw_names, list):
                raw_names = []
            names: list[str] = []
            for raw_name in raw_names:
                canonical = known_channels.get(
                    normalize_channel_id(_clean(raw_name, MAX_AUTHOR_LENGTH))
                )
                if canonical is None:
                    logging.info(
                        "Dropping %r from a %s claim for '%s': not a source here",
                        raw_name,
                        block_type,
                        context,
                    )
                    continue
                if canonical not in names:
                    names.append(canonical)
            if not names:
                logging.info(
                    "Dropping an unattributed %s claim for '%s': %r",
                    block_type,
                    context,
                    text,
                )
                continue
            names = names[:MAX_CLAIM_CHANNELS]
            # The same channels credited twice in one block reads as two
            # independent reports of the thing they said once. Keyed on the
            # names that will actually be printed, so two claims that differ
            # only in a name past the cut still count as one.
            key = " ".join(sorted(names))
            if key in seen:
                continue
            seen.add(key)
            claims.append({"text": text, "channels": names})

    if claims:
        return SummaryBlock(type=block_type, claims=claims)

    # No usable attribution left. A `disputed` line still stands on its own —
    # that is the shape every stored post before this used — but an
    # `attributed` block without names has nothing left to say.
    text = _clean(raw.get("text"), MAX_TEXT_LENGTH)
    if block_type == DISPUTED and text:
        return SummaryBlock(type=DISPUTED, text=text)
    return None


def _parse_links(
    raw: dict[str, Any], context: str, known_urls: frozenset[str]
) -> SummaryBlock | None:
    """A list of headlines, each pointing at a post that exists."""
    raw_links = raw.get("links")
    if not isinstance(raw_links, list):
        return None

    links: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_link in raw_links[:MAX_LINKS]:
        if not isinstance(raw_link, dict):
            continue
        text = _clean(raw_link.get("text"), MAX_TEXT_LENGTH)
        url = _clean(raw_link.get("url"), MAX_TEXT_LENGTH)
        if not text or not url:
            continue
        if url not in known_urls:
            logging.info("Dropping an invented link %r for '%s'", url, context)
            continue
        # The same post under two headlines reads as two events.
        if url in seen:
            continue
        seen.add(url)
        links.append({"text": text, "url": url})

    if not links:
        return None
    return SummaryBlock(type=LINKS, links=links)


def _clean(value: Any, max_length: int) -> str:
    """A trimmed single-spaced string, or "" for anything unusable."""
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    return text[:max_length].strip()
