"""A deliberately tiny markup language for model-written text.

The model writes news copy, and copy needs emphasis: a number, a name, a
verdict. The obvious route — let it emit Markdown or HTML and parse that — is
where the bugs live, because a general parser has to answer questions the model
will get wrong sooner or later: an unclosed `**`, a `<b>` around a `<a>`, an
ampersand that may or may not need escaping.

So the language here is two constructs and nothing else: `**bold**` and
`__italic__`. Everything the model writes that is not one of those is literal
text. `parse_markup` is total — every possible string maps to valid RichText,
including strings full of stray delimiters — so a badly formatted answer
degrades to plain text instead of raising inside the daemon.

`__` rather than `_` for italics on purpose: channel handles (`it_news`,
`andro_price`) carry single underscores, and marking those up as italics would
mangle names that appear in the copy.

The same two constructs do double duty in a digest, where `link_emphasis` reads
the emphasized span as the phrase to hang the link on. That reuse is the point:
the model marks up what matters in the sentence it just wrote, and no code has
to match a separately-returned phrase back into the headline.
"""

import html
import re
from collections.abc import Collection

from nyan.rich import RichText, bold, italic, link


# One pattern for both constructs. Non-greedy, so the shortest balanced pair
# wins and an unpaired delimiter simply never matches. DOTALL because a span
# may wrap across a newline the model put in.
_MARKUP = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)

# The third construct, for one place only: a digest lede. When a single event
# carries the whole period, the digest opens with a sentence or two about it,
# and the posts that sentence draws on have to be reachable from it — the
# alternative is the same facts again as link rows underneath, which is the
# duplication a lede is meant to avoid. Same idea as the emphasis span picking
# the link anchor: the model marks the phrase in the sentence it is writing.
# Which URLs are real is not this module's business; `nyan.summary` strips the
# ones the model was never given before the text gets here.
_LINK = re.compile(r"\[([^\[\]]+?)\]\((https?://[^\s()]+)\)")

# Both kinds in one scan, so a link and an emphasis span never overlap: the
# earlier one in the text wins and the scan resumes after it.
_TOKEN = re.compile(
    _MARKUP.pattern + r"|\[(?P<link_text>[^\[\]]+?)\]\((?P<url>https?://[^\s()]+)\)",
    re.DOTALL,
)

# Delimiters left over after the paired ones are consumed. They are the model's
# mistakes, and a reader who sees "**" reads it as our bug, so they are dropped
# rather than shown.
_STRAY_DELIMITERS = re.compile(r"\*\*|__")

# Emphasis works by contrast, so a span covering most of the text is not
# emphasis at all. Past this share of the string, markup is discarded and the
# words are kept.
MAX_EMPHASIS_RATIO = 0.7

# Shortest span that may become a link on its own. A two-character anchor is a
# tap target nobody can hit, so such a span is treated as no choice at all and
# the whole line becomes the link.
MIN_ANCHOR_LENGTH = 3


def parse_markup(text: str) -> RichText:
    """`text` with `**bold**` and `__italic__` turned into inline entities.

    Returns a plain string when there is nothing to mark up, so unformatted
    text stays as simple in the payload as it was before.
    """
    if not text:
        return ""

    parts: list[RichText] = []
    emphasized = 0
    position = 0
    for match in _TOKEN.finditer(text):
        plain = text[position : match.start()]
        if plain:
            parts.append(strip_markup(plain))
        if match.group("link_text") is not None:
            content = strip_markup(match.group("link_text")).strip()
            if content:
                parts.append(link(content, match.group("url")))
        else:
            inner, is_bold = (
                (match.group(1), True)
                if match.group(1) is not None
                else (match.group(2), False)
            )
            content = strip_markup(inner)
            if content:
                emphasized += len(content)
                parts.append(bold(content) if is_bold else italic(content))
        position = match.end()

    tail = text[position:]
    if tail:
        parts.append(strip_markup(tail))

    parts = [part for part in parts if part != ""]
    if not parts:
        return ""

    # So much emphasis that the contrast is gone: the words are what matter,
    # so the emphasis goes. Links stay, since a link is navigation rather than
    # contrast, and dropping one would lose the reader a post.
    if emphasized > len(strip_markup(text)) * MAX_EMPHASIS_RATIO:
        parts = [
            part["text"]
            if isinstance(part, dict) and part["type"] in ("bold", "italic")
            else part
            for part in parts
        ]
    return _collapse(parts)


def _collapse(parts: list[RichText]) -> RichText:
    """Adjacent plain strings merged; a lone string returned as itself."""
    merged: list[RichText] = []
    for part in parts:
        if isinstance(part, str) and merged and isinstance(merged[-1], str):
            merged[-1] += part
        else:
            merged.append(part)
    if len(merged) == 1 and isinstance(merged[0], str):
        return merged[0]
    return merged


def link_emphasis(text: str, url: str) -> RichText:
    """`text` with its emphasized span turned into a link to `url`.

    A digest is a page of headlines, and a page where every headline is entirely
    blue has no emphasis left in it: nothing on it stands out, so a reader gets
    no help deciding what to tap. Linking only the phrase that carries the news
    gives every line one point of contrast.

    Which phrase that is, only the model can say — but it says it by marking up
    the sentence it is already writing, so there is nothing to match afterwards.
    That was the whole flaw in asking for the phrase as a separate field: a
    phrase returned apart from its sentence has to be found in it again, and it
    fails to be found for reasons nobody can see from the output — a different
    dash, a collapsed space, a word declined differently the second time.

    Total, like `parse_markup`: a headline with no markup, an unclosed
    delimiter, a span covering the entire line or a two-letter one all fall back
    to linking the whole headline. A digest never loses a link.
    """
    plain = strip_markup(text)
    if not plain:
        return ""

    match = _MARKUP.search(text)
    if match is None:
        return link(plain, url)

    inner = strip_markup(
        match.group(1) if match.group(1) is not None else match.group(2)
    )
    anchor = inner.strip()
    # Whitespace the model left inside the delimiters belongs outside the link:
    # an underlined trailing space is visible, and ugly.
    lead = inner[: len(inner) - len(inner.lstrip())]
    trail = inner[len(inner.rstrip()) :]
    before = strip_markup(text[: match.start()]) + lead
    after = trail + strip_markup(text[match.end() :])

    if (
        len(anchor) < MIN_ANCHOR_LENGTH
        # Same contrast rule as bold: a span covering most of the line leaves
        # nothing for it to contrast against.
        or len(anchor) > len(plain) * MAX_EMPHASIS_RATIO
        # Nothing outside the span, so the "phrase" is the headline.
        or not (before.strip() or after.strip())
    ):
        return link(plain, url)

    return [part for part in (before, link(anchor, url), after) if part != ""]


def strip_markup(text: str) -> str:
    """`text` with the delimiters removed and the words kept.

    For anywhere the markup is meaningless: a prompt reading a stored summary
    needs the facts, and a stray `**` in one only teaches the next model to
    write more of them.
    """
    return _STRAY_DELIMITERS.sub("", _LINK.sub(r"\1", text))


def find_links(text: str) -> list[dict[str, str]]:
    """Every `[text](url)` in the copy, in order, as `{"text", "url"}` pairs."""
    return [
        {"text": strip_markup(m.group(1)).strip(), "url": m.group(2)}
        for m in _LINK.finditer(text)
    ]


def drop_links(text: str, keep: Collection[str]) -> str:
    """`text` with every link whose URL is not in `keep` reduced to its words.

    A link the model invented still carries a true phrase — the phrase is what
    it wrote about the news — so the words stay and only the destination goes.
    """
    return _LINK.sub(lambda m: m.group(0) if m.group(2) in keep else m.group(1), text)


# Inline entity types and the HTML tag Telegram's parse_mode expects for each.
# `marked` has no HTML counterpart, so it degrades to bold: the intent was
# emphasis, and bold is the emphasis HTML has.
_HTML_TAGS = {"bold": "b", "italic": "i", "marked": "b"}


def to_html(text: RichText) -> str:
    """The same inline entities, serialized for a plain `sendMessage`.

    `parse_markup` and `link_emphasis` build entity trees for rich messages,
    and a post that goes out as ordinary HTML — the digest, which is a page of
    lines and gains nothing from rich blocks but their padding — needs the very
    same trees as tags. One serializer keeps the two paths from drifting: the
    model's markup means the same thing whichever way the message is sent.

    Every literal string is escaped here and nowhere else, so news copy that
    contains `<` or `&` can never break the message.
    """
    if isinstance(text, str):
        return html.escape(text, quote=False)
    if isinstance(text, list):
        return "".join(to_html(part) for part in text)
    inner = to_html(text.get("text", ""))
    kind = text.get("type")
    if kind == "url":
        return (
            f'<a href="{html.escape(str(text.get("url", "")), quote=True)}">{inner}</a>'
        )
    tag = _HTML_TAGS.get(str(kind))
    return f"<{tag}>{inner}</{tag}>" if tag else inner
