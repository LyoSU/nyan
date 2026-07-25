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
"""

import re

from nyan.rich import RichText, bold, italic


# One pattern for both constructs. Non-greedy, so the shortest balanced pair
# wins and an unpaired delimiter simply never matches. DOTALL because a span
# may wrap across a newline the model put in.
_MARKUP = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)

# Delimiters left over after the paired ones are consumed. They are the model's
# mistakes, and a reader who sees "**" reads it as our bug, so they are dropped
# rather than shown.
_STRAY_DELIMITERS = re.compile(r"\*\*|__")

# Emphasis works by contrast, so a span covering most of the text is not
# emphasis at all. Past this share of the string, markup is discarded and the
# words are kept.
MAX_EMPHASIS_RATIO = 0.7


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
    for match in _MARKUP.finditer(text):
        plain = text[position : match.start()]
        if plain:
            parts.append(_strip_strays(plain))
        inner, is_bold = (
            (match.group(1), True) if match.group(1) is not None else (match.group(2), False)
        )
        content = _strip_strays(inner)
        if content:
            emphasized += len(content)
            parts.append(bold(content) if is_bold else italic(content))
        position = match.end()

    tail = text[position:]
    if tail:
        parts.append(_strip_strays(tail))

    parts = [part for part in parts if part != ""]
    if not parts:
        return ""

    # No markup found, or so much of it that the contrast is gone: the words
    # are what matter, so return them unadorned.
    plain_text = _strip_strays(text)
    if emphasized == 0 or emphasized > len(plain_text) * MAX_EMPHASIS_RATIO:
        return plain_text
    if len(parts) == 1 and isinstance(parts[0], str):
        return parts[0]
    return parts


def _strip_strays(text: str) -> str:
    return _STRAY_DELIMITERS.sub("", text)
