import logging
import re
from typing import Any


MACRO = re.compile(r"%(\w+)%")

#: The marker a channel puts in front of each item of a list: an arrow, a bullet,
#: a coloured circle. Bounded, because a line may start with two or three of them.
LEADING_MARKER = re.compile(r"^([^\w\s]{1,3})\s*")

#: Characters that begin a line of prose rather than an item of a list, and are
#: therefore not markers. Written as an exclusion because the markers themselves
#: are an open set — every channel picks its own square or circle — while the
#: ways of opening a paragraph are few and known. Without this the rule read
#: quoted paragraphs as a list and took the news with them: a report on the
#: strike against the Chernihiv supermarket has four paragraphs opening with
#: «, and a translated analysis has five opening with a dash.
PROSE_LEADS = set("«»\"“”„'‘’—–-−>‹›(),.…:;!?*")

#: Stripped before the line is read at all. A zero-width joiner at the head of a
#: paragraph is invisible, is neither a word character nor a space, and so passed
#: for a bullet — which is how an analysis of eight thousand Facebook posts came
#: to look like a digest.
INVISIBLE = re.compile(r"[​-‏⁠﻿]")

#: Removed from the marker before counting, so that a channel writing ▪️ on one
#: line and ▪ on the next is understood to be writing the same list.
VARIATION_SELECTOR = "️"

#: Shortest line that can carry a whole news item. Below this a marked line is
#: decoration — a footer, a row of links, a sign-off.
MIN_ITEM_LENGTH = 40

#: Four rather than three, decided by reading six days of live posts. At three
#: the rule reached longform articles that cite three of the channel's own
#: earlier pieces — an InformNapalm investigation, a StopFake explainer — and
#: those are the posts it must never touch. At four every remaining case was a
#: list of separate stories.
MIN_ITEMS = 4

#: A link to one of the channel's own earlier posts, which is what a digest is
#: made of: each item points at the post that carried the story.
OWN_POST = re.compile(r"^https?://t\.me/(?P<channel>[\w]+)/\d+/?$", re.IGNORECASE)

#: Share of the post's substantial lines that have to be items. A digest is a
#: list and almost nothing else; an article is prose with a list inside it. This
#: is what separates them once both have four marked lines and four links to the
#: channel's own posts — which an analysis of one subject does have, because it
#: cites its own earlier pieces. Seven tenths leaves room for a heading and a
#: sign-off, and no room for the paragraph of argument that follows a real
#: article's bullets.
MIN_ITEM_SHARE = 0.7


class RubricDetector:
    """Recognises posts that are a channel's routine rather than news.

    The daily 9am minute of silence, a digest of what already happened, a
    funeral notice, the siren going off and the all-clear — a reader scrolling
    the feed does not want any of these, and the category model does not catch
    them: it was trained on what a post is about, and these are about war,
    politics and grief like the news around them.

    This is deliberately not part of `TextProcessor`. That class decides whether
    text is usable — obscene, empty, mangled — and answers by returning an empty
    string, which erases the post. A ritual post is perfectly good text that is
    simply not news, and saying so by setting the category keeps the post in the
    database, visible to the channel statistics, and excluded by the same
    `is_discarded()` rule that already drops what the model calls `not_news`.

    Patterns are anchored wherever the distinction actually lives. A rubric
    heading sits at the start of a post; a news story that merely mentions the
    minute of silence mentions it in the middle of a longer text. Matching a bare
    substring anywhere would take the news with the ritual.

    A pattern may name a macro as `%name%`, expanded from `config["macros"]`
    before compiling. Whatever holds for the whole air-raid group is written
    once as a macro, because writing it out eight times is how seven of them
    come to disagree with the eighth — which is exactly what happened: only the
    siren pattern spelled out the `<place> — <event>` prefix a regional channel
    writes, so the all-clear and the ballistic-threat warning kept reaching the
    feed. Four conditions: a post naming casualties or damage is a report of
    the strike and not of the siren (`not_a_strike`), the event may be preceded
    by the place (`where`) and by the verb that announces it (`announced`), and
    an alert may end by saying what triggered it (`why`). An unknown macro
    raises rather than compiling to a literal `%name%` that would silently
    never match.

    `where` is what 262 live posts turned out to need, and no more. A channel
    puts the place ahead of the event in three ways: after a dash, on a line of
    its own, or behind the word УВАГА — and the marker that follows the newline
    is usually an emoji, so a run of non-word characters has to be allowed after
    the separator. Commas stay excluded, because a place name has none and a
    news lede usually does; that exclusion alone is what keeps "Мерія
    розповіла, чому в Києві оголосили тривогу" out of it.

    The `\\W` runs are bounded rather than starred. Written the obvious way,
    with a bounded class inside an unbounded repeat, the prefix took over five
    seconds on a post of four thousand dashes — a post short enough for Telegram
    to accept.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        macros: dict[str, str] = config.get("macros", {})
        # Kept beside the compiled form so `explain` can answer with the line as
        # written in the config, which is the line a human has to edit.
        self.patterns: list[tuple[str, re.Pattern[str]]] = [
            (pattern, re.compile(self.expand(pattern, macros), re.IGNORECASE))
            for pattern in config.get("patterns", [])
        ]
        # Said out loud because the alternative is silence. `configs` is a
        # mounted volume, so new code can run against a config file that predates
        # it and has no `rubric_detector` section; the detector then matches
        # nothing, every ritual post is published, and the only evidence is in
        # the feed. That is how it went unnoticed for a day.
        if not self.patterns:
            logging.warning(
                "RubricDetector built with no patterns: rubric posts will not be "
                "recognised. Check that annotator_config.json is the version this "
                "code expects."
            )
        else:
            logging.info("RubricDetector compiled %d patterns", len(self.patterns))

    @staticmethod
    def expand(pattern: str, macros: dict[str, str]) -> str:
        def substitute(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in macros:
                raise KeyError(f"Unknown rubric macro '{name}' in {pattern!r}")
            return macros[name]

        return MACRO.sub(substitute, pattern)

    def __call__(self, text: str) -> bool:
        return self.explain(text) is not None

    def explain(self, text: str) -> str | None:
        """The pattern that matched, for working out why a post disappeared."""
        stripped = text.strip()
        if not stripped:
            return None
        for source, regexp in self.patterns:
            if regexp.search(stripped):
                return source
        return None

    @staticmethod
    def listing(text: str, own_post_links: int) -> str | None:
        """A digest recognised by its shape, for the channels that head it with none.

        Some channels publish the day's news as four arrowed lines, each a whole
        story with a link to the post that carried it, and never write the word
        "дайджест" anywhere. No vocabulary reaches those, and adding one channel's
        arrow to a list of words would only start a second list to keep current.

        Read the raw text, not `patched_text`: the text processor takes the emoji
        out, and the marker is usually an emoji. On `patched_text` this rule saw
        only the markers that survive — ▪ and • — which are the ones a single
        story uses for its sub-points, so it found no digests and plenty of news.

        Two conditions, and the second is the one that does the work. Marked lines
        say the post is a list; links to the channel's **own earlier posts** say
        each item is a story of its own. Counting all links instead lets a footer
        decide: Ukrinform signs every post with four links to its social accounts,
        and that alone was enough to make an enumerated report of one night's
        strikes look like a digest of four.
        """
        markers: dict[str, int] = dict()
        substantial = 0
        for raw_line in text.split("\n"):
            line = INVISIBLE.sub("", raw_line).strip()
            if len(line) < MIN_ITEM_LENGTH:
                continue
            substantial += 1
            match = LEADING_MARKER.match(line)
            if not match:
                continue
            marker = match.group(1).replace(VARIATION_SELECTOR, "")
            if not marker or set(marker) & PROSE_LEADS:
                continue
            markers[marker] = markers.get(marker, 0) + 1

        if not markers:
            return None
        marker, items = max(markers.items(), key=lambda pair: pair[1])
        if items < MIN_ITEMS or own_post_links < items:
            return None
        if items < substantial * MIN_ITEM_SHARE:
            return None
        return (
            f"listing: {items} of {substantial} lines led by {marker!r}, "
            f"{own_post_links} own links"
        )


def count_own_posts(links: list[str], channel_id: str) -> int:
    """How many of the links point at an earlier post of this same channel.

    Only a link to a numbered post counts. A link to the channel's front page is
    what a sign-off is made of, and every post has one.
    """
    if not channel_id:
        return 0
    wanted = channel_id.lower().lstrip("@")
    count = 0
    for link in links:
        match = OWN_POST.match(link.strip())
        if match and match.group("channel").lower() == wanted:
            count += 1
    return count
