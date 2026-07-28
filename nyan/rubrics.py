import logging
import re
from typing import Any


MACRO = re.compile(r"%(\w+)%")


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
        stripped = text.strip()
        if not stripped:
            return False
        return any(regexp.search(stripped) for _, regexp in self.patterns)

    def explain(self, text: str) -> str | None:
        """The pattern that matched, for working out why a post disappeared."""
        stripped = text.strip()
        if not stripped:
            return None
        for source, regexp in self.patterns:
            if regexp.search(stripped):
                return source
        return None
