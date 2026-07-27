import re
from typing import Any


class RubricDetector:
    """Recognises posts that are a channel's routine rather than news.

    The daily 9am minute of silence, a digest of what already happened, a
    funeral notice — a reader scrolling the feed does not want any of these, and
    the category model does not catch them: it was trained on what a post is
    about, and these are about war, politics and grief like the news around them.

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
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in config.get("patterns", [])
        ]

    def __call__(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        return any(pattern.search(stripped) for pattern in self.patterns)

    def explain(self, text: str) -> str | None:
        """The pattern that matched, for working out why a post disappeared."""
        stripped = text.strip()
        if not stripped:
            return None
        for pattern in self.patterns:
            if pattern.search(stripped):
                return pattern.pattern
        return None
