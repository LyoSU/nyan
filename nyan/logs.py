"""Log formatting for the daemon and the digest.

The daemon's log is read in two places that want opposite things. Coolify's log
viewer strips ANSI escapes before streaming them to the browser, so colour there
is invisible at best; a terminal over ssh renders it fine. The split this module
takes: readability comes from layout, which survives everywhere, and colour is a
bonus applied only when the stream is a terminal.
"""

import logging
import os
import re
import sys
from datetime import datetime
from typing import IO, Literal

Outcome = Literal["added", "important", "skipped"]

RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"

# Shortened so every level fits one column: WARNING and CRITICAL are the only
# names longer than ERROR, and the extra width would be paid on every line.
_LEVEL_NAMES = {logging.WARNING: "WARN", logging.CRITICAL: "CRIT"}
_LEVEL_COLOURS = {
    logging.DEBUG: DIM,
    logging.WARNING: YELLOW,
    logging.ERROR: RED,
    logging.CRITICAL: BOLD + RED,
}

_OUTCOME_LABELS: dict[str, str] = {
    "added": "added",
    "important": "added ★",
    "skipped": "skipped",
}
_OUTCOME_COLOURS = {"added": GREEN, "important": YELLOW, "skipped": DIM}

_LEVEL_WIDTH = 5
_OUTCOME_WIDTH = 9
# Eleven fits 999 999 999 views per hour with room to spare. Fixed rather than
# fitted to the data: lines are formatted one at a time, so there is no way to
# know how wide the next number will be.
_METRIC_WIDTH = 11
_BANNER_WIDTH = 72

_logger = logging.getLogger("nyan")


def should_colorize(stream: IO[str]) -> bool:
    """Decide whether `stream` can carry colour, following the usual conventions.

    NO_COLOR (no-color.org) is the override of last resort, so it beats an
    explicit LOG_COLOR=always: the point of the convention is that a user can
    turn colour off across every tool at once.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False

    mode = (os.environ.get("LOG_COLOR") or "auto").lower()
    if mode == "always":
        return True
    if mode == "never":
        return False

    if os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


# The bot token sits in the path of every Bot API call, and httpx logs the
# request url in full. Matching the path segment rather than a bare token keeps
# this from firing on ordinary text that happens to look like one.
_BOT_TOKEN = re.compile(r"/bot\d+:[A-Za-z0-9_-]+")
_REDACTED = "/bot<token>"


class RedactSecrets(logging.Filter):
    """Strip bot tokens from records before any handler writes them out.

    A filter rather than formatting: the token has to be gone from the record
    itself, so it cannot reach a second handler — or a log aggregator — through
    a path this module does not control.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _BOT_TOKEN.sub(_REDACTED, record.msg)
        # httpx passes the url as an argument, so the token is not in `msg` yet
        # at the time this runs.
        if isinstance(record.args, tuple):
            record.args = tuple(
                _BOT_TOKEN.sub(_REDACTED, arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True


def _paint(text: str, colour: str, colorize: bool) -> str:
    if not colorize or not colour:
        return text
    return f"{colour}{text}{RESET}"


def _group_thousands(value: int) -> str:
    # Locale-independent on purpose: locale.format_string mutates process-wide
    # state, and the daemon runs background threads that did not ask for it.
    # A plain space, not U+00A0: the two look identical in a terminal, but a
    # number copied out of the log has to match what the reader then greps for.
    return f"{value:,}".replace(",", " ")


def iteration_banner(colorize: bool, when: datetime | None = None) -> str:
    """The full-width rule between iterations, and the only place the date lives.

    Per-line stamps carry the time alone — the date on every line was ten
    characters of noise repeated for the length of a run.
    """
    stamp = (when or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    head = "───── new iteration "
    tail = f" {stamp} ─────"
    fill = max(1, _BANNER_WIDTH - len(head) - len(tail))
    return _paint(f"{head}{'─' * fill}{tail}", DIM, colorize)


class NyanFormatter(logging.Formatter):
    """Renders `HH:MM:SS  LEVEL  [outcome] [views]  message`.

    The two optional columns are filled from `outcome` and `metric` attributes
    that `log_cluster` attaches; records without them — including every line
    from httpx, scrapy and the rest — render as plain messages.
    """

    def __init__(self, colorize: bool = False) -> None:
        super().__init__()
        self.colorize = colorize

    def _columns(self, record: logging.LogRecord) -> str:
        outcome = getattr(record, "outcome", None)
        if outcome not in _OUTCOME_LABELS:
            return ""

        # Padded before painting: escape sequences count towards a format
        # spec's width, so colouring first would misalign every coloured line.
        label = f"{_OUTCOME_LABELS[outcome]:<{_OUTCOME_WIDTH}}"
        metric = getattr(record, "metric", None)
        cell = _group_thousands(metric) if isinstance(metric, int) else ""
        painted = _paint(label, _OUTCOME_COLOURS[outcome], self.colorize)
        return f"{painted}{cell:>{_METRIC_WIDTH}}  "

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "banner", False):
            return iteration_banner(self.colorize, datetime.fromtimestamp(record.created))

        stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        name = _LEVEL_NAMES.get(record.levelno, record.levelname)
        level = f"{name:<{_LEVEL_WIDTH}}"

        line = (
            f"{_paint(stamp, DIM, self.colorize)}  "
            f"{_paint(level, _LEVEL_COLOURS.get(record.levelno, ''), self.colorize)}  "
            f"{self._columns(record)}{record.getMessage()}"
        )

        # Replayed by hand because overriding format() bypasses the base class's
        # own handling — and daemon.py's logging.exception calls exist for the
        # traceback, not the sentence in front of it.
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            line += "\n" + self.formatStack(record.stack_info)
        return line


def log_cluster(outcome: Outcome, views_per_hour: int, title: str) -> None:
    """Log one ranker verdict about a cluster.

    The title stays the record's message rather than moving into `extra`, so the
    line still reads correctly under any other formatter — pytest's caplog, or a
    handler some library installs.
    """
    # Passed with no args, so logging leaves the string alone: headlines contain
    # percent signs, and %-formatting them would raise inside the logging call.
    _logger.info(title, extra={"outcome": outcome, "metric": views_per_hour})


def log_new_iteration() -> None:
    _logger.info("new iteration", extra={"banner": True})


def setup_logging() -> None:
    """Install the formatter on the root logger. Safe to call more than once."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(NyanFormatter(colorize=should_colorize(handler.stream)))
    handler.addFilter(RedactSecrets())

    # force=True so a second entry point (digest after send, or a library that
    # called basicConfig first) replaces the handler instead of doubling it.
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL") or "INFO", handlers=[handler], force=True
    )
