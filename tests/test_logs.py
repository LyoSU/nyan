import io
import logging
import re

from nyan.logs import (
    NyanFormatter,
    RedactSecrets,
    iteration_banner,
    log_cluster,
    should_colorize,
)


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def _record(message: str, level: int = logging.INFO, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("nyan", level, __file__, 0, message, None, None)
    record.__dict__.update(extra)
    return record


def test_cluster_lines_line_up_their_columns() -> None:
    """The ranker emits twenty of these in a row, so the titles must share an edge.

    Ragged left edges are what made the original log hard to skim: a six-digit
    view count and a seven-digit one pushed their titles to different columns.
    """
    formatter = NyanFormatter(colorize=False)

    added = formatter.format(_record("Туреччина обмежила", outcome="added", metric=161996))
    skipped = formatter.format(_record("ЄС надав Україні", outcome="skipped", metric=30327))
    important = formatter.format(
        _record("Найближчими ночами", outcome="important", metric=1347322)
    )

    assert added.index("Туреччина") == skipped.index("ЄС") == important.index("Найближчими")


def test_view_counts_are_grouped_into_thousands() -> None:
    formatter = NyanFormatter(colorize=False)

    line = formatter.format(_record("Найближчими ночами", outcome="important", metric=1347322))

    assert "1 347 322" in line


def test_plain_output_carries_no_escape_sequences() -> None:
    """Coolify strips ANSI before streaming, and log files keep it as garbage.

    Either way an escape sequence in non-terminal output is pure loss, so the
    uncoloured formatter must never emit one.
    """
    formatter = NyanFormatter(colorize=False)

    line = formatter.format(_record("dropping 7 clusters", level=logging.WARNING))

    assert "\x1b" not in line


def test_colour_appears_only_when_asked_for() -> None:
    line = NyanFormatter(colorize=True).format(_record("failed", level=logging.ERROR))

    assert "\x1b[" in line
    assert "failed" in line


def test_every_line_keeps_its_own_timestamp() -> None:
    """Deliberately not abbreviated to a repeat marker: grep must stay useful.

    A line found by grep has to say when it happened without its neighbours.
    """
    formatter = NyanFormatter(colorize=False)

    line = formatter.format(_record("13 clusters after the first filter"))

    assert re.match(r"^\d{2}:\d{2}:\d{2} ", line)


def test_a_traceback_survives_the_custom_formatter() -> None:
    """The daemon calls logging.exception when an iteration fails.

    Overriding format() without replaying formatException() silently throws the
    traceback away, which is the one thing that log line exists to carry.
    """
    formatter = NyanFormatter(colorize=False)
    try:
        raise ValueError("mongo is unreachable")
    except ValueError:
        import sys

        record = _record("Could not read documents, waiting", level=logging.ERROR)
        record.exc_info = sys.exc_info()

    line = formatter.format(record)

    assert "ValueError: mongo is unreachable" in line
    assert "Traceback" in line


def test_third_party_lines_pass_through_untouched() -> None:
    """httpx logs every Telegram call through the same root logger."""
    formatter = NyanFormatter(colorize=False)

    line = formatter.format(_record('HTTP Request: POST https://api.telegram.org "200 OK"'))

    assert line.endswith('HTTP Request: POST https://api.telegram.org "200 OK"')


def test_no_color_wins_over_an_explicit_request_for_colour(monkeypatch) -> None:
    """no-color.org makes NO_COLOR the override of last resort. Honour that."""
    monkeypatch.setenv("LOG_COLOR", "always")
    monkeypatch.setenv("NO_COLOR", "1")

    assert should_colorize(_Terminal()) is False


def test_colour_is_off_when_the_stream_is_not_a_terminal(monkeypatch) -> None:
    monkeypatch.delenv("LOG_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)

    assert should_colorize(io.StringIO()) is False
    assert should_colorize(_Terminal()) is True


def test_colour_can_be_forced_into_a_pipe(monkeypatch) -> None:
    """For `docker logs | less -R`, where the stream is a pipe but eyes are real."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("LOG_COLOR", "always")

    assert should_colorize(io.StringIO()) is True


def test_a_dumb_terminal_gets_no_colour(monkeypatch) -> None:
    monkeypatch.delenv("LOG_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")

    assert should_colorize(_Terminal()) is False


def test_log_cluster_hands_the_formatter_the_pieces_it_needs(caplog) -> None:
    """The title stays the message so the line still reads without the formatter."""
    with caplog.at_level(logging.INFO, logger="nyan"):
        log_cluster("added", 161996, "Туреччина обмежила рух суден")

    record = caplog.records[-1]
    assert record.getMessage() == "Туреччина обмежила рух суден"
    assert record.outcome == "added"
    assert record.metric == 161996


def test_a_title_with_a_percent_sign_is_not_treated_as_a_format_string(caplog) -> None:
    """Headlines carry percentages, and %-formatting on user text raises."""
    with caplog.at_level(logging.INFO, logger="nyan"):
        log_cluster("skipped", 100, "Інфляція 12% за рік")

    assert caplog.records[-1].getMessage() == "Інфляція 12% за рік"


def test_the_iteration_banner_carries_the_full_date() -> None:
    """The per-line timestamps drop the date; the banner is where it lives."""
    banner = iteration_banner(colorize=False)

    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", banner)
    assert "new iteration" in banner


def test_a_bot_token_never_reaches_the_log() -> None:
    """httpx logs the full request URL, and ours carries the bot token in it.

    Every one of those lines went to the Coolify log panel, to any file the
    output was piped into, and to whoever was shown a screenshot.
    """
    record = _record(
        'HTTP Request: POST '
        'https://api.telegram.org/bot1234567890:FAKE-TOKEN-FOR-TESTS-ONLY'
        '/sendRichMessage "HTTP/1.1 400 Bad Request"'
    )

    assert RedactSecrets().filter(record) is True
    line = NyanFormatter(colorize=False).format(record)

    assert "FAKE-TOKEN-FOR-TESTS-ONLY" not in line
    assert "1234567890" not in line
    # The method still has to be readable: that is why the line is kept at all.
    assert "sendRichMessage" in line


def test_redaction_survives_deferred_formatting() -> None:
    """httpx passes the url as an argument, not baked into the message."""
    record = logging.LogRecord(
        "httpx", logging.INFO, __file__, 0,
        'HTTP Request: %s "%s"',
        ("https://api.telegram.org/bot123456:SECRET_TOKEN_HERE/sendMessage", "200 OK"),
        None,
    )

    RedactSecrets().filter(record)

    assert "SECRET_TOKEN_HERE" not in record.getMessage()


def test_redaction_leaves_ordinary_lines_alone() -> None:
    record = _record("13 clusters after the first filter")

    RedactSecrets().filter(record)

    assert record.getMessage() == "13 clusters after the first filter"
