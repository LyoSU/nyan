"""Tests for the tiny markup language the model writes in.

`parse_markup` has to be total: any string at all, including strings full of
stray delimiters, must produce valid RichText rather than an exception, because
the caller is a daemon rendering a post nobody is watching.
"""

import pytest

from nyan.markup import parse_markup


def flatten(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(flatten(item) for item in value)
    if isinstance(value, dict):
        return flatten(value["text"])
    return ""


def test_plain_text_stays_a_plain_string() -> None:
    """Unformatted text must not grow a wrapper it does not need."""
    assert parse_markup("Просто текст") == "Просто текст"


def test_bold_becomes_an_entity() -> None:
    parsed = parse_markup("Загинула **58-річна жінка**, дев'ятеро поранені")

    assert parsed == [
        "Загинула ",
        {"type": "bold", "text": "58-річна жінка"},
        ", дев'ятеро поранені",
    ]


def test_italic_uses_double_underscores() -> None:
    parsed = parse_markup("Джерело називає це __ймовірною__ причиною аварії")

    assert isinstance(parsed, list)
    assert parsed[1] == {"type": "italic", "text": "ймовірною"}


def test_a_single_underscore_is_left_alone() -> None:
    """Channel handles carry them, and italicizing a name mangles it."""
    assert parse_markup("Канал it_news повідомив") == "Канал it_news повідомив"


@pytest.mark.parametrize(
    "text",
    [
        "Незакритий **жирний",
        "**",
        "____",
        "Текст ** з ** пробілами навколо",
        "***",
        "**__",
    ],
)
def test_broken_markup_degrades_to_text(text: str) -> None:
    """The model will get this wrong eventually; nothing may raise."""
    parsed = parse_markup(text)

    assert "**" not in flatten(parsed)
    assert "__" not in flatten(parsed)


def test_emphasis_over_most_of_the_text_is_dropped() -> None:
    """Emphasis works by contrast, so a bolded sentence is not emphasis."""
    parsed = parse_markup("**Уся ця фраза виділена жирним від початку до кінця**")

    assert parsed == "Уся ця фраза виділена жирним від початку до кінця"


def test_an_empty_string_stays_empty() -> None:
    assert parse_markup("") == ""


def test_two_spans_in_one_sentence() -> None:
    parsed = parse_markup("Загинуло **троє**, поранено __дванадцять__ людей у місті")

    assert isinstance(parsed, list)
    kinds = [part["type"] for part in parsed if isinstance(part, dict)]
    assert kinds == ["bold", "italic"]


def test_a_span_across_a_line_break_still_parses() -> None:
    parsed = parse_markup("Заява **на дві\nстрічки** тексту")

    assert isinstance(parsed, list)
    assert parsed[1] == {"type": "bold", "text": "на дві\nстрічки"}


def test_nested_delimiters_do_not_produce_leftovers() -> None:
    parsed = parse_markup("**жирний з __курсивом__ усередині** і ще текст")

    assert "__" not in flatten(parsed)
    assert "**" not in flatten(parsed)
