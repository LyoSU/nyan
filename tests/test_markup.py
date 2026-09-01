"""Tests for the tiny markup language the model writes in.

`parse_markup` has to be total: any string at all, including strings full of
stray delimiters, must produce valid RichText rather than an exception, because
the caller is a daemon rendering a post nobody is watching.
"""

import pytest

from nyan.markup import (
    drop_links,
    find_links,
    link_emphasis,
    parse_markup,
    strip_markup,
    to_html,
)


URL = "https://t.me/UAliveNews/1"


def anchors(value: object) -> list[str]:
    """The text of every link in a parsed value."""
    if isinstance(value, dict):
        return [flatten(value["text"])] if value.get("type") == "url" else []
    if isinstance(value, list):
        return [anchor for item in value for anchor in anchors(item)]
    return []


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


# ------------------------------------------------------------- link_emphasis
#
# The same delimiters mean something else in a digest headline: the span is the
# phrase to hang the link on. Every degenerate case has to fall back to linking
# the whole headline, because a digest line with no link at all is a dead end.


def test_only_the_marked_phrase_becomes_the_link() -> None:
    parsed = link_emphasis("Рада ухвалила **бюджет на 2027 рік**", URL)

    assert parsed == [
        "Рада ухвалила ",
        {"type": "url", "text": "бюджет на 2027 рік", "url": URL},
    ]


def test_the_words_around_the_phrase_stay_plain() -> None:
    parsed = link_emphasis("ППО **збила 15 дронів** над Київщиною", URL)

    assert anchors(parsed) == ["збила 15 дронів"]
    assert flatten(parsed) == "ППО збила 15 дронів над Київщиною"


def test_a_phrase_at_the_start_needs_no_leading_part() -> None:
    parsed = link_emphasis("**Блекаут у Тбілісі** триває другу добу поспіль", URL)

    assert isinstance(parsed, list)
    assert parsed[0] == {"type": "url", "text": "Блекаут у Тбілісі", "url": URL}


def test_a_headline_without_markup_is_linked_whole() -> None:
    """The old behaviour, kept as the fallback: never lose the link."""
    assert link_emphasis("Рада ухвалила бюджет", URL) == {
        "type": "url",
        "text": "Рада ухвалила бюджет",
        "url": URL,
    }


def test_an_unclosed_delimiter_falls_back_to_the_whole_headline() -> None:
    parsed = link_emphasis("Рада ухвалила **бюджет на 2027 рік", URL)

    assert anchors(parsed) == ["Рада ухвалила бюджет на 2027 рік"]
    assert "**" not in flatten(parsed)


def test_a_phrase_covering_the_whole_headline_is_not_a_phrase() -> None:
    """Marking everything is the failure this feature exists to fix."""
    parsed = link_emphasis("**Рада ухвалила бюджет**", URL)

    assert anchors(parsed) == ["Рада ухвалила бюджет"]


def test_a_phrase_covering_most_of_the_headline_loses_its_contrast() -> None:
    parsed = link_emphasis(
        "Рада **ухвалила бюджет на 2027 рік у першому читанні**", URL
    )

    assert anchors(parsed) == ["Рада ухвалила бюджет на 2027 рік у першому читанні"]


def test_a_two_letter_phrase_is_too_small_to_tap() -> None:
    parsed = link_emphasis("Рада ухвалила бюджет **на** 2027 рік", URL)

    assert anchors(parsed) == ["Рада ухвалила бюджет на 2027 рік"]


def test_only_the_first_phrase_becomes_the_link() -> None:
    """Two links on one line would read as two separate news items."""
    parsed = link_emphasis("**Рада** ухвалила **бюджет** на наступний рік", URL)

    assert anchors(parsed) == ["Рада"]
    assert flatten(parsed) == "Рада ухвалила бюджет на наступний рік"


def test_spaces_inside_the_delimiters_stay_outside_the_link() -> None:
    """An underlined trailing space is visible, and looks like a bug."""
    parsed = link_emphasis("Рада ухвалила ** бюджет на 2027 рік ** у читанні", URL)

    assert anchors(parsed) == ["бюджет на 2027 рік"]
    assert flatten(parsed) == "Рада ухвалила  бюджет на 2027 рік  у читанні"


def test_an_italic_span_works_as_the_anchor_too() -> None:
    """The digest does not care which of the two constructs the model reached for."""
    parsed = link_emphasis("Кабмін підвищив __виплати ветеранам__ з січня", URL)

    assert anchors(parsed) == ["виплати ветеранам"]


def test_an_empty_headline_produces_nothing() -> None:
    assert link_emphasis("", URL) == ""
    assert link_emphasis("**", URL) == ""


def test_strip_markup_keeps_the_words() -> None:
    assert strip_markup("Загинуло **троє** людей") == "Загинуло троє людей"


def test_to_html_serializes_the_same_entities_as_tags() -> None:
    text = to_html(link_emphasis("Курс **>42 грн** & далі", "https://t.me/x/1?a=1&b=2"))

    assert text == (
        'Курс <a href="https://t.me/x/1?a=1&amp;b=2">&gt;42 грн</a> &amp; далі'
    )


def test_to_html_handles_bold_italic_and_plain() -> None:
    assert to_html(parse_markup("Без світла **1,2 млн**, __курсив__")) == (
        "Без світла <b>1,2 млн</b>, <i>курсив</i>"
    )
    assert to_html("a < b") == "a &lt; b"
    assert to_html("") == ""


def test_a_link_in_prose_becomes_a_url_entity() -> None:
    """The digest lede: the posts it draws on are reachable from the sentence."""
    assert parse_markup("Загиблих [зросла до восьми](https://t.me/x/1), решта") == [
        "Загиблих ",
        {"type": "url", "text": "зросла до восьми", "url": "https://t.me/x/1"},
        ", решта",
    ]


def test_strip_markup_keeps_the_words_of_a_link() -> None:
    assert strip_markup("Загиблих [зросла **до** восьми](https://t.me/x/1).") == (
        "Загиблих зросла до восьми."
    )


def test_links_survive_when_emphasis_is_dropped_for_covering_the_line() -> None:
    """Emphasis is contrast and goes when there is nothing to contrast with;
    a link is navigation and losing it loses the reader a post."""
    assert parse_markup("**Усе жирне тут** [далі](https://t.me/x/1)") == [
        "Усе жирне тут ",
        {"type": "url", "text": "далі", "url": "https://t.me/x/1"},
    ]


def test_drop_links_reduces_unknown_ones_to_their_words() -> None:
    text = "[Добре](https://t.me/x/1) і [зле](https://evil.example/1)"

    assert drop_links(text, {"https://t.me/x/1"}) == "[Добре](https://t.me/x/1) і зле"
    assert find_links(text) == [
        {"text": "Добре", "url": "https://t.me/x/1"},
        {"text": "зле", "url": "https://evil.example/1"},
    ]
