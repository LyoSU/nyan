"""Tests for the trust boundary between the model and the renderer.

Everything `parse_summary` receives is model output, so the cases here are the
ways a model gets a schema wrong: types that do not exist, fields that are the
wrong kind, a quote with nobody behind it. None of them may raise, and none may
reach the renderer.
"""

from typing import Any

from nyan.summary import (
    DISPUTED,
    HIDDEN,
    LIST,
    QUOTE,
    SUBHEADING,
    TEXT,
    Summary,
    parse_summary,
)


def blocks(*raw: dict[str, Any]) -> dict[str, Any]:
    return {"headline": "Заголовок", "blocks": list(raw)}


def types(summary: Summary) -> list[str]:
    return [block.type for block in summary.blocks]


def test_a_post_of_one_paragraph_survives_intact() -> None:
    summary = parse_summary(blocks({"type": TEXT, "text": "Що сталося."}))

    assert summary
    assert summary.headline == "Заголовок"
    assert types(summary) == [TEXT]


def test_the_model_chooses_the_shape() -> None:
    """No fixed template: whatever order the model picks is kept."""
    summary = parse_summary(
        blocks(
            {"type": TEXT, "text": "Лід."},
            {"type": QUOTE, "text": "Слова", "author": "Посадовець"},
            {"type": SUBHEADING, "text": "Друга частина"},
            {"type": LIST, "items": ["Раз", "Два"]},
            {"type": HIDDEN, "summary": "Передісторія", "text": "Було раніше."},
            {"type": DISPUTED, "text": "Джерела називають різні цифри"},
        )
    )

    assert types(summary) == [TEXT, QUOTE, SUBHEADING, LIST, HIDDEN, DISPUTED]


def test_an_unknown_block_type_is_dropped() -> None:
    summary = parse_summary(
        blocks(
            {"type": TEXT, "text": "Лід."},
            {"type": "table", "rows": [[1, 2]]},
            {"type": "image", "url": "https://example.com/a.jpg"},
        )
    )

    assert types(summary) == [TEXT]


def test_an_unattributed_quote_is_dropped() -> None:
    """Words in quote marks that nobody is on record saying is the one thing
    this feature must never produce."""
    summary = parse_summary(
        blocks(
            {"type": TEXT, "text": "Лід."},
            {"type": QUOTE, "text": "Ми цього не робили"},
        )
    )

    assert types(summary) == [TEXT]


def test_a_post_cannot_open_with_bullets() -> None:
    summary = parse_summary(
        blocks(
            {"type": LIST, "items": ["Раз", "Два"]},
            {"type": TEXT, "text": "Лід."},
        )
    )

    assert types(summary) == [TEXT]


def test_a_post_may_open_with_a_quote() -> None:
    """A quote lead is legitimate journalism, unlike a bullet lead."""
    summary = parse_summary(
        blocks({"type": QUOTE, "text": "Слова", "author": "Посадовець"})
    )

    assert types(summary) == [QUOTE]


def test_a_one_item_list_becomes_a_paragraph() -> None:
    summary = parse_summary(
        blocks(
            {"type": TEXT, "text": "Лід."},
            {"type": LIST, "items": ["Єдиний факт"]},
        )
    )

    assert types(summary) == [TEXT, TEXT]
    assert summary.blocks[1].text == "Єдиний факт"


def test_a_hidden_block_without_a_summary_is_shown_instead() -> None:
    """Otherwise the reader is asked to tap on nothing in particular."""
    summary = parse_summary(
        blocks(
            {"type": TEXT, "text": "Лід."},
            {"type": HIDDEN, "text": "Подробиці."},
        )
    )

    assert types(summary) == [TEXT, TEXT]


def test_repeated_quotes_beyond_the_limit_are_dropped() -> None:
    """A stack of quote cards inverts the post's hierarchy."""
    summary = parse_summary(
        blocks(
            {"type": TEXT, "text": "Лід."},
            *[
                {"type": QUOTE, "text": f"Слова {i}", "author": f"Автор {i}"}
                for i in range(5)
            ],
        )
    )

    assert types(summary).count(QUOTE) == 2


def test_disputed_returned_as_a_field_is_still_used() -> None:
    """An older prompt shape returned it that way; accepting both costs a branch."""
    summary = parse_summary(
        {
            "headline": "Заголовок",
            "blocks": [{"type": TEXT, "text": "Лід."}],
            "disputed": "Джерела називають різні цифри",
        }
    )

    assert types(summary) == [TEXT, DISPUTED]
    assert summary.blocks[1].text == "Джерела називають різні цифри"


def test_a_disputed_block_wins_over_the_field() -> None:
    summary = parse_summary(
        {
            "headline": "Заголовок",
            "blocks": [
                {"type": TEXT, "text": "Лід."},
                {"type": DISPUTED, "text": "З блоку"},
            ],
            "disputed": "З поля",
        }
    )

    assert [b.text for b in summary.blocks if b.type == DISPUTED] == ["З блоку"]


def test_junk_instead_of_an_answer_yields_nothing() -> None:
    for raw in ([], "текст", None, 42, {}, {"blocks": "не список"}):
        assert not parse_summary(raw)


def test_wrongly_typed_fields_are_ignored() -> None:
    summary = parse_summary(
        {
            "headline": 42,
            "blocks": [
                "не об'єкт",
                {"type": TEXT, "text": ["не", "рядок"]},
                {"type": LIST, "items": "не список"},
                {"type": TEXT, "text": "Єдиний придатний блок."},
            ],
        }
    )

    assert summary.headline == ""
    assert types(summary) == [TEXT]


def test_whitespace_is_normalized() -> None:
    summary = parse_summary(blocks({"type": TEXT, "text": "  Текст\n\n  з дірами  "}))

    assert summary.blocks[0].text == "Текст з дірами"


def test_a_runaway_answer_is_capped() -> None:
    summary = parse_summary(
        blocks(*[{"type": TEXT, "text": f"Абзац {i}."} for i in range(40)])
    )

    assert len(summary.blocks) <= 10


def test_a_summary_survives_storage() -> None:
    original = parse_summary(
        blocks(
            {"type": TEXT, "text": "Лід."},
            {"type": QUOTE, "text": "Слова", "author": "Посадовець"},
            {"type": LIST, "items": ["Раз", "Два"]},
        )
    )

    restored = Summary.fromdict(original.asdict())

    assert restored == original
