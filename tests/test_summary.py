"""Tests for the trust boundary between the model and the renderer.

Everything `parse_summary` receives is model output, so the cases here are the
ways a model gets a schema wrong: types that do not exist, fields that are the
wrong kind, a quote with nobody behind it. None of them may raise, and none may
reach the renderer.
"""

from typing import Any

from nyan.summary import (
    ATTRIBUTED,
    DISPUTED,
    HIDDEN,
    LIST,
    QUOTE,
    SUBHEADING,
    TEXT,
    Summary,
    SummaryBlock,
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


def test_as_text_leaves_the_markup_behind() -> None:
    """A prompt reading a stored summary needs the facts, not the delimiters.

    Left in, they would also teach the next model to write more of them — and in
    a digest headline a `**` span means something else entirely: the link anchor.
    """
    summary = Summary(
        blocks=[
            SummaryBlock(type=TEXT, text="Загинуло **троє** людей"),
            SummaryBlock(type=LIST, items=["__Дев'ятеро__ поранені", "Дві будівлі"]),
        ]
    )

    assert summary.as_text() == "Загинуло троє людей Дев'ятеро поранені Дві будівлі"


# --------------------------------------------------- attribution on the claims
#
# The point of these two blocks is whose claim it is, so the cases below are
# about attribution surviving, being checked, and failing safely — a claim
# credited to a channel that was never a source here is worse than no claim.


def test_a_dispute_keeps_every_version_with_its_channels() -> None:
    summary = parse_summary(
        blocks(
            {"type": TEXT, "text": "Лід."},
            {
                "type": DISPUTED,
                "claims": [
                    {"text": "поранених дев'ятеро", "channels": ["suspilne"]},
                    {"text": "поранених одинадцятеро", "channels": ["trukha", "ok"]},
                ],
            },
        ),
        allowed_channels={"suspilne", "trukha", "ok"},
    )

    assert types(summary) == [TEXT, DISPUTED]
    assert summary.blocks[1].claims == [
        {"text": "поранених дев'ятеро", "channels": ["suspilne"]},
        {"text": "поранених одинадцятеро", "channels": ["trukha", "ok"]},
    ]


def test_a_channel_that_was_not_a_source_cannot_be_credited() -> None:
    """The one failure this feature must not have: an invented byline."""
    summary = parse_summary(
        blocks(
            {"type": TEXT, "text": "Лід."},
            {
                "type": ATTRIBUTED,
                "claims": [
                    {"text": "уламки впали на школу", "channels": ["suspilne", "нема"]},
                    {"text": "тривога тривала дві години", "channels": ["вигадка"]},
                ],
            },
        ),
        allowed_channels={"suspilne"},
    )

    assert types(summary) == [TEXT, ATTRIBUTED]
    assert summary.blocks[1].claims == [
        {"text": "уламки впали на школу", "channels": ["suspilne"]}
    ]


def test_an_unattributed_claim_is_dropped_rather_than_shown_bare() -> None:
    summary = parse_summary(
        blocks(
            {"type": TEXT, "text": "Лід."},
            {"type": ATTRIBUTED, "claims": [{"text": "подробиця", "channels": []}]},
        ),
        allowed_channels={"suspilne"},
    )

    assert types(summary) == [TEXT]


def test_one_version_is_not_a_disagreement() -> None:
    """A `disputed` block with a single side is a lone claim, so it is labelled
    as one: telling the reader the sources conflict when only one of them said
    anything is a stronger claim than the sources support."""
    summary = parse_summary(
        blocks(
            {"type": TEXT, "text": "Лід."},
            {
                "type": DISPUTED,
                "claims": [{"text": "загиблих п'ятеро", "channels": ["trukha"]}],
            },
        ),
        allowed_channels={"trukha"},
    )

    assert types(summary) == [TEXT, ATTRIBUTED]


def test_the_same_channels_are_not_credited_twice_in_one_block() -> None:
    summary = parse_summary(
        blocks(
            {"type": TEXT, "text": "Лід."},
            {
                "type": ATTRIBUTED,
                "claims": [
                    {"text": "перша подробиця", "channels": ["ok"]},
                    {"text": "та сама подробиця інакше", "channels": ["ok"]},
                ],
            },
        ),
        allowed_channels={"ok"},
    )

    assert types(summary) == [TEXT, ATTRIBUTED]
    assert summary.blocks[1].claims == [{"text": "перша подробиця", "channels": ["ok"]}]


def test_a_dispute_with_no_usable_attribution_keeps_its_line() -> None:
    """Every post stored before attribution existed has this shape."""
    summary = parse_summary(
        blocks(
            {"type": TEXT, "text": "Лід."},
            {"type": DISPUTED, "text": "поранених від дев'яти до одинадцяти"},
        )
    )

    assert types(summary) == [TEXT, DISPUTED]
    assert summary.blocks[1].text == "поранених від дев'яти до одинадцяти"


def test_claims_survive_a_round_trip_through_storage() -> None:
    original = parse_summary(
        blocks(
            {"type": TEXT, "text": "Лід."},
            {
                "type": ATTRIBUTED,
                "claims": [{"text": "подробиця", "channels": ["ok"]}],
            },
        ),
        allowed_channels={"ok"},
    )

    assert Summary.fromdict(original.asdict()).asdict() == original.asdict()


def test_a_channel_named_a_little_wrong_is_still_matched() -> None:
    """The model copies these ids out of the prompt and misses by a case or an @.

    Matching verbatim would drop real attributions, and storing what it wrote
    would leave the renderer looking up an id no document has — so the id it is
    stored under is the document's own.
    """
    summary = parse_summary(
        blocks(
            {"type": TEXT, "text": "Лід."},
            {
                "type": ATTRIBUTED,
                "claims": [{"text": "подробиця", "channels": ["@Suspilne_ZP"]}],
            },
        ),
        allowed_channels={"suspilne_zp"},
    )

    assert summary.blocks[1].claims == [
        {"text": "подробиця", "channels": ["suspilne_zp"]}
    ]
