from nyan.restatement import (
    LEDE_RESTATED,
    QUOTED_RESTATED,
    find_sentence_end,
    restatement,
    stems,
)


HEADLINE = "Путін заперечив мобілізацію в РФ після виборів"


def test_the_same_sentence_in_more_words_is_a_restatement() -> None:
    """The case from the channel: every word of the headline is back, padded
    with 'повідомлення про нову хвилю' and 'осінніх', which add nothing."""
    lede = (
        "Путін заперечив повідомлення про нову хвилю мобілізації "
        "в Росії після осінніх виборів."
    )
    assert restatement(HEADLINE, lede) >= LEDE_RESTATED


def test_a_continuation_is_not_a_restatement() -> None:
    """Same subject, new facts: the weapon, the count, the victim. Cutting this
    would lose the story's numbers, so it must stay under the threshold."""
    headline = "Росія вдарила по Запоріжжю, є загибла"
    lede = (
        "Росія вдарила по Запоріжжю керованими авіабомбами, "
        "загинула 58-річна жінка, дев'ятеро поранені."
    )
    assert restatement(headline, lede) < LEDE_RESTATED


def test_a_sentence_about_something_else_scores_nothing() -> None:
    headline = "Зеленський анонсував нові санкції проти Росії"
    lede = "Президент заявив, що наступний пакет торкнеться тіньового флоту."
    assert restatement(headline, lede) == 0.0


def test_a_verbatim_copy_scores_one() -> None:
    assert restatement(HEADLINE, HEADLINE + ".") == 1.0


def test_inflection_does_not_hide_a_repeated_word() -> None:
    """Ukrainian declines its nouns, so 'мобілізацію' and 'мобілізації' have to
    count as the same word or nearly every restatement would slip through."""
    assert stems("мобілізацію") == stems("мобілізації")
    assert stems("виборів") == stems("вибори")


def test_function_words_and_markup_are_ignored() -> None:
    assert stems("у в на про після **через** та") == set()
    assert stems("**Путін**") == stems("Путін")


def test_a_number_is_a_word_even_when_short() -> None:
    """'58' is two characters, and still the fact that matters in a sentence."""
    assert "58" in stems("загинуло 58 людей")


def test_the_quoted_threshold_is_stricter_than_the_lede_one() -> None:
    """A channel's own sentence is only dropped when it is almost nothing but the
    headline; a sentence the model wrote against its instructions goes sooner."""
    assert QUOTED_RESTATED > LEDE_RESTATED


def test_a_channel_sentence_that_adds_a_reason_is_kept() -> None:
    headline = "У Києві оголосили повітряну тривогу"
    body = "У Києві та області оголосили повітряну тривогу через загрозу балістики з півночі."
    assert LEDE_RESTATED <= restatement(headline, body) < QUOTED_RESTATED


def test_a_channel_sentence_that_is_the_headline_is_not() -> None:
    headline = "У Києві оголосили повітряну тривогу"
    assert restatement(headline, "Увага! У Києві оголосили повітряну тривогу.") >= QUOTED_RESTATED


def test_sentence_end_is_after_the_punctuation_or_at_a_newline() -> None:
    assert find_sentence_end("Перше речення. Друге.") == len("Перше речення.")
    assert find_sentence_end("Лід без крапки\nДалі текст") == len("Лід без крапки")
    assert find_sentence_end("Одне речення.") is None
