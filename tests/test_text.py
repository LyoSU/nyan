from nyan.text import TextProcessor


def make_processor(
    skip: list[str] | None = None,
    rm: list[str] | None = None,
    obscene: list[str] | None = None,
) -> TextProcessor:
    return TextProcessor(
        {
            "skip_substrings": skip or [],
            "rm_substrings": rm or [],
            "obscene_substrings": obscene or [],
        }
    )


def test_a_post_carrying_a_skip_substring_is_dropped() -> None:
    processor = make_processor(skip=["хвилина повчання"])

    assert processor("Хвилина повчання: не переходьте дорогу на червоне.") == ""


def test_case_does_not_hide_a_skip_substring() -> None:
    """A channel that shouts its rubric heading is running the same rubric."""
    processor = make_processor(skip=["хвилина повчання"])

    assert processor("ХВИЛИНА ПОВЧАННЯ від нашого редактора.") == ""


def test_an_ordinary_post_survives() -> None:
    processor = make_processor(skip=["хвилина повчання"])

    assert processor("Уряд ухвалив постанову про підвищення тарифів.") != ""


def test_a_skip_substring_is_looked_for_after_cleaning_too() -> None:
    """Removing emoji and hashtags can bring the marker into view."""
    processor = make_processor(skip=["хвилина повчання від редакції"])

    text = "🔴 Хвилина повчання #новини від редакції"

    assert processor(text) == ""


def test_obscene_is_flagged_regardless_of_case() -> None:
    processor = make_processor(obscene=["лайно"])

    assert processor.has_obscene("Повне ЛАЙНО ця постанова")


def test_removal_of_a_substring_keeps_the_rest_of_the_post() -> None:
    processor = make_processor(rm=["Читайте нас у Telegram."])

    result = processor("Уряд ухвалив постанову. Читайте нас у Telegram.")

    assert "Читайте нас" not in result
    assert "Уряд ухвалив постанову." in result


def test_an_empty_configuration_changes_nothing_it_should_not() -> None:
    """The production config ships all three lists nearly empty."""
    processor = make_processor()

    assert processor("Уряд ухвалив постанову про тарифи.") == (
        "Уряд ухвалив постанову про тарифи."
    )
    assert not processor.has_obscene("будь-який текст")
