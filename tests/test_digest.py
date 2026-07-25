"""Tests for the digest: its window, its link validation and its rendering.

The window is the part with teeth. A digest that does not publish must leave its
posts for the next one, and a digest that does publish must not count the same
post twice — both of which are properties of where the window starts, not of the
text the model writes.
"""

import json
from typing import Any

import pytest

from nyan import digest
from nyan.summary import DIGEST_LIMITS, LINKS, SUBHEADING, TEXT, Summary, parse_summary
from nyan.util import format_period_uk


HOUR = 3600
NOW = 1753300000


class FakeCollection:
    """Just enough of a Mongo collection for the watermark."""

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.records = records or []

    def find_one(self, query: Any, sort: Any = None) -> dict[str, Any] | None:
        found = [r for r in self.records if "published_until" in r]
        if not found:
            return None
        return max(found, key=lambda r: r["published_until"])

    def insert_one(self, record: dict[str, Any]) -> None:
        self.records.append(record)


@pytest.fixture
def collection(monkeypatch: Any) -> FakeCollection:
    fake = FakeCollection()
    monkeypatch.setattr(digest, "get_topics_collection", lambda _: fake)
    return fake


def test_the_first_ever_digest_uses_the_requested_duration(
    collection: FakeCollection,
) -> None:
    start, end = digest.window("mongo.json", duration_hours=8, now=NOW)

    assert (start, end) == (NOW - 8 * HOUR, NOW)


def test_the_window_starts_where_the_last_digest_stopped(
    collection: FakeCollection,
) -> None:
    """Nothing may fall between two digests, and nothing may be counted twice."""
    collection.records.append({"published_until": NOW - 3 * HOUR})

    start, end = digest.window("mongo.json", duration_hours=8, now=NOW)

    assert (start, end) == (NOW - 3 * HOUR, NOW)


def test_a_quiet_shift_is_carried_into_the_next_digest(
    collection: FakeCollection,
) -> None:
    """The watermark only moves when something was published, so an unpublished
    shift is not lost: the next window still reaches back over it."""
    collection.records.append({"published_until": NOW - 20 * HOUR})

    start, _ = digest.window("mongo.json", duration_hours=8, now=NOW)

    # Twenty hours back, not eight: the posts of the shift that never published
    # are inside the window.
    assert start == NOW - 20 * HOUR


def test_a_long_silence_is_clamped(collection: FakeCollection) -> None:
    """Otherwise a week of failures builds a prompt too large to send."""
    collection.records.append({"published_until": NOW - 500 * HOUR})

    start, _ = digest.window("mongo.json", duration_hours=8, now=NOW)

    assert start == NOW - digest.MAX_CATCHUP_HOURS * HOUR


def test_a_watermark_in_the_future_is_ignored(collection: FakeCollection) -> None:
    """A clock change or a manual run must not produce an empty window forever."""
    collection.records.append({"published_until": NOW + 5 * HOUR})

    start, end = digest.window("mongo.json", duration_hours=8, now=NOW)

    assert (start, end) == (NOW - 8 * HOUR, NOW)


def digest_summary(*blocks: dict[str, Any], urls: set[str]) -> Summary:
    return parse_summary(
        {"headline": "Головне", "blocks": list(blocks)},
        context="digest",
        allowed_urls=urls,
        limits=DIGEST_LIMITS,
    )


def test_only_links_the_model_was_given_survive() -> None:
    """An invented link sends the reader to a post that does not exist."""
    summary = digest_summary(
        {
            "type": LINKS,
            "links": [
                {"text": "Справжня новина", "url": "https://t.me/UAliveNews/1"},
                {"text": "Вигадана новина", "url": "https://t.me/UAliveNews/999"},
            ],
        },
        urls={"https://t.me/UAliveNews/1"},
    )

    assert [link["text"] for link in summary.blocks[0].links] == ["Справжня новина"]


def test_a_block_of_only_invented_links_is_dropped() -> None:
    summary = digest_summary(
        {"type": LINKS, "links": [{"text": "Вигадана", "url": "https://evil.example"}]},
        urls={"https://t.me/UAliveNews/1"},
    )

    assert not summary


def test_the_same_post_is_not_listed_twice() -> None:
    """One post under two headlines reads as two events."""
    url = "https://t.me/UAliveNews/1"
    summary = digest_summary(
        {
            "type": LINKS,
            "links": [
                {"text": "Перший заголовок", "url": url},
                {"text": "Другий заголовок", "url": url},
            ],
        },
        urls={url},
    )

    assert len(summary.blocks[0].links) == 1


def test_a_digest_may_carry_a_heading_per_topic() -> None:
    """Unlike a story, where a third heading means the model lost the plot."""
    blocks: list[dict[str, Any]] = []
    urls = {f"https://t.me/UAliveNews/{i}" for i in range(6)}
    for i in range(6):
        blocks.append({"type": SUBHEADING, "text": f"Тема {i}"})
        blocks.append(
            {
                "type": LINKS,
                "links": [{"text": f"Новина {i}", "url": f"https://t.me/UAliveNews/{i}"}],
            }
        )
    summary = digest_summary(*blocks, urls=urls)

    assert [b.type for b in summary.blocks].count(SUBHEADING) == 6


def test_a_headline_without_a_marked_phrase_is_linked_whole() -> None:
    """The fallback: an unmarked headline still gets the reader to the post."""
    summary = digest_summary(
        {
            "type": LINKS,
            "links": [{"text": "Новина", "url": "https://t.me/UAliveNews/1"}],
        },
        urls={"https://t.me/UAliveNews/1"},
    )

    blocks = digest.render_digest(summary, NOW - 8 * HOUR, NOW, "8 годин")

    assert [b["type"] for b in blocks] == ["heading", "list", "divider", "footer"]
    item = blocks[1]["items"][0]["blocks"][0]
    assert item["text"] == {
        "type": "url",
        "text": "Новина",
        "url": "https://t.me/UAliveNews/1",
    }


def test_only_the_marked_phrase_of_a_headline_is_linked() -> None:
    """A digest of end-to-end blue headlines emphasizes nothing at all."""
    summary = digest_summary(
        {
            "type": LINKS,
            "links": [
                {
                    "text": "Рада ухвалила **бюджет на 2027 рік**",
                    "url": "https://t.me/UAliveNews/1",
                }
            ],
        },
        urls={"https://t.me/UAliveNews/1"},
    )

    blocks = digest.render_digest(summary, NOW - 8 * HOUR, NOW, "8 годин")

    assert blocks[1]["items"][0]["blocks"][0]["text"] == [
        "Рада ухвалила ",
        {
            "type": "url",
            "text": "бюджет на 2027 рік",
            "url": "https://t.me/UAliveNews/1",
        },
    ]


def test_the_footer_names_the_span_the_digest_covers() -> None:
    summary = digest_summary(
        {"type": TEXT, "text": "Текст."}, urls=set()
    )

    blocks = digest.render_digest(summary, NOW - 8 * HOUR, NOW, "8 годин")

    assert "—" in blocks[-1]["text"]


def test_a_digest_without_a_headline_names_the_period() -> None:
    summary = digest_summary({"type": TEXT, "text": "Текст."}, urls=set())
    summary.headline = ""

    blocks = digest.render_digest(summary, NOW - 8 * HOUR, NOW, "8 годин")

    assert blocks[0]["text"] == "Головне за 8 годин"


def test_missing_posts_are_reported() -> None:
    clusters = [
        {"url": "https://t.me/UAliveNews/1"},
        {"url": "https://t.me/UAliveNews/2"},
    ]
    summary = digest_summary(
        {
            "type": LINKS,
            "links": [{"text": "Новина", "url": "https://t.me/UAliveNews/1"}],
        },
        urls={"https://t.me/UAliveNews/1", "https://t.me/UAliveNews/2"},
    )

    assert digest.count_missing(summary, clusters) == ["https://t.me/UAliveNews/2"]


@pytest.mark.parametrize(
    "hours,expected",
    [
        (1, "1 годину"),
        (2, "2 години"),
        (4, "4 години"),
        (5, "5 годин"),
        (8, "8 годин"),
        (11, "11 годин"),
        (21, "21 годину"),
        (24, "добу"),
        (26, "добу"),
        (48, "2 дні"),
        (72, "3 дні"),
        (120, "5 днів"),
    ],
)
def test_the_period_is_declined_correctly(hours: float, expected: str) -> None:
    assert format_period_uk(hours) == expected


def _patch_digest_llm(monkeypatch: Any) -> list[dict[str, Any]]:
    """Replace the LLM, returning the list that records every call."""
    calls: list[dict[str, Any]] = []

    def fake_openai_completion(**kwargs: Any) -> str:
        calls.append(kwargs)
        return json.dumps(
            {"headline": "Головне", "blocks": [{"type": TEXT, "text": "Текст."}]},
            ensure_ascii=False,
        )

    monkeypatch.setattr("nyan.digest.openai_completion", fake_openai_completion)
    return calls


DIGEST_PROMPT = str(digest.BASE_DIR / "prompts/digest.txt")
ONE_CLUSTER = [
    {
        "url": "https://t.me/UAliveNews/1",
        "headline": "Новина",
        "text": "Текст новини.",
        "views": 1000,
        "sources_count": 2,
    }
]


def test_the_digest_prompt_tells_the_model_what_day_it_is(monkeypatch: Any) -> None:
    """Otherwise it writes "як до кінця року, так і до кінця 2026 року"."""
    calls = _patch_digest_llm(monkeypatch)

    summary = digest.write_digest(
        ONE_CLUSTER,
        prompt_path=DIGEST_PROMPT,
        model_name="model",
        period="8 годин",
        today="25 липня 2026 року",
    )

    assert summary
    assert "Сьогодні 25 липня 2026 року." in calls[0]["messages"][0]["content"]


def test_a_digest_written_without_a_date_falls_back_to_today(monkeypatch: Any) -> None:
    calls = _patch_digest_llm(monkeypatch)

    digest.write_digest(
        ONE_CLUSTER, prompt_path=DIGEST_PROMPT, model_name="model", period="8 годин"
    )

    prompt = calls[0]["messages"][0]["content"]
    assert "Сьогодні" in prompt
    assert "{{today}}" not in prompt


PREVIOUS_RECORD = {
    "published_until": NOW - 8 * HOUR,
    "summary": {
        "headline": "Головне за 8 годин: удари по енергетиці",
        "blocks": [
            {"type": SUBHEADING, "text": "⚡ Енергетика"},
            {
                "type": LINKS,
                "links": [
                    {
                        "text": "Росія вдарила по **енергетиці Харкова**",
                        "url": "https://t.me/UAliveNews/1",
                    }
                ],
            },
            {"type": TEXT, "text": "Без світла залишилися 1,2 млн абонентів."},
        ],
    },
}


def test_the_last_digest_is_the_most_recent_one(collection: FakeCollection) -> None:
    collection.records.append({"published_until": NOW - 16 * HOUR, "summary": {}})
    collection.records.append(PREVIOUS_RECORD)

    record = digest.read_last_digest("mongo.json")

    assert record is not None
    assert record["published_until"] == NOW - 8 * HOUR


def test_the_previous_digest_is_summed_up_by_its_headlines() -> None:
    form = digest.previous_form(PREVIOUS_RECORD)

    assert form["headline"] == "Головне за 8 годин: удари по енергетиці"
    assert form["topics"] == ["⚡ Енергетика"]
    # Markup stripped: in a digest headline the ** span picks the link anchor, so
    # handing it back would teach the model to copy asterisks into a place where
    # they mean something else.
    assert form["headlines"] == ["Росія вдарила по енергетиці Харкова"]


def test_no_body_of_the_previous_digest_is_passed_on() -> None:
    """Headlines are context; facts are not.

    A number from the previous digest belongs to no link in this one, and a
    digest may only state what the posts it lists actually say. So the model is
    given enough to recognize a continuing story and not enough to describe one.
    """
    form = digest.previous_form(PREVIOUS_RECORD)

    assert "1,2 млн" not in json.dumps(form, ensure_ascii=False)


def test_a_first_ever_digest_has_no_previous_form() -> None:
    assert digest.previous_form(None) == {}


def test_the_previous_headlines_reach_the_prompt(monkeypatch: Any) -> None:
    """A long story runs across digests: the strike, then the confirmed toll."""
    calls = _patch_digest_llm(monkeypatch)

    digest.write_digest(
        ONE_CLUSTER,
        prompt_path=DIGEST_PROMPT,
        model_name="model",
        period="8 годин",
        previous=digest.previous_form(PREVIOUS_RECORD),
    )

    prompt = calls[0]["messages"][0]["content"]
    assert "Росія вдарила по енергетиці Харкова" in prompt
    assert "Нічого з цього списку в добірку не переноси" in prompt


def test_a_digest_with_no_predecessor_says_nothing_about_one(monkeypatch: Any) -> None:
    calls = _patch_digest_llm(monkeypatch)

    digest.write_digest(
        ONE_CLUSTER, prompt_path=DIGEST_PROMPT, model_name="model", period="8 годин"
    )

    assert "Про попередню добірку" not in calls[0]["messages"][0]["content"]
