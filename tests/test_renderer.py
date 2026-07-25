import re
from typing import Any
from collections.abc import Sequence

import pytest

from nyan.channels import Channels
from nyan.clusters import Cluster
from nyan.document import Document
from nyan.renderer import Renderer, pluralize_sources

# Channels present in tests/channels.json, one per trust group of the "main"
# issue, so the rendered breakdown has something to group by.
OFFICIAL = "rian_ru"
VERIFIED = "rbc_news"
AGGREGATOR = "nexta_live"


def make_doc(
    channel_id: str,
    url: str,
    pub_time: int = 1700000000,
    views: int = 500,
    text: str = "Основний текст новини.",
    images: Sequence[str] = (),
    videos: Sequence[str] = (),
    links: Sequence[str] = (),
) -> Document:
    return Document(
        url=url,
        channel_id=channel_id,
        post_id=1,
        views=views,
        pub_time=pub_time,
        patched_text=text,
        channel_title=channel_id.upper(),
        images=images,
        videos=videos,
        links=links,
    )


def make_cluster(
    docs: Sequence[Document],
    headline: str | None = "Заголовок новини",
    differences: Sequence[dict[str, Any]] = (),
    embedded_images: Sequence[dict[str, str]] = (),
) -> Cluster:
    cluster = Cluster()
    for doc in docs:
        cluster.add(doc)
    annotation_doc = docs[0]
    annotation_doc.embedded_images = list(embedded_images)
    cluster.saved_annotation_doc = annotation_doc
    cluster.saved_analysis = {"headline": headline, "differences": list(differences)}
    return cluster


def find(blocks: Sequence[dict[str, Any]], block_type: str) -> dict[str, Any]:
    matches = [b for b in blocks if b["type"] == block_type]
    assert matches, "No {} block in {}".format(
        block_type, [b["type"] for b in blocks]
    )
    return matches[0]


# Keys that carry readable content; media urls and ids are deliberately left out.
_CONTENT_KEYS = ("summary", "text", "credit", "caption", "blocks", "items")


def flatten_text(value: Any) -> str:
    """All plain text inside a block or RichText tree, for substring assertions."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(
            flatten_text(value[key]) for key in _CONTENT_KEYS if key in value
        )
    return ""


@pytest.fixture
def two_group_cluster() -> Cluster:
    return make_cluster(
        [
            make_doc(VERIFIED, "https://t.me/rbc_news/1", pub_time=100),
            make_doc(AGGREGATOR, "https://t.me/nexta_live/1", pub_time=200),
        ]
    )


def test_post_is_rich_by_default(renderer: Renderer, two_group_cluster: Cluster) -> None:
    post = renderer.render_cluster(two_group_cluster, "main")
    assert post is not None
    assert post.is_rich


def test_block_order_is_stable(renderer: Renderer) -> None:
    """Readers learn to scan a fixed shape, so the order must not vary."""
    cluster = make_cluster(
        [
            make_doc(
                VERIFIED, "https://t.me/rbc_news/1", pub_time=100, images=("a",)
            ),
            make_doc(
                AGGREGATOR, "https://t.me/nexta_live/1", pub_time=200, images=("b",)
            ),
        ],
        differences=[{"channel_ids": [AGGREGATOR], "text": "додаткова деталь"}],
        embedded_images=[{"url": "https://example.com/a.jpg"}],
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    assert [b["type"] for b in post.blocks] == [
        "heading",
        "photo",
        "paragraph",
        "blockquote",
        "details",
        "divider",
        "footer",
    ]


def test_llm_headline_becomes_the_heading(renderer: Renderer) -> None:
    cluster = make_cluster(
        [make_doc(VERIFIED, "https://t.me/rbc_news/1", text="Довгий текст новини.")],
        headline="Коротко про суть",
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    assert find(post.blocks, "heading")["text"] == "Коротко про суть"
    assert find(post.blocks, "paragraph")["text"] == "Довгий текст новини."


def test_important_cluster_gets_a_bigger_heading(renderer: Renderer) -> None:
    cluster = make_cluster([make_doc(VERIFIED, "https://t.me/rbc_news/1")])
    normal = renderer.render_cluster(cluster, "main")
    cluster.is_important = True
    important = renderer.render_cluster(cluster, "main")

    assert normal is not None and normal.blocks is not None
    assert important is not None and important.blocks is not None
    assert find(normal.blocks, "heading")["size"] == 3
    assert find(important.blocks, "heading")["size"] == 2


def test_without_a_headline_the_first_sentence_stands_in(renderer: Renderer) -> None:
    cluster = make_cluster(
        [
            make_doc(
                VERIFIED,
                "https://t.me/rbc_news/1",
                text="Головне сталося вчора. А це подробиці події.",
            )
        ],
        headline=None,
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    assert find(post.blocks, "heading")["text"] == "Головне сталося вчора."
    # The remainder must not repeat the heading.
    assert find(post.blocks, "paragraph")["text"] == "А це подробиці події."


def test_a_lead_on_its_own_line_becomes_the_heading(renderer: Renderer) -> None:
    """Telegram posts break the lead with a newline, not ". ".

    Looking only for a period followed by a space left most posts with no
    heading at all, since their text never contains one.
    """
    cluster = make_cluster(
        [
            make_doc(
                VERIFIED,
                "https://t.me/rbc_news/1",
                text=(
                    "Федорову запропонували посаду радника — Reuters.\n"
                    "«Я зателефонував йому», — сказав міністр."
                ),
            )
        ],
        headline=None,
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    heading = find(post.blocks, "heading")
    assert heading["text"] == "Федорову запропонували посаду радника — Reuters."
    body = find(post.blocks, "paragraph")["text"]
    assert body == "«Я зателефонував йому», — сказав міністр."


def test_a_lead_without_final_punctuation_still_becomes_a_heading(
    renderer: Renderer,
) -> None:
    cluster = make_cluster(
        [
            make_doc(
                VERIFIED,
                "https://t.me/rbc_news/1",
                text=(
                    "Зеленський підтвердив ураження\n"
                    "За даними Президента, атаковано три об'єкти."
                ),
            )
        ],
        headline=None,
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    assert find(post.blocks, "heading")["text"] == "Зеленський підтвердив ураження"


def test_an_overlong_lead_is_left_in_the_body(renderer: Renderer) -> None:
    long_lead = "Дуже довгий вступ, " * 10
    cluster = make_cluster(
        [make_doc(VERIFIED, "https://t.me/rbc_news/1", text=f"{long_lead}\nдалі текст.")],
        headline=None,
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    assert [b for b in post.blocks if b["type"] == "heading"] == []


def test_single_sentence_without_a_headline_gets_no_heading(renderer: Renderer) -> None:
    cluster = make_cluster(
        [make_doc(VERIFIED, "https://t.me/rbc_news/1", text="Одне речення без кінця")],
        headline=None,
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    assert [b["type"] for b in post.blocks if b["type"] == "heading"] == []
    assert find(post.blocks, "paragraph")["text"] == "Одне речення без кінця"


def test_several_photos_become_a_slideshow(renderer: Renderer) -> None:
    cluster = make_cluster(
        [
            make_doc(VERIFIED, "https://t.me/rbc_news/1", images=("a",), pub_time=100),
            make_doc(
                AGGREGATOR, "https://t.me/nexta_live/1", images=("b",), pub_time=200
            ),
        ],
        embedded_images=[
            {"url": "https://example.com/a.jpg"},
            {"url": "https://example.com/b.jpg"},
        ],
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    slideshow = find(post.blocks, "slideshow")
    assert [b["type"] for b in slideshow["blocks"]] == ["photo", "photo"]


def test_video_wins_over_photos(renderer: Renderer) -> None:
    cluster = make_cluster(
        [
            make_doc(
                VERIFIED,
                "https://t.me/rbc_news/1",
                videos=("https://example.com/v.mp4",),
            )
        ],
        embedded_images=[{"url": "https://example.com/a.jpg"}],
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    assert [b["type"] for b in post.blocks if b["type"] in ("video", "photo")] == [
        "video"
    ]


def test_difference_is_credited_to_the_channel_reporting_it(renderer: Renderer) -> None:
    cluster = make_cluster(
        [
            make_doc(VERIFIED, "https://t.me/rbc_news/1", pub_time=100),
            make_doc(AGGREGATOR, "https://t.me/nexta_live/1", pub_time=200),
        ],
        differences=[{"channel_ids": [AGGREGATOR], "text": "затримали підозрюваного."}],
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    quote = find(post.blocks, "blockquote")
    assert quote["blocks"][0]["text"] == "затримали підозрюваного"
    assert flatten_text(quote["credit"]) == AGGREGATOR.upper()


def test_difference_from_an_unknown_channel_is_dropped(renderer: Renderer) -> None:
    cluster = make_cluster(
        [make_doc(VERIFIED, "https://t.me/rbc_news/1")],
        differences=[{"channel_ids": ["hallucinated_channel"], "text": "деталь"}],
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    assert [b for b in post.blocks if b["type"] == "blockquote"] == []


def test_sources_summary_is_just_a_count(renderer: Renderer) -> None:
    """A per-group row of emoji and digits read as a badge, not as information."""
    cluster = make_cluster(
        [
            make_doc(OFFICIAL, "https://t.me/rian_ru/1", pub_time=100),
            make_doc(VERIFIED, "https://t.me/rbc_news/1", pub_time=200),
            make_doc(AGGREGATOR, "https://t.me/nexta_live/1", pub_time=300),
        ]
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    assert find(post.blocks, "details")["summary"] == "3 джерела"


def test_group_titles_are_spaced_after_their_emoji(
    renderer: Renderer, two_group_cluster: Cluster
) -> None:
    """A glyph set solid against the next word reads as one token."""
    post = renderer.render_cluster(two_group_cluster, "main")

    assert post is not None
    assert post.blocks is not None
    listing = flatten_text(find(post.blocks, "details")["blocks"][0])
    assert not re.search(r"[^\s\w][\wА-Яа-яІіЇїЄєҐґ]", listing), listing


def test_documents_from_an_unlisted_channel_are_skipped(renderer: Renderer) -> None:
    """A channel removed from channels.json still has documents in Mongo.

    Rendering used to raise KeyError on the first such cluster, which took down
    the whole iteration.
    """
    cluster = make_cluster(
        [
            make_doc(VERIFIED, "https://t.me/rbc_news/1", pub_time=100),
            make_doc(VERIFIED, "https://t.me/rbc_news/2", pub_time=200),
        ]
    )
    cluster.docs[1].channel_id = "channel_that_was_removed"

    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    listing = flatten_text(find(post.blocks, "details"))
    assert "channel_that_was_removed" not in listing


def test_a_cluster_of_only_unlisted_channels_renders_nothing(
    renderer: Renderer,
) -> None:
    cluster = make_cluster([make_doc(VERIFIED, "https://t.me/rbc_news/1", pub_time=100)])
    cluster.docs[0].channel_id = "channel_that_was_removed"

    assert renderer.render_cluster(cluster, "main") is None


def test_sources_are_collapsed_but_group_titles_are_readable(
    renderer: Renderer, two_group_cluster: Cluster
) -> None:
    post = renderer.render_cluster(two_group_cluster, "main")

    assert post is not None
    assert post.blocks is not None
    details = find(post.blocks, "details")
    assert "is_open" not in details
    listing = flatten_text(details["blocks"][0])
    assert "Перевірені медіа" in listing
    assert "Новинні" in listing


def test_provenance_lives_inside_the_disclosure(renderer: Renderer) -> None:
    cluster = make_cluster(
        [
            make_doc(
                VERIFIED,
                "https://t.me/rbc_news/1",
                pub_time=100,
                links=("https://lb.ua/story",),
            ),
            make_doc(
                AGGREGATOR,
                "https://t.me/nexta_live/1",
                pub_time=200,
                links=("https://lb.ua/story",),
            ),
        ]
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    details = flatten_text(find(post.blocks, "details"))
    assert "Першим" in details
    assert "Ймовірне першоджерело" in details
    assert "lb.ua" in details


def test_external_link_needs_two_channels(renderer: Renderer) -> None:
    cluster = make_cluster(
        [make_doc(VERIFIED, "https://t.me/rbc_news/1", links=("https://lb.ua/story",))]
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    assert "Ймовірне першоджерело" not in flatten_text(find(post.blocks, "details"))


def test_first_publication_time_is_localized_for_the_reader(renderer: Renderer) -> None:
    doc = make_doc(VERIFIED, "https://t.me/rbc_news/1", pub_time=1700000000)
    post = renderer.render_cluster(make_cluster([doc]), "main")

    assert post is not None
    assert post.blocks is not None
    details = find(post.blocks, "details")
    entities = _collect(details, lambda b: b.get("type") == "date_time")
    assert len(entities) == 1
    assert entities[0]["unix_time"] == 1700000000
    # "r" would read as "a week ago" once the post is in the archive.
    assert entities[0]["date_time_format"] == "t"


def _collect(value: Any, predicate: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if predicate(value):
            found.append(value)
        for item in value.values():
            found.extend(_collect(item, predicate))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect(item, predicate))
    return found


def test_footer_credits_the_quoted_channel_and_reach(renderer: Renderer) -> None:
    cluster = make_cluster(
        [make_doc(VERIFIED, "https://t.me/rbc_news/1", views=12400)]
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    footer = flatten_text(find(post.blocks, "footer"))
    assert VERIFIED.upper() in footer
    assert "12,4K" in footer


def test_channel_titles_need_no_escaping(renderer: Renderer) -> None:
    doc = make_doc(VERIFIED, "https://t.me/rbc_news/1")
    doc.channel_title = "Кіно & Театр <18>"
    post = renderer.render_cluster(make_cluster([doc]), "main")

    assert post is not None
    assert post.blocks is not None
    assert "Кіно & Театр <18>" in flatten_text(find(post.blocks, "footer"))


def test_cluster_without_a_configured_group_renders_nothing(
    renderer: Renderer, channels: Channels
) -> None:
    cluster = make_cluster([make_doc(VERIFIED, "https://t.me/rbc_news/1")])
    assert renderer.render_cluster(cluster, "no_such_issue") is None


def test_legacy_format_still_produces_text_and_media(
    renderer_config_path: str, channels: Channels
) -> None:
    import json
    import tempfile

    with open(renderer_config_path) as f:
        config = json.load(f)
    config["post_format"] = "legacy"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        legacy_config_path = f.name

    legacy_renderer = Renderer(legacy_config_path, channels)
    cluster = make_cluster(
        [
            make_doc(VERIFIED, "https://t.me/rbc_news/1", pub_time=100),
            make_doc(AGGREGATOR, "https://t.me/nexta_live/1", pub_time=200),
        ],
        differences=[{"channel_ids": [AGGREGATOR], "text": "деталь"}],
    )
    post = legacy_renderer.render_cluster(cluster, "main")

    assert post is not None
    assert not post.is_rich
    assert post.text is not None
    assert "Основний текст новини" in post.text
    # The legacy template needs the channel credit as markup.
    assert '<a href="https://t.me/nexta_live/1">' in post.text


@pytest.mark.parametrize(
    "count,expected",
    [
        (1, "джерело"),
        (2, "джерела"),
        (4, "джерела"),
        (5, "джерел"),
        (11, "джерел"),
        (12, "джерел"),
        (21, "джерело"),
        (22, "джерела"),
        (25, "джерел"),
    ],
)
def test_source_count_is_declined_correctly(count: int, expected: str) -> None:
    assert pluralize_sources(count) == expected
