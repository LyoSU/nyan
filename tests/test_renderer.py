import json
import logging
import re
from typing import Any
from collections.abc import Sequence

import pytest

from nyan.channels import Channels
from nyan.clusters import Cluster
from nyan.document import Document
from nyan.renderer import Renderer, pluralize_sources
from tests.conftest import get_renderer_config_path

# Channels present in tests/channels.json, one per accountability tier of the
# "main" issue, so the rendered breakdown has something to group by.
OFFICIAL = "rian_ru"
VERIFIED = "rbc_news"
AGGREGATOR = "nexta_live"
# Crawled so that whether it carried a story can be counted, never printed.
MONITORED = "meduzalive"


def make_doc(
    channel_id: str,
    url: str,
    pub_time: int = 1700000000,
    views: int = 500,
    text: str = "Основний текст новини.",
    images: Sequence[str] = (),
    videos: Sequence[str] = (),
    links: Sequence[str] = (),
    embedded_images: Sequence[dict[str, Any]] = (),
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
        embedded_images=list(embedded_images),
    )


def make_cluster(
    docs: Sequence[Document],
    headline: str | None = "Заголовок новини",
    embedded_images: Sequence[dict[str, str]] = (),
    summary: dict[str, Any] | None = None,
) -> Cluster:
    """A cluster whose analysis is already filled in, so nothing calls the LLM.

    Without `summary` the cluster renders the fallback shape: the chosen
    channel's own text, credited to it.
    """
    cluster = Cluster()
    for doc in docs:
        cluster.add(doc)
    annotation_doc = docs[0]
    if embedded_images:
        annotation_doc.embedded_images = list(embedded_images)
    cluster.saved_annotation_doc = annotation_doc
    cluster.saved_analysis = {
        "headline": headline,
        "summary": summary,
        "generation": cluster.generation,
    }
    return cluster


def make_summary(*blocks: dict[str, Any], headline: str = "Заголовок новини") -> dict[str, Any]:
    return {"headline": headline, "blocks": list(blocks)}


def find(blocks: Sequence[dict[str, Any]], block_type: str) -> dict[str, Any]:
    matches = [b for b in blocks if b["type"] == block_type]
    assert matches, "No {} block in {}".format(
        block_type, [b["type"] for b in blocks]
    )
    return matches[0]


def headline_text(blocks: Sequence[dict[str, Any]]) -> Any:
    """The post's headline, unwrapped from the bold entity inside the heading."""
    return find(blocks, "heading")["text"]["text"]


def squeeze(text: str) -> str:
    """Collapse runs of whitespace.

    `flatten_text` joins nested inline parts with a space of its own, so a
    separator inside the tree shows up doubled. Telegram renders the parts
    without that seam, and these assertions are about what the reader sees.
    """
    return re.sub(r"\s+", " ", text).strip()


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
    """The frame around the story is fixed, whatever shape the story has."""
    cluster = make_cluster(
        [
            make_doc(
                VERIFIED, "https://t.me/rbc_news/1", pub_time=100, images=("a",)
            ),
            make_doc(
                AGGREGATOR, "https://t.me/nexta_live/1", pub_time=200, images=("b",)
            ),
        ],
        embedded_images=[{"url": "https://example.com/a.jpg"}],
        summary=make_summary(
            {"type": "text", "text": "Що сталося."},
            {"type": "list", "items": ["Раз", "Два"]},
        ),
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    assert [b["type"] for b in post.blocks] == [
        "heading",
        # The model's own blocks, in the order it chose them, with the media
        # after the lede.
        "paragraph",
        "photo",
        "list",
        "details",
        "divider",
        "footer",
    ]


def media_cluster(*blocks: dict[str, Any]) -> Cluster:
    """A cluster with one photo and, if given, a summary of `blocks`."""
    return make_cluster(
        [make_doc(VERIFIED, "https://t.me/rbc_news/1", pub_time=100, images=("a",))],
        embedded_images=[{"url": "https://example.com/a.jpg"}],
        summary=make_summary(*blocks) if blocks else None,
    )


def block_types(renderer: Renderer, cluster: Cluster) -> list[str]:
    post = renderer.render_cluster(cluster, "main")
    assert post is not None and post.blocks is not None
    return [b["type"] for b in post.blocks]


def test_media_follows_the_lede(renderer: Renderer) -> None:
    """Headline, lede, photo — the order a newspaper page puts them in.

    The reader gets what happened before the picture of it, instead of scrolling
    a slideshow to reach the first sentence.
    """
    types = block_types(
        renderer,
        media_cluster(
            {"type": "text", "text": "Що сталося."},
            {"type": "text", "text": "Подробиці."},
        ),
    )

    assert types[:4] == ["heading", "paragraph", "photo", "paragraph"]


def test_media_stays_on_top_when_the_post_opens_with_a_subheading(
    renderer: Renderer,
) -> None:
    """A block that introduces what follows it must not be cut off from it.

    Only a paragraph is a lede. A subheading, a list of links or a quote owns
    the blocks under it, and a photo dropped in between separates a label from
    the thing it labels.
    """
    types = block_types(
        renderer,
        media_cluster(
            {"type": "subheading", "text": "Головне"},
            {"type": "text", "text": "Що сталося."},
        ),
    )

    assert types[:4] == ["heading", "photo", "heading", "paragraph"]


def test_the_quoted_post_keeps_its_media_on_top(renderer: Renderer) -> None:
    """Without a summary there is no lede to hold the photo back.

    The fallback shape carries the channel's whole post as one paragraph, so
    "after the first paragraph" would mean after all of the text.
    """
    types = block_types(renderer, media_cluster())

    assert types[:3] == ["heading", "photo", "paragraph"]


def test_a_quoted_post_is_credited_and_a_summarized_one_is_not(
    renderer: Renderer,
) -> None:
    """A byline under text the channel did not write would be a lie."""
    docs = [
        make_doc(VERIFIED, "https://t.me/rbc_news/1", pub_time=100),
        make_doc(AGGREGATOR, "https://t.me/nexta_live/1", pub_time=200),
    ]
    quoted = renderer.render_cluster(make_cluster(docs), "main")
    summarized = renderer.render_cluster(
        make_cluster(docs, summary=make_summary({"type": "text", "text": "Зведено."})),
        "main",
    )

    assert quoted is not None and quoted.blocks is not None
    assert summarized is not None and summarized.blocks is not None
    assert f"— {VERIFIED.upper()}" in squeeze(flatten_text(quoted.blocks))
    assert VERIFIED.upper() not in flatten_text(
        [b for b in summarized.blocks if b["type"] == "paragraph"]
    )


def test_llm_headline_becomes_the_heading(renderer: Renderer) -> None:
    cluster = make_cluster(
        [make_doc(VERIFIED, "https://t.me/rbc_news/1", text="Довгий текст новини.")],
        headline="Коротко про суть",
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    assert headline_text(post.blocks) == "Коротко про суть"
    assert find(post.blocks, "paragraph")["text"] == "Довгий текст новини."


def test_the_headline_is_bold_inside_its_heading(renderer: Renderer) -> None:
    """A heading's own weight is semibold, which at this size reads as body text
    in a larger font. Side by side in a real client, the bold one is the one that
    looks like a news headline."""
    cluster = make_cluster(
        [make_doc(VERIFIED, "https://t.me/rbc_news/1")], headline="Коротко про суть"
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None and post.blocks is not None
    assert find(post.blocks, "heading")["text"] == {
        "type": "bold",
        "text": "Коротко про суть",
    }


def test_important_cluster_gets_a_bigger_heading(renderer: Renderer) -> None:
    cluster = make_cluster([make_doc(VERIFIED, "https://t.me/rbc_news/1")])
    normal = renderer.render_cluster(cluster, "main")
    cluster.is_important = True
    important = renderer.render_cluster(cluster, "main")

    assert normal is not None and normal.blocks is not None
    assert important is not None and important.blocks is not None
    # Smaller number, bigger type: sizes run 1-6 the way h1-h6 do.
    assert find(important.blocks, "heading")["size"] < find(normal.blocks, "heading")["size"]


def make_renderer(tmp_path: Any, channels: Channels, **overrides: Any) -> Renderer:
    with open(get_renderer_config_path()) as r:
        config = json.load(r)
    config.update(overrides)
    path = tmp_path / "renderer_config.json"
    path.write_text(json.dumps(config))
    return Renderer(str(path), channels)


def heading_sizes(renderer: Renderer) -> list[int]:
    """The post's own headline, then a section heading inside the story."""
    cluster = make_cluster(
        [
            make_doc(VERIFIED, "https://t.me/rbc_news/1"),
            make_doc(AGGREGATOR, "https://t.me/nexta_live/1"),
        ],
        summary=make_summary(
            {"type": "text", "text": "Лід."},
            {"type": "subheading", "text": "Друга частина"},
            {"type": "text", "text": "Продовження."},
        ),
    )
    post = renderer.render_cluster(cluster, "main")
    assert post is not None and post.blocks is not None
    return [b["size"] for b in post.blocks if b["type"] == "heading"]


def test_headline_size_is_configurable(tmp_path: Any, channels: Channels) -> None:
    """One knob, so a retune cannot leave two levels the same size."""
    smaller = make_renderer(tmp_path, channels, headline_size=5)

    assert heading_sizes(smaller) == [5, 6]
    assert heading_sizes(make_renderer(tmp_path, channels, headline_size=2)) == [2, 3]


def test_headline_size_out_of_range_is_clamped(
    tmp_path: Any, channels: Channels
) -> None:
    """Telegram rejects a size outside 1-6, and a bad config must not post nothing."""
    renderer = make_renderer(tmp_path, channels, headline_size=9)

    assert renderer.headline_size == 6
    # Nothing goes below the smallest size, so the section heading shares it.
    assert renderer.section_size == 6
    assert renderer.important_headline_size == 5


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
    assert headline_text(post.blocks) == "Головне сталося вчора."
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
    assert heading["text"]["text"] == "Федорову запропонували посаду радника — Reuters."
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
    assert headline_text(post.blocks) == "Зеленський підтвердив ураження"


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
    """One photo per channel, so a well-covered event shows several angles."""
    cluster = make_cluster(
        [
            make_doc(
                VERIFIED,
                "https://t.me/rbc_news/1",
                images=("a",),
                pub_time=100,
                embedded_images=[{"url": "https://example.com/a.jpg"}],
            ),
            make_doc(
                AGGREGATOR,
                "https://t.me/nexta_live/1",
                images=("b",),
                pub_time=200,
                embedded_images=[{"url": "https://example.com/b.jpg"}],
            ),
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


def summarized_post(renderer: Renderer, *blocks: dict[str, Any]) -> list[dict[str, Any]]:
    cluster = make_cluster(
        [
            make_doc(VERIFIED, "https://t.me/rbc_news/1", pub_time=100),
            make_doc(AGGREGATOR, "https://t.me/nexta_live/1", pub_time=200),
        ],
        summary=make_summary(*blocks),
    )
    post = renderer.render_cluster(cluster, "main")
    assert post is not None and post.blocks is not None
    return post.blocks


def test_the_summary_headline_becomes_the_heading(renderer: Renderer) -> None:
    blocks = summarized_post(renderer, {"type": "text", "text": "Лід."})

    assert headline_text(blocks) == "Заголовок новини"


def test_a_quote_names_the_person_who_said_it(renderer: Renderer) -> None:
    """A quotation belongs to a speaker, which is what makes it a quotation."""
    blocks = summarized_post(
        renderer,
        {"type": "text", "text": "Лід."},
        {"type": "quote", "text": "Ми цього не робили", "author": "Іван Федоров"},
    )

    quote = find(blocks, "blockquote")
    assert quote["credit"] == "Іван Федоров"
    assert flatten_text(quote["blocks"]) == "Ми цього не робили"


def test_markup_in_the_text_becomes_entities(renderer: Renderer) -> None:
    blocks = summarized_post(
        renderer, {"type": "text", "text": "Загинула **58-річна жінка** в місті"}
    )

    paragraph = find(blocks, "paragraph")
    assert paragraph["text"][1] == {"type": "bold", "text": "58-річна жінка"}
    assert "**" not in flatten_text(blocks)


def test_a_hidden_block_becomes_a_disclosure(renderer: Renderer) -> None:
    blocks = summarized_post(
        renderer,
        {"type": "text", "text": "Лід."},
        {"type": "hidden", "summary": "Передісторія", "text": "Було раніше."},
    )

    disclosures = [b for b in blocks if b["type"] == "details"]
    # Two: the model's, then the source list.
    assert disclosures[0]["summary"] == "Передісторія"
    assert flatten_text(disclosures[0]["blocks"]) == "Було раніше."


def test_a_disagreement_between_sources_is_labelled(renderer: Renderer) -> None:
    """A reader skimming has to see that the sources do not agree."""
    blocks = summarized_post(
        renderer,
        {"type": "text", "text": "Лід."},
        {"type": "disputed", "text": "поранених від дев'яти до одинадцяти"},
    )

    paragraphs = [b for b in blocks if b["type"] == "paragraph"]
    assert paragraphs[-1]["text"] == [
        {"type": "bold", "text": "Джерела різняться"},
        ": ",
        "поранених від дев'яти до одинадцяти",
    ]


def test_each_version_of_a_dispute_names_the_channels_behind_it(
    renderer: Renderer,
) -> None:
    """The finding is who says what, so the credit has to be in the post.

    An unattributed "sources differ" line tells a reader that somebody is wrong
    without saying who, which is the one thing they cannot check.
    """
    blocks = summarized_post(
        renderer,
        {"type": "text", "text": "Лід."},
        {
            "type": "disputed",
            "claims": [
                {"text": "поранених дев'ятеро", "channels": [VERIFIED]},
                {"text": "поранених одинадцятеро", "channels": [AGGREGATOR]},
            ],
        },
    )

    paragraphs = [b for b in blocks if b["type"] == "paragraph"]
    assert paragraphs[-1]["text"] == {"type": "bold", "text": "Джерела різняться"}

    items = find(blocks, "list")["items"]
    assert [squeeze(flatten_text(item["blocks"])) for item in items] == [
        "поранених дев'ятеро — 📰 RBC_NEWS ⚪",
        "поранених одинадцятеро — 🎭 NEXTA_LIVE",
    ]


def test_a_credit_carries_the_channel_tier(renderer: Renderer) -> None:
    """Who is alone on a claim decides how the claim reads.

    An anonymous aggregator by itself on a detail is the shape a planted item
    takes; a state body by itself on one is the body announcing its own
    business. The reader can only tell them apart if the mark is on the line.
    """
    blocks = summarized_post(
        renderer,
        {"type": "text", "text": "Лід."},
        {
            "type": "attributed",
            "claims": [{"text": "подробиця", "channels": [AGGREGATOR]}],
        },
    )

    item = find(blocks, "list")["items"][0]
    credit = item["blocks"][0]["text"][-1][0]
    assert credit[0] == renderer.channels.group_emoji("grey")
    assert credit[-1]["url"] == "https://t.me/nexta_live/1"


def test_a_credited_channel_links_to_the_post_it_made(renderer: Renderer) -> None:
    blocks = summarized_post(
        renderer,
        {"type": "text", "text": "Лід."},
        {
            "type": "attributed",
            "claims": [{"text": "уламки впали на школу", "channels": [AGGREGATOR]}],
        },
    )

    item = find(blocks, "list")["items"][0]
    link = item["blocks"][0]["text"][-1][0][-1]
    assert link == {
        "type": "url",
        "text": "NEXTA_LIVE",
        "url": "https://t.me/nexta_live/1",
    }


def test_a_claim_whose_channel_left_the_cluster_is_not_printed_bare(
    renderer: Renderer,
) -> None:
    """A raw channel id mid-sentence is noise; better no line than that."""
    blocks = summarized_post(
        renderer,
        {"type": "text", "text": "Лід."},
        {
            "type": "attributed",
            "claims": [{"text": "подробиця", "channels": ["gone_channel"]}],
        },
    )

    assert "подробиця" not in flatten_text(blocks)
    assert "gone_channel" not in flatten_text(blocks)


def test_summary_list_items_become_bullets(renderer: Renderer) -> None:
    blocks = summarized_post(
        renderer,
        {"type": "text", "text": "Лід."},
        {"type": "list", "items": ["Перший факт", "Другий факт"]},
    )

    items = find(blocks, "list")["items"]
    assert [flatten_text(item["blocks"]) for item in items] == [
        "Перший факт",
        "Другий факт",
    ]


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


def test_a_channel_outside_the_issue_is_skipped_quietly(
    renderer: Renderer, caplog: Any
) -> None:
    """No group for an issue means "not republished there", which is not news.

    126 of the 163 channels carry no group for 'war', so a war cluster logged a
    warning for every document that was behaving exactly as configured — and
    buried the warning right above it, which fires when a channel has vanished
    from channels.json altogether.
    """
    cluster = make_cluster(
        [
            make_doc(VERIFIED, "https://t.me/rbc_news/1", pub_time=100),
            make_doc(AGGREGATOR, "https://t.me/nexta_live/1", pub_time=200),
        ]
    )

    # Only rbc_news has a group for "economy" in tests/channels.json.
    with caplog.at_level(logging.DEBUG):
        post = renderer.render_cluster(cluster, "economy")

    assert post is not None
    assert post.blocks is not None
    listing = flatten_text(find(post.blocks, "details"))
    assert "nexta" not in listing.lower()
    assert any("has no group" in record.getMessage() for record in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


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
    assert "Медіа та автори" in listing
    assert "Анонімні" in listing


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


def test_footer_carries_the_reach_alone(renderer: Renderer) -> None:
    """The channel is credited under its text, so repeating it here is noise."""
    cluster = make_cluster(
        [make_doc(VERIFIED, "https://t.me/rbc_news/1", views=12400)]
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    footer = flatten_text(find(post.blocks, "footer"))
    assert "12,4K" in footer
    assert VERIFIED.upper() not in footer


def test_footer_links_the_story_by_id_not_by_headline(renderer: Renderer) -> None:
    """The link has to outlive the headline it was posted with.

    A channel post stays in the archive forever, while the site's canonical URL
    carries a slug transliterated from a headline that gets rewritten while the
    cluster is still taking sources. So the post addresses the story by id and
    lets the site redirect.
    """
    cluster = make_cluster(
        [make_doc(VERIFIED, "https://t.me/rbc_news/1")],
        headline="Заголовок, який ще перепишуть",
    )
    cluster.clid = 41337
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    links = _collect(find(post.blocks, "footer"), lambda b: b.get("type") == "url")
    assert len(links) == 1
    assert links[0]["url"] == "https://news.yuri.ly/n/41337"
    # Not "читати далі": the post already carries the story and the sources, so
    # the label names what only the site has.
    assert links[0]["text"] == "Як поширювалося"


def test_footer_keeps_the_reach_when_there_is_no_story_url(renderer: Renderer) -> None:
    """A cluster is rendered before it is filed, and a deployment may have no
    site at all. Either way the footer degrades to the view count."""
    cluster = make_cluster([make_doc(VERIFIED, "https://t.me/rbc_news/1", views=800)])
    assert cluster.clid is None
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    footer = find(post.blocks, "footer")
    assert "800" in flatten_text(footer)
    assert _collect(footer, lambda b: b.get("type") == "url") == []


def test_the_text_is_credited_to_the_channel_it_came_from(renderer: Renderer) -> None:
    """A reader has to know whose words these are before weighing them."""
    cluster = make_cluster(
        [
            make_doc(VERIFIED, "https://t.me/rbc_news/1", text="Текст новини."),
            make_doc(AGGREGATOR, "https://t.me/nexta_live/1", pub_time=1700000001),
        ]
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    paragraphs = [b for b in post.blocks if b["type"] == "paragraph"]
    # Directly under the text, not somewhere below the source list.
    assert flatten_text(paragraphs[-2]) == "Текст новини."
    credit = paragraphs[-1]
    assert squeeze(flatten_text(credit)) == f"— {VERIFIED.upper()}"
    assert _collect(credit, lambda b: b.get("type") == "url")[0]["url"] == (
        "https://t.me/rbc_news/1"
    )


def test_a_post_without_an_llm_headline_is_still_credited(renderer: Renderer) -> None:
    """A one-sentence post has no heading, so the credit must not hang off one."""
    cluster = make_cluster(
        [make_doc(VERIFIED, "https://t.me/rbc_news/1", text="Одне речення.")],
        headline=None,
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    paragraphs = [b for b in post.blocks if b["type"] == "paragraph"]
    assert squeeze(flatten_text(paragraphs[-1])) == f"— {VERIFIED.upper()}"


def test_channel_titles_need_no_escaping(renderer: Renderer) -> None:
    doc = make_doc(VERIFIED, "https://t.me/rbc_news/1")
    doc.channel_title = "Кіно & Театр <18>"
    post = renderer.render_cluster(make_cluster([doc]), "main")

    assert post is not None
    assert post.blocks is not None
    # Blocks carry text as data, so nothing has to be escaped anywhere it appears.
    assert "Кіно & Театр <18>" in flatten_text(post.blocks)


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
        summary=make_summary({"type": "text", "text": "Зведений текст."}),
    )
    post = legacy_renderer.render_cluster(cluster, "main")

    assert post is not None
    assert not post.is_rich
    assert post.text is not None
    # The rollback path stays what it was: one channel's text, quoted.
    assert "Основний текст новини" in post.text
    assert "Зведений текст" not in post.text
    # The legacy template needs its channel links as markup.
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


def test_a_photo_and_a_video_share_one_slideshow(renderer: Renderer) -> None:
    """Both kinds are chosen together, so both go into the same swipeable group.

    The renderer used to return a video *instead of* the photos, which meant a
    well-filmed event lost every picture the other channels had.
    """
    cluster = make_cluster(
        [
            make_doc(
                VERIFIED,
                "https://t.me/rbc_news/1",
                videos=("https://example.com/v.mp4",),
                pub_time=100,
            ),
            make_doc(
                AGGREGATOR,
                "https://t.me/nexta_live/1",
                images=("b",),
                pub_time=200,
                embedded_images=[{"url": "https://example.com/b.jpg"}],
            ),
        ],
    )
    post = renderer.render_cluster(cluster, "main")

    assert post is not None
    assert post.blocks is not None
    slideshow = find(post.blocks, "slideshow")
    assert [b["type"] for b in slideshow["blocks"]] == ["video", "photo"]


def test_the_legacy_format_carries_the_media_the_cluster_chose(
    renderer: Renderer,
) -> None:
    """Both formats show the same attachments, in the same order.

    They used to disagree: the rich renderer preferred video, the legacy client
    preferred photos, so the config value decided what a reader saw.
    """
    cluster = make_cluster(
        [
            make_doc(
                VERIFIED,
                "https://t.me/rbc_news/1",
                videos=("https://example.com/v.mp4",),
                pub_time=100,
            ),
            make_doc(
                AGGREGATOR,
                "https://t.me/nexta_live/1",
                images=("b",),
                pub_time=200,
                embedded_images=[{"url": "https://example.com/b.jpg"}],
            ),
        ],
    )
    post = renderer.render_cluster(cluster, "main", post_format="legacy")

    assert post is not None
    assert [(item.type, item.url) for item in post.media] == [
        ("video", "https://example.com/v.mp4"),
        ("photo", "https://example.com/b.jpg"),
    ]
