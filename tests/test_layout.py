"""The author's own layout of a post: blocks it may choose, where the media goes,
and the old template kept for Telegram to fall back on. Everything here is
behind `presentation.layout` and off by default."""

import json
from typing import Any

import pytest

import nyan.clusters
from nyan import media_brief
from nyan.channels import Channels
from nyan.client import TelegramClient
from nyan.clusters import Cluster
from nyan.document import Document
from nyan.media import MEDIA_PHOTO, MEDIA_VIDEO, MediaItem
from nyan.rich import RenderedPost, heading, paragraph
from nyan.summary import TABLE, parse_summary
from tests.test_client import FakeResponse, record_calls
from tests.test_client import client as client
from tests.test_renderer import (
    AGGREGATOR,
    VERIFIED,
    make_cluster,
    make_doc,
    make_renderer,
    make_summary,
)

LAYOUT = {"presentation": {"layout": True}}

# 14 November, fuel prices: numbers a reader compares.
PRICES = {
    "type": "table",
    "header": ["Мережа", "А-95", "Дизель"],
    "rows": [["ОККО", "62,99", "59,99"], ["WOG", 62.49, "59,49"]],
}


def blocks_of(raw: dict[str, Any], layout: bool = True) -> list[str]:
    summary = parse_summary(make_summary(*raw["blocks"]), layout=layout)
    return [block.type for block in summary.blocks]


def test_a_table_is_kept_only_when_the_author_was_offered_one() -> None:
    raw = {"blocks": [{"type": "text", "text": "Ціни знизилися."}, PRICES]}

    assert blocks_of(raw) == ["text", TABLE]
    assert blocks_of(raw, layout=False) == ["text"], "the old template never had one"


@pytest.mark.parametrize(
    "table",
    [
        {"type": "table", "rows": [["a", "b", "c", "d", "e"]]},
        {"type": "table", "header": ["a", "b"], "rows": [["a", "b"], ["c"]]},
        {"type": "table", "rows": [["a", "b"]] * 11},
    ],
    ids=["too wide for a phone", "ragged", "too long"],
)
def test_a_table_that_would_not_read_is_dropped(table: dict[str, Any]) -> None:
    assert blocks_of({"blocks": [{"type": "text", "text": "Лід."}, table]}) == ["text"]


def laid_out(tmp_path: Any, channels: Channels, **analysis: Any) -> RenderedPost:
    cluster = make_cluster(
        [
            make_doc(
                VERIFIED,
                "https://t.me/rbc_news/1",
                images=("a",),
                embedded_images=[{"url": "https://example.com/a.jpg"}],
            ),
            make_doc(
                AGGREGATOR,
                "https://t.me/nexta_live/1",
                images=("b",),
                pub_time=1700000100,
                embedded_images=[{"url": "https://example.com/b.jpg"}],
            ),
        ],
        summary=make_summary(
            {"type": "text", "text": "Ціни на пальне знизилися."},
            PRICES | {"rows": [["ОККО", "62,99", "59,99"]]},
            {"type": "quote", "text": "Довга заява. " * 20, "author": "Мінекономіки", "fold": True},
        ),
    )
    cluster.saved_analysis.update(analysis)
    post = make_renderer(tmp_path, channels, **LAYOUT).render_cluster(cluster, "main")
    assert post is not None and post.blocks is not None
    return post


def test_the_authors_blocks_become_their_telegram_blocks(tmp_path: Any, channels: Channels) -> None:
    post = laid_out(tmp_path, channels)

    kinds = [block["type"] for block in post.blocks or []]
    assert "table" in kinds and "expandable_blockquote" in kinds
    assert kinds.index("slideshow") < kinds.index("table"), "media after the lede, as before"


def test_a_laid_out_post_carries_the_old_template_of_itself(
    tmp_path: Any, channels: Channels
) -> None:
    post = laid_out(tmp_path, channels)

    assert post.fallback is not None
    kinds = [block["type"] for block in post.fallback]
    assert "table" not in kinds and "expandable_blockquote" not in kinds
    assert "blockquote" in kinds and "list" in kinds, "the same facts in the old blocks"


def test_the_author_folds_the_media(tmp_path: Any, channels: Channels) -> None:
    post = laid_out(tmp_path, channels, media={"place": "fold"})

    kinds = [block["type"] for block in post.blocks or []]
    assert "slideshow" not in kinds
    folded = next(b for b in post.blocks or [] if b.get("summary") == "Фото й відео")
    assert folded["blocks"][0]["type"] == "slideshow"


def test_the_author_leaves_the_media_out(tmp_path: Any, channels: Channels) -> None:
    post = laid_out(tmp_path, channels, media={"place": "none"})

    assert "example.com" not in json.dumps(post.blocks)


def test_the_author_picks_the_picture_that_leads(tmp_path: Any, channels: Channels) -> None:
    post = laid_out(tmp_path, channels, media={"place": "show", "first": 2})

    slideshow = next(b for b in post.blocks or [] if b["type"] == "slideshow")
    assert slideshow["blocks"][0]["photo"]["media"].endswith("b.jpg")


def test_with_the_flag_off_the_media_decision_is_ignored(tmp_path: Any, channels: Channels) -> None:
    cluster = make_cluster(
        [
            make_doc(
                VERIFIED,
                "https://t.me/rbc_news/1",
                images=("a",),
                embedded_images=[{"url": "https://example.com/a.jpg"}],
            )
        ],
        summary=make_summary({"type": "text", "text": "Лід."}),
    )
    plain = make_renderer(tmp_path, channels).render_cluster(cluster, "main")
    cluster.saved_analysis["media"] = {"place": "none"}

    again = make_renderer(tmp_path, channels).render_cluster(cluster, "main")

    assert plain is not None and again is not None
    assert again.blocks == plain.blocks and again.fallback is None


# What the author is told about the media.


def video_doc() -> Document:
    return make_doc(
        VERIFIED,
        "https://t.me/rbc_news/1",
        videos=("https://example.com/v.mp4", "https://example.com/w.mp4"),
    )


def test_a_video_is_named_with_its_length_and_never_taken_for_a_photo() -> None:
    doc = video_doc()
    doc.video_thumbs = ("https://example.com/v.jpg", "")
    doc.video_durations = (40, 75)
    media = [
        MediaItem(type=MEDIA_VIDEO, url="https://example.com/v.mp4"),
        MediaItem(type=MEDIA_VIDEO, url="https://example.com/w.mp4"),
        MediaItem(type=MEDIA_PHOTO, url="https://example.com/p.jpg"),
    ]

    described = media_brief.describe(media, [doc])

    assert [item.label for item in described] == ["відео, 0:40", "відео, 1:15, без прев'ю", "фото"]
    assert described[0].still_url == "https://example.com/v.jpg"
    note = media_brief.media_note(described, shown=1)
    assert "1. відео, 0:40" in note and "Перші 1 з них прикладено" in note


def test_the_pictures_stop_at_the_first_one_without_a_still(monkeypatch: Any) -> None:
    """The n-th image has to be the n-th line, or the author describes the wrong one."""
    monkeypatch.setattr(media_brief, "fetch_still", lambda url: f"data:{url}")
    attachments = [
        media_brief.Attachment("фото", "a"),
        media_brief.Attachment("відео, 0:40, без прев'ю"),
        media_brief.Attachment("фото", "c"),
    ]

    assert media_brief.stills(attachments) == ["data:a"]


def test_the_pictures_go_after_the_text_at_low_detail() -> None:
    message = media_brief.with_stills({"role": "user", "content": "Текст"}, ["data:x"])

    assert message["content"][0] == {"type": "text", "text": "Текст"}
    assert message["content"][1]["image_url"] == {"url": "data:x", "detail": "low"}


def test_a_media_decision_out_of_range_is_no_lead() -> None:
    assert media_brief.parse_media({"place": "fold", "first": 9}, 2) == {"place": "fold"}
    assert media_brief.parse_media({"place": "maybe"}, 2) is None
    assert media_brief.parse_media({"place": "show"}, 0) is None


# The author's call with the flag on.


def two_channel_story() -> Cluster:
    cluster = Cluster()
    cluster.add(make_doc(VERIFIED, "https://t.me/rbc_news/1", images=("a",)))
    cluster.add(make_doc(AGGREGATOR, "https://t.me/nexta_live/1", pub_time=1700000100))
    cluster.saved_annotation_doc = cluster.docs[0]
    return cluster


def test_the_author_is_given_the_layout_rules_and_the_pictures(monkeypatch: Any) -> None:
    calls: list[Any] = []

    def author(**kwargs: Any) -> str:
        calls.append(kwargs)
        return json.dumps(
            {
                "headline": "Ціни на пальне знизилися",
                "blocks": [{"type": "text", "text": "Лід."}, PRICES],
                "media": {"place": "fold", "notes": [{"what": "логотип", "kind": "filler"}]},
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(nyan.clusters, "openai_completion", author)
    monkeypatch.setattr(nyan.clusters, "_ANALYSIS_CACHE", {})
    monkeypatch.setattr(media_brief, "fetch_still", lambda url: "data:still")
    monkeypatch.setattr(
        Cluster, "media", property(lambda self: [MediaItem(type=MEDIA_PHOTO, url="u")])
    )
    nyan.clusters.configure_author(layout=True, vision=True)
    try:
        analysis = two_channel_story().analysis
    finally:
        nyan.clusters.configure_author()

    system, user = calls[0]["messages"]
    assert "Верстка." in system["content"]
    assert user["content"][0]["text"].rstrip().endswith("1. фото")
    assert user["content"][1]["type"] == "image_url"
    assert analysis["media"]["place"] == "fold"
    assert [block["type"] for block in analysis["summary"]["blocks"]] == ["text", "table"]


def test_a_refused_picture_costs_the_picture_not_the_post(monkeypatch: Any) -> None:
    calls: list[Any] = []

    def author(**kwargs: Any) -> str:
        calls.append(kwargs)
        if isinstance(kwargs["messages"][1]["content"], list):
            raise RuntimeError("image input is not supported")
        return json.dumps({"headline": "Заголовок", "blocks": [{"type": "text", "text": "Лід."}]})

    monkeypatch.setattr(nyan.clusters, "openai_completion", author)
    monkeypatch.setattr(nyan.clusters, "_ANALYSIS_CACHE", {})
    monkeypatch.setattr(media_brief, "fetch_still", lambda url: "data:still")
    monkeypatch.setattr(
        Cluster, "media", property(lambda self: [MediaItem(type=MEDIA_PHOTO, url="u")])
    )
    nyan.clusters.configure_author(layout=True, vision=True)
    try:
        analysis = two_channel_story().analysis
    finally:
        nyan.clusters.configure_author()

    assert analysis["headline"] == "Заголовок" and len(calls) == 2


def test_with_the_flag_off_the_author_is_asked_what_it_always_was(monkeypatch: Any) -> None:
    calls: list[Any] = []

    def author(**kwargs: Any) -> str:
        calls.append(kwargs)
        return json.dumps({"headline": "Заголовок", "blocks": [{"type": "text", "text": "Лід."}]})

    monkeypatch.setattr(nyan.clusters, "openai_completion", author)
    monkeypatch.setattr(nyan.clusters, "_ANALYSIS_CACHE", {})

    assert two_channel_story().analysis["headline"] == "Заголовок"
    system, user = calls[0]["messages"]
    assert "Верстка." not in system["content"] and "МЕДІА" not in user["content"]


# Telegram refusing the layout.


def test_a_refused_layout_goes_out_in_the_old_template(client: TelegramClient) -> None:
    rejected = FakeResponse(400, {"ok": False, "description": "Bad Request: TABLE_INVALID"})
    calls = record_calls(client, [rejected, FakeResponse()])
    post = RenderedPost(
        blocks=[heading("Заголовок"), {"type": "table", "cells": []}],
        fallback=[heading("Заголовок"), paragraph("Текст")],
    )

    message = client.send_post(post, "main")

    assert message is not None
    sent = json.loads(calls[-1][1]["rich_message"])
    assert [block["type"] for block in sent["blocks"]] == ["heading", "paragraph"]
