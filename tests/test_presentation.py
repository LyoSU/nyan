"""How the editor's decisions show in a post: levels, update lines, the storyline
links and the quote in a reply. Every part sits behind its own config flag."""

import json
from typing import Any

from nyan.channels import Channels
from nyan.client import TelegramClient
from nyan.rich import RenderedPost, heading, paragraph
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

ALL_ON = {
    "presentation": {
        "levels": True,
        "update_line": True,
        "more_on_this": True,
        "quote_parent": True,
        "channel_urls": {"main": "https://t.me/UAliveNews"},
    }
}

# 11 November, the hospital in Odesa region: one update to an evening post.
DEATHS = {"time": 1794400200, "text": "у лікарні померли двоє постраждалих"}


def story(headline: str = "На Одещині загинув чоловік, троє людей постраждали") -> Any:
    return make_cluster(
        [
            make_doc(VERIFIED, "https://t.me/rbc_news/1"),
            make_doc(AGGREGATOR, "https://t.me/nexta_live/1"),
        ],
        headline=headline,
        summary=make_summary(
            {"type": "text", "text": "Росія вдарила по Одещині вночі."},
            {"type": "text", "text": "На місці працюють рятувальники."},
            headline=headline,
        ),
    )


def types(post: RenderedPost | None) -> list[str]:
    assert post is not None and post.blocks is not None
    return [block["type"] for block in post.blocks]


def test_with_every_flag_off_the_post_is_the_post_it_was(tmp_path: Any, channels: Channels) -> None:
    cluster = story()
    plain = make_renderer(tmp_path, channels).render_cluster(cluster, "main")
    cluster.level = "breaking"
    cluster.updates = [DEATHS]
    cluster.storyline_posts = [
        {"clid": 1, "message_id": 5, "issue": "main", "time": 1794390000, "headline": "Х"}
    ]
    cluster.reply_to_clid, cluster.reply_to_headline = 1, "Х"

    again = make_renderer(tmp_path, channels).render_cluster(cluster, "main")

    assert again is not None and plain is not None
    assert again.blocks == plain.blocks
    assert again.reply_quote is None and again.notify is False


def test_an_update_is_said_under_the_headline(tmp_path: Any, channels: Channels) -> None:
    cluster = story()
    cluster.updates = [DEATHS]

    post = make_renderer(tmp_path, channels, **ALL_ON).render_cluster(cluster, "main")

    assert types(post)[:2] == ["heading", "paragraph"]
    assert post is not None and post.blocks is not None
    line = post.blocks[1]["text"]["text"]
    assert line[0] == "оновлено " and line[1]["type"] == "date_time"
    assert line[2] == ": у лікарні померли двоє постраждалих"
    assert "table" not in types(post), "one update is a line, not a history"


def test_several_updates_add_a_table_before_the_sources(tmp_path: Any, channels: Channels) -> None:
    cluster = story()
    cluster.updates = [{"time": 1794390000, "text": "загинула людина"}, DEATHS]

    post = make_renderer(tmp_path, channels, **ALL_ON).render_cluster(cluster, "main")

    kinds = types(post)
    assert kinds.index("table") < len(kinds) - 3
    assert post is not None and post.blocks is not None
    table = post.blocks[kinds.index("table")]
    assert [row[1]["text"] for row in table["cells"][1:]] == [
        "загинула людина",
        "у лікарні померли двоє постраждалих",
    ]
    assert post.blocks[1]["text"]["text"][2].endswith("двоє постраждалих"), "latest on top"


def test_more_on_this_links_into_the_readers_own_channel(tmp_path: Any, channels: Channels) -> None:
    cluster = story("Влучання дрона в Київському районі Харкова")
    cluster.storyline_posts = [
        {
            "clid": 66119,
            "message_id": 6119,
            "issue": "main",
            "time": 1794300000,
            "headline": "Дрон влучив у дах дев’ятиповерхівки в Харкові",
        }
    ]
    renderer = make_renderer(tmp_path, channels, **ALL_ON)

    post = renderer.render_cluster(cluster, "main")

    text = json.dumps(post.blocks if post else None, ensure_ascii=False)
    assert "Ще про це" in text and "https://t.me/UAliveNews/6119" in text
    assert renderer.render_more_on_this(cluster.storyline_posts, "war") == [], (
        "no address for a war channel, so no links that would lead to main"
    )


def test_breaking_has_a_label_and_rings(tmp_path: Any, channels: Channels) -> None:
    cluster = story("Сили оборони вночі збили 144 ворожі цілі")
    cluster.level = "breaking"

    post = make_renderer(tmp_path, channels, **ALL_ON).render_cluster(cluster, "main")

    assert post is not None and post.blocks is not None
    assert "Терміново" in json.dumps(post.blocks[0], ensure_ascii=False)
    assert post.notify is True


def test_a_minor_post_takes_two_lines_of_the_feed(tmp_path: Any, channels: Channels) -> None:
    cluster = story("НБУ випустив пам’ятну монету до року Кози")
    cluster.level = "minor"

    post = make_renderer(tmp_path, channels, **ALL_ON).render_cluster(cluster, "main")

    kinds = types(post)
    assert "heading" not in kinds
    assert kinds[:3] == ["paragraph", "paragraph", "details"], "headline, lede, the rest folded"
    assert post is not None and post.notify is False


def test_a_minor_quoted_post_folds_the_quote(tmp_path: Any, channels: Channels) -> None:
    long_text = "Ціни на пальне на АЗС змінилися. " * 12
    cluster = make_cluster(
        [make_doc(VERIFIED, "https://t.me/rbc_news/1", text=long_text)],
        headline="Бензин дешевшає, автогаз дорожчає",
    )
    cluster.level = "minor"

    post = make_renderer(tmp_path, channels, **ALL_ON).render_cluster(cluster, "main")

    assert post is not None and post.blocks is not None
    quote = [b for b in post.blocks if b["type"] == "expandable_blockquote"]
    assert quote and "Ціни на пальне" in json.dumps(quote[0]["text"], ensure_ascii=False)
    assert "credit" in quote[0]


def test_a_follow_up_quotes_the_headline_it_continues(tmp_path: Any, channels: Channels) -> None:
    cluster = story("Кількість загиблих у Чорноморську зросла до трьох")
    cluster.reply_to_clid = 66325
    cluster.reply_to_headline = "Удар по ринку в Чорноморську: одна людина загинула"

    post = make_renderer(tmp_path, channels, **ALL_ON).render_cluster(cluster, "main")

    assert post is not None
    assert post.reply_quote == [
        {"type": "bold", "text": "Удар по ринку в Чорноморську: одна людина загинула"}
    ]


def rich(quote: Any = None, notify: bool = False) -> RenderedPost:
    return RenderedPost(
        blocks=[heading("Заголовок"), paragraph("Текст")], reply_quote=quote, notify=notify
    )


def test_the_quote_goes_into_reply_parameters(client: TelegramClient) -> None:
    calls = record_calls(client, [FakeResponse()])
    quote = [{"type": "bold", "text": "Удар по ринку в Чорноморську"}]

    client.send_post(rich(quote, notify=True), "main", reply_to=42)

    params = calls[0][1]
    assert json.loads(params["reply_parameters"])["quote_rich_texts"] == quote
    assert params["disable_notification"] is False


def test_a_quote_the_parent_no_longer_has_is_dropped_not_the_reply(
    client: TelegramClient,
) -> None:
    rejected = FakeResponse(400, {"ok": False, "description": "Bad Request: QUOTE_TEXT_INVALID"})
    calls = record_calls(client, [rejected, FakeResponse()])

    message = client.send_post(rich([{"type": "bold", "text": "Старий заголовок"}]), "main", 42)

    assert message is not None
    second = json.loads(calls[1][1]["reply_parameters"])
    assert second["message_id"] == 42 and "quote_rich_texts" not in second


def test_without_a_reply_there_is_no_quote_and_no_sound(client: TelegramClient) -> None:
    calls = record_calls(client, [FakeResponse()])

    client.send_post(rich([{"type": "bold", "text": "Х"}]), "main")

    assert "reply_parameters" not in calls[0][1]
    assert calls[0][1]["disable_notification"] is True
