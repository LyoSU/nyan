"""How the editor's decisions show in a post: levels, update lines, the storyline
links and the quote in a reply. Every part sits behind its own config flag."""

import json
from typing import Any

from nyan.channels import Channels
from nyan.client import TelegramClient
from nyan import rich as blocks_of
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
# The post itself went out that morning, 11.11 11:00 Kyiv time.
PUBLISHED = 1794387600


def story(headline: str = "На Одещині загинув чоловік, троє людей постраждали") -> Any:
    cluster = make_cluster(
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
    cluster.create_time = PUBLISHED
    return cluster


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
    line = post.blocks[1]["text"]
    assert line == {
        "type": "italic",
        "text": "оновлено о 14:30: у лікарні померли двоє постраждалих",
    }
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
    assert [row[0]["text"] for row in table["cells"][1:]] == ["11:40", "14:30"]
    assert post.blocks[1]["text"]["text"].endswith("двоє постраждалих"), "latest on top"


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
    assert post is not None and post.blocks is not None
    more = json.dumps(next(b for b in post.blocks if b["type"] == "list"), ensure_ascii=False)
    assert "date_time" not in more, "a client would say the date in its own language"
    assert renderer.render_more_on_this(cluster.storyline_posts, "war", PUBLISHED) == [], (
        "no address for a war channel, so no links that would lead to main"
    )


def test_breaking_has_a_label_and_rings(tmp_path: Any, channels: Channels) -> None:
    cluster = story("Сили оборони вночі збили 144 ворожі цілі")
    cluster.level = "breaking"

    post = make_renderer(tmp_path, channels, **ALL_ON).render_cluster(cluster, "main")

    assert post is not None and post.blocks is not None
    assert post.blocks[0] == {"type": "paragraph", "text": {"type": "bold", "text": "🔴 Терміново"}}
    assert post.blocks[1]["type"] == "heading", "the label is its own line above the headline"
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


def test_dates_are_ukrainian_text_and_only_the_headline_is_a_link(
    tmp_path: Any, channels: Channels
) -> None:
    cluster = story()
    earlier = {"clid": 1, "message_id": 43816, "issue": "main", "headline": "Обмін полоненими"}
    cluster.storyline_posts = [{**earlier, "time": 1794300000}]

    post = make_renderer(tmp_path, channels, **ALL_ON).render_cluster(cluster, "main")

    assert post is not None and post.blocks is not None
    item = next(b for b in post.blocks if b["type"] == "list")["items"][0]["blocks"][0]
    assert item["text"] == [
        "10 листопада, 10:40 — ",
        {"type": "url", "text": "Обмін полоненими", "url": "https://t.me/UAliveNews/43816"},
    ]


def test_an_update_on_a_later_day_names_the_day(tmp_path: Any, channels: Channels) -> None:
    cluster = story()
    next_morning = {"time": 1794467400, "text": "загиблих троє"}
    cluster.updates = [DEATHS, next_morning]

    post = make_renderer(tmp_path, channels, **ALL_ON).render_cluster(cluster, "main")

    assert post is not None and post.blocks is not None
    assert post.blocks[1]["text"]["text"] == "оновлено 12 листопада о 09:10: загиблих троє"
    table = next(b for b in post.blocks if b["type"] == "table")
    assert [row[0]["text"] for row in table["cells"][1:]] == ["14:30", "12 листопада, 09:10"]


def test_a_date_from_another_year_carries_the_year(tmp_path: Any, channels: Channels) -> None:
    renderer = make_renderer(tmp_path, channels, **ALL_ON)

    assert renderer.say_moment(1794300000, PUBLISHED + 365 * 86400) == "10 листопада 2026, 10:40"


def test_an_update_rewrites_the_headline_and_keeps_its_line(
    monkeypatch: Any, tmp_path: Any, channels: Channels
) -> None:
    """11 November, Chornomorsk: "one dead" in the headline under a line saying
    three would contradict itself; the update rewrites the whole post."""
    from nyan.clusters import Cluster
    from tests.test_editor import NOW, _Client, _answer, _daemon, _doc, _post, _posted

    daemon = _daemon()
    daemon.renderer = make_renderer(tmp_path, channels, **ALL_ON)
    edited: list[Any] = []

    class Client(_Client):
        def update_post(self, message: Any, post: Any) -> bool:
            edited.append(post)
            return super().update_post(message, post)

    daemon.client = Client()  # type: ignore[assignment]
    parent = _post(66325, "Удар по ринку в Чорноморську: одна людина загинула", 1, [1.0, 0.0])
    # Channels the renderer knows, or it has nothing to show.
    parent.docs = []
    parent.add(_doc("https://t.me/rbc_news/1", "Удар по ринку: одна загибла", 80, [1.0, 0.0]))
    parent.add(_doc("https://t.me/nexta_live/1", "Одна загибла на ринку", 75, [1.0, 0.0]))
    headline = parent.saved_analysis["headline"]
    parent.saved_analysis = {"headline": headline, "generation": parent.generation}
    parent.saved_hash = parent.hash
    generation_before = parent.generation
    story = Cluster()
    # Same two channels: coverage did not grow, so only the update can force
    # the rewrite.
    story.add(_doc("https://t.me/rbc_news/2", "Загиблих стало троє", 5, [1.0, 0.0]))
    _answer(
        monkeypatch,
        {"relation": "update", "post": 1, "decision": "publish", "change": "загиблих стало троє"},
    )
    monkeypatch.setattr("nyan.clusters.get_current_ts", lambda: NOW)
    rewrite = {
        "headline": "У Чорноморську загинули троє людей",
        "blocks": [{"type": "text", "text": "У лікарні померли двоє постраждалих."}],
    }
    monkeypatch.setattr(
        "nyan.clusters.openai_completion",
        lambda **kwargs: json.dumps(rewrite, ensure_ascii=False),
    )
    monkeypatch.setattr("nyan.clusters._ANALYSIS_CACHE", {})

    daemon.send_cluster(
        story,
        "main",
        _posted(parent),
        None,
        None,
    )

    assert parent.generation == generation_before
    assert len(edited) == 1 and edited[0].blocks is not None
    blocks = edited[0].blocks
    assert blocks[0]["type"] == "heading"
    assert "У Чорноморську загинули троє людей" in json.dumps(blocks[0], ensure_ascii=False)
    assert blocks[1]["text"] == {"type": "italic", "text": "оновлено о 10:00: загиблих стало троє"}


def test_a_minor_post_keeps_its_photo_and_drops_what_single_channels_add(
    tmp_path: Any, channels: Channels
) -> None:
    """The NBU coin: two lines of text and a square photo that filled the screen."""
    renderer = make_renderer(tmp_path, channels, **ALL_ON)
    photo = blocks_of.photo("https://cdn4.telegram-cdn.org/file/coin.jpg")
    blocks = [
        heading(blocks_of.bold("НБУ випустив пам’ятну монету до року Кози")),
        paragraph("Монета має номінал 5 гривень."),
        blocks_of.slideshow(photo, photo),
        paragraph(blocks_of.bold("Пишуть окремі джерела")),
        blocks_of.bullet_list([paragraph("Придбати монету можна в інтернет-магазині НБУ")]),
        paragraph("Тираж — 80 тисяч штук."),
        blocks_of.details("23 джерела", paragraph("…")),
    ]

    minor = renderer.render_level(blocks, "minor")

    assert [b["type"] for b in minor] == ["paragraph", "paragraph", "details", "details"]
    more = minor[2]
    assert more["summary"] == "Докладніше"
    assert [b["type"] for b in more["blocks"]] == ["slideshow", "paragraph"], "photo first"
    assert "Пишуть окремі джерела" not in json.dumps(minor, ensure_ascii=False)


def test_a_long_legend_takes_two_short_lines(tmp_path: Any, channels: Channels) -> None:
    on = make_renderer(tmp_path, channels, presentation={"short_legend": True})
    off = make_renderer(tmp_path, channels)
    parts = ["👥 Організація", "👤 Персона", "⚪ Білий список ІМІ", "🔍 Є розслідування"]

    lines = on.render_short_legend(parts, kind_count=2)

    assert [line["text"]["text"] for line in lines] == [
        "👥 Організація · 👤 Персона",
        "⚪ Білий список ІМІ · 🔍 Є розслідування",
    ]
    assert on.render_short_legend(parts[2:3] + parts[3:] + ["x"], 0)[0]["text"]["text"] == (
        "⚪ Білий список ІМІ · 🔍 Є розслідування"
    )
    assert off.short_legend is False


def test_a_channels_own_quote_marks_do_not_go_into_our_quote(
    tmp_path: Any, channels: Channels
) -> None:
    """13 November, "Труха⚡️Київ": the channel's text began every line with "> "."""
    from nyan.renderer import without_quote_marks

    text = [
        "> У Подільському районі неповнолітній прийшов до знайомого.\n>> На крики ",
        {"type": "bold", "text": "вибіг сусід"},
        " і викликав медиків > не маркер.\n  > Нападнику повідомили про підозру.",
    ]

    assert without_quote_marks(text) == [
        "У Подільському районі неповнолітній прийшов до знайомого.\nНа крики ",
        {"type": "bold", "text": "вибіг сусід"},
        " і викликав медиків > не маркер.\nНападнику повідомили про підозру.",
    ]
    assert without_quote_marks({"type": "bold", "text": "> Х"}) == {"type": "bold", "text": "Х"}
