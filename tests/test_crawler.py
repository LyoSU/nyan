"""Tests for the crawler: parsing helpers, the spider's entry points and pipelines.

The spider is exercised against a saved fragment of t.me markup rather than the
live site, so a Telegram layout change shows up as a failing test here.
"""

import asyncio
import json
import logging
from datetime import datetime, UTC
from typing import Any

import pytest
from parsel import Selector
from pymongo.errors import OperationFailure
from scrapy.exceptions import DropItem
from scrapy.http import HtmlResponse, Request
from twisted.python.failure import Failure

from crawler.pipelines import (
    ChannelStatsPipeline,
    JsonlPipeline,
    MongoPipeline,
    PostHistoryPipeline,
    as_record,
    keep_history,
    text_hash,
)
from crawler.spiders.telegram import (
    CHANNEL_STATS_KIND,
    DEFAULT_RECRAWL_TIME,
    MAX_PAGES_PER_CHANNEL,
    TelegramSpider,
    extract_images,
    extract_mentions,
    get_current_ts,
    parse_post_url,
    process_counter,
    process_views,
    to_timestamp,
)


def as_datetime_attr(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")


# Anchored to the moment the suite runs, not to a calendar date.
#
# These used to be literal 2026-07-25 timestamps, which made the paging test a
# time bomb: the spider's window is the last 24 hours counted from now, so the
# day after that date the fixture posts fell outside it, the spider correctly
# stopped paging, and the test that checks paging happens began to fail with no
# change to the crawler at all.
POST_TIMES = tuple(get_current_ts() - hours * 3600 for hours in (1, 2, 3))

# A cut-down copy of the structure t.me/s/<channel> serves: the channel header
# with its counters, then posts with text, a photo, a view count and a time.
CHANNEL_HTML = f"""
<body><main><div>
<div class="tgme_channel_info">
  <div class="tgme_channel_info_header">
    <div class="tgme_channel_info_header_title"><span dir="auto">UA News</span></div>
  </div>
  <div class="tgme_channel_info_counters">
    <div class="tgme_channel_info_counter">
      <span class="counter_value">713K</span> <span class="counter_type">subscribers</span>
    </div>
    <div class="tgme_channel_info_counter">
      <span class="counter_value">108K</span> <span class="counter_type">photos</span>
    </div>
  </div>
</div>
<section class="tgme_channel_history"><div>
  <div class="tgme_widget_message" data-post="uanews/100">
    <div class="tgme_widget_message_bubble">
      <div class="tgme_widget_message_text">
        Новина про подію, пише @suspilnenews.
        <a href="https://example.com/story">джерело</a>
      </div>
      <a class="tgme_widget_message_photo_wrap"
         style="width:100px;background-image:url('https://cdn.telesco.pe/a.jpg')"></a>
      <a class="tgme_widget_message_photo_wrap"
         style="background-image:url('https://cdn.telesco.pe/a.jpg');width:50px"></a>
      <span class="tgme_widget_message_views">12.5K</span>
      <time class="time" datetime="{as_datetime_attr(POST_TIMES[0])}"></time>
    </div>
  </div>
  <div class="tgme_widget_message" data-post="uanews/101">
    <div class="tgme_widget_message_bubble">
      <div class="media_supported_cont">
        <div class="tgme_widget_message_text">Пост із медіа-обгорткою</div>
      </div>
      <video class="tgme_widget_message_video" src="https://cdn.telesco.pe/v.mp4"></video>
      <span class="tgme_widget_message_meta">
        <span class="tgme_widget_message_views">900</span>
        <span class="tgme_widget_message_meta_edited">edited</span>
        <time class="time" datetime="{as_datetime_attr(POST_TIMES[1])}"></time>
      </span>
    </div>
  </div>
  <div class="tgme_widget_message" data-post="uanews/102">
    <div class="tgme_widget_message_bubble">
      <div class="tgme_widget_message_text">Пост без переглядів (службовий)</div>
      <time class="time" datetime="{as_datetime_attr(POST_TIMES[2])}"></time>
    </div>
  </div>
</div></section></div></main></body>
"""


@pytest.fixture
def spider(tmp_path: Any) -> TelegramSpider:
    channels = tmp_path / "channels.json"
    channels.write_text(json.dumps({"channels": [{"name": "uanews"}]}))
    fetch_times = tmp_path / "fetch_times.json"
    fetch_times.write_text("{}")
    return TelegramSpider(
        channels_file=str(channels),
        fetch_times=str(fetch_times),
        hours="24",
    )


def channel_response(html: str = CHANNEL_HTML) -> HtmlResponse:
    url = "https://t.me/s/uanews"
    return HtmlResponse(
        url=url, body=html.encode("utf-8"), encoding="utf-8", request=Request(url)
    )


def test_views_are_parsed_from_every_form_telegram_uses() -> None:
    assert process_views("1234") == 1234
    assert process_views("12.5K") == 12500
    assert process_views("1.2M") == 1200000
    assert process_views(" 900 ") == 900
    assert process_views(None) == 0
    assert process_views("nonsense") == 0


def test_post_urls_are_split_into_channel_and_id() -> None:
    assert parse_post_url("https://t.me/UaNews/123?embed=1") == {
        "url": "https://t.me/uanews/123",
        "channel_id": "uanews",
        "post_id": 123,
    }


def test_timestamps_are_read_as_utc() -> None:
    expected = int(datetime(2026, 7, 25, 9, 0, tzinfo=UTC).timestamp())
    assert to_timestamp("2026-07-25T09:00:00+00:00") == expected
    assert to_timestamp("") == 0
    assert to_timestamp("not a date") == 0


def test_the_spider_yields_a_request_per_channel(spider: TelegramSpider) -> None:
    """Scrapy 2.13 renamed this entry point and 2.17 dropped the old name.

    A spider that only defines start_requests() crawls nothing on 2.17 without
    reporting an error, so both have to work.
    """
    legacy = list(spider.start_requests())

    assert [r.url for r in legacy] == ["https://t.me/s/uanews"]


def test_the_modern_entry_point_yields_the_same(spider: TelegramSpider) -> None:
    async def collect() -> list[str]:
        return [request.url async for request in spider.start()]

    assert asyncio.run(collect()) == ["https://t.me/s/uanews"]


def test_channels_are_left_alone_for_the_default_interval(tmp_path: Any) -> None:
    """Without a default, every pass re-read every channel.

    Re-reading refreshes view counts, so it has to happen — but once a minute it
    was a full crawl of every channel for posts that had not changed.
    """
    channels = tmp_path / "channels.json"
    channels.write_text(json.dumps({"channels": [{"name": "uanews"}]}))
    fetch_times = tmp_path / "fetch_times.json"
    fetch_times.write_text(json.dumps({"uanews": get_current_ts() - 30}))

    spider = TelegramSpider(
        channels_file=str(channels), fetch_times=str(fetch_times), hours="24"
    )

    assert spider.default_recrawl_time == DEFAULT_RECRAWL_TIME
    assert list(spider.channel_requests()) == []


def test_the_interval_can_be_overridden_globally(tmp_path: Any) -> None:
    channels = tmp_path / "channels.json"
    channels.write_text(json.dumps({"channels": [{"name": "uanews"}]}))
    fetch_times = tmp_path / "fetch_times.json"
    fetch_times.write_text(json.dumps({"uanews": get_current_ts() - 30}))

    spider = TelegramSpider(
        channels_file=str(channels),
        fetch_times=str(fetch_times),
        hours="24",
        recrawl_time="10",
    )

    assert [r.url for r in spider.channel_requests()] == ["https://t.me/s/uanews"]


def test_a_channel_can_set_its_own_interval(tmp_path: Any) -> None:
    channels = tmp_path / "channels.json"
    channels.write_text(
        json.dumps({"channels": [{"name": "uanews", "recrawl_time": 10}]})
    )
    fetch_times = tmp_path / "fetch_times.json"
    fetch_times.write_text(json.dumps({"uanews": get_current_ts() - 30}))

    spider = TelegramSpider(
        channels_file=str(channels), fetch_times=str(fetch_times), hours="24"
    )

    # The channel's own 10s beats the 300s default.
    assert [r.url for r in spider.channel_requests()] == ["https://t.me/s/uanews"]


def test_recently_fetched_channels_are_skipped(tmp_path: Any) -> None:
    channels = tmp_path / "channels.json"
    channels.write_text(
        json.dumps({"channels": [{"name": "uanews", "recrawl_time": 86400}]})
    )
    fetch_times = tmp_path / "fetch_times.json"
    fetch_times.write_text(json.dumps({"uanews": get_current_ts() - 60}))

    spider = TelegramSpider(
        channels_file=str(channels), fetch_times=str(fetch_times), hours="24"
    )

    assert list(spider.channel_requests()) == []


def test_disabled_channels_are_not_requested(tmp_path: Any) -> None:
    channels = tmp_path / "channels.json"
    channels.write_text(
        json.dumps(
            {"channels": [{"name": "uanews", "disabled": True}, {"name": "other"}]}
        )
    )
    fetch_times = tmp_path / "fetch_times.json"
    fetch_times.write_text("{}")

    spider = TelegramSpider(
        channels_file=str(channels), fetch_times=str(fetch_times), hours="24"
    )

    assert [r.url for r in spider.channel_requests()] == ["https://t.me/s/other"]


def posts_from(results: list[Any]) -> list[dict[str, Any]]:
    """Only the post items. The same crawl also yields audience measurements."""
    return [
        r for r in results if isinstance(r, dict) and r.get("_kind") != CHANNEL_STATS_KIND
    ]


def test_posts_are_parsed_from_channel_markup(spider: TelegramSpider) -> None:
    results = list(spider.parse_channel(channel_response()))
    items = posts_from(results)

    # The third post has no view counter, which marks it as a service message.
    assert len(items) == 2

    first = items[0]
    assert first["url"] == "https://t.me/uanews/100"
    assert first["channel_id"] == "uanews"
    assert first["post_id"] == 100
    assert first["views"] == 12500
    assert first["pub_time"] == POST_TIMES[0]
    assert "Новина про подію" in first["text"]
    assert first["links"] == ["https://example.com/story"]
    # The same url appears in two style blocks and must be stored once.
    assert first["images"] == ["https://cdn.telesco.pe/a.jpg"]
    assert first["mentions"] == ["suspilnenews"]
    assert first["edited"] is False

    second = items[1]
    assert second["text"] == "Пост із медіа-обгорткою."
    assert second["videos"] == ["https://cdn.telesco.pe/v.mp4"]
    # The label sits in the meta line beside the view counter, which is the only
    # place the preview admits a post was changed.
    assert second["edited"] is True


def test_the_crawl_pages_back_through_history(spider: TelegramSpider) -> None:
    """Posts older than the window are reached by following ?before=."""
    requests = [
        r for r in spider.parse_channel(channel_response()) if isinstance(r, Request)
    ]

    assert [r.url for r in requests] == ["https://t.me/s/uanews?before=100"]


def test_paging_stops_once_the_window_is_covered(tmp_path: Any) -> None:
    channels = tmp_path / "channels.json"
    channels.write_text(json.dumps({"channels": [{"name": "uanews"}]}))
    fetch_times = tmp_path / "fetch_times.json"
    fetch_times.write_text("{}")
    # A window that ends after the posts in the fixture were published.
    spider = TelegramSpider(
        channels_file=str(channels), fetch_times=str(fetch_times), hours="0"
    )

    requests = [
        r for r in spider.parse_channel(channel_response()) if isinstance(r, Request)
    ]

    assert requests == []


def test_a_missing_fetch_times_file_is_not_an_error(tmp_path: Any) -> None:
    """State lives under data/, so a fresh deployment starts without the file."""
    channels = tmp_path / "channels.json"
    channels.write_text(json.dumps({"channels": [{"name": "uanews"}]}))
    missing = tmp_path / "data" / "fetch_times.json"

    spider = TelegramSpider(
        channels_file=str(channels), fetch_times=str(missing), hours="24"
    )

    assert spider.fetch_times == {}
    # Closing creates the directory it needs instead of failing.
    spider.closed("finished")
    assert missing.exists()


def test_fetch_times_are_saved_when_the_spider_closes(spider: TelegramSpider) -> None:
    list(spider.parse_channel(channel_response()))
    spider.closed("finished")

    with open(spider.fetch_times_path) as r:
        saved = json.load(r)
    assert "uanews" in saved


class FakeStats:
    def __init__(self, values: dict[str, int]) -> None:
        self.values = values

    def get_value(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)


def attach_stats(spider: TelegramSpider, **values: int) -> None:
    spider.crawler = type(  # type: ignore[assignment]
        "FakeCrawler", (), {"stats": FakeStats(values)}
    )()


def test_an_empty_crawl_is_reported_as_an_error(
    spider: TelegramSpider, caplog: Any
) -> None:
    """Scrapy calls a crawl that requested nothing a clean success."""
    attach_stats(spider, item_scraped_count=0, **{"downloader/request_count": 5})
    spider.requested_channels = 5

    with caplog.at_level(logging.ERROR):
        spider.closed("finished")

    assert "scraped no posts" in caplog.text


def test_skipping_every_channel_is_not_an_error(
    spider: TelegramSpider, caplog: Any
) -> None:
    """All channels read recently is an expected no-op, not a failure."""
    attach_stats(spider, item_scraped_count=0, **{"downloader/request_count": 0})
    spider.requested_channels = 0

    with caplog.at_level(logging.INFO):
        spider.closed("finished")

    assert "scraped no posts" not in caplog.text
    assert "every channel was read recently" in caplog.text


def test_a_productive_crawl_logs_its_count(
    spider: TelegramSpider, caplog: Any
) -> None:
    attach_stats(spider, item_scraped_count=57)

    with caplog.at_level(logging.INFO):
        spider.closed("finished")

    assert "Scraped 57 posts" in caplog.text
    assert "scraped no posts" not in caplog.text


def test_closing_without_a_crawler_does_not_fail(spider: TelegramSpider) -> None:
    # Constructed directly, as the tests above do: there is no crawler to ask.
    spider.closed("finished")


# The shapes t.me serves that the two hardcoded text paths could not reach. Each
# of these was found on the live site, not invented: a pass over all 342 channels
# lost 185 of 5329 posts to the first one alone.
NESTED_HTML = f"""
<body><main><div>
<section class="tgme_channel_history"><div>
  <div class="tgme_widget_message" data-post="uanews/200">
    <div class="tgme_widget_message_bubble">
      <div class="media_supported_cont">
        <div class="tgme_widget_message_one_media">
          <div class="media_supported_cont">
            <a class="tgme_widget_message_photo_wrap"
               style='background-image:url("https://cdn.telesco.pe/one.jpg")'></a>
            <div class="tgme_widget_message_text js-message_text">
              Підпис під альбомом, який раніше зникав.
            </div>
          </div>
        </div>
      </div>
      <div class="tgme_widget_message_reactions js-message_reactions">
        <span class="tgme_reaction"><i class="emoji"><b>👍</b></i>1.58K</span>
        <span class="tgme_reaction"
              style="background-image:url('//telegram.org/img/emoji/40/F09F9881.png')">
          <i class="emoji"
             style="background-image:url('//telegram.org/img/emoji/40/F09F9881.png')"><b></b></i>42</span>
      </div>
      <span class="tgme_widget_message_views">5K</span>
      <time class="time" datetime="{as_datetime_attr(POST_TIMES[0])}"></time>
    </div>
  </div>
  <div class="tgme_widget_message" data-post="uanews/201">
    <div class="tgme_widget_message_bubble">
      <a class="tgme_widget_message_reply" href="https://t.me/uanews/199">
        <div class="tgme_widget_message_text js-message_reply_text">Текст цитати</div>
      </a>
      <div class="tgme_widget_message_text js-message_text">Власний текст поста.</div>
      <span class="tgme_widget_message_views">7</span>
      <time class="time" datetime="{as_datetime_attr(POST_TIMES[1])}"></time>
    </div>
  </div>
</div></section></div></main></body>
"""


def test_text_is_found_however_deep_telegram_nests_it(spider: TelegramSpider) -> None:
    """A post with one media item sits two wrappers deeper than the old selector.

    The two structural paths that used to be hardcoded stopped at a direct child
    of the bubble, so `bubble > media_supported_cont > one_media >
    media_supported_cont > text` returned nothing and the post was dropped as if
    it were a photo without a caption — the one path in the spider that logs
    nothing at all.
    """
    items = posts_from(list(spider.parse_channel(channel_response(NESTED_HTML))))

    assert [item["post_id"] for item in items] == [200, 201]
    assert items[0]["text"] == "Підпис під альбомом, який раніше зникав."
    assert items[0]["images"] == ["https://cdn.telesco.pe/one.jpg"]


def test_a_quoted_message_is_not_mistaken_for_the_post(spider: TelegramSpider) -> None:
    """The reply block comes first in the DOM, so document order alone is wrong."""
    items = posts_from(list(spider.parse_channel(channel_response(NESTED_HTML))))

    assert items[1]["text"] == "Власний текст поста."


def test_reactions_are_collected(spider: TelegramSpider) -> None:
    """Already on the page, and a stronger signal than a view: a view is passive."""
    items = posts_from(list(spider.parse_channel(channel_response(NESTED_HTML))))

    assert items[0]["reactions"] == [
        {"emoji": "👍", "count": 1580},
        # No <b> to read, so the emoji comes from the image name: the same UTF-8
        # bytes in hex, which is how Telegram names every emoji sprite.
        {"emoji": "😁", "count": 42},
    ]
    assert items[0]["reactions_count"] == 1622
    # A post without a reaction block gets an empty list, not a missing field.
    assert items[1]["reactions"] == []
    assert items[1]["reactions_count"] == 0


def test_image_urls_survive_every_quoting_style() -> None:
    assert extract_images(_selector('<a style="background-image:url(x.jpg)"></a>')) == [
        "x.jpg"
    ]
    assert extract_images(
        _selector("<a style='background-image:url(\"y.jpg\");width:5px'></a>")
    ) == ["y.jpg"]
    assert extract_images(
        _selector("<a style=\"background-image:url('z.jpg')\"></a>")
    ) == ["z.jpg"]


def _selector(html: str) -> Any:
    """One post-sized fragment, with the class the image extractor looks for."""
    body = html.replace("<a ", '<a class="tgme_widget_message_photo_wrap" ')
    return Selector(text=f'<div class="tgme_widget_message">{body}</div>')


def test_timestamps_accept_the_forms_telegram_may_serve() -> None:
    """One hardcoded strptime format meant a changed suffix dropped every post."""
    expected = int(datetime(2026, 7, 25, 9, 0, tzinfo=UTC).timestamp())
    assert to_timestamp("2026-07-25T09:00:00+00:00") == expected
    assert to_timestamp("2026-07-25T09:00:00Z") == expected
    assert to_timestamp("2026-07-25T12:00:00+03:00") == expected


def test_a_fetch_time_from_the_future_is_ignored(tmp_path: Any, caplog: Any) -> None:
    """A clock skew used to silence a channel forever, at DEBUG level.

    `current_ts - last_fetch_time` is negative for a future timestamp, so it is
    always below any recrawl interval: the channel is skipped on every pass, and
    the only trace is a DEBUG line the production log level does not print.
    """
    channels = tmp_path / "channels.json"
    channels.write_text(json.dumps({"channels": [{"name": "uanews"}]}))
    fetch_times = tmp_path / "fetch_times.json"
    fetch_times.write_text(json.dumps({"uanews": get_current_ts() + 86400}))

    with caplog.at_level(logging.WARNING):
        spider = TelegramSpider(
            channels_file=str(channels), fetch_times=str(fetch_times), hours="24"
        )

    assert "future" in caplog.text.lower()
    assert [r.url for r in spider.channel_requests()] == ["https://t.me/s/uanews"]


def test_unknown_channels_are_dropped_from_fetch_times(tmp_path: Any) -> None:
    """Otherwise every channel ever removed stays in the file for good."""
    channels = tmp_path / "channels.json"
    channels.write_text(json.dumps({"channels": [{"name": "uanews"}]}))
    fetch_times = tmp_path / "fetch_times.json"
    fetch_times.write_text(json.dumps({"uanews": 1, "gone": 2}))

    spider = TelegramSpider(
        channels_file=str(channels), fetch_times=str(fetch_times), hours="24"
    )
    spider.closed("finished")

    with open(fetch_times) as r:
        assert json.load(r).keys() == {"uanews"}


def test_fetch_times_are_saved_during_the_pass(spider: TelegramSpider) -> None:
    """A pass that is killed halfway must not lose what it already read.

    Written only in `closed()`, a restart meant the next pass re-read every
    channel — twice the requests to Telegram, on a crawler that is being
    throttled precisely when it restarts.
    """
    spider.fetch_times_save_interval = 0
    list(spider.parse_channel(channel_response()))

    with open(spider.fetch_times_path) as r:
        assert "uanews" in json.load(r)


def test_a_pinned_notice_is_not_stored_as_a_post(spider: TelegramSpider) -> None:
    """It carries the pinned post's own text, which we already store from the post.

    Counted apart from a post we simply failed to measure: both used to leave by
    the same door, because a service message has no view counter either.
    """
    service = f"""
    <body><main><div><section class="tgme_channel_history"><div>
      <div class="tgme_widget_message service_message" data-post="uanews/400">
        <div class="tgme_widget_message_bubble">
          <div class="tgme_widget_message_text">BBC pinned «Головне за день»</div>
          <time class="time" datetime="{as_datetime_attr(POST_TIMES[0])}"></time>
        </div>
      </div>
    </div></section></div></main></body>
    """

    assert posts_from(list(spider.parse_channel(channel_response(service)))) == []
    assert spider.report("uanews").service == 1
    assert spider.report("uanews").no_views == 0


def test_a_channel_with_no_posts_is_named_in_the_log(
    spider: TelegramSpider, caplog: Any
) -> None:
    """A silent channel is the failure the old summary could not see.

    `check_scraped_anything` only fires when the whole pass scraped nothing, so
    one channel whose markup no longer parses is invisible among 341 that do.
    """
    empty = """
    <body><main><div><section class="tgme_channel_history"><div>
      <div class="tgme_widget_message" data-post="uanews/300">
        <div class="tgme_widget_message_bubble">
          <span class="tgme_widget_message_views">5</span>
          <time class="time" datetime="2026-07-29T09:00:00+00:00"></time>
        </div>
      </div>
    </div></section></div></main></body>
    """
    list(spider.parse_channel(channel_response(empty)))
    attach_stats(spider, item_scraped_count=0)

    with caplog.at_level(logging.WARNING):
        spider.closed("finished")

    assert "uanews" in caplog.text
    assert "no posts" in caplog.text


def test_a_page_without_messages_is_reported(
    spider: TelegramSpider, caplog: Any
) -> None:
    """Which is what a private, renamed or deleted channel serves: a 200 and nothing."""
    blank = (
        '<body><main><div><section class="tgme_channel_history"><div>'
        "</div></section></div></main></body>"
    )
    list(spider.parse_channel(channel_response(blank)))
    attach_stats(spider, item_scraped_count=0)

    with caplog.at_level(logging.WARNING):
        spider.closed("finished")

    assert "uanews" in caplog.text
    assert "no messages" in caplog.text


def test_a_failed_request_is_reported_with_its_channel(
    spider: TelegramSpider, caplog: Any
) -> None:
    """HttpError used to be swallowed by the middleware without a channel name."""
    request = Request("https://t.me/s/uanews", meta={"channel": "uanews"})
    failure = Failure(ValueError("boom"))
    failure.request = request  # type: ignore[attr-defined]

    with caplog.at_level(logging.ERROR):
        spider.channel_failed(failure)
        attach_stats(spider, item_scraped_count=0)
        spider.closed("finished")

    assert "uanews" in caplog.text
    assert "ValueError" in caplog.text


def test_paging_is_bounded(spider: TelegramSpider, caplog: Any) -> None:
    """A window that never closes must not walk a channel's whole history."""
    response = channel_response()
    response.request.meta["page"] = MAX_PAGES_PER_CHANNEL
    response.request.meta["channel"] = "uanews"

    with caplog.at_level(logging.WARNING):
        results = list(spider.parse_channel(response))

    assert [r for r in results if isinstance(r, Request)] == []
    assert "page limit" in caplog.text


def test_paging_stops_when_before_does_not_move(spider: TelegramSpider) -> None:
    """A page whose lowest id is the one we asked before would loop forever."""
    response = channel_response()
    response.request.meta.update({"channel": "uanews", "page": 1, "before": 100})

    results = list(spider.parse_channel(response))

    assert [r for r in results if isinstance(r, Request)] == []


def test_incomplete_posts_are_dropped() -> None:
    complete = {
        "url": "https://T.me/UaNews/1",
        "text": "Новина.",
        "pub_time": 1784973600,
        "views": 10,
    }

    assert as_record(complete)["url"] == "https://t.me/uanews/1"

    for missing in ("url", "text", "pub_time", "views"):
        with pytest.raises(DropItem):
            as_record({**complete, missing: None})


class FakeCollection:
    def __init__(self) -> None:
        self.batches: list[list[Any]] = []
        self.indexes: list[Any] = []

    def bulk_write(self, operations: list[Any], ordered: bool = True) -> None:
        self.batches.append(list(operations))

    def create_index(self, keys: Any, **kwargs: Any) -> str:
        self.indexes.append(keys)
        return str(kwargs.get("name", ""))


class FakeSettings:
    """Just enough of Scrapy's settings object for from_crawler."""

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def getint(self, name: str, default: int) -> int:
        return int(self.values.get(name, default))

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)


def fake_crawler(**values: Any) -> Any:
    return type("FakeCrawler", (), {"settings": FakeSettings(values)})()


def post(index: int = 0, text: str = "Новина.") -> dict[str, Any]:
    return {
        "url": f"https://t.me/uanews/{index}",
        "text": text,
        "pub_time": 1784973600,
        "views": 10,
    }


def test_pipelines_read_their_settings_from_the_crawler() -> None:
    """Scrapy deprecated passing `spider` into the hooks, so nothing may rely on it."""
    mongo = MongoPipeline.from_crawler(fake_crawler(MONGO_BATCH_SIZE=7))
    jsonl = JsonlPipeline.from_crawler(fake_crawler(JSONL_OUTPUT_PATH="/tmp/x.jsonl"))

    assert mongo.batch_size == 7
    assert jsonl.output_path == "/tmp/x.jsonl"


def test_mongo_pipeline_writes_in_batches(monkeypatch: Any) -> None:
    collection = FakeCollection()
    monkeypatch.setattr(
        "crawler.pipelines.get_documents_collection", lambda _: collection
    )

    pipeline = MongoPipeline.from_crawler(fake_crawler(MONGO_BATCH_SIZE=2))
    pipeline.open_spider()
    for i in range(5):
        pipeline.process_item(post(i))
    # Batch size 2: two full batches written, the fifth item still pending.
    assert [len(b) for b in collection.batches] == [2, 2]

    pipeline.close_spider()
    assert [len(b) for b in collection.batches] == [2, 2, 1]
    assert pipeline.written == 5


def test_mentions_are_read_from_both_spellings() -> None:
    """A handle in the text and a t.me link are the same act, so both count."""
    mentions = extract_mentions(
        "Пише @truexanewsua, дивіться t.me/s/hueviykharkov",
        ["https://t.me/suspilnenews/123"],
    )

    assert mentions == ["hueviykharkov", "suspilnenews", "truexanewsua"]


def test_things_shaped_like_handles_and_not_handles_are_rejected() -> None:
    """Every one of these produced a phantom channel before the rules tightened."""
    assert extract_mentions("напишіть на mail@gmail.com", []) == []
    # Below Telegram's five-character minimum for a public handle.
    assert extract_mentions("@ab", []) == []
    # Invite links name a channel we cannot resolve to a handle.
    assert extract_mentions("t.me/+AbCdEf123", []) == []
    # A private channel is a numeric id under /c/.
    assert extract_mentions("https://t.me/c/1234567/89", []) == []
    assert extract_mentions("t.me/addstickers/pack", []) == []


def test_reflowing_a_paragraph_is_not_an_edit() -> None:
    """html2text wraps the same sentence differently between crawls.

    Without normalizing, every one of those became a revision and buried the
    edits that changed what a post said.
    """
    assert text_hash("Загинули  двоє.\n") == text_hash("Загинули двоє.")
    assert text_hash("Загинули двоє.") != text_hash("Загинули четверо.")


def test_the_stored_text_is_archived_before_it_is_overwritten() -> None:
    """The two stages must stay in this order.

    The first reads `$text`, which is still the *stored* text only because the
    second stage has not run yet. Swap them and the revision log records the new
    text as though it were the old one.
    """
    record = as_record({**post(1, text="Загинули четверо."), "fetch_time": 3000})
    archive, overwrite = keep_history(record)

    assert list(archive["$set"]) == ["revisions", "first_edit_time"]
    assert archive["$set"]["revisions"]["$cond"][0]["$and"][1] == {
        "$ne": ["$text_hash", record["text_hash"]]
    }
    assert overwrite["$set"]["text"] == {"$literal": "Загинули четверо."}
    # An aggregation `$set` may not rewrite the field the query matched on.
    assert "url" not in overwrite["$set"]


def test_values_are_written_as_values_and_not_as_field_paths() -> None:
    """A post opening «$1,7 млрд» could not be stored at all without this.

    In an aggregation update a string beginning with `$` is a field path, so that
    text was read as a path to a field named «1,7 млрд на експорті…» — which ends
    in a full stop, which is not legal — and the whole batch failed with
    "FieldPath must not end with a '.'". Arrays are parsed the same way, so a url
    containing a leading `$` would have done it too.
    """
    record = as_record({**post(1, text="$1,7 млрд на експорті."), "links": ["$odd"]})
    _, overwrite = keep_history(record)

    assert overwrite["$set"]["text"] == {"$literal": "$1,7 млрд на експорті."}
    assert overwrite["$set"]["links"] == {"$literal": ["$odd"]}
    # Every value, not only the ones that happen to start with a dollar today.
    assert all(
        isinstance(value, dict) and "$literal" in value
        for value in overwrite["$set"].values()
    )


def test_an_edit_is_not_timestamped_when_the_crawl_time_is_unknown() -> None:
    """Better no timestamp than one standing in for a time nobody measured."""
    archive, _ = keep_history(as_record(post(1)))

    assert list(archive["$set"]) == ["revisions"]


def test_view_samples_of_one_hour_collapse_into_one_row(monkeypatch: Any) -> None:
    """Twelve re-reads an hour must not leave twelve rows to average over."""
    collection = FakeCollection()
    monkeypatch.setattr(
        "crawler.pipelines.get_post_history_collection", lambda _: collection
    )

    pipeline = PostHistoryPipeline.from_crawler(fake_crawler())
    pipeline.open_spider()
    for fetch_time, views in ((3600, 100), (4500, 180), (7200, 400)):
        pipeline.process_item({**post(1), "views": views, "fetch_time": fetch_time})
    pipeline.close_spider()

    written = collection.batches[0]
    assert [op._filter["hour_ts"] for op in written] == [3600, 3600, 7200]
    # Two of the three share a key, so Mongo upserts two rows out of three
    # operations — the second overwriting the first with the later sample.
    assert len({op._filter["hour_ts"] for op in written}) == 2


def test_a_sample_with_no_time_is_not_recorded(monkeypatch: Any) -> None:
    """It would collapse into whichever hour happened to be current."""
    collection = FakeCollection()
    monkeypatch.setattr(
        "crawler.pipelines.get_post_history_collection", lambda _: collection
    )

    pipeline = PostHistoryPipeline.from_crawler(fake_crawler())
    pipeline.open_spider()
    pipeline.process_item(post(1))
    pipeline.close_spider()

    assert collection.batches == []


def test_a_database_that_cannot_hold_the_history_does_not_stop_the_crawl(
    monkeypatch: Any,
) -> None:
    """This exact failure took the crawler down once.

    The database was out of disk, `create_index` raised, Scrapy let it propagate
    out of `open_spider`, and the spider aborted before reading a single channel —
    so the cost of a missing view-count series was every post of that crawl.
    """

    class RefusingCollection(FakeCollection):
        def create_index(self, keys: Any, **kwargs: Any) -> str:
            raise OperationFailure("available disk space is less than required")

    collection = RefusingCollection()
    monkeypatch.setattr(
        "crawler.pipelines.get_post_history_collection", lambda _: collection
    )

    pipeline = PostHistoryPipeline.from_crawler(fake_crawler())
    pipeline.open_spider()

    assert pipeline.enabled is False
    # And it keeps waving items through rather than collecting them for a write
    # that cannot happen.
    item = {**post(1), "fetch_time": 3600}
    assert pipeline.process_item(item) is item
    pipeline.close_spider()
    assert collection.batches == []


def test_a_failed_write_of_measurements_is_not_fatal(monkeypatch: Any) -> None:
    """A disk that fills mid-crawl is the same problem arriving later."""

    class RefusingCollection(FakeCollection):
        def bulk_write(self, operations: list[Any], ordered: bool = True) -> None:
            raise OperationFailure("available disk space is less than required")

    collection = RefusingCollection()
    monkeypatch.setattr(
        "crawler.pipelines.get_post_history_collection", lambda _: collection
    )

    pipeline = PostHistoryPipeline.from_crawler(fake_crawler(MONGO_BATCH_SIZE=1))
    pipeline.open_spider()
    pipeline.process_item({**post(1), "fetch_time": 3600})
    pipeline.close_spider()

    assert pipeline.written == 0
    assert pipeline.enabled is False


def test_measurements_pass_through_the_post_history_pipeline(monkeypatch: Any) -> None:
    """Every pipeline has to wave the kinds it does not handle through."""
    collection = FakeCollection()
    monkeypatch.setattr(
        "crawler.pipelines.get_post_history_collection", lambda _: collection
    )

    pipeline = PostHistoryPipeline.from_crawler(fake_crawler())
    pipeline.open_spider()
    measurement = {"_kind": CHANNEL_STATS_KIND, "channel_id": "uanews", "hour_ts": 3600}

    assert pipeline.process_item(measurement) is measurement
    pipeline.close_spider()
    assert collection.batches == []


def test_counter_values_are_parsed_from_the_header_forms() -> None:
    """The header spaces its thousands where the per-post view count does not."""
    assert process_counter("713K") == 713000
    assert process_counter("1 234") == 1234
    assert process_counter("1,234") == 1234
    assert process_counter("23") == 23
    assert process_counter(None) == 0


def test_the_subscriber_count_is_read_from_the_channel_header(
    spider: TelegramSpider,
) -> None:
    results = list(spider.parse_channel(channel_response()))
    stats = [r for r in results if isinstance(r, dict) and r.get("_kind") == CHANNEL_STATS_KIND]

    assert len(stats) == 1
    assert stats[0]["channel_id"] == "uanews"
    assert stats[0]["subscribers"] == 713000
    assert stats[0]["channel_title"] == "UA News"
    # Bucketed to the hour, so a channel crawled every five minutes leaves one
    # row an hour rather than twelve.
    assert stats[0]["hour_ts"] % 3600 == 0
    assert stats[0]["hour_ts"] <= stats[0]["ts"]


def test_the_photo_counter_is_not_mistaken_for_subscribers(
    spider: TelegramSpider,
) -> None:
    """Counters differ only by their label, so the label is what must match."""
    results = list(spider.parse_channel(channel_response()))
    stats = [r for r in results if isinstance(r, dict) and r.get("_kind") == CHANNEL_STATS_KIND]

    assert stats[0]["subscribers"] != 108000


def test_history_pages_do_not_repeat_the_measurement(spider: TelegramSpider) -> None:
    """?before= pages carry the same header; measuring again would be noise."""
    url = "https://t.me/s/uanews?before=100"
    response = HtmlResponse(
        url=url, body=CHANNEL_HTML.encode("utf-8"), encoding="utf-8", request=Request(url)
    )

    results = list(spider.parse_channel(response))

    assert not [r for r in results if isinstance(r, dict) and r.get("_kind") == CHANNEL_STATS_KIND]
    assert posts_from(results), "posts are still parsed on a history page"


def test_a_channel_that_hides_its_count_yields_no_measurement(
    spider: TelegramSpider,
) -> None:
    """A missing count is a fact about the channel, not a zero to record."""
    html = CHANNEL_HTML.replace("subscribers", "photos")
    results = list(spider.parse_channel(channel_response(html)))

    assert not [r for r in results if isinstance(r, dict) and r.get("_kind") == CHANNEL_STATS_KIND]


def measurement(channel: str = "uanews", hour_ts: int = 1784973600) -> dict[str, Any]:
    return {
        "_kind": CHANNEL_STATS_KIND,
        "channel_id": channel,
        "channel_title": "UA News",
        "ts": hour_ts + 42,
        "hour_ts": hour_ts,
        "subscribers": 713000,
    }


def test_measurements_never_reach_the_post_pipelines(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """A measurement has no url or text, so a post pipeline must wave it through
    rather than reject it — otherwise every crawl logs a DropItem for an item
    that is working exactly as intended."""
    collection = FakeCollection()
    monkeypatch.setattr(
        "crawler.pipelines.get_documents_collection", lambda _: collection
    )

    mongo = MongoPipeline.from_crawler(fake_crawler(MONGO_BATCH_SIZE=1))
    mongo.open_spider()
    assert mongo.process_item(measurement()) is not None
    mongo.close_spider()
    assert collection.batches == []
    assert mongo.written == 0

    output = tmp_path / "out.jsonl"
    jsonl = JsonlPipeline.from_crawler(fake_crawler(JSONL_OUTPUT_PATH=str(output)))
    jsonl.open_spider()
    jsonl.process_item(measurement())
    jsonl.close_spider()
    assert output.read_text() == ""


def test_channel_stats_are_upserted_once_per_channel_hour(monkeypatch: Any) -> None:
    collection = FakeCollection()
    monkeypatch.setattr(
        "crawler.pipelines.get_channel_stats_collection", lambda _: collection
    )

    pipeline = ChannelStatsPipeline.from_crawler(fake_crawler())
    pipeline.open_spider()
    pipeline.process_item(measurement("uanews", 1784973600))
    pipeline.process_item(measurement("other", 1784973600))
    # A post passes straight through and must not be recorded as a measurement.
    pipeline.process_item(post(1))
    pipeline.close_spider()

    assert pipeline.written == 2
    assert [("channel_id", 1), ("hour_ts", 1)] in collection.indexes

    written = collection.batches[0]
    assert len(written) == 2
    # The routing marker is storage-layer noise and must not be persisted.
    for operation in written:
        assert "_kind" not in operation._doc["$set"]


def test_jsonl_pipeline_deduplicates_by_url(tmp_path: Any) -> None:
    output = tmp_path / "out.jsonl"
    pipeline = JsonlPipeline.from_crawler(fake_crawler(JSONL_OUTPUT_PATH=str(output)))
    pipeline.open_spider()

    for text in ("Перша версія.", "Виправлена версія."):
        # The trailing slash normalizes to the same url as the first post.
        pipeline.process_item({**post(1, text), "url": "https://t.me/uanews/1/"})
    pipeline.close_spider()

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["text"] == "Виправлена версія."
