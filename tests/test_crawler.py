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
from scrapy.exceptions import DropItem
from scrapy.http import HtmlResponse, Request

from crawler.pipelines import JsonlPipeline, MongoPipeline, as_record
from crawler.spiders.telegram import (
    TelegramSpider,
    get_current_ts,
    parse_post_url,
    process_views,
    to_timestamp,
)


# A cut-down copy of the structure t.me/s/<channel> serves: one post with text,
# a photo, a view count and a timestamp.
CHANNEL_HTML = """
<body><main><div><section class="tgme_channel_history"><div>
  <div class="tgme_widget_message" data-post="uanews/100">
    <div class="tgme_widget_message_bubble">
      <div class="tgme_widget_message_text">
        Новина про подію. <a href="https://example.com/story">джерело</a>
      </div>
      <a class="tgme_widget_message_photo_wrap"
         style="width:100px;background-image:url('https://cdn.telesco.pe/a.jpg')"></a>
      <a class="tgme_widget_message_photo_wrap"
         style="background-image:url('https://cdn.telesco.pe/a.jpg');width:50px"></a>
      <span class="tgme_widget_message_views">12.5K</span>
      <time class="time" datetime="2026-07-25T09:00:00+00:00"></time>
    </div>
  </div>
  <div class="tgme_widget_message" data-post="uanews/101">
    <div class="tgme_widget_message_bubble">
      <div class="media_supported_cont">
        <div class="tgme_widget_message_text">Пост із медіа-обгорткою</div>
      </div>
      <video class="tgme_widget_message_video" src="https://cdn.telesco.pe/v.mp4"></video>
      <span class="tgme_widget_message_views">900</span>
      <time class="time" datetime="2026-07-25T08:00:00+00:00"></time>
    </div>
  </div>
  <div class="tgme_widget_message" data-post="uanews/102">
    <div class="tgme_widget_message_bubble">
      <div class="tgme_widget_message_text">Пост без переглядів (службовий)</div>
      <time class="time" datetime="2026-07-25T07:00:00+00:00"></time>
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


def test_posts_are_parsed_from_channel_markup(spider: TelegramSpider) -> None:
    results = list(spider.parse_channel(channel_response()))
    items = [r for r in results if isinstance(r, dict)]

    # The third post has no view counter, which marks it as a service message.
    assert len(items) == 2

    first = items[0]
    assert first["url"] == "https://t.me/uanews/100"
    assert first["channel_id"] == "uanews"
    assert first["post_id"] == 100
    assert first["views"] == 12500
    assert first["pub_time"] == to_timestamp("2026-07-25T09:00:00+00:00")
    assert "Новина про подію." in first["text"]
    assert first["links"] == ["https://example.com/story"]
    # The same url appears in two style blocks and must be stored once.
    assert first["images"] == ["https://cdn.telesco.pe/a.jpg"]

    second = items[1]
    assert second["text"] == "Пост із медіа-обгорткою."
    assert second["videos"] == ["https://cdn.telesco.pe/v.mp4"]


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
    attach_stats(spider, item_scraped_count=0, **{"downloader/request_count": 0})

    with caplog.at_level(logging.ERROR):
        spider.closed("finished")

    assert "scraped no posts" in caplog.text


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

    def bulk_write(self, operations: list[Any], ordered: bool = True) -> None:
        self.batches.append(list(operations))


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
