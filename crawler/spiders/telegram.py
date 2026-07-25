import json
import logging
import os
import shutil
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, UTC
from typing import Any

import scrapy
import html2text
from scrapy.http import Response


# Post ids are the only way back through a channel's history: the next page is
# requested as ?before=<lowest id seen>.
Item = dict[str, Any]


def get_current_ts() -> int:
    # now(utc), not now().replace(tzinfo=utc): the latter relabels local
    # wall-clock time as UTC and only agrees with the rest of the pipeline when
    # the host happens to run on UTC.
    return int(datetime.now(UTC).timestamp())


def process_views(views: str | None) -> int:
    if not views:
        return 0
    try:
        views = views.strip()
        if "K" in views:
            return int(float(views.replace("K", "")) * 1000)
        elif "M" in views:
            return int(float(views.replace("M", "")) * 1000000)
        else:
            return int(views)
    except (ValueError, AttributeError):
        return 0


def parse_post_url(url: str) -> Item:
    url = url.split("?")[0].lower()
    channel_id, post_id = url.split("/")[-2:]
    return {
        "url": url,
        "channel_id": channel_id,
        "post_id": int(post_id)
    }


def to_timestamp(dt_str: str | None) -> int:
    if not dt_str:
        return 0
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S+00:00")
        dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp())
    except (ValueError, TypeError):
        logging.warning("Invalid datetime string: %s", dt_str)
        return 0


def html2text_setup() -> html2text.HTML2Text:
    instance = html2text.HTML2Text(bodywidth=0)
    instance.ignore_links = True
    instance.ignore_images = True
    instance.ignore_tables = True
    instance.ignore_emphasis = True
    instance.ul_item_mark = ""
    return instance


class TelegramSpider(scrapy.Spider):
    name = "telegram"
    channel_url_template = "https://t.me/s/{}"
    post_url_template = "https://t.me/{}?embed=1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        assert "channels_file" in kwargs
        with open(kwargs.pop("channels_file")) as r:
            self.channels = json.load(r)["channels"]
            self.channels = {ch["name"].lower(): ch for ch in self.channels}
        assert "fetch_times" in kwargs
        self.fetch_times_path = kwargs.pop("fetch_times")
        self.fetch_times = self.read_fetch_times(self.fetch_times_path)

        assert "hours" in kwargs
        hours = int(kwargs.pop("hours"))
        self.until_ts = get_current_ts() - hours * 3600
        logging.info("Considering last %d hours", hours)

        self.html2text = html2text_setup()

        super().__init__(*args, **kwargs)

    async def start(self) -> AsyncIterator[scrapy.Request]:
        # Scrapy 2.13 replaced start_requests() with an async start(), and 2.17
        # removed the old name entirely. A spider that only defines
        # start_requests() there yields nothing at all — no error, no requests,
        # just an empty crawl — so both entry points are kept.
        for request in self.channel_requests():
            yield request

    def start_requests(self) -> Iterator[scrapy.Request]:
        """Entry point for Scrapy older than 2.13."""
        return self.channel_requests()

    def channel_requests(self) -> Iterator[scrapy.Request]:
        channels = [ch for ch in self.channels.values() if not ch.get("disabled", False)]
        urls = {self.channel_url_template.format(ch["name"]) for ch in channels}
        current_ts = get_current_ts()
        requested = 0
        for url in sorted(urls):
            channel_name = url.split("/")[-1].lower()
            last_fetch_time = self.fetch_times.get(channel_name, 0)
            recrawl_time = self.channels[channel_name].get("recrawl_time", 0)
            if current_ts - last_fetch_time < recrawl_time:
                logging.debug(
                    "Skip %s, fetched %ds ago, recrawl interval %ds",
                    url,
                    current_ts - last_fetch_time,
                    recrawl_time,
                )
                continue
            requested += 1
            yield scrapy.Request(url=url, callback=self.parse_channel)
        logging.info("Requesting %d of %d channels", requested, len(urls))

    @staticmethod
    def read_fetch_times(path: str) -> dict[str, int]:
        """When each channel was last read, or empty on a first ever run.

        A missing file is normal rather than an error: this is state, so it
        lives under data/ and a fresh deployment does not have it yet.
        """
        if not os.path.exists(path):
            logging.info("No %s yet, treating every channel as unread", path)
            return {}
        with open(path) as r:
            times: dict[str, int] = json.load(r)
        return times

    def closed(self, reason: str) -> None:
        os.makedirs(os.path.dirname(self.fetch_times_path) or ".", exist_ok=True)
        temp_path = self.fetch_times_path + ".new"
        with open(temp_path, "w") as w:
            json.dump(self.fetch_times, w)
        shutil.move(temp_path, self.fetch_times_path)
        self.check_scraped_anything(reason)

    def check_scraped_anything(self, reason: str) -> None:
        """Report an empty crawl loudly.

        Scrapy reports a crawl that requested nothing as a clean success, which
        is how an incompatible Scrapy release turned into a feed that quietly
        stopped filling instead of an error anyone could see.
        """
        crawler = getattr(self, "crawler", None)
        if crawler is None or crawler.stats is None:
            return
        scraped = crawler.stats.get_value("item_scraped_count", 0)
        if scraped:
            logging.info("Scraped %d posts", scraped)
            return
        requests = crawler.stats.get_value("downloader/request_count", 0)
        logging.error(
            "Crawl scraped no posts at all (%d requests, reason: %s). "
            "Either every channel was skipped by recrawl_time, or the site "
            "layout changed, or this Scrapy version does not call the spider's "
            "entry point.",
            requests,
            reason,
        )

    def parse_channel(self, response: Response) -> Iterator[Item | scrapy.Request]:
        url = response.url
        channel_name = url.split("/")[-1].split("?")[0].lower()
        history_path = "//body/main/div/section[contains(@class, 'tgme_channel_history')]/div"
        posts = response.xpath(history_path + "/div")

        min_post_id, min_post_ts = None, None
        for post in posts:
            # Rebound to plain strings, hence the separate names: a selector
            # list and its extracted text are not the same kind of thing.
            post_path = post.xpath("@data-post").get()
            post_time = post.css("time.time::attr(datetime)").get()
            if not post_path or not post_time:
                continue

            post_id = int(post_path.split("/")[-1])
            post_ts = to_timestamp(post_time)

            min_post_id = min(post_id, min_post_id) if min_post_id is not None else post_id
            min_post_ts = min(post_ts, min_post_ts) if min_post_ts is not None else post_ts

            post_url = self.post_url_template.format(post_path)
            try:
                item = self._parse_post(post, post_url)
                if item is None:
                    continue
                yield item
            except Exception:
                logging.exception("Unexpected error at %s", post_url)
                continue

        current_ts = get_current_ts()
        self.fetch_times[channel_name] = current_ts
        if not min_post_ts or min_post_ts < self.until_ts:
            return
        url = url.split("?")[0]
        url += f"?before={min_post_id}"
        yield scrapy.Request(url=url, callback=self.parse_channel)

    def _parse_post(self, post_element: Any, post_url: str) -> Item | None:
        text_path = "div.tgme_widget_message_bubble > div.tgme_widget_message_text"
        text_alt_path = (
            "div.tgme_widget_message_bubble > div.media_supported_cont"
            " > div.tgme_widget_message_text"
        )
        views_path = "span.tgme_widget_message_views::text"
        time_path = "time.time::attr(datetime)"
        images_path = "a.tgme_widget_message_photo_wrap::attr(style)"
        videos_path = "video.tgme_widget_message_video::attr(src)"
        reply_path = "a.tgme_widget_message_reply::attr(href)"
        forward_path = "a.tgme_widget_message_forwarded_from_name::attr(href)"

        item = parse_post_url(post_url)
        text_element = post_element.css(text_path)
        text_alt_element = post_element.css(text_alt_path)
        if not text_element and text_alt_element:
            text_element = text_alt_element

        if not text_element:
            # Images only
            return None

        item["text"] = self._parse_html(text_element.extract_first())
        item["links"] = text_element.css("a::attr(href)").getall()
        item["fetch_time"] = get_current_ts()

        views_element = post_element.css(views_path)
        if not views_element:
            # Service messages
            return None

        item["views"] = process_views(views_element.get())

        time_element = post_element.css(time_path)
        item["pub_time"] = to_timestamp(time_element.get())

        # Telegram repeats the same background-image across style blocks for
        # different sizes, so the same url arrives more than once. Deduplicated
        # in place, since the order is the order of the album.
        images = []
        for image_style in post_element.css(images_path):
            for style in image_style.get().split(";"):
                style = style.strip()
                if "background-image" not in style:
                    continue
                image_url = style.split("url(")[-1][1:-2]
                if image_url and image_url not in images:
                    images.append(image_url)
        item["images"] = images

        videos = []
        for video in post_element.css(videos_path):
            video_url = video.get()
            if video_url and video_url not in videos:
                videos.append(video_url)
        item["videos"] = videos

        reply_element = post_element.css(reply_path)
        if reply_element:
            item["reply_to"] = reply_element.get()
        forward_element = post_element.css(forward_path)
        if forward_element:
            item["forward_from"] = forward_element.get()

        return item

    def _parse_html(self, html: str) -> str:
        text = self.html2text.handle(html)
        sentences = [s.strip() for s in text.strip().split("\n") if s.strip()]
        for i, sentence in enumerate(sentences):
            if sentence[-1].isalpha():
                sentences[i] = sentence + "."
        return "\n".join(sentences)
