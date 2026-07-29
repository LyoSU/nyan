import json
import logging
import os
import re
import shutil
from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import datetime, UTC
from typing import Any

import scrapy
import html2text
from scrapy.http import Response


# Post ids are the only way back through a channel's history: the next page is
# requested as ?before=<lowest id seen>.
Item = dict[str, Any]

# Marks an item as an audience measurement rather than a post, so the pipelines
# can tell the two apart. Posts carry no `_kind` at all.
CHANNEL_STATS_KIND = "channel_stats"

# Default delay before a channel is read again. Five minutes keeps the feed
# close to real time while cutting the crawl volume fivefold compared to the
# per-minute loop the sender's restart cycle produces. Override globally with
# `-a recrawl_time=`, or per channel in channels.json.
DEFAULT_RECRAWL_TIME = 300


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


def process_counter(value: str | None) -> int:
    """A header counter such as "713K" or "1 234" as a number.

    Separate from `process_views` only because the channel header spaces its
    thousands where the per-post view count does not.
    """
    if not value:
        return 0
    return process_views(value.replace(" ", "").replace(" ", "").replace(",", ""))


# Telegram's own rule for a public handle: 5 to 32 characters of [A-Za-z0-9_].
# The lookbehind is what keeps "someone@gmail.com" and a trailing "…/@x" out —
# without it every email address in a post became a mentioned channel.
MENTION_RE = re.compile(r"(?<![\w@/.])@([A-Za-z0-9_]{5,32})\b")

# A link to a channel, with or without a post id, and with or without the /s/
# preview prefix. Unanchored on purpose: it has to run over post text as well as
# over extracted hrefs, because a channel can write out t.me/name in the body and
# Telegram autolinks it — matching only whole hrefs missed exactly those.
#
# The trailing lookahead is what rejects a deeper path, which is how every
# non-channel t.me url is shaped: `t.me/joinchat/AAA` matches "joinchat" and is
# then thrown out because a slash follows. `+invite` links never match at all,
# since `+` is not in the handle class.
TME_RE = re.compile(
    r"(?:https?://)?t\.me/(?:s/)?([A-Za-z0-9_]{5,32})(?:/\d+)?(?![\w/])"
)

# Paths that look like handles and are not. The lookahead above already rejects
# these when they carry a path of their own; this catches the bare forms, and
# `c` — which prefixes private-channel links whose numeric id names no handle.
RESERVED_HANDLES = frozenset(
    {
        "joinchat",
        "addstickers",
        "addlist",
        "setlanguage",
        "share",
        "proxy",
        "socks",
        "boost",
    }
)


def extract_mentions(text: str, links: Sequence[str]) -> list[str]:
    """Which other channels a post names, as bare lowercase handles.

    A second kind of edge between channels, and the more informative of the two:
    co-occurrence in a story says two channels cover the same events, which is
    symmetric and says nothing about who follows whom. A mention has a direction.
    A channel that is named by forty others and names none is a source; one that
    names forty and is named by none is an aggregator, and the two are
    indistinguishable by co-occurrence alone.

    Both spellings count — «@handle» in the text and a t.me link — because they
    are the same act, and which one a channel uses is a habit of its editor.
    `forward_from` is deliberately not folded in here: a repost is a stronger
    relation than a mention and it already has its own field.
    """
    handles = {match.group(1).lower() for match in MENTION_RE.finditer(text)}
    for source in (text, *links):
        for match in TME_RE.finditer(source):
            handles.add(match.group(1).lower())
    return sorted(handles - RESERVED_HANDLES)


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

        # How long to leave a channel alone after reading it. Re-reading the
        # whole window is deliberate — it refreshes view counts, which ranking
        # depends on — but doing that every minute costs a full crawl of every
        # channel for posts that have not changed. Per-channel `recrawl_time` in
        # channels.json still wins.
        self.default_recrawl_time = int(
            kwargs.pop("recrawl_time", DEFAULT_RECRAWL_TIME)
        )
        logging.info(
            "Considering last %d hours, recrawling channels every %ds",
            hours,
            self.default_recrawl_time,
        )

        self.html2text = html2text_setup()
        self.requested_channels = 0

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
            recrawl_time = self.channels[channel_name].get(
                "recrawl_time", self.default_recrawl_time
            )
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
        self.requested_channels = requested
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
        if not self.requested_channels:
            # Every channel was read recently enough: an expected no-op, not a
            # failure worth an error in the log.
            logging.info("Nothing to crawl yet, every channel was read recently")
            return
        requests = crawler.stats.get_value("downloader/request_count", 0)
        logging.error(
            "Crawl scraped no posts from %d channels (%d requests, reason: %s). "
            "Either the site layout changed, or this Scrapy version does not "
            "call the spider's entry point.",
            self.requested_channels,
            requests,
            reason,
        )

    def parse_channel(self, response: Response) -> Iterator[Item | scrapy.Request]:
        url = response.url
        channel_name = url.split("/")[-1].split("?")[0].lower()
        # The subscriber count sits in the header of the very page we are
        # already reading for posts, so tracking audience size over time costs
        # no extra request. Head page only: the ?before= pages repeat the same
        # header, and parsing it again would write four identical measurements
        # per crawl of a busy channel.
        if "before=" not in url:
            stats = self.parse_channel_stats(response, channel_name)
            if stats is not None:
                yield stats

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

    @staticmethod
    def parse_channel_stats(response: Response, channel_name: str) -> Item | None:
        """How many people the channel was talking to, right now.

        The header carries several counters — subscribers, photos, videos,
        links — distinguished only by their label, so the label is what we match
        on rather than position. Telegram writes "subscriber" in the singular
        for a channel with one, hence the prefix test.

        Returns None when the count is absent, which happens on a channel that
        hides it. That is a fact about the channel, not an error, and a missing
        measurement is better than a zero that would look like collapse.
        """
        subscribers = 0
        for counter in response.css("div.tgme_channel_info_counter"):
            label = (counter.css("span.counter_type::text").get() or "").strip().lower()
            if label.startswith("subscriber"):
                subscribers = process_counter(counter.css("span.counter_value::text").get())
                break

        if subscribers <= 0:
            return None

        title = (
            response.css("div.tgme_channel_info_header_title span::text").get()
            or response.css("div.tgme_channel_info_header_title::text").get()
            or ""
        ).strip()

        ts = get_current_ts()
        return {
            "_kind": CHANNEL_STATS_KIND,
            "channel_id": channel_name,
            "channel_title": title,
            "ts": ts,
            # Bucketed to the hour so a channel crawled every five minutes
            # leaves one row an hour instead of twelve. Subscriber counts do not
            # move fast enough for the finer resolution to say anything.
            "hour_ts": ts - ts % 3600,
            "subscribers": subscribers,
        }

    def _parse_post(self, post_element: Any, post_url: str) -> Item | None:
        text_path = "div.tgme_widget_message_bubble > div.tgme_widget_message_text"
        text_alt_path = (
            "div.tgme_widget_message_bubble > div.media_supported_cont"
            " > div.tgme_widget_message_text"
        )
        views_path = "span.tgme_widget_message_views::text"
        meta_path = "span.tgme_widget_message_meta"
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
        item["mentions"] = extract_mentions(item["text"], item["links"])
        item["fetch_time"] = get_current_ts()

        # The preview writes "edited" into the same meta line as the view count
        # and the time. It is the only place Telegram admits a post was changed,
        # and it is a weaker signal than our own revision log — it says a post
        # was edited at some point, not what changed or when, and it is lost the
        # moment the post falls out of the crawl window. Recorded anyway, because
        # it catches the edits that happened between two of our crawls and that
        # the revision log therefore never sees as a change.
        meta_text = " ".join(post_element.css(f"{meta_path} ::text").getall())
        item["edited"] = "edited" in meta_text.lower()

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
