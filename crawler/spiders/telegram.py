import json
import logging
import os
import re
import shutil
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
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

# How far back through a channel's history one pass may page. The window is
# supposed to close on its own once posts fall outside `hours`, so this only
# matters when something about the window is wrong — a channel with no
# timestamps, a clock far off, a `hours` argument that never ends. Without a
# bound, that case is not a slow crawl but a walk through the channel's entire
# history, at the expense of every other channel in the pass.
MAX_PAGES_PER_CHANNEL = 20

# How often, at most, the fetch times are written out mid-pass. They used to be
# written only when the spider closed, so a container restart in the middle of a
# pass discarded everything that pass had read and the next one re-read every
# channel — double the requests to Telegram, at the moment a restart makes that
# least affordable.
FETCH_TIMES_SAVE_INTERVAL = 30

# How far ahead of now a stored fetch time may be before it is treated as junk.
# A future timestamp makes `now - last_fetch` negative, which is below every
# recrawl interval, which silences the channel on every pass from then on.
FUTURE_FETCH_TIME_TOLERANCE = 60

# The longest a channel may be left alone, and the ceiling on the backoff below.
#
# 900s is not a taste: `post_history` and `channel_stats` bucket their samples by
# the hour, so two reads an hour is what guarantees every hour a sample. A cap of
# 1800 buys seven more points of saving on the roster and allows an hour with no
# measurement at all, which is the one thing those series exist to prevent.
MAX_RECRAWL_TIME = 900

# How much of a channel's own silence we are willing to fall behind it. A quarter:
# a channel silent for an hour is read within fifteen minutes, one silent for a
# week within fifteen minutes as well (the cap), and one posting more often than
# every twenty minutes is read on the floor — that is, exactly as before.
#
# The roster this runs on is why: 63 of 342 channels post less than twice a day
# and account for 0.3% of all posts, while receiving the same 288 reads a day as
# `kpszsu` with its 198 posts a day. Simulated on the measured rate of every
# channel, this rule removes 35% of the reads — 80% of them from the quiet tail,
# 1% from the channels the feed actually depends on.
QUIET_BACKOFF_DIVISOR = 4


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
    """A `<time datetime=...>` attribute as a UTC epoch.

    Through `fromisoformat` rather than one hardcoded `strptime` pattern. The
    pattern demanded a literal `+00:00`, so the day Telegram wrote `Z` or a local
    offset instead, every post would parse as 0 and be dropped by the pipeline as
    incomplete — a total outage caused by a suffix.
    """
    if not dt_str:
        return 0
    try:
        dt = datetime.fromisoformat(dt_str.strip())
    except (ValueError, TypeError):
        logging.warning("Invalid datetime string: %s", dt_str)
        return 0
    # A time with no zone can only be read as UTC: that is what the preview
    # serves, and guessing the host's zone instead is how timestamps drift.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


# The text node of a quoted message, which sits *before* the post's own text in
# the DOM. Document order alone would therefore pick the quote.
REPLY_TEXT_CLASS = "js-message_reply_text"


def find_message_text(post_element: Any) -> Any | None:
    """The post's own text node, however deeply Telegram has nested it.

    Two structural paths used to be hardcoded, both direct children of the
    bubble: `bubble > text` and `bubble > media_supported_cont > text`. Telegram
    renders a post with a single media item one wrapper deeper —
    `bubble > media_supported_cont > tgme_widget_message_one_media >
    media_supported_cont > text` — and a `>` selector does not reach it. The post
    then looked textless and was dropped through the "Images only" branch, which
    logs nothing at all.

    A live pass over all 342 channels lost 185 of 5329 posts that way, and not at
    random: the loss lands on posts with photo albums and long captions, which is
    what the official channels publish. `mvs_ukraine` lost four of seven.

    So the lookup is by meaning — the first text node that is the post's own —
    rather than by a path that the next wrapper breaks again, silently.
    """
    for node in post_element.css("div.tgme_widget_message_text"):
        classes = (node.xpath("@class").get() or "").split()
        if REPLY_TEXT_CLASS in classes:
            continue
        if node.xpath("ancestor::a[contains(@class, 'tgme_widget_message_reply')]"):
            continue
        # A link preview's own description is the linked page's words, not the
        # channel's, and it is served under its own wrapper.
        if node.xpath("ancestor::*[contains(@class, 'link_preview')]"):
            continue
        return node
    return None


# `background-image: url(...)`, with single quotes, double quotes or none — all
# three of which Telegram serves. This used to be string surgery
# (`style.split("url(")[-1][1:-2]`) that assumed exactly one quoting style and
# silently produced a truncated url for the others.
BACKGROUND_IMAGE_RE = re.compile(r"background-image\s*:\s*url\((['\"]?)(.*?)\1\)")


# Wrappers whose contents belong to some other post, not to this one. The text
# extractor has always known about them; the media extractors did not, and held
# together only because Telegram happens to render a quote's thumbnail under a
# class they do not match. That is an accident of markup, and the next redesign
# is free to end it — silently, by putting the quoted post's picture at the
# front of ours, where the carousel and the site both use it as the cover.
FOREIGN_MEDIA_ANCESTORS = (
    "tgme_widget_message_reply",
    "link_preview",
)

_FOREIGN_ANCESTOR_XPATH = " or ".join(
    f"contains(@class, '{name}')" for name in FOREIGN_MEDIA_ANCESTORS
)


def own_media_nodes(post_element: Any, css: str) -> list[Any]:
    """Nodes matching `css` that are the post's own, not a quoted post's."""
    return [
        node
        for node in post_element.css(css)
        if not node.xpath(f"ancestor::*[{_FOREIGN_ANCESTOR_XPATH}]")
    ]


def parse_duration(text: str) -> int:
    """Telegram's "1:23" or "1:02:03" as seconds, or 0 when unreadable.

    Worth having and free: it is printed on the player already, and it is the
    one comparable thing a clip has besides its poster. Two channels reposting
    one clip get different urls and, because Telegram re-encodes what it is
    given, often different posters too.
    """
    parts = text.strip().split(":")
    if not parts or not all(part.isdigit() for part in parts):
        return 0
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds


def extract_videos(post_element: Any) -> tuple[list[str], list[str], list[int]]:
    """Video urls and the still Telegram renders for each, index-aligned.

    The still is the only comparable thing a video has. Two channels posting the
    same footage get a different CDN url each — the problem photos have, which
    photos solve by embedding the picture — and there is nothing to embed for an
    mp4 we never download. Telegram renders a poster for it, and that poster
    embeds like any other image, which is what lets the same clip from six
    channels be recognized as one.

    Players are read first, because a player is what pairs a video with its
    still. Then any video outside one is picked up with no still: the wrapper is
    an assumption about markup that the next redesign can break, and losing every
    video to it — silently — is how the text extraction broke before.
    """
    videos: list[str] = []
    thumbs: list[str] = []
    durations: list[int] = []

    def add(url: str, thumb: str, duration: int) -> None:
        if url and url not in videos:
            videos.append(url)
            thumbs.append(thumb)
            durations.append(duration)

    for player in own_media_nodes(post_element, "a.tgme_widget_message_video_player"):
        style = player.css("i.tgme_widget_message_video_thumb::attr(style)").get() or ""
        match = BACKGROUND_IMAGE_RE.search(style)
        add(
            (player.css("video::attr(src)").get() or "").strip(),
            match.group(2).strip() if match else "",
            parse_duration(player.css("time.message_video_duration::text").get() or ""),
        )

    for node in own_media_nodes(post_element, "video.tgme_widget_message_video"):
        add((node.xpath("@src").get() or "").strip(), "", 0)

    return videos, thumbs, durations


def extract_images(post_element: Any) -> list[str]:
    """Photo urls, in album order, without repeats.

    Telegram repeats the same background-image across style blocks for different
    sizes, so the same url arrives more than once.
    """
    images: list[str] = []
    for node in own_media_nodes(post_element, "a.tgme_widget_message_photo_wrap"):
        style = node.xpath("@style").get() or ""
        for match in BACKGROUND_IMAGE_RE.finditer(style):
            url = match.group(2).strip()
            if url and url not in images:
                images.append(url)
    return images


# How Telegram names an emoji sprite: the emoji's own UTF-8 bytes in hex, e.g.
# `/img/emoji/40/F09F9881.png` for 😁. The fallback for a reaction whose <b> is
# empty, which is what a custom emoji renders as.
EMOJI_FROM_SPRITE_RE = re.compile(r"/emoji/[^/]+/([0-9A-Fa-f]+)\.")


def reaction_emoji(reaction: Any) -> str:
    emoji = "".join(reaction.css("i.emoji b::text").getall()).strip()
    if emoji:
        return emoji
    style = reaction.css("i.emoji::attr(style)").get() or ""
    match = EMOJI_FROM_SPRITE_RE.search(style)
    if not match:
        return ""
    try:
        return bytes.fromhex(match.group(1)).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return ""


def parse_reactions(post_element: Any) -> list[Item]:
    """Which reactions a post drew, in Telegram's own order (most first).

    Worth having and free: the counts are already on the page being parsed, and
    they say something views cannot. A view is passive — it counts everyone the
    post scrolled past. A reaction is an act, so the ratio between the two is a
    measure of how much a post moved its readers rather than how many it reached.

    A list of pairs rather than a dict keyed by emoji: a custom emoji can leave
    the key empty, Mongo will not take an empty key, and Telegram's ordering is
    information that a dict would not promise to keep.
    """
    reactions: list[Item] = []
    for reaction in post_element.css(
        "div.tgme_widget_message_reactions span.tgme_reaction"
    ):
        # The count is the span's own text, beside the <i> holding the emoji.
        count = process_counter("".join(reaction.xpath("text()").getall()))
        emoji = reaction_emoji(reaction)
        if not emoji and not count:
            continue
        reactions.append({"emoji": emoji, "count": count})
    return reactions


@dataclass
class ChannelReport:
    """What one channel gave up during one pass.

    The old summary could only say whether the *whole* pass scraped anything, so
    a single channel that had stopped parsing was invisible among the hundreds
    that had not — which is exactly the shape of failure this crawler produces.
    """

    pages: int = 0
    messages: int = 0
    posts: int = 0
    no_text: int = 0
    no_views: int = 0
    service: int = 0
    outside_window: int = 0
    error: str = ""


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
        self.fetch_times, self.newest_post_times = self.read_fetch_times(
            self.fetch_times_path
        )

        assert "hours" in kwargs
        hours = int(kwargs.pop("hours"))
        self.until_ts = get_current_ts() - hours * 3600

        # The floor on how long to leave a channel alone after reading it, and the
        # interval for a channel we know nothing about yet. Re-reading the whole
        # window is deliberate — it refreshes view counts, which ranking depends
        # on — but doing that every minute costs a full crawl of every channel for
        # posts that have not changed. Per-channel `recrawl_time` in channels.json
        # overrides both this and the backoff.
        self.default_recrawl_time = int(
            kwargs.pop("recrawl_time", DEFAULT_RECRAWL_TIME)
        )
        # Never below the floor, whatever the floor has been set to: a run with
        # `-a recrawl_time=1800` is asking for 1800, not for a cap of 900.
        self.max_recrawl_time = max(
            int(kwargs.pop("max_recrawl_time", MAX_RECRAWL_TIME)),
            self.default_recrawl_time,
        )
        logging.info(
            "Considering last %d hours, reading channels every %d-%ds by how "
            "recently each one posted",
            hours,
            self.default_recrawl_time,
            self.max_recrawl_time,
        )

        self.html2text = html2text_setup()
        self.requested_channels = 0
        #: What each channel gave up this pass, by channel name. Written by
        #: `parse_channel` and `channel_failed`, read by `report_channels`.
        self.reports: dict[str, ChannelReport] = {}
        self.fetch_times_save_interval = FETCH_TIMES_SAVE_INTERVAL
        self.fetch_times_saved_at = 0

        super().__init__(*args, **kwargs)

    def report(self, channel_name: str) -> ChannelReport:
        return self.reports.setdefault(channel_name, ChannelReport())

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
        backed_off = 0
        for url in sorted(urls):
            channel_name = url.split("/")[-1].lower()
            last_fetch_time = self.fetch_times.get(channel_name, 0)
            recrawl_time = self.recrawl_interval(channel_name, current_ts)
            if recrawl_time > self.default_recrawl_time:
                backed_off += 1
            if current_ts - last_fetch_time < recrawl_time:
                logging.debug(
                    "Skip %s, fetched %ds ago, recrawl interval %ds",
                    url,
                    current_ts - last_fetch_time,
                    recrawl_time,
                )
                continue
            requested += 1
            self.report(channel_name)
            yield scrapy.Request(
                url=url,
                callback=self.parse_channel,
                errback=self.channel_failed,
                # The channel travels with the request so a failure can name it.
                # Without this, an HttpError arrives at the log as a url inside a
                # middleware's message and nothing groups it by channel.
                meta={"channel": channel_name, "page": 1},
            )
        self.requested_channels = requested
        logging.info(
            "Requesting %d of %d channels, %d of them backed off past the %ds floor",
            requested,
            len(urls),
            backed_off,
            self.default_recrawl_time,
        )

    def recrawl_interval(self, channel_name: str, current_ts: int) -> int:
        """How long to leave this channel alone, from how recently it posted.

        A quarter of its own silence, between the floor and the cap. The roster is
        why: 63 of 342 channels post less than twice a day and produce 0.3% of all
        posts, while being read as often as the channel that posts two hundred
        times a day. A channel posting more often than every `floor * divisor`
        seconds is read on the floor, which is to say unchanged.

        An explicit `recrawl_time` in channels.json wins outright — it is someone
        stating a rate they want, not an estimate. A channel we have never read
        gets the floor, so a first pass behaves exactly as before.
        """
        configured = self.channels[channel_name].get("recrawl_time")
        if configured is not None:
            return int(configured)
        newest_post = self.newest_post_times.get(channel_name, 0)
        if not newest_post:
            return self.default_recrawl_time
        silence = max(0, current_ts - newest_post)
        interval = silence // QUIET_BACKOFF_DIVISOR
        return min(max(interval, self.default_recrawl_time), self.max_recrawl_time)

    @staticmethod
    def read_fetch_times(path: str) -> tuple[dict[str, int], dict[str, int]]:
        """When each channel was last read, and when it last posted.

        A missing file is normal rather than an error: this is state, so it
        lives under data/ and a fresh deployment does not have it yet. So is the
        older shape of the file — a bare `{channel: fetched_ts}` — which is read
        as "we know when it was read, and nothing about when it posts". That means
        the first pass after a deploy paces every channel on the floor and the
        backoff starts applying from the second, which is the right way round.

        A timestamp from the future is dropped. It cannot be honest — nothing has
        been read after now — and its effect is the worst kind of failure this
        crawler has: `now - last_fetch` is negative, so it is below every recrawl
        interval, so the channel is skipped on every pass from then on, with the
        only trace a DEBUG line that production's log level does not print.
        """
        if not os.path.exists(path):
            logging.info("No %s yet, treating every channel as unread", path)
            return {}, {}
        with open(path) as r:
            stored: dict[str, Any] = json.load(r)

        if "fetched" in stored or "newest_posts" in stored:
            fetched: dict[str, int] = dict(stored.get("fetched") or {})
            newest_posts: dict[str, int] = dict(stored.get("newest_posts") or {})
        else:
            # The shape this file had before the backoff existed.
            fetched = {name: int(ts) for name, ts in stored.items()}
            newest_posts = {}

        horizon = get_current_ts() + FUTURE_FETCH_TIME_TOLERANCE
        times = {name: ts for name, ts in fetched.items() if ts <= horizon}
        if len(times) != len(fetched):
            logging.warning(
                "Ignoring fetch times in the future for %s: a channel with one "
                "is never read again",
                ", ".join(sorted(set(fetched) - set(times))),
            )
        return times, newest_posts

    def note_fetch_time(self, channel_name: str, newest_post_ts: int = 0) -> None:
        self.fetch_times[channel_name] = get_current_ts()
        if newest_post_ts:
            # Never backwards: the `?before=` pages of the same channel carry older
            # posts, and a deleted latest post must not make us think the channel
            # went quiet. Staying too fast is the harmless direction of this error.
            known = self.newest_post_times.get(channel_name, 0)
            self.newest_post_times[channel_name] = max(known, newest_post_ts)
        if get_current_ts() - self.fetch_times_saved_at >= self.fetch_times_save_interval:
            self.save_fetch_times()

    def save_fetch_times(self) -> None:
        """The crawl's pacing state, atomically, without channels we dropped.

        Written mid-pass as well as at close: a pass killed halfway used to leave
        no record of what it had already read.
        """
        state = {
            "fetched": {
                name: ts for name, ts in self.fetch_times.items() if name in self.channels
            },
            "newest_posts": {
                name: ts
                for name, ts in self.newest_post_times.items()
                if name in self.channels
            },
        }
        os.makedirs(os.path.dirname(self.fetch_times_path) or ".", exist_ok=True)
        temp_path = self.fetch_times_path + ".new"
        with open(temp_path, "w") as w:
            json.dump(state, w)
        shutil.move(temp_path, self.fetch_times_path)
        self.fetch_times_saved_at = get_current_ts()

    def channel_failed(self, failure: Any) -> None:
        """A request that never produced a page, named by channel.

        Scrapy's own handling of these is a middleware message about a url, at a
        level and in a wording that does not group by channel. A channel failing
        every pass for a week is the thing we most need to be able to see.
        """
        request = getattr(failure, "request", None)
        channel_name = (request.meta.get("channel") if request else None) or "unknown"
        reason = getattr(failure, "type", None)
        reason_name = getattr(reason, "__name__", None) or str(reason or failure)
        self.report(channel_name).error = reason_name
        logging.error(
            "Channel %s failed: %s (%s)",
            channel_name,
            reason_name,
            getattr(request, "url", "no url"),
        )

    def closed(self, reason: str) -> None:
        self.save_fetch_times()
        self.report_channels()
        self.check_scraped_anything(reason)

    def report_channels(self) -> None:
        """One summary of the pass, per-channel where a channel is the problem.

        Three kinds of silence, and they have different causes, so they are
        reported apart: a channel that failed outright (already logged by
        `channel_failed`), a channel that served a page with no messages on it —
        which is what a private, renamed or deleted channel serves, with a 200 —
        and a channel whose messages parsed into no posts at all, which is what a
        stale text selector looks like from the outside.
        """
        if not self.reports:
            return

        crawled = [name for name, r in self.reports.items() if not r.error]
        blank = sorted(n for n in crawled if self.reports[n].pages and not self.reports[n].messages)
        # Messages inside the window, not messages on the page: a quiet channel's
        # first page is mostly older than the window, and every one of those is
        # skipped on purpose. Counting them here would accuse the parser of being
        # broken on exactly the channels this crawler is right about.
        textless = sorted(
            n
            for n in crawled
            if self.reports[n].messages - self.reports[n].outside_window > 0
            and not self.reports[n].posts
        )
        never_answered = sorted(n for n in crawled if not self.reports[n].pages)
        no_text = sum(r.no_text for r in self.reports.values())
        no_views = sum(r.no_views for r in self.reports.values())

        logging.info(
            "Pass over %d channels: %d posts from %d messages, dropped %d without "
            "text and %d without a view count, skipped %d service messages and %d "
            "posts older than the window",
            len(self.reports),
            sum(r.posts for r in self.reports.values()),
            sum(r.messages for r in self.reports.values()),
            no_text,
            no_views,
            sum(r.service for r in self.reports.values()),
            sum(r.outside_window for r in self.reports.values()),
        )
        if blank:
            logging.warning(
                "Served a page with no messages (private, renamed or deleted?): %s",
                ", ".join(blank),
            )
        if textless:
            logging.warning(
                "Served messages but no posts, which is what a stale text "
                "selector looks like: %s",
                ", ".join(textless),
            )
        if never_answered:
            logging.warning(
                "Requested but never answered, and with no error either: %s",
                ", ".join(never_answered),
            )

        stats = getattr(getattr(self, "crawler", None), "stats", None)
        if stats is not None and hasattr(stats, "set_value"):
            stats.set_value("nyan/channels_crawled", len(crawled))
            stats.set_value("nyan/channels_failed", len(self.reports) - len(crawled))
            stats.set_value("nyan/channels_without_messages", len(blank))
            stats.set_value("nyan/channels_without_posts", len(textless))
            stats.set_value("nyan/messages_without_text", no_text)
            stats.set_value("nyan/messages_without_views", no_views)
            stats.set_value(
                "nyan/service_messages", sum(r.service for r in self.reports.values())
            )
            stats.set_value(
                "nyan/messages_outside_window",
                sum(r.outside_window for r in self.reports.values()),
            )

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
        meta = response.request.meta if response.request is not None else {}
        report = self.report(meta.get("channel") or channel_name)
        report.pages += 1
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
        max_post_ts = 0
        for post in posts:
            # Rebound to plain strings, hence the separate names: a selector
            # list and its extracted text are not the same kind of thing.
            post_path = post.xpath("@data-post").get()
            post_time = post.css("time.time::attr(datetime)").get()
            if not post_path or not post_time:
                continue
            # «BBC NEWS Україна pinned «Головне за день…»» and its kind. Telegram
            # marks these with a class, and they carry the *pinned post's* text —
            # which we already store from the post itself, so keeping them would
            # duplicate it under a new id.
            #
            # They used to be dropped anyway, but incidentally: a service message
            # has no view counter, so it fell out of the same branch as a post
            # whose views we genuinely failed to read. Separating them keeps
            # `no_views` meaning "a post we could not measure", which is a number
            # worth watching.
            if post.css(".service_message"):
                report.service += 1
                continue
            report.messages += 1

            post_id = int(post_path.split("/")[-1])
            post_ts = to_timestamp(post_time)

            # Both minima are taken over every post on the page, including the
            # ones skipped below: they drive paging, which has to know how far
            # back this page reached even when nothing on it was worth storing.
            min_post_id = min(post_id, min_post_id) if min_post_id is not None else post_id
            min_post_ts = min(post_ts, min_post_ts) if min_post_ts is not None else post_ts
            # How recently the channel posted, which is what paces the next read.
            # Taken before the window filter below — a channel whose newest post is
            # three weeks old still has one, and that is exactly the fact the
            # backoff needs. Pinned notices are excluded by the branch above, on
            # purpose: pinning is not posting, and a channel that pins something
            # once a week is still a channel that is silent.
            max_post_ts = max(max_post_ts, post_ts)

            # Older than the window this crawl declares. Paging already stops at
            # `until_ts`, but nothing stopped us *writing* posts past it, so a
            # quiet channel — whose first page reaches back weeks — had its whole
            # page rewritten every pass. Nobody reads those: the daemon selects
            # `pub_time >= now - documents_offset`, which is the same 24 hours.
            #
            # Measured across the roster: 35% of the posts on the first pages are
            # outside the window, and for the 63 channels that post less than
            # twice a day it is 93-98% of them.
            #
            # This couples `-a hours=` to the daemon's `documents_offset`. They
            # are already equal, and they have to stay that way: a window here
            # narrower than the daemon's would leave it reading posts that no
            # longer get updated.
            if post_ts and post_ts < self.until_ts:
                report.outside_window += 1
                continue

            post_url = self.post_url_template.format(post_path)
            try:
                item = self._parse_post(post, post_url, report)
                if item is None:
                    continue
                report.posts += 1
                yield item
            except Exception:
                logging.exception("Unexpected error at %s", post_url)
                continue

        self.note_fetch_time(channel_name, max_post_ts)
        if not min_post_ts or min_post_ts < self.until_ts:
            return

        page = int(meta.get("page", 1))
        if page >= MAX_PAGES_PER_CHANNEL:
            logging.warning(
                "Stopping at %s after the page limit of %d; the window is not "
                "closing, so posts older than it are left for the next pass",
                channel_name,
                MAX_PAGES_PER_CHANNEL,
            )
            return
        # `before` has to move, or the same page is fetched forever. Scrapy's dupe
        # filter would catch the identical url, but silently and only after the
        # request was built — and a channel stuck on one page would still burn its
        # share of the pass.
        previous_before = meta.get("before")
        stuck = (
            previous_before is not None
            and min_post_id is not None
            and min_post_id >= int(previous_before)
        )
        if stuck:
            logging.warning(
                "Stopping at %s: paging before=%s did not move past %s",
                channel_name,
                min_post_id,
                previous_before,
            )
            return

        url = url.split("?")[0] + f"?before={min_post_id}"
        yield scrapy.Request(
            url=url,
            callback=self.parse_channel,
            errback=self.channel_failed,
            meta={"channel": channel_name, "page": page + 1, "before": min_post_id},
        )

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

    def _parse_post(
        self, post_element: Any, post_url: str, report: ChannelReport | None = None
    ) -> Item | None:
        views_path = "span.tgme_widget_message_views::text"
        meta_path = "span.tgme_widget_message_meta"
        time_path = "time.time::attr(datetime)"
        reply_path = "a.tgme_widget_message_reply::attr(href)"
        forward_path = "a.tgme_widget_message_forwarded_from_name::attr(href)"

        item = parse_post_url(post_url)
        text_element = find_message_text(post_element)

        if text_element is None:
            # A photo or video with no caption, or an album's continuation: there
            # is nothing to cluster, so it is not stored. Counted, because the
            # same silence is what a stale selector produces, and a number that
            # jumps is the only way to tell the two apart.
            if report is not None:
                report.no_text += 1
            return None

        item["text"] = self._parse_html(text_element.get())
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
            if report is not None:
                report.no_views += 1
            return None

        item["views"] = process_views(views_element.get())
        item["reactions"] = parse_reactions(post_element)
        item["reactions_count"] = sum(r["count"] for r in item["reactions"])

        time_element = post_element.css(time_path)
        item["pub_time"] = to_timestamp(time_element.get())

        item["images"] = extract_images(post_element)

        item["videos"], item["video_thumbs"], item["video_durations"] = extract_videos(
            post_element
        )

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
