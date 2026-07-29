import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

from itemadapter import ItemAdapter
from pymongo import UpdateOne
from pymongo.errors import PyMongoError
from scrapy.exceptions import DropItem

from crawler.spiders.telegram import CHANNEL_STATS_KIND
from nyan.mongo import (
    get_channel_stats_collection,
    get_documents_collection,
    get_post_history_collection,
)
from nyan.util import normalize_url


# Posts are written in batches: a crawl of a hundred channels is thousands of
# posts, and a round trip each makes Mongo the bottleneck.
DEFAULT_BATCH_SIZE = 100

DEFAULT_MONGO_CONFIG_PATH = os.getenv("MONGO_CONFIG_PATH") or "configs/mongo_config.json"
DEFAULT_JSONL_OUTPUT_PATH = "telegram_news.jsonl"

REQUIRED_FIELDS = ("url", "text", "pub_time", "views")

# How many past versions of a post to keep. A channel that edits a post twice is
# saying something; one that has edited it forty times is running a live blog,
# and the forty-first entry answers no question the tenth did not. Bounded
# because this array lives in the post document and an unbounded one is how a
# 16MB document limit gets hit by a single busy channel.
MAX_REVISIONS = 10

# How long a view sample is worth keeping. Thirty days: the point of these is the
# slope between two of them while a story is live, and a month on the story is
# settled and the row is dead weight. The database has already been filled once
# by a collection with no expiry, so this one gets its expiry on the day it ships
# rather than after it becomes a problem.
HISTORY_TTL_SECONDS = 30 * 24 * 3600


def text_hash(text: str) -> str:
    """A short digest of a post's text, for spotting silent edits.

    Whitespace-normalized, so a reflowed paragraph is not reported as a change
    of substance — html2text can wrap the same sentence differently between two
    crawls, and a revision log full of those would bury the real edits.

    Truncated to 16 hex characters: this is a change detector, not a signature,
    and 64 bits of it makes a collision between two versions of one short post
    something that does not happen.
    """
    normalized = " ".join(text.split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def is_channel_stats(item: Any) -> bool:
    """Whether this item is an audience measurement rather than a post.

    The crawl now yields two kinds of thing from the same page, and every
    pipeline has to let the kind it does not handle pass through untouched —
    otherwise the post pipeline rejects every measurement as a post with no
    text, and the log fills with DropItem noise for items working as intended.
    """
    return ItemAdapter(item).get("_kind") == CHANNEL_STATS_KIND


def as_record(item: Any) -> dict[str, Any]:
    """Item as a Mongo document, keyed by its normalized url.

    Raises DropItem for posts missing anything the pipeline needs, which is how
    Scrapy expects a pipeline to reject an item.
    """
    adapter = ItemAdapter(item)
    for name in REQUIRED_FIELDS:
        if not adapter.get(name):
            raise DropItem(f"Missing {name} field in {item}")

    record: dict[str, Any] = adapter.asdict()
    record["url"] = normalize_url(str(adapter.get("url")))
    record["text_hash"] = text_hash(str(adapter.get("text")))
    return record


def keep_history(record: dict[str, Any]) -> list[dict[str, Any]]:
    """An update that files the stored text away before overwriting it.

    ## Why this is not a ReplaceOne

    It used to be. A crawl re-reads every post in a channel's recent history
    every few minutes, and a `ReplaceOne` on the url meant each re-read
    discarded whatever was stored and wrote the current state in its place. Two
    things were lost that way, both silently.

    The first was the view count of an hour ago, so `views` was always the newest
    sample and never a series — see `get_post_history_collection` for what that
    costs. The second matters more: if a channel edited a post, the old text was
    gone, and nothing anywhere recorded that it had ever been different. A
    channel that publishes a casualty figure and quietly corrects it twenty
    minutes later is doing the single most interesting thing a news channel can
    do in front of a monitor, and we were overwriting the evidence on the next
    crawl.

    ## How

    As an aggregation-pipeline update, in two stages that must stay in this
    order: the first reads `$text` and `$text_hash` — which are still the *stored*
    values, because the second stage has not run yet — and appends them to
    `revisions` if the hash differs from the one being written. Doing it
    server-side is what makes it atomic and what avoids reading every post's full
    text back over the wire just to compare it.

    The else branch is a bare `"$revisions"` rather than `{"$ifNull": [...]}` on
    purpose: an aggregation `$set` of a missing path leaves the field absent, so
    unedited posts — nearly all of them — never grow a `revisions: []` key.
    """
    stored_revision = {
        "ts": "$fetch_time",
        "text": "$text",
        "text_hash": "$text_hash",
        "views": "$views",
    }
    changed = {
        "$and": [
            # Absent on posts written before this existed, and on the upsert of a
            # post we have never seen. Neither is an edit.
            {"$ne": [{"$type": "$text_hash"}, "missing"]},
            {"$ne": ["$text_hash", record["text_hash"]]},
        ]
    }
    archive: dict[str, Any] = {
        "revisions": {
            "$cond": [
                changed,
                {
                    "$slice": [
                        {
                            "$concatArrays": [
                                {"$ifNull": ["$revisions", []]},
                                [stored_revision],
                            ]
                        },
                        -MAX_REVISIONS,
                    ]
                },
                "$revisions",
            ]
        }
    }

    # When an edit was first *detected*, which is all we can honestly claim: the
    # change happened somewhere between the crawl that saw the old text and this
    # one. The site needs one timestamp to print beside a headline, and
    # `revisions` is the detail behind it. Set from the incoming value rather than
    # from `$fetch_time`, which at this stage is still the previous crawl's.
    detected_at = record.get("fetch_time")
    if detected_at is not None:
        archive["first_edit_time"] = {
            "$cond": [changed, {"$ifNull": ["$first_edit_time", detected_at]}, "$first_edit_time"]
        }

    return [
        {"$set": archive},
        # Every value wrapped in `$literal`, and this is not defensive style — it
        # is required. In an aggregation update a string beginning with `$` is a
        # field path, not a value, so a post opening «$1,7 млрд на експорті» was
        # read as a path to a field named «1,7 млрд на експорті», which ends in a
        # full stop, which is not a legal path: the write failed with
        # "FieldPath must not end with a '.'" and those posts could never be
        # stored at all. Arrays are parsed the same way, so `links` and `mentions`
        # would have broken on any url containing a leading `$` too.
        #
        # This is the one thing `ReplaceOne` gave for free — there, a value is
        # only ever a value — and the cost of the pipeline that keeps the
        # revision log.
        {
            "$set": {
                key: {"$literal": value}
                for key, value in record.items()
                if key != "url"
            }
        },
    ]


class MongoPipeline:
    """Upserts crawled posts into Mongo.

    Settings are read in from_crawler rather than from the `spider` argument of
    the hooks: Scrapy deprecated passing it, and a future version dropping it
    would break a pipeline that depends on it.
    """

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        config_path: str = DEFAULT_MONGO_CONFIG_PATH,
    ) -> None:
        self.batch_size = batch_size
        self.config_path = config_path
        self.operations: list[UpdateOne] = []
        self.written = 0

    @classmethod
    def from_crawler(cls, crawler: Any) -> "MongoPipeline":
        settings = crawler.settings
        return cls(
            batch_size=settings.getint("MONGO_BATCH_SIZE", DEFAULT_BATCH_SIZE),
            config_path=settings.get("MONGO_CONFIG_PATH", DEFAULT_MONGO_CONFIG_PATH),
        )

    def open_spider(self, spider: Any = None) -> None:
        self.collection = get_documents_collection(self.config_path)

    def process_item(self, item: Any, spider: Any = None) -> Any:
        if is_channel_stats(item):
            return item
        record = as_record(item)
        self.operations.append(
            UpdateOne({"url": record["url"]}, keep_history(record), upsert=True)
        )
        if len(self.operations) >= self.batch_size:
            self.flush()
        return item

    def close_spider(self, spider: Any = None) -> None:
        self.flush()
        logging.info("Wrote %d posts to Mongo", self.written)

    def flush(self) -> None:
        if not self.operations:
            return
        self.collection.bulk_write(self.operations, ordered=False)
        self.written += len(self.operations)
        self.operations = []


class PostHistoryPipeline:
    """Records how many views each post had, hour by hour.

    The companion to `ChannelStatsPipeline`, and for the same reason: a single
    measurement of a moving quantity is not a measurement of anything. Views as
    crawled are a floor — whatever the post had when we happened to look, minutes
    after publication — so the absolute number is not comparable between two
    channels crawled at different intervals. The *slope* between two samples of
    the same post is, because the bias is identical at both ends.

    That slope is what makes «поширюється швидко» a fact rather than a feeling,
    and it is the honest replacement for view counts in ranking.

    Written from the post item rather than from a second crawl, so this costs no
    extra request: the numbers are already on the page being parsed. Upserts on
    (url, hour_ts), so the twelve re-reads of a post within one hour leave one
    row carrying the last of them.
    """

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        config_path: str = DEFAULT_MONGO_CONFIG_PATH,
    ) -> None:
        self.batch_size = batch_size
        self.config_path = config_path
        self.operations: list[UpdateOne] = []
        self.written = 0
        #: Cleared when the collection cannot be prepared, so a database that
        #: cannot take these samples costs the samples and nothing else.
        self.enabled = True

    @classmethod
    def from_crawler(cls, crawler: Any) -> "PostHistoryPipeline":
        settings = crawler.settings
        return cls(
            batch_size=settings.getint("MONGO_BATCH_SIZE", DEFAULT_BATCH_SIZE),
            config_path=settings.get("MONGO_CONFIG_PATH", DEFAULT_MONGO_CONFIG_PATH),
        )

    def open_spider(self, spider: Any = None) -> None:
        self.collection = get_post_history_collection(self.config_path)
        # Nothing here may abort the crawl. Scrapy runs `open_spider` on every
        # pipeline before the spider starts and lets an exception propagate, so a
        # failure creating an index on a *derived* collection takes down the
        # collection of posts as well — which is what happened the first time this
        # shipped: the database was out of disk, `create_index` raised
        # OutOfDiskSpace, and the crawler went into a retry loop writing nothing
        # at all. A view-count series is worth having and it is not worth an
        # archive; if this collection cannot be set up, the crawl proceeds without
        # it.
        try:
            self.collection.create_index(
                [("url", 1), ("hour_ts", 1)], unique=True, name="post_hour"
            )
            # The site reads this per story — every post of a cluster over a
            # window — and the ranker reads it by recency. Both are served by the
            # hour.
            self.collection.create_index([("hour_ts", -1)], name="hour")
            # And it expires. The reason to keep these samples is the slope
            # between them while a story is live; a month later the story is
            # settled and the rows are the same dead weight that filled the disk
            # once already. Mongo needs a date, not an epoch, for a TTL index, so
            # the field is written alongside the integer rather than instead of it.
            self.collection.create_index(
                [("sampled_at", 1)], expireAfterSeconds=HISTORY_TTL_SECONDS, name="ttl"
            )
        except PyMongoError:
            logging.exception("Could not prepare post history; crawling without it")
            self.enabled = False

    def process_item(self, item: Any, spider: Any = None) -> Any:
        if is_channel_stats(item) or not self.enabled:
            return item

        adapter = ItemAdapter(item)
        fetch_time = adapter.get("fetch_time")
        views = adapter.get("views")
        # No timestamp means nothing to place the sample at, and a sample with no
        # time is worse than no sample: it would collapse into whichever hour
        # bucket happened to be current. Views of zero are dropped for the same
        # reason `as_record` requires them — the preview omits the counter on
        # service messages, and a zero there is "not measured", not "nobody saw
        # it".
        if not fetch_time or not views:
            return item

        url = normalize_url(str(adapter.get("url")))
        hour_ts = int(fetch_time) - int(fetch_time) % 3600
        self.operations.append(
            UpdateOne(
                {"url": url, "hour_ts": hour_ts},
                {
                    "$set": {
                        "views": int(views),
                        "ts": int(fetch_time),
                        # The same instant as `ts`, as a date, because that is the
                        # only type a TTL index will act on.
                        "sampled_at": datetime.fromtimestamp(int(fetch_time), UTC),
                    },
                    # Written once, on the row's first sample. The channel is
                    # what makes this collection groupable without a join back
                    # to `documents`, and pub_time is what turns a view count
                    # into an age.
                    "$setOnInsert": {
                        "channel_id": adapter.get("channel_id"),
                        "pub_time": adapter.get("pub_time"),
                    },
                },
                upsert=True,
            )
        )
        if len(self.operations) >= self.batch_size:
            self.flush()
        return item

    def close_spider(self, spider: Any = None) -> None:
        self.flush()
        logging.info("Wrote %d post measurements to Mongo", self.written)

    def flush(self) -> None:
        if not self.operations:
            return
        # Same rule as `open_spider`: these are derived numbers and a database
        # that will not take them must not cost us the posts as well.
        try:
            self.collection.bulk_write(self.operations, ordered=False)
            self.written += len(self.operations)
        except PyMongoError:
            logging.exception("Could not write post measurements; dropping the batch")
            self.enabled = False
        self.operations = []


class ChannelStatsPipeline:
    """Records how many subscribers each channel had, hour by hour.

    Views tell you how far one post travelled; subscribers tell you how far it
    could have. Only the ratio of the two compares a 700k channel to a 20k one,
    and only a series of measurements shows a channel growing, stalling, or
    buying its audience overnight.

    Upserts on (channel_id, hour_ts), so a channel crawled twelve times an hour
    leaves one row rather than twelve, and re-running a crawl is harmless.
    """

    def __init__(self, config_path: str = DEFAULT_MONGO_CONFIG_PATH) -> None:
        self.config_path = config_path
        self.operations: list[UpdateOne] = []
        self.written = 0
        self.enabled = True

    @classmethod
    def from_crawler(cls, crawler: Any) -> "ChannelStatsPipeline":
        return cls(
            config_path=crawler.settings.get("MONGO_CONFIG_PATH", DEFAULT_MONGO_CONFIG_PATH)
        )

    def open_spider(self, spider: Any = None) -> None:
        self.collection = get_channel_stats_collection(self.config_path)
        # Idempotent, and the only place that knows this collection's shape.
        # Wrapped for the reason spelled out in `PostHistoryPipeline.open_spider`:
        # Scrapy lets an exception here abort the spider, and audience
        # measurements are not worth an archive.
        try:
            self.collection.create_index(
                [("channel_id", 1), ("hour_ts", 1)], unique=True, name="channel_hour"
            )
            self.collection.create_index([("hour_ts", -1)], name="hour")
        except PyMongoError:
            logging.exception("Could not prepare channel stats; crawling without them")
            self.enabled = False

    def process_item(self, item: Any, spider: Any = None) -> Any:
        if not is_channel_stats(item) or not self.enabled:
            return item

        record = ItemAdapter(item).asdict()
        record.pop("_kind", None)
        self.operations.append(
            UpdateOne(
                {"channel_id": record["channel_id"], "hour_ts": record["hour_ts"]},
                {"$set": record},
                upsert=True,
            )
        )
        return item

    def close_spider(self, spider: Any = None) -> None:
        self.flush()
        logging.info("Wrote %d channel measurements to Mongo", self.written)

    def flush(self) -> None:
        if not self.operations:
            return
        try:
            self.collection.bulk_write(self.operations, ordered=False)
            self.written += len(self.operations)
        except PyMongoError:
            logging.exception("Could not write channel measurements; dropping the batch")
            self.enabled = False
        self.operations = []


class JsonlPipeline:
    """Writes posts to a file instead of Mongo, for local runs.

    Enable with `-s ITEM_PIPELINES='{"crawler.pipelines.JsonlPipeline": 300}'`
    and choose the destination with `-s JSONL_OUTPUT_PATH=...`.
    """

    def __init__(self, output_path: str = DEFAULT_JSONL_OUTPUT_PATH) -> None:
        self.output_path = output_path
        self.items: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_crawler(cls, crawler: Any) -> "JsonlPipeline":
        return cls(
            output_path=crawler.settings.get(
                "JSONL_OUTPUT_PATH", DEFAULT_JSONL_OUTPUT_PATH
            )
        )

    def open_spider(self, spider: Any = None) -> None:
        self.items = {}

    def close_spider(self, spider: Any = None) -> None:
        with open(self.output_path, "w") as w:
            for record in self.items.values():
                w.write(json.dumps(record, ensure_ascii=False) + "\n")
        logging.info("Wrote %d posts to %s", len(self.items), self.output_path)

    def process_item(self, item: Any, spider: Any = None) -> Any:
        if is_channel_stats(item):
            return item
        record = as_record(item)
        self.items[record["url"]] = record
        return item
