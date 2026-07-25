import json
import logging
import os
from typing import Any

from itemadapter import ItemAdapter
from pymongo import ReplaceOne
from scrapy.exceptions import DropItem

from nyan.mongo import get_documents_collection
from nyan.util import normalize_url


# Posts are written in batches: a crawl of a hundred channels is thousands of
# posts, and a round trip each makes Mongo the bottleneck.
DEFAULT_BATCH_SIZE = 100

DEFAULT_MONGO_CONFIG_PATH = os.getenv("MONGO_CONFIG_PATH") or "configs/mongo_config.json"
DEFAULT_JSONL_OUTPUT_PATH = "telegram_news.jsonl"

REQUIRED_FIELDS = ("url", "text", "pub_time", "views")


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
    return record


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
        self.operations: list[ReplaceOne[dict[str, Any]]] = []
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
        record = as_record(item)
        self.operations.append(ReplaceOne({"url": record["url"]}, record, upsert=True))
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
        record = as_record(item)
        self.items[record["url"]] = record
        return item
