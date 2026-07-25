import os
from typing import Any
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from pymongo import ReplaceOne
from tqdm import tqdm

from nyan.mongo import get_documents_collection, get_annotated_documents_collection
from nyan.util import Serializable, gen_batch, normalize_url


CURRENT_VERSION = 6

MAX_CROPPED_WORDS = 50

# Mongo lookups are batched in chunks so a day's worth of documents neither
# becomes one oversized query nor one query per document.
MONGO_BATCH_SIZE = 1000


def crop_words(text: str | None, max_words: int) -> str:
    """`text` shortened to whole words, with an ellipsis if anything was cut."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


@dataclass
class Document(Serializable):
    url: str
    channel_id: str
    post_id: int
    views: int
    pub_time: int
    pub_time_dt: datetime | None = None
    text: str | None = None
    fetch_time: int | None = None
    images: Sequence[str] = tuple()
    links: Sequence[str] = tuple()
    videos: Sequence[str] = tuple()
    reply_to: str | None = None
    forward_from: str | None = None

    channel_title: str = ""
    has_obscene: bool = False
    patched_text: str | None = None
    groups: dict[str, str] = field(default_factory=dict)
    issue: str | None = None
    language: str | None = None
    category: str | None = None
    category_scores: dict[str, float] = field(default_factory=dict)
    tokens: str | None = None
    embedding: list[float] | None = None
    embedding_key: str = "multilingual_e5_base"
    embedded_images: Sequence[dict[str, Any]] = tuple()

    version: int = CURRENT_VERSION

    def is_reannotation_needed(self, new_doc: "Document") -> bool:
        assert normalize_url(new_doc.url) == normalize_url(self.url)
        if self.version != CURRENT_VERSION:
            return True
        return new_doc.text != self.text

    def is_discarded(self) -> bool:
        if self.issue is None:
            return True
        if not self.groups:
            return True
        if not self.patched_text or len(self.patched_text) < 12:
            return True
        return self.category == "not_news"

    def update_meta(self, new_doc: "Document") -> None:
        self.fetch_time = new_doc.fetch_time
        self.views = new_doc.views

    def asdict(self, is_short: bool = False) -> dict[str, Any]:
        record = super().asdict()
        if is_short:
            record.pop("text")
            record.pop("embedding")
            record.pop("embedded_images")
        return record

    @property
    def cropped_text(self) -> str:
        return crop_words(self.patched_text, MAX_CROPPED_WORDS)


def read_documents_file(
    file_path: str, current_ts: int | None = None, offset: int | None = None
) -> list[Document]:
    assert os.path.exists(file_path)
    with open(file_path) as r:
        docs = [Document.deserialize(line) for line in r]
        if current_ts and offset:
            docs = [doc for doc in docs if doc.pub_time >= current_ts - offset]
    return docs


def read_documents_mongo(
    mongo_config_path: str, current_ts: int, offset: int
) -> list[Document]:
    collection = get_documents_collection(mongo_config_path)
    docs = list(collection.find({"pub_time": {"$gte": current_ts - offset}}))
    return [Document.fromdict(doc) for doc in docs]


def read_annotated_documents_mongo(
    mongo_config_path: str, docs: list[Document]
) -> tuple[list[Document], list[Document]]:
    """Split `docs` into those already annotated in Mongo and those still to do.

    Documents are fetched in batches rather than one query per document: a day
    of crawling is thousands of documents, and a round trip each turns this into
    the slowest step of the iteration.
    """
    collection = get_annotated_documents_collection(mongo_config_path)

    url2annotated: dict[str, dict[str, Any]] = dict()
    urls = [normalize_url(doc.url) for doc in docs]
    batches = list(gen_batch(urls, MONGO_BATCH_SIZE))
    for batch in tqdm(batches, desc="Reading annotated docs from Mongo"):
        for record in collection.find({"url": {"$in": batch}}):
            url2annotated[record["url"]] = record

    annotated_docs = []
    remaining_docs = []
    for doc, url in zip(docs, urls, strict=True):
        annotated_doc = url2annotated.get(url)
        if not annotated_doc:
            remaining_docs.append(doc)
            continue

        annotated_doc_loaded: Document = Document.fromdict(annotated_doc)
        if annotated_doc_loaded.is_reannotation_needed(doc):
            remaining_docs.append(doc)
            continue

        annotated_doc_loaded.update_meta(doc)
        assert annotated_doc_loaded.embedding is not None
        assert annotated_doc_loaded.patched_text is not None
        annotated_docs.append(annotated_doc_loaded)
    return annotated_docs, remaining_docs


def write_annotated_documents_mongo(
    mongo_config_path: str, docs: list[Document]
) -> None:
    collection = get_annotated_documents_collection(mongo_config_path)

    indices = collection.index_information()
    if "url_1" not in indices:
        collection.create_index([("url", 1)], name="url_1")

    operations = []
    for doc in docs:
        assert doc.embedding is not None
        assert doc.patched_text is not None
        doc_dict = doc.asdict()
        doc_dict["url"] = normalize_url(doc.url)
        operations.append(
            ReplaceOne({"url": doc_dict["url"]}, doc_dict, upsert=True)
        )

    # One round trip per batch instead of one per document.
    for batch in gen_batch(operations, MONGO_BATCH_SIZE):
        collection.bulk_write(batch, ordered=False)
