import os
from typing import Any
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from pymongo import ReplaceOne
from tqdm import tqdm

from nyan.channels import Channels
from nyan.mongo import get_documents_collection, get_annotated_documents_collection
from nyan.util import Serializable, gen_batch, normalize_url


# Bumped whenever a change to the annotation pipeline makes stored annotations
# unusable, which sends every cached document back through the annotator.
# 7: the image encoder became SigLIP 2 and the language detector changed, so
# annotations from before carry 512-wide image vectors next to the new 768-wide
# ones, and a `language` decided by a different classifier.
CURRENT_VERSION = 7

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
    #: The still Telegram renders for each video, index-aligned with `videos` and
    #: empty where it rendered none. The only comparable thing a video has: see
    #: `extract_videos` in the spider.
    video_thumbs: Sequence[str] = tuple()
    reply_to: str | None = None
    forward_from: str | None = None
    #: Other channels this post names, as bare handles. A directed edge, unlike
    #: co-occurrence in a story — see `extract_mentions` in the spider.
    mentions: Sequence[str] = tuple()
    #: Whether Telegram's preview labels the post as edited. Weaker than
    #: `first_edit_time`, and set for edits made between two of our crawls.
    edited: bool = False
    #: Digest of the crawled text, whitespace-normalized. What the crawler
    #: compares to notice a silent edit.
    text_hash: str | None = None
    #: When an edit was first detected. `revisions` in the `documents` collection
    #: holds the superseded texts themselves; deliberately not mirrored here,
    #: because every field on this dataclass is copied into annotated documents
    #: and then into clusters, and old post bodies have no business there.
    first_edit_time: int | None = None

    channel_title: str = ""
    has_obscene: bool = False
    patched_text: str | None = None
    groups: dict[str, str] = field(default_factory=dict)
    #: What outside registers say about the channel, copied off the registry at
    #: annotation time so `choose_title` can prefer a source that is in one.
    #: Deliberately not part of CURRENT_VERSION: documents written before this
    #: existed keep an empty list rather than being re-embedded for a label.
    badges: list[str] = field(default_factory=list)
    #: Mirrors `Channel.monitor_only`. Whether a channel may be quoted is a fact
    #: about the channel, but the decision is made where only documents are in
    #: hand — picking a post's text, counting a story's reach — so it rides along.
    monitor_only: bool = False
    #: The channel this one is a satellite of, mirroring `Channel.master`. Ten
    #: Труха channels post the same text within minutes, so ranking has to be
    #: able to count them once — and ranking sees documents, not the registry.
    master: str | None = None
    issue: str | None = None
    language: str | None = None
    category: str | None = None
    category_scores: dict[str, float] = field(default_factory=dict)
    tokens: str | None = None
    embedding: list[float] | None = None
    embedding_key: str = "multilingual_e5_base"
    embedded_images: Sequence[dict[str, Any]] = tuple()
    #: `{"url": <video url>, "embedding": <vector of its still>}` per video whose
    #: still could be fetched. Keyed by the video's own url, not the still's, so
    #: the vector can be found from `videos` alone.
    embedded_videos: Sequence[dict[str, Any]] = tuple()

    version: int = CURRENT_VERSION

    def is_reannotation_needed(
        self, new_doc: "Document", is_known_channel: bool = False
    ) -> bool:
        assert normalize_url(new_doc.url) == normalize_url(self.url)
        if self.version != CURRENT_VERSION:
            return True
        # An annotation of a configured channel that carries no channel data was
        # written while the channel list was unreachable. Neither the version nor
        # the text betrays it, so without this check the document keeps its empty
        # issue forever and `is_discarded()` drops it from every feed — which is
        # what happened to several hundred documents on 2026-07-25.
        #
        # Only for channels that are configured now: for a channel absent from
        # channels.json the empty annotation is the correct answer, and asking
        # for it again would recompute an embedding every iteration to no end.
        if is_known_channel and (self.issue is None or not self.groups):
            return True
        # A video whose still was never embedded. Cheaper than bumping
        # CURRENT_VERSION, which would re-embed every text in the window for the
        # sake of the few documents that carry video — and without it a stored
        # annotation only gains the vector if the post's text happens to change,
        # which for video posts is rarely.
        if new_doc.video_thumbs and not self.embedded_videos:
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
        """`is_short` drops what a stored cluster can do without.

        The text is re-read from the annotation cache and the embedding is only
        needed while clustering, so neither has to be carried in
        `posted_clusters`. The image vectors do: the daemon re-reads posted
        clusters from storage on every iteration, and `Cluster.images` picks its
        photos out of `embedded_images` while the ratio gate counts `images`.
        Dropping one and keeping the other passed the gate with nothing to
        choose from, so a published post lost its photos — and got them back
        whenever a source happened to be re-crawled that iteration, which is
        what made the media look random.
        """
        record = super().asdict()
        if is_short:
            record.pop("text")
            record.pop("embedding")
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
    mongo_config_path: str, docs: list[Document], channels: "Channels | None" = None
) -> tuple[list[Document], list[Document]]:
    """Split `docs` into those already annotated in Mongo and those still to do.

    Documents are fetched in batches rather than one query per document: a day
    of crawling is thousands of documents, and a round trip each turns this into
    the slowest step of the iteration.

    `channels` lets a stored annotation be recognized as damaged: one that has
    no issue although its channel is configured has to be redone. Without it the
    check cannot tell that case apart from a channel nobody configured.
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
        is_known_channel = channels is not None and doc.channel_id in channels
        if annotated_doc_loaded.is_reannotation_needed(
            doc, is_known_channel=is_known_channel
        ):
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
    # Without it, pruning scans every document in the collection to find the old
    # ones — which on the half-million entries that accumulated here is the whole
    # 6.8GB read off disk on an interval, to delete twenty thousand rows.
    if "pub_time_1" not in indices:
        collection.create_index([("pub_time", 1)], name="pub_time_1")

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


# How long an annotation is worth keeping. The daemon only ever asks for
# annotations of documents inside `documents_offset`, which is a day — so a week
# is already several times the longest window that reads this, with room for the
# offset to be widened without silently throwing away work.
ANNOTATION_TTL_DAYS = 7

# Ceiling on one pass. The collection had grown to half a million documents
# before anything pruned it, and deleting that in a single statement would hold
# the collection while the daemon needs it. Bounded, so the backlog drains over
# several iterations and each one stays short.
MAX_PRUNE_PER_PASS = 20000


def prune_annotated_documents_mongo(
    mongo_config_path: str, current_ts: int, ttl_days: int = ANNOTATION_TTL_DAYS
) -> int:
    """Drop annotations nothing will ask for again.

    This collection is a cache, and it was the only one in the database with no
    expiry. Each entry carries an embedding — 768 doubles, about 6KB, plus one
    vector per attached image — so half a million of them reached 6.8GB, which
    was 81% of the database and eventually the whole disk. When the disk filled,
    every write failed, including the crawler's: the cost of keeping vectors
    nobody reads was the archive stopping.

    Nothing is lost that could be needed. `read_annotated_documents_mongo` is
    only ever handed documents from the last `documents_offset` — a day — and an
    entry that falls out of that window is never looked up again. If one is
    deleted early the annotator recomputes it, which is exactly what a
    `CURRENT_VERSION` bump already does to the entire collection.

    Returns how many were removed, so the daemon can log a number rather than
    claim it did something.
    """
    collection = get_annotated_documents_collection(mongo_config_path)
    cutoff = current_ts - ttl_days * 24 * 3600

    # `pub_time`, not `fetch_time`: a document's own age is what decides whether
    # the daemon can still ask for it, and fetch_time is absent on entries
    # written before the crawler recorded it.
    stale = collection.find(
        {"pub_time": {"$lt": cutoff}}, {"_id": 1}
    ).limit(MAX_PRUNE_PER_PASS)
    ids = [row["_id"] for row in stale]
    if not ids:
        return 0
    return int(collection.delete_many({"_id": {"$in": ids}}).deleted_count)
