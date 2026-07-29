import json
from functools import cache
from typing import Any

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection


def read_config(mongo_config_path: str) -> dict[str, Any]:
    with open(mongo_config_path) as r:
        mongo_config: dict[str, Any] = json.load(r)
    return mongo_config


@cache
def get_database(mongo_config_path: str) -> Database[dict[str, Any]]:
    """The database handle for a config file, built once per process.

    MongoClient owns a connection pool and is meant to be long-lived. Creating
    one per collection lookup — which the daemon did on every iteration —
    leaks pools and sockets, because nothing ever closes them.
    """
    mongo_config = read_config(mongo_config_path)
    client: MongoClient[dict[str, Any]] = MongoClient(**mongo_config["client"])
    database_name: str = mongo_config["database_name"]
    return client[database_name]


def get_collection(
    mongo_config_path: str, config_key: str, default: str
) -> Collection[dict[str, Any]]:
    collection_name: str = read_config(mongo_config_path).get(config_key, default)
    return get_database(mongo_config_path)[collection_name]


def get_documents_collection(mongo_config_path: str) -> Collection[dict[str, Any]]:
    return get_collection(mongo_config_path, "documents_collection_name", "documents")


def get_annotated_documents_collection(
    mongo_config_path: str,
) -> Collection[dict[str, Any]]:
    return get_collection(
        mongo_config_path,
        "annotated_documents_collection_name",
        "annotated_documents",
    )


def get_clusters_collection(mongo_config_path: str) -> Collection[dict[str, Any]]:
    return get_collection(mongo_config_path, "clusters_collection_name", "clusters")


def get_memes_collection(mongo_config_path: str) -> Collection[dict[str, Any]]:
    return get_collection(mongo_config_path, "memes_collection_name", "memes")


def get_topics_collection(mongo_config_path: str) -> Collection[dict[str, Any]]:
    return get_collection(mongo_config_path, "topics_collection_name", "topics")


def get_channel_graph_collection(mongo_config_path: str) -> Collection[dict[str, Any]]:
    """Which channels behave as one voice, one document per channel.

    Built by `scripts/build_channel_graph.py` over months of clusters, because
    the question it answers is invisible in any single one: two channels that
    always carry the same event within seconds of each other are one source
    whether or not a given post proves it. A story page reading «56 джерел» needs
    that to avoid presenting a clone network as fifty-six confirmations.

    Derived data, and rebuildable from `clusters` at any time — so it is written
    with plain upserts and nothing depends on it surviving.
    """
    return get_collection(
        mongo_config_path, "channel_graph_collection_name", "channel_graph"
    )


def get_post_history_collection(mongo_config_path: str) -> Collection[dict[str, Any]]:
    """View counts over time, one document per post per hour.

    `documents` keeps the newest measurement of a post and nothing else, because
    a crawl overwrites the fields it re-reads. That is enough to say how far a
    post travelled and useless for saying how fast: a story carried by fifty
    channels in ten minutes and one carried by fifty over two days are the same
    number there. The rate is the more honest figure of the two — views as
    crawled are a floor, sampled minutes after publication, while their
    *derivative* survives that bias because both samples are biased the same way.

    Bucketed to the hour, like `channel_stats`, so a post re-read every five
    minutes leaves one row an hour rather than twelve.
    """
    return get_collection(
        mongo_config_path, "post_history_collection_name", "post_history"
    )


def get_channel_stats_collection(mongo_config_path: str) -> Collection[dict[str, Any]]:
    """Subscriber counts over time, one document per channel per hour.

    Separate from `documents` because it answers a different kind of question:
    documents are what a channel said, this is how big its audience was while it
    said it. Together they give reach per subscriber, which is the only view
    figure that compares one channel to another.
    """
    return get_collection(
        mongo_config_path, "channel_stats_collection_name", "channel_stats"
    )
