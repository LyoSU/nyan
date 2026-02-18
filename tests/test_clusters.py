from nyan.client import MessageId
from nyan.clusters import Cluster, Clusters
from nyan.document import Document
from nyan.util import normalize_channel_id, normalize_url


def _make_doc(url: str, channel_id: str = "channel") -> Document:
    return Document(
        url=url,
        channel_id=channel_id,
        post_id=1,
        views=1,
        pub_time=1,
    )


def _make_cluster(url: str, message_id: int) -> Cluster:
    cluster = Cluster()
    cluster.add(_make_doc(url))
    cluster.messages = [MessageId(message_id=message_id, issue="main")]
    return cluster


def test_urls2messages_cache_invalidates_on_add() -> None:
    clusters = Clusters()

    first = _make_cluster("https://t.me/source/1", 101)
    clusters.add(first)

    # Materialize cached mapping before adding the next cluster.
    _ = clusters.urls2messages

    second = _make_cluster("https://t.me/source/2", 202)
    clusters.add(second)

    candidate = Cluster()
    candidate.add(_make_doc("https://t.me/source/2?single"))

    similar = clusters.find_similar(candidate, "main")
    assert similar is second


def test_normalize_url_removes_query_fragment_and_case() -> None:
    url = " HTTPS://T.ME/UA/123/?Single#Top "
    assert normalize_url(url) == "https://t.me/ua/123"


def test_normalize_channel_id_handles_mentions_and_tme_links() -> None:
    assert normalize_channel_id("@UAliveNews") == "ualivenews"
    assert normalize_channel_id("https://t.me/s/UAliveNews/123?single") == "ualivenews"
