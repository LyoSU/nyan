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


def _make_multi_channel_cluster(message_id: int) -> Cluster:
    cluster = Cluster()
    cluster.add(_make_doc("https://t.me/source_a/1", channel_id="channel_a"))
    cluster.add(_make_doc("https://t.me/source_b/1", channel_id="channel_b"))
    cluster.messages = [MessageId(message_id=message_id, issue="main")]
    cluster.saved_annotation_doc = cluster.docs[0]
    return cluster


def _patch_llm(monkeypatch, response='{"headline": "Заголовок", "differences": []}'):  # type: ignore[no-untyped-def]
    calls = []

    def fake_openai_completion(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return response

    monkeypatch.setattr("nyan.clusters.openai_completion", fake_openai_completion)
    monkeypatch.setattr("nyan.clusters._ANALYSIS_CACHE", {})
    return calls


def _make_single_channel_cluster() -> Cluster:
    cluster = Cluster()
    cluster.add(_make_doc("https://t.me/source/1", channel_id="only_channel"))
    cluster.add(_make_doc("https://t.me/source/2", channel_id="only_channel"))
    cluster.saved_annotation_doc = cluster.docs[0]
    return cluster


def test_cluster_analysis_memoizes_llm_call(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cluster = _make_multi_channel_cluster(101)
    calls = _patch_llm(monkeypatch)

    assert cluster.diff == []
    assert cluster.diff == []
    assert cluster.headline == "Заголовок"
    assert len(calls) == 1


def test_cluster_analysis_reuses_cache_across_objects(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = _patch_llm(monkeypatch)

    first_object = _make_multi_channel_cluster(101)
    second_object = _make_multi_channel_cluster(101)

    assert first_object.headline == "Заголовок"
    assert second_object.headline == "Заголовок"
    assert len(calls) == 1


def test_single_channel_cluster_gets_a_headline_but_no_differences(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cluster = _make_single_channel_cluster()
    calls = _patch_llm(
        monkeypatch,
        response='{"headline": "Заголовок", "differences": ['
        '{"channel_ids": ["only_channel"], "text": "вигадана відмінність"}]}',
    )

    # Nobody else covered the story, so any difference is a hallucination.
    assert cluster.diff == []
    assert cluster.headline == "Заголовок"
    assert len(calls) == 1


def test_cluster_analysis_survives_broken_llm_output(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cluster = _make_multi_channel_cluster(101)
    _patch_llm(monkeypatch, response="not json at all")

    assert cluster.headline is None
    assert cluster.diff == []


def test_cluster_analysis_drops_incomplete_differences(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cluster = _make_multi_channel_cluster(101)
    _patch_llm(
        monkeypatch,
        response='{"headline": "Заголовок", "differences": ['
        '{"channel_ids": ["channel_b"], "text": "справжня"},'
        '{"channel_ids": [], "text": "без джерела"},'
        '{"channel_ids": ["channel_b"], "text": ""}]}',
    )

    assert [d["text"] for d in cluster.diff] == ["справжня"]


def test_stored_cluster_without_headline_is_not_reanalysed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = _patch_llm(monkeypatch)
    stored = {
        "clid": 1,
        "docs": [_make_doc("https://t.me/source/1").asdict(is_short=True)],
        "diff": [],
    }

    cluster = Cluster.fromdict(stored)

    # Clusters saved before headlines existed must not pay for a new LLM call.
    assert cluster.headline is None
    assert cluster.diff == []
    assert len(calls) == 0


def test_normalize_url_removes_query_fragment_and_case() -> None:
    url = " HTTPS://T.ME/UA/123/?Single#Top "
    assert normalize_url(url) == "https://t.me/ua/123"


def test_normalize_channel_id_handles_mentions_and_tme_links() -> None:
    assert normalize_channel_id("@UAliveNews") == "ualivenews"
    assert normalize_channel_id("https://t.me/s/UAliveNews/123?single") == "ualivenews"
