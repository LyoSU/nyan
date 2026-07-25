import json
from typing import Any

from nyan.client import MessageId
from nyan.clusters import Cluster, Clusters
from nyan.document import Document
from nyan.util import normalize_channel_id, normalize_url


SUMMARY_RESPONSE = json.dumps(
    {
        "headline": "Заголовок",
        "blocks": [{"type": "text", "text": "Що сталося."}],
    },
    ensure_ascii=False,
)


def _make_doc(
    url: str,
    channel_id: str = "channel",
    group: str = "",
    images: tuple[str, ...] = (),
    embedded_images: tuple[dict[str, Any], ...] = (),
    pub_time: int = 1,
) -> Document:
    return Document(
        url=url,
        channel_id=channel_id,
        post_id=1,
        views=1,
        pub_time=pub_time,
        groups={"main": group} if group else {},
        images=images,
        embedded_images=embedded_images,
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


def _make_multi_channel_cluster(message_id: int = 101) -> Cluster:
    cluster = Cluster()
    cluster.add(_make_doc("https://t.me/source_a/1", channel_id="channel_a"))
    cluster.add(_make_doc("https://t.me/source_b/1", channel_id="channel_b"))
    cluster.messages = [MessageId(message_id=message_id, issue="main")]
    cluster.saved_annotation_doc = cluster.docs[0]
    return cluster


def _patch_llm(monkeypatch, response=SUMMARY_RESPONSE):  # type: ignore[no-untyped-def]
    """Replace the LLM, returning the list that records every call."""
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
    cluster = _make_multi_channel_cluster()
    calls = _patch_llm(monkeypatch)

    assert cluster.summary.headline == "Заголовок"
    assert cluster.headline == "Заголовок"
    assert cluster.summary.blocks[0].text == "Що сталося."
    assert len(calls) == 1


def test_cluster_analysis_reuses_cache_across_objects(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = _patch_llm(monkeypatch)

    first_object = _make_multi_channel_cluster()
    second_object = _make_multi_channel_cluster()

    assert first_object.headline == "Заголовок"
    assert second_object.headline == "Заголовок"
    assert len(calls) == 1


def test_a_single_channel_story_is_not_rewritten(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """One source means nothing to merge, so paraphrasing it only adds risk."""
    cluster = _make_single_channel_cluster()
    calls = _patch_llm(monkeypatch, response='{"headline": "Заголовок"}')

    assert cluster.headline == "Заголовок"
    assert not cluster.summary
    assert len(calls) == 1
    # The cheaper prompt, and one that cannot invent a cross-source claim.
    assert "Зведи їх в один пост" not in calls[0]["messages"][0]["content"]


def test_a_multi_source_story_is_summarized_from_every_channel(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cluster = _make_multi_channel_cluster()
    calls = _patch_llm(monkeypatch)

    assert cluster.summary
    prompt = calls[0]["messages"][0]["content"]
    assert "channel_a" in prompt
    assert "channel_b" in prompt


def test_cluster_analysis_survives_broken_llm_output(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cluster = _make_multi_channel_cluster()
    _patch_llm(monkeypatch, response="not json at all")

    assert cluster.headline is None
    # Falsy, so the renderer falls back to quoting a channel.
    assert not cluster.summary


def test_a_summary_without_blocks_does_not_count_as_one(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A headline with nothing under it is not a post."""
    cluster = _make_multi_channel_cluster()
    _patch_llm(monkeypatch, response='{"headline": "Заголовок", "blocks": []}')

    assert cluster.headline == "Заголовок"
    assert not cluster.summary


def test_growing_coverage_rewrites_the_post(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A story told by twice as many channels deserves a fresh telling."""
    cluster = _make_multi_channel_cluster()
    calls = _patch_llm(monkeypatch)

    assert cluster.summary
    assert len(calls) == 1

    cluster.add(_make_doc("https://t.me/source_c/1", channel_id="channel_c"))
    assert cluster.summary
    assert len(calls) == 2


def test_a_new_source_alone_does_not_rewrite_the_post(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Otherwise every arriving post would cost an LLM call for every story."""
    cluster = Cluster()
    for index in range(8):
        cluster.add(
            _make_doc(f"https://t.me/source_{index}/1", channel_id=f"channel_{index}")
        )
    cluster.saved_annotation_doc = cluster.docs[0]
    calls = _patch_llm(monkeypatch)

    assert cluster.summary
    cluster.add(_make_doc("https://t.me/source_9/1", channel_id="channel_9"))
    assert cluster.summary
    assert len(calls) == 1


def test_an_official_source_joining_rewrites_the_post(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """An official channel can confirm or correct everything written before it."""
    cluster = _make_multi_channel_cluster()
    calls = _patch_llm(monkeypatch)

    assert cluster.summary
    cluster.add(
        _make_doc("https://t.me/official/1", channel_id="official", group="red")
    )
    assert cluster.summary
    assert len(calls) == 2


def test_a_stored_summary_is_reused_verbatim(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cluster = _make_multi_channel_cluster()
    _patch_llm(monkeypatch)
    assert cluster.summary

    calls = _patch_llm(monkeypatch)
    restored = Cluster.fromdict(json.loads(cluster.serialize()))

    assert restored.summary.blocks[0].text == "Що сталося."
    assert restored.headline == "Заголовок"
    assert len(calls) == 0


def test_stored_cluster_from_before_summaries_is_not_reanalysed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = _patch_llm(monkeypatch)
    stored = {
        "clid": 1,
        "docs": [_make_doc("https://t.me/source/1").asdict(is_short=True)],
        "diff": [],
    }

    cluster = Cluster.fromdict(stored)

    # Its post is already published; rewriting a day-old story helps nobody.
    assert cluster.headline is None
    assert not cluster.summary
    assert len(calls) == 0


def test_the_prompt_carries_one_document_per_channel(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Two posts from one channel say the same thing twice."""
    cluster = Cluster()
    cluster.add(_make_doc("https://t.me/a/1", channel_id="a", pub_time=10))
    cluster.add(_make_doc("https://t.me/a/2", channel_id="a", pub_time=20))
    cluster.add(_make_doc("https://t.me/b/1", channel_id="b", pub_time=30))
    cluster.saved_annotation_doc = cluster.docs[2]

    urls = [doc.url for doc in cluster.prompt_docs]

    # The chosen document leads, since it is what the fallback would quote.
    assert urls == ["https://t.me/b/1", "https://t.me/a/1"]


def test_normalize_url_removes_query_fragment_and_case() -> None:
    url = " HTTPS://T.ME/UA/123/?Single#Top "
    assert normalize_url(url) == "https://t.me/ua/123"


def test_normalize_channel_id_handles_mentions_and_tme_links() -> None:
    assert normalize_channel_id("@UAliveNews") == "ualivenews"
    assert normalize_channel_id("https://t.me/s/UAliveNews/123?single") == "ualivenews"
