"""What the daemon asks before publishing a story it may already have told."""

import json
from typing import Any

from nyan.client import MessageId
from nyan.clusters import Cluster
from nyan.document import Document
from nyan.relation import (
    FOLLOW_UP,
    SAME,
    UNRELATED,
    judge_relation,
    nearest_clusters,
)
from nyan.util import get_current_ts


def _cluster(
    embedding: list[float],
    message_id: int | None = None,
    age_seconds: int = 600,
    text: str = "Текст",
) -> Cluster:
    cluster = Cluster()
    now = get_current_ts()
    post_id = message_id if message_id is not None else 0
    cluster.add(
        Document(
            url=f"https://t.me/uanews/{post_id}",
            channel_id="uanews",
            post_id=post_id,
            views=100,
            pub_time=now - age_seconds,
            fetch_time=now,
            text=text,
            patched_text=text,
            groups={"main": "blue"},
            issue="main",
            language="uk",
            embedding=embedding,
        )
    )
    if message_id is not None:
        cluster.messages.append(MessageId(message_id=message_id, issue="main"))
    return cluster


def _patch_llm(monkeypatch: Any, response: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_openai_completion(**kwargs: Any) -> str:
        calls.append(kwargs)
        return response

    monkeypatch.setattr("nyan.relation.openai_completion", fake_openai_completion)
    return calls


def test_only_close_enough_and_older_clusters_are_candidates() -> None:
    """Who is worth asking about.

    The floor is set for recall rather than precision — the judge is what
    decides, so a candidate only has to be plausible. Anything published after
    this story cannot be what this story is a development of.
    """
    story = _cluster([1.0, 0.0], age_seconds=600)
    close = _cluster([0.99, 0.14], message_id=1, age_seconds=3600)
    distant = _cluster([0.0, 1.0], message_id=2, age_seconds=3600)
    newer = _cluster([1.0, 0.0], message_id=3, age_seconds=10)

    candidates = nearest_clusters(story, [close, distant, newer])

    assert candidates == [close]


def test_candidates_are_capped_and_ordered_by_closeness() -> None:
    story = _cluster([1.0, 0.0], age_seconds=600)
    published = [
        _cluster([1.0, 0.05 * step], message_id=step, age_seconds=3600 + step)
        for step in range(1, 6)
    ]

    candidates = nearest_clusters(story, published, limit=2)

    assert [c.messages[0].message_id for c in candidates] == [1, 2]


def test_the_verdict_names_the_cluster_it_is_about(monkeypatch: Any) -> None:
    story = _cluster([1.0, 0.0])
    first = _cluster([0.99, 0.14], message_id=1, age_seconds=3600)
    second = _cluster([0.98, 0.19], message_id=2, age_seconds=3600)
    _patch_llm(monkeypatch, json.dumps({"match": 2, "verdict": "same"}))

    relation = judge_relation(story, [first, second])

    assert relation.verdict == SAME
    assert relation.cluster is second


def test_no_match_means_a_story_of_its_own(monkeypatch: Any) -> None:
    story = _cluster([1.0, 0.0])
    other = _cluster([0.99, 0.14], message_id=1, age_seconds=3600)
    _patch_llm(monkeypatch, json.dumps({"match": None, "verdict": "unrelated"}))

    relation = judge_relation(story, [other])

    assert relation.verdict == UNRELATED
    assert relation.cluster is None


def test_a_development_is_told_apart_from_a_repeat(monkeypatch: Any) -> None:
    story = _cluster([1.0, 0.0])
    other = _cluster([0.99, 0.14], message_id=1, age_seconds=3600)
    _patch_llm(monkeypatch, json.dumps({"match": 1, "verdict": "follow_up"}))

    relation = judge_relation(story, [other])

    assert relation.verdict == FOLLOW_UP
    assert relation.cluster is other


def test_nothing_to_compare_against_asks_nothing(monkeypatch: Any) -> None:
    """No candidates, no call: the judge runs once per post at most."""
    calls = _patch_llm(monkeypatch, json.dumps({"match": 1, "verdict": "same"}))

    relation = judge_relation(_cluster([1.0, 0.0]), [])

    assert relation.verdict == UNRELATED
    assert calls == []


def test_an_answer_that_cannot_be_read_publishes_the_story(monkeypatch: Any) -> None:
    """The feed does not stop because a model returned prose.

    Standing alone is the outcome this replaces, so falling back to it costs a
    merge that would have been nice to have and nothing else.
    """
    story = _cluster([1.0, 0.0])
    other = _cluster([0.99, 0.14], message_id=1, age_seconds=3600)
    _patch_llm(monkeypatch, "не можу відповісти")

    assert judge_relation(story, [other]).verdict == UNRELATED


def test_an_index_outside_the_list_is_refused(monkeypatch: Any) -> None:
    story = _cluster([1.0, 0.0])
    other = _cluster([0.99, 0.14], message_id=1, age_seconds=3600)
    _patch_llm(monkeypatch, json.dumps({"match": 7, "verdict": "same"}))

    assert judge_relation(story, [other]).verdict == UNRELATED


def test_a_failed_call_publishes_the_story(monkeypatch: Any) -> None:
    def explode(**kwargs: Any) -> str:
        raise RuntimeError("gateway is down")

    monkeypatch.setattr("nyan.relation.openai_completion", explode)
    story = _cluster([1.0, 0.0])
    other = _cluster([0.99, 0.14], message_id=1, age_seconds=3600)

    assert judge_relation(story, [other]).verdict == UNRELATED


def test_the_judge_is_told_stale_facts_make_a_follow_up(monkeypatch: Any) -> None:
    """Materially new facts of the same event must go out, not be swallowed.

    A "same" verdict is absorbed silently even past the editing window, so the
    boundary between "same" and "follow_up" carries real weight: grown casualty
    counts or an official denial reach the reader only as a "follow_up".
    """
    story = _cluster([1.0, 0.0])
    other = _cluster([0.99, 0.14], message_id=1, age_seconds=3600)
    calls = _patch_llm(monkeypatch, json.dumps({"match": 1, "verdict": "same"}))

    judge_relation(story, [other])

    prompt = "\n".join(str(m["content"]) for m in calls[0]["messages"])
    assert "кількість жертв зросла" in prompt


def test_the_candidates_reach_the_prompt(monkeypatch: Any) -> None:
    """The model has to see the text it is judging, numbered as it answers."""
    story = _cluster([1.0, 0.0], text="Нова подія сталася зранку.")
    other = _cluster(
        [0.99, 0.14], message_id=1, age_seconds=3600, text="Стара подія була вчора."
    )
    calls = _patch_llm(monkeypatch, json.dumps({"match": 1, "verdict": "same"}))

    judge_relation(story, [other])

    prompt = "\n".join(str(m["content"]) for m in calls[0]["messages"])
    assert "Нова подія сталася зранку." in prompt
    assert "Стара подія була вчора." in prompt


def test_the_rules_stay_one_prefix_across_judgements(monkeypatch: Any) -> None:
    """The judge runs once per post, so its rules are worth caching.

    The system half carries no variables, and the cache key names the prompt so
    that every judgement asks for the worker already holding those tokens.
    """
    calls = _patch_llm(monkeypatch, json.dumps({"match": 1, "verdict": "same"}))
    first = _cluster([1.0, 0.0])
    second = _cluster([0.9, 0.4], message_id=2, age_seconds=7200)

    judge_relation(first, [_cluster([0.99, 0.14], message_id=1, age_seconds=3600)])
    judge_relation(second, [_cluster([0.98, 0.2], message_id=3, age_seconds=5400)])

    assert calls[0]["messages"][0] == calls[1]["messages"][0]
    assert calls[0]["messages"][0]["role"] == "system"
    assert calls[0]["prompt_cache_key"] == "relation"
