"""What the daemon asks before publishing a story it may already have told."""

import json
from typing import Any

from nyan.client import MessageId
from nyan.clusters import Cluster
from nyan.document import Document
from nyan.relation import (
    FAILED,
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


def test_only_close_enough_clusters_are_candidates() -> None:
    """Who is worth asking about.

    The floor is set for recall rather than precision — the judge is what
    decides, so a candidate only has to be plausible.
    """
    story = _cluster([1.0, 0.0], age_seconds=600)
    close = _cluster([0.99, 0.14], message_id=1, age_seconds=3600)
    distant = _cluster([0.0, 1.0], message_id=2, age_seconds=3600)

    assert nearest_clusters(story, [close, distant]) == [close]


def test_a_post_with_newer_sources_is_still_a_candidate() -> None:
    """"У Києві після падіння БпЛА загорілося авто", then four minutes later
    "Уламки БпЛА впали на дорогу в Києві" — 0.969 apart, and never compared:
    the second cluster held an earlier channel's post, so the published one
    looked newer than the story and was dropped as a candidate."""
    story = _cluster([1.0, 0.0], age_seconds=1800, text="Уламки БпЛА впали на дорогу в Києві")
    published = _cluster(
        [0.99, 0.14],
        message_id=1,
        age_seconds=600,
        text="У Києві після падіння БпЛА загорілося авто",
    )

    assert nearest_clusters(story, [published]) == [published]


def _published_ago(
    embedding: list[float], message_id: int, hours: float, reply_to: int | None = None
) -> Cluster:
    cluster = _cluster(embedding, message_id=message_id, age_seconds=int(hours * 3600) + 600)
    cluster.clid = 70000 + message_id
    cluster.create_time = get_current_ts() - int(hours * 3600)
    cluster.reply_to_clid = reply_to
    return cluster


def test_a_thread_takes_one_slot() -> None:
    """Twelve posts on the Kraków shooting were each other's nearest
    neighbours; offered one by one they took every slot."""
    root = _published_ago([1.0, 0.0], 1, 5)
    reply = _published_ago([1.0, 0.01], 2, 4, reply_to=root.clid)
    reply_of_reply = _published_ago([1.0, 0.02], 3, 3, reply_to=reply.clid)
    other = _published_ago([0.95, 0.31], 4, 2)
    story = _cluster([1.0, 0.012])

    candidates = nearest_clusters(story, [root, reply, reply_of_reply, other], now=get_current_ts())

    assert candidates == [reply, other]


def test_an_older_post_has_to_be_much_closer() -> None:
    """Past a day only a near-copy is offered: "вибухи в Києві" reads the same
    every night, while "the Kraków shooter died in hospital" sat at 0.97 to
    the shooting's posts of the day before."""
    now = get_current_ts()
    yesterday_close = _published_ago([0.97, 0.243], 1, 30)
    yesterday_loose = _published_ago([0.9, 0.436], 2, 30)
    today_loose = _published_ago([0.9, 0.436], 3, 3)
    story = _cluster([1.0, 0.0])

    candidates = nearest_clusters(story, [yesterday_close, yesterday_loose, today_loose], now=now)

    assert candidates == [yesterday_close, today_loose]


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


def test_an_answer_that_cannot_be_read_holds_the_story(monkeypatch: Any) -> None:
    """Prose instead of a verdict is no answer, and no answer is not "new".

    Read as UNRELATED, it published the very stories the judge is there to
    stop: the outage that broke this call broke the summary too, and Moldova
    giving up Russian gas went out twice, seven hours and 0.995 apart.
    """
    story = _cluster([1.0, 0.0])
    other = _cluster([0.99, 0.14], message_id=1, age_seconds=3600)
    _patch_llm(monkeypatch, "не можу відповісти")

    assert judge_relation(story, [other]).verdict == FAILED


def test_an_index_outside_the_list_is_no_answer(monkeypatch: Any) -> None:
    story = _cluster([1.0, 0.0])
    other = _cluster([0.99, 0.14], message_id=1, age_seconds=3600)
    _patch_llm(monkeypatch, json.dumps({"match": 7, "verdict": "same"}))

    assert judge_relation(story, [other]).verdict == FAILED


def test_a_failed_call_holds_the_story(monkeypatch: Any) -> None:
    def explode(**kwargs: Any) -> str:
        raise RuntimeError("gateway is down")

    monkeypatch.setattr("nyan.relation.openai_completion", explode)
    story = _cluster([1.0, 0.0])
    other = _cluster([0.99, 0.14], message_id=1, age_seconds=3600)

    assert judge_relation(story, [other]).verdict == FAILED


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


def test_a_published_candidate_is_shown_as_its_readers_saw_it(monkeypatch: Any) -> None:
    """The post's headline and prose, not the one channel it was annotated from.

    That channel wrote only about the meeting; the post led with the attack the
    president warned of there, and a second story with that warning alone was
    judged a different event against the channel's text and published again.
    """
    story = _cluster([1.0, 0.0], text="Є дані розвідки, що Росія готує нову масовану атаку.")
    published = _cluster(
        [0.99, 0.14],
        message_id=1,
        age_seconds=3600,
        text="Зеленський зустрівся з прем'єром Британії та обговорив підготовку до зими.",
    )
    published.saved_analysis = {
        "headline": "Зеленський заявив про підготовку РФ до нової масованої атаки",
        "summary": {
            "blocks": [{"type": "text", "text": "Розвідка має інформацію про нову атаку."}]
        },
        "generation": "0-",
    }
    unwritten = _cluster(
        [0.98, 0.2], message_id=2, age_seconds=5400, text="Пост лишився словами каналу."
    )
    calls = _patch_llm(monkeypatch, json.dumps({"match": 1, "verdict": "same"}))

    judge_relation(story, [published, unwritten])

    prompt = "\n".join(str(m["content"]) for m in calls[0]["messages"])
    assert "Зеленський заявив про підготовку РФ до нової масованої атаки" in prompt
    assert "Розвідка має інформацію про нову атаку." in prompt
    assert "обговорив підготовку до зими" not in prompt
    assert "Пост лишився словами каналу." in prompt
    # The story itself has no post yet: it is judged on its own text.
    assert "Є дані розвідки" in prompt


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
