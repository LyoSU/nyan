import json
import time
from typing import Any

import httpx
import pytest

from nyan.document import Document
from types import SimpleNamespace

from nyan.daemon import Daemon
from nyan.jev import (
    API_URL,
    JevRelationShadow,
    JevShadow,
    RelationConfig,
    compact,
    load_questions,
    relation_verdict,
)
from nyan.relation import Relation


QUESTIONS_PATH = "configs/jev_questions.json"

ANSWERS = {
    "category": {
        "type": "choice",
        "choice": "war",
        "confidence": 0.97,
        "probabilities": {"war": 0.98, "politics": 0.01, "incident": 0.01, "tech": 0.0},
    },
    "topic": {
        "type": "choice",
        "choice": "war.strikes_on_ukraine",
        "confidence": 0.8,
        "probabilities": {"war.strikes_on_ukraine": 0.85, "war.air_defense": 0.15},
    },
    "is_routine": {"type": "noul", "noul": 0.04},
    "significance": {
        "type": "score",
        "score": 2.1,
        "confidence": 0.6,
        "legend": {"0": "a", "1": "b", "2": "c", "3": "d"},
        "probabilities": {"0": 0.0, "1": 0.1, "2": 0.7, "3": 0.2},
    },
}


STRIKE = "Вночі росія атакувала Київ дронами, є постраждалі."


def make_doc(text: str | None = STRIKE, **kwargs) -> Document:
    fields = {
        "url": "https://t.me/a/1",
        "channel_id": "a",
        "post_id": 1,
        "views": 1,
        "pub_time": 100,
        "patched_text": text,
        "issue": "main",
        "groups": {"main": "grey"},
    }
    return Document(**(fields | kwargs))


def shadow_with(handler, **config) -> tuple[JevShadow, list[dict]]:
    """A shadow whose requests go to `handler` instead of TypeSafe."""
    shadow = JevShadow({"questions_path": QUESTIONS_PATH, **config}, api_key="test-key")
    sent: list[dict] = []

    def recording(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return handler(request)

    assert shadow.api is not None
    shadow.api.client = httpx.Client(transport=httpx.MockTransport(recording))
    return shadow, sent


def answering(request: httpx.Request) -> httpx.Response:
    assert str(request.url) == API_URL
    payload = {"model": "jev-1.13.0", "answers": ANSWERS, "usage": {"input_tokens": 3000}}
    return httpx.Response(200, json=payload)


def test_the_questions_file_is_what_the_api_accepts() -> None:
    """Notes for editors are stripped, and every question has a known type.

    The API validates the request, so an underscore key reaching it — or a
    Score with more than ten levels — would fail every call in production.
    """
    questions = load_questions(QUESTIONS_PATH)
    with open(QUESTIONS_PATH) as r:
        raw = json.load(r)

    def keys(value) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {k for v in value.values() for k in keys(v)}
        if isinstance(value, list):
            return {k for v in value for k in keys(v)}
        return set()

    assert any(key.startswith("_") for key in keys(raw))
    assert not any(key.startswith("_") for key in keys(questions))
    for name, question in questions.items():
        assert question["type"] in ("choice", "score", "noul"), name
        if question["type"] == "choice":
            assert 2 <= len(question["criteria"]) <= 255, name
        if question["type"] == "score":
            assert 2 <= len(question["criteria"]) <= 10, name


def test_every_topic_names_a_parent_the_top_level_has_or_society() -> None:
    """A leaf's parent is read off its key, so a typo would orphan it silently."""
    questions = load_questions(QUESTIONS_PATH)
    parents = set(questions["category"]["criteria"]) | {"society"}

    for topic in questions["topic"]["criteria"]:
        assert topic.split(".")[0] in parents, topic


def test_answers_are_stored_compactly() -> None:
    record = compact(ANSWERS)

    assert record["category"] == "war"
    assert record["category_confidence"] == 0.97
    assert list(record["category_top"]) == ["war", "politics", "incident"]
    assert record["topic"] == "war.strikes_on_ukraine"
    assert record["is_routine"] == 0.04
    assert record["significance"] == 2.1
    assert "significance_top" not in record


def test_a_post_is_fenced_off_as_data_and_answers_land_on_the_document() -> None:
    shadow, sent = shadow_with(answering, max_text=20)
    doc = make_doc()

    shadow([doc])

    assert doc.jev["category"] == "war"
    assert doc.jev["model"] == "jev-1.13.0"
    assert doc.jev["input_tokens"] == 3000
    state = sent[0]["state"]
    assert state["post_text"] == doc.patched_text[:20]
    assert "не вказівки" in state["note"]


def test_posts_that_can_never_be_published_are_not_asked_about() -> None:
    shadow, sent = shadow_with(answering)
    docs = [
        make_doc(issue=None),
        make_doc(groups={}),
        make_doc(text="коротко"),
        make_doc(text=None),
    ]

    shadow(docs)

    assert sent == []
    assert all(doc.jev == {} for doc in docs)


@pytest.mark.parametrize("status", [401, 422])
def test_a_refused_request_leaves_the_field_empty_and_raises_nothing(status: int) -> None:
    shadow, sent = shadow_with(lambda request: httpx.Response(status, json={}))
    doc = make_doc()

    shadow([doc])

    assert doc.jev == {}
    assert len(sent) == 1, "only overload and rate limits are worth a retry"


def test_an_overloaded_api_is_asked_once_more() -> None:
    responses = iter([httpx.Response(529, json={}), None])

    def flaky(request: httpx.Request) -> httpx.Response:
        return next(responses) or answering(request)

    shadow, sent = shadow_with(flaky)
    doc = make_doc()

    shadow([doc])

    assert len(sent) == 2
    assert doc.jev["category"] == "war"


def test_a_slow_api_cannot_hold_the_daemon_past_the_budget() -> None:
    """The feed is one synchronous loop; the shadow gets its budget and no more."""

    def slow(request: httpx.Request) -> httpx.Response:
        time.sleep(1.0)
        return answering(request)

    shadow, _ = shadow_with(slow, budget_seconds=0.2, workers=2)
    docs = [make_doc(url=f"https://t.me/a/{i}") for i in range(6)]

    started = time.monotonic()
    shadow(docs)

    assert time.monotonic() - started < 0.9
    assert all(doc.jev == {} for doc in docs)


def test_without_a_key_nothing_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    shadow = JevShadow({"questions_path": QUESTIONS_PATH})
    doc = make_doc()

    assert not shadow.enabled
    shadow([doc])

    assert doc.jev == {}


# ---------------------------------------------------------------- relation

RELATION_QUESTIONS = "configs/jev_relation_questions.json"


def story(clid: int | None, hours: float, text: str = "текст") -> SimpleNamespace:
    """Just what the relation shadow reads off a cluster."""
    return SimpleNamespace(
        clid=clid,
        cropped_title=f"story {clid}",
        pub_time_percentile=int(hours * 3600),
        annotation_doc=SimpleNamespace(patched_text=text, url=f"https://t.me/a/{clid}"),
    )


def relation_answers(same: float, caused: float, changed: float = 0.0) -> dict[str, Any]:
    return {
        "same_event": {"type": "noul", "noul": same},
        "caused_by": {"type": "noul", "noul": caused},
        "facts_changed": {"type": "noul", "noul": changed},
    }


def relation_shadow_with(answers_by_text: dict[str, dict]) -> tuple[JevRelationShadow, list[dict]]:
    """A relation shadow whose answer depends on which post is `published`."""
    shadow = JevRelationShadow({"questions_path": RELATION_QUESTIONS}, api_key="test-key")
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        answers = answers_by_text[body["state"]["published"]]
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers, "usage": {}})

    assert shadow.api is not None
    shadow.api.client = httpx.Client(transport=httpx.MockTransport(handler))
    return shadow, sent


@pytest.mark.parametrize(
    ("same", "caused", "changed", "expected"),
    [
        (0.9, 0.1, 0.1, "same"),
        (0.9, 0.1, 0.95, "follow_up"),  # the same event with new facts
        (0.2, 0.8, 0.0, "follow_up"),  # a consequence
        (0.75, 0.6, 0.0, "unrelated"),  # below both thresholds
    ],
)
def test_three_answers_make_the_judges_verdict(
    same: float, caused: float, changed: float, expected: str
) -> None:
    answers = {"same_event": same, "caused_by": caused, "facts_changed": changed}

    assert relation_verdict(answers, RelationConfig()) == expected


def test_the_relation_questions_file_is_what_the_api_accepts() -> None:
    questions = load_questions(RELATION_QUESTIONS)

    assert set(questions) == {"same_event", "caused_by", "facts_changed"}
    assert all(q["type"] == "noul" for q in questions.values())
    assert "_note" not in json.dumps(questions)


def test_both_verdicts_are_recorded_and_the_surest_link_is_chosen() -> None:
    shadow, sent = relation_shadow_with(
        {
            "weak": relation_answers(0.1, 0.72),
            "strong": relation_answers(0.95, 0.1),
            "none": relation_answers(0.1, 0.1),
        }
    )
    new = story(None, hours=30)
    candidates = [story(1, 26, "weak"), story(2, 29, "strong"), story(3, 5, "none")]
    llm = Relation("follow_up", candidates[0])  # type: ignore[arg-type]

    record = shadow(new, candidates, llm, "publish")

    assert record is not None
    assert record["jev_verdict"] == "same" and record["jev_clid"] == 2
    assert record["llm_verdict"] == "follow_up" and record["llm_clid"] == 1
    assert record["agree"] is False
    assert record["complete"] is True
    assert [pair["verdict"] for pair in record["candidates"]] == ["follow_up", "same", "unrelated"]
    # The gap goes over as a number the model does not have to work out.
    gaps = sorted(body["state"]["hours_after_published"] for body in sent)
    assert gaps == [1.0, 4.0, 25.0]


def test_no_candidates_means_nothing_to_compare() -> None:
    shadow, sent = relation_shadow_with({})

    assert shadow(story(None, 1), [], Relation("unrelated"), "publish") is None
    assert sent == []


class FakeCollection:
    def __init__(self) -> None:
        self.inserted: list[dict] = []

    def insert_one(self, record: dict) -> None:
        self.inserted.append(record)


def daemon_with_shadow(
    shadow: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[Daemon, FakeCollection]:
    daemon = object.__new__(Daemon)
    daemon.relation_shadow = shadow
    daemon.mongo_config_path = "configs/mongo_config.json"
    collection = FakeCollection()
    monkeypatch.setattr("nyan.daemon.get_relation_shadow_collection", lambda path: collection)
    return daemon, collection


def test_the_daemon_acts_on_the_llm_and_files_the_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow, _ = relation_shadow_with({"a": relation_answers(0.95, 0.0)})
    daemon, collection = daemon_with_shadow(shadow, monkeypatch)
    candidate = story(7, 1, "a")
    llm = Relation("unrelated")
    monkeypatch.setattr("nyan.daemon.judge_relation", lambda story, candidates: llm)

    relation = daemon.judge(story(None, 2), [candidate], "publish")  # type: ignore[list-item]

    assert relation is llm
    assert collection.inserted[0]["jev_verdict"] == "same"
    assert collection.inserted[0]["site"] == "publish"


def test_a_failing_shadow_never_reaches_the_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        enabled = True

        def __call__(self, *args: Any) -> dict:
            raise RuntimeError("TypeSafe is down")

    daemon, collection = daemon_with_shadow(Broken(), monkeypatch)
    llm = Relation("unrelated")
    monkeypatch.setattr("nyan.daemon.judge_relation", lambda story, candidates: llm)

    assert daemon.judge(story(None, 2), [story(7, 1)], "late") is llm  # type: ignore[list-item]
    assert collection.inserted == []
