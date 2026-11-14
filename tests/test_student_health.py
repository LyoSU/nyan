from typing import Any

import httpx
import pytest

from nyan.student_health import alert, problems, summarize


def post(
    best: float | None, model: str = "e5_multi_h3c", text: str | None = "текст"
) -> dict[str, Any]:
    doc: dict[str, Any] = {"patched_text": text}
    if best is not None:
        doc["category_scores"] = {"war": best, "politics": 1 - best}
        doc["student"] = {"model": model}
    return doc


def test_a_normal_day_raises_nothing() -> None:
    health = summarize([post(0.95)] * 900 + [post(0.6)] * 100)

    assert health.posts == 1000
    assert health.unsure_share == pytest.approx(0.1)
    assert problems(health) == []


def test_posts_without_text_are_not_counted() -> None:
    """They get no category from anything, so they say nothing about the model."""
    health = summarize([post(0.95)] * 300 + [post(None, text=None)] * 300)

    assert health.posts == 300
    assert problems(health) == []


def test_a_host_that_lost_the_model_is_reported() -> None:
    """The older head leaves scores but no student record."""
    old_head = {"patched_text": "текст", "category_scores": {"war": 0.9}}
    health = summarize([post(0.95)] * 500 + [old_head] * 500)

    assert health.unlabelled_share == pytest.approx(0.5)
    assert any("not loaded" in problem for problem in problems(health))


def test_a_model_unsure_of_too_much_is_reported() -> None:
    health = summarize([post(0.95)] * 700 + [post(0.55)] * 300)

    assert any("unsure" in problem for problem in problems(health))


def test_two_models_at_once_are_reported() -> None:
    """A half-finished rollout: one container on the new model, one on the old."""
    health = summarize([post(0.95)] * 400 + [post(0.95, model="e5_multi")] * 400)

    assert any("different models" in problem for problem in problems(health))


def test_a_quiet_window_is_reported_as_quiet_not_as_a_bad_model() -> None:
    found = problems(summarize([post(0.55)] * 20))

    assert len(found) == 1 and "only 20 posts" in found[0]


def test_alert_is_skipped_without_a_bot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALERT_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ALERT_CHAT_ID", raising=False)

    assert alert("x") is False


def test_alert_posts_to_the_configured_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALERT_BOT_TOKEN", "T")
    monkeypatch.setenv("ALERT_CHAT_ID", "42")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    assert alert("щось не так", transport=httpx.MockTransport(handler)) is True
    assert seen[0].url.path == "/botT/sendMessage"
    assert b'"chat_id":"42"' in seen[0].content.replace(b" ", b"")
