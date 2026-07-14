from types import SimpleNamespace
from typing import Any, Dict

from nyan.openai import openai_completion


def _fake_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_openai_completion_forwards_reasoning_effort(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: Dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _fake_response("ok")

    monkeypatch.setattr("nyan.openai.openai.ChatCompletion.create", fake_create)

    result = openai_completion(
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="low",
    )

    assert result == "ok"
    assert captured["reasoning_effort"] == "low"


def test_openai_completion_omits_reasoning_effort_by_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: Dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _fake_response("ok")

    monkeypatch.setattr("nyan.openai.openai.ChatCompletion.create", fake_create)

    openai_completion(messages=[{"role": "user", "content": "hi"}])

    assert "reasoning_effort" not in captured
