from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from nyan import openai as llm
from nyan.openai import openai_completion, parse_unsupported_params


class FakeCompletions:
    """Stands in for client.chat.completions, recording every request."""

    def __init__(
        self, errors: Optional[List[Exception]] = None, content: str = "ok"
    ) -> None:
        self.errors = list(errors or ())
        self.content = content
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


@pytest.fixture
def fake_client(monkeypatch):  # type: ignore[no-untyped-def]
    def _install(
        errors: Optional[List[Exception]] = None, content: str = "ok"
    ) -> FakeCompletions:
        completions = FakeCompletions(errors, content)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(llm, "get_client", lambda: client)
        monkeypatch.setattr(llm, "_unsupported_params", {})
        monkeypatch.setattr(llm.time, "sleep", lambda _: None)
        return completions

    return _install


def _messages() -> List[Dict[str, Any]]:
    return [{"role": "user", "content": "hi"}]


def test_forwards_reasoning_effort(fake_client) -> None:  # type: ignore[no-untyped-def]
    completions = fake_client()

    assert openai_completion(_messages(), reasoning_effort="low") == "ok"
    assert completions.calls[0]["reasoning_effort"] == "low"


def test_omits_reasoning_effort_by_default(fake_client) -> None:  # type: ignore[no-untyped-def]
    completions = fake_client()

    openai_completion(_messages())

    assert "reasoning_effort" not in completions.calls[0]


def test_parse_unsupported_params_reads_a_litellm_list() -> None:
    error = (
        "litellm.UnsupportedParamsError: custom_openai does not support "
        "parameters: ['reasoning_effort'], for model=gpt-5.6-luna."
    )
    assert parse_unsupported_params(error) == {"reasoning_effort"}


def test_parse_unsupported_params_reads_an_openai_single_name() -> None:
    error = "Unsupported parameter: 'temperature' is not supported with this model."
    assert parse_unsupported_params(error) == {"temperature"}


def test_retries_without_a_parameter_the_gateway_rejects(fake_client) -> None:  # type: ignore[no-untyped-def]
    error = Exception(
        "litellm.UnsupportedParamsError: custom_openai does not support "
        "parameters: ['reasoning_effort'], for model=gpt-5.6-luna."
    )
    completions = fake_client(errors=[error])

    assert openai_completion(_messages(), reasoning_effort="low") == "ok"
    assert len(completions.calls) == 2
    assert completions.calls[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in completions.calls[1]


def test_remembers_the_rejection_so_the_next_call_costs_nothing(fake_client) -> None:  # type: ignore[no-untyped-def]
    error = Exception(
        "custom_openai does not support parameters: ['reasoning_effort']"
    )
    completions = fake_client(errors=[error])

    openai_completion(_messages(), reasoning_effort="low")
    openai_completion(_messages(), reasoning_effort="low")

    # Two failed-then-retried calls would be four requests; learning makes three.
    assert len(completions.calls) == 3
    assert "reasoning_effort" not in completions.calls[2]


def test_renames_max_tokens_for_reasoning_models(fake_client) -> None:  # type: ignore[no-untyped-def]
    error = Exception(
        "Unsupported value: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead."
    )
    completions = fake_client(errors=[error])

    openai_completion(_messages())

    assert "max_tokens" not in completions.calls[1]
    assert completions.calls[1]["max_completion_tokens"] == 2400


def test_shrinks_the_output_when_asked_to_reduce(fake_client) -> None:  # type: ignore[no-untyped-def]
    completions = fake_client(errors=[Exception("Please reduce the length")])

    openai_completion(_messages())

    assert completions.calls[1]["max_tokens"] < completions.calls[0]["max_tokens"]


def test_raises_errors_it_cannot_recover_from(fake_client) -> None:  # type: ignore[no-untyped-def]
    completions = fake_client(errors=[Exception("Invalid API key")])

    with pytest.raises(Exception, match="Invalid API key"):
        openai_completion(_messages())

    # Retrying an unrecoverable error would just burn the quota.
    assert len(completions.calls) == 1


def test_batch_completion_bounds_its_thread_pool(fake_client) -> None:  # type: ignore[no-untyped-def]
    completions = fake_client()

    results = llm.openai_batch_completion([_messages() for _ in range(20)])

    assert results == ["ok"] * 20
    assert len(completions.calls) == 20


def test_empty_batch_starts_no_pool(fake_client) -> None:  # type: ignore[no-untyped-def]
    fake_client()
    assert llm.openai_batch_completion([]) == []
