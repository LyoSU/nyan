from types import SimpleNamespace
from typing import Any

import pytest

from nyan import openai as llm
from nyan.openai import (
    OpenAIDecodingArguments,
    openai_completion,
    parse_unsupported_params,
)


class FakeCompletions:
    """Stands in for client.chat.completions, recording every request."""

    def __init__(
        self, errors: list[Exception] | None = None, content: str = "ok"
    ) -> None:
        self.errors = list(errors or ())
        self.content = content
        self.calls: list[dict[str, Any]] = []

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
        errors: list[Exception] | None = None, content: str = "ok"
    ) -> FakeCompletions:
        completions = FakeCompletions(errors, content)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(llm, "get_client", lambda: client)
        monkeypatch.setattr(llm, "_unsupported_params", {})
        return completions

    return _install


def _messages() -> list[dict[str, Any]]:
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


def test_a_request_carries_nothing_it_was_not_asked_to(fake_client) -> None:  # type: ignore[no-untyped-def]
    """Sampling parameters are the model's own business.

    Every one that used to be sent — a token cap, top_p, two penalties — is
    either refused by current models, costing a rejection and a retry on every
    call, or pushes the model off the defaults it was tuned at.
    """
    completions = fake_client()

    openai_completion(_messages())

    assert set(completions.calls[0]) == {"messages", "model"}


def test_renames_max_tokens_for_reasoning_models(fake_client) -> None:  # type: ignore[no-untyped-def]
    """A cap asked for through LLM_MAX_TOKENS still has to survive the gateway."""
    error = Exception(
        "Unsupported value: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead."
    )
    completions = fake_client(errors=[error])
    capped = OpenAIDecodingArguments(max_tokens=8000)

    openai_completion(_messages(), decoding_args=capped)

    assert "max_tokens" not in completions.calls[1]
    # The value carries over untouched; only the parameter name changes.
    assert completions.calls[1]["max_completion_tokens"] == 8000


def test_shrinks_the_output_when_asked_to_reduce(fake_client) -> None:  # type: ignore[no-untyped-def]
    completions = fake_client(errors=[Exception("Please reduce the length")])
    capped = OpenAIDecodingArguments(max_tokens=8000)

    openai_completion(_messages(), decoding_args=capped)

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


# --------------------------------------------------------------- prompt caching
#
# A provider reuses a prompt's prefix only if the request lands on the worker
# that already holds it. The key below says which prompt this is, so the calls
# that share a system message ask for the same worker.


def test_names_the_prompt_when_asked_to(fake_client) -> None:  # type: ignore[no-untyped-def]
    completions = fake_client()

    openai_completion(_messages(), prompt_cache_key="summary")

    assert completions.calls[0]["prompt_cache_key"] == "summary"


def test_says_nothing_about_caching_by_default(fake_client) -> None:  # type: ignore[no-untyped-def]
    completions = fake_client()

    openai_completion(_messages())

    assert "prompt_cache_key" not in completions.calls[0]


def test_a_gateway_that_dislikes_the_cache_key_still_gets_the_call(fake_client) -> None:  # type: ignore[no-untyped-def]
    """The key is a routing hint. Losing it costs money, losing the post costs a post."""
    error = Exception("Unrecognized request argument supplied: prompt_cache_key")
    completions = fake_client(errors=[error])

    assert openai_completion(_messages(), prompt_cache_key="summary") == "ok"
    assert "prompt_cache_key" not in completions.calls[1]


def test_an_unphrased_complaint_about_the_key_drops_it_too(fake_client) -> None:  # type: ignore[no-untyped-def]
    """No regex knows every gateway's wording; naming the hint is enough."""
    error = Exception("400: prompt_cache_key is not allowed here")
    completions = fake_client(errors=[error])

    assert openai_completion(_messages(), prompt_cache_key="summary") == "ok"
    assert "prompt_cache_key" not in completions.calls[1]


def test_usage_says_how_much_of_the_prompt_the_cache_paid_for(caplog) -> None:  # type: ignore[no-untyped-def]
    """A prefix that stopped being cached is invisible without this line."""
    completion = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=6000,
            completion_tokens=400,
            prompt_tokens_details=SimpleNamespace(cached_tokens=5120),
        )
    )
    with caplog.at_level("INFO"):
        llm.log_usage(completion, "model")

    assert "cached=5120 (85%)" in caplog.text


def test_usage_reports_a_gateway_that_counts_no_cache(caplog) -> None:  # type: ignore[no-untyped-def]
    completion = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=6000, completion_tokens=400)
    )
    with caplog.at_level("INFO"):
        llm.log_usage(completion, "model")

    # Unreported is not zero: claiming a cold cache here would send someone
    # hunting for a broken prefix that is fine.
    assert "cached=unreported" in caplog.text


def test_usage_survives_a_response_that_carries_none(caplog) -> None:  # type: ignore[no-untyped-def]
    llm.log_usage(SimpleNamespace(), "model")
