import logging
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, cast
from collections.abc import Sequence
from multiprocessing.pool import ThreadPool

from openai import OpenAI


@dataclass
class OpenAIDecodingArguments:
    max_tokens: int = 2400
    top_p: float = 0.95
    n: int = 1
    stream: bool = False
    stop: Sequence[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def as_params(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


DEFAULT_ARGS = OpenAIDecodingArguments()


def env_str(name: str, default: str) -> str:
    # docker-compose forwards unset variables as empty strings, which a
    # getenv default would not replace.
    return os.getenv(name) or default


def env_number(name: str, default: str) -> float:
    value = env_str(name, default)
    try:
        return float(value)
    except ValueError:
        logging.warning("Invalid %s=%r, using %s", name, value, default)
        return float(default)


# Model id passed to the gateway. Override via the LLM_MODEL env var to switch
# models without a code change / redeploy.
DEFAULT_MODEL = env_str("LLM_MODEL", "openai/gpt-5.4-mini")

# OpenAI-compatible endpoint. Defaults to OpenRouter, but can be pointed at a
# self-hosted gateway (e.g. LiteLLM / OmniRoute) via the LLM_BASE_URL env var.
LLM_BASE_URL = env_str("LLM_BASE_URL", "https://openrouter.ai/api/v1")

# API key for the OpenAI-compatible gateway. Prefer the generic LLM_API_KEY so
# the credential follows LLM_BASE_URL; falls back to OPENROUTER_API_KEY for
# backwards compatibility with existing deployments.
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")

# The daemon is a synchronous loop, so a slow request stalls the whole feed.
# These calls ask for one headline and a few short differences, so a minute is
# already generous — and the bound has to be multiplied by the retries below to
# see the real worst case.
LLM_TIMEOUT = env_number("LLM_TIMEOUT", "60")

# Transport-level retries (429s, 5xx, connection resets) are handled by the
# SDK with proper backoff; the loop below only handles errors that need the
# request itself to change. Two, not three: with the timeout above this caps a
# single cluster at about three minutes of waiting instead of eight.
LLM_MAX_RETRIES = int(env_number("LLM_MAX_RETRIES", "2"))

# How many times a single completion may be rewritten and resent before giving
# up. Each attempt must make progress (drop a parameter, shrink the output),
# so this is a safety net rather than a rate limit.
MAX_ATTEMPTS = 6

# Reasoning budget for the short analysis/digest calls. The recovery below can
# discover that a gateway rejects the parameter, but that costs one failed
# request per process; setting LLM_REASONING_EFFORT=none skips it up front.
_DISABLED_VALUES = frozenset(("none", "off", "no", "0", "disabled"))


def env_reasoning_effort(name: str = "LLM_REASONING_EFFORT") -> str | None:
    value = env_str(name, "low").strip()
    return None if value.lower() in _DISABLED_VALUES else value


DEFAULT_REASONING_EFFORT = env_reasoning_effort()

_client: OpenAI | None = None

# Parameters a given model has already rejected, remembered per process so a
# gateway limitation costs one failed request in total rather than one per
# call. Populated from the errors described below.
_unsupported_params: dict[str, set[str]] = {}

# Gateways in front of the model name the parameters they do not implement:
#   litellm.UnsupportedParamsError: custom_openai does not support
#   parameters: ['reasoning_effort']
_UNSUPPORTED_LIST_RE = re.compile(r"does not support parameters:\s*\[([^\]]*)\]")
# OpenAI itself reports them one at a time:
#   Unsupported parameter: 'temperature' is not supported with this model.
_UNSUPPORTED_SINGLE_RE = re.compile(
    r"[Uu]nsupported parameter:\s*'([^']+)'|'([^']+)' is not supported with this model"
)


def get_client() -> OpenAI:
    """The shared client. Kept lazy so importing this module needs no API key."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            timeout=LLM_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
        )
    return _client


def parse_unsupported_params(error: str) -> set[str]:
    """Parameter names a gateway or model complained about, if any."""
    names: set[str] = set()
    match = _UNSUPPORTED_LIST_RE.search(error)
    if match:
        names |= {
            name.strip().strip("'\"") for name in match.group(1).split(",") if name.strip()
        }
    for groups in _UNSUPPORTED_SINGLE_RE.findall(error):
        names |= {name for name in groups if name}
    return names


def openai_completion(
    messages: list[dict[str, Any]],
    decoding_args: OpenAIDecodingArguments = DEFAULT_ARGS,
    model_name: str = DEFAULT_MODEL,
    response_format: dict[str, str] | None = None,
    reasoning_effort: str | None = None,
) -> str:
    assert decoding_args.n == 1

    params: dict[str, Any] = decoding_args.as_params()
    # Only forward response_format/reasoning_effort when explicitly requested,
    # so behavior stays opt-in per call and does not leak into unrelated
    # completions. Reasoning models (e.g. gpt-5.x) otherwise default to a
    # higher effort and silently burn hidden reasoning tokens on simple tasks.
    if response_format is not None:
        params["response_format"] = response_format
    if reasoning_effort is not None:
        params["reasoning_effort"] = reasoning_effort

    known_unsupported = _unsupported_params.get(model_name, set())
    for name in known_unsupported & set(params):
        params.pop(name)

    prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
    logging.info(
        "LLM call: model=%s, prompt_chars=%d, reasoning_effort=%s",
        model_name,
        prompt_chars,
        params.get("reasoning_effort"),
    )

    client = get_client()
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            completion = client.chat.completions.create(
                # The SDK types messages as a union of per-role TypedDicts;
                # callers here build plain dicts, which are equivalent at
                # runtime but not assignable.
                messages=cast(Any, messages),
                model=model_name,
                **params,
            )
            content = completion.choices[0].message.content
            return cast(str, content).strip() if content else ""
        except Exception as e:
            last_error = e
            error = str(e)

            if not rewrite_params(params, error, model_name):
                logging.error("LLM call failed: %s", error)
                raise
            # Recoverable: rewrite_params already logged what it changed, so the
            # raw error is context rather than a problem in its own right.
            # No sleep: every branch above changes the request itself, and
            # rate limits are the SDK's job (LLM_MAX_RETRIES).
            logging.info("LLM error (attempt %d), retrying: %s", attempt + 1, error)

    assert last_error is not None
    raise last_error


def rewrite_params(params: dict[str, Any], error: str, model_name: str) -> bool:
    """Adjust `params` in place so a retry can succeed. False if it cannot.

    Every branch must change the request, otherwise retrying just repeats the
    same failure.
    """
    if "Please reduce" in error:
        for name in ("max_tokens", "max_completion_tokens"):
            if name in params:
                params[name] = int(params[name] * 0.8)
                logging.warning("Reducing %s to %d, retrying", name, params[name])
                return True
        return False

    # Reasoning models replaced max_tokens with max_completion_tokens, but
    # gateways still accept the old name, so we send it and rename on demand.
    if "max_completion_tokens" in error and "max_tokens" in params:
        params["max_completion_tokens"] = params.pop("max_tokens")
        logging.warning("Renaming max_tokens to max_completion_tokens, retrying")
        return True

    unsupported = parse_unsupported_params(error) & set(params)
    if unsupported:
        _unsupported_params.setdefault(model_name, set()).update(unsupported)
        for name in unsupported:
            params.pop(name)
        logging.warning(
            "Model %s rejects %s, retrying without them",
            model_name,
            sorted(unsupported),
        )
        return True

    return False


def openai_batch_completion(
    batch: list[list[dict[str, Any]]],
    decoding_args: OpenAIDecodingArguments = DEFAULT_ARGS,
    model_name: str = DEFAULT_MODEL,
    max_workers: int = 8,
) -> list[str]:
    if not batch:
        return []
    # One thread per item saturates the gateway on large batches and makes
    # rate-limit backoff useless.
    with ThreadPool(min(len(batch), max_workers)) as pool:
        return list(
            pool.starmap(
                openai_completion,
                [(messages, decoding_args, model_name) for messages in batch],
            )
        )
