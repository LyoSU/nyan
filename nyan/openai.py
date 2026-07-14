import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence, List, Dict, Any, cast
from multiprocessing.pool import ThreadPool

import openai
import copy


@dataclass
class OpenAIDecodingArguments:
    max_tokens: int = 2400
    top_p: float = 0.95
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


DEFAULT_ARGS = OpenAIDecodingArguments()

# Model id passed to OpenRouter. Override via the LLM_MODEL env var to switch
# models without a code change / redeploy.
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-5.4-mini")

# OpenAI-compatible endpoint. Defaults to OpenRouter, but can be pointed at a
# self-hosted gateway (e.g. LiteLLM / OmniRoute) via the LLM_BASE_URL env var.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

# API key for the OpenAI-compatible gateway. Prefer the generic LLM_API_KEY so
# the credential follows LLM_BASE_URL; falls back to OPENROUTER_API_KEY for
# backwards compatibility with existing deployments.
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")


def openai_completion(
    messages: List[Dict[str, Any]],
    decoding_args: OpenAIDecodingArguments = DEFAULT_ARGS,
    model_name: str = DEFAULT_MODEL,
    sleep_time: int = 2,
    response_format: Optional[Dict[str, str]] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    decoding_args = copy.deepcopy(decoding_args)
    assert decoding_args.n == 1

    # Configure OpenAI client for the configured OpenAI-compatible gateway
    openai.api_base = LLM_BASE_URL
    openai.api_key = LLM_API_KEY

    # Only forward response_format/reasoning_effort when explicitly requested,
    # so behavior stays opt-in per call and does not leak into unrelated
    # completions. Reasoning models (e.g. gpt-5.x) otherwise default to a
    # higher effort and silently burn hidden reasoning tokens on simple tasks.
    extra_args: Dict[str, Any] = {}
    if response_format is not None:
        extra_args["response_format"] = response_format
    if reasoning_effort is not None:
        extra_args["reasoning_effort"] = reasoning_effort

    while True:
        try:
            completions = openai.ChatCompletion.create(
                messages=messages,
                model=model_name,
                **decoding_args.__dict__,
                **extra_args,
            )
            break
        except Exception as e:
            logging.warning("OpenAI error: %s.", e)
            if "Please reduce" in str(e):
                decoding_args.max_tokens = int(decoding_args.max_tokens * 0.8)
                logging.warning(
                    "Reducing target length to %d, Retrying...",
                    decoding_args.max_tokens,
                )
            else:
                raise e
    return cast(str, completions.choices[0].message.content.strip())


def openai_batch_completion(
    batch: List[List[Dict[str, Any]]],
    decoding_args: OpenAIDecodingArguments = DEFAULT_ARGS,
    model_name: str = DEFAULT_MODEL,
    sleep_time: int = 2,
) -> List[str]:
    completions = []
    with ThreadPool(len(batch)) as pool:
        results = pool.starmap(
            openai_completion,
            [(messages, decoding_args, model_name, sleep_time) for messages in batch],
        )
        for result in results:
            completions.append(result)
    return completions
