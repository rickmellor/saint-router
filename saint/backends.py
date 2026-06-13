from __future__ import annotations

import os
from typing import Any

import litellm

from saint.config import BackendConfig


def _resolve_api_key(b: BackendConfig) -> str | None:
    if b.api_key:
        return b.api_key
    if b.api_key_env:
        return os.environ.get(b.api_key_env)
    return None


def _resolve_model_id(b: BackendConfig) -> str:
    """Translate (provider, model) → litellm model id.

    LiteLLM requires an explicit provider prefix on the model name; it does NOT
    infer the provider from `api_base` alone. Without a prefix, LiteLLM raises
    `BadRequestError: LLM Provider NOT provided` (and prints a "Provider List"
    link to its docs). We always prefix.
    """
    if b.provider == "anthropic":
        return f"anthropic/{b.model}"
    if b.provider == "openai":
        # Works for OpenAI itself AND any OpenAI-compatible endpoint reached via
        # api_base (LM Studio, OpenRouter, vLLM, llama.cpp server, etc.).
        return f"openai/{b.model}"
    # Unknown provider: pass through as-is and let the user/LiteLLM error speak.
    return b.model


async def call_backend(
    backend: BackendConfig,
    *,
    messages: list[dict[str, Any]],
    stream: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    extra_params: dict[str, Any] | None = None,
) -> Any:
    """Dispatch a chat completion to a backend via the LiteLLM SDK.

    For stream=True, returns an async generator of chunks (litellm's CustomStreamWrapper).
    For stream=False, returns the response object.
    """
    kwargs: dict[str, Any] = {
        "model": _resolve_model_id(backend),
        "messages": messages,
        "stream": stream,
        "timeout": backend.timeout_s,
        "api_key": _resolve_api_key(backend),
    }
    if backend.base_url:
        kwargs["api_base"] = backend.base_url
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if extra_params:
        for k, v in extra_params.items():
            kwargs.setdefault(k, v)
    return await litellm.acompletion(**kwargs)
