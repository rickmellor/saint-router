from __future__ import annotations

import os
from typing import Any

import litellm

from saint.config import BackendConfig
from saint.route_cache import content_text


def _mark_content(content: Any, cc: dict[str, Any]) -> Any:
    """Copy `content` into Anthropic block form with cache_control on the last text
    block. Never mutates the input. Returns the input unchanged if there is no text."""
    if isinstance(content, str):
        if not content:
            return content
        return [{"type": "text", "text": content, "cache_control": cc}]
    if isinstance(content, list):
        last_text = max((i for i, p in enumerate(content)
                         if isinstance(p, dict) and p.get("type") == "text" and p.get("text")),
                        default=None)
        if last_text is None:
            return content
        out = [dict(p) if isinstance(p, dict) else p for p in content]
        out[last_text]["cache_control"] = cc
        return out
    return content


def inject_cache_control(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    min_chars: int,
    ttl: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Rolling Anthropic prompt-cache breakpoints (copy-on-write; inputs untouched).

    Marks the first system message and the last text-bearing message with
    cache_control — because the breakpoint sits on the *last* message, every
    agent-loop turn re-marks the new tail and the previous prefix becomes a cache
    read. When there is no system message, the last tool is marked instead (in
    Anthropic request order tools precede system precede messages, so a system
    breakpoint already covers tools). <= 2 of Anthropic's 4 breakpoints used.

    Skips entirely (returns inputs unchanged) below `min_chars` of total message
    text: cache writes cost 1.25x, and short prompts are below Anthropic's
    cacheable minimum anyway.
    """
    total = sum(len(content_text(m.get("content"))) for m in messages)
    if total < min_chars:
        return messages, tools

    cc: dict[str, Any] = {"type": "ephemeral"}
    if ttl:
        cc["ttl"] = ttl

    out = list(messages)
    sys_idx = next((i for i, m in enumerate(out) if m.get("role") == "system"), None)
    if sys_idx is not None:
        marked = _mark_content(out[sys_idx].get("content"), cc)
        if marked is not out[sys_idx].get("content"):
            out[sys_idx] = {**out[sys_idx], "content": marked}

    # rolling breakpoint: last message with actual text (skips content=None
    # assistant tool-call messages and pure-image turns)
    for i in range(len(out) - 1, -1, -1):
        if content_text(out[i].get("content")):
            marked = _mark_content(out[i].get("content"), cc)
            if marked is not out[i].get("content"):
                out[i] = {**out[i], "content": marked}
            break

    out_tools = tools
    if sys_idx is None and tools:
        out_tools = [*tools[:-1], {**tools[-1], "cache_control": cc}]
    return out, out_tools


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
    if b.provider == "bedrock":
        # model is an inference-profile id, e.g. global.anthropic.claude-sonnet-5
        return f"bedrock/{b.model}"
    # Unknown provider: pass through as-is and let the user/LiteLLM error speak.
    return b.model


def _provider_kwargs(backend: BackendConfig) -> dict[str, Any]:
    """Auth/endpoint kwargs per provider. Bedrock uses the AWS credential chain
    (aws_profile → credential_process/SSO via saint.bedrock_auth's patched path),
    never a static api_key."""
    if backend.provider == "bedrock":
        kw: dict[str, Any] = {"aws_region_name": backend.aws_region}
        if backend.aws_profile:
            kw["aws_profile_name"] = backend.aws_profile
        return kw
    kw = {"api_key": _resolve_api_key(backend)}
    if backend.base_url:
        kw["api_base"] = backend.base_url
    return kw


def _shape_request(backend: BackendConfig, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Per-backend request shaping: drop model-forbidden params; inject a default
    max_tokens only when the client omitted one (Bedrock/Anthropic require it)."""
    for k in backend.drop_params:
        kwargs.pop(k, None)
    if backend.default_max_tokens is not None:
        kwargs.setdefault("max_tokens", backend.default_max_tokens)
    return kwargs


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
        **_provider_kwargs(backend),
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if extra_params:
        for k, v in extra_params.items():
            kwargs.setdefault(k, v)
    return await litellm.acompletion(**_shape_request(backend, kwargs))


async def call_embeddings(backend: BackendConfig, input: Any) -> Any:
    """Dispatch an embeddings request (str or list of str) to a backend."""
    kwargs: dict[str, Any] = {
        "model": _resolve_model_id(backend),
        "input": input,
        "timeout": backend.timeout_s,
        **_provider_kwargs(backend),
    }
    return await litellm.aembedding(**kwargs)
