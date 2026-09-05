from __future__ import annotations

import os
from typing import Any

import litellm

# Drop OpenAI-style params a target model rejects instead of failing the dispatch.
# Seen 2026-09-04: litellm raised UnsupportedParamsError("claude-sonnet-5 does not
# support temperature") on a pinned cloud-medium request, and the on_error fallback
# then served it from the LOCAL seat — a silent tier downgrade. Per-backend
# `drop_params` in config.toml still applies on top (explicit blocks only; the auto
# Anthropic ladder has none), this is the catch-all.
litellm.drop_params = True

import re

from saint.config import BackendConfig

_CLAUDE5_NO_TEMPERATURE = re.compile(r"claude-(sonnet|opus|fable|haiku)-5(\b|[-.])")
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


def split_volatile(
    messages: list[dict[str, Any]], sentinel: str,
) -> tuple[list[dict[str, Any]], str | None]:
    """Extract per-turn volatile context marked by `sentinel` in the first system message.

    Everything after the sentinel is removed from the system message (the marker with it)
    and returned as `volatile`. The caller relocates it to the tail via `append_volatile`
    AFTER cache_control is placed, so it never sits inside the cached prefix. Copy-on-write;
    inputs untouched. Returns (messages, None) when there's no sentinel to act on.
    """
    if not sentinel:
        return messages, None
    for i, m in enumerate(messages):
        if m.get("role") != "system":
            continue
        content = m.get("content")
        text = content if isinstance(content, str) else content_text(content)
        if not text or sentinel not in text:
            continue
        head, _, tail = text.partition(sentinel)
        head, tail = head.rstrip(), tail.strip()
        out = [dict(x) for x in messages]
        if head:
            out[i] = {**out[i], "content": head}   # list/image system content collapses to
        else:                                       # its text head — a non-issue for real callers
            out.pop(i)
        return out, (tail or None)
    return messages, None


def append_volatile(
    messages: list[dict[str, Any]], volatile: str,
) -> list[dict[str, Any]]:
    """Append `volatile` as a trailing text block on the last user message (a new user
    message if there is none). Placed after any cache_control block, so it rides *after*
    the last cache breakpoint — present for the model, invisible to the cache prefix.
    Copy-on-write."""
    block = {"type": "text", "text": volatile}
    out = [dict(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            content = out[i].get("content")
            if isinstance(content, list):
                out[i]["content"] = [*content, block]
            elif isinstance(content, str) and content:
                out[i]["content"] = [{"type": "text", "text": content}, block]
            else:
                out[i]["content"] = [block]
            return out
    out.append({"role": "user", "content": [block]})
    return out


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
    # Anthropic rejects `temperature` outright on the Claude 5 family ("`temperature` is
    # deprecated for this model", invalid_request_error) and litellm's drop_params table
    # doesn't know that yet, so every cloud-medium/-large/-flagship dispatch that carried a
    # client temperature 400'd and fell through to on_error (seen 2026-09-04). Strip it
    # ourselves for those models; the auto-ladder backends have no config block to set
    # drop_params on.
    if backend.provider in ("anthropic", "bedrock") and _CLAUDE5_NO_TEMPERATURE.search(kwargs.get("model") or backend.model or ""):
        kwargs.pop("temperature", None)
    if backend.default_max_tokens is not None:
        kwargs.setdefault("max_tokens", backend.default_max_tokens)
    # chat_template_kwargs is a vLLM/llama.cpp extension, not an OpenAI param: litellm only
    # delivers it inside `extra_body`, and Anthropic/Bedrock reject it outright.
    ctk = kwargs.pop("chat_template_kwargs", None)
    if ctk and backend.provider not in ("anthropic", "bedrock"):
        extra = dict(kwargs.get("extra_body") or {})
        extra.setdefault("chat_template_kwargs", ctk)
        kwargs["extra_body"] = extra
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


async def call_backend_messages(backend: BackendConfig, *, params: dict[str, Any],
                                stream: bool) -> Any:
    """Dispatch an Anthropic Messages API request via litellm's anthropic interface.

    Native passthrough for anthropic/bedrock backends; openai-compatible backends go
    through litellm's messages→completion translation. Returns an
    AnthropicMessagesResponse, or an async iterator of Anthropic-format SSE bytes when
    stream=True."""
    model_id = _resolve_model_id(backend)
    if backend.provider == "openai" and backend.base_url:
        # litellm's Messages bridge sends `openai/…` models to the OpenAI *Responses* API
        # (vLLM's implementation rejects Claude Code's tool/system history with pydantic
        # union errors); `hosted_vllm/…` keeps it on plain chat completions (2026-08-27).
        model_id = f"hosted_vllm/{backend.model}"
    kwargs: dict[str, Any] = {
        "model": model_id,
        "timeout": backend.timeout_s,
        **_provider_kwargs(backend),
        **params,
    }
    return await litellm.anthropic.messages.acreate(
        stream=stream, **_shape_request(backend, kwargs))


async def call_embeddings(backend: BackendConfig, input: Any) -> Any:
    """Dispatch an embeddings request (str or list of str) to a backend."""
    kwargs: dict[str, Any] = {
        "model": _resolve_model_id(backend),
        "input": input,
        "timeout": backend.timeout_s,
        **_provider_kwargs(backend),
    }
    return await litellm.aembedding(**kwargs)
