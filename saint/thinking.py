"""Signed-thinking stripping on a backend switch.

A Claude thinking block carries a `signature` — an HMAC bound to the exact model that
minted it. If a conversation's history is replayed to a DIFFERENT backend (after a routing
reclassification or a dispatch-fallback hop), the receiving Bedrock/Anthropic backend 400s
on the foreign signature. Turn-pinning (conversation affinity) keeps a turn on its minting
backend as an optimization; this is the correctness floor when a switch happens anyway.

Two wire shapes (both handled):
1. typed content blocks: `{"type": "thinking"|"redacted_thinking", ...}` inside a message's
   content list — the /v1/messages (Anthropic) path.
2. top-level assistant keys: `thinking_blocks` / `reasoning_content` — the OpenAI chat
   completions path (litellm surfaces signed thinking there as message-level fields).

Fail-closed: when continuity can't be confirmed (no affinity entry, or the previous
backend differs), strip — losing reasoning context is cheap; a 400 is not.
"""

from __future__ import annotations

from typing import Any

_SIGNATURE_VALIDATING = ("anthropic", "bedrock")
_THINKING_BLOCK_TYPES = ("thinking", "redacted_thinking")
_TOP_LEVEL_THINKING_KEYS = ("thinking_blocks", "reasoning_content")


def should_strip(effective_provider: str, prev_backend: str | None,
                 effective_backend_name: str) -> bool:
    """Strip iff the target validates signatures AND turn continuity is unconfirmed."""
    if effective_provider not in _SIGNATURE_VALIDATING:
        return False
    return prev_backend is None or prev_backend != effective_backend_name


def strip_signed_thinking(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove signed thinking from assistant history (copy-on-write; returns the input
    object unchanged when nothing needed stripping)."""
    changed = False
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") != "assistant":
            out.append(m)
            continue
        new_m = m
        # shape 2: top-level keys
        present = [k for k in _TOP_LEVEL_THINKING_KEYS if k in m]
        if present:
            new_m = {k: v for k, v in m.items() if k not in _TOP_LEVEL_THINKING_KEYS}
            changed = True
        # shape 1: typed content blocks
        content = new_m.get("content")
        if isinstance(content, list):
            filtered = [b for b in content
                        if not (isinstance(b, dict)
                                and b.get("type") in _THINKING_BLOCK_TYPES)]
            if len(filtered) != len(content):
                new_m = {**new_m, "content": filtered}
                changed = True
        out.append(new_m)
    return out if changed else messages


# Models that accept `thinking: {"type": "adaptive"}` (Claude 5 family). Anything else gets the
# param removed — Claude Code sends adaptive for every request, and Haiku 4.5 400s on it.
_ADAPTIVE_OK = ("claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-mythos-5")


def shape_thinking_param(params: dict, provider: str, model: str) -> dict:
    """Make the request-level `thinking` param acceptable to the target:
    - non-Anthropic backends: drop it (a local seat's thinking is its own template knob; with
      it present, litellm's bridge re-routes through the OpenAI Responses API, which vLLM only
      half-implements — Claude Code's tool history 400'd there, 2026-08-27);
    - Anthropic/Bedrock models outside the Claude 5 family: drop `adaptive` (400 otherwise)."""
    th = params.get("thinking")
    if not isinstance(th, dict):
        return params
    if provider not in _SIGNATURE_VALIDATING:
        return {k: v for k, v in params.items() if k != "thinking"}
    if th.get("type") == "adaptive" and not any(t in (model or "") for t in _ADAPTIVE_OK):
        return {k: v for k, v in params.items() if k != "thinking"}
    return params


def sanitize_thinking_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop thinking blocks that can never validate at Anthropic: empty text or no
    signature. A local seat's turn (via litellm's bridge) leaves `{"type":"thinking",
    "thinking":""}` in the client's history; Anthropic rejects the whole request
    ("each thinking block must contain thinking"). Copy-on-write like strip_signed_thinking."""
    changed = False
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if m.get("role") == "assistant" and isinstance(content, list):
            keep = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "thinking" \
                        and (not (b.get("thinking") or "").strip() or not b.get("signature")):
                    changed = True
                    continue
                if isinstance(b, dict) and b.get("type") == "redacted_thinking" and not b.get("data"):
                    changed = True
                    continue
                keep.append(b)
            if len(keep) != len(content):
                m = {**m, "content": keep}
            if not keep:                       # an assistant turn can't be empty — leave a stub
                m = {**m, "content": [{"type": "text", "text": "…"}]}
                changed = True
        out.append(m)
    return out if changed else messages


def fold_system_role_messages(messages: list[dict[str, Any]], provider: str, model: str) -> list[dict[str, Any]]:
    """Claude Code sends hook context as `{"role": "system"}` entries inside `messages`
    (SessionStart "additional context"). Claude 5 models accept that; Haiku 4.5 answers
    "role 'system' is not supported on this model" (2026-08-27). For Anthropic/Bedrock
    targets outside the Claude 5 family, convert each such entry into a user message whose
    text is wrapped in <system-reminder> — the representation those models were trained on.
    Copy-on-write; other providers are left alone (chat-completions bridges accept system roles)."""
    if provider not in _SIGNATURE_VALIDATING or any(t in (model or "") for t in _ADAPTIVE_OK):
        return messages
    changed = False
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") != "system":
            out.append(m)
            continue
        changed = True
        content = m.get("content")
        text = content if isinstance(content, str) else "\n".join(
            (b.get("text") or "") for b in (content or []) if isinstance(b, dict) and b.get("type") == "text")
        out.append({"role": "user", "content": [{"type": "text",
                    "text": f"<system-reminder>\n{text}\n</system-reminder>"}]})
    return out if changed else messages
