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
