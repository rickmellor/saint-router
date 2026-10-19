"""Capability vocabulary + detection. "text" is implicit and never listed.

A turn that carries non-text content must only reach a backend that can serve it. The
vocabulary is plain strings in one registry so a new modality is one dict entry, never a
new `if image` branch somewhere in the dispatch path.
"""

from __future__ import annotations

from typing import Any

# content-block type → capability a backend must have to accept it. Extend here only.
BLOCK_CAPABILITIES: dict[str, str] = {
    # OpenAI chat-completions content parts
    "image_url": "vision",
    "input_audio": "audio",
    "file": "document",
    # Anthropic Messages blocks
    "image": "vision",
    "document": "document",
}
KNOWN_CAPABILITIES: frozenset[str] = frozenset(BLOCK_CAPABILITIES.values())


def _scan_blocks(blocks: Any, out: set[str]) -> None:
    """Union in the capabilities the listed blocks need, descending into Anthropic
    `tool_result` content (an image riding back inside a tool result still needs vision)."""
    if not isinstance(blocks, list):
        return
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if not isinstance(btype, str):
            continue                    # odd shape: a list/dict "type" must not raise here
        cap = BLOCK_CAPABILITIES.get(btype)
        if cap:
            out.add(cap)
        elif btype == "tool_result":
            _scan_blocks(block.get("content"), out)


def required_capabilities(messages: list[dict[str, Any]]) -> frozenset[str]:
    """Capabilities the WHOLE conversation needs (every message, any role — the full history is
    replayed to the backend, so an image three turns back still needs vision). Looks inside
    content-block lists and, for Anthropic `tool_result` blocks, inside their `content` list.
    str content, text/tool_use/tool_result/thinking blocks contribute nothing. Never raises on
    odd shapes (non-dict blocks, missing keys)."""
    out: set[str] = set()
    if not isinstance(messages, list):
        return frozenset()
    for m in messages:
        if isinstance(m, dict):
            _scan_blocks(m.get("content"), out)
    return frozenset(out)


def parse_capabilities(value: Any) -> frozenset[str] | None:
    """johnny's `capabilities` value → set, or None = unknown.
    Accepts {"modalities": [...]} (dict), or a bare list/tuple/set of strings. "text" is dropped.
    None / non-parseable / dict without a list under "modalities" → None."""
    if isinstance(value, dict):
        value = value.get("modalities")
    if not isinstance(value, (list, tuple, set, frozenset)):
        return None
    return frozenset(str(v) for v in value if isinstance(v, str) and v != "text")


def missing(required: frozenset[str], available: frozenset[str] | None) -> frozenset[str]:
    """required − available; available None (unknown) counts as empty (text-only)."""
    return required - (frozenset() if available is None else available)
