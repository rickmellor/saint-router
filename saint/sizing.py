"""Rough request sizing for decisions that must happen BEFORE the seat counts tokens.

The router has no tokenizer on this path (see `/tokenize` in server.py, which does reach a
seat's real one), so anything that needs a size up front — a spill cold-prefill guard — works
from characters. Measured ~3.1 chars/token on Claude Code traffic once thinking text is
excluded (see `_SKIP_KEYS`); 3.0 errs slightly high, the safe side for a context guard, and is
deliberately compared against generous limits: this is a guard against pointless cold prefills
and guaranteed context-overflow failures, not a billing number.
"""

from __future__ import annotations

from typing import Any

_CHARS_PER_TOKEN = 3.0

# Skipped outright — none of it is text the seat tokenises:
#   `data`       — base64 payloads (an image/audio/document); a string starting with `data:` is
#                  the inline-URL form of the same thing. Both are enormous, and counting them
#                  would size an image-only turn as if it were a novel.
#   `thinking`   — earlier turns' reasoning text, which the chat template drops. 70 % of a long
#                  Claude Code body (measured 2026-10-22) and none of the real prompt.
#   `signature`  — opaque thinking-block signatures.
# A key named `thinking` whose value is a dict — the top-level request option
# `"thinking": {"type": "enabled", "budget_tokens": N}` — is skipped too; fine, it sizes ~0.
_SKIP_KEYS = frozenset({"data", "thinking", "signature"})


def _walk(node: Any, total: int) -> int:
    if isinstance(node, str):
        return total if node.startswith("data:") else total + len(node)
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _SKIP_KEYS:
                continue
            total = _walk(value, total)
        return total
    if isinstance(node, (list, tuple)):
        for item in node:
            total = _walk(item, total)
        return total
    return total


def estimate_request_tokens(body: dict) -> int:
    """Estimated prompt tokens for a request body: every string value in the JSON counted by
    length, base64 payloads and thinking text skipped, divided by 3.0. Non-dict/None input
    estimates 0."""
    if not isinstance(body, dict):
        return 0
    return int(_walk(body, 0) / _CHARS_PER_TOKEN)


def requested_output_tokens(body: dict) -> int:
    """What the client asked to get back: `max_tokens`, else `max_completion_tokens`, else 0.
    Anything that is not an int (absent, null, a string) is 0 — an unknown output budget must
    never be the reason a seat is skipped."""
    if not isinstance(body, dict):
        return 0
    for key in ("max_tokens", "max_completion_tokens"):
        value = body.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0
