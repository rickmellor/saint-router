"""Runtime fixes for litellm bugs that hit SAINT before upstream does.

Applied once from `saint.backends` (imported by both the server and the CLI).
Each patch is idempotent and checks that the attribute it replaces still exists,
so a litellm upgrade that renames it fails loudly in tests instead of silently.
"""
from __future__ import annotations

import copy
from typing import Any, List

_APPLIED = False


def _split_reasoning_from_content(chunk: Any) -> List[Any]:
    """`[reasoning_chunk, content_chunk]` if one OpenAI chunk carries both
    `reasoning_content` and `content`/`tool_calls`, else `[chunk]`.

    Seen 2026-10-19 (litellm 1.91.0, vLLM reasoning parser on the Flash-Next seats): at
    the reasoning→answer boundary a single chunk holds the reasoning tail ("\\n") AND the
    first answer token. The Messages bridge picks the new block's type with content
    first (text) but builds the delta with reasoning first (thinking_delta), dropping
    the text — so the stream has a thinking_delta inside a text block. Claude Code
    aborts the turn ("API Error: Content block is not a thinking block", ~1 in 5 pool
    jobs) and the first answer token is lost either way."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return [chunk]
    delta = getattr(choices[0], "delta", None)
    if delta is None or not getattr(delta, "reasoning_content", None):
        return [chunk]
    if not (getattr(delta, "content", None) or getattr(delta, "tool_calls", None)):
        return [chunk]
    reasoning = copy.deepcopy(chunk)
    reasoning.choices[0].finish_reason = None
    reasoning.choices[0].delta.content = None
    if hasattr(reasoning.choices[0].delta, "tool_calls"):
        reasoning.choices[0].delta.tool_calls = None
    content = copy.deepcopy(chunk)
    content.choices[0].delta.reasoning_content = None
    return [reasoning, content]


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    from litellm.llms.anthropic.experimental_pass_through.adapters import streaming_iterator as si

    splitter = si._CombinedChunkSplitter        # AttributeError here = litellm moved it; re-check the bug
    original = splitter.__dict__["_split"].__func__

    def _split(chunk: Any) -> List[Any]:
        out: List[Any] = []
        for part in _split_reasoning_from_content(chunk):
            out.extend(original(part))          # then the stock content/finish_reason split
        return out

    splitter._split = staticmethod(_split)
    _APPLIED = True
