"""Anthropic Messages API (/v1/messages) adaptation layer.

Claude Code (and any Anthropic-SDK client) speaks the Messages API: top-level `system`,
content-block messages, Anthropic-schema tools, required `max_tokens`. SAINT routes these
with the same brain as chat completions and dispatches via litellm's Anthropic interface
(`litellm.anthropic.messages.acreate`) — native passthrough for anthropic/bedrock
backends, translation for openai-compatible seats.

Two adaptation problems live here:

1. **Routing view** (`openai_view`): decide_route reads OpenAI-shaped messages. Anthropic
   `tool_result` blocks — which Claude Code sends as the LAST USER MESSAGE on every tool
   turn — would trip the router's multimodal detection (any non-text block) and misroute
   entire agent loops. They are neutralized to text here, and multimodal detection is done
   Anthropic-aware (`detect_multimodal`: image/document blocks only, including nested
   inside tool_result content) and passed to decide_route as an override.
2. **Dispatch params** (`build_dispatch_params`): whitelist of Messages-API fields
   forwarded verbatim. The client's own `cache_control` blocks pass through untouched —
   Claude Code manages its own cache breakpoints; SAINT never injects on this surface.
"""

from __future__ import annotations

from typing import Any

from saint.config import Config, resolve_backend
from saint.route_cache import content_text

# Messages-API fields forwarded to the backend verbatim (model/stream handled explicitly).
MESSAGES_FORWARDED_PARAMS = (
    "system",
    "tools",
    "tool_choice",
    "thinking",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "metadata",
)


def messages_model_field(cfg: Config, model: str) -> str:
    """Map the client's model string to a saint model field.

    saint-* names pass through; a backend name/alias hit (raw Bedrock inference-profile
    ids are expected aliases, e.g. 'global.anthropic.claude-opus-4-8') pins that backend;
    anything else — including 'auto' and unknown claude-* ids — routes via the classifier.
    Claude Code sends model ids we don't control, so unknown must never be an error."""
    if model.startswith("saint-"):
        return model
    b = resolve_backend(cfg, model)
    if b is not None:
        return f"saint-{b.name}"
    return "saint-auto"


def _block_text(block: Any) -> str:
    if isinstance(block, dict) and block.get("type") == "text":
        return block.get("text", "")
    return ""


def _neutralize_block(block: Any) -> Any:
    """tool_result → plain text block (its textual content), so the router's generic
    non-text-block multimodal check never fires on agent tool turns."""
    if isinstance(block, dict) and block.get("type") == "tool_result":
        inner = block.get("content")
        if isinstance(inner, str):
            text = inner
        elif isinstance(inner, list):
            text = "\n".join(_block_text(b) for b in inner if _block_text(b))
        else:
            text = ""
        return {"type": "text", "text": text}
    return block


def openai_view(body: dict) -> list[dict[str, Any]]:
    """Messages-like list for decide_route: [system-as-message?] + anthropic messages
    with tool_result blocks neutralized. Read-only adaptation — never dispatched."""
    view: list[dict[str, Any]] = []
    system = body.get("system")
    if system:
        sys_text = system if isinstance(system, str) else "\n".join(
            _block_text(b) for b in system if _block_text(b))
        view.append({"role": "system", "content": sys_text})
    for m in body.get("messages", []):
        content = m.get("content")
        if isinstance(content, list):
            content = [_neutralize_block(b) for b in content]
        view.append({"role": m.get("role"), "content": content})
    return view


def detect_multimodal(messages: list[dict]) -> bool:
    """Anthropic-aware multimodal check on the LAST user message: image/document blocks
    count (including nested inside tool_result content); tool_use/tool_result/text don't."""
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user is None or not isinstance(last_user.get("content"), list):
        return False

    def _is_media(block: Any) -> bool:
        if not isinstance(block, dict):
            return False
        if block.get("type") in ("image", "document"):
            return True
        if block.get("type") == "tool_result" and isinstance(block.get("content"), list):
            return any(isinstance(b, dict) and b.get("type") in ("image", "document")
                       for b in block["content"])
        return False

    return any(_is_media(b) for b in last_user["content"])


def strip_prefix_from_messages(messages: list[dict], raw: str) -> list[dict]:
    """Remove a parsed !/@ routing prefix from the last user message before dispatch
    (copy-on-write). Handles str content and the first text block of block content."""
    if not raw:
        return messages
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") != "user":
            continue
        content = out[i].get("content")
        if isinstance(content, str) and content.startswith(raw):
            out[i] = {**out[i], "content": content[len(raw):].lstrip(" ")}
        elif isinstance(content, list):
            blocks = list(content)
            for j, b in enumerate(blocks):
                if isinstance(b, dict) and b.get("type") == "text":
                    if b.get("text", "").startswith(raw):
                        blocks[j] = {**b, "text": b["text"][len(raw):].lstrip(" ")}
                        out[i] = {**out[i], "content": blocks}
                    break
        break
    return out


def build_dispatch_params(body: dict) -> dict[str, Any]:
    """Whitelisted Messages-API params for litellm's anthropic interface."""
    params: dict[str, Any] = {
        "messages": body.get("messages", []),
        "max_tokens": body["max_tokens"],
    }
    for k in MESSAGES_FORWARDED_PARAMS:
        if k in body and body[k] is not None:
            params[k] = body[k]
    return params


def anthropic_usage(response: Any) -> dict:
    """Usage dict from an Anthropic-format response (dict or pydantic)."""
    if isinstance(response, dict):
        return response.get("usage") or {}
    if hasattr(response, "model_dump"):
        return (response.model_dump().get("usage")) or {}
    return {}


class SseUsageTracker:
    """Extract usage from an Anthropic SSE stream while relaying bytes untouched.

    Anthropic streams carry usage in two events:
      - message_start: data.message.usage — input_tokens, cache_creation_input_tokens,
        cache_read_input_tokens (and an initial output_tokens)
      - message_delta: data.usage.output_tokens — cumulative; take the max
    Parsing failures must never kill the relay — this is accounting, not correctness."""

    def __init__(self):
        self.usage: dict[str, int] = {}
        self._buf = b""

    def feed(self, chunk: Any) -> None:
        try:
            self._buf += chunk if isinstance(chunk, bytes) else str(chunk).encode()
            while b"\n\n" in self._buf:
                block, self._buf = self._buf.split(b"\n\n", 1)
                self._parse(block)
        except Exception:
            pass

    def _bump(self, key: str, value: Any) -> None:
        if isinstance(value, int):
            self.usage[key] = max(self.usage.get(key, 0), value)

    def _parse(self, block: bytes) -> None:
        import json as _json
        for line in block.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            try:
                d = _json.loads(line[5:].strip())
            except Exception:
                continue
            t = d.get("type")
            if t == "message_start":
                u = (d.get("message") or {}).get("usage") or {}
                for k in ("input_tokens", "cache_creation_input_tokens",
                          "cache_read_input_tokens", "output_tokens"):
                    self._bump(k, u.get(k))
            elif t == "message_delta":
                self._bump("output_tokens", (d.get("usage") or {}).get("output_tokens"))


def anthropic_error(status: int, err_type: str, message: str):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=status, content={
        "type": "error", "error": {"type": err_type, "message": message}})


def last_user_text(body: dict) -> str:
    """Joined text of the last user message (for logging parity with chat)."""
    for m in reversed(body.get("messages", [])):
        if m.get("role") == "user":
            return content_text(m.get("content"))
    return ""
