"""In-memory routing caches for agent-loop traffic.

Agent clients (pi, hermes, opencode) turn one user prompt into N chat-completion
calls: the last user message stays constant while assistant/tool turns accumulate.
Two small caches exploit that:

- **Turn cache**: classifier labels keyed by (classifier input, urgency). The first
  turn of a loop classifies; the rest reuse the labels. Kills both the repeated
  classifier latency and label flap (a nondeterministic LLM classifier re-rolling
  medium→hard mid-loop and escalating one turn to cloud).
- **Conversation affinity**: the conversation's last labels keyed by
  hash(first system message + first non-system message) — OpenRouter's sticky-routing
  key — or an explicit client-supplied session id. Short follow-ups ("yes, do that")
  inherit the conversation's labels instead of classifying out of context.

Caches store LABELS, never resolved backends: policy, urgency and johnny liveness
still resolve on every request. Entries are ~100 bytes; memory is negligible.

Single-process by design: state lives on the server's app.state and uvicorn runs one
event loop. Races are benign (worst case a redundant classification).
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


def content_text(content: Any) -> str:
    """Text of an OpenAI-style message content (str or content-block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(parts)
    return ""


@dataclass(frozen=True)
class CachedLabels:
    domain: str
    complexity: str
    reason: str
    source: str  # the classifier_used that originally produced these labels


@dataclass(frozen=True)
class ConversationEntry:
    labels: CachedLabels
    backend: str  # last intended backend (observability; policy re-resolves anyway)


class TTLCache:
    """LRU + TTL cache on a monotonic clock. Lazy expiry on get, size-bound on set."""

    def __init__(self, ttl_s: float, max_entries: int):
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if time.monotonic() >= expires_at:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        self._data[key] = (time.monotonic() + self.ttl_s, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


@dataclass(frozen=True)
class RouteCaches:
    turns: TTLCache | None          # None when [cache].turn_cache = false
    conversations: TTLCache | None  # None when [cache].conversation_affinity = false


def turn_key(clf_input: str, urgency: str) -> str:
    return hashlib.sha256(f"{urgency}\x00{clf_input}".encode()).hexdigest()


def conversation_key(messages: list[dict[str, Any]], session_id: str | None) -> str | None:
    """Conversation identity: explicit session id wins (x-session-id semantics); else
    hash(first system message + first non-system message) — stable as turns append."""
    if session_id:
        return hashlib.sha256(f"sid\x00{session_id}".encode()).hexdigest()
    first_system = next((m for m in messages if m.get("role") == "system"), None)
    first_non_system = next((m for m in messages if m.get("role") != "system"), None)
    if first_non_system is None:
        return None  # nothing stable to key on
    sys_text = content_text(first_system.get("content")) if first_system else ""
    first_text = content_text(first_non_system.get("content"))
    return hashlib.sha256(f"{sys_text}\x00{first_text}".encode()).hexdigest()
