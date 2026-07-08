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


class Breaker:
    """Per-backend circuit breaker for dispatch failures.

    After `threshold` consecutive failures a backend's circuit opens for `cooldown_s`:
    while open, dispatch skips straight to the backend's on_error fallback (when one is
    configured — with no fallback the primary is still tried). Any success closes it."""

    def __init__(self, threshold: int, cooldown_s: float):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}
        self._auth_flagged: set[str] = set()

    def record_failure(self, backend: str) -> None:
        n = self._failures.get(backend, 0) + 1
        self._failures[backend] = n
        if n >= self.threshold:
            self._open_until[backend] = time.monotonic() + self.cooldown_s

    def open_now(self, backend: str, cooldown_s: float) -> None:
        """Force the circuit open — auth failure: retrying is pointless until the
        credentials refresh, so skip the threshold and use the (longer) auth cooldown."""
        self._failures[backend] = self.threshold
        self._open_until[backend] = time.monotonic() + cooldown_s
        self._auth_flagged.add(backend)

    def is_auth_flagged(self, backend: str) -> bool:
        return backend in self._auth_flagged

    def record_success(self, backend: str) -> None:
        self._failures.pop(backend, None)
        self._open_until.pop(backend, None)
        self._auth_flagged.discard(backend)

    def is_open(self, backend: str) -> bool:
        until = self._open_until.get(backend)
        if until is None:
            return False
        if time.monotonic() >= until:
            # cooldown elapsed: half-open — allow a try; failure re-opens via threshold
            self._open_until.pop(backend, None)
            self._failures[backend] = self.threshold - 1
            return False
        return True


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
