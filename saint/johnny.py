"""johnny integration — the resolver + telemetry-provide (over CLI/HTTP, never import).

SAINT asks johnny "where is this role/seat, what model, is it ready?" via a focused
`resolve` call (cached, short TTL), and may trigger a non-blocking ensure-load. It also
*provides* normalized per-request telemetry to johnny's ingest spool (johnny owns that
schema). Two transports behind one protocol: CLI (subprocess `johnny`) and HTTP (johnnyd).
Any johnny failure returns None / is swallowed — SAINT degrades to its static baseline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from saint.config import JohnnyConfig


@dataclass(frozen=True)
class Resolution:
    seat: str | None
    endpoint: str | None
    model: str | None
    state: str  # ready | loading | absent | failed
    eta_s: float | None
    queue_depth: int | None


def _parse(d: dict) -> Resolution:
    return Resolution(
        seat=d.get("seat"), endpoint=d.get("endpoint"), model=d.get("model"),
        state=d.get("state", "absent"), eta_s=d.get("eta_s"), queue_depth=d.get("queue_depth"),
    )


class JohnnyResolver(Protocol):
    def resolve(self, target: str) -> Resolution | None: ...
    def ensure_load(self, target: str) -> None: ...


class _TTLCache:
    def __init__(self, ttl: float):
        self.ttl = ttl
        self._d: dict[str, tuple[float, Resolution]] = {}

    def get(self, key: str) -> Resolution | None:
        v = self._d.get(key)
        if not v:
            return None
        ts, val = v
        return val if (time.monotonic() - ts) <= self.ttl else None

    def put(self, key: str, val: Resolution) -> None:
        self._d[key] = (time.monotonic(), val)


class CliResolver:
    """v0 transport: shell out to the `johnny` CLI."""

    def __init__(self, cli_path: str = "johnny", ttl: float = 1.0):
        self.cli = cli_path
        self.cache = _TTLCache(ttl)

    def resolve(self, target: str) -> Resolution | None:
        cached = self.cache.get(target)
        if cached is not None:
            return cached
        try:
            p = subprocess.run([self.cli, "resolve", target, "--json"], capture_output=True, text=True, timeout=5)
            if p.returncode != 0 or not p.stdout.strip():
                return None
            res = _parse(json.loads(p.stdout))
        except Exception:
            return None
        self.cache.put(target, res)
        return res

    def ensure_load(self, target: str) -> None:
        try:
            subprocess.run([self.cli, "up", target, "--json"], capture_output=True, text=True, timeout=10)
        except Exception:
            pass


class HttpResolver:
    """v1 transport: johnnyd HTTP."""

    def __init__(self, base_url: str, ttl: float = 1.0):
        self.base = base_url.rstrip("/")
        self.cache = _TTLCache(ttl)

    def resolve(self, target: str) -> Resolution | None:
        cached = self.cache.get(target)
        if cached is not None:
            return cached
        try:
            with urllib.request.urlopen(f"{self.base}/resolve?target={quote(target)}", timeout=5) as r:  # noqa: S310
                res = _parse(json.loads(r.read()))
        except Exception:
            return None
        self.cache.put(target, res)
        return res

    def ensure_load(self, target: str) -> None:
        try:
            data = json.dumps({"model": target}).encode()
            req = urllib.request.Request(self.base + "/up", data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()  # noqa: S310
        except Exception:
            pass


def build_resolver(johnny: JohnnyConfig | None) -> JohnnyResolver | None:
    if johnny is None:
        return None
    if johnny.transport == "http" and johnny.base_url:
        return HttpResolver(johnny.base_url, ttl=johnny.resolve_cache_ttl_s)
    return CliResolver(johnny.cli_path, ttl=johnny.resolve_cache_ttl_s)


# --- telemetry provide: SAINT provides, johnny accepts (append-only spool) ---
def _ingest_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return Path(base) / "johnny" / "ingest"


def provide_telemetry(record: dict) -> None:
    """Append one normalized JSONL record to johnny's ingest spool. Best-effort + non-fatal
    (same discipline as _safe_log): a failure here never affects the response."""
    try:
        d = _ingest_dir()
        d.mkdir(parents=True, exist_ok=True)
        line = json.dumps({**record, "source": "proxy"}) + "\n"
        with open(d / "saint.jsonl", "a") as f:  # O_APPEND keeps small line writes atomic on local fs
            f.write(line)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[saint] telemetry provide failed: {type(e).__name__}: {e}", file=sys.stderr)
