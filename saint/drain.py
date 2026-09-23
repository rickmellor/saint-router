"""Drain: stop routing NEW requests to a seat while it stays up, so in-flight work finishes and johnny can then
take it down cleanly. State lives in a small JSON file next to the config (edited by `saint drain`, johnny, or by
hand) and is re-read on every dispatch (mtime-cached), so no router restart is needed — restarts under a fan-out are
exactly what drain exists to avoid.

Entries match a seat by name (johnny container name), by port ("8006"), or by johnny role ("worker").
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def drain_file() -> Path:
    env = os.environ.get("SAINT_DRAIN_FILE")
    return Path(env) if env else Path(os.path.expanduser("~/.config/saint/drain.json"))


_cache: dict = {"path": None, "mtime": None, "checked": 0.0, "set": frozenset()}


def load_drain() -> frozenset[str]:
    p = drain_file(); now = time.monotonic()
    if str(p) != _cache["path"]:                       # path changed (tests, env) → drop everything cached
        _cache.update(path=str(p), mtime=None, checked=0.0, set=frozenset())
    if now - _cache["checked"] < 1.0:
        return _cache["set"]
    _cache["checked"] = now
    try:
        mtime = p.stat().st_mtime
    except OSError:
        _cache.update(mtime=None, set=frozenset()); return _cache["set"]
    if mtime == _cache["mtime"]:
        return _cache["set"]
    try:
        data = json.loads(p.read_text() or "{}")
        names = frozenset(str(x) for x in (data.get("draining") or []))
    except Exception:
        names = frozenset()
    _cache.update(mtime=mtime, set=names)
    return names


def is_draining(seat: str | None, endpoint: str | None = None, role: str | None = None, drain: frozenset[str] | None = None) -> bool:
    d = load_drain() if drain is None else drain
    if not d:
        return False
    keys = {k for k in (seat, role) if k}
    if endpoint:
        port = endpoint.rstrip("/").rsplit(":", 1)[-1].split("/", 1)[0]
        keys.add(port)
    return bool(keys & d)


def set_drain(names: list[str], on: bool) -> frozenset[str]:
    p = drain_file(); p.parent.mkdir(parents=True, exist_ok=True)
    cur = set(load_drain())
    cur = (cur | set(names)) if on else (cur - set(names))
    p.write_text(json.dumps({"draining": sorted(cur), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2) + "\n")
    _cache["checked"] = 0.0
    return frozenset(cur)
