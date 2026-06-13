"""Tests for the johnny integration (add-johnny-integration)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from saint.binding import resolve_for_dispatch
from saint.config import load_config
from saint.johnny import Resolution
from saint.storage import SCHEMA_VERSION, _load_migration, open_db, schema_version

_CELLS = "\n".join(f'"{d},{c}" = "cloud-small"' for d in ("code", "general") for c in ("trivial", "medium", "hard"))


def _write_config(tmp_path: Path, *, while_loading: bool = True, johnny_block: bool = True,
                  johnny_only_backend: bool = True) -> Path:
    coder_wl = '\nwhile_loading = "cloud-small"' if while_loading else ""
    vision = (
        '\n[backends.managed-vision]\njohnny_seat = "vision"\njohnny_only = true\n'
        if johnny_only_backend else ""
    )
    routing_wl = '\nwhile_loading = "cloud-small"' if while_loading else ""
    jblock = (
        '\n[johnny]\ntransport = "cli"\ncli_path = "johnny"\nresolve_cache_ttl_s = 1.0\nensure_load = true\n'
        if johnny_block else ""
    )
    toml = f"""
[server]
host = "127.0.0.1"
port = 4000

[backends.cloud-small]
provider = "anthropic"
model = "claude-haiku"
api_key_env = "ANTHROPIC_API_KEY"

[backends.local-coder]
provider = "openai"
model = "static-x"
base_url = "http://127.0.0.1:1234/v1"
johnny_role = "coder"{coder_wl}
{vision}
[classifier]
backend = "cloud-small"

[routing]
default_urgency = "normal"
default_on_failure = "cloud-small"{routing_wl}

[routing.policy.normal]
{_CELLS}
[routing.policy.urgent]
{_CELLS}
[routing.policy.patient]
{_CELLS}
{jblock}
[logging]
db_path = "/tmp/saint-test.db"
prompt_storage = "none"
"""
    p = tmp_path / "config.toml"
    p.write_text(toml)
    return p


class _Stub:
    def __init__(self, res: Resolution | None):
        self.res = res
        self.loaded: list[str] = []
        self.resolve_calls = 0

    def resolve(self, target: str):
        self.resolve_calls += 1
        return self.res

    def ensure_load(self, target: str):
        self.loaded.append(target)


# --- config ---
def test_johnny_binding_parsed(tmp_path: Path):
    cfg = load_config(_write_config(tmp_path))
    b = cfg.backends["local-coder"]
    assert b.johnny_bound and b.johnny_target == "coder"
    assert cfg.backends["managed-vision"].johnny_only is True
    assert cfg.johnny is not None and cfg.johnny.transport == "cli"


def test_bound_without_johnny_block_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="block is required"):
        load_config(_write_config(tmp_path, johnny_block=False))


def test_non_johnny_only_requires_static(tmp_path: Path):
    # a bound backend missing base_url + not johnny_only -> rejected
    toml = _write_config(tmp_path).read_text().replace(
        'base_url = "http://127.0.0.1:1234/v1"\n', ""
    )
    p = tmp_path / "c2.toml"
    p.write_text(toml)
    with pytest.raises(ValueError, match="static base_url"):
        load_config(p)


# --- binding / routing ---
def test_ready_overrides_static(tmp_path: Path):
    cfg = load_config(_write_config(tmp_path))
    stub = _Stub(Resolution("coder@boxA", "http://127.0.0.1:8002/v1", "qwen3-coder", "ready", None, 0))
    eff = resolve_for_dispatch(cfg, "local-coder", stub)
    assert eff.state_at_dispatch == "johnny_ready"
    assert eff.backend.base_url == "http://127.0.0.1:8002/v1"
    assert eff.backend.model == "qwen3-coder"
    assert eff.johnny_seat == "coder@boxA"


def test_loading_serves_while_loading_and_triggers_load(tmp_path: Path):
    cfg = load_config(_write_config(tmp_path))
    stub = _Stub(Resolution("coder@boxA", None, None, "loading", 40, None))
    eff = resolve_for_dispatch(cfg, "local-coder", stub)
    assert eff.backend.name == "cloud-small"
    assert eff.state_at_dispatch == "while_loading"
    assert stub.loaded == ["coder"]  # non-blocking ensure-load fired


def test_unreachable_falls_to_static_baseline_when_no_while_loading(tmp_path: Path):
    cfg = load_config(_write_config(tmp_path, while_loading=False))
    stub = _Stub(None)  # johnny unreachable
    eff = resolve_for_dispatch(cfg, "local-coder", stub)
    assert eff.backend.name == "local-coder"
    assert eff.state_at_dispatch == "static_baseline"


def test_johnny_only_no_static_floor_uses_default(tmp_path: Path):
    cfg = load_config(_write_config(tmp_path, while_loading=False))
    stub = _Stub(Resolution(None, None, None, "absent", None, None))
    eff = resolve_for_dispatch(cfg, "managed-vision", stub)
    assert eff.backend.name == "cloud-small"  # default_on_failure
    assert eff.state_at_dispatch == "fallback"


def test_standalone_unbound_backend_makes_no_johnny_call(tmp_path: Path):
    cfg = load_config(_write_config(tmp_path))
    stub = _Stub(Resolution("x", "y", "z", "ready", None, None))
    eff = resolve_for_dispatch(cfg, "cloud-small", stub)  # cloud-small is unbound
    assert eff.state_at_dispatch is None and eff.johnny_seat is None
    assert stub.resolve_calls == 0  # never asks johnny for an unbound backend


# --- storage migration ---
def test_migration_v1_to_v2_roundtrip(tmp_path: Path):
    db_path = tmp_path / "log.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(_load_migration("0001_init.sql"))
    conn.execute("PRAGMA user_version = 1")
    conn.close()
    conn2 = open_db(db_path)  # should migrate 1 -> current
    assert schema_version(conn2) == SCHEMA_VERSION
    cols = {r[1] for r in conn2.execute("PRAGMA table_info(requests)")}
    assert "johnny_seat" in cols and "state_at_dispatch" in cols


# --- telemetry provide ---
def test_provide_telemetry_spool(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    from saint import johnny as J

    J.provide_telemetry({"seat": "coder@boxA", "latency_ms": 12, "ttft_ms": 40})
    spool = tmp_path / "state" / "johnny" / "ingest" / "saint.jsonl"
    assert spool.exists()
    import json
    rec = json.loads(spool.read_text().strip())
    assert rec["seat"] == "coder@boxA" and rec["source"] == "proxy"


def test_provide_telemetry_non_fatal(monkeypatch, capsys):
    # unwritable ingest dir -> must not raise, only stderr
    monkeypatch.setenv("XDG_STATE_HOME", "/proc/cannot-write-here")
    from saint import johnny as J

    J.provide_telemetry({"seat": "x"})  # should not raise
