"""Spill: a johnny_role backend borrows another role's seat when its own is saturated or not ready."""
from __future__ import annotations

from dataclasses import replace

import pytest

from saint import binding
from saint.config import BackendConfig, Config, JohnnyConfig
from saint.johnny import Resolution


def _cfg(spill=("coder",), spill_at=6):
    b = {
        "local-worker": BackendConfig(name="local-worker", provider="openai", model="m", api_key_env=None, api_key="x",
                                      base_url="http://static:8006/v1", aliases=(), timeout_s=10, johnny_role="worker",
                                      spill=spill, spill_at=spill_at),
        "local-coder": BackendConfig(name="local-coder", provider="openai", model="m", api_key_env=None, api_key="x",
                                     base_url="http://static:8002/v1", aliases=(), timeout_s=10, johnny_role="coder"),
        "cloud": BackendConfig(name="cloud", provider="anthropic", model="c", api_key_env=None, api_key="x",
                               base_url=None, aliases=(), timeout_s=10),
    }
    cfg = Config.__new__(Config)
    object.__setattr__(cfg, "backends", b)
    object.__setattr__(cfg, "johnny", JohnnyConfig(transport="cli", cli_path="johnny", base_url=None, resolve_cache_ttl_s=1, ensure_load=False))

    class R:  # routing stub
        while_loading = None
        default_on_failure = "cloud"
    object.__setattr__(cfg, "routing", R())
    return cfg


class FakeResolver:
    def __init__(self, seats):
        self.seats = seats
        self.loads_requested = []

    def resolve(self, target):
        return self.seats.get(target)

    def ensure_load(self, target):
        pass


def _ready(seat, port):
    return Resolution(seat=seat, endpoint=f"http://127.0.0.1:{port}/v1", model="m", state="ready", eta_s=None, queue_depth=None)


@pytest.fixture
def loads(monkeypatch):
    table = {}
    monkeypatch.setattr(binding, "seat_load", lambda endpoint, timeout=0.5: table.get(endpoint))
    return table


def test_no_spill_when_own_seat_has_headroom(loads):
    cfg = _cfg(); r = FakeResolver({"worker": _ready("w", 8006), "coder": _ready("c", 8002)})
    loads["http://127.0.0.1:8006/v1"] = 2; loads["http://127.0.0.1:8002/v1"] = 0
    eff = binding.resolve_for_dispatch(cfg, "local-worker", r)
    assert eff.johnny_seat == "w" and eff.spilled_from is None and eff.seat_load == 2


def test_spills_to_least_loaded_ready_seat_when_saturated(loads):
    cfg = _cfg(spill=("chat", "coder")); r = FakeResolver({"worker": _ready("w", 8006), "coder": _ready("c", 8002), "chat": _ready("h", 8003)})
    loads["http://127.0.0.1:8006/v1"] = 7; loads["http://127.0.0.1:8002/v1"] = 1; loads["http://127.0.0.1:8003/v1"] = 4
    eff = binding.resolve_for_dispatch(cfg, "local-worker", r)
    assert eff.johnny_seat == "c" and eff.spilled_from == "worker" and eff.seat_load == 1
    assert eff.backend.base_url == "http://127.0.0.1:8002/v1" and eff.state_at_dispatch == "johnny_ready"


def test_no_spill_when_alternatives_are_no_better(loads):
    cfg = _cfg(); r = FakeResolver({"worker": _ready("w", 8006), "coder": _ready("c", 8002)})
    loads["http://127.0.0.1:8006/v1"] = 8; loads["http://127.0.0.1:8002/v1"] = 8
    eff = binding.resolve_for_dispatch(cfg, "local-worker", r)
    assert eff.johnny_seat == "w" and eff.spilled_from is None


def test_unknown_load_never_spills(loads):
    cfg = _cfg(); r = FakeResolver({"worker": _ready("w", 8006), "coder": _ready("c", 8002)})
    # own load unreadable → treated as unknown, not as saturated
    loads["http://127.0.0.1:8002/v1"] = 0
    eff = binding.resolve_for_dispatch(cfg, "local-worker", r)
    assert eff.johnny_seat == "w" and eff.spilled_from is None and eff.seat_load is None


def test_spill_beats_fallback_when_own_seat_not_ready(loads):
    cfg = _cfg(); r = FakeResolver({"worker": Resolution(seat="w", endpoint=None, model=None, state="loading", eta_s=30, queue_depth=None),
                                    "coder": _ready("c", 8002)})
    loads["http://127.0.0.1:8002/v1"] = 3
    eff = binding.resolve_for_dispatch(cfg, "local-worker", r)
    assert eff.johnny_seat == "c" and eff.spilled_from == "worker" and eff.state_at_dispatch == "johnny_ready"


def test_role_without_spill_never_borrows(loads):
    cfg = _cfg(); r = FakeResolver({"worker": _ready("w", 8006), "coder": _ready("c", 8002)})
    loads["http://127.0.0.1:8002/v1"] = 9; loads["http://127.0.0.1:8006/v1"] = 0
    eff = binding.resolve_for_dispatch(cfg, "local-coder", r)   # coder has no spill list here
    assert eff.johnny_seat == "c" and eff.spilled_from is None


def test_spill_columns_logged(tmp_path):
    from saint.storage import LogRow, log_request, open_db, spill_stats
    conn = open_db(tmp_path / "log.sqlite")
    base = dict(request_id="r", model_field="saint-local-worker", prefixes_raw=None, pinned_backend=None, urgency_used="normal",
                classifier_used=None, classifier_fallback_reason=None, classifier_input_chars=None, classifier_input_truncated_from=None,
                classifier_latency_ms=None, classifier_domain=None, classifier_complexity=None, classifier_reason=None,
                backend_chosen="local-worker", backend_latency_ms=1, tokens_in=1, tokens_out=1, success=True, error_kind=None,
                prompt_content=None, prompt_storage_mode="none", state_at_dispatch="johnny_ready")
    log_request(conn, LogRow(**base, johnny_seat="w", seat_load=2))
    log_request(conn, LogRow(**base, johnny_seat="c", spilled_from="worker", spilled_to="c", seat_load=1))
    [row] = spill_stats(conn)
    assert row["backend"] == "local-worker" and row["requests"] == 2 and row["spilled"] == 1 and row["spilled_to"] == {"c": 1}
