"""Drain: a draining seat takes no new requests — its role spills elsewhere and it is never a spill target."""
import json

import pytest

from saint import binding, drain
from saint.johnny import Resolution
from tests.test_spill import FakeResolver, _cfg, _ready


@pytest.fixture
def drainfile(tmp_path, monkeypatch):
    f = tmp_path / "drain.json"; monkeypatch.setenv("SAINT_DRAIN_FILE", str(f)); drain._cache.update(mtime=None, checked=0.0, set=frozenset())
    return f


@pytest.fixture
def loads(monkeypatch):
    table = {}
    monkeypatch.setattr(binding, "seat_load", lambda endpoint, timeout=0.5: table.get(endpoint))
    return table


def test_set_and_match_by_name_port_role(drainfile):
    drain.set_drain(["johnny-x-8006"], True)
    assert drain.is_draining("johnny-x-8006")
    drain.set_drain(["8002"], True)
    assert drain.is_draining("other", endpoint="http://127.0.0.1:8002/v1")
    drain.set_drain(["worker"], True)
    assert drain.is_draining(None, role="worker")
    drain.set_drain(["8002", "worker"], False)
    assert not drain.is_draining("other", endpoint="http://127.0.0.1:8002/v1") and json.loads(drainfile.read_text())["draining"] == ["johnny-x-8006"]


def test_draining_own_seat_spills_regardless_of_load(drainfile, loads):
    cfg = _cfg(spill=("coder",)); r = FakeResolver({"worker": _ready("w", 8006), "coder": _ready("c", 8002)})
    loads["http://127.0.0.1:8006/v1"] = 0; loads["http://127.0.0.1:8002/v1"] = 5
    drain.set_drain(["w"], True)
    eff = binding.resolve_for_dispatch(cfg, "local-worker", r)
    assert eff.johnny_seat == "c" and eff.spilled_from == "worker" and eff.drained == "w"


def test_draining_seat_is_never_a_spill_target(drainfile, loads):
    cfg = _cfg(spill=("coder",)); r = FakeResolver({"worker": _ready("w", 8006), "coder": _ready("c", 8002)})
    loads["http://127.0.0.1:8006/v1"] = 9; loads["http://127.0.0.1:8002/v1"] = 0
    drain.set_drain(["c"], True)
    eff = binding.resolve_for_dispatch(cfg, "local-worker", r)
    assert eff.johnny_seat == "w" and eff.spilled_from is None      # saturated, but the only alternative is draining


def test_draining_with_no_alternative_falls_back(drainfile, loads):
    cfg = _cfg(spill=()); r = FakeResolver({"worker": _ready("w", 8006)})
    drain.set_drain(["8006"], True)
    eff = binding.resolve_for_dispatch(cfg, "local-worker", r)
    assert eff.johnny_seat != "w" or eff.state_at_dispatch != "johnny_ready"
    assert eff.drained == "w"


def test_file_edits_are_picked_up_without_restart(drainfile, loads):
    cfg = _cfg(spill=("coder",)); r = FakeResolver({"worker": _ready("w", 8006), "coder": _ready("c", 8002)})
    loads["http://127.0.0.1:8006/v1"] = 0; loads["http://127.0.0.1:8002/v1"] = 0
    assert binding.resolve_for_dispatch(cfg, "local-worker", r).johnny_seat == "w"
    drainfile.write_text(json.dumps({"draining": ["w"]})); drain._cache["checked"] = 0.0
    assert binding.resolve_for_dispatch(cfg, "local-worker", r).johnny_seat == "c"
