"""hybrid-usage/1: attribution columns, the client label, and the /usage/v1 report.

Contract: /home/rick/hybrid-usage/contract.md (canonical copy in neurio/hsp-arch-tools).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from saint import usage as U
from saint.config import (
    BackendConfig,
    ClassifierConfig,
    ClientRule,
    Config,
    EnergyConfig,
    LoggingConfig,
    RoutingConfig,
    ServerConfig,
)
from saint.server import build_app
from saint.storage import SCHEMA_VERSION, build_log_row, log_request, open_db, schema_version

# ------------------------------------------------------------------ fixtures

_ANCHOR = datetime(2026, 10, 20, tzinfo=UTC)   # pinned so the window math is deterministic


def _ts(day: int, hour: int = 12, month: int = 10) -> str:
    """A logged ts: ISO-8601 UTC with the +00:00 offset server.py writes."""
    return _ANCHOR.replace(day=day, month=month, hour=hour).isoformat()


def _cfg() -> Config:
    """Local seats (one with a bench tok/s, one without), one backend per billing family, and a
    cloud tier with no prices. [energy] and the seat watts are pinned so money is exact."""
    def openai(name, **kw):
        base = dict(provider="openai", model=f"{name}-model", api_key=None,
                    base_url="http://x/v1", aliases=(), timeout_s=60)
        base.update(kw)
        return BackendConfig(name=name, **base)

    backends = {
        # johnny-bound seat with a bench decode rate -> billing local, energy priced
        "local-coder": openai("local-coder", johnny_role="coder", tok_s=100.0, watts=600.0),
        # named local*, no tok_s and no log samples -> local with NO energy model
        "local-pool": openai("local-pool"),
        "cloud-anthropic": BackendConfig(
            name="cloud-anthropic", provider="anthropic", model="claude-opus-5-5",
            api_key_env="ANTHROPIC_API_KEY", api_key=None, base_url=None, aliases=("opus",),
            timeout_s=120, price_in=4.0, price_out=20.0),
        "cloud-bedrock": BackendConfig(
            name="cloud-bedrock", provider="bedrock", model="global.anthropic.claude-opus-5-5",
            api_key=None, api_key_env=None, base_url=None, aliases=(), timeout_s=120,
            aws_region="us-east-1", price_in=4.0, price_out=20.0),
        "cloud-openai": openai("cloud-openai", provider="openai", model="gpt-x",
                               base_url=None, price_in=2.0, price_out=8.0),
        "cloud-vertex": openai("cloud-vertex", provider="vertex_ai", model="gemini-x",
                               base_url=None),
        "cloud-azure": openai("cloud-azure", provider="azure", model="gpt-x", base_url=None),
        "groq-fast": openai("groq-fast", provider="groq", model="llama", base_url=None),
        "cloud-unpriced": openai("cloud-unpriced", provider="anthropic", model="claude-x",
                                 base_url=None),
    }
    policy = {u: {f"{d},{c}": "local-coder" for d in ("code", "general")
                  for c in ("trivial", "medium", "hard")}
              for u in ("normal", "urgent", "patient")}
    return Config(
        server=ServerConfig(host="127.0.0.1", port=4000),
        backends=backends,
        classifier=ClassifierConfig(backend="local-coder", fallback_backend=None,
                                    max_input_chars=8000, timeout_s=5, prompt_template_path=None),
        routing=RoutingConfig(default_urgency="normal", default_on_failure="cloud-anthropic",
                              policy=policy),
        logging=LoggingConfig(db_path=":memory:", prompt_storage="full"),
        energy=EnergyConfig(price_kwh=0.36, host_watts=1500.0, gpu_watts=300.0, seat_gpus=2),
        clients=(ClientRule(name="claude-code", user_agent="claude-cli/", backend="local-coder"),),
    )


def _insert(conn, *, ts, backend="local-coder", model_field="saint-auto", tokens_in=100,
            tokens_out=100, cache_read=None, cache_write=None, client="claude-code",
            origin=None, session=None, job_id=None, latency=None, success=True):
    conn.execute(
        "INSERT INTO requests (ts, request_id, model_field, urgency_used, backend_chosen,"
        " tokens_in, tokens_out, cache_read_tokens, cache_write_tokens, backend_latency_ms,"
        " success, prompt_storage_mode, client, origin, session, job_id)"
        " VALUES (?, 'r', ?, 'normal', ?, ?, ?, ?, ?, ?, ?, 'full', ?, ?, ?, ?)",
        (ts, model_field, backend, tokens_in, tokens_out, cache_read, cache_write, latency,
         1 if success else 0, client, origin, session, job_id))


def _db(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    return conn


def _q(conn, cfg=None, **params):
    status, body = U.query(conn, cfg or _cfg(), params)
    assert status == 200, body
    return body


# ----------------------------------------------------------------- migration

def test_migration_0005_from_v4_keeps_rows(tmp_path):
    from importlib import resources

    old = sqlite3.connect(tmp_path / "old.sqlite")
    for m in ("0001_init.sql", "0002_johnny.sql", "0003_cache_metrics.sql", "0004_spill.sql"):
        old.executescript(resources.files("saint.migrations").joinpath(m).read_text())
    old.execute("PRAGMA user_version = 4")
    old.execute("INSERT INTO requests (ts, request_id, model_field, urgency_used, backend_chosen,"
                " success, prompt_storage_mode, tokens_out)"
                " VALUES ('2026-10-01T09:00:00+00:00', 'r', 'saint-auto', 'normal',"
                " 'local-coder', 1, 'none', 42)")
    old.commit()
    old.close()

    conn = open_db(tmp_path / "old.sqlite")
    assert SCHEMA_VERSION == 5 and schema_version(conn) == 5
    cols = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
    assert {"client", "origin", "session", "job_id"} <= cols
    # the old row survives, unattributed — which is exactly what HEADERS_SINCE exists to say
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone() == (1,)
    assert conn.execute("SELECT client, origin, session, job_id FROM requests").fetchone() \
        == (None, None, None, None)


def test_build_log_row_and_insert_roundtrip(tmp_path):
    conn = _db(tmp_path)
    decision = _Decision()
    row = build_log_row(decision, model_field="saint-auto", backend_latency_ms=1, success=True,
                        error_kind=None, tokens_in=1, tokens_out=1, prompt_storage_mode="none",
                        client="claude-code", origin="GEN-PW0", session="sess-1", job_id="j03")
    from saint.storage import log_request
    log_request(conn, row)
    assert conn.execute("SELECT client, origin, session, job_id FROM requests").fetchone() \
        == ("claude-code", "GEN-PW0", "sess-1", "j03")
    # and a row built without them stays NULL
    row2 = build_log_row(decision, model_field="saint-auto", backend_latency_ms=1, success=True,
                         error_kind=None, tokens_in=1, tokens_out=1, prompt_storage_mode="none")
    log_request(conn, row2)
    assert conn.execute("SELECT client FROM requests ORDER BY id DESC LIMIT 1").fetchone() == (None,)


class _Decision:
    """Minimal RoutingDecision stand-in for build_log_row."""

    def __init__(self):
        import uuid
        self.request_id = str(uuid.uuid4())
        self.parsed = type("P", (), {"raw": None, "stripped": ""})()
        self.pinned_backend = None
        self.urgency = "normal"
        self.classifier_outcome = None
        self.classifier_result = None
        self.backend = "local-coder"
        self.last_user_content_original = None


# -------------------------------------------------------------- client_label

def test_client_label_prefers_the_rule_match():
    cfg = _cfg()
    assert U.client_label(cfg, "claude-cli/2.1.280 (external; node)") == "claude-code"


def test_client_label_rule_is_case_insensitive_and_prefix_only():
    cfg = replace(_cfg(), clients=(ClientRule(name="claude-code", user_agent="CLAUDE-CLI/",
                                              backend="local-coder"),))
    assert U.client_label(cfg, "Claude-CLI/2.1.280") == "claude-code"
    assert U.client_label(cfg, "notclaude-cli/1") == "notclaude-cli"   # prefix, not substring
    assert U.client_label(cfg, "curl/8.5.0") == "curl"


def test_client_label_rule_order_wins_first():
    cfg = replace(_cfg(), clients=(
        ClientRule(name="pi", user_agent="pi/", backend="local-coder"),
        ClientRule(name="claude-code", user_agent="claude-cli/", backend="local-coder")))
    assert U.client_label(cfg, "claude-cli/2") == "claude-code"
    assert U.client_label(cfg, "pi/0.85.2") == "pi"


def test_client_label_falls_back_to_first_product_token():
    assert U.client_label(_cfg(), "curl/8.5.0 (x86_64-pc-linux-gnu)") == "curl"
    assert U.client_label(_cfg(), "Pi/0.85.2") == "pi"          # lowercased
    assert U.client_label(_cfg(), "openai-python 1.2") == "openai-python"
    assert U.client_label(_cfg(), "") == "unknown"
    assert U.client_label(_cfg(), None) == "unknown"
    assert U.client_label(_cfg(), "   ///") == "unknown"


def test_client_label_does_not_change_routing():
    """It labels every request; the [clients] ROUTING rule still only fires when unpinned."""
    import asyncio

    from saint.router import decide_route
    cfg = _cfg()
    with patch("saint.classifier.call_backend", AsyncMock()):
        d = asyncio.run(decide_route(cfg=cfg, model_field="saint-cloud-anthropic",
                                     messages=[{"role": "user", "content": "hi"}],
                                     client_user_agent="claude-cli/2.1.280"))
    assert d.backend == "cloud-anthropic"      # the model-field pin wins, as before
    assert d.client_rule is None


# -------------------------------------------------------------- attribution

def test_attribution_reads_and_cleans_headers():
    hdrs = {
        "user-agent": "claude-cli/2.1.280",
        "x-inference-origin": "  GEN-PW0DXJ6E\t",
        "x-inference-session": "c33f\u0000-488a\u001b[0m",
        "x-inference-job": "j03\n",
    }
    a = U.attribution(hdrs, _cfg())
    assert a == {"client": "claude-code", "origin": "GEN-PW0DXJ6E",
                 "session": "c33f-488a[0m", "job_id": "j03"}


def test_attribution_truncates_and_nulls():
    hdrs = {"user-agent": "pi/0.85", "x-inference-origin": "x" * 400,
            "x-inference-session": "   ", "x-inference-job": "\x00\x01"}
    a = U.attribution(hdrs, _cfg())
    assert len(a["origin"]) == 128 and a["origin"] == "x" * 128
    assert a["session"] is None and a["job_id"] is None
    assert a["client"] == "pi"


def test_attribution_without_headers_labels_client_only():
    assert U.attribution({"user-agent": "curl/8.5"}, _cfg()) == {
        "client": "curl", "origin": None, "session": None, "job_id": None}
    assert U.attribution({}, _cfg())["client"] == "unknown"


# ------------------------------------------------------------------- tokens

def test_token_split_subtracts_cache_and_clamps():
    assert U._tokens(1000, 50, 300, 100) == {"input": 600, "cache_read": 300,
                                             "cache_creation": 100, "output": 50}
    # a backend that reports prompt tokens WITHOUT including the cache still can't go negative
    assert U._tokens(100, 0, 200, 50) == {"input": 0, "cache_read": 200,
                                          "cache_creation": 50, "output": 0}
    assert U._tokens(None, None, None, None) == {"input": 0, "cache_read": 0,
                                                 "cache_creation": 0, "output": 0}


def test_totals_always_carry_four_keys(tmp_path):
    conn = _db(tmp_path)          # no rows at all
    body = _q(conn, **{"from": "2026-10-01", "to": "2026-10-05"})
    assert sorted(body["totals"]["tokens_by_type"]) == [
        "cache_creation", "cache_read", "input", "output"]
    assert body["totals"] == {"requests": 0, "tokens_by_type": {"input": 0, "cache_read": 0,
                                                                "cache_creation": 0, "output": 0},
                              "total_tokens": 0, "energy_cost_usd": None,
                              "billed_cost_usd": None}
    assert body["data_through"] is None


# ------------------------------------------------------------------ billing

@pytest.mark.parametrize("backend,want", [
    ("local-coder", "local"),          # johnny-bound
    ("local-pool", "local"),           # name starts with local
    ("cloud-anthropic", "anthropic"),
    ("cloud-bedrock", "bedrock"),
    ("cloud-openai", "openai"),
    ("cloud-vertex", "vertex"),        # vertex_ai
    ("cloud-azure", "azure"),
    ("groq-fast", "other"),            # provider we can't name a biller for
    ("cloud-unpriced", "anthropic"),   # billing is who bills, not who is priced
    ("backend-that-vanished", "other"),
])
def test_billing_mapping(backend, want):
    assert U.billing_for(_cfg(), backend)[0] == want


def test_billing_resolves_a_renamed_backend():
    cfg = replace(_cfg(), backends={**_cfg().backends, "cloud-anthropic": replace(
        _cfg().backends["cloud-anthropic"], aliases=("opus", "flagship"))})
    assert U.billing_for(cfg, "cloud-flagship")[0] == "anthropic"


# -------------------------------------------------------------------- money

def test_money_cloud_and_local(tmp_path):
    conn = _db(tmp_path)
    # cloud: uncached 200 x $4 + 600 cache reads x $0.4 + 200 cache writes x $5 + 300 out x $20
    _insert(conn, ts=_ts(3), backend="cloud-anthropic", tokens_in=1000, tokens_out=300,
            cache_read=600, cache_write=200)
    # local: 0.5 Mtok out x (600 W x $0.36 / (100 tok/s x 3.6)) = 0.5 x $0.60
    _insert(conn, ts=_ts(3), backend="local-coder", tokens_in=900, tokens_out=500_000,
            cache_read=900, latency=1000)
    body = _q(conn, **{"from": "2026-10-01", "to": "2026-10-05"})

    by = {r["backend"]: r for r in body["by_backend"]}
    assert by["cloud-anthropic"]["billed_cost_usd"] == round((800 + 240 + 1000 + 6000) / 1e6, 4)
    assert by["cloud-anthropic"]["energy_cost_usd"] is None
    assert by["local-coder"]["energy_cost_usd"] == 0.3
    assert by["local-coder"]["billed_cost_usd"] is None      # no per-token bill, not zero
    assert by["local-coder"]["billing"] == "local" and by["local-coder"]["model"] == "local-coder-model"

    assert body["totals"]["energy_cost_usd"] == 0.3
    assert body["totals"]["billed_cost_usd"] == 0.008
    # input = 200 (cloud, after cache) + 0 (local, all cached) — never double counted
    assert body["totals"]["tokens_by_type"] == {"input": 200, "cache_read": 1500,
                                               "cache_creation": 200, "output": 500_300}


def test_money_null_when_unpriced(tmp_path):
    conn = _db(tmp_path)
    _insert(conn, ts=_ts(3), backend="cloud-unpriced", tokens_out=10)
    _insert(conn, ts=_ts(3), backend="local-pool", tokens_out=10)   # no tok_s, no log samples
    body = _q(conn, **{"from": "2026-10-01", "to": "2026-10-05"})
    assert body["totals"]["billed_cost_usd"] is None
    assert body["totals"]["energy_cost_usd"] is None
    codes = {n["code"] for n in body["notes"]}
    assert "NO_ENERGY_MODEL" in codes


def test_money_known_zero_is_not_null(tmp_path):
    conn = _db(tmp_path)
    _insert(conn, ts=_ts(3), backend="cloud-anthropic", tokens_in=0, tokens_out=0)
    body = _q(conn, **{"from": "2026-10-01", "to": "2026-10-05"})
    assert body["totals"]["billed_cost_usd"] == 0.0


def test_energy_rate_uses_log_when_config_is_silent(tmp_path):
    """No `tok_s` in config -> the p75 measured from the log, the same rate `saint savings` uses."""
    conn = _db(tmp_path)
    for _ in range(6):    # >=5 substantial samples, or the seat counts as un-costed
        _insert(conn, ts=_ts(2), backend="local-pool", tokens_out=200, latency=2000,
                client="curl")
    body = _q(conn, **{"from": "2026-10-01", "to": "2026-10-05"})
    by = {r["backend"]: r for r in body["by_backend"]}
    assert by["local-pool"]["energy_cost_usd"] is not None


# ------------------------------------------------------------------ windows

def test_window_bounds_are_midnight_utc(tmp_path):
    conn = _db(tmp_path)
    _insert(conn, ts="2026-10-01T00:00:00+00:00", client="c")       # `from` inclusive
    _insert(conn, ts="2026-09-30T23:59:59+00:00", client="c")       # one second before: out
    _insert(conn, ts="2026-10-05T23:59:59+00:00", client="c")       # `to` day, last second: in
    _insert(conn, ts="2026-10-06T00:00:00+00:00", client="c")       # `to`+1 midnight: out
    body = _q(conn, **{"from": "2026-10-01", "to": "2026-10-05"})
    assert body["totals"]["requests"] == 2
    assert body["data_through"] == "2026-10-05T23:59:59Z"


def test_defaults_to_last_thirty_days(tmp_path):
    today = datetime.now(UTC).date()
    status, body = U.query(_db(tmp_path), _cfg(), {})
    assert status == 200
    assert body["window"]["to"] == today.isoformat()
    assert body["window"]["from"] == (today - timedelta(days=29)).isoformat()
    assert "by_day" in body                       # granularity defaults to day
    assert "PARTIAL_PERIOD" in {n["code"] for n in body["notes"]}


def test_by_day_and_granularity_total(tmp_path):
    conn = _db(tmp_path)
    _insert(conn, ts=_ts(1), client="a")
    _insert(conn, ts=_ts(1), client="b", origin="GEN-1")
    _insert(conn, ts=_ts(3), client="a")          # day 2 has no traffic -> omitted
    body = _q(conn, **{"from": "2026-10-01", "to": "2026-10-05"})
    assert [d["day"] for d in body["by_day"]] == ["2026-10-01", "2026-10-03"]
    assert body["by_day"][0]["requests"] == 2
    assert {"energy_cost_usd", "billed_cost_usd", "total_tokens"} <= set(body["by_day"][0])
    total = _q(conn, **{"from": "2026-10-01", "to": "2026-10-05", "granularity": "total"})
    assert "by_day" not in total
    assert total["totals"]["requests"] == 3


def test_breakdowns_group_and_sort(tmp_path):
    conn = _db(tmp_path)
    _insert(conn, ts=_ts(1), client="claude-code", origin="GEN-A", tokens_out=10)
    _insert(conn, ts=_ts(1), client="pi", origin="GEN-A", tokens_out=20)
    _insert(conn, ts=_ts(2), client="pi", tokens_out=30)      # no origin -> its own entry
    body = _q(conn, **{"from": "2026-10-01", "to": "2026-10-05"})
    assert [c["client"] for c in body["by_client"]] == ["claude-code", "pi"]   # requests desc
    assert body["by_client"][1]["total_tokens"] == 50
    assert body["by_client"][1]["tokens_by_type"]["output"] == 50
    origins = {o["origin"]: o for o in body["by_origin"]}
    assert origins[None]["requests"] == 1 and origins["GEN-A"]["requests"] == 2


# ------------------------------------------------------------------ filters

def test_filters_each_combined_and_null_origin(tmp_path):
    conn = _db(tmp_path)
    _insert(conn, ts=_ts(1), client="claude-code", origin="GEN-A", session="s1")
    _insert(conn, ts=_ts(1), client="pi", origin="GEN-B", session="s2")
    _insert(conn, ts=_ts(2), client="pi", origin=None, session=None)   # headerless
    w = {"from": "2026-10-01", "to": "2026-10-05"}
    assert _q(conn, client="pi", **w)["totals"]["requests"] == 2
    assert _q(conn, origin="GEN-A,GEN-B", **w)["totals"]["requests"] == 2   # OR within a filter
    assert _q(conn, session="s2", **w)["totals"]["requests"] == 1
    # AND across filters
    assert _q(conn, client="pi", origin="GEN-B", **w)["totals"]["requests"] == 1
    # a request without the header never matches an origin/session filter
    assert _q(conn, origin="GEN-A", client="pi", **w)["totals"]["requests"] == 0
    assert _q(conn, client="claude-code,pi", origin="GEN-A", **w)["totals"]["requests"] == 1


def test_filters_echo_and_dedupe(tmp_path):
    conn = _db(tmp_path)
    body = _q(conn, client="pi,pi, claude-code ", **{"from": "2026-10-01", "to": "2026-10-05"})
    assert body["filters"] == {"client": ["pi", "claude-code"], "origin": None, "session": None}


def test_excludes_explain_and_embeddings(tmp_path):
    conn = _db(tmp_path)
    _insert(conn, ts=_ts(1), model_field="saint-auto")
    _insert(conn, ts=_ts(1), model_field="saint-explain", tokens_out=5000)
    _insert(conn, ts=_ts(1), model_field="saint-embeddings", tokens_in=9000)
    body = _q(conn, **{"from": "2026-10-01", "to": "2026-10-05"})
    assert body["totals"]["requests"] == 1
    assert body["totals"]["total_tokens"] == 200


# ------------------------------------------------------------------- errors

@pytest.mark.parametrize("params,code,extra", [
    ({"grain": "day"}, "UNKNOWN_PARAM", {"param": "grain",
                                         "allowed": list(U.PARAM_ORDER)}),
    ({"from": "2026-13-01", "to": "2026-10-05"}, "INVALID_PARAM", {"param": "from"}),
    ({"to": "not-a-date"}, "INVALID_PARAM", {"param": "to"}),
    ({"from": "2026-10-20", "to": "2026-10-01"}, "INVALID_PARAM", {"param": "from"}),
    ({"from": "2025-01-01", "to": "2026-10-05"}, "INVALID_PARAM", {"param": "from"}),  # >366d
    ({"granularity": "week"}, "INVALID_PARAM", {"param": "granularity",
                                                "allowed": ["day", "total"]}),
    ({"client": ""}, "INVALID_PARAM", {"param": "client"}),
    ({"origin": "GEN-A,,"}, "INVALID_PARAM", {"param": "origin"}),
    ({"session": " "}, "INVALID_PARAM", {"param": "session"}),
])
def test_error_bodies(tmp_path, params, code, extra):
    status, body = U.query(_db(tmp_path), _cfg(), params)
    assert status == 400 and body["code"] == code
    for k, v in extra.items():
        assert body[k] == v, (k, body)
    assert "error" in body


def test_366_day_window_is_allowed(tmp_path):
    status, body = U.query(_db(tmp_path), _cfg(),
                           {"from": "2025-10-05", "to": "2026-10-05"})
    assert status == 200


def test_unavailable_when_the_log_is_unreadable(tmp_path):
    conn = sqlite3.connect(":memory:")           # no requests table at all
    status, body = U.query(conn, _cfg(), {})
    assert status == 503 and body["code"] == "UNAVAILABLE"


# ------------------------------------------------------------------- notes

def test_notes_headers_since_dates_the_first_attributed_row(tmp_path):
    conn = _db(tmp_path)
    _insert(conn, ts=_ts(1, month=9), client=None)          # pre-migration row
    _insert(conn, ts=_ts(15, month=9), client="claude-code")
    body = _q(conn, **{"from": "2026-08-01", "to": "2026-10-05"})
    note = next(n for n in body["notes"] if n["code"] == "HEADERS_SINCE")
    assert "2026-09-15" in note["text"]
    # a window entirely after that date says nothing about it
    assert "HEADERS_SINCE" not in {n["code"] for n in
                                   _q(conn, **{"from": "2026-10-01", "to": "2026-10-05"})["notes"]}


def test_note_headers_since_when_nothing_is_attributed(tmp_path):
    conn = _db(tmp_path)
    _insert(conn, ts=_ts(1), client=None)
    note = next(n for n in _q(conn, **{"from": "2026-10-01", "to": "2026-10-05"})["notes"]
                if n["code"] == "HEADERS_SINCE")
    assert "no attributed rows" in note["text"]


def test_notes_shape_and_no_partial_period_for_a_closed_window(tmp_path):
    body = _q(_db(tmp_path), **{"from": "2026-10-01", "to": "2026-10-05"})
    assert all({"code", "text"} == set(n) for n in body["notes"])
    assert body["notes"] == []


# -------------------------------------------------------------- http surface

_ANTHROPIC_OK = {
    "id": "msg_01", "type": "message", "role": "assistant", "model": "m",
    "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
    "usage": {"input_tokens": 40, "output_tokens": 5},
}
_CHAT_OK = {
    "id": "x", "object": "chat.completion", "created": 0, "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 7, "completion_tokens": 2},
}
_HEADERS = {"X-Inference-Origin": "GEN-PW0DXJ6E", "X-Inference-Session": "c33f-488a",
            "X-Inference-Job": "sa-20261020-j03", "User-Agent": "claude-cli/2.1.280 (external)"}


def _app(tmp_path, cfg=None):
    app = build_app_for_tests(cfg or _cfg(), db_path=tmp_path / "log.sqlite")
    return app, TestClient(app)


def build_app_for_tests(cfg, db_path):
    from saint.server import build_app
    return build_app(cfg, db_path=db_path)


def _row(tmp_path, query):
    return sqlite3.connect(tmp_path / "log.sqlite").execute(query).fetchone()


@pytest.mark.parametrize("path,payload", [
    ("/v1/messages", {"model": "saint-cloud-anthropic", "max_tokens": 10,
                      "messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/chat/completions", {"model": "saint-cloud-anthropic",
                             "messages": [{"role": "user", "content": "hi"}]}),
])
def test_attribution_recorded_on_both_apis(tmp_path, path, payload):
    _, client = _app(tmp_path)
    with patch("saint.backends.litellm.anthropic.messages.acreate",
               AsyncMock(return_value=_ANTHROPIC_OK)), \
         patch("saint.backends.litellm.acompletion", AsyncMock(return_value=_CHAT_OK)):
        resp = client.post(path, json=payload, headers=_HEADERS)
    assert resp.status_code == 200
    assert _row(tmp_path, "SELECT client, origin, session, job_id FROM requests") == (
        "claude-code", "GEN-PW0DXJ6E", "c33f-488a", "sa-20261020-j03")


def test_client_labelled_even_without_attribution_headers(tmp_path):
    _, client = _app(tmp_path)
    with patch("saint.backends.litellm.acompletion", AsyncMock(return_value=_CHAT_OK)):
        client.post("/v1/chat/completions", json={
            "model": "saint-cloud-anthropic", "messages": [{"role": "user", "content": "hi"}]},
            headers={"User-Agent": "curl/8.5.0"})
    assert _row(tmp_path, "SELECT client, origin FROM requests") == ("curl", None)


@pytest.mark.parametrize("path,payload,attr", [
    ("/v1/messages", {"model": "saint-cloud-anthropic", "max_tokens": 10, "stream": True,
                      "messages": [{"role": "user", "content": "hi"}]}, True),
    ("/v1/chat/completions", {"model": "saint-cloud-anthropic", "stream": True,
                             "messages": [{"role": "user", "content": "hi"}]}, False),
])
def test_attribution_recorded_on_streaming_and_failed_requests(tmp_path, path, payload, attr):
    _, client = _app(tmp_path)
    boom = AsyncMock(side_effect=RuntimeError("down"))
    with patch("saint.backends.litellm.anthropic.messages.acreate", boom), \
         patch("saint.backends.litellm.acompletion", boom):
        with client.stream("POST", path, json=payload, headers=_HEADERS) as resp:
            list(resp.iter_bytes())
    # the failed request is still logged, still attributed
    assert _row(tmp_path, "SELECT client, origin, success FROM requests") == (
        "claude-code", "GEN-PW0DXJ6E", 0)


def test_attribution_headers_never_forwarded_upstream(tmp_path):
    """Contract §1: providers MUST NOT forward them to an upstream model."""
    _, client = _app(tmp_path)
    acompletion = AsyncMock(return_value=_CHAT_OK)
    acreate = AsyncMock(return_value=_ANTHROPIC_OK)
    with patch("saint.backends.litellm.acompletion", acompletion), \
         patch("saint.backends.litellm.anthropic.messages.acreate", acreate):
        client.post("/v1/chat/completions", json={
            "model": "saint-cloud-anthropic", "messages": [{"role": "user", "content": "hi"}]},
            headers=_HEADERS)
        client.post("/v1/messages", json={
            "model": "saint-cloud-anthropic", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}]}, headers=_HEADERS)
    for mock in (acompletion, acreate):
        kwargs = mock.call_args.kwargs
        flat = json.dumps(kwargs, default=str)
        for value in ("GEN-PW0DXJ6E", "c33f-488a", "sa-20261020-j03",
                      "X-Inference", "claude-cli"):
            assert value not in flat, (value, kwargs)
        assert "x-inference-origin" not in str(kwargs.get("extra_headers"))


def test_usage_route_reports_what_the_apis_recorded(tmp_path):
    _, client = _app(tmp_path)
    with patch("saint.backends.litellm.acompletion", AsyncMock(return_value=_CHAT_OK)):
        client.post("/v1/chat/completions", json={
            "model": "saint-cloud-anthropic", "messages": [{"role": "user", "content": "hi"}]},
            headers=_HEADERS)
    today = datetime.now(UTC).date().isoformat()
    resp = client.get(f"/usage/v1?from=2020-01-01&to={today}&granularity=total")
    assert resp.status_code == 200
    body = resp.json()
    assert body["contract"] == "hybrid-usage/1"
    assert body["provider"]["name"] == "saint"
    assert body["window"] == {"from": "2020-01-01", "to": today, "timezone": "UTC"}
    assert body["currency"] == "USD"
    assert body["totals"]["requests"] == 1
    assert body["by_client"][0]["client"] == "claude-code"
    assert body["by_origin"][0]["origin"] == "GEN-PW0DXJ6E"
    assert body["by_backend"][0]["backend"] == "cloud-anthropic"
    assert "prompt" not in json.dumps(body)          # never prompt text
    # and the errors keep the contract's shape over HTTP
    assert client.get("/usage/v1?grain=day").status_code == 400
    assert client.get("/usage/v1?grain=day").json()["code"] == "UNKNOWN_PARAM"


def test_usage_route_repeats_take_the_first_value(tmp_path):
    _, client = _app(tmp_path)
    resp = client.get("/usage/v1?client=claude-code&client=pi&to=2026-10-05&from=2026-10-01")
    assert resp.json()["filters"]["client"] == ["claude-code"]


def test_usage_route_auth(tmp_path, monkeypatch):
    _, client = _app(tmp_path)
    monkeypatch.delenv("SAINT_USAGE_TOKEN", raising=False)
    assert client.get("/usage/v1").status_code == 200        # unset = open

    monkeypatch.setenv("SAINT_USAGE_TOKEN", "sekret")
    assert client.get("/usage/v1").status_code == 401
    assert client.get("/usage/v1").json()["code"] == "UNAUTHORIZED"
    assert client.get("/usage/v1", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/usage/v1", headers={"Authorization": "sekret"}).status_code == 401
    ok = client.get("/usage/v1", headers={"Authorization": "Bearer sekret"})
    assert ok.status_code == 200

    monkeypatch.setenv("SAINT_USAGE_TOKEN", "")              # empty is not a token
    assert client.get("/usage/v1").status_code == 200


def test_usage_route_unavailable_on_db_error(tmp_path, monkeypatch):
    app, client = _app(tmp_path)

    class Boom:
        def execute(self, *a, **k):
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(app.state, "db", Boom())
    resp = client.get("/usage/v1")
    assert resp.status_code == 503 and resp.json()["code"] == "UNAVAILABLE"


def test_provider_version_is_the_package_version():
    assert U.provider_version() in ("0.1.0", "unknown")
    assert U.CONTRACT == "hybrid-usage/1"
