"""Capability-aware routing (spec §9).

A turn that carries non-text content (images today, audio/documents later) must only be
dispatched to a backend/seat that can serve it. Live seat capabilities come from johnny,
static ones from config; incapable candidates are EXCLUDED, not tried-and-failed.

Written against the spec alone — the implementation (saint/capabilities.py plus the
config/johnny/binding/dispatch/router/server hooks it describes) is being written in
parallel, so these imports fail until it lands.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from saint.capabilities import (
    BLOCK_CAPABILITIES,
    KNOWN_CAPABILITIES,
    missing,
    parse_capabilities,
    required_capabilities,
)

from saint import binding
from saint import config as config_mod
from saint import dispatch as dispatch_mod
from saint.config import (
    BackendConfig,
    ClassifierConfig,
    Config,
    JohnnyConfig,
    LoggingConfig,
    RoutingConfig,
    ServerConfig,
    load_config,
)
from saint.dispatch import dispatch_candidates, run_candidates
from saint.johnny import Resolution, _parse
from saint.route_cache import Breaker
from saint.server import build_app

# The spec names ConfigError; the loader today raises ValueError. Accept either (a
# ConfigError that subclasses ValueError is covered by the ValueError entry too).
_CONFIG_ERRORS: tuple[type[BaseException], ...] = tuple(dict.fromkeys(
    [ValueError, getattr(config_mod, "ConfigError", ValueError)]))

_IMG_OAI = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
_IMG_ANTH = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}
_DOC_ANTH = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                            "data": "AA"}}
_AUDIO_OAI = {"type": "input_audio", "input_audio": {"data": "AA", "format": "wav"}}
_FILE_OAI = {"type": "file", "file": {"file_data": "data:application/pdf;base64,AA",
                                      "filename": "spec.pdf"}}
_OK = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"}],
       "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


# ---------------------------------------------------------------- test scaffolding

def _b(name: str, *, provider: str = "openai", model: str | None = None,
       base_url: str | None = "http://static.invalid/v1",
       capabilities: tuple[str, ...] = (), **kw) -> BackendConfig:
    """BackendConfig with the required fields filled in; `capabilities` is the STATIC
    declaration (johnny's live answer is supplied separately, via the resolver)."""
    return BackendConfig(
        name=name, provider=provider, model=model or name,
        api_key_env=None, api_key=("x" if provider == "openai" else None),
        base_url=(base_url if provider == "openai" else None),
        aliases=(), timeout_s=10, capabilities=tuple(capabilities), **kw,
    )


def _cfg(backends: list[BackendConfig], *, default_on_failure: str | None = None,
         multimodal_backend: str | None = None, policy_target: str | None = None,
         classifier_backend: str | None = None, embeddings_backend: str | None = None,
         while_loading: str | None = None, johnny: JohnnyConfig | None = None) -> Config:
    """A real Config over `backends` (list order == config order, which the candidate
    extension depends on). retry_same_backend is off so attempt counts stay readable."""
    names = [b.name for b in backends]
    target = policy_target or names[0]
    policy = {u: {f"{d},{c}": target for d in ("code", "general") for c in ("trivial", "medium", "hard")}
              for u in ("normal", "urgent", "patient")}
    return Config(
        server=ServerConfig(host="127.0.0.1", port=4000),
        backends={b.name: b for b in backends},
        classifier=ClassifierConfig(backend=classifier_backend or target, fallback_backend=None,
                                    max_input_chars=8000, timeout_s=5, prompt_template_path=None),
        routing=RoutingConfig(default_urgency="normal",
                              default_on_failure=default_on_failure or names[-1],
                              policy=policy, multimodal_backend=multimodal_backend,
                              embeddings_backend=embeddings_backend, while_loading=while_loading,
                              retry_same_backend=False, breaker_failures=3, breaker_cooldown_s=60.0),
        logging=LoggingConfig(db_path=":memory:", prompt_storage="none"),
        johnny=johnny,
    )


def _johnny() -> JohnnyConfig:
    """A [johnny] block that must never actually be shelled out — every test uses a fake
    resolver, and resolve_for_dispatch only consults it when cfg.johnny is set."""
    return JohnnyConfig(transport="cli", cli_path="johnny-must-not-be-called", base_url=None,
                        resolve_cache_ttl_s=1.0, ensure_load=False)


class FakeResolver:
    """johnny stand-in: canned Resolutions per role/seat target, no CLI, no network."""

    def __init__(self, **by_target: Resolution | None):
        self.seats = dict(by_target)
        self.resolve_calls = 0
        self.loads_requested: list[str] = []

    def resolve(self, target: str) -> Resolution | None:
        self.resolve_calls += 1
        return self.seats.get(target)

    def ensure_load(self, target: str) -> None:
        self.loads_requested.append(target)


def _ready(seat: str, port: int, capabilities: frozenset[str] | None = None,
           model: str = "m") -> Resolution:
    """A ready johnny seat; `capabilities` None = johnny told us nothing (unknown),
    frozenset() = it declared text-only. Those are different things."""
    return Resolution(seat=seat, endpoint=f"http://127.0.0.1:{port}/v1", model=model,
                      state="ready", eta_s=None, queue_depth=None, capabilities=capabilities)


@pytest.fixture
def loads(monkeypatch):
    """Seat-load table (endpoint -> in-flight count); absent endpoint = unknown load."""
    table: dict[str, int] = {}
    monkeypatch.setattr(binding, "seat_load", lambda endpoint, timeout=0.5: table.get(endpoint))
    return table


def _app(tmp_path, cfg):
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    return app, TestClient(app)


def _row(tmp_path, query):
    import sqlite3
    return sqlite3.connect(tmp_path / "log.sqlite").execute(query).fetchone()


# ---------------------------------------------------------------- §1 required_capabilities

def test_required_str_content_needs_nothing():
    msgs = [{"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello there"}]
    assert required_capabilities(msgs) == frozenset()


def test_required_text_only_blocks_needs_nothing():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "a"},
                                         {"type": "text", "text": "b"}]}]
    assert required_capabilities(msgs) == frozenset()


def test_required_openai_image_part_needs_vision():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "look"}, _IMG_OAI]}]
    assert required_capabilities(msgs) == frozenset({"vision"})


def test_required_input_audio_part_needs_audio():
    msgs = [{"role": "user", "content": ["", _AUDIO_OAI]}]
    assert required_capabilities(msgs) == frozenset({"audio"})


def test_required_file_part_needs_document():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "summarize"}, _FILE_OAI]}]
    assert required_capabilities(msgs) == frozenset({"document"})


def test_required_anthropic_image_block_needs_vision():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "what is this?"}, _IMG_ANTH]}]
    assert required_capabilities(msgs) == frozenset({"vision"})


def test_required_anthropic_document_block_needs_document():
    msgs = [{"role": "user", "content": [_DOC_ANTH]}]
    assert required_capabilities(msgs) == frozenset({"document"})


def test_required_sees_image_nested_in_tool_result():
    """Agent turns hide media inside tool_result content — the whole message tree counts."""
    msgs = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1",
         "content": [{"type": "text", "text": "screenshot captured"}, _IMG_ANTH]}]}]
    assert required_capabilities(msgs) == frozenset({"vision"})


def test_required_media_in_an_earlier_turn_still_counts():
    """The full history is replayed to the backend, so an image three turns back still
    needs a vision seat — this is the case the last-user-message multimodal check misses."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": "look"}, _IMG_OAI]},
            {"role": "assistant", "content": "a cat"},
            {"role": "user", "content": "what breed?"}]
    assert required_capabilities(msgs) == frozenset({"vision"})


def test_required_media_on_an_assistant_turn_counts():
    msgs = [{"role": "user", "content": "generate one"},
            {"role": "assistant", "content": [{"type": "text", "text": "sure"}, _IMG_OAI]},
            {"role": "user", "content": "again"}]
    assert required_capabilities(msgs) == frozenset({"vision"})


def test_required_is_the_union_over_the_whole_conversation():
    msgs = [{"role": "user", "content": [_IMG_OAI]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [_AUDIO_OAI, _DOC_ANTH]}]
    assert required_capabilities(msgs) == frozenset({"vision", "audio", "document"})


def test_required_never_raises_on_odd_shapes():
    for weird in (
        [],
        [{"role": "user"}],                                        # no content key
        [{"role": "user", "content": None}],
        [{"role": "user", "content": ["bare string part", 42, {"type": "text"}]}],  # junk blocks
        [{"role": "user", "content": {"type": "text", "text": "dict content"}}],    # not a list
        [{"role": "user", "content": [{"type": "tool_result"}]}],  # tool_result, no content
        [{"role": "user", "content": [{"type": "tool_result", "content": "not a list"}]}],
        [{"role": "user", "content": [{"type": "tool_result", "content": ["x", None]}]}],
    ):
        assert required_capabilities(weird) == frozenset()


def test_required_returns_a_frozenset_and_handles_repeats():
    got = required_capabilities([{"role": "user", "content": [_IMG_OAI, _IMG_OAI]}])
    assert isinstance(got, frozenset) and got == frozenset({"vision"})


def test_capability_registry_is_plain_strings():
    """Capabilities are registry names, not `if image` branches: the block map's values are
    the whole vocabulary, and text is implicit (never listed)."""
    assert KNOWN_CAPABILITIES == frozenset(BLOCK_CAPABILITIES.values())
    assert {"vision", "audio", "document"} <= KNOWN_CAPABILITIES
    assert "text" not in KNOWN_CAPABILITIES
    assert set(BLOCK_CAPABILITIES) >= {"image_url", "input_audio", "file", "image", "document"}


# ---------------------------------------------------------------- §1 parse_capabilities

def test_parse_dict_modalities_form():
    assert parse_capabilities({"modalities": ["vision"]}) == frozenset({"vision"})


def test_parse_bare_list_tuple_or_set():
    assert parse_capabilities(["vision", "document"]) == frozenset({"vision", "document"})
    assert parse_capabilities(("vision",)) == frozenset({"vision"})
    assert parse_capabilities({"vision"}) == frozenset({"vision"})


def test_parse_drops_implicit_text():
    assert parse_capabilities({"modalities": ["text", "vision"]}) == frozenset({"vision"})


def test_parse_text_only_is_known_empty_not_unknown():
    """Empty set = "this seat is text-only" (a real answer); None = "johnny didn't say".
    The two must not collapse — binding falls back to static config only for None."""
    assert parse_capabilities(["text"]) == frozenset()


def test_parse_none_is_unknown():
    assert parse_capabilities(None) is None


def test_parse_garbage_is_unknown():
    for garbage in ({}, {"modalities": "vision"}, {"modalities": None}, {"modalities": {}},
                    {"capabilities": ["vision"]}, 42, object()):
        assert parse_capabilities(garbage) is None


# ---------------------------------------------------------------- §1 missing()

def test_missing_subtracts_available_from_required():
    assert missing(frozenset({"vision", "audio"}), frozenset({"vision"})) == frozenset({"audio"})


def test_missing_unknown_available_counts_as_text_only():
    assert missing(frozenset({"vision"}), None) == frozenset({"vision"})
    assert missing(frozenset({"vision", "document"}), None) == frozenset({"vision", "document"})


def test_missing_nothing_required_is_always_satisfied():
    assert missing(frozenset(), None) == frozenset()
    assert missing(frozenset(), frozenset({"vision"})) == frozenset()


# ---------------------------------------------------------------- §2 config parsing

def _toml(tmp_path: Path, backends_toml: str) -> Path:
    cells = "\n".join(f'"{d},{c}" = "cloud"' for d in ("code", "general")
                      for c in ("trivial", "medium", "hard"))
    p = tmp_path / "config.toml"
    p.write_text(f"""
[server]
host = "127.0.0.1"
port = 4000

[backends.cloud]
provider = "openai"
model = "cloud-m"
base_url = "http://127.0.0.1:8099/v1"
{backends_toml}
[classifier]
backend = "cloud"

[routing]
default_urgency = "normal"
default_on_failure = "cloud"

[routing.policy.normal]
{cells}
[routing.policy.urgent]
{cells}
[routing.policy.patient]
{cells}

[logging]
db_path = "{tmp_path / 'log.sqlite'}"
prompt_storage = "none"
""")
    return p


def test_config_explicit_capabilities_parsed(tmp_path):
    cfg = load_config(_toml(tmp_path, """
[backends.visionseat]
provider = "openai"
model = "v"
base_url = "http://127.0.0.1:8010/v1"
capabilities = ["vision"]
"""))
    b = cfg.backends["visionseat"]
    assert b.capabilities == ("vision",)
    assert b.static_capabilities == frozenset({"vision"})


def test_config_without_capabilities_key_is_text_only(tmp_path):
    cfg = load_config(_toml(tmp_path, """
[backends.plain]
provider = "openai"
model = "p"
base_url = "http://127.0.0.1:8011/v1"
"""))
    assert cfg.backends["plain"].capabilities == ()
    assert cfg.backends["plain"].static_capabilities == frozenset()


def test_config_anthropic_defaults_to_vision_document(tmp_path):
    cfg = load_config(_toml(tmp_path, """
[backends.cloud-haiku]
provider = "anthropic"
model = "claude-x"
api_key_env = "ANTHROPIC_API_KEY"
"""))
    assert cfg.backends["cloud-haiku"].static_capabilities == frozenset({"vision", "document"})


def test_config_bedrock_defaults_to_vision_document(tmp_path):
    cfg = load_config(_toml(tmp_path, """
[backends.bray]
provider = "bedrock"
model = "claude-x"
aws_region = "us-east-1"
"""))
    assert cfg.backends["bray"].static_capabilities == frozenset({"vision", "document"})


def test_config_explicit_empty_list_overrides_the_provider_default(tmp_path):
    """Present-but-empty is a declaration ("text only"), distinct from absent."""
    cfg = load_config(_toml(tmp_path, """
[backends.cloud-haiku]
provider = "anthropic"
model = "claude-x"
api_key_env = "ANTHROPIC_API_KEY"
capabilities = []
"""))
    assert cfg.backends["cloud-haiku"].static_capabilities == frozenset()


def test_config_capabilities_non_list_is_a_config_error(tmp_path):
    p = _toml(tmp_path, """
[backends.plain]
provider = "openai"
model = "p"
base_url = "http://127.0.0.1:8011/v1"
capabilities = "vision"
""")
    with pytest.raises(_CONFIG_ERRORS, match="capabilities"):
        load_config(p)


def test_config_unknown_capability_is_allowed_and_warned_once(tmp_path, capsys):
    """The registry is extensible: an unknown name is never rejected (a newer johnny must
    not break an older router), it just gets one heads-up on stderr per name."""
    cfg = load_config(_toml(tmp_path, """
[backends.seat-a]
provider = "openai"
model = "a"
base_url = "http://127.0.0.1:8012/v1"
capabilities = ["telepathy"]

[backends.seat-b]
provider = "openai"
model = "b"
base_url = "http://127.0.0.1:8013/v1"
capabilities = ["telepathy", "vision"]
"""))
    assert cfg.backends["seat-a"].static_capabilities == frozenset({"telepathy"})
    assert cfg.backends["seat-b"].static_capabilities == frozenset({"telepathy", "vision"})
    # 0 or 1 lines naming it: deduped per name, not per occurrence/backend (a module-level
    # registry may already have warned in an earlier test, hence no lower bound).
    assert len([ln for ln in capsys.readouterr().err.splitlines() if "telepathy" in ln]) <= 1


def test_config_auto_cloud_ladder_declares_vision_document(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = load_config(_toml(tmp_path, ""))
    for name in ("cloud-small", "cloud-medium", "cloud-large", "cloud-exquisite"):
        if name in cfg.backends:                      # auto-configured, not explicit
            assert cfg.backends[name].static_capabilities == frozenset({"vision", "document"})


# ---------------------------------------------------------------- §3 Resolution carries caps

def test_resolution_parse_reads_capabilities():
    r = _parse({"seat": "s1", "endpoint": "http://127.0.0.1:8002/v1", "model": "m",
                "state": "ready", "eta_s": None, "queue_depth": None,
                "capabilities": {"modalities": ["vision"]}})
    assert r.capabilities == frozenset({"vision"})


def test_resolution_parse_absent_capabilities_is_unknown():
    r = _parse({"seat": "s1", "endpoint": "http://127.0.0.1:8002/v1", "model": "m",
                "state": "ready", "eta_s": None, "queue_depth": None})
    assert r.capabilities is None


def test_resolution_default_capabilities_is_unknown():
    r = Resolution(seat="s", endpoint=None, model=None, state="absent", eta_s=None, queue_depth=None)
    assert r.capabilities is None


# ---------------------------------------------------------------- §4 Effective.capabilities

def test_effective_caps_ready_seat_beats_static():
    cfg = _cfg([_b("local-coder", johnny_role="coder", capabilities=("document",))], johnny=_johnny())
    r = FakeResolver(coder=_ready("c", 8002, capabilities=frozenset({"vision"})))
    eff = binding.resolve_for_dispatch(cfg, "local-coder", r)
    assert eff.state_at_dispatch == "johnny_ready"
    assert eff.capabilities == frozenset({"vision"})       # live answer wins


def test_effective_caps_ready_seat_without_caps_falls_back_to_static():
    cfg = _cfg([_b("local-coder", johnny_role="coder", capabilities=("vision",))], johnny=_johnny())
    r = FakeResolver(coder=_ready("c", 8002, capabilities=None))
    eff = binding.resolve_for_dispatch(cfg, "local-coder", r)
    assert eff.capabilities == frozenset({"vision"})


def test_effective_caps_ready_seat_declaring_text_only_stays_text_only():
    cfg = _cfg([_b("local-coder", johnny_role="coder", capabilities=("vision",))], johnny=_johnny())
    r = FakeResolver(coder=_ready("c", 8002, capabilities=frozenset()))
    eff = binding.resolve_for_dispatch(cfg, "local-coder", r)
    assert eff.capabilities == frozenset()                 # known-and-empty, not "fall back"


def test_effective_caps_static_baseline_uses_static():
    cfg = _cfg([_b("local-coder", johnny_role="coder", capabilities=("vision",))], johnny=_johnny())
    eff = binding.resolve_for_dispatch(cfg, "local-coder", FakeResolver(coder=None))  # johnny down
    assert eff.state_at_dispatch == "static_baseline"
    assert eff.capabilities == frozenset({"vision"})


def test_effective_caps_while_loading_uses_the_serving_backend_static():
    cfg = _cfg([_b("local-coder", johnny_role="coder", capabilities=("vision",)),
                _b("cloud", capabilities=("vision", "document"))],
               while_loading="cloud", johnny=_johnny())
    r = FakeResolver(coder=Resolution(seat="c", endpoint=None, model=None, state="loading",
                                      eta_s=30, queue_depth=None))
    eff = binding.resolve_for_dispatch(cfg, "local-coder", r)
    assert eff.backend.name == "cloud" and eff.state_at_dispatch == "while_loading"
    assert eff.capabilities == frozenset({"vision", "document"})   # the one that serves


def test_effective_caps_fallback_uses_the_serving_backend_static():
    cfg = _cfg([_b("managed-vision", provider=None, model=None, base_url=None,
                   johnny_seat="vision", johnny_only=True),
                _b("cloud", capabilities=("vision", "document"))], johnny=_johnny())
    r = FakeResolver(vision=Resolution(seat=None, endpoint=None, model=None, state="absent",
                                       eta_s=None, queue_depth=None))
    eff = binding.resolve_for_dispatch(cfg, "managed-vision", r)
    assert eff.state_at_dispatch == "fallback"
    assert eff.capabilities == frozenset({"vision", "document"})


def test_effective_caps_unbound_backend_is_static():
    """Cloud backends have no johnny binding, and run_candidates must still be able to
    exclude them for a vision turn — so the unbound path carries static caps too."""
    cfg = _cfg([_b("cloud", provider="anthropic", base_url=None,
                   capabilities=("vision", "document"))])
    eff = binding.resolve_for_dispatch(cfg, "cloud", None)
    assert eff.state_at_dispatch is None
    assert eff.capabilities == frozenset({"vision", "document"})


def test_describe_for_explain_reports_capabilities():
    cfg = _cfg([_b("local-coder", johnny_role="coder")], johnny=_johnny())
    r = FakeResolver(coder=_ready("c", 8002, capabilities=frozenset({"vision"})))
    d = binding.describe_for_explain(cfg, "local-coder", r)
    assert d is not None and d["capabilities"] == ["vision"]


# ---------------------------------------------------------------- §4 capability spill

def test_capability_spill_prefers_a_capable_sibling_seat(loads, capsys):
    cfg = _cfg([_b("local-worker", johnny_role="worker", spill=("coder",)),
                _b("local-coder", johnny_role="coder")], johnny=_johnny())
    r = FakeResolver(worker=_ready("w", 8006, capabilities=frozenset()),
                     coder=_ready("c", 8002, capabilities=frozenset({"vision"})))
    loads["http://127.0.0.1:8002/v1"] = 0
    eff = binding.resolve_for_dispatch(cfg, "local-worker", r, required=frozenset({"vision"}))
    assert eff.johnny_seat == "c" and eff.spilled_from == "worker"
    assert eff.capabilities == frozenset({"vision"})
    assert eff.backend.base_url == "http://127.0.0.1:8002/v1"
    err = capsys.readouterr().err
    assert "capability spill" in err and "vision" in err


def test_no_capability_spill_when_own_seat_can_already_serve(loads):
    cfg = _cfg([_b("local-worker", johnny_role="worker", spill=("coder",)),
                _b("local-coder", johnny_role="coder")], johnny=_johnny())
    r = FakeResolver(worker=_ready("w", 8006, capabilities=frozenset({"vision"})),
                     coder=_ready("c", 8002, capabilities=frozenset({"vision"})))
    eff = binding.resolve_for_dispatch(cfg, "local-worker", r, required=frozenset({"vision"}))
    assert eff.johnny_seat == "w" and eff.spilled_from is None


def test_capability_spill_skips_siblings_that_also_lack_the_capability(loads):
    cfg = _cfg([_b("local-worker", johnny_role="worker", spill=("coder",)),
                _b("local-coder", johnny_role="coder")], johnny=_johnny())
    r = FakeResolver(worker=_ready("w", 8006, capabilities=frozenset()),
                     coder=_ready("c", 8002, capabilities=frozenset()))
    eff = binding.resolve_for_dispatch(cfg, "local-worker", r, required=frozenset({"vision"}))
    # nowhere capable to go → stay on the own seat; run_candidates does the exclusion
    assert eff.johnny_seat == "w" and eff.spilled_from is None


def test_load_spill_never_lands_on_a_seat_that_cannot_serve_the_turn(loads):
    """The own seat is saturated, so a spill is warranted — but the only ready sibling is
    text-only, so borrowing it would fail the request. Stay on the busy seat."""
    cfg = _cfg([_b("local-worker", johnny_role="worker", spill=("coder",), spill_at=6),
                _b("local-coder", johnny_role="coder")], johnny=_johnny())
    r = FakeResolver(worker=_ready("w", 8006, capabilities=frozenset({"vision"})),
                     coder=_ready("c", 8002, capabilities=frozenset()))
    loads["http://127.0.0.1:8006/v1"] = 9
    loads["http://127.0.0.1:8002/v1"] = 0        # idle: it WOULD be picked if not filtered out
    eff = binding.resolve_for_dispatch(cfg, "local-worker", r, required=frozenset({"vision"}))
    assert eff.johnny_seat == "w" and eff.spilled_from is None


def test_spill_without_required_is_unfiltered_as_before(loads):
    """Same saturated setup, no `required` kwarg → the pre-capabilities behaviour (spill to
    the least-loaded ready seat). Guards the default against over-filtering."""
    cfg = _cfg([_b("local-worker", johnny_role="worker", spill=("coder",), spill_at=6),
                _b("local-coder", johnny_role="coder")], johnny=_johnny())
    r = FakeResolver(worker=_ready("w", 8006, capabilities=frozenset({"vision"})),
                     coder=_ready("c", 8002, capabilities=frozenset()))
    loads["http://127.0.0.1:8006/v1"] = 9
    loads["http://127.0.0.1:8002/v1"] = 0
    eff = binding.resolve_for_dispatch(cfg, "local-worker", r)
    assert eff.johnny_seat == "c" and eff.spilled_from == "worker"


# ---------------------------------------------------------------- §5 dispatch_candidates

def test_dispatch_candidates_without_required_is_primary_plus_on_error():
    cfg = _cfg([_b("primary", on_error="fallback"), _b("fallback"),
                _b("static-v", capabilities=("vision",))])
    br = Breaker(threshold=3, cooldown_s=60.0)
    assert dispatch_candidates(cfg, "primary", br) == ["primary", "fallback"]
    assert dispatch_candidates(cfg, "primary", br, required=frozenset()) == ["primary", "fallback"]


def test_dispatch_candidates_appends_johnny_bound_then_static():
    cfg = _cfg([_b("primary", on_error="fallback"), _b("fallback"),
                _b("seat-x", johnny_role="x"), _b("static-v", capabilities=("vision",)),
                _b("plain")])
    br = Breaker(threshold=3, cooldown_s=60.0)
    got = dispatch_candidates(cfg, "primary", br, required=frozenset({"vision"}))
    assert got == ["primary", "fallback", "seat-x", "static-v"]
    assert "plain" not in got                    # cannot serve it → never suggested
    assert got[0] == "primary"                   # the incapable primary stays first


def test_dispatch_candidates_appends_a_candidate_once():
    cfg = _cfg([_b("primary", on_error="static-v"), _b("static-v", capabilities=("vision",))])
    br = Breaker(threshold=3, cooldown_s=60.0)
    got = dispatch_candidates(cfg, "primary", br, required=frozenset({"vision"}))
    assert got == ["primary", "static-v"]


def test_dispatch_candidates_does_not_exclude_classifier_or_embeddings_by_name():
    cfg = _cfg([_b("primary"), _b("static-v", capabilities=("vision",))],
               classifier_backend="static-v", embeddings_backend="static-v")
    br = Breaker(threshold=3, cooldown_s=60.0)
    got = dispatch_candidates(cfg, "primary", br, required=frozenset({"vision"}))
    assert got == ["primary", "static-v"]


def test_dispatch_candidates_pinned_turn_keeps_the_short_list():
    """A pin is a pin: never silently upgrade a pinned turn to a different backend."""
    cfg = _cfg([_b("primary", on_error="fallback"), _b("fallback"),
                _b("seat-x", johnny_role="x"), _b("static-v", capabilities=("vision",))])
    br = Breaker(threshold=3, cooldown_s=60.0)
    got = dispatch_candidates(cfg, "primary", br, required=frozenset({"vision"}), pinned=True)
    assert got == ["primary", "fallback"]


# ---------------------------------------------------------------- §5 run_candidates

def _recorder(log: list, value="ok", fail_for: str | None = None):
    async def _attempt(eff):
        log.append(eff.backend.name)
        if fail_for == eff.backend.name:
            raise RuntimeError("boom")
        return value
    return _attempt


async def test_run_candidates_skips_incapable_candidate_and_serves_capable_one(capsys):
    cfg = _cfg([_b("textseat"), _b("visionseat", capabilities=("vision",))])
    tried: list[str] = []
    br = Breaker(threshold=1, cooldown_s=60.0)     # one failure would open the circuit
    served, error_kind, eff = await run_candidates(
        cfg=cfg, candidates=["textseat", "visionseat"], rid="tst", resolver=None, breaker=br,
        attempt=_recorder(tried), required=frozenset({"vision"}))
    assert tried == ["visionseat"]                 # the incapable one was never attempted
    assert served is not None and served.backend == "visionseat" and error_kind is None
    assert eff.backend.name == "visionseat"
    err = capsys.readouterr().err
    assert "skipping 'textseat'" in err and "lacks vision" in err
    assert not br.is_open("textseat")              # skipping is not a backend failure


async def test_run_candidates_capability_error_when_every_candidate_is_skipped(capsys):
    cfg = _cfg([_b("a"), _b("b")])
    tried: list[str] = []
    served, error_kind, _ = await run_candidates(
        cfg=cfg, candidates=["a", "b"], rid="tst", resolver=None,
        breaker=Breaker(threshold=3, cooldown_s=60.0), attempt=_recorder(tried),
        required=frozenset({"vision"}))
    assert served is None and tried == []
    assert error_kind is not None and error_kind.startswith("CapabilityError: ")
    assert "no backend can serve vision" in error_kind
    assert "a/a lacks vision" in error_kind and "b/b lacks vision" in error_kind


async def test_run_candidates_capability_error_names_every_gap(capsys):
    cfg = _cfg([_b("a"), _b("b", capabilities=("vision",))])
    served, error_kind, _ = await run_candidates(
        cfg=cfg, candidates=["a", "b"], rid="tst", resolver=None,
        breaker=Breaker(threshold=3, cooldown_s=60.0), attempt=_recorder([]),
        required=frozenset({"vision", "document"}))
    assert served is None
    assert "no backend can serve document,vision" in error_kind
    assert "a/a lacks document,vision" in error_kind
    assert "b/b lacks document" in error_kind


async def test_run_candidates_attempted_failure_keeps_the_normal_error_kind():
    """An incapable candidate was skipped but another was tried and failed: that is a
    backend error (502), not a capability error (400)."""
    cfg = _cfg([_b("textseat"), _b("visionseat", capabilities=("vision",))])
    tried: list[str] = []
    served, error_kind, _ = await run_candidates(
        cfg=cfg, candidates=["textseat", "visionseat"], rid="tst", resolver=None,
        breaker=Breaker(threshold=3, cooldown_s=60.0),
        attempt=_recorder(tried, fail_for="visionseat"), required=frozenset({"vision"}))
    assert tried == ["visionseat"]
    assert served is None and error_kind == "RuntimeError"


async def test_run_candidates_without_required_attempts_everything(capsys):
    cfg = _cfg([_b("a"), _b("b")])
    tried: list[str] = []
    served, error_kind, _ = await run_candidates(
        cfg=cfg, candidates=["a", "b"], rid="tst", resolver=None,
        breaker=Breaker(threshold=3, cooldown_s=60.0), attempt=_recorder(tried))
    assert tried == ["a"] and served.backend == "a" and error_kind is None
    assert "skipping" not in capsys.readouterr().err


async def test_run_candidates_resolves_each_candidate_with_required(monkeypatch):
    """The resolve call must see `required`, else a johnny seat's live capabilities (and the
    capability spill it enables) are decided blind."""
    seen: list = []
    real = binding.resolve_for_dispatch

    def spy(cfg_, name, resolver, required=frozenset(), size=None):
        seen.append(required)
        return real(cfg_, name, resolver, required, size=size)

    monkeypatch.setattr(dispatch_mod, "resolve_for_dispatch", spy)
    cfg = _cfg([_b("seat", johnny_role="coder")], johnny=_johnny())
    r = FakeResolver(coder=_ready("c", 8002, capabilities=frozenset({"vision"})))
    served, error_kind, _ = await run_candidates(
        cfg=cfg, candidates=["seat"], rid="tst", resolver=r,
        breaker=Breaker(threshold=3, cooldown_s=60.0),
        attempt=_recorder([]), required=frozenset({"vision"}))
    assert seen == [frozenset({"vision"})]
    assert served is not None and served.backend == "seat" and error_kind is None


# ---------------------------------------------------------------- §6 decide_route carries it

async def test_decide_route_sets_required_capabilities_on_the_dispatch_path():
    from saint.router import decide_route
    cfg = _cfg([_b("local-coder"), _b("cloud", provider="anthropic", base_url=None)])
    d = await decide_route(cfg=cfg, model_field="saint-auto",
                           messages=[{"role": "user", "content": [{"type": "text", "text": "hi"},
                                                                  _IMG_OAI]}])
    assert d.required_capabilities == frozenset({"vision"})


async def test_decide_route_sets_required_capabilities_on_the_pinned_path():
    from saint.router import decide_route
    cfg = _cfg([_b("local-coder"), _b("cloud", provider="anthropic", base_url=None)])
    d = await decide_route(cfg=cfg, model_field="saint-local-coder",
                           messages=[{"role": "user", "content": [_DOC_ANTH]}])
    assert d.pinned_backend == "local-coder"
    assert d.required_capabilities == frozenset({"document"})


async def test_decide_route_sets_required_capabilities_for_text_turns_and_explain():
    from saint.router import decide_route
    cfg = _cfg([_b("local-coder"), _b("cloud", provider="anthropic", base_url=None)])
    with patch("saint.classifier.call_backend", AsyncMock(side_effect=RuntimeError("not needed"))):
        d = await decide_route(cfg=cfg, model_field="saint-explain",
                               messages=[{"role": "user", "content": "hello"}])
    assert d.mode == "explain" and d.required_capabilities == frozenset()


async def test_decide_route_required_override_wins_over_message_scan():
    from saint.router import decide_route
    cfg = _cfg([_b("local-coder"), _b("cloud", provider="anthropic", base_url=None)])
    with patch("saint.classifier.call_backend", AsyncMock(side_effect=RuntimeError("not needed"))):
        d = await decide_route(cfg=cfg, model_field="saint-auto",
                               messages=[{"role": "user", "content": "and now?"}],
                               required_override=frozenset({"vision"}))
    assert d.required_capabilities == frozenset({"vision"})


async def test_routing_decision_default_is_empty():
    from saint.prefixes import ParsedPrefixes
    from saint.router import RoutingDecision
    d = RoutingDecision(request_id="r", mode="dispatch", backend="x", pinned_backend=None,
                        urgency="normal", parsed=ParsedPrefixes(urgency=None, pinned_backend=None,
                                                               stripped="", raw=""),
                        classifier_outcome=None, classifier_result=None, multimodal=False,
                        model_field="saint-auto", last_user_content_original=None,
                        stripped_last_user=None)
    assert d.required_capabilities == frozenset()


# ---------------------------------------------------------------- §7 /v1/chat/completions

def test_chat_completions_pinned_to_inc_capable_seat_is_400(tmp_path, capsys):
    """A pin is a pin, so the turn cannot be moved to a vision seat — it is refused with a
    400 that says what was missing, instead of a 502 or a garbled answer."""
    cfg = _cfg([_b("seat-txt", johnny_role="coder"), _b("cloud", provider="anthropic",
                                                          base_url=None)], johnny=_johnny())
    app, client = _app(tmp_path, cfg)
    app.state.resolver = FakeResolver(coder=_ready("s", 8002, capabilities=frozenset()))
    mock = AsyncMock(return_value=_OK)
    with patch("saint.backends.litellm.acompletion", mock):
        resp = client.post("/v1/chat/completions", json={
            "model": "saint-seat-txt",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                                      _IMG_OAI]}]})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "capability_unavailable"
    assert err["message"].startswith("no backend can serve vision")
    assert "seat-txt" in err["message"] and "lacks vision" in err["message"]
    assert "CapabilityError" not in err["message"]     # the prefix is stripped for clients
    assert mock.await_count == 0                       # nothing was attempted
    row = _row(tmp_path, "SELECT success, error_kind FROM requests ORDER BY id DESC LIMIT 1")
    assert row[0] == 0 and row[1].startswith("CapabilityError")


def test_chat_completions_unpinned_image_finds_the_capable_backend(tmp_path):
    cfg = _cfg([_b("textseat"), _b("visionseat", capabilities=("vision",)),
                _b("cloud", provider="anthropic", base_url=None)],
               default_on_failure="cloud", multimodal_backend="textseat")
    _, client = _app(tmp_path, cfg)
    mock = AsyncMock(return_value=_OK)
    with patch("saint.backends.litellm.acompletion", mock):
        resp = client.post("/v1/chat/completions", json={
            "model": "saint-auto",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                                      _IMG_OAI]}]})
    assert resp.status_code == 200
    assert resp.headers["x-saint-backend"] == "visionseat"
    assert mock.await_count == 1                       # the text-only seat was never tried
    assert _row(tmp_path, "SELECT success FROM requests ORDER BY id DESC LIMIT 1")[0] == 1


def test_chat_completions_image_in_an_earlier_turn_still_needs_vision(tmp_path):
    """Detection is over the whole conversation: the media may be far from the last message."""
    cfg = _cfg([_b("textseat"), _b("visionseat", capabilities=("vision",)),
                _b("cloud", provider="anthropic", base_url=None)],
               default_on_failure="cloud", policy_target="textseat")
    _, client = _app(tmp_path, cfg)
    mock = AsyncMock(return_value=_OK)
    with patch("saint.backends.litellm.acompletion", mock):
        resp = client.post("/v1/chat/completions", json={
            "model": "saint-textseat",
            "messages": [{"role": "user", "content": [_IMG_OAI]},
                         {"role": "assistant", "content": "a cat"},
                         {"role": "user", "content": "what breed?"}]})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "capability_unavailable"
    assert mock.await_count == 0


def test_chat_completions_text_turn_unaffected(tmp_path):
    cfg = _cfg([_b("textseat"), _b("visionseat", capabilities=("vision",)),
                _b("cloud", provider="anthropic", base_url=None)],
               default_on_failure="cloud", policy_target="textseat")
    _, client = _app(tmp_path, cfg)
    mock = AsyncMock(return_value=_OK)
    with patch("saint.backends.litellm.acompletion", mock):
        resp = client.post("/v1/chat/completions", json={
            "model": "saint-textseat",
            "messages": [{"role": "user", "content": "hello"}]})
    assert resp.status_code == 200 and resp.headers["x-saint-backend"] == "textseat"


# ---------------------------------------------------------------- §7 /v1/messages

_MSG_OK = {"id": "msg_01", "type": "message", "role": "assistant", "model": "m",
           "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
           "usage": {"input_tokens": 10, "output_tokens": 2}}
_CLS_RESP = {"choices": [{"message": {"content": json.dumps(
    {"domain": "code", "complexity": "medium", "reason": "."})}}]}


def test_messages_image_block_skips_the_inc_capable_primary(tmp_path):
    cfg = _cfg([_b("textseat"), _b("visionseat", capabilities=("vision",)),
                _b("cloud", provider="anthropic", base_url=None)],
               default_on_failure="cloud", policy_target="textseat")
    _, client = _app(tmp_path, cfg)
    acreate = AsyncMock(return_value=_MSG_OK)
    with patch("saint.backends.litellm.anthropic.messages.acreate", acreate):
        resp = client.post("/v1/messages", json={
            "model": "saint-textseat", "max_tokens": 10,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                                      _IMG_ANTH]}]})
    assert resp.status_code == 400                       # pinned → no capable fallback
    assert resp.json()["error"]["type"] == "invalid_request_error"
    assert "no backend can serve vision" in resp.json()["error"]["message"]
    assert acreate.await_count == 0


def test_messages_tool_result_media_needs_the_override(tmp_path):
    """openai_view flattens tool_result content to text, so detection must run on the RAW
    body — otherwise a screenshot returned by a tool routes to a text-only seat."""
    cfg = _cfg([_b("textseat"), _b("visionseat", capabilities=("vision",)),
                _b("cloud", provider="anthropic", base_url=None)],
               default_on_failure="cloud", policy_target="textseat")
    _, client = _app(tmp_path, cfg)
    acreate = AsyncMock(return_value=_MSG_OK)
    clf = AsyncMock(return_value=_CLS_RESP)
    with patch("saint.backends.litellm.anthropic.messages.acreate", acreate), \
         patch("saint.classifier.call_backend", clf):
        resp = client.post("/v1/messages", json={
            "model": "auto", "max_tokens": 10, "system": "agent",
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1",
                     "content": [{"type": "text", "text": "captured"}, _IMG_ANTH]}]},
                {"role": "assistant", "content": [{"type": "text", "text": "saw it"}]},
                {"role": "user", "content": "and now?"}]})
    assert resp.status_code == 200
    assert resp.headers["x-saint-backend"] == "visionseat"
    assert acreate.await_count == 1                      # textseat skipped, not failed
