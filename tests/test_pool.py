"""Pool backends (strict load balancing) + client rules (spec §7).

Rick wants one behaviour per client: pi keeps routing as today (classify, cloud for hard
turns), while Claude Code must ALWAYS round-robin across the three local Flash-Next seats
and never leave the box. Two mechanisms:

- a **pool backend** — `pool = [members]` + `local_only = true`: it owns no endpoint, it
  resolves to an ordered list of member Effectives (round robin / least loaded, optionally
  sticky per conversation), and it never hops to cloud — no ready member is a 503
  `overloaded_error`, which Claude Code retries while a seat loads.
- a **client rule** — a `User-Agent` prefix mapped to a backend, applied to UNPINNED turns
  before the classifier, so a plain model name (`claude-fable-5-1`) stops meaning "classify
  this" for a client that identified itself.

Written against the spec alone — the implementation (saint/config.py pool + ClientRule,
saint/router.py client rules, saint/binding.py pool_order, saint/dispatch.py +
saint/server.py pool plumbing, saint/explain.py) is being written in parallel, so these
imports fail until it lands. Do not stub the implementation here.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from saint import binding
from saint import config as config_mod
from saint.config import (
    BackendConfig,
    ClassifierConfig,
    ClientRule,
    Config,
    JohnnyConfig,
    LoggingConfig,
    RoutingConfig,
    ServerConfig,
    load_config,
)
from saint.dispatch import dispatch_candidates, run_candidates
from saint.johnny import Resolution
from saint.route_cache import Breaker, ConversationEntry, RouteCaches, TTLCache
from saint.server import build_app

# The spec names ConfigError; the loader today raises ValueError. Accept either (a
# ConfigError that subclasses ValueError is covered by the ValueError entry too).
_CONFIG_ERRORS: tuple[type[BaseException], ...] = tuple(dict.fromkeys(
    [ValueError, getattr(config_mod, "ConfigError", ValueError)]))

_OK = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"}],
       "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
_MSG_OK = {"id": "msg_01", "type": "message", "role": "assistant", "model": "m",
           "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
           "usage": {"input_tokens": 10, "output_tokens": 2}}
_CLS_RESP = {"choices": [{"message": {"content": json.dumps(
    {"domain": "code", "complexity": "medium", "reason": "."})}}]}
_IMG_ANTH = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": "AA"}}

# The three Flash-Next seats of the real config, in config order == rotation order.
MEMBERS = ("local-coder", "local-worker", "local-worker2")
ROLES = ("coder", "worker", "worker2")
POOL = "local-pool"
CLAUDE_UA = "claude-cli/2.1.303 (external, sdk-cli)"
_RULE = (ClientRule(name="claude-code", user_agent="claude-cli/", backend=POOL),)


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


def _pool(name: str = POOL, members: tuple[str, ...] = MEMBERS, **kw) -> BackendConfig:
    """A pool backend: no provider/model/base_url of its own (a pool is exempt from the
    static-baseline requirement exactly like johnny_only is). TOML spells these `strategy`
    and `sticky`; the dataclass fields are `pool_strategy`/`pool_sticky` (spec §1)."""
    return _b(name, provider=None, pool=tuple(members), **kw)


def _cfg(backends: list[BackendConfig], *, default_on_failure: str | None = None,
         multimodal_backend: str | None = None, policy_target: str | None = None,
         classifier_backend: str | None = None, embeddings_backend: str | None = None,
         while_loading: str | None = None, johnny: JohnnyConfig | None = None,
         clients: tuple[ClientRule, ...] = ()) -> Config:
    """A real Config over `backends` (list order == config order, which the pool rotation
    depends on). retry_same_backend is off so attempt counts stay readable."""
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
        clients=clients,
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


def _loading(seat: str) -> Resolution:
    """A warming seat — for a pool member that is an escape route, never an answer."""
    return Resolution(seat=seat, endpoint=None, model=None, state="loading", eta_s=42.0,
                      queue_depth=None)


def _down(seat: str) -> Resolution:
    """A seat johnny reports as absent (not loaded at all)."""
    return Resolution(seat=seat, endpoint=None, model=None, state="absent", eta_s=None,
                      queue_depth=None)


@pytest.fixture
def loads(monkeypatch):
    """Seat-load table (endpoint -> in-flight count); absent endpoint = unknown load."""
    table: dict[str, int] = {}
    monkeypatch.setattr(binding, "seat_load", lambda endpoint, timeout=0.5: table.get(endpoint))
    return table


@pytest.fixture(autouse=True)
def _fresh_rotation():
    """The round-robin counter is module state (spec §3: `_RR`, keyed per pool), so without a
    reset the sequence a test asserts would depend on which tests ran before it."""
    rr = getattr(binding, "_RR", None)
    if isinstance(rr, dict):
        rr.clear()
    yield


def _seats() -> FakeResolver:
    """Three ready seats on distinct ports, one per role, text-only unless overridden."""
    return FakeResolver(coder=_ready("seat-coder", 8101),
                        worker=_ready("seat-worker", 8102),
                        worker2=_ready("seat-worker2", 8103))


def _all_down() -> FakeResolver:
    """No member is reachable — the pool is empty."""
    return FakeResolver(coder=_loading("seat-coder"), worker=_down("seat-worker"),
                        worker2=None)


def _pool_cfg(*, members: tuple[str, ...] = MEMBERS, pool_kw: dict | None = None,
              clients: tuple[ClientRule, ...] = ()) -> Config:
    """cloud-large (the only cloud backend, and the failure fallback) + the named members +
    the pool on top. Members are johnny_only seats: their static config carries no endpoint,
    so a member is eligible only through a READY seat — the real Flash-Next shape, and the
    reason a warming seat empties the pool instead of serving cloud."""
    seats = [_b(m, provider=None, johnny_role=role, johnny_only=True)
             for m, role in zip(members, ROLES, strict=True)]
    kw = {"local_only": True}
    kw.update(pool_kw or {})
    cloud = replace(_b("cloud-large", provider="anthropic", base_url=None), aliases=("opus",))
    return _cfg([cloud, *seats,
                 _pool(POOL, members, **kw)],
                default_on_failure="cloud-large", policy_target="cloud-large",
                johnny=_johnny(), clients=clients)


def _caches() -> RouteCaches:
    """Route caches with room for a few conversations (sticky pools read `.conversations`)."""
    return RouteCaches(turns=TTLCache(300.0, 128), conversations=TTLCache(900.0, 128))


def _sticky(caches: RouteCaches, key: str, member: str) -> None:
    """Pretend `member` served this conversation's previous turn (what the server writes)."""
    caches.conversations.set(key, ConversationEntry(backend=member))


def _names(order: list) -> list[str]:
    """The member names a pool_order result would try, in order."""
    return [e.pool_member for e in order]


def _first(cfg, resolver, **kw) -> str:
    """The member pool_order would try first."""
    return _names(binding.pool_order(cfg, POOL, resolver, **kw))[0]


def _br(threshold: int = 3) -> Breaker:
    return Breaker(threshold=threshold, cooldown_s=60.0)


def _app(tmp_path, cfg):
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    return app, TestClient(app)


def _row(tmp_path, query):
    import sqlite3
    return sqlite3.connect(tmp_path / "log.sqlite").execute(query).fetchone()


async def _decide(cfg, *, model_field: str, ua: str | None = None,
                  messages: list | None = None, caches=None):
    from saint.router import decide_route

    return await decide_route(cfg=cfg, model_field=model_field,
                              messages=messages or [{"role": "user", "content": "hi"}],
                              caches=caches, client_user_agent=ua)


def _recorder(log: list, *, fail: bool = False):
    async def _attempt(eff):
        log.append(getattr(eff, "pool_member", None) or eff.backend.name)
        if fail:
            raise RuntimeError("boom")
        return "ok"
    return _attempt


def _fail_first(log: list):
    """Succeed from the second attempt on — the pool's next member IS the retry."""
    async def _attempt(eff):
        log.append(eff.pool_member)
        if len(log) == 1:
            raise RuntimeError("boom")
        return "ok"
    return _attempt


class _FakeStreamChunks:
    """Async iterator yielding fake litellm chunks."""

    def __init__(self, chunks: list[dict]):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


# ---------------------------------------------------------------- §1 config: pools

_TOML = """
[server]
host = "127.0.0.1"
port = 4000

[backends.cloud-large]
provider = "anthropic"
model = "claude-opus-5-5"
api_key_env = "ANTHROPIC_API_KEY"
aliases = ["opus"]
timeout_s = 120

[backends.local-coder]
provider = "openai"
base_url = "http://127.0.0.1:8101/v1"
model = "flash-next"
api_key = "x"

[backends.local-worker]
provider = "openai"
base_url = "http://127.0.0.1:8102/v1"
model = "flash-next"
api_key = "x"

[backends.local-worker2]
provider = "openai"
base_url = "http://127.0.0.1:8103/v1"
model = "flash-next"
api_key = "x"

[backends.local-pool]
pool = ["local-coder", "local-worker", "local-worker2"]
strategy = "round_robin"
sticky = true
local_only = true
aliases = ["pool"]

[classifier]
backend = "local-coder"
max_input_chars = 8000
timeout_s = 5

[routing]
default_urgency = "normal"
default_on_failure = "cloud-large"

[routing.policy.normal]
"code,trivial"    = "local-coder"
"code,medium"     = "local-coder"
"code,hard"       = "cloud-large"
"general,trivial" = "local-coder"
"general,medium"  = "local-coder"
"general,hard"    = "cloud-large"

[routing.policy.urgent]
"code,trivial"    = "local-coder"
"code,medium"     = "local-coder"
"code,hard"       = "cloud-large"
"general,trivial" = "local-coder"
"general,medium"  = "local-coder"
"general,hard"    = "cloud-large"

[routing.policy.patient]
"code,trivial"    = "local-coder"
"code,medium"     = "local-coder"
"code,hard"       = "local-coder"
"general,trivial" = "local-coder"
"general,medium"  = "local-coder"
"general,hard"    = "local-coder"

[logging]
db_path = "log.sqlite"
prompt_storage = "none"

[[clients]]
name = "claude-code"
user_agent = "claude-cli/"
backend = "local-pool"
"""


def _toml(tmp_path, text: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(text)
    return p


def test_pool_backend_parses_from_toml(tmp_path):
    cfg = load_config(_toml(tmp_path, _TOML))
    pool = cfg.backends[POOL]
    assert pool.pool == MEMBERS
    assert pool.is_pool is True
    assert pool.pool_strategy == "round_robin"
    assert pool.pool_sticky is True
    assert pool.local_only is True
    assert pool.aliases == ("pool",)
    assert cfg.backends["local-coder"].is_pool is False
    assert cfg.backends["local-coder"].pool == ()


def test_client_rules_parse_from_toml(tmp_path):
    cfg = load_config(_toml(tmp_path, _TOML))
    assert cfg.clients == _RULE
    assert cfg.backends["cloud-large"].pool == ()


def test_pool_needs_no_provider_model_or_base_url(tmp_path):
    """A pool is exempt from the static-baseline requirement, like johnny_only (spec §1)."""
    pool = load_config(_toml(tmp_path, _TOML)).backends[POOL]
    assert pool.provider is None and pool.model is None and pool.base_url is None


def test_pool_member_must_exist(tmp_path):
    toml = _TOML.replace('"local-worker2"]', '"local-worker2", "ghost"]')
    with pytest.raises(_CONFIG_ERRORS, match="ghost"):
        load_config(_toml(tmp_path, toml))


def test_pool_may_not_nest_a_pool(tmp_path):
    toml = _TOML.replace('pool = ["local-coder", "local-worker", "local-worker2"]',
                         'pool = ["local-coder", "local-pool"]')
    with pytest.raises(_CONFIG_ERRORS, match=POOL):
        load_config(_toml(tmp_path, toml))


def test_pool_strategy_must_be_known(tmp_path):
    with pytest.raises(_CONFIG_ERRORS, match="strategy"):
        load_config(_toml(tmp_path, _TOML.replace('strategy = "round_robin"',
                                                  'strategy = "random"')))


def test_pool_on_error_rejected_when_local_only(tmp_path):
    """local_only means "never leave the pool"; an on_error hop would contradict it."""
    toml = _TOML.replace("local_only = true", 'local_only = true\non_error = "cloud-large"')
    with pytest.raises(_CONFIG_ERRORS, match="on_error"):
        load_config(_toml(tmp_path, toml))


def test_pool_on_error_allowed_when_not_local_only(tmp_path):
    toml = _TOML.replace("local_only = true", 'local_only = false\non_error = "cloud-large"')
    cfg = load_config(_toml(tmp_path, toml))
    assert cfg.backends[POOL].on_error == "cloud-large"


@pytest.mark.parametrize("key,value", [
    ("johnny_role", '"coder"'),
    ("johnny_seat", '"seat-1"'),
    ("spill", '["coder"]'),
])
def test_pool_must_not_carry_a_johnny_binding_or_spill(tmp_path, key, value):
    """A pool resolves its MEMBERS; a binding on the pool itself would be a second,
    contradictory resolution path."""
    toml = _TOML.replace("local_only = true", f"local_only = true\n{key} = {value}")
    with pytest.raises(_CONFIG_ERRORS, match=key):
        load_config(_toml(tmp_path, toml))


def test_client_rule_backend_must_exist(tmp_path):
    toml = _TOML.replace('backend = "local-pool"', 'backend = "ghost"')
    with pytest.raises(_CONFIG_ERRORS, match="ghost"):
        load_config(_toml(tmp_path, toml))


def test_client_rule_user_agent_is_required(tmp_path):
    toml = _TOML.replace('user_agent = "claude-cli/"', "user_agent = ''")
    with pytest.raises(_CONFIG_ERRORS, match="user_agent"):
        load_config(_toml(tmp_path, toml))


def test_local_only_is_legal_on_a_plain_backend(tmp_path):
    """local_only is a property of a backend, not only of a pool: on a plain backend it means
    "no on_error / while_loading / default_on_failure hop for this one" (spec §1)."""
    toml = _TOML.replace("[backends.local-coder]",
                         '[backends.local-coder]\nlocal_only = true\non_error = "cloud-large"')
    cfg = load_config(_toml(tmp_path, toml))
    assert cfg.backends["local-coder"].local_only is True


# ---------------------------------------------------------------- §2 client rules

async def test_client_rule_matches_ua_prefix_case_insensitively():
    cfg = _pool_cfg(clients=_RULE)
    clf = AsyncMock(return_value=_CLS_RESP)
    with patch("saint.classifier.call_backend", clf):
        d = await _decide(cfg, model_field="claude-fable-5-1", ua="CLAUDE-CLI/2.1.303")
    assert d.backend == POOL
    assert d.client_rule == "claude-code"
    assert d.pinned_backend is None
    assert d.classifier_outcome is None and d.classifier_result is None
    assert clf.await_count == 0                       # the classifier never ran


async def test_client_rule_first_match_in_config_order_wins():
    cfg = _pool_cfg(clients=(*_RULE, ClientRule(name="other", user_agent="claude-cli/2.1",
                                                backend="local-worker")))
    d = await _decide(cfg, model_field="claude-fable-5-1", ua="claude-cli/2.1.0")
    assert d.backend == POOL and d.client_rule == "claude-code"


async def test_no_user_agent_takes_the_classifier_path():
    cfg = _pool_cfg(clients=_RULE)
    clf = AsyncMock(return_value=_CLS_RESP)
    with patch("saint.classifier.call_backend", clf):
        d = await _decide(cfg, model_field="saint-auto", ua=None)
    assert clf.await_count == 1
    assert d.backend == "cloud-large"                 # policy target, not the pool
    assert d.client_rule is None


async def test_unrelated_user_agent_takes_the_classifier_path():
    cfg = _pool_cfg(clients=_RULE)
    clf = AsyncMock(return_value=_CLS_RESP)
    with patch("saint.classifier.call_backend", clf):
        d = await _decide(cfg, model_field="saint-auto", ua="pi/0.9")
    assert clf.await_count == 1 and d.backend == "cloud-large" and d.client_rule is None


async def test_client_rule_does_not_override_a_model_field_pin():
    cfg = _pool_cfg(clients=_RULE)
    d = await _decide(cfg, model_field="saint-local-worker", ua=CLAUDE_UA)
    assert d.backend == "local-worker"
    assert d.pinned_backend == "local-worker"
    assert d.client_rule is None


async def test_client_rule_does_not_override_a_prefix_pin():
    cfg = _pool_cfg(clients=_RULE)
    d = await _decide(cfg, model_field="saint-auto", ua=CLAUDE_UA,
                      messages=[{"role": "user", "content": "!opus hi"}])
    assert d.backend == "cloud-large"                 # cloud-large aliases "opus"
    assert d.pinned_backend == "cloud-large"
    assert d.client_rule is None
    assert d.stripped_last_user == "hi"


async def test_saint_auto_from_a_matched_ua_goes_to_the_rule_backend():
    """`saint-auto` is not a pin — a matched client still gets its rule backend."""
    cfg = _pool_cfg(clients=_RULE)
    clf = AsyncMock(return_value=_CLS_RESP)
    with patch("saint.classifier.call_backend", clf):
        d = await _decide(cfg, model_field="saint-auto", ua=CLAUDE_UA)
    assert d.backend == POOL and d.client_rule == "claude-code"
    assert clf.await_count == 0


async def test_multimodal_turn_from_a_matched_ua_still_uses_the_rule_backend():
    """Capabilities are enforced at DISPATCH, not here — an image from Claude Code must not
    escape to the cloud multimodal backend."""
    cfg = _pool_cfg(clients=_RULE)
    d = await _decide(cfg, model_field="claude-fable-5-1", ua=CLAUDE_UA,
                      messages=[{"role": "user", "content": [
                          {"type": "text", "text": "what is this?"}, _IMG_ANTH]}])
    assert d.backend == POOL and d.client_rule == "claude-code"
    assert d.multimodal is True
    assert d.required_capabilities == frozenset({"vision"})


async def test_client_rule_sets_the_conversation_fields():
    """Affinity still needs its key + previous backend on this path: the pool is sticky, and
    the thinking-signature guard reads prev_backend."""
    cfg = _pool_cfg(clients=_RULE)
    caches = _caches()
    msgs = [{"role": "user", "content": "start a conversation"}]
    d = await _decide(cfg, model_field="claude-fable-5-1", ua=CLAUDE_UA, messages=msgs,
                      caches=caches)
    assert d.conversation_key
    _sticky(caches, d.conversation_key, "local-worker")
    d2 = await _decide(cfg, model_field="claude-fable-5-1", ua=CLAUDE_UA, messages=msgs,
                       caches=caches)
    assert d2.conversation_key == d.conversation_key
    assert d2.prev_backend == "local-worker"


async def test_explain_mode_still_works_for_a_matched_client():
    cfg = _pool_cfg(clients=_RULE)
    d = await _decide(cfg, model_field="saint-explain", ua=CLAUDE_UA)
    assert d.mode == "explain"


async def test_explain_prints_the_client_rule():
    from saint.explain import format_decision

    cfg = _pool_cfg(clients=_RULE)
    d = await _decide(cfg, model_field="claude-fable-5-1", ua=CLAUDE_UA)
    text = format_decision(d, cfg, resolver=_seats())
    assert "Client rule:" in text
    assert "claude-code" in text and POOL in text


# ---------------------------------------------------------------- §3 pool resolution

def test_effective_has_a_pool_member_field():
    """Effective records WHICH member a resolution came from (spec §3)."""
    assert binding.Effective(_b("x"), None, None).pool_member is None


def test_pool_order_round_robin_cycles_over_four_calls():
    cfg, r = _pool_cfg(), _seats()
    picks = [_first(cfg, r) for _ in range(4)]
    assert set(picks[:3]) == set(MEMBERS)              # all three seats served in a cycle
    assert picks[3] == picks[0]                        # and the fourth wraps to the first


def test_pool_order_rotates_the_whole_list_not_just_the_head():
    cfg, r = _pool_cfg(), _seats()
    first = _names(binding.pool_order(cfg, POOL, r))
    assert sorted(first) == sorted(MEMBERS)
    assert _first(cfg, r) == first[1]                  # what was second is now first


def test_pool_order_skips_a_loading_member():
    cfg, r = _pool_cfg(), _seats()
    r.seats["worker"] = _loading("seat-worker")
    assert _names(binding.pool_order(cfg, POOL, r)) == ["local-coder", "local-worker2"]


def test_pool_order_skips_an_absent_member():
    cfg, r = _pool_cfg(), _seats()
    r.seats["worker"] = _down("seat-worker")
    assert _names(binding.pool_order(cfg, POOL, r)) == ["local-coder", "local-worker2"]


def test_pool_order_skips_a_member_whose_johnny_is_unreachable():
    """None from johnny on a johnny_only member resolves to the cloud fallback — an escape,
    so the member is out of a local_only pool."""
    cfg, r = _pool_cfg(), _seats()
    r.seats["worker"] = None
    assert _names(binding.pool_order(cfg, POOL, r)) == ["local-coder", "local-worker2"]


def test_pool_order_skips_a_member_lacking_a_required_capability():
    cfg = _pool_cfg()
    r = FakeResolver(coder=_ready("seat-coder", 8101, capabilities=frozenset({"vision"})),
                     worker=_ready("seat-worker", 8102, capabilities=frozenset()),
                     worker2=_ready("seat-worker2", 8103, capabilities=frozenset({"vision"})))
    got = _names(binding.pool_order(cfg, POOL, r, required=frozenset({"vision"})))
    assert got == ["local-coder", "local-worker2"]


def test_pool_order_with_reasons_reports_why_a_member_dropped():
    cfg = _pool_cfg()
    r = FakeResolver(coder=_ready("seat-coder", 8101), worker=_loading("seat-worker"),
                     worker2=None)
    order, reasons = binding.pool_order_with_reasons(cfg, POOL, r)
    assert _names(order) == ["local-coder"]
    assert set(reasons) == {"local-worker", "local-worker2"}
    assert all(reasons[m] for m in reasons)            # a one-word reason is enough


def test_pool_order_least_loaded_picks_the_lowest_load(loads):
    cfg = _pool_cfg(pool_kw={"pool_strategy": "least_loaded"})
    r = _seats()
    loads["http://127.0.0.1:8101/v1"] = 7
    loads["http://127.0.0.1:8102/v1"] = 1
    loads["http://127.0.0.1:8103/v1"] = 3
    assert _names(binding.pool_order(cfg, POOL, r)) == ["local-worker", "local-worker2",
                                                        "local-coder"]


def test_pool_order_least_loaded_sorts_unknown_load_last(loads):
    """Unknown load is NOT idle (spec: `seat_load or 10**6`) — an unreadable seat must not
    win the sort over a seat measured idle."""
    cfg = _pool_cfg(pool_kw={"pool_strategy": "least_loaded"})
    r = _seats()
    loads["http://127.0.0.1:8102/v1"] = 0
    assert _first(cfg, r) == "local-worker"


def test_pool_order_least_loaded_ignores_the_rotation(loads):
    """Strategy, not history: two calls with the same loads pick the same member."""
    cfg = _pool_cfg(pool_kw={"pool_strategy": "least_loaded"})
    r = _seats()
    loads["http://127.0.0.1:8103/v1"] = 0
    assert _first(cfg, r) == _first(cfg, r) == "local-worker2"


def test_pool_order_sticky_prefers_the_conversation_member():
    cfg, r, caches = _pool_cfg(), _seats(), _caches()
    _sticky(caches, "conv-a", "local-worker2")
    assert _first(cfg, r, conversation_key="conv-a", caches=caches) == "local-worker2"


def test_pool_order_sticky_only_applies_to_the_conversation_that_had_one():
    cfg, r, caches = _pool_cfg(), _seats(), _caches()
    _sticky(caches, "conv-a", "local-worker2")
    assert _first(cfg, r, conversation_key="conv-b", caches=caches) in MEMBERS
    assert _first(cfg, r, conversation_key="conv-a", caches=caches) == "local-worker2"


def test_pool_order_sticky_does_not_advance_the_rotation():
    """The exact sequence: a sticky hit must not consume a rotation slot, or one chatty
    conversation would skew every other one (spec §3 step 2, the "Else" branch)."""
    cfg, r, caches = _pool_cfg(), _seats(), _caches()
    _sticky(caches, "conv-a", "local-worker2")
    first = _first(cfg, r)                                            # no conversation: rotation
    sticky = _first(cfg, r, conversation_key="conv-a", caches=caches)
    sticky_again = _first(cfg, r, conversation_key="conv-a", caches=caches)
    nxt = _first(cfg, r)                                              # rotation continues
    assert sticky == sticky_again == "local-worker2"
    assert nxt == MEMBERS[(MEMBERS.index(first) + 1) % len(MEMBERS)]  # advanced ONE slot


def test_pool_order_sticky_off_rotates_even_for_a_known_conversation():
    cfg = _pool_cfg(pool_kw={"pool_sticky": False})
    r, caches = _seats(), _caches()
    _sticky(caches, "conv-a", "local-worker2")
    picks = {_first(cfg, r, conversation_key="conv-a", caches=caches) for _ in range(3)}
    assert picks == set(MEMBERS)


def test_pool_order_sticky_falls_back_to_rotation_for_an_unknown_conversation():
    cfg, r = _pool_cfg(), _seats()
    picks = [_first(cfg, r, conversation_key="never-seen", caches=_caches()) for _ in range(3)]
    assert set(picks) == set(MEMBERS)


def test_pool_order_without_caches_still_rotates():
    """The CLI path passes no caches at all (spec §3 step 2)."""
    cfg, r = _pool_cfg(), _seats()
    picks = [_first(cfg, r, conversation_key="conv-a") for _ in range(3)]
    assert set(picks) == set(MEMBERS)


def test_pool_order_effectives_carry_the_member_and_its_own_endpoint():
    cfg, r = _pool_cfg(), _seats()
    for eff in binding.pool_order(cfg, POOL, r):
        assert eff.pool_member in MEMBERS
        assert eff.backend.name == eff.pool_member     # the member's OWN seat serves
        assert eff.state_at_dispatch in ("johnny_ready", "static_baseline", None)


def test_pool_order_empty_when_nothing_is_eligible():
    assert binding.pool_order(_pool_cfg(), POOL, _all_down()) == []


def test_resolve_for_dispatch_pool_returns_the_next_member():
    eff = binding.resolve_for_dispatch(_pool_cfg(), POOL, _seats())
    assert eff.pool_member in MEMBERS
    assert eff.backend.name == eff.pool_member


def test_resolve_for_dispatch_empty_pool_reports_pool_empty():
    eff = binding.resolve_for_dispatch(_pool_cfg(), POOL, _all_down())
    assert eff.state_at_dispatch == "pool_empty"
    assert eff.backend.name == POOL


def test_resolve_for_dispatch_non_pool_ignores_the_new_kwargs():
    eff = binding.resolve_for_dispatch(_pool_cfg(), "local-worker", _seats(), frozenset(),
                                       conversation_key="conv-a", caches=_caches())
    assert eff.backend.name == "local-worker" and eff.pool_member is None
    assert eff.state_at_dispatch == "johnny_ready"


def test_describe_for_explain_lists_members_and_the_next_pick():
    r = _seats()
    r.seats["worker"] = _loading("seat-worker")
    d = binding.describe_for_explain(_pool_cfg(), POOL, r)
    assert d["target"] == POOL and d["state"] == "pool"
    assert sorted(m["name"] for m in d["members"]) == sorted(MEMBERS)
    by_name = {m["name"]: m for m in d["members"]}
    assert by_name["local-coder"]["state"] == "eligible"
    assert by_name["local-coder"]["seat"] == "seat-coder"
    assert isinstance(by_name["local-coder"]["capabilities"], (list, tuple, set, frozenset))
    assert by_name["local-worker"]["state"] != "eligible"      # the reason word instead
    assert d["next"] in ("local-coder", "local-worker2")


def test_explain_peek_does_not_advance_the_rotation():
    cfg, r = _pool_cfg(), _seats()
    first = binding.describe_for_explain(cfg, POOL, r)["next"]
    assert binding.describe_for_explain(cfg, POOL, r)["next"] == first   # read-only, twice
    assert _first(cfg, r) == first                           # and the dispatch agrees


def test_explain_prints_pool_members_and_the_next_pick():
    from saint.explain import format_decision
    from saint.prefixes import ParsedPrefixes
    from saint.router import RoutingDecision

    r = _seats()
    r.seats["worker"] = _loading("seat-worker")
    decision = RoutingDecision(
        request_id="0" * 32, mode="explain", backend=POOL, pinned_backend=POOL,
        urgency="normal", parsed=ParsedPrefixes(urgency=None, pinned_backend=POOL,
                                               stripped="hi", raw=""),
        classifier_outcome=None, classifier_result=None, multimodal=False,
        model_field="saint-explain", last_user_content_original="hi", stripped_last_user="hi",
        required_capabilities=frozenset())
    text = format_decision(decision, _pool_cfg(), resolver=r)
    assert "member local-coder" in text and "eligible" in text
    assert "member local-worker" in text
    assert "next pick" in text


# ---------------------------------------------------------------- §4 dispatch

def test_dispatch_candidates_local_only_pool_is_the_only_candidate():
    cfg = _pool_cfg(pool_kw={"on_error": "cloud-large"})
    assert dispatch_candidates(cfg, POOL, _br()) == [POOL]


def test_dispatch_candidates_local_only_pool_does_not_widen_for_capabilities():
    """local_only = never leave the pool: an image turn must not add the cloud backend."""
    cfg = _pool_cfg()
    assert dispatch_candidates(cfg, POOL, _br(), required=frozenset({"vision"})) == [POOL]


def test_dispatch_candidates_local_only_pool_ignores_an_open_breaker():
    """The pool has no endpoint to skip — a member's breaker is decided in the walk."""
    cfg = _pool_cfg()
    br = _br(threshold=1)
    for m in MEMBERS:
        br.record_failure(m)
    assert dispatch_candidates(cfg, POOL, br) == [POOL]


def test_dispatch_candidates_pool_without_local_only_keeps_the_normal_walk():
    cfg = _pool_cfg(pool_kw={"local_only": False, "on_error": "cloud-large"})
    assert dispatch_candidates(cfg, POOL, _br()) == [POOL, "cloud-large"]


def test_dispatch_candidates_local_only_on_a_plain_backend():
    cfg = _cfg([_b("plain", on_error="other", local_only=True), _b("other")])
    assert dispatch_candidates(cfg, "plain", _br()) == ["plain"]


async def test_run_candidates_pool_first_member_fails_second_serves():
    cfg, r = _pool_cfg(), _seats()
    tried: list[str] = []
    served, error_kind, eff = await run_candidates(
        cfg=cfg, candidates=[POOL], rid="tst", resolver=r, breaker=_br(threshold=1),
        attempt=_fail_first(tried))
    assert error_kind is None and served is not None
    assert served.backend == POOL                      # the POOL is the served backend name
    assert served.eff.pool_member == tried[1] != tried[0]
    assert eff is served.eff
    assert len(tried) == 2                             # no same-backend retry inside a pool


async def test_run_candidates_records_the_breaker_under_the_member():
    cfg, r = _pool_cfg(), _seats()
    tried: list[str] = []
    br = _br(threshold=1)
    served, _, _ = await run_candidates(cfg=cfg, candidates=[POOL], rid="tst", resolver=r,
                                        breaker=br, attempt=_fail_first(tried))
    assert served is not None
    assert br.is_open(tried[0])                        # the member that failed
    assert not br.is_open(POOL)                        # never the pool name
    assert not br.is_open(tried[1])


async def test_run_candidates_skips_a_breaker_open_member():
    cfg, r = _pool_cfg(), _seats()
    tried: list[str] = []
    br = _br(threshold=1)
    br.record_failure("local-coder")
    br.record_failure("local-worker")
    served, error_kind, _ = await run_candidates(cfg=cfg, candidates=[POOL], rid="tst",
                                                 resolver=r, breaker=br,
                                                 attempt=_recorder(tried))
    assert tried == ["local-worker2"]
    assert served.eff.pool_member == "local-worker2" and error_kind is None


async def test_run_candidates_all_members_breaker_open_is_pool_unavailable():
    cfg, r = _pool_cfg(), _seats()
    br = _br(threshold=1)
    for m in MEMBERS:
        br.record_failure(m)
    tried: list[str] = []
    served, error_kind, _ = await run_candidates(cfg=cfg, candidates=[POOL], rid="tst",
                                                 resolver=r, breaker=br, attempt=_recorder(tried))
    assert served is None and tried == []
    assert error_kind.startswith("PoolUnavailable: no ready member in " + POOL)


async def test_run_candidates_empty_pool_lists_every_reason():
    cfg = _pool_cfg()
    served, error_kind, _ = await run_candidates(
        cfg=cfg, candidates=[POOL], rid="tst", resolver=_all_down(), breaker=_br(),
        attempt=_recorder([]))
    assert served is None
    assert error_kind.startswith("PoolUnavailable: no ready member in " + POOL)
    assert "(" in error_kind and ")" in error_kind
    for m in MEMBERS:                                  # the text says why each one is out
        assert m in error_kind


async def test_run_candidates_all_members_fail_keeps_the_real_error_kind():
    """An attempted-and-failed pool is a backend error (502), not an outage signal (503)."""
    tried: list[str] = []
    served, error_kind, _ = await run_candidates(
        cfg=_pool_cfg(), candidates=[POOL], rid="tst", resolver=_seats(),
        breaker=_br(threshold=9), attempt=_recorder(tried, fail=True))
    assert served is None and error_kind == "RuntimeError"
    assert len(tried) == len(MEMBERS)                  # each member tried exactly once


async def test_run_candidates_pool_never_attempts_an_incapable_member():
    cfg = _pool_cfg()
    r = FakeResolver(coder=_ready("seat-coder", 8101, capabilities=frozenset()),
                     worker=_ready("seat-worker", 8102, capabilities=frozenset({"vision"})),
                     worker2=_ready("seat-worker2", 8103, capabilities=frozenset({"vision"})))
    tried: list[str] = []
    served, error_kind, _ = await run_candidates(
        cfg=cfg, candidates=[POOL], rid="tst", resolver=r, breaker=_br(),
        attempt=_recorder(tried), required=frozenset({"vision"}))
    assert served is not None and error_kind is None
    assert "local-coder" not in tried
    assert served.eff.pool_member in ("local-worker", "local-worker2")


async def test_run_candidates_pool_honours_sticky():
    cfg, r, caches = _pool_cfg(), _seats(), _caches()
    _sticky(caches, "conv-a", "local-worker")
    tried: list[str] = []
    served, _, _ = await run_candidates(cfg=cfg, candidates=[POOL], rid="tst", resolver=r,
                                        breaker=_br(), attempt=_recorder(tried),
                                        conversation_key="conv-a", caches=caches)
    assert tried == ["local-worker"] and served.eff.pool_member == "local-worker"


async def test_run_candidates_non_pool_candidate_is_unchanged():
    cfg, r = _pool_cfg(), _seats()
    tried: list[str] = []
    served, _, eff = await run_candidates(cfg=cfg, candidates=["local-worker"], rid="tst",
                                          resolver=r, breaker=_br(), attempt=_recorder(tried),
                                          conversation_key="conv-a", caches=_caches())
    assert tried == ["local-worker"] and served.backend == "local-worker"
    assert eff.pool_member is None


# ---------------------------------------------------------------- §5 endpoints

def _endpoint_cfg() -> Config:
    """Static (unbound) members, so these tests need no resolver: eligibility comes from a
    plain static backend and each member has its own port for readable failures."""
    seats = [_b(m, base_url=f"http://127.0.0.1:{8101 + i}/v1") for i, m in enumerate(MEMBERS)]
    return _cfg([_b("cloud-large", provider="anthropic", base_url=None), *seats,
                 _pool(POOL, MEMBERS, local_only=True)],
                default_on_failure="cloud-large", policy_target="cloud-large",
                clients=_RULE)


def test_messages_claude_code_served_by_a_pool_member(tmp_path):
    _, client = _app(tmp_path, _endpoint_cfg())
    acreate = AsyncMock(return_value=_MSG_OK)
    clf = AsyncMock(return_value=_CLS_RESP)
    with patch("saint.backends.litellm.anthropic.messages.acreate", acreate), \
         patch("saint.classifier.call_backend", clf):
        resp = client.post("/v1/messages", headers={"user-agent": "claude-cli/2.1.0"},
                           json={"model": "claude-fable-5-1", "max_tokens": 10,
                                 "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert resp.headers["x-saint-backend"] == POOL
    assert resp.headers["x-saint-pool-member"] in MEMBERS
    assert clf.await_count == 0                        # the classifier was skipped
    assert acreate.await_count == 1                    # exactly one member was tried


def test_messages_sticky_member_is_reused_within_a_conversation(tmp_path):
    _, client = _app(tmp_path, _endpoint_cfg())
    acreate = AsyncMock(return_value=_MSG_OK)
    first_turn = [{"role": "user", "content": "first turn of this conversation"}]
    with patch("saint.backends.litellm.anthropic.messages.acreate", acreate):
        r1 = client.post("/v1/messages", headers={"user-agent": CLAUDE_UA},
                         json={"model": "claude-fable-5-1", "max_tokens": 10,
                               "messages": first_turn})
        r2 = client.post("/v1/messages", headers={"user-agent": CLAUDE_UA},
                         json={"model": "claude-fable-5-1", "max_tokens": 10,
                               "messages": [*first_turn,
                                            {"role": "assistant", "content": "ok"},
                                            {"role": "user", "content": "and now?"}]})
        # a NEW conversation must take the next seat, not the sticky one
        r3 = client.post("/v1/messages", headers={"user-agent": CLAUDE_UA},
                         json={"model": "claude-fable-5-1", "max_tokens": 10,
                               "messages": [{"role": "user",
                                             "content": "a different chat"}]})
    first = r1.headers["x-saint-pool-member"]
    assert first in MEMBERS
    assert r2.headers["x-saint-pool-member"] == first  # the seat's prefix cache is reused
    assert r3.headers["x-saint-pool-member"] != first  # rotation advanced for a new chat


def test_messages_pinned_member_sets_no_pool_header(tmp_path):
    """A named backend served this turn, not a pool — the header would be a lie."""
    _, client = _app(tmp_path, _endpoint_cfg())
    with patch("saint.backends.litellm.anthropic.messages.acreate",
               AsyncMock(return_value=_MSG_OK)):
        resp = client.post("/v1/messages", headers={"user-agent": CLAUDE_UA},
                           json={"model": "saint-local-worker", "max_tokens": 10,
                                 "messages": [{"role": "user", "content": "hi"}]})
    assert resp.headers["x-saint-backend"] == "local-worker"
    assert "x-saint-pool-member" not in resp.headers


def test_chat_completions_pi_shape_still_classifies(tmp_path):
    """No Claude UA + saint-auto: pi keeps today's behaviour — classifier, policy target,
    no pool involvement."""
    _, client = _app(tmp_path, _endpoint_cfg())
    completion = AsyncMock(return_value=_OK)
    clf = AsyncMock(return_value=_CLS_RESP)
    with patch("saint.backends.litellm.acompletion", completion), \
         patch("saint.classifier.call_backend", clf):
        resp = client.post("/v1/chat/completions", headers={"user-agent": "pi/0.9"},
                           json={"model": "saint-auto",
                                 "messages": [{"role": "user",
                                               "content": "refactor the widget parser"}]})
    assert resp.status_code == 200
    assert resp.headers["x-saint-backend"] == "cloud-large"
    assert "x-saint-pool-member" not in resp.headers
    assert clf.await_count == 1


def test_chat_completions_claude_code_never_reaches_the_cloud(tmp_path):
    """A hard-looking turn from Claude Code stays local: the rule beats the classifier."""
    _, client = _app(tmp_path, _endpoint_cfg())
    completion = AsyncMock(return_value=_OK)
    clf = AsyncMock(return_value=_CLS_RESP)
    with patch("saint.backends.litellm.acompletion", completion), \
         patch("saint.classifier.call_backend", clf):
        resp = client.post("/v1/chat/completions", headers={"user-agent": CLAUDE_UA},
                           json={"model": "claude-fable-5-1",
                                 "messages": [{"role": "user",
                                               "content": "design a distributed rate limiter"}]})
    assert resp.status_code == 200
    assert resp.headers["x-saint-backend"] == POOL
    assert resp.headers["x-saint-pool-member"] in MEMBERS
    assert clf.await_count == 0


def test_streaming_pool_turn_sets_the_member_header(tmp_path):
    _, client = _app(tmp_path, _endpoint_cfg())
    chunks = [
        {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
         "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]},
        {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    with patch("saint.backends.litellm.acompletion",
               AsyncMock(return_value=_FakeStreamChunks(chunks))):
        with client.stream("POST", "/v1/chat/completions", headers={"user-agent": CLAUDE_UA},
                           json={"model": "claude-fable-5-1", "stream": True,
                                 "messages": [{"role": "user", "content": "hi"}]}) as resp:
            headers = dict(resp.headers)
            text = "".join(resp.iter_text())
    assert headers["x-saint-backend"] == POOL
    assert headers["x-saint-pool-member"] in MEMBERS
    assert "Hel" in text


def _no_seat_app(tmp_path, cfg: Config | None = None):
    """An app whose every pool member is loading/absent — the pool resolves empty."""
    app, client = _app(tmp_path, cfg or _pool_cfg())
    app.state.resolver = _all_down()
    return app, client


def test_pool_unavailable_openai_is_503_no_local_seat(tmp_path):
    _, client = _no_seat_app(tmp_path)
    completion = AsyncMock(return_value=_OK)
    clf = AsyncMock(return_value=_CLS_RESP)
    with patch("saint.backends.litellm.acompletion", completion), \
         patch("saint.classifier.call_backend", clf):
        resp = client.post("/v1/chat/completions",
                           json={"model": "saint-" + POOL,
                                 "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 503
    err = resp.json()["error"]
    assert err["type"] == "server_error"
    assert err["code"] == "no_local_seat"
    assert not err["message"].startswith("PoolUnavailable")   # the prefix is for the log
    assert POOL in err["message"]
    assert resp.headers["retry-after"] == "5"
    assert completion.await_count == 0                 # nothing left the box


def test_pool_unavailable_anthropic_is_overloaded_error(tmp_path):
    """Claude Code retries overloaded_error — exactly what we want while a seat loads."""
    _, client = _no_seat_app(tmp_path, _pool_cfg(clients=_RULE))
    acreate = AsyncMock(return_value=_MSG_OK)
    clf = AsyncMock(return_value=_CLS_RESP)
    with patch("saint.backends.litellm.anthropic.messages.acreate", acreate), \
         patch("saint.classifier.call_backend", clf):
        resp = client.post("/v1/messages", headers={"user-agent": CLAUDE_UA},
                           json={"model": "claude-fable-5-1", "max_tokens": 10,
                                 "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 503
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "overloaded_error"
    assert POOL in body["error"]["message"]
    assert acreate.await_count == 0
    # anthropic_error() takes no headers today; if it grew the parameter, the hint is required
    from saint.anthropic_api import anthropic_error
    if "headers" in inspect.signature(anthropic_error).parameters:
        assert resp.headers["retry-after"] == "5"


def test_pool_unavailable_is_logged_as_the_error(tmp_path):
    _, client = _no_seat_app(tmp_path)
    with patch("saint.backends.litellm.acompletion", AsyncMock(return_value=_OK)), \
         patch("saint.classifier.call_backend", AsyncMock(return_value=_CLS_RESP)):
        client.post("/v1/chat/completions",
                    json={"model": "saint-" + POOL,
                          "messages": [{"role": "user", "content": "hi"}]})
    row = _row(tmp_path, "SELECT success, error_kind FROM requests ORDER BY id DESC LIMIT 1")
    assert row[0] == 0
    assert row[1].startswith("PoolUnavailable")


def test_pool_unavailable_does_not_route_through_default_on_failure(tmp_path):
    """local_only is the whole point: a warming seat must not quietly become a cloud bill."""
    cfg = _pool_cfg()
    cfg = replace(cfg, routing=replace(cfg.routing, while_loading="cloud-large"))
    _, client = _no_seat_app(tmp_path, cfg)
    completion = AsyncMock(return_value=_OK)
    with patch("saint.backends.litellm.acompletion", completion), \
         patch("saint.classifier.call_backend", AsyncMock(return_value=_CLS_RESP)):
        resp = client.post("/v1/chat/completions",
                           json={"model": "saint-" + POOL,
                                 "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 503
    assert completion.await_count == 0
