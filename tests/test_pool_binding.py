"""Pool backends — member resolution (saint/binding.py §3) and the pool dispatch walk
(saint/dispatch.py §4).

A pool is a backend whose `pool` field names other backends: it never serves itself, it
picks one of its members per turn (round-robin or least-loaded, optionally sticky per
conversation) and never escapes to cloud.

FIXTURE NOTE — these tests are written against the spec while the config work happens in
parallel, so `BackendConfig` here has NO `pool`/`pool_strategy`/`pool_sticky`/`local_only`
fields. Rather than branch on whether they exist, `_attrs()` copies the BackendConfig and
stamps the attributes directly (`object.__setattr__`, which works on a frozen dataclass).
That is correct both before the fields land and after: the implementation reads them with
`getattr(b, "pool", ())` etc., so a stamped attribute and a real field are the same thing.
Once the fields exist upstream, this can become a plain `dataclasses.replace`.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from saint import binding
from saint.binding import Effective, describe_for_explain, pool_order, pool_order_with_reasons
from saint.config import (
    BackendConfig,
    ClassifierConfig,
    Config,
    JohnnyConfig,
    LoggingConfig,
    RoutingConfig,
    ServerConfig,
)
from saint.dispatch import dispatch_candidates, run_candidates
from saint.johnny import Resolution
from saint.route_cache import Breaker, ConversationEntry, RouteCaches, TTLCache

# ---------------------------------------------------------------- test scaffolding


def _b(name: str, *, provider: str | None = "openai", model: str | None = None,
       base_url: str | None = "http://static.invalid/v1", api_key_env: str | None = None,
       capabilities: tuple[str, ...] = (), **kw) -> BackendConfig:
    return BackendConfig(
        name=name, provider=provider, model=model or name,
        api_key_env=api_key_env, api_key=("x" if provider == "openai" else None),
        base_url=(base_url if provider == "openai" else None),
        aliases=(), timeout_s=10, capabilities=tuple(capabilities), **kw,
    )


def _attrs(b: BackendConfig, **attrs: Any) -> BackendConfig:
    """See the module docstring: stamp pool fields onto a copy of a BackendConfig."""
    out = copy.copy(b)
    for k, v in attrs.items():
        object.__setattr__(out, k, v)
    return out


def _pool(name: str, members: list[str], *, strategy: str = "round_robin", sticky: bool = True,
          local_only: bool = True, **kw) -> BackendConfig:
    """The pool backend itself carries no provider/model/endpoint — its members do."""
    return _attrs(_b(name, provider=None, model=None, base_url=None, **kw),
                  pool=tuple(members), pool_strategy=strategy, pool_sticky=sticky,
                  local_only=local_only)


def _cfg(backends: list[BackendConfig], *, default_on_failure: str | None = None,
         while_loading: str | None = None, johnny: JohnnyConfig | None = None) -> Config:
    names = [b.name for b in backends]
    target = names[0]
    policy = {u: {f"{d},{c}": target for d in ("code", "general") for c in ("trivial", "medium", "hard")}
              for u in ("normal", "urgent", "patient")}
    return Config(
        server=ServerConfig(host="127.0.0.1", port=4000),
        backends={b.name: b for b in backends},
        classifier=ClassifierConfig(backend=target, fallback_backend=None,
                                    max_input_chars=8000, timeout_s=5, prompt_template_path=None),
        routing=RoutingConfig(default_urgency="normal",
                              default_on_failure=default_on_failure or names[-1],
                              policy=policy, while_loading=while_loading,
                              retry_same_backend=False, breaker_failures=3, breaker_cooldown_s=60.0),
        logging=LoggingConfig(db_path=":memory:", prompt_storage="none"),
        johnny=johnny,
    )


def _johnny() -> JohnnyConfig:
    return JohnnyConfig(transport="cli", cli_path="johnny-must-not-be-called", base_url=None,
                        resolve_cache_ttl_s=1.0, ensure_load=False)


class FakeResolver:
    def __init__(self, **by_target: Resolution | None):
        self.seats = dict(by_target)
        self.loads_requested: list[str] = []

    def resolve(self, target: str) -> Resolution | None:
        return self.seats.get(target)

    def ensure_load(self, target: str) -> None:
        self.loads_requested.append(target)


def _ready(seat: str, port: int, capabilities: frozenset[str] | None = None,
           model: str = "m") -> Resolution:
    return Resolution(seat=seat, endpoint=f"http://127.0.0.1:{port}/v1", model=model,
                      state="ready", eta_s=None, queue_depth=None, capabilities=capabilities)


def _not_ready(seat: str, state: str = "loading") -> Resolution:
    return Resolution(seat=seat, endpoint=None, model=None, state=state, eta_s=30.0,
                      queue_depth=None)


def _members(*specs: tuple[str, str, int]):
    """[(name, johnny role, port)] -> ready member backends, in config order."""
    return [_b(n, johnny_role=r, base_url=f"http://static.invalid/{n}/v1") for n, r, _ in specs]


def _seat_only(name: str, role: str) -> BackendConfig:
    """A member with no static fallback: while its seat is not ready it resolves to an escape
    (while_loading/fallback), which makes it INELIGIBLE for a pool. A member that keeps a
    static base_url stays eligible through a reload — that is the existing binding contract."""
    return _b(name, johnny_role=role, johnny_only=True, base_url=None)


def _resolver(*specs: tuple[str, str, int], **extra: Resolution | None) -> FakeResolver:
    """Ready seats keyed by role, one per (name, role, port) spec."""
    return FakeResolver(**{r: _ready(n, p) for n, r, p in specs}, **extra)


@pytest.fixture(autouse=True)
def _reset_rr(monkeypatch):
    """The round-robin cursor is module state; each test starts at zero."""
    binding._RR.clear()
    yield
    binding._RR.clear()


@pytest.fixture
def loads(monkeypatch):
    table: dict[str, int] = {}
    monkeypatch.setattr(binding, "seat_load", lambda endpoint, timeout=0.5: table.get(endpoint))
    return table


def _caches(**by_key: str) -> RouteCaches:
    """RouteCaches whose conversation cache holds ConversationEntry(backend=...) per key."""
    conv = TTLCache(ttl_s=900.0, max_entries=512)
    for k, backend in by_key.items():
        conv.set(k, ConversationEntry(backend=backend))
    return RouteCaches(turns=None, conversations=conv)


def _names(order: list[Effective]) -> list[str | None]:
    return [e.pool_member for e in order]


def _recorder(log: list, value="ok", fail_for: str | tuple[str, ...] | None = None):
    fails = (fail_for,) if isinstance(fail_for, str) else tuple(fail_for or ())

    async def _attempt(eff):
        log.append(eff.backend.name)
        if eff.backend.name in fails:
            raise RuntimeError("boom")
        return value
    return _attempt


# ---------------------------------------------------------------- Effective / helpers

def test_effective_pool_member_defaults_to_none():
    eff = Effective(_b("x"), None, None)
    assert eff.pool_member is None


def test_is_pool_reads_the_field_defensively():
    from saint.binding import is_pool
    assert is_pool(_pool("p", ["a"])) is True
    assert is_pool(_b("plain")) is False          # no pool attribute at all -> not a pool


# ---------------------------------------------------------------- §3 pool_order

def test_round_robin_cycles_over_four_calls():
    cfg = _cfg([_pool("p", ["a", "b", "c"]), *_members(("a", "ra", 8001), ("b", "rb", 8002),
                                                        ("c", "rc", 8003))], johnny=_johnny())
    r = _resolver(("a", "ra", 8001), ("b", "rb", 8002), ("c", "rc", 8003))
    picks = [_names(pool_order(cfg, "p", r))[0] for _ in range(4)]
    assert picks == ["a", "b", "c", "a"]


def test_round_robin_tags_each_effective_with_its_member():
    cfg = _cfg([_pool("p", ["a", "b"]), *_members(("a", "ra", 8001), ("b", "rb", 8002))],
               johnny=_johnny())
    r = _resolver(("a", "ra", 8001), ("b", "rb", 8002))
    order = pool_order(cfg, "p", r)
    assert _names(order) == ["a", "b"]
    assert order[0].state_at_dispatch == "johnny_ready"
    assert order[0].backend.base_url == "http://127.0.0.1:8001/v1"


def test_round_robin_skips_loading_and_absent_members():
    cfg = _cfg([_pool("p", ["a", "b", "c"]),
                _b("a", johnny_role="ra", base_url="http://static.invalid/a/v1"),
                _seat_only("b", "rb"),
                _b("c", johnny_role="rc", base_url="http://static.invalid/c/v1")],
               johnny=_johnny())
    r = FakeResolver(ra=_ready("sa", 8001), rb=_not_ready("sb"), rc=_ready("sc", 8003))
    # the cursor advances once per call over the ELIGIBLE list, not the configured one
    assert [_names(pool_order(cfg, "p", r))[0] for _ in range(5)] == ["a", "c", "a", "c", "a"]
    assert pool_order_with_reasons(cfg, "p", r)[1] == {"b": "loading"}


def test_a_member_escaping_to_cloud_is_ineligible_with_a_reason(loads):
    """while_loading is an escape hatch to cloud — a pool member that takes it is not 'ready
    locally', so it must drop out rather than quietly send the turn off-box."""
    cloud = _b("cloud", provider="anthropic", base_url=None, api_key_env="K")
    cfg = _cfg([_pool("p", ["a", "b"]), cloud,
                _b("a", johnny_role="ra", johnny_only=True, base_url=None),
                _b("b", johnny_role="rb", base_url="http://static.invalid/b/v1")],
               while_loading="cloud", johnny=_johnny())
    r = FakeResolver(ra=_not_ready("sa"), rb=_ready("sb", 8002))
    order, reasons = pool_order_with_reasons(cfg, "p", r)
    assert _names(order) == ["b"]
    assert reasons == {"a": "loading"}


def test_a_member_lacking_a_required_capability_is_dropped():
    cfg = _cfg([_pool("p", ["a", "b"]),
                _b("a", johnny_role="ra", base_url="http://static.invalid/a/v1"),
                _b("b", johnny_role="rb", base_url="http://static.invalid/b/v1")],
               johnny=_johnny())
    r = FakeResolver(ra=_ready("sa", 8001, capabilities=frozenset()),
                     rb=_ready("sb", 8002, capabilities=frozenset({"vision"})))
    order, reasons = pool_order_with_reasons(cfg, "p", r, required=frozenset({"vision"}))
    assert _names(order) == ["b"]
    assert reasons == {"a": "lacks vision"}


def test_least_loaded_picks_the_lowest_load_first(loads):
    cfg = _cfg([_pool("p", ["a", "b", "c"], strategy="least_loaded"),
                *_members(("a", "ra", 8001), ("b", "rb", 8002), ("c", "rc", 8003))],
               johnny=_johnny())
    r = _resolver(("a", "ra", 8001), ("b", "rb", 8002), ("c", "rc", 8003))
    loads["http://127.0.0.1:8001/v1"] = 7
    loads["http://127.0.0.1:8002/v1"] = 1
    loads["http://127.0.0.1:8003/v1"] = 4
    assert _names(pool_order(cfg, "p", r)) == ["b", "c", "a"]


def test_least_loaded_sorts_unknown_load_last(loads):
    cfg = _cfg([_pool("p", ["a", "b"], strategy="least_loaded"),
                *_members(("a", "ra", 8001), ("b", "rb", 8002))], johnny=_johnny())
    r = _resolver(("a", "ra", 8001), ("b", "rb", 8002))
    loads["http://127.0.0.1:8002/v1"] = 9      # a's load is unreadable
    assert _names(pool_order(cfg, "p", r)) == ["b", "a"]


def test_sticky_puts_the_previous_member_first_without_advancing_the_cursor():
    cfg = _cfg([_pool("p", ["a", "b", "c"]), *_members(("a", "ra", 8001), ("b", "rb", 8002),
                                                        ("c", "rc", 8003))], johnny=_johnny())
    r = _resolver(("a", "ra", 8001), ("b", "rb", 8002), ("c", "rc", 8003))
    caches = _caches()
    assert _names(pool_order(cfg, "p", r, conversation_key="k", caches=caches))[0] == "a"
    caches.conversations.set("k", ConversationEntry(backend="b"))
    assert _names(pool_order(cfg, "p", r, conversation_key="k", caches=caches))[0] == "b"
    # every later turn of THIS conversation stays on b, and the cursor did not move, so a new
    # conversation continues the rotation where it would have been.
    assert _names(pool_order(cfg, "p", r, conversation_key="k", caches=caches))[0] == "b"
    # one rotation slot was consumed (the first call); a new conversation takes the NEXT one
    assert _names(pool_order(cfg, "p", r, conversation_key="other", caches=caches))[0] == "b"


def test_sticky_falls_back_to_rotation_when_the_member_is_no_longer_eligible():
    cfg = _cfg([_pool("p", ["a", "b"]),
                _b("a", johnny_role="ra", base_url="http://static.invalid/a/v1"),
                _seat_only("b", "rb")], johnny=_johnny())
    r = FakeResolver(ra=_ready("sa", 8001), rb=_not_ready("sb"))
    caches = _caches(k="b")                    # affinity points at the seat that just went away
    assert _names(pool_order(cfg, "p", r, conversation_key="k", caches=caches)) == ["a"]


def test_sticky_off_rotates_even_within_one_conversation():
    cfg = _cfg([_pool("p", ["a", "b"], sticky=False), *_members(("a", "ra", 8001), ("b", "rb", 8002))],
               johnny=_johnny())
    r = _resolver(("a", "ra", 8001), ("b", "rb", 8002))
    caches = _caches(k="b")
    assert [_names(pool_order(cfg, "p", r, conversation_key="k", caches=caches))[0]
            for _ in range(3)] == ["a", "b", "a"]


def test_no_conversation_key_or_caches_is_pure_rotation():
    cfg = _cfg([_pool("p", ["a", "b"]), *_members(("a", "ra", 8001), ("b", "rb", 8002))],
               johnny=_johnny())
    r = _resolver(("a", "ra", 8001), ("b", "rb", 8002))
    assert [_names(pool_order(cfg, "p", r, caches=_caches(k="b")))[0] for _ in range(2)] == ["a", "b"]


def test_missing_member_name_is_absent():
    cfg = _cfg([_pool("p", ["a", "ghost"]), _b("a", johnny_role="ra",
                                               base_url="http://static.invalid/a/v1")],
               johnny=_johnny())
    order, reasons = pool_order_with_reasons(cfg, "p", FakeResolver(ra=_ready("sa", 8001)))
    assert _names(order) == ["a"] and reasons == {"ghost": "absent"}


def test_a_static_only_member_is_eligible():
    """johnny absent entirely: the member's own static endpoint is still local, so it counts."""
    cfg = _cfg([_pool("p", ["a"]), _b("a", base_url="http://127.0.0.1:8009/v1")])
    order = pool_order(cfg, "p", None)
    assert _names(order) == ["a"] and order[0].state_at_dispatch is None


def test_empty_pool_orders_to_nothing():
    cfg = _cfg([_pool("p", ["a"]), _b("a", johnny_role="ra", johnny_only=True, base_url=None)],
               johnny=_johnny())
    r = FakeResolver(ra=_not_ready("sa"))
    assert pool_order(cfg, "p", r) == []
    assert pool_order_with_reasons(cfg, "p", r)[1] == {"a": "loading"}


# ---------------------------------------------------------------- §3 resolve_for_dispatch

def test_resolve_for_dispatch_on_a_pool_returns_the_next_member():
    cfg = _cfg([_pool("p", ["a", "b"]), *_members(("a", "ra", 8001), ("b", "rb", 8002))],
               johnny=_johnny())
    r = _resolver(("a", "ra", 8001), ("b", "rb", 8002))
    eff = binding.resolve_for_dispatch(cfg, "p", r)
    assert eff.pool_member == "a" and eff.backend.name == "a"
    assert binding.resolve_for_dispatch(cfg, "p", r).pool_member == "b"


def test_resolve_for_dispatch_on_an_empty_pool_says_so():
    cfg = _cfg([_pool("p", ["a"]), _b("a", johnny_role="ra", johnny_only=True, base_url=None)],
               johnny=_johnny())
    eff = binding.resolve_for_dispatch(cfg, "p", FakeResolver(ra=_not_ready("sa")))
    assert eff.state_at_dispatch == "pool_empty" and eff.pool_member is None


def test_resolve_for_dispatch_non_pool_ignores_the_new_kwargs():
    cfg = _cfg([_b("a", johnny_role="ra", base_url="http://static.invalid/a/v1")], johnny=_johnny())
    eff = binding.resolve_for_dispatch(cfg, "a", FakeResolver(ra=_ready("sa", 8001)),
                                       conversation_key="k", caches=_caches(k="a"))
    assert eff.state_at_dispatch == "johnny_ready" and eff.pool_member is None


# ---------------------------------------------------------------- §3 describe_for_explain

def test_describe_reports_members_and_next_pick():
    cfg = _cfg([_pool("p", ["a", "b"]), *_members(("a", "ra", 8001), ("b", "rb", 8002))],
               johnny=_johnny())
    r = FakeResolver(ra=_ready("sa", 8001), rb=_ready("sb", 8002))
    d = describe_for_explain(cfg, "p", r)
    assert d["state"] == "pool" and d["target"] == "p" and d["next"] == "a"
    assert d["members"][0] == {"name": "a", "state": "eligible", "seat": "sa",
                               "capabilities": []}


def test_describe_names_the_reason_for_a_dropped_member():
    cfg = _cfg([_pool("p", ["a", "b"]),
                _b("a", johnny_role="ra", johnny_only=True, base_url=None),
                _b("b", johnny_role="rb", base_url="http://static.invalid/b/v1")],
               johnny=_johnny())
    r = FakeResolver(ra=_not_ready("sa", "absent"), rb=_ready("sb", 8002))
    d = describe_for_explain(cfg, "p", r)
    assert [m["state"] for m in d["members"]] == ["absent", "eligible"]


def test_describe_does_not_advance_the_round_robin_cursor():
    cfg = _cfg([_pool("p", ["a", "b"]), *_members(("a", "ra", 8001), ("b", "rb", 8002))],
               johnny=_johnny())
    r = _resolver(("a", "ra", 8001), ("b", "rb", 8002))
    assert describe_for_explain(cfg, "p", r)["next"] == "a"
    assert describe_for_explain(cfg, "p", r)["next"] == "a"     # peeking is free
    assert _names(pool_order(cfg, "p", r))[0] == "a"            # ...and the real pick is too
    assert _names(pool_order(cfg, "p", r))[0] == "b"


def test_describe_never_triggers_a_member_load():
    """explain is a look, not a dispatch — even with ensure_load on in the config."""
    cfg = _cfg([_pool("p", ["a", "b"]), _seat_only("a", "ra"),
                _b("b", johnny_role="rb", base_url="http://static.invalid/b/v1")],
               johnny=JohnnyConfig(transport="cli", cli_path="j", base_url=None,
                                   resolve_cache_ttl_s=1.0, ensure_load=True))
    r = FakeResolver(ra=_not_ready("sa"), rb=_ready("sb", 8002))
    describe_for_explain(cfg, "p", r)
    assert r.loads_requested == []
    pool_order(cfg, "p", r)                       # a real dispatch DOES kick the loading seat
    assert r.loads_requested == ["ra"]


async def test_an_empty_pool_outranks_a_capability_error():
    """A warming local seat must read as the retryable PoolUnavailable, not a 400 telling the
    client its turn is impossible."""
    cfg = _cfg([_pool("p", ["a"]), _seat_only("a", "ra"),
                _b("text", base_url="http://127.0.0.1:8009/v1")], johnny=_johnny())
    r = FakeResolver(ra=_not_ready("sa"))
    served, error_kind, _ = await run_candidates(
        cfg=cfg, candidates=["text", "p"], rid="tst", resolver=r, breaker=Breaker(3, 60.0),
        attempt=_recorder([]), required=frozenset({"vision"}))
    assert served is None and error_kind.startswith("PoolUnavailable: ")


# ---------------------------------------------------------------- §4 dispatch_candidates

def test_local_only_candidate_is_alone_even_with_capabilities_and_on_error():
    pool = _pool("p", ["a"], local_only=True, on_error="cloud")
    cfg = _cfg([pool, _b("cloud", provider="anthropic", base_url=None, api_key_env="K",
                         capabilities=("vision",)),
                _b("a", johnny_role="ra", base_url="http://static.invalid/a/v1",
                   capabilities=("vision",))])
    br = Breaker(threshold=1, cooldown_s=60.0)
    br.record_failure("a")                    # an open breaker is moot: the members are the failover
    assert dispatch_candidates(cfg, "p", br, required=frozenset({"vision"})) == ["p"]


def test_a_pool_without_local_only_behaves_like_a_normal_backend():
    pool = _pool("p", ["a"], local_only=False, on_error="cloud")
    cfg = _cfg([pool, _b("cloud", provider="anthropic", base_url=None, api_key_env="K")])
    got = dispatch_candidates(cfg, "p", Breaker(threshold=3, cooldown_s=60.0))
    assert got == ["p", "cloud"]


# ---------------------------------------------------------------- §4 run_candidates

async def test_pool_failure_falls_to_the_next_member_and_blames_the_member():
    cfg = _cfg([_pool("p", ["a", "b"]), *_members(("a", "ra", 8001), ("b", "rb", 8002))],
               johnny=_johnny())
    r = _resolver(("a", "ra", 8001), ("b", "rb", 8002))
    tried: list[str] = []
    br = Breaker(threshold=1, cooldown_s=60.0)     # one failure opens a member's circuit
    served, error_kind, _ = await run_candidates(
        cfg=cfg, candidates=["p"], rid="tst", resolver=r, breaker=br,
        attempt=_recorder(tried, fail_for="a"))
    assert tried == ["a", "b"]
    assert served is not None and served.backend == "p" and error_kind is None
    assert served.eff.pool_member == "b" and served.value == "ok"
    assert br.is_open("a")                        # the sick seat is out of rotation...
    assert not br.is_open("p")                    # ...the healthy pool name is not


async def test_pool_member_is_attempted_exactly_once():
    cfg = _cfg([_pool("p", ["a"]), *_members(("a", "ra", 8001))], johnny=_johnny())
    r = _resolver(("a", "ra", 8001))
    tried: list[str] = []
    served, error_kind, _ = await run_candidates(
        cfg=cfg, candidates=["p"], rid="tst", resolver=r, breaker=Breaker(3, 60.0),
        attempt=_recorder(tried, fail_for="a"))
    assert tried == ["a"] and served is None and error_kind == "RuntimeError"


async def test_breaker_open_member_is_skipped_without_attempt(capsys):
    cfg = _cfg([_pool("p", ["a", "b"]), *_members(("a", "ra", 8001), ("b", "rb", 8002))],
               johnny=_johnny())
    r = _resolver(("a", "ra", 8001), ("b", "rb", 8002))
    br = Breaker(threshold=1, cooldown_s=60.0)
    br.record_failure("a")
    tried: list[str] = []
    served, _, eff = await run_candidates(cfg=cfg, candidates=["p"], rid="tst", resolver=r,
                                          breaker=br, attempt=_recorder(tried))
    assert tried == ["b"] and served.backend == "p" and eff.pool_member == "b"
    assert "circuit open" in capsys.readouterr().err


async def test_every_member_breaker_open_reports_pool_unavailable():
    cfg = _cfg([_pool("p", ["a", "b"]), *_members(("a", "ra", 8001), ("b", "rb", 8002))],
               johnny=_johnny())
    r = _resolver(("a", "ra", 8001), ("b", "rb", 8002))
    br = Breaker(threshold=1, cooldown_s=60.0)
    br.record_failure("a")
    br.record_failure("b")
    tried: list[str] = []
    served, error_kind, eff = await run_candidates(cfg=cfg, candidates=["p"], rid="tst",
                                                   resolver=r, breaker=br, attempt=_recorder(tried))
    assert tried == [] and served is None and eff is None
    assert error_kind == "PoolUnavailable: no ready member in p (a: breaker open; b: breaker open)"


async def test_empty_pool_error_lists_every_reason():
    cfg = _cfg([_pool("p", ["a", "b", "ghost"]),
                _b("a", johnny_role="ra", johnny_only=True, base_url=None),
                _b("b", johnny_role="rb", base_url="http://static.invalid/b/v1")],
               johnny=_johnny())
    r = FakeResolver(ra=_not_ready("sa"), rb=_ready("sb", 8002, capabilities=frozenset()))
    served, error_kind, _ = await run_candidates(
        cfg=cfg, candidates=["p"], rid="tst", resolver=r, breaker=Breaker(3, 60.0),
        attempt=_recorder([]), required=frozenset({"vision"}))
    assert served is None
    assert error_kind == ("PoolUnavailable: no ready member in p "
                          "(a: loading; b: lacks vision; ghost: absent)")


async def test_pool_walk_is_sticky_across_turns_of_one_conversation():
    cfg = _cfg([_pool("p", ["a", "b", "c"]), *_members(("a", "ra", 8001), ("b", "rb", 8002),
                                                        ("c", "rc", 8003))], johnny=_johnny())
    r = _resolver(("a", "ra", 8001), ("b", "rb", 8002), ("c", "rc", 8003))
    caches = _caches()
    tried: list[str] = []

    async def _turn(key: str) -> str:
        _, _, eff = await run_candidates(cfg=cfg, candidates=["p"], rid="tst", resolver=r,
                                         breaker=Breaker(3, 60.0), attempt=_recorder(tried),
                                         conversation_key=key, caches=caches)
        caches.conversations.set(key, ConversationEntry(backend=eff.pool_member))
        return eff.pool_member

    assert [await _turn("k") for _ in range(3)] == ["a", "a", "a"]   # one conversation, one seat
    assert await _turn("fresh") == "b"                               # a new one takes the next slot


async def test_pool_after_a_failed_candidate_keeps_the_real_error():
    """A pool tried after something else failed reports its own outcome, not the stale one."""
    cfg = _cfg([_b("first"), _pool("p", ["a"]), *_members(("a", "ra", 8001))], johnny=_johnny())
    r = _resolver(("a", "ra", 8001))
    tried: list[str] = []
    served, error_kind, _ = await run_candidates(
        cfg=cfg, candidates=["first", "p"], rid="tst", resolver=r, breaker=Breaker(3, 60.0),
        attempt=_recorder(tried, fail_for=("first", "a")))
    assert tried == ["first", "a"] and served is None and error_kind == "RuntimeError"


async def test_non_pool_candidate_path_is_unchanged():
    cfg = _cfg([_b("a", johnny_role="ra", base_url="http://static.invalid/a/v1")], johnny=_johnny())
    tried: list[str] = []
    served, error_kind, eff = await run_candidates(
        cfg=cfg, candidates=["a"], rid="tst", resolver=FakeResolver(ra=_ready("sa", 8001)),
        breaker=Breaker(3, 60.0), attempt=_recorder(tried), conversation_key="k",
        caches=_caches(k="a"))
    assert tried == ["a"] and served.backend == "a" and error_kind is None
    assert eff.pool_member is None
