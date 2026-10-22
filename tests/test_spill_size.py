"""Seat load from a llama.cpp seat, and size-aware spill (see saint/sizing.py).

Two halves of the same feature: a spill target that is not vLLM still has to report a load,
and a spill target with a smaller context window (or a slow cold prefill) must not be handed a
turn it cannot hold. Sizing is an ESTIMATE — chars/3.0 with base64 and thinking text skipped —
so the limits it is compared against are guards, not billing numbers."""
from __future__ import annotations

import urllib.request

import pytest

from saint import binding, johnny
from saint.config import BackendConfig, Config, JohnnyConfig, load_config
from saint.johnny import Resolution, seat_load
from saint.sizing import estimate_request_tokens, requested_output_tokens

# ---------------------------------------------------------------- seat_load


class _Resp:
    def __init__(self, text: str):
        self._text = text

    def read(self) -> bytes:
        return self._text.encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def metrics(monkeypatch):
    """A setter for the /metrics text any seat answers with; clears the ~1 s load cache so no
    cached answer leaks between tests."""
    johnny._load_cache.clear()
    state = {"text": ""}
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout=None: _Resp(state["text"]))
    yield lambda text: state.__setitem__("text", text)
    johnny._load_cache.clear()


_VLLM = (
    "# HELP vllm:num_requests_running Number of requests currently running on GPU.\n"
    "# TYPE vllm:num_requests_running gauge\n"
    'vllm:num_requests_running{model_name="qwen"} 3.0\n'
    "# HELP vllm:num_requests_waiting Number of requests waiting to be processed.\n"
    "# TYPE vllm:num_requests_waiting gauge\n"
    'vllm:num_requests_waiting{model_name="qwen"} 2.0\n'
)

_LLAMA = (
    "# HELP llamacpp:requests_processing Number of requests processing\n"
    "# TYPE llamacpp:requests_processing gauge\n"
    "llamacpp:requests_processing 2\n"
    "# HELP llamacpp:requests_deferred Number of requests deferred\n"
    "# TYPE llamacpp:requests_deferred gauge\n"
    "llamacpp:requests_deferred 1\n"
)

def test_seat_load_vllm_running_plus_waiting(metrics):
    metrics(_VLLM)
    assert seat_load("http://127.0.0.1:8006/v1") == 5


def test_seat_load_llamacpp_processing_plus_deferred(metrics):
    metrics(_LLAMA)
    assert seat_load("http://127.0.0.1:8080/v1") == 3


def test_seat_load_prefers_vllm_lines_when_both_present(metrics):
    metrics(_LLAMA + _VLLM)
    assert seat_load("http://127.0.0.1:8006/v1") == 5


def test_seat_load_neither_format_is_unknown(metrics):
    metrics("# HELP something_else x\n# TYPE something_else gauge\nfoo 7\n")
    assert seat_load("http://127.0.0.1:8006/v1") is None


def test_seat_load_unreadable_is_unknown(monkeypatch):
    johnny._load_cache.clear()

    def boom(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert seat_load("http://127.0.0.1:9/v1") is None      # must not raise


# ---------------------------------------------------------------- sizing


def test_estimate_counts_strings():
    assert estimate_request_tokens({"a": "x" * 300}) == 100
    turn = {"messages": [{"role": "user", "content": "hello there"}]}
    assert estimate_request_tokens(turn) == int((len("user") + len("hello there")) / 3.0)


def test_estimate_skips_base64_payloads():
    # a `data` value and a `data:` URL are base64, not text — a 1 MB image is not a 300K-token turn
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 7000}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "A" * 7000}},
    ]}]}
    assert estimate_request_tokens(body) < 20


def test_estimate_skips_thinking_text():
    # earlier turns' `thinking` text is ~70 % of a long Claude Code body but the chat template
    # drops it, so counting it tripled the estimate (measured 2026-10-22: 344K vs 133K real)
    body = {"messages": [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "T" * 10_000, "signature": "S" * 500},
        {"type": "text", "text": "t" * 3_000},
    ]}]}
    # only the 3,000-char text block is tokenised: 3000 / 3.0 = 1000, plus the ~20 chars the walk
    # does count — the two block `type` tags and the role
    assert abs(estimate_request_tokens(body) - 1000) <= 10
    bare = {"messages": [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "", "signature": ""},
        {"type": "text", "text": "t" * 3_000},
    ]}]}
    assert estimate_request_tokens(body) == estimate_request_tokens(bare)   # thinking text is free
    # the top-level request option is a dict under the same key name — skipped with it, sizes 0
    assert estimate_request_tokens({"thinking": {"type": "enabled", "budget_tokens": 8192}}) == 0


def test_estimate_empty_and_odd_input():
    assert estimate_request_tokens({}) == 0
    assert estimate_request_tokens(None) == 0          # type: ignore[arg-type]


def test_requested_output_tokens():
    assert requested_output_tokens({"max_tokens": 4096}) == 4096
    assert requested_output_tokens({"max_completion_tokens": 1024}) == 1024
    assert requested_output_tokens({"max_tokens": 4096, "max_completion_tokens": 1024}) == 4096
    assert requested_output_tokens({}) == 0
    assert requested_output_tokens({"max_tokens": "4096"}) == 0
    assert requested_output_tokens({"max_tokens": None}) == 0


# ---------------------------------------------------------------- _spill size gate


def _cfg(chat_ctx=None, chat_spill_max=None, spill=("chat",)):
    """worker spills to role `chat`; `local-chat` is the backend that serves that role, and the
    two limits under test live on IT (the spill target), not on the borrower."""
    b = {
        "local-worker": BackendConfig(name="local-worker", provider="openai", model="m",
                                      api_key_env=None, api_key="x",
                                      base_url="http://static:8006/v1", aliases=(), timeout_s=10,
                                      johnny_role="worker", spill=spill, spill_at=6),
        "local-chat": BackendConfig(name="local-chat", provider="openai", model="m",
                                    api_key_env=None, api_key="x",
                                    base_url="http://static:8003/v1", aliases=(), timeout_s=10,
                                    johnny_role="chat", context=chat_ctx,
                                    spill_max_prompt=chat_spill_max),
        "local-coder": BackendConfig(name="local-coder", provider="openai", model="m",
                                     api_key_env=None, api_key="x",
                                     base_url="http://static:8002/v1", aliases=(), timeout_s=10,
                                     johnny_role="coder", context=262144),
        "cloud": BackendConfig(name="cloud", provider="anthropic", model="c",
                               api_key_env=None, api_key="x", base_url=None, aliases=(),
                               timeout_s=10),
    }
    cfg = Config.__new__(Config)
    object.__setattr__(cfg, "backends", b)
    object.__setattr__(cfg, "johnny", JohnnyConfig(
        transport="cli", cli_path="johnny", base_url=None, resolve_cache_ttl_s=1, ensure_load=False))

    class R:  # routing stub
        while_loading = None
        default_on_failure = "cloud"
        retry_same_backend = False
    object.__setattr__(cfg, "routing", R())
    return cfg


class FakeResolver:
    def __init__(self, seats):
        self.seats = seats

    def resolve(self, target):
        return self.seats.get(target)

    def ensure_load(self, target):
        pass


def _ready(seat, port):
    return Resolution(seat=seat, endpoint=f"http://127.0.0.1:{port}/v1", model="m",
                      state="ready", eta_s=None, queue_depth=None)


@pytest.fixture
def loads(monkeypatch):
    table = {}
    monkeypatch.setattr(binding, "seat_load", lambda endpoint, timeout=0.5: table.get(endpoint))
    return table


def _spill(cfg, resolver, size, own_seat="w"):
    return binding._spill(cfg, cfg.backends["local-worker"], resolver, own_seat, None,
                          frozenset(), size)


def test_spill_skipped_when_turn_exceeds_target_context(loads, capsys):
    cfg = _cfg(chat_ctx=131072)
    loads["http://127.0.0.1:8003/v1"] = 0
    r = FakeResolver({"chat": _ready("h", 8003)})
    assert _spill(cfg, r, (100000, 32000)) is None           # 132000 > 131072
    assert "spill skip chat: request ~100000+32000 tokens > context 131072" in capsys.readouterr().err
    got = _spill(cfg, r, (50000, 32000))
    assert got is not None and got[0].seat == "h"


def test_spill_size_gate_only_applies_when_size_known(loads):
    cfg = _cfg(chat_ctx=131072)
    loads["http://127.0.0.1:8003/v1"] = 0
    r = FakeResolver({"chat": _ready("h", 8003)})
    got = _spill(cfg, r, None)                               # callers that pass no size are unchanged
    assert got is not None and got[0].seat == "h"


def test_spill_skipped_when_prompt_exceeds_spill_max_prompt(loads, capsys):
    cfg = _cfg(chat_spill_max=65536)
    loads["http://127.0.0.1:8003/v1"] = 0
    r = FakeResolver({"chat": _ready("h", 8003)})
    assert _spill(cfg, r, (70000, 32000)) is None
    assert "spill skip chat: request ~70000+32000 tokens > spill_max_prompt 65536" in capsys.readouterr().err
    got = _spill(cfg, r, (60000, 32000))                     # spill_max_prompt gates the PROMPT only
    assert got is not None and got[0].seat == "h"


def test_size_skip_falls_through_to_next_spill_candidate(loads):
    # chat is listed first AND is the less-loaded seat, so it wins on load — until size rules it out
    cfg = _cfg(chat_ctx=131072, spill=("chat", "coder"))
    loads["http://127.0.0.1:8003/v1"] = 0
    loads["http://127.0.0.1:8002/v1"] = 1
    r = FakeResolver({"chat": _ready("h", 8003), "coder": _ready("c", 8002)})
    assert _spill(cfg, r, (50000, 32000))[0].seat == "h"
    assert _spill(cfg, r, (200000, 32000))[0].seat == "c"    # fits coder's 262K, not chat's 131K


def test_size_gate_reaches_spill_via_dispatch(loads):
    """The gate has to be reachable from the real entry point, not only from _spill itself."""
    cfg = _cfg(chat_ctx=131072)
    loads["http://127.0.0.1:8006/v1"] = 9                    # own seat saturated -> spill
    loads["http://127.0.0.1:8003/v1"] = 0
    r = FakeResolver({"worker": _ready("w", 8006), "chat": _ready("h", 8003)})
    eff = binding.resolve_for_dispatch(cfg, "local-worker", r, size=(100000, 32000))
    assert eff.johnny_seat == "w" and eff.spilled_from is None
    eff = binding.resolve_for_dispatch(cfg, "local-worker", r, size=(10000, 32000))
    assert eff.johnny_seat == "h" and eff.spilled_from == "worker"


async def test_size_gate_reaches_spill_via_run_candidates(loads):
    """And through the dispatch walk the endpoints actually use."""
    from saint.dispatch import run_candidates
    from saint.route_cache import Breaker

    cfg = _cfg(chat_ctx=131072)
    loads["http://127.0.0.1:8006/v1"] = 9                     # own seat saturated -> spill
    loads["http://127.0.0.1:8003/v1"] = 0
    r = FakeResolver({"worker": _ready("w", 8006), "chat": _ready("h", 8003)})
    tried = []

    async def attempt(eff):
        tried.append(eff.backend.base_url)
        return "ok"

    async def go(size):
        tried.clear()
        served, error_kind, _ = await run_candidates(
            cfg=cfg, candidates=["local-worker"], rid="tst", resolver=r,
            breaker=Breaker(threshold=3, cooldown_s=60.0), attempt=attempt, size=size)
        assert served is not None and error_kind is None
        return tried[0]

    assert await go((100000, 32000)) == "http://127.0.0.1:8006/v1"   # too big to borrow: own seat
    assert await go((10000, 32000)) == "http://127.0.0.1:8003/v1"    # fits: the spill seat


# ---------------------------------------------------------------- config parse

_TOML = """
[server]
host = "127.0.0.1"
port = 4000

[backends.cloud]
provider = "anthropic"
model = "claude-opus"
api_key_env = "ANTHROPIC_API_KEY"

[backends.local-chat]
provider = "openai"
model = "qwen"
base_url = "http://127.0.0.1:8080/v1"
johnny_role = "chat"
context = 131072
spill_max_prompt = 65536

[backends.local-worker]
provider = "openai"
model = "flash"
base_url = "http://127.0.0.1:8006/v1"
johnny_role = "worker"
spill = ["chat"]

[classifier]
backend = "cloud"

[routing]
default_urgency = "normal"
default_on_failure = "cloud"
while_loading = "cloud"

[routing.policy.normal]
"code,trivial" = "local-worker"
"code,medium"  = "local-worker"
"code,hard"    = "cloud"
"general,trivial" = "local-worker"
"general,medium"  = "local-worker"
"general,hard"    = "cloud"

[routing.policy.urgent]
"code,trivial" = "local-worker"
"code,medium"  = "local-worker"
"code,hard"    = "cloud"
"general,trivial" = "local-worker"
"general,medium"  = "local-worker"
"general,hard"    = "cloud"

[routing.policy.patient]
"code,trivial" = "local-worker"
"code,medium"  = "local-worker"
"code,hard"    = "cloud"
"general,trivial" = "local-worker"
"general,medium"  = "local-worker"
"general,hard"    = "local-worker"

[johnny]
transport = "cli"
cli_path = "johnny"
ensure_load = false

[logging]
db_path = "/tmp/saint-test-spillsize.db"
prompt_storage = "none"
"""


def test_spill_max_prompt_parsed(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(_TOML)
    cfg = load_config(p)
    assert cfg.backends["local-chat"].spill_max_prompt == 65536
    assert cfg.backends["local-chat"].context == 131072
    assert cfg.backends["local-worker"].spill_max_prompt is None   # absent = no limit
