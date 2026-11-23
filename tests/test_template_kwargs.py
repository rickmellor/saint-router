"""Per-backend `chat_template_kwargs` defaults ride in extra_body under the client's own keys;
cloud providers never see them. A seat's chat template has its own defaults (the RTX text seat
defaults reasoning_effort to `xhigh` → 433K chars of thinking for a 1.5K-char docs turn,
2026-11-23), so a backend can pin them in config."""
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import pytest

from saint.backends import _shape_request, call_backend_messages
from saint.config import BackendConfig
from tests.test_config import EXAMPLE_TOML


def _b(provider="openai", ctk=(), **kw):
    return BackendConfig(name="b", provider=provider, model="m", api_key_env=None, api_key="x",
                         base_url="http://x/v1", aliases=(), timeout_s=10,
                         chat_template_kwargs=ctk, **kw)


# --- shaping ---

def test_backend_default_lands_in_extra_body():
    k = _shape_request(_b(ctk=(("reasoning_effort", "medium"),)), {"model": "m", "messages": []})
    assert k["extra_body"] == {"chat_template_kwargs": {"reasoning_effort": "medium"}}


def test_client_keys_win_per_key_and_backend_keys_survive():
    k = _shape_request(_b(ctk=(("reasoning_effort", "medium"), ("enable_thinking", True))),
                       {"model": "m", "messages": [],
                        "chat_template_kwargs": {"reasoning_effort": "low"}})
    assert k["extra_body"]["chat_template_kwargs"] == {"reasoning_effort": "low",
                                                       "enable_thinking": True}
    assert "chat_template_kwargs" not in k       # only ever delivered inside extra_body


def test_client_sending_one_key_still_gets_backend_defaults():
    k = _shape_request(_b(ctk=(("reasoning_effort", "medium"),)),
                       {"model": "m", "messages": [], "chat_template_kwargs": {"enable_thinking": False}})
    assert k["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False,
                                                        "reasoning_effort": "medium"}}


def test_client_empty_dict_and_backend_default():
    # A client that sends an empty dict still gets the backend defaults (nothing to override).
    k = _shape_request(_b(ctk=(("reasoning_effort", "medium"),)),
                       {"model": "m", "messages": [], "chat_template_kwargs": {}})
    assert k["extra_body"] == {"chat_template_kwargs": {"reasoning_effort": "medium"}}


def test_non_dict_client_value_not_merged():
    k = _shape_request(_b(ctk=(("reasoning_effort", "medium"),)),
                       {"model": "m", "messages": [], "chat_template_kwargs": ["nope"]})
    assert k["extra_body"] == {"chat_template_kwargs": ["nope"]}


def test_caller_extra_body_key_wins():
    # `extra.setdefault` semantics unchanged: an explicit extra_body entry is not overwritten.
    k = _shape_request(_b(ctk=(("reasoning_effort", "medium"),)),
                       {"model": "m", "messages": [],
                        "extra_body": {"chat_template_kwargs": {"a": 1}}})
    assert k["extra_body"] == {"chat_template_kwargs": {"a": 1}}


def test_cloud_never_gets_template_kwargs():
    for provider in ("anthropic", "bedrock"):
        b = _b(provider=provider, ctk=(("reasoning_effort", "medium"),))
        if provider == "bedrock":
            b = replace(b, aws_region="us-east-1", api_key=None, base_url=None)
        k = _shape_request(b, {"model": "m", "messages": [],
                               "chat_template_kwargs": {"enable_thinking": False}})
        assert "chat_template_kwargs" not in k
        assert "extra_body" not in k


def test_nothing_set_adds_no_extra_body():
    k = _shape_request(_b(), {"model": "m", "messages": []})
    assert "extra_body" not in k
    k = _shape_request(_b(ctk=(("reasoning_effort", "medium"),)),
                       {"model": "m", "messages": [], "chat_template_kwargs": None})
    assert k["extra_body"] == {"chat_template_kwargs": {"reasoning_effort": "medium"}}


def test_empty_client_value_and_no_backend_default_adds_no_extra_body():
    k = _shape_request(_b(), {"model": "m", "messages": [], "chat_template_kwargs": {}})
    assert "extra_body" not in k


def test_template_kwargs_with_priority():
    k = _shape_request(_b(priority=5, ctk=(("reasoning_effort", "medium"),)),
                       {"model": "m", "messages": [], "chat_template_kwargs": {"enable_thinking": False}})
    assert k["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False,
                                                        "reasoning_effort": "medium"},
                               "priority": 5}


# --- dispatch path: /v1/messages → an openai backend goes through litellm's messages bridge ---

async def test_call_backend_messages_sends_template_kwargs():
    b = _b(ctk=(("reasoning_effort", "medium"),))
    acreate = AsyncMock(return_value={"id": "msg_01", "content": []})
    with patch("saint.backends.litellm.anthropic.messages.acreate", acreate):
        await call_backend_messages(b, params={"max_tokens": 100, "messages": []}, stream=False)
    kw = acreate.call_args.kwargs
    assert kw["model"] == "hosted_vllm/m"
    assert kw["extra_body"]["chat_template_kwargs"] == {"reasoning_effort": "medium"}


# --- the field survives the derivations in saint/binding.py (johnny override, spill, pool) ---

def test_johnny_resolve_and_spill_carry_the_field(monkeypatch):
    from saint import binding
    from saint.config import Config, JohnnyConfig
    from saint.johnny import Resolution

    def seat(port, **kw):
        return BackendConfig(name="b", provider="openai", model="m", api_key_env=None, api_key="x",
                             base_url=f"http://static:{port}/v1", aliases=(), timeout_s=10,
                             johnny_role=("worker" if port == 8006 else "coder"), **kw)
    b = {"local-worker": seat(8006, spill=("coder",), spill_at=6,
                              chat_template_kwargs=(("reasoning_effort", "medium"),)),
         "local-coder": seat(8002),
         "cloud": BackendConfig(name="cloud", provider="anthropic", model="c", api_key_env=None,
                                api_key="x", base_url=None, aliases=(), timeout_s=10)}
    cfg = Config.__new__(Config)
    object.__setattr__(cfg, "backends", b)
    object.__setattr__(cfg, "johnny", JohnnyConfig(transport="cli", cli_path="johnny", base_url=None,
                                                   resolve_cache_ttl_s=1, ensure_load=False))
    class R:
        while_loading = None
        default_on_failure = "cloud"
    object.__setattr__(cfg, "routing", R())

    class FakeResolver:
        def resolve(self, target):
            return Resolution(seat=target, endpoint=f"http://127.0.0.1:{8006 if target == 'worker' else 8002}/v1",
                              model="m", state="ready", eta_s=None, queue_depth=None)
        def ensure_load(self, target):
            pass

    loads = {"http://127.0.0.1:8006/v1": 2, "http://127.0.0.1:8002/v1": 0}
    monkeypatch.setattr(binding, "seat_load", lambda endpoint, timeout=0.5: loads[endpoint])

    # own seat ready: johnny swapped endpoint/model only — the default still shapes the request
    eff = binding.resolve_for_dispatch(cfg, "local-worker", FakeResolver())
    assert eff.spilled_from is None
    assert eff.backend.chat_template_kwargs_map == {"reasoning_effort": "medium"}

    # own seat saturated → the CODER's seat serves, but with the REQUESTING backend's config:
    # spill only replaces base_url/model, so local-worker's chat_template_kwargs applies.
    loads["http://127.0.0.1:8006/v1"] = 9                   # own seat saturated
    eff = binding.resolve_for_dispatch(cfg, "local-worker", FakeResolver())
    assert eff.spilled_from == "worker" and eff.johnny_seat == "coder"
    assert eff.backend.base_url == "http://127.0.0.1:8002/v1"
    assert eff.backend.chat_template_kwargs_map == {"reasoning_effort": "medium"}

    # pool: the MEMBER's own config shapes the request (the pool itself may not set it)
    pool_name = "pool-ctk"                          # unique: binding._RR is process-global
    b[pool_name] = BackendConfig(name=pool_name, provider=None, model=None, api_key_env=None,
                                 api_key=None, base_url=None, aliases=(), timeout_s=10,
                                 pool=("local-worker", "local-coder"))
    loads["http://127.0.0.1:8006/v1"] = 2
    order = binding.pool_order(cfg, pool_name, FakeResolver())
    assert [e.pool_member for e in order] == ["local-worker", "local-coder"]
    assert order[0].backend.chat_template_kwargs_map == {"reasoning_effort": "medium"}
    assert order[1].backend.chat_template_kwargs_map == {}


# --- config ---

def _write(tmp_path, toml, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("TEST_HOME", str(tmp_path))
    p = tmp_path / "c.toml"
    p.write_text(toml)
    return p


def test_toml_table_round_trips(tmp_path, monkeypatch):
    from saint.config import load_config
    toml = EXAMPLE_TOML.replace('aliases = []\ntimeout_s = 60',
                                'aliases = []\ntimeout_s = 60\n'
                                'chat_template_kwargs = { reasoning_effort = "medium", enable_thinking = false }')
    cfg = load_config(_write(tmp_path, toml, monkeypatch))
    assert cfg.backends["local-small"].chat_template_kwargs_map == {"reasoning_effort": "medium",
                                                                    "enable_thinking": False}
    assert cfg.backends["local-small"].chat_template_kwargs == (("enable_thinking", False),
                                                                ("reasoning_effort", "medium"))
    assert hash(cfg.backends["local-small"])          # frozen + hashable, as before
    assert cfg.backends["cloud-large"].chat_template_kwargs == ()


def test_non_table_is_a_config_error(tmp_path, monkeypatch):
    from saint.config import load_config
    toml = EXAMPLE_TOML.replace('aliases = []\ntimeout_s = 60',
                                'aliases = []\ntimeout_s = 60\nchat_template_kwargs = "medium"')
    with pytest.raises(ValueError, match="local-small.*chat_template_kwargs"):
        load_config(_write(tmp_path, toml, monkeypatch))


def test_setting_it_on_an_anthropic_backend_is_a_config_error(tmp_path, monkeypatch):
    from saint.config import load_config
    toml = EXAMPLE_TOML.replace('aliases = ["opus"]',
                                'aliases = ["opus"]\nchat_template_kwargs = { reasoning_effort = "medium" }')
    with pytest.raises(ValueError, match="cloud-large.*chat_template_kwargs"):
        load_config(_write(tmp_path, toml, monkeypatch))


def test_setting_it_on_a_pool_is_a_config_error(tmp_path, monkeypatch):
    from saint.config import load_config
    toml = EXAMPLE_TOML + """
[backends.local-pool]
provider = "openai"
model = "m"
pool = ["local-small"]
chat_template_kwargs = { reasoning_effort = "medium" }
"""
    with pytest.raises(ValueError, match="local-pool.*chat_template_kwargs"):
        load_config(_write(tmp_path, toml, monkeypatch))
