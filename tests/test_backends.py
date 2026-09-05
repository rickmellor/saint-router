from unittest.mock import AsyncMock, patch

import pytest

from saint.backends import call_backend
from saint.config import BackendConfig


@pytest.fixture
def cloud_backend() -> BackendConfig:
    return BackendConfig(
        name="cloud-large", provider="anthropic", model="claude-opus-4-7",
        api_key_env="ANTHROPIC_API_KEY", api_key=None, base_url=None,
        aliases=("opus",), timeout_s=120,
    )


@pytest.fixture
def local_backend() -> BackendConfig:
    return BackendConfig(
        name="local-small", provider="openai", model="qwen2.5-3b-instruct",
        api_key_env=None, api_key="lm-studio",
        base_url="http://localhost:1234/v1", aliases=(), timeout_s=60,
    )


async def test_call_anthropic_backend_translates_model(cloud_backend, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    mock = AsyncMock(return_value={"choices": [{"message": {"content": "hi"}}]})
    with patch("saint.backends.litellm.acompletion", mock):
        await call_backend(cloud_backend, messages=[{"role": "user", "content": "hi"}], stream=False)
    _, kwargs = mock.call_args
    # litellm anthropic models get the "anthropic/" prefix
    assert kwargs["model"] == "anthropic/claude-opus-4-7"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["timeout"] == 120
    assert kwargs["stream"] is False


async def test_call_openai_compatible_with_base_url(local_backend):
    mock = AsyncMock(return_value={"choices": []})
    with patch("saint.backends.litellm.acompletion", mock):
        await call_backend(local_backend, messages=[{"role": "user", "content": "x"}], stream=False)
    _, kwargs = mock.call_args
    # OpenAI-compatible endpoints (LM Studio, vLLM, OpenRouter, etc.) need the
    # explicit "openai/" prefix; LiteLLM does NOT infer the provider from api_base.
    assert kwargs["model"] == "openai/qwen2.5-3b-instruct"
    assert kwargs["api_base"] == "http://localhost:1234/v1"
    assert kwargs["api_key"] == "lm-studio"


async def test_passes_through_tools(cloud_backend, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    mock = AsyncMock(return_value={"choices": []})
    tools = [{"type": "function", "function": {"name": "search"}}]
    with patch("saint.backends.litellm.acompletion", mock):
        await call_backend(
            cloud_backend, messages=[{"role": "user", "content": "x"}],
            stream=False, tools=tools, tool_choice="auto",
        )
    kwargs = mock.call_args.kwargs
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"


# --- volatile sentinel (per-turn context kept out of the cache prefix) -----------
from saint.backends import append_volatile, inject_cache_control, split_volatile

SENT = "<<<saint:volatile>>>"


def test_split_volatile_extracts_tail_and_strips_marker():
    msgs = [
        {"role": "system", "content": f"Today is 2026-08-19.\n{SENT}\nCurrent time: 8:20 PM."},
        {"role": "user", "content": "hi"},
    ]
    out, vol = split_volatile(msgs, SENT)
    assert vol == "Current time: 8:20 PM."
    assert out[0]["content"] == "Today is 2026-08-19."          # head kept, marker gone
    assert SENT not in out[0]["content"]
    assert msgs[0]["content"].count(SENT) == 1                   # input untouched (copy-on-write)


def test_split_volatile_absent_or_disabled():
    msgs = [{"role": "system", "content": "stable"}, {"role": "user", "content": "hi"}]
    assert split_volatile(msgs, SENT) == (msgs, None)
    assert split_volatile([{"role": "system", "content": f"x{SENT}y"}], "") == (
        [{"role": "system", "content": f"x{SENT}y"}], None)   # empty sentinel disables


def test_split_volatile_whole_system_is_volatile_drops_message():
    msgs = [{"role": "system", "content": f"{SENT}\nlive"}, {"role": "user", "content": "hi"}]
    out, vol = split_volatile(msgs, SENT)
    assert vol == "live"
    assert all(m["role"] != "system" for m in out)              # empty head → system dropped


def test_append_volatile_lands_after_cache_control_breakpoint():
    # emulate the dispatch order: split → inject_cache_control → append
    msgs = [
        {"role": "system", "content": f"stable prefix\n{SENT}\nlive note"},
        {"role": "user", "content": "x" * 5000},               # over min_chars → caching engages
    ]
    stable, vol = split_volatile(msgs, SENT)
    cached, _ = inject_cache_control(stable, None, min_chars=4000)
    final = append_volatile(cached, vol)
    last = final[-1]["content"]                                 # last user message content (blocks)
    assert isinstance(last, str) is False and last[-1] == {"type": "text", "text": "live note"}
    # the volatile block is last and carries NO cache_control; the breakpoint sits before it
    assert "cache_control" not in last[-1]
    assert any("cache_control" in b for b in last[:-1])


def test_append_volatile_creates_user_turn_when_none():
    out = append_volatile([{"role": "system", "content": "s"}], "note")
    assert out[-1] == {"role": "user", "content": [{"type": "text", "text": "note"}]}


def test_shape_request_strips_temperature_for_claude5():
    from saint.backends import _shape_request
    from saint.config import BackendConfig
    def mk(name, provider, model, base_url=None):
        return BackendConfig(name=name, provider=provider, model=model, api_key_env=None, api_key="k",
                             base_url=base_url, aliases=(), timeout_s=30)
    b = mk("cloud-medium", "anthropic", "claude-sonnet-5")
    kw = _shape_request(b, {"model": "claude-sonnet-5", "temperature": 0.2, "max_tokens": 10})
    assert "temperature" not in kw
    # Claude 4.x keeps it
    b4 = mk("cloud-small", "anthropic", "claude-haiku-4-5-20251001")
    kw4 = _shape_request(b4, {"model": "claude-haiku-4-5-20251001", "temperature": 0.2, "max_tokens": 10})
    assert kw4["temperature"] == 0.2
    # local seats untouched
    bl = mk("local-chat", "openai", "x", base_url="http://localhost:8002/v1")
    assert _shape_request(bl, {"model": "x", "temperature": 0.2})["temperature"] == 0.2
