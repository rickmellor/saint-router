from unittest.mock import AsyncMock, patch

import pytest

from goorouter.backends import call_backend
from goorouter.config import BackendConfig


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
    with patch("goorouter.backends.litellm.acompletion", mock):
        await call_backend(cloud_backend, messages=[{"role": "user", "content": "hi"}], stream=False)
    args, kwargs = mock.call_args
    # litellm anthropic models get the "anthropic/" prefix
    assert kwargs["model"] == "anthropic/claude-opus-4-7"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["timeout"] == 120
    assert kwargs["stream"] is False


async def test_call_openai_compatible_with_base_url(local_backend):
    mock = AsyncMock(return_value={"choices": []})
    with patch("goorouter.backends.litellm.acompletion", mock):
        await call_backend(local_backend, messages=[{"role": "user", "content": "x"}], stream=False)
    args, kwargs = mock.call_args
    # OpenAI-compatible: just the model name (no provider prefix)
    assert kwargs["model"] == "qwen2.5-3b-instruct"
    assert kwargs["api_base"] == "http://localhost:1234/v1"
    assert kwargs["api_key"] == "lm-studio"


async def test_passes_through_tools(cloud_backend, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    mock = AsyncMock(return_value={"choices": []})
    tools = [{"type": "function", "function": {"name": "search"}}]
    with patch("goorouter.backends.litellm.acompletion", mock):
        await call_backend(
            cloud_backend, messages=[{"role": "user", "content": "x"}],
            stream=False, tools=tools, tool_choice="auto",
        )
    kwargs = mock.call_args.kwargs
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"
