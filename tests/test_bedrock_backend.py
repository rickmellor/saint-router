from unittest.mock import AsyncMock, patch

from saint.backends import _resolve_model_id, call_backend
from saint.config import BackendConfig


def _bedrock(**over) -> BackendConfig:
    base = dict(
        name="bedrock-sonnet", provider="bedrock",
        model="global.anthropic.claude-sonnet-5",
        api_key_env=None, api_key=None, base_url=None, aliases=(), timeout_s=120,
        aws_region="us-east-1", aws_profile="ClaudeCode",
    )
    base.update(over)
    return BackendConfig(**base)


def test_resolve_model_id_bedrock():
    assert _resolve_model_id(_bedrock()) == "bedrock/global.anthropic.claude-sonnet-5"


async def test_call_backend_bedrock_kwargs():
    mock = AsyncMock(return_value={"ok": True})
    with patch("saint.backends.litellm.acompletion", mock):
        await call_backend(_bedrock(), messages=[{"role": "user", "content": "hi"}],
                           stream=False)
    kw = mock.call_args.kwargs
    assert kw["model"] == "bedrock/global.anthropic.claude-sonnet-5"
    assert kw["aws_region_name"] == "us-east-1"
    assert kw["aws_profile_name"] == "ClaudeCode"
    assert "api_key" not in kw and "api_base" not in kw


async def test_call_backend_bedrock_no_profile_uses_default_chain():
    mock = AsyncMock(return_value={})
    with patch("saint.backends.litellm.acompletion", mock):
        await call_backend(_bedrock(aws_profile=None),
                           messages=[{"role": "user", "content": "hi"}], stream=False)
    assert "aws_profile_name" not in mock.call_args.kwargs


async def test_drop_params_and_default_max_tokens():
    b = _bedrock(drop_params=("temperature",), default_max_tokens=8192)
    mock = AsyncMock(return_value={})
    with patch("saint.backends.litellm.acompletion", mock):
        await call_backend(b, messages=[{"role": "user", "content": "hi"}], stream=False,
                           extra_params={"temperature": 0.7, "top_p": 0.9})
    kw = mock.call_args.kwargs
    assert "temperature" not in kw          # dropped
    assert kw["top_p"] == 0.9               # untouched
    assert kw["max_tokens"] == 8192         # injected (client omitted)
    # client-specified max_tokens wins over the default
    mock2 = AsyncMock(return_value={})
    with patch("saint.backends.litellm.acompletion", mock2):
        await call_backend(b, messages=[{"role": "user", "content": "hi"}], stream=False,
                           extra_params={"max_tokens": 100})
    assert mock2.call_args.kwargs["max_tokens"] == 100


async def test_openai_backend_kwargs_unchanged():
    b = BackendConfig(name="local", provider="openai", model="m", api_key_env=None,
                      api_key="local", base_url="http://x/v1", aliases=(), timeout_s=60)
    mock = AsyncMock(return_value={})
    with patch("saint.backends.litellm.acompletion", mock):
        await call_backend(b, messages=[{"role": "user", "content": "hi"}], stream=False)
    kw = mock.call_args.kwargs
    assert kw["api_key"] == "local" and kw["api_base"] == "http://x/v1"
    assert "aws_region_name" not in kw


def test_cache_control_gate_includes_bedrock():
    from dataclasses import replace
    from saint.router import _prepare_dispatch, RoutingDecision  # noqa: F401
    from tests.test_router import _cfg
    cfg = _cfg()
    big = "review this carefully " * 400
    msgs = [{"role": "system", "content": big}, {"role": "user", "content": "go"}]

    class _D:  # minimal decision stub for _prepare_dispatch
        stripped_last_user = "go"
        backend = "cloud-large"
        prev_backend = "cloud-large"  # same backend → no thinking strip

    _, out, _ = _prepare_dispatch(cfg, _D(), msgs, None, _bedrock())
    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_effective_prices_bedrock_derivation():
    from saint.cli import _effective_prices
    import pytest
    p = _effective_prices(_bedrock(price_in=3.0, price_out=15.0))
    assert p["cache_read"] == pytest.approx(0.3) and p["cache_write"] == pytest.approx(3.75)


def test_cache_tokens_converse_field_names():
    from saint.server import _cache_tokens
    read, write = _cache_tokens({"cacheReadInputTokens": 500, "cacheWriteInputTokens": 60})
    assert (read, write) == (500, 60)
