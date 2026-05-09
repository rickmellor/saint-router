import json
from unittest.mock import AsyncMock, patch

import pytest

from goorouter.classifier import (
    ClassifierError,
    classify,
    classify_with_fallback,
    load_prompt_template,
)
from goorouter.config import BackendConfig


def _backend() -> BackendConfig:
    return BackendConfig(
        name="local-small", provider="openai", model="qwen2.5-3b-instruct",
        api_key_env=None, api_key="lm-studio",
        base_url="http://localhost:1234/v1", aliases=(), timeout_s=5,
    )


def _response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


async def test_classify_parses_json():
    payload = json.dumps({"domain": "code", "complexity": "medium", "reason": "ok"})
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=_response(payload))):
        result = await classify(_backend(), prompt="refactor X", template=load_prompt_template(None))
    assert result.domain == "code"
    assert result.complexity == "medium"
    assert result.reason == "ok"
    assert result.latency_ms >= 0


async def test_classify_strips_code_fences():
    payload = '```json\n{"domain":"general","complexity":"trivial","reason":"hi"}\n```'
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=_response(payload))):
        result = await classify(_backend(), prompt="hi", template=load_prompt_template(None))
    assert result.domain == "general"
    assert result.complexity == "trivial"


async def test_classify_invalid_json_raises():
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=_response("not json"))):
        with pytest.raises(ClassifierError):
            await classify(_backend(), prompt="x", template=load_prompt_template(None))


async def test_classify_invalid_enum_raises():
    payload = json.dumps({"domain": "wat", "complexity": "medium", "reason": "."})
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=_response(payload))):
        with pytest.raises(ClassifierError):
            await classify(_backend(), prompt="x", template=load_prompt_template(None))


def _other_backend() -> BackendConfig:
    return BackendConfig(
        name="local-large", provider="openai", model="qwen2.5-32b-instruct",
        api_key_env=None, api_key="lm-studio",
        base_url="http://localhost:1234/v1", aliases=(), timeout_s=180,
    )


async def test_oversize_skips_primary_uses_fallback():
    primary_mock = AsyncMock()  # should not be called
    payload = json.dumps({"domain": "code", "complexity": "medium", "reason": "ok"})
    fb_mock = AsyncMock(return_value=_response(payload))

    async def fake_call(b, **kw):
        if b.name == "local-small":
            return await primary_mock(b, **kw)
        return await fb_mock(b, **kw)

    with patch("goorouter.classifier.call_backend", side_effect=fake_call):
        outcome = await classify_with_fallback(
            primary=_backend(), fallback=_other_backend(),
            prompt="x" * 12000, max_input_chars=8000,
            template=load_prompt_template(None),
        )
    assert outcome.result is not None
    assert outcome.fallback_reason == "oversize"
    assert outcome.classifier_used == "local-large"
    assert outcome.input_chars == 12000
    assert outcome.input_truncated_from is None
    primary_mock.assert_not_called()
    fb_mock.assert_called_once()


async def test_primary_error_triggers_fallback():
    payload = json.dumps({"domain": "general", "complexity": "trivial", "reason": "."})

    async def fake_call(b, **kw):
        if b.name == "local-small":
            raise RuntimeError("boom")
        return _response(payload)

    with patch("goorouter.classifier.call_backend", side_effect=fake_call):
        outcome = await classify_with_fallback(
            primary=_backend(), fallback=_other_backend(),
            prompt="hi", max_input_chars=8000,
            template=load_prompt_template(None),
        )
    assert outcome.result is not None
    assert outcome.fallback_reason == "primary_error"
    assert outcome.classifier_used == "local-large"


async def test_no_fallback_oversize_truncates():
    payload = json.dumps({"domain": "code", "complexity": "medium", "reason": "."})
    seen_lengths: list[int] = []

    async def fake_call(b, *, messages, **kw):
        seen_lengths.append(len(messages[0]["content"]))
        return _response(payload)

    with patch("goorouter.classifier.call_backend", side_effect=fake_call):
        outcome = await classify_with_fallback(
            primary=_backend(), fallback=None,
            prompt="x" * 12000, max_input_chars=8000,
            template=load_prompt_template(None),
        )
    assert outcome.fallback_reason is None
    assert outcome.input_chars == 8000
    assert outcome.input_truncated_from == 12000
    assert seen_lengths[0] > 0


async def test_both_fail_returns_none_result():
    async def fake_call(b, **kw):
        raise RuntimeError("everything down")

    with patch("goorouter.classifier.call_backend", side_effect=fake_call):
        outcome = await classify_with_fallback(
            primary=_backend(), fallback=_other_backend(),
            prompt="x", max_input_chars=8000,
            template=load_prompt_template(None),
        )
    assert outcome.result is None
    assert outcome.fallback_reason == "primary_error"
    assert outcome.classifier_used is None
