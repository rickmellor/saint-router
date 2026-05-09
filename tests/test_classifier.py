import json
from unittest.mock import AsyncMock, patch

import pytest

from goorouter.classifier import (
    ClassifierError,
    ClassifierResult,
    classify,
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
