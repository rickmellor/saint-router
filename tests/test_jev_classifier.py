"""mode="jev": request shape, the confidence gate, error handling, and the router's
fall-through chain (jev → embedding head → LLM labeller). No network — httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from saint import jev_classifier as JC
from saint.classifier import ClassifierResult


def _answer(domain="code", dconf=0.93, complexity="medium", cconf=0.81, model="jev-1.13.0"):
    return {"model": model, "answers": {
        "domain": {"type": "choice", "choice": domain, "confidence": dconf, "probabilities": {domain: dconf}},
        "complexity": {"type": "choice", "choice": complexity, "confidence": cconf, "probabilities": {complexity: cconf}},
    }, "usage": {"input_tokens": 120, "output_tokens": 2}}


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_request_asks_both_axes_in_one_call_with_pinned_model():
    body = JC.build_request("fix this stack trace", JC.JevSettings())
    assert body["model"] == "jev-1.13.0" and body["state"] == "fix this stack trace"
    assert set(body["questions"]) == {"domain", "complexity"}
    assert set(body["questions"]["domain"]["criteria"]) == {"code", "general"}
    assert set(body["questions"]["complexity"]["criteria"]) == {"trivial", "medium", "hard"}
    assert all(q["type"] == "choice" for q in body["questions"].values())


def test_parse_applies_min_of_both_confidences():
    r = JC.parse_response(_answer(), min_confidence=0.6, latency_ms=88)
    assert isinstance(r, ClassifierResult) and (r.domain, r.complexity, r.latency_ms) == ("code", "medium", 88)
    assert "0.93" in r.reason and "0.81" in r.reason
    assert JC.parse_response(_answer(cconf=0.41), min_confidence=0.6, latency_ms=1) is None  # weak axis defers
    assert JC.parse_response(_answer(dconf=0.2), min_confidence=0.6, latency_ms=1) is None


@pytest.mark.parametrize("bad", [
    {}, {"answers": "x"}, _answer(domain="poetry"), _answer(complexity="extreme"),
    {"answers": {"domain": {"choice": "code"}, "complexity": {"choice": "hard", "confidence": 0.9}}},
])
def test_parse_rejects_malformed_answers(bad):
    with pytest.raises(JC.JevError):
        JC.parse_response(bad, min_confidence=0.6, latency_ms=1)


@pytest.mark.asyncio
async def test_classify_posts_bearer_and_returns_result(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["body"] = str(req.url), req.headers.get("authorization"), json.loads(req.content)
        return httpx.Response(200, json=_answer(domain="general", complexity="hard"))

    async with _client(handler) as c:
        r = await JC.classify("design a sharding scheme", settings=JC.JevSettings(), min_confidence=0.6, client=c)
    assert (r.domain, r.complexity) == ("general", "hard")
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone" and seen["auth"] == "Bearer sk-test"
    assert seen["body"]["state"] == "design a sharding scheme"


@pytest.mark.asyncio
async def test_classify_uses_jev_specific_gate_when_set(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    async with _client(lambda req: httpx.Response(200, json=_answer(cconf=0.7))) as c:
        assert await JC.classify("x", settings=JC.JevSettings(min_confidence=0.75), min_confidence=0.6, client=c) is None
        assert await JC.classify("x", settings=JC.JevSettings(), min_confidence=0.6, client=c) is not None


@pytest.mark.asyncio
async def test_classify_failures_raise_jev_error(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JC.JevError, match="not set"):
        await JC.classify("x", settings=JC.JevSettings(), min_confidence=0.6)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    async with _client(lambda req: httpx.Response(429, text="slow down")) as c:
        with pytest.raises(JC.JevError, match="HTTP 429"):
            await JC.classify("x", settings=JC.JevSettings(), min_confidence=0.6, client=c)

    def boom(req):
        raise httpx.ConnectTimeout("timed out")

    async with _client(boom) as c:
        with pytest.raises(JC.JevError, match="ConnectTimeout"):
            await JC.classify("x", settings=JC.JevSettings(), min_confidence=0.6, client=c)
    async with _client(lambda req: httpx.Response(200, text="<html>")) as c:
        with pytest.raises(JC.JevError, match="non-JSON"):
            await JC.classify("x", settings=JC.JevSettings(), min_confidence=0.6, client=c)


# --- router chain: jev → (embedding head) → LLM labeller --------------------------------

from dataclasses import replace  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

from saint.router import decide_route  # noqa: E402
from tests.test_router import _cfg  # noqa: E402


def _jev_cfg():
    cfg = _cfg()
    return replace(cfg, classifier=replace(cfg.classifier, mode="jev"))


async def _route(cfg, text="refactor the scheduler"):
    return await decide_route(cfg=cfg, model_field="saint-auto", messages=[{"role": "user", "content": text}])


async def test_router_uses_jev_when_confident():
    ok = ClassifierResult(domain="code", complexity="hard", reason="jev", latency_ms=90)
    with patch("saint.jev_classifier.classify", AsyncMock(return_value=ok)), \
         patch("saint.classifier.call_backend", AsyncMock(side_effect=AssertionError("LLM must not be called"))):
        d = await _route(_jev_cfg())
    assert d.backend == "cloud-large" and d.classifier_outcome.classifier_used == "typesafe-jev"
    assert d.classifier_outcome.fallback_reason is None


async def test_router_falls_to_llm_when_jev_defers_and_no_embedding_configured():
    llm = {"choices": [{"message": {"content": json.dumps({"domain": "general", "complexity": "trivial", "reason": "hi"})}}]}
    with patch("saint.jev_classifier.classify", AsyncMock(return_value=None)), \
         patch("saint.classifier.call_backend", AsyncMock(return_value=llm)):
        d = await _route(_jev_cfg())
    assert d.classifier_result.complexity == "trivial"
    assert d.classifier_outcome.fallback_reason == "jev_defer"
    assert d.classifier_outcome.classifier_used != "typesafe-jev"


async def test_router_survives_jev_outage():
    llm = {"choices": [{"message": {"content": json.dumps({"domain": "code", "complexity": "medium", "reason": "fn"})}}]}
    with patch("saint.jev_classifier.classify", AsyncMock(side_effect=JC.JevError("HTTP 503"))), \
         patch("saint.classifier.call_backend", AsyncMock(return_value=llm)):
        d = await _route(_jev_cfg())
    assert d.classifier_result.domain == "code" and d.classifier_outcome.fallback_reason == "jev_error"


async def test_existing_modes_are_untouched_by_the_jev_flag():
    llm = {"choices": [{"message": {"content": json.dumps({"domain": "code", "complexity": "hard", "reason": "r"})}}]}
    with patch("saint.jev_classifier.classify", AsyncMock(side_effect=AssertionError("jev must not be called"))), \
         patch("saint.classifier.call_backend", AsyncMock(return_value=llm)):
        d = await _route(_cfg())  # default mode = llm
    assert d.backend == "cloud-large" and d.classifier_outcome.fallback_reason is None
