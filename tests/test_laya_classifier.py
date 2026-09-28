"""mode="laya": the trained question text is loaded from the checkpoint's saint-training.json and
sent verbatim, no bearer header, the gate reads answer_confidence, and the router chain is
laya → LLM labeller (or → embedding head when laya_fallthrough = "embedding"). No network —
httpx.MockTransport and patched classify()."""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from saint import jev_classifier as JC
from saint import laya_classifier as LC
from saint.classifier import ClassifierResult
from saint.router import decide_route
from saint.storage import LogRow, fetch_training_rows, log_request, open_db
from tests.test_router import _cfg

QUESTIONS = {
    "complexity": {"type": "choice", "instructions": "Who should do this work?",
                   "criteria": {"trivial": "greeting", "medium": "manual work", "hard": "thinking work"}},
    "domain": {"type": "choice", "instructions": "What kind of work?",
               "criteria": {"code": "source code", "general": "everything else"}},
}


@pytest.fixture
def meta_path(tmp_path):
    p = tmp_path / "saint-training.json"
    p.write_text(json.dumps({"question": QUESTIONS, "labels": ["trivial", "medium", "hard"],
                             "trained_on": 2584, "date": "2026-09-26", "base": "convaiinnovations/laya english"}))
    return p


def _laya_cfg(meta_path, **over):
    cfg = _cfg()
    return replace(cfg, classifier=replace(cfg.classifier, mode="laya", laya_questions_path=str(meta_path),
                                           laya_url="http://laya.test", min_confidence=0.8, **over))


def _answer(domain="code", dconf=0.98, complexity="medium", cconf=0.97, entropy_conf=0.5):
    """laya-serve's shape: `confidence` is entropy-based, `answer_confidence` is the max probability."""
    return {"model": "laya-rl-agent", "answers": {
        "domain": {"type": "choice", "choice": domain, "confidence": entropy_conf, "answer_confidence": dconf,
                   "probabilities": {domain: dconf}},
        "complexity": {"type": "choice", "choice": complexity, "confidence": entropy_conf, "answer_confidence": cconf,
                       "probabilities": {complexity: cconf}},
    }, "usage": {"input_tokens": 170, "output_tokens": 0}}


def test_training_meta_is_loaded_validated_and_cached(meta_path):
    meta = LC.load_training_meta(meta_path)
    assert meta["questions"] == QUESTIONS and meta["trained_on"] == 2584 and meta["trained_at"] == "2026-09-26"
    assert LC.load_training_meta(meta_path) is meta                       # mtime cache
    cfg = _laya_cfg(meta_path)
    assert LC.trained_at(cfg.classifier) == "2026-09-26T00:00:00"
    s = LC.settings_from(cfg.classifier)
    assert s.api_key_env is None and s.model == "english" and s.base_url == "http://laya.test"
    assert s.questions == QUESTIONS and s.classifier_used == "laya"
    body = JC.build_request("hi", s)
    assert body["questions"] == QUESTIONS and body["model"] == "english"


def test_complexity_only_checkpoint_is_rejected(tmp_path):
    p = tmp_path / "saint-training.json"
    p.write_text(json.dumps({"question": {"complexity": QUESTIONS["complexity"]}}))
    with pytest.raises(ValueError, match="domain"):
        LC.load_training_meta(p)


def test_gate_reads_answer_confidence_not_entropy_confidence():
    # entropy `confidence` 0.5 would fail a 0.8 gate; the max-probability answer_confidence passes it
    r = JC.parse_response(_answer(), min_confidence=0.8, latency_ms=21, classifier_used="laya")
    assert isinstance(r, ClassifierResult) and (r.domain, r.complexity) == ("code", "medium")
    assert r.reason.startswith("laya ") and "0.98" in r.reason and "0.97" in r.reason
    assert JC.parse_response(_answer(cconf=0.79), min_confidence=0.8, latency_ms=1) is None
    # a server without answer_confidence still works off `confidence`
    old = _answer(); [a.pop("answer_confidence") for a in old["answers"].values()]
    assert JC.parse_response(old, min_confidence=0.4, latency_ms=1) is not None
    assert JC.parse_response(old, min_confidence=0.6, latency_ms=1) is None


async def test_classify_sends_trained_questions_without_bearer(meta_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["body"] = str(req.url), req.headers.get("authorization"), json.loads(req.content)
        return httpx.Response(200, json=_answer(domain="general", complexity="hard"))

    cfg = _laya_cfg(meta_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        r = await LC.classify("design a sharding scheme", classifier_cfg=cfg.classifier, min_confidence=0.8, client=c)
        d, x, conf, ms = await LC.predict("design a sharding scheme", classifier_cfg=cfg.classifier, client=c)
    assert (r.domain, r.complexity) == ("general", "hard")
    assert seen["url"] == "http://laya.test/v1/systemone" and seen["auth"] is None
    assert seen["body"]["questions"] == QUESTIONS and seen["body"]["state"] == "design a sharding scheme"
    assert (d, x, conf) == ("general", "hard", 0.97)


async def test_classify_failures_raise_jev_error(meta_path):
    cfg = _laya_cfg(meta_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503, text="loading"))) as c:
        with pytest.raises(JC.JevError, match="HTTP 503"):
            await LC.classify("x", classifier_cfg=cfg.classifier, min_confidence=0.8, client=c)


# --- router chain --------------------------------------------------------------------------

async def _route(cfg, text="refactor the scheduler"):
    return await decide_route(cfg=cfg, model_field="saint-auto", messages=[{"role": "user", "content": text}])


async def test_router_uses_laya_when_confident(meta_path):
    ok = ClassifierResult(domain="code", complexity="hard", reason="laya", latency_ms=22)
    with patch("saint.laya_classifier.classify", AsyncMock(return_value=ok)), \
         patch("saint.classifier.call_backend", AsyncMock(side_effect=AssertionError("LLM must not be called"))):
        d = await _route(_laya_cfg(meta_path))
    assert d.backend == "cloud-large" and d.classifier_outcome.classifier_used == "laya"
    assert d.classifier_outcome.fallback_reason is None


async def test_router_falls_to_llm_when_laya_defers(meta_path):
    llm = {"choices": [{"message": {"content": json.dumps({"domain": "general", "complexity": "trivial", "reason": "hi"})}}]}
    with patch("saint.laya_classifier.classify", AsyncMock(return_value=None)), \
         patch("saint.classifier.call_backend", AsyncMock(return_value=llm)):
        d = await _route(_laya_cfg(meta_path))
    assert d.classifier_result.complexity == "trivial"
    assert d.classifier_outcome.fallback_reason == "laya_defer" and d.classifier_outcome.classifier_used != "laya"


async def test_router_survives_laya_outage(meta_path):
    llm = {"choices": [{"message": {"content": json.dumps({"domain": "code", "complexity": "medium", "reason": "fn"})}}]}
    with patch("saint.laya_classifier.classify", AsyncMock(side_effect=JC.JevError("ConnectError"))), \
         patch("saint.classifier.call_backend", AsyncMock(return_value=llm)):
        d = await _route(_laya_cfg(meta_path))
    assert d.classifier_result.domain == "code" and d.classifier_outcome.fallback_reason == "laya_error"


async def test_router_can_fall_through_to_the_embedding_head(meta_path):
    """laya_fallthrough='embedding' with no head trained records both hops before the LLM answers."""
    llm = {"choices": [{"message": {"content": json.dumps({"domain": "code", "complexity": "medium", "reason": "fn"})}}]}
    cfg = _laya_cfg(meta_path, laya_fallthrough="embedding", embedding_backend="local-small", head_path=str(meta_path.parent / "none.npz"))
    with patch("saint.laya_classifier.classify", AsyncMock(return_value=None)), \
         patch("saint.classifier.call_backend", AsyncMock(return_value=llm)):
        d = await _route(cfg)
    assert d.classifier_outcome.fallback_reason == "laya_defer+embedding_untrained"


def test_laya_rows_never_train_the_head(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    base = dict(request_id="r", model_field="saint-auto", prefixes_raw=None, pinned_backend=None, urgency_used="normal",
                classifier_fallback_reason=None, classifier_input_chars=10, classifier_input_truncated_from=None,
                classifier_latency_ms=20, classifier_domain="code", classifier_complexity="medium", classifier_reason="r",
                backend_chosen="local-coder", backend_latency_ms=100, tokens_in=1, tokens_out=1, success=True, error_kind=None,
                prompt_storage_mode="full")
    for i, used in enumerate(["laya", "typesafe-jev", "local-embed (embed-head)", "cache", "local-chat"]):
        log_request(conn, LogRow(**{**base, "request_id": f"r-{i}", "classifier_used": used, "prompt_content": f"prompt {i}"}))
    rows = list(fetch_training_rows(conn, 100))
    assert [r[0] for r in rows] == ["prompt 4"]
