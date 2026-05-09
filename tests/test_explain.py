import json
from unittest.mock import AsyncMock, patch

from goorouter.explain import format_decision
from goorouter.router import decide_route
from tests.test_router import _cfg


async def test_format_decision_explain():
    cfg = _cfg()
    payload = json.dumps(
        {"domain": "code", "complexity": "medium", "reason": "scoped refactor"}
    )
    response = {"choices": [{"message": {"content": payload}}]}
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=response)):
        decision = await decide_route(
            cfg=cfg,
            model_field="goo-explain",
            messages=[{"role": "user", "content": "!urgent rewrite x"}],
        )
    text = format_decision(decision, cfg)
    assert "Routing decision" in text
    assert "urgent" in text
    assert "code" in text and "medium" in text
    assert "scoped refactor" in text
    assert decision.backend in text


async def test_format_decision_pinned_backend():
    cfg = _cfg()
    decision = await decide_route(
        cfg=cfg,
        model_field="goo-cloud-large",
        messages=[{"role": "user", "content": "hi"}],
    )
    text = format_decision(decision, cfg)
    assert "Pinned backend" in text
    assert "cloud-large" in text


async def test_format_decision_multimodal():
    cfg = _cfg()
    decision = await decide_route(
        cfg=cfg,
        model_field="goo-auto",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ],
    )
    text = format_decision(decision, cfg)
    assert "Multimodal" in text
    assert "default_on_failure" in text
