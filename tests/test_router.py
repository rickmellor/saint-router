import json
from unittest.mock import AsyncMock, patch

import pytest

from goorouter.config import (
    BackendConfig, Config, ServerConfig, ClassifierConfig, RoutingConfig, LoggingConfig
)
from goorouter.router import RoutingDecision, decide_route


def _cfg() -> Config:
    backends = {
        name: BackendConfig(
            name=name, provider="openai", model=name,
            api_key_env=None, api_key="lm", base_url="http://x", aliases=(), timeout_s=60,
        )
        for name in ("cloud-large", "cloud-small", "local-large", "local-small", "local-coder")
    }
    backends["cloud-large"] = BackendConfig(
        name="cloud-large", provider="anthropic", model="claude-opus-4-7",
        api_key_env="ANTHROPIC_API_KEY", api_key=None, base_url=None,
        aliases=("opus",), timeout_s=120,
    )
    policy = {
        u: {f"{d},{c}": "local-coder" if d == "code" else "local-small"
            for d in ("code", "general") for c in ("trivial", "medium", "hard")}
        for u in ("normal", "urgent", "patient")
    }
    policy["normal"]["code,hard"] = "cloud-large"
    policy["urgent"]["general,medium"] = "cloud-small"
    return Config(
        server=ServerConfig(host="127.0.0.1", port=4000),
        backends=backends,
        classifier=ClassifierConfig(backend="local-small", fallback_backend=None,
                                     max_input_chars=8000, timeout_s=5,
                                     prompt_template_path=None),
        routing=RoutingConfig(default_urgency="normal", default_on_failure="cloud-large",
                              policy=policy),
        logging=LoggingConfig(db_path=":memory:", prompt_storage="full"),
    )


async def test_decide_explain_mode():
    cfg = _cfg()
    payload = json.dumps({"domain": "general", "complexity": "trivial", "reason": "small"})
    response = {"choices": [{"message": {"content": payload}}]}
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=response)):
        decision = await decide_route(
            cfg=cfg, model_field="goo-explain",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert decision.mode == "explain"
    # general/trivial under default normal urgency → policy.normal["general,trivial"] = local-small
    assert decision.backend == "local-small"
    assert decision.classifier_result is not None


async def test_decide_pinned_by_model_field():
    cfg = _cfg()
    decision = await decide_route(
        cfg=cfg, model_field="goo-cloud-large",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert decision.mode == "dispatch"
    assert decision.backend == "cloud-large"
    assert decision.classifier_result is None
    assert decision.pinned_backend == "cloud-large"


async def test_decide_pinned_by_prefix_overrides_model_field():
    cfg = _cfg()
    decision = await decide_route(
        cfg=cfg, model_field="goo-cloud-large",
        messages=[{"role": "user", "content": "!local-small foo"}],
    )
    assert decision.backend == "local-small"
    assert decision.stripped_last_user == "foo"


async def test_decide_auto_runs_classifier():
    cfg = _cfg()
    payload = json.dumps({"domain": "code", "complexity": "hard", "reason": "novel refactor"})
    response = {"choices": [{"message": {"content": payload}}]}
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=response)):
        decision = await decide_route(
            cfg=cfg, model_field="goo-auto",
            messages=[{"role": "user", "content": "rewrite this thing"}],
        )
    assert decision.backend == "cloud-large"  # policy.normal["code,hard"]
    assert decision.classifier_result is not None
    assert decision.classifier_result.domain == "code"


async def test_decide_urgency_prefix_changes_policy():
    cfg = _cfg()
    payload = json.dumps({"domain": "general", "complexity": "medium", "reason": "."})
    response = {"choices": [{"message": {"content": payload}}]}
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=response)):
        decision = await decide_route(
            cfg=cfg, model_field="goo-auto",
            messages=[{"role": "user", "content": "!urgent help"}],
        )
    # policy.urgent["general,medium"] = cloud-small
    assert decision.backend == "cloud-small"
    assert decision.urgency == "urgent"


async def test_decide_classifier_failure_uses_default_on_failure():
    cfg = _cfg()
    with patch("goorouter.classifier.call_backend",
               AsyncMock(side_effect=RuntimeError("down"))):
        decision = await decide_route(
            cfg=cfg, model_field="goo-auto",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert decision.backend == "cloud-large"  # default_on_failure
    assert decision.classifier_result is None


async def test_decide_unknown_prefix_raises():
    from goorouter.prefixes import UnknownPrefixError
    cfg = _cfg()
    with pytest.raises(UnknownPrefixError):
        await decide_route(
            cfg=cfg, model_field="goo-auto",
            messages=[{"role": "user", "content": "!doesnotexist hi"}],
        )


async def test_decide_multimodal_routes_to_default_on_failure():
    cfg = _cfg()
    decision = await decide_route(
        cfg=cfg, model_field="goo-auto",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]}],
    )
    assert decision.backend == "cloud-large"  # default_on_failure
    assert decision.multimodal is True


async def test_decide_empty_messages_defaults_general_trivial():
    cfg = _cfg()
    decision = await decide_route(cfg=cfg, model_field="goo-auto", messages=[])
    # policy.normal["general,trivial"] = local-small
    assert decision.backend == "local-small"
    assert decision.classifier_result is None
