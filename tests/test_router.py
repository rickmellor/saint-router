import json
from unittest.mock import AsyncMock, patch

import pytest

from saint.config import (
    BackendConfig,
    ClassifierConfig,
    Config,
    LoggingConfig,
    RoutingConfig,
    ServerConfig,
)
from saint.router import decide_route


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
    with patch("saint.classifier.call_backend", AsyncMock(return_value=response)):
        decision = await decide_route(
            cfg=cfg, model_field="saint-explain",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert decision.mode == "explain"
    # general/trivial under default normal urgency → policy.normal["general,trivial"] = local-small
    assert decision.backend == "local-small"
    assert decision.classifier_result is not None


async def test_decide_pinned_by_model_field():
    cfg = _cfg()
    decision = await decide_route(
        cfg=cfg, model_field="saint-cloud-large",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert decision.mode == "dispatch"
    assert decision.backend == "cloud-large"
    assert decision.classifier_result is None
    assert decision.pinned_backend == "cloud-large"


async def test_decide_pinned_by_prefix_overrides_model_field():
    cfg = _cfg()
    decision = await decide_route(
        cfg=cfg, model_field="saint-cloud-large",
        messages=[{"role": "user", "content": "!local-small foo"}],
    )
    assert decision.backend == "local-small"
    assert decision.stripped_last_user == "foo"


async def test_decide_auto_runs_classifier():
    cfg = _cfg()
    payload = json.dumps({"domain": "code", "complexity": "hard", "reason": "novel refactor"})
    response = {"choices": [{"message": {"content": payload}}]}
    with patch("saint.classifier.call_backend", AsyncMock(return_value=response)):
        decision = await decide_route(
            cfg=cfg, model_field="saint-auto",
            messages=[{"role": "user", "content": "rewrite this thing"}],
        )
    assert decision.backend == "cloud-large"  # policy.normal["code,hard"]
    assert decision.classifier_result is not None
    assert decision.classifier_result.domain == "code"


async def test_decide_ignore_after_cuts_classifier_input_not_dispatch():
    from dataclasses import replace
    cfg = _cfg()
    cfg = replace(cfg, classifier=replace(cfg.classifier,
                                          ignore_after=("\n\n[fabric]", "<memory-context>")))
    payload = json.dumps({"domain": "general", "complexity": "trivial", "reason": "greeting"})
    response = {"choices": [{"message": {"content": payload}}]}
    injected = "Say hi to the team.\n\n[fabric] relevant to your request:\n  [ts] agent: Verilog modules"
    mock = AsyncMock(return_value=response)
    with patch("saint.classifier.call_backend", mock):
        decision = await decide_route(
            cfg=cfg, model_field="saint-auto",
            messages=[{"role": "user", "content": injected}],
        )
    # classifier saw only the user's request
    assert decision.classifier_outcome.input_chars == len("Say hi to the team.")
    sent = mock.call_args.kwargs["messages"][-1]["content"]
    assert "[fabric]" not in sent
    # the logged prompt matches what was classified; dispatch text keeps the full message
    assert decision.last_user_content_original == "Say hi to the team."
    assert "[fabric]" in decision.stripped_last_user


async def test_decide_ignore_after_injection_only_message_routes_general_trivial():
    from dataclasses import replace
    cfg = _cfg()
    cfg = replace(cfg, classifier=replace(cfg.classifier, ignore_after=("<memory-context>",)))
    decision = await decide_route(
        cfg=cfg, model_field="saint-auto",
        messages=[{"role": "user", "content": "<memory-context>\nrecalled stuff"}],
    )
    assert decision.backend == "local-small"  # policy general,trivial — no classifier call
    assert decision.classifier_result is None


async def test_decide_urgency_prefix_changes_policy():
    cfg = _cfg()
    payload = json.dumps({"domain": "general", "complexity": "medium", "reason": "."})
    response = {"choices": [{"message": {"content": payload}}]}
    with patch("saint.classifier.call_backend", AsyncMock(return_value=response)):
        decision = await decide_route(
            cfg=cfg, model_field="saint-auto",
            messages=[{"role": "user", "content": "!urgent help"}],
        )
    # policy.urgent["general,medium"] = cloud-small
    assert decision.backend == "cloud-small"
    assert decision.urgency == "urgent"


async def test_decide_classifier_failure_uses_default_on_failure():
    cfg = _cfg()
    with patch("saint.classifier.call_backend",
               AsyncMock(side_effect=RuntimeError("down"))):
        decision = await decide_route(
            cfg=cfg, model_field="saint-auto",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert decision.backend == "cloud-large"  # default_on_failure
    assert decision.classifier_result is None


async def test_decide_unknown_prefix_raises():
    from saint.prefixes import UnknownPrefixError
    cfg = _cfg()
    with pytest.raises(UnknownPrefixError):
        await decide_route(
            cfg=cfg, model_field="saint-auto",
            messages=[{"role": "user", "content": "!doesnotexist hi"}],
        )


async def test_decide_multimodal_routes_to_default_on_failure():
    cfg = _cfg()
    decision = await decide_route(
        cfg=cfg, model_field="saint-auto",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]}],
    )
    assert decision.backend == "cloud-large"  # default_on_failure
    assert decision.multimodal is True


async def test_decide_empty_messages_defaults_general_trivial():
    cfg = _cfg()
    decision = await decide_route(cfg=cfg, model_field="saint-auto", messages=[])
    # policy.normal["general,trivial"] = local-small
    assert decision.backend == "local-small"
    assert decision.classifier_result is None


from saint.router import apply_stripping


def test_apply_stripping_replaces_last_user_content():
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "!urgent please"},
    ]
    out = apply_stripping(messages, stripped_last_user="please")
    assert out is not messages
    assert out[-1]["content"] == "please"
    assert out[-3]["content"] == "earlier"


def test_apply_stripping_no_user_message_unchanged():
    messages = [{"role": "system", "content": "x"}]
    out = apply_stripping(messages, stripped_last_user=None)
    assert out == messages


# --- routing caches (turn cache + conversation affinity) ---
def _caches():
    from saint.route_cache import RouteCaches, TTLCache
    return RouteCaches(turns=TTLCache(300, 64), conversations=TTLCache(900, 64))


def _payload(domain, complexity):
    p = json.dumps({"domain": domain, "complexity": complexity, "reason": "r"})
    return {"choices": [{"message": {"content": p}}]}


async def test_turn_cache_classifies_once():
    cfg = _cfg()
    caches = _caches()
    prompt = "refactor the widget parser so it streams tokens instead of buffering"
    msgs = [{"role": "system", "content": "agent"}, {"role": "user", "content": prompt}]
    mock = AsyncMock(return_value=_payload("code", "medium"))
    with patch("saint.classifier.call_backend", mock):
        d1 = await decide_route(cfg=cfg, model_field="saint-auto", messages=msgs, caches=caches)
        d2 = await decide_route(cfg=cfg, model_field="saint-auto",
                                messages=msgs + [{"role": "assistant", "content": "ok"},
                                                 {"role": "user", "content": prompt}],
                                caches=caches)
    assert mock.call_count == 1
    assert d1.backend == d2.backend == "local-coder"
    assert d2.classifier_outcome.classifier_used == "cache"
    assert d2.classifier_result.latency_ms == 0


async def test_turn_cache_kills_flap():
    cfg = _cfg()
    caches = _caches()
    msgs = [{"role": "user", "content": "review the code and tell me more please"}]
    mock = AsyncMock(side_effect=[_payload("code", "medium"), _payload("code", "hard")])
    with patch("saint.classifier.call_backend", mock):
        d1 = await decide_route(cfg=cfg, model_field="saint-auto", messages=msgs, caches=caches)
        d2 = await decide_route(cfg=cfg, model_field="saint-auto", messages=msgs, caches=caches)
    # without the cache, turn 2 would have re-rolled to code,hard -> cloud-large
    assert d1.backend == d2.backend == "local-coder"


async def test_caches_bypassed_for_pins_and_explain():
    cfg = _cfg()
    caches = _caches()
    mock = AsyncMock(return_value=_payload("general", "trivial"))
    with patch("saint.classifier.call_backend", mock):
        # pinned via model field: no classification, nothing cached
        await decide_route(cfg=cfg, model_field="saint-cloud-large",
                           messages=[{"role": "user", "content": "hello there friend"}],
                           caches=caches)
        assert len(caches.turns) == 0 and len(caches.conversations) == 0
        # explain classifies but never caches
        await decide_route(cfg=cfg, model_field="saint-explain",
                           messages=[{"role": "user", "content": "hello there friend"}],
                           caches=caches)
    assert len(caches.turns) == 0 and len(caches.conversations) == 0


async def test_classifier_failure_not_cached():
    cfg = _cfg()
    caches = _caches()
    msgs = [{"role": "user", "content": "some prompt that fails to classify"}]
    ok = _payload("code", "trivial")
    mock = AsyncMock(side_effect=[RuntimeError("boom"), ok])
    with patch("saint.classifier.call_backend", mock):
        d1 = await decide_route(cfg=cfg, model_field="saint-auto", messages=msgs, caches=caches)
        d2 = await decide_route(cfg=cfg, model_field="saint-auto", messages=msgs, caches=caches)
    assert d1.backend == "cloud-large"      # default_on_failure
    assert d2.backend == "local-coder"      # re-classified, not served from cache
    assert mock.call_count == 2


async def test_short_follow_up_inherits_conversation_labels():
    cfg = _cfg()
    caches = _caches()
    first = [{"role": "system", "content": "agent"},
             {"role": "user", "content": "explore the saint-router repo and refactor decide_route into pure helpers"}]
    follow = first + [{"role": "assistant", "content": "Which repo did you mean?"},
                      {"role": "user", "content": "yes, sorry. saint-router"}]
    mock = AsyncMock(return_value=_payload("code", "hard"))
    with patch("saint.classifier.call_backend", mock):
        d1 = await decide_route(cfg=cfg, model_field="saint-auto", messages=first, caches=caches)
        d2 = await decide_route(cfg=cfg, model_field="saint-auto", messages=follow, caches=caches)
    assert mock.call_count == 1
    assert d1.backend == d2.backend == "cloud-large"   # policy.normal[code,hard]
    assert d2.classifier_outcome.classifier_used == "inherited"


async def test_long_new_message_does_not_inherit():
    cfg = _cfg()
    caches = _caches()
    first = [{"role": "user", "content": "explore the saint-router repo and refactor decide_route"}]
    follow = first + [{"role": "assistant", "content": "done"},
                      {"role": "user", "content": "now write me a limerick about routers and their little classifier heads"}]
    mock = AsyncMock(side_effect=[_payload("code", "hard"), _payload("general", "trivial")])
    with patch("saint.classifier.call_backend", mock):
        await decide_route(cfg=cfg, model_field="saint-auto", messages=first, caches=caches)
        d2 = await decide_route(cfg=cfg, model_field="saint-auto", messages=follow, caches=caches)
    assert mock.call_count == 2
    assert d2.backend == "local-small"


async def test_sticky_conversations_inherits_regardless_of_length():
    from dataclasses import replace
    cfg = _cfg()
    cfg = replace(cfg, cache=replace(cfg.cache, sticky_conversations=True))
    caches = _caches()
    first = [{"role": "user", "content": "explore the saint-router repo and refactor decide_route"}]
    follow = first + [{"role": "assistant", "content": "done"},
                      {"role": "user", "content": "now write me a limerick about routers and their little classifier heads"}]
    mock = AsyncMock(return_value=_payload("code", "hard"))
    with patch("saint.classifier.call_backend", mock):
        await decide_route(cfg=cfg, model_field="saint-auto", messages=first, caches=caches)
        d2 = await decide_route(cfg=cfg, model_field="saint-auto", messages=follow, caches=caches)
    assert mock.call_count == 1
    assert d2.classifier_outcome.classifier_used == "inherited"


async def test_inherited_labels_respect_live_urgency_prefix():
    cfg = _cfg()
    caches = _caches()
    first = [{"role": "user", "content": "explore the saint-router repo and refactor decide_route"}]
    follow = first + [{"role": "assistant", "content": "done"},
                      {"role": "user", "content": "!urgent yes do it"}]
    mock = AsyncMock(return_value=_payload("general", "medium"))
    with patch("saint.classifier.call_backend", mock):
        d1 = await decide_route(cfg=cfg, model_field="saint-auto", messages=first, caches=caches)
        d2 = await decide_route(cfg=cfg, model_field="saint-auto", messages=follow, caches=caches)
    assert d1.backend == "local-small"   # policy.normal[general,medium]
    assert d2.classifier_outcome.classifier_used == "inherited"
    assert d2.backend == "cloud-small"   # policy.URGENT[general,medium] — urgency stays live


async def test_injection_only_message_inherits_instead_of_general_trivial():
    from dataclasses import replace
    cfg = _cfg()
    cfg = replace(cfg, classifier=replace(cfg.classifier, ignore_after=("<memory-context>",)))
    caches = _caches()
    first = [{"role": "user", "content": "explore the saint-router repo and refactor decide_route"}]
    follow = first + [{"role": "assistant", "content": "done"},
                      {"role": "user", "content": "<memory-context>\nrecalled stuff only"}]
    mock = AsyncMock(return_value=_payload("code", "hard"))
    with patch("saint.classifier.call_backend", mock):
        await decide_route(cfg=cfg, model_field="saint-auto", messages=first, caches=caches)
        d2 = await decide_route(cfg=cfg, model_field="saint-auto", messages=follow, caches=caches)
    assert d2.classifier_outcome.classifier_used == "inherited"
    assert d2.backend == "cloud-large"


async def test_no_caches_arg_behaves_as_before():
    cfg = _cfg()
    msgs = [{"role": "user", "content": "refactor the parser"}]
    mock = AsyncMock(return_value=_payload("code", "medium"))
    with patch("saint.classifier.call_backend", mock):
        await decide_route(cfg=cfg, model_field="saint-auto", messages=msgs)
        await decide_route(cfg=cfg, model_field="saint-auto", messages=msgs)
    assert mock.call_count == 2


async def test_multimodal_backend_knob():
    from dataclasses import replace
    cfg = _cfg()
    mm_msg = [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:..."}}]}]
    d1 = await decide_route(cfg=cfg, model_field="saint-auto", messages=mm_msg)
    assert d1.backend == "cloud-large"  # default_on_failure when knob unset
    cfg2 = replace(cfg, routing=replace(cfg.routing, multimodal_backend="local-coder"))
    d2 = await decide_route(cfg=cfg2, model_field="saint-auto", messages=mm_msg)
    assert d2.backend == "local-coder"
