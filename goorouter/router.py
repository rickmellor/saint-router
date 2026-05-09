from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from goorouter.backends import call_backend
from goorouter.classifier import (
    ClassifierResult,
    FallbackOutcome,
    classify_with_fallback,
    load_prompt_template,
)
from goorouter.config import Config
from goorouter.policy import resolve_policy
from goorouter.prefixes import ParsedPrefixes, parse_prefixes

GOO_AUTO = "goo-auto"
GOO_EXPLAIN = "goo-explain"
GOO_PREFIX = "goo-"


@dataclass(frozen=True)
class RoutingDecision:
    request_id: str
    mode: Literal["dispatch", "explain"]
    backend: str
    pinned_backend: str | None
    urgency: str
    parsed: ParsedPrefixes
    classifier_outcome: FallbackOutcome | None
    classifier_result: ClassifierResult | None
    multimodal: bool
    model_field: str
    last_user_content_original: str | None
    stripped_last_user: str | None


def _last_user_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m
    return None


def _is_multimodal_content(content: Any) -> bool:
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") not in (None, "text"):
                return True
    return False


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(parts)
    return ""


async def decide_route(
    *,
    cfg: Config,
    model_field: str,
    messages: list[dict[str, Any]],
) -> RoutingDecision:
    request_id = str(uuid.uuid4())

    mode: Literal["dispatch", "explain"] = "explain" if model_field == GOO_EXPLAIN else "dispatch"

    last_user = _last_user_message(messages)
    multimodal = bool(last_user and _is_multimodal_content(last_user.get("content")))
    last_text_original = _content_text(last_user.get("content")) if last_user else None

    urgencies = {"urgent", "patient", "normal"}
    backend_alias_map = {b.name: set(b.aliases) for b in cfg.backends.values()}

    parsed = ParsedPrefixes(urgency=None, pinned_backend=None,
                             stripped=last_text_original or "", raw="")
    if last_text_original is not None and not multimodal:
        parsed = parse_prefixes(last_text_original, urgencies, backend_alias_map)

    pinned_via_model: str | None = None
    if model_field.startswith(GOO_PREFIX) and model_field not in (GOO_AUTO, GOO_EXPLAIN):
        candidate = model_field[len(GOO_PREFIX):]
        if candidate in cfg.backends:
            pinned_via_model = candidate

    # prefix pin takes precedence over model-field pin
    pinned = parsed.pinned_backend or pinned_via_model
    urgency = parsed.urgency or cfg.routing.default_urgency

    if multimodal and pinned is None:
        return RoutingDecision(
            request_id=request_id, mode=mode,
            backend=cfg.routing.default_on_failure,
            pinned_backend=None, urgency=urgency, parsed=parsed,
            classifier_outcome=None, classifier_result=None,
            multimodal=True, model_field=model_field,
            last_user_content_original=last_text_original,
            stripped_last_user=None,
        )

    if pinned is not None:
        return RoutingDecision(
            request_id=request_id, mode=mode, backend=pinned,
            pinned_backend=pinned, urgency=urgency, parsed=parsed,
            classifier_outcome=None, classifier_result=None,
            multimodal=multimodal, model_field=model_field,
            last_user_content_original=last_text_original,
            stripped_last_user=parsed.stripped if last_text_original is not None else None,
        )

    if last_user is None or not parsed.stripped.strip():
        backend = resolve_policy(cfg.routing.policy, urgency, "general", "trivial")
        return RoutingDecision(
            request_id=request_id, mode=mode, backend=backend,
            pinned_backend=None, urgency=urgency, parsed=parsed,
            classifier_outcome=None, classifier_result=None,
            multimodal=multimodal, model_field=model_field,
            last_user_content_original=last_text_original,
            stripped_last_user=parsed.stripped if last_text_original is not None else None,
        )

    primary = cfg.backends[cfg.classifier.backend]
    fallback = cfg.backends[cfg.classifier.fallback_backend] if cfg.classifier.fallback_backend else None
    template = load_prompt_template(cfg.classifier.prompt_template_path)
    outcome = await classify_with_fallback(
        primary=primary, fallback=fallback,
        prompt=parsed.stripped,
        max_input_chars=cfg.classifier.max_input_chars,
        template=template,
    )

    if outcome.result is None:
        backend = cfg.routing.default_on_failure
    else:
        backend = resolve_policy(cfg.routing.policy, urgency,
                                  outcome.result.domain, outcome.result.complexity)
    return RoutingDecision(
        request_id=request_id, mode=mode, backend=backend,
        pinned_backend=None, urgency=urgency, parsed=parsed,
        classifier_outcome=outcome, classifier_result=outcome.result,
        multimodal=multimodal, model_field=model_field,
        last_user_content_original=last_text_original,
        stripped_last_user=parsed.stripped,
    )


def apply_stripping(
    messages: list[dict[str, Any]], stripped_last_user: str | None,
) -> list[dict[str, Any]]:
    """Return a new messages list with the last user message's content replaced by `stripped_last_user`."""
    if stripped_last_user is None:
        return list(messages)
    out = [dict(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i]["content"] = stripped_last_user
            break
    return out


async def dispatch_non_streaming(
    cfg: Config, decision: RoutingDecision, messages: list[dict[str, Any]],
    *, tools: list[dict] | None = None, tool_choice: Any = None,
    extra_params: dict[str, Any] | None = None,
) -> Any:
    out_messages = apply_stripping(messages, decision.stripped_last_user)
    backend = cfg.backends[decision.backend]
    return await call_backend(
        backend, messages=out_messages, stream=False,
        tools=tools, tool_choice=tool_choice,
        extra_params=extra_params,
    )


async def dispatch_streaming(
    cfg: Config, decision: RoutingDecision, messages: list[dict[str, Any]],
    *, tools: list[dict] | None = None, tool_choice: Any = None,
    extra_params: dict[str, Any] | None = None,
) -> Any:
    out_messages = apply_stripping(messages, decision.stripped_last_user)
    backend = cfg.backends[decision.backend]
    return await call_backend(
        backend, messages=out_messages, stream=True,
        tools=tools, tool_choice=tool_choice,
        extra_params=extra_params,
    )
