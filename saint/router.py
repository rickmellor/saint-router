from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from pathlib import Path

from saint.backends import call_backend
from saint.classifier import (
    ClassifierError,
    ClassifierResult,
    FallbackOutcome,
    _log_classifier_failure,
    classify_with_fallback,
    load_prompt_template,
)
from saint.config import BackendConfig, Config
from saint.policy import resolve_policy
from saint.prefixes import ParsedPrefixes, parse_prefixes
from saint.route_cache import (
    CachedLabels,
    ConversationEntry,
    RouteCaches,
    content_text as _content_text,
    conversation_key,
    turn_key,
)

# Trained embedding heads are cached by path, invalidated on mtime (so `classifier train`
# takes effect without a restart).
_HEAD_CACHE: dict[str, tuple[float, object]] = {}


def _load_head(path: str | None):
    from saint.embed_classifier import DEFAULT_HEAD_PATH, Head

    p = Path(path or DEFAULT_HEAD_PATH).expanduser()
    if not p.exists():
        return None
    mtime = p.stat().st_mtime
    cached = _HEAD_CACHE.get(str(p))
    if cached and cached[0] == mtime:
        return cached[1]
    head = Head.load(p)
    _HEAD_CACHE[str(p)] = (mtime, head)
    return head


async def _classify_for_route(cfg: Config, prompt: str) -> FallbackOutcome:
    """Classify per config: the embedding head first (mode='embedding'), else/on low
    confidence/on error, the LLM classifier. Always returns a FallbackOutcome for logging."""
    primary = cfg.backends[cfg.classifier.backend]
    fallback = cfg.backends[cfg.classifier.fallback_backend] if cfg.classifier.fallback_backend else None
    template = load_prompt_template(cfg.classifier.prompt_template_path)

    async def _llm(reason_prefix: str | None) -> FallbackOutcome:
        outcome = await classify_with_fallback(
            primary=primary, fallback=fallback, prompt=prompt,
            max_input_chars=cfg.classifier.max_input_chars, template=template,
        )
        if reason_prefix and outcome.fallback_reason is None:
            outcome = FallbackOutcome(
                result=outcome.result, classifier_used=outcome.classifier_used,
                fallback_reason=reason_prefix, input_chars=outcome.input_chars,
                input_truncated_from=outcome.input_truncated_from,
            )
        return outcome

    if cfg.classifier.mode != "embedding":
        return await _llm(None)

    from saint import embed_classifier as EC

    head = _load_head(cfg.classifier.head_path)
    embed_backend = cfg.backends.get(cfg.classifier.embedding_backend or "")
    if head is None or embed_backend is None:
        return await _llm("embedding_untrained")

    text = prompt[: cfg.classifier.max_input_chars]
    try:
        result = await EC.classify(head, embed_backend, prompt=text, min_confidence=cfg.classifier.min_confidence)
    except Exception as e:
        _log_classifier_failure(embed_backend.name, ClassifierError(str(e)))
        return await _llm("embedding_error")
    if result is None:  # head not confident enough — defer to the LLM
        return await _llm("embedding_defer")
    return FallbackOutcome(
        result=result, classifier_used=f"{embed_backend.name} (embed-head)",
        fallback_reason=None, input_chars=len(text),
        input_truncated_from=(len(prompt) if len(prompt) > len(text) else None),
    )

def _outcome_from_labels(labels: CachedLabels, used: str, clf_input: str) -> FallbackOutcome:
    """Rebuild a FallbackOutcome from cached labels. `used` is 'cache' (turn cache) or
    'inherited' (conversation affinity); both are excluded from classifier training
    (storage.fetch_training_rows) so reuse never manufactures training rows."""
    return FallbackOutcome(
        result=ClassifierResult(
            domain=labels.domain, complexity=labels.complexity,
            reason=f"{labels.reason} [reused from {labels.source}]", latency_ms=0,
        ),
        classifier_used=used,
        fallback_reason=None,
        input_chars=len(clf_input),
        input_truncated_from=None,
    )


SAINT_AUTO = "saint-auto"
SAINT_EXPLAIN = "saint-explain"
SAINT_PREFIX = "saint-"


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


def _cut_at_markers(text: str, markers: tuple[str, ...]) -> str:
    """Truncate at the earliest occurrence of any classifier.ignore_after marker.

    Agent clients (hermes, opencode, …) append memory-recall / context blocks to the
    user message; classifying that blob instead of the user's request skews routing.
    Only the classifier input (and the logged prompt, so training stays coherent with
    what the head sees at runtime) is cut — dispatch forwards the full message.
    """
    cut = len(text)
    for m in markers:
        i = text.find(m)
        if 0 <= i < cut:
            cut = i
    return text[:cut].rstrip()


async def decide_route(
    *,
    cfg: Config,
    model_field: str,
    messages: list[dict[str, Any]],
    caches: RouteCaches | None = None,   # server passes app.state.route_caches; CLI passes nothing
    session_id: str | None = None,       # x-session-id header / session_id body field
) -> RoutingDecision:
    request_id = str(uuid.uuid4())

    mode: Literal["dispatch", "explain"] = "explain" if model_field == SAINT_EXPLAIN else "dispatch"

    last_user = _last_user_message(messages)
    multimodal = bool(last_user and _is_multimodal_content(last_user.get("content")))
    full_text = _content_text(last_user.get("content")) if last_user else None
    # classifier + log view of the message: client-injected context cut off.
    # `full_text` (via parsed.stripped) is what dispatch forwards.
    last_text_original = (
        _cut_at_markers(full_text, cfg.classifier.ignore_after)
        if full_text is not None and cfg.classifier.ignore_after else full_text
    )

    urgencies = {"urgent", "patient", "normal"}
    backend_alias_map = {b.name: set(b.aliases) for b in cfg.backends.values()}

    parsed = ParsedPrefixes(urgency=None, pinned_backend=None,
                             stripped=full_text or "", raw="")
    if full_text is not None and not multimodal:
        parsed = parse_prefixes(full_text, urgencies, backend_alias_map)

    pinned_via_model: str | None = None
    if model_field.startswith(SAINT_PREFIX) and model_field not in (SAINT_AUTO, SAINT_EXPLAIN):
        candidate = model_field[len(SAINT_PREFIX):]
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

    clf_input = (
        _cut_at_markers(parsed.stripped, cfg.classifier.ignore_after)
        if cfg.classifier.ignore_after else parsed.stripped
    )

    # Routing caches participate only for real dispatches with no pin and no multimodal
    # content (pins/multimodal returned above; explain and CLI paths bypass here).
    use_caches = caches is not None and mode == "dispatch"
    conv_key = (
        conversation_key(messages, session_id)
        if use_caches and caches.conversations is not None else None
    )
    conv_entry: ConversationEntry | None = (
        caches.conversations.get(conv_key) if conv_key else None
    )

    outcome: FallbackOutcome | None = None
    labels: CachedLabels | None = None  # canonical labels behind `outcome`, for cache writes
    empty_input = last_user is None or not clf_input.strip()

    if empty_input and conv_entry is None:
        backend = resolve_policy(cfg.routing.policy, urgency, "general", "trivial")
        return RoutingDecision(
            request_id=request_id, mode=mode, backend=backend,
            pinned_backend=None, urgency=urgency, parsed=parsed,
            classifier_outcome=None, classifier_result=None,
            multimodal=multimodal, model_field=model_field,
            last_user_content_original=last_text_original,
            stripped_last_user=parsed.stripped if full_text is not None else None,
        )

    # (a) empty input (incl. injection-only after ignore_after) inherits when it can;
    # (b) short follow-ups inherit the conversation's labels instead of classifying
    #     "yes, do that" out of context; sticky_conversations inherits regardless.
    if conv_entry is not None and (
        empty_input
        or cfg.cache.sticky_conversations
        or len(clf_input) < cfg.cache.short_follow_up_max_chars
    ):
        labels = conv_entry.labels
        outcome = _outcome_from_labels(labels, "inherited", clf_input)

    # (c) turn cache: agent-loop turns repeat the same last user message.
    tk = turn_key(clf_input, urgency)
    if outcome is None and use_caches and caches.turns is not None:
        hit = caches.turns.get(tk)
        if hit is not None:
            labels = hit
            outcome = _outcome_from_labels(labels, "cache", clf_input)

    # (d) classify; cache labels on success (never cache failures).
    if outcome is None:
        outcome = await _classify_for_route(cfg, clf_input)
        if outcome.result is not None:
            labels = CachedLabels(
                domain=outcome.result.domain, complexity=outcome.result.complexity,
                reason=outcome.result.reason, source=outcome.classifier_used or "?",
            )
            if use_caches and caches.turns is not None:
                caches.turns.set(tk, labels)

    if outcome.result is None:
        backend = cfg.routing.default_on_failure
    else:
        backend = resolve_policy(cfg.routing.policy, urgency,
                                  outcome.result.domain, outcome.result.complexity)
        # (e) refresh conversation affinity every successful turn — the sliding TTL.
        if conv_key and labels is not None:
            caches.conversations.set(conv_key, ConversationEntry(labels=labels, backend=backend))
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
    effective_backend: BackendConfig | None = None,
) -> Any:
    out_messages = apply_stripping(messages, decision.stripped_last_user)
    backend = effective_backend or cfg.backends[decision.backend]
    return await call_backend(
        backend, messages=out_messages, stream=False,
        tools=tools, tool_choice=tool_choice,
        extra_params=extra_params,
    )


async def dispatch_streaming(
    cfg: Config, decision: RoutingDecision, messages: list[dict[str, Any]],
    *, tools: list[dict] | None = None, tool_choice: Any = None,
    extra_params: dict[str, Any] | None = None,
    effective_backend: BackendConfig | None = None,
) -> Any:
    out_messages = apply_stripping(messages, decision.stripped_last_user)
    backend = effective_backend or cfg.backends[decision.backend]
    return await call_backend(
        backend, messages=out_messages, stream=True,
        tools=tools, tool_choice=tool_choice,
        extra_params=extra_params,
    )
