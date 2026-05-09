from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from goorouter.backends import call_backend
from goorouter.config import BackendConfig


def _truncate(value: object, limit: int = 500) -> str:
    s = str(value)
    return s if len(s) <= limit else s[:limit] + "...(truncated)"


def _extract_litellm_detail(err: BaseException) -> str:
    """Walk an exception chain looking for HTTP-level diagnostic info.

    LiteLLM wraps provider exceptions in BadRequestError / APIError with
    attributes like `.status_code`, `.message`, `.response`, `.body` —
    most of which `str(err)` doesn't print. Pull whatever's there. Dedupe
    repeats (LiteLLM re-wraps multiple times along the chain).
    """
    seen_keys: set[tuple[str, str]] = set()
    parts: list[str] = []

    def _add(label: str, value: object) -> None:
        if value is None or value == "":
            return
        s = _truncate(value)
        key = (label, s)
        if key in seen_keys:
            return
        seen_keys.add(key)
        parts.append(f"{label}={s}")

    seen_excs: set[int] = set()
    cur: BaseException | None = err
    while cur is not None and id(cur) not in seen_excs:
        seen_excs.add(id(cur))
        _add("status", getattr(cur, "status_code", None))
        _add("body", getattr(cur, "body", None) or getattr(cur, "response_text", None))
        # LiteLLM-specific
        _add("provider", getattr(cur, "llm_provider", None))
        _add("model", getattr(cur, "model", None))
        # httpx Response attached as .response
        resp = getattr(cur, "response", None)
        if resp is not None:
            text = getattr(resp, "text", None)
            if text:
                _add("response.text", text)
            else:
                # Some httpx responses expose `.content` (bytes) but no .text yet
                content = getattr(resp, "content", None)
                if content:
                    try:
                        _add("response.text", content.decode("utf-8", errors="replace"))
                    except Exception:
                        pass
        cur = cur.__cause__ or cur.__context__
    return "; ".join(parts) if parts else ""


def _log_classifier_failure(backend_name: str, err: Exception) -> None:
    """Print the actual classifier error to stderr so users can diagnose.

    `classify_with_fallback` swallows ClassifierError to drive the fallback
    chain; without this, real errors (LM Studio not running, model name
    mismatch, network refused, etc.) become invisible behind the generic
    "(both failed)" stdout marker. We log full detail to stderr instead,
    including any HTTP status / response body the underlying SDK exposes.
    """
    detail = _extract_litellm_detail(err)
    suffix = f" [{detail}]" if detail else ""
    print(
        f"[router] classifier '{backend_name}' failed: "
        f"{type(err).__name__}: {err}{suffix}",
        file=sys.stderr, flush=True,
    )

VALID_DOMAINS = {"code", "general"}
VALID_COMPLEXITIES = {"trivial", "medium", "hard"}


class ClassifierError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClassifierResult:
    domain: str
    complexity: str
    reason: str
    latency_ms: int


def load_prompt_template(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return (
        resources.files("goorouter")
        .joinpath("classifier_prompt.txt")
        .read_text(encoding="utf-8")
    )


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _extract_json(text: str) -> Any:
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ClassifierError(f"classifier did not return parseable JSON: {text!r}") from e


async def classify(backend: BackendConfig, *, prompt: str, template: str) -> ClassifierResult:
    """Run the classifier on `prompt`. Returns ClassifierResult or raises ClassifierError."""
    rendered = template.replace("{prompt}", prompt)
    started = time.monotonic()
    try:
        response = await call_backend(
            backend,
            messages=[{"role": "user", "content": rendered}],
            stream=False,
        )
    except Exception as e:
        raise ClassifierError(f"classifier call failed: {e}") from e
    latency_ms = int((time.monotonic() - started) * 1000)

    content = (
        response["choices"][0]["message"]["content"]
        if isinstance(response, dict)
        else response.choices[0].message.content
    )
    parsed = _extract_json(content)
    if not isinstance(parsed, dict):
        raise ClassifierError(f"classifier JSON not an object: {parsed!r}")
    domain = parsed.get("domain")
    complexity = parsed.get("complexity")
    reason = parsed.get("reason", "")
    if domain not in VALID_DOMAINS:
        raise ClassifierError(f"classifier domain '{domain}' not in {VALID_DOMAINS}")
    if complexity not in VALID_COMPLEXITIES:
        raise ClassifierError(f"classifier complexity '{complexity}' not in {VALID_COMPLEXITIES}")
    return ClassifierResult(
        domain=domain, complexity=complexity, reason=str(reason), latency_ms=latency_ms,
    )


@dataclass(frozen=True)
class FallbackOutcome:
    result: ClassifierResult | None
    classifier_used: str | None
    fallback_reason: str | None  # "oversize" | "primary_error" | None
    input_chars: int             # chars actually sent to whichever classifier ran
    input_truncated_from: int | None


async def classify_with_fallback(
    *,
    primary: BackendConfig,
    fallback: BackendConfig | None,
    prompt: str,
    max_input_chars: int,
    template: str,
) -> FallbackOutcome:
    original_len = len(prompt)

    # Case 1: oversize and fallback configured → skip primary, full prompt to fallback
    if original_len > max_input_chars and fallback is not None:
        try:
            result = await classify(fallback, prompt=prompt, template=template)
            return FallbackOutcome(
                result=result, classifier_used=fallback.name,
                fallback_reason="oversize", input_chars=original_len,
                input_truncated_from=None,
            )
        except ClassifierError as e:
            _log_classifier_failure(fallback.name, e)
            return FallbackOutcome(
                result=None, classifier_used=None,
                fallback_reason="oversize", input_chars=original_len,
                input_truncated_from=None,
            )

    # Case 2: oversize but no fallback → head-truncate, try primary
    if original_len > max_input_chars and fallback is None:
        truncated = prompt[:max_input_chars]
        try:
            result = await classify(primary, prompt=truncated, template=template)
            return FallbackOutcome(
                result=result, classifier_used=primary.name,
                fallback_reason=None, input_chars=max_input_chars,
                input_truncated_from=original_len,
            )
        except ClassifierError as e:
            _log_classifier_failure(primary.name, e)
            return FallbackOutcome(
                result=None, classifier_used=None,
                fallback_reason="primary_error", input_chars=max_input_chars,
                input_truncated_from=original_len,
            )

    # Case 3: fits → primary; fall back on error if configured
    try:
        result = await classify(primary, prompt=prompt, template=template)
        return FallbackOutcome(
            result=result, classifier_used=primary.name,
            fallback_reason=None, input_chars=original_len,
            input_truncated_from=None,
        )
    except ClassifierError as primary_err:
        _log_classifier_failure(primary.name, primary_err)
        if fallback is None:
            return FallbackOutcome(
                result=None, classifier_used=None,
                fallback_reason="primary_error", input_chars=original_len,
                input_truncated_from=None,
            )
        try:
            result = await classify(fallback, prompt=prompt, template=template)
            return FallbackOutcome(
                result=result, classifier_used=fallback.name,
                fallback_reason="primary_error", input_chars=original_len,
                input_truncated_from=None,
            )
        except ClassifierError as fb_err:
            _log_classifier_failure(fallback.name, fb_err)
            return FallbackOutcome(
                result=None, classifier_used=None,
                fallback_reason="primary_error", input_chars=original_len,
                input_truncated_from=None,
            )
