"""System One ("Jev" wire protocol) request classifier — ``classifier.mode = "jev"`` and the
transport behind ``mode = "laya"``.

One POST with the text ("state") and a set of typed questions returns, in a single
parallel pass, a choice + a calibrated probability distribution + a confidence per
question. SAINT asks two ``choice`` questions in ONE call — domain and complexity —
mirroring the two independent heads of the local embedding classifier, and applies the
same gate: confidence = min(domain, complexity); below ``min_confidence`` it returns
``None`` (= defer), and the router falls through to the embedding head and/or the LLM
labeller. Nothing about the older classifiers is removed; these are modes behind the
same flag.

Two servers speak this protocol:
- TypeSafe AI's hosted Jev (``mode = "jev"``, api.typesafe.ai, bearer key, early access —
  parked 2026-09-21: cloud only, never measured here).
- Convai's open Laya via ``laya-serve`` (``mode = "laya"``, local, ``tools/laya_saint_serve.py``)
  — SAINT's fine-tuned checkpoint since 2026-09-26; see ``saint/laya_classifier.py`` for the
  question text, which is part of the model's input and must match training.

Every failure raises ``JevError`` so the router falls through instead of guessing. The
gate reads ``answer_confidence`` (the answer's max probability — what the offline gate
sweeps measured) and falls back to ``confidence`` for servers that omit it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from saint.classifier import VALID_COMPLEXITIES, VALID_DOMAINS, ClassifierResult

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-1.13.0"  # pinned on purpose — aliases drift
DEFAULT_API_KEY_ENV = "TYPESAFE_API_KEY"
CLASSIFIER_USED = "typesafe-jev"

# Default (zero-shot) option descriptions for the hosted Jev — its only steering lever.
# mode="laya" replaces these with the exact questions the checkpoint was trained on.
DOMAIN_QUESTION = {
    "type": "choice",
    "instructions": "What kind of work is the user's message asking for?",
    "criteria": {
        "code": ("Writing, reading, debugging, reviewing or explaining source code, shell commands, "
                 "configuration files, queries, build/CI problems, stack traces, or software architecture."),
        "general": ("Anything else: questions, writing, analysis, planning, math, science, conversation, "
                    "summaries, advice — no programming artifact is involved."),
    },
}
COMPLEXITY_QUESTION = {
    "type": "choice",
    "instructions": "How demanding is the user's message for an AI assistant to answer well?",
    "criteria": {
        "trivial": ("A greeting, acknowledgement, short factual lookup, simple rephrase, or a one-line "
                    "edit. A small model answers it correctly in a sentence or two."),
        "medium": ("A routine, well-specified task: write or fix a single function, explain a concept, "
                   "summarise a text, answer a multi-part question. A capable 30B-class local model "
                   "handles it reliably in one pass."),
        "hard": ("Needs deep multi-step reasoning, design or architecture trade-offs, subtle debugging "
                 "across several components, long-horizon planning, or expert judgment where a wrong "
                 "answer is costly. Warrants the strongest available model."),
    },
}
DEFAULT_QUESTIONS = {"domain": DOMAIN_QUESTION, "complexity": COMPLEXITY_QUESTION}


class JevError(RuntimeError):
    """Any failure talking to the server — the router treats it as 'fall through to the local path'."""


@dataclass(frozen=True)
class JevSettings:
    api_key_env: str | None = DEFAULT_API_KEY_ENV  # None → no Authorization header (a local server)
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 3.0           # hot path: a slow classifier is worse than the local head
    min_confidence: float | None = None  # None → classifier.min_confidence
    questions: dict | None = None    # None → DEFAULT_QUESTIONS; mode="laya" passes the trained text
    classifier_used: str = CLASSIFIER_USED


def api_key(settings: JevSettings) -> str | None:
    if not settings.api_key_env:
        return None
    return os.environ.get(settings.api_key_env) or None


def build_request(prompt: str, settings: JevSettings) -> dict:
    return {"model": settings.model, "state": prompt,
            "questions": dict(settings.questions or DEFAULT_QUESTIONS)}


def parse_picks(data: dict) -> dict[str, tuple[str, float]]:
    """Validate the two answers → {axis: (label, confidence)}. Raises JevError on anything odd."""
    answers = data.get("answers")
    if not isinstance(answers, dict):
        raise JevError("response has no 'answers' object")
    picked: dict[str, tuple[str, float]] = {}
    for axis, valid in (("domain", VALID_DOMAINS), ("complexity", VALID_COMPLEXITIES)):
        a = answers.get(axis) or {}
        choice = a.get("choice")
        conf = a.get("answer_confidence")
        if not isinstance(conf, (int, float)):
            conf = a.get("confidence")
        if choice not in valid:
            raise JevError(f"{axis}: unexpected choice {choice!r}")
        if not isinstance(conf, (int, float)):
            raise JevError(f"{axis}: missing confidence")
        picked[axis] = (choice, float(conf))
    return picked


def parse_response(data: dict, *, min_confidence: float, latency_ms: int,
                   classifier_used: str = CLASSIFIER_USED) -> ClassifierResult | None:
    """Validate the answers; ``None`` when either axis is under the confidence gate."""
    picked = parse_picks(data)
    conf = min(picked["domain"][1], picked["complexity"][1])
    if conf < min_confidence:
        return None
    return ClassifierResult(
        domain=picked["domain"][0], complexity=picked["complexity"][0],
        reason=(f"{classifier_used} {data.get('model') or '?'}: domain {picked['domain'][1]:.2f}, "
                f"complexity {picked['complexity'][1]:.2f}"),
        latency_ms=latency_ms,
    )


async def fetch(prompt: str, *, settings: JevSettings, client: httpx.AsyncClient | None = None) -> tuple[dict, int]:
    """One call for both axes → (raw JSON, latency_ms). Raises JevError on any failure."""
    headers = {}
    if settings.api_key_env:
        key = api_key(settings)
        if not key:
            raise JevError(f"${settings.api_key_env} is not set")
        headers["Authorization"] = f"Bearer {key}"
    own = client is None
    client = client or httpx.AsyncClient(timeout=settings.timeout_s)
    t0 = time.monotonic()
    try:
        resp = await client.post(settings.base_url.rstrip("/") + "/v1/systemone",
                                 json=build_request(prompt, settings), headers=headers,
                                 timeout=settings.timeout_s)
        if resp.status_code != 200:
            raise JevError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
    except httpx.HTTPError as e:
        raise JevError(f"{type(e).__name__}: {e}") from e
    except ValueError as e:
        raise JevError(f"non-JSON response: {e}") from e
    finally:
        if own:
            await client.aclose()
    return data, int((time.monotonic() - t0) * 1000)


async def classify(prompt: str, *, settings: JevSettings, min_confidence: float,
                   client: httpx.AsyncClient | None = None) -> ClassifierResult | None:
    """Returns None to defer; raises JevError on any failure."""
    gate = settings.min_confidence if settings.min_confidence is not None else min_confidence
    data, latency_ms = await fetch(prompt, settings=settings, client=client)
    return parse_response(data, min_confidence=gate, latency_ms=latency_ms,
                          classifier_used=settings.classifier_used)
