"""TypeSafe AI "Jev" as a request classifier (``classifier.mode = "jev"``).

Jev is a non-autoregressive "System One" model: one POST with the text ("state") and a
set of typed questions returns, in a single parallel pass, a choice + a calibrated
probability distribution + a confidence per question. SAINT asks two ``choice``
questions in ONE call — domain and complexity — mirroring the two independent heads of
the local embedding classifier, and applies the same gate: confidence =
min(domain, complexity); below ``min_confidence`` it returns ``None`` (= defer), and the
router falls through to the existing embedding head and then the LLM labeller. Nothing
about the older classifiers is removed; this is one more mode behind the same flag.

Why try it: the embedding head is fast but nearly blind on the medium/hard boundary
(55.7 % complexity agreement, 48 % coverage on 2026-09-21), and every deferral costs a
~1.4 s Haiku call. Jev claims 70–500 ms with calibrated confidence.

Trade-offs to keep in view: it is a hosted API (api.typesafe.ai, early access) — the last
user message leaves the LAN, there is no local/offline mode, and an outage must degrade to
the local path, which is why every failure here raises ``JevError`` for the router to
catch rather than guessing. Pin ``model`` to a version id: ``jev-latest`` can change
behaviour without notice. Known model weaknesses that matter here: literal reading,
"context rot" on long noisy input (so the router's ``ignore_after`` trimming and
``max_input_chars`` cap apply before we get the text), and no defence against
instructions embedded in the text.
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

# Option descriptions are the whole "prompt" for a System One model — keep them concrete
# and anchored to SAINT's routing consequence (local seat vs cloud), not to vibes.
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


class JevError(RuntimeError):
    """Any failure talking to Jev — the router treats it as 'fall through to the local path'."""


@dataclass(frozen=True)
class JevSettings:
    api_key_env: str = DEFAULT_API_KEY_ENV
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 3.0           # hot path: a slow Jev is worse than the local head
    min_confidence: float | None = None  # None → classifier.min_confidence


def api_key(settings: JevSettings) -> str | None:
    return os.environ.get(settings.api_key_env) or None


def build_request(prompt: str, settings: JevSettings) -> dict:
    return {"model": settings.model, "state": prompt,
            "questions": {"domain": DOMAIN_QUESTION, "complexity": COMPLEXITY_QUESTION}}


def parse_response(data: dict, *, min_confidence: float, latency_ms: int) -> ClassifierResult | None:
    """Validate Jev's answers; ``None`` when either axis is under the confidence gate."""
    answers = data.get("answers")
    if not isinstance(answers, dict):
        raise JevError("response has no 'answers' object")
    picked: dict[str, tuple[str, float]] = {}
    for axis, valid in (("domain", VALID_DOMAINS), ("complexity", VALID_COMPLEXITIES)):
        a = answers.get(axis) or {}
        choice, conf = a.get("choice"), a.get("confidence")
        if choice not in valid:
            raise JevError(f"{axis}: unexpected choice {choice!r}")
        if not isinstance(conf, (int, float)):
            raise JevError(f"{axis}: missing confidence")
        picked[axis] = (choice, float(conf))
    conf = min(picked["domain"][1], picked["complexity"][1])
    if conf < min_confidence:
        return None
    return ClassifierResult(
        domain=picked["domain"][0], complexity=picked["complexity"][0],
        reason=(f"jev {data.get('model') or '?'}: domain {picked['domain'][1]:.2f}, "
                f"complexity {picked['complexity'][1]:.2f}"),
        latency_ms=latency_ms,
    )


async def classify(prompt: str, *, settings: JevSettings, min_confidence: float,
                   client: httpx.AsyncClient | None = None) -> ClassifierResult | None:
    """One Jev call for both axes. Returns None to defer; raises JevError on any failure."""
    key = api_key(settings)
    if not key:
        raise JevError(f"${settings.api_key_env} is not set")
    gate = settings.min_confidence if settings.min_confidence is not None else min_confidence
    own = client is None
    client = client or httpx.AsyncClient(timeout=settings.timeout_s)
    t0 = time.monotonic()
    try:
        resp = await client.post(settings.base_url.rstrip("/") + "/v1/systemone",
                                 json=build_request(prompt, settings),
                                 headers={"Authorization": f"Bearer {key}"})
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
    return parse_response(data, min_confidence=gate, latency_ms=int((time.monotonic() - t0) * 1000))
