"""``classifier.mode = "laya"`` — SAINT's fine-tuned Laya decision model, served locally.

Laya (Convai Innovations, Apache 2.0) is a 421M non-autoregressive decision model: a
ModernBERT-large encoder plus a typed decision head that reads a question and its option
descriptions at request time and returns a choice with probabilities. Zero-shot it was
useless for routing (60–75 %); fine-tuned on the rubric-v3 labels with its own decision
head kept intact it routes cloud/local at 97 % with 96 % coverage at the 0.8 gate, ~22 ms
per request on the RTX 4080 (2026-09-26 evaluation, infrastructure/saint/README.md).

The server is ``laya-serve`` pinned to the checkpoint (``tools/laya_saint_serve.py``,
systemd --user ``laya-saint.service``, :8130) speaking the Jev ``/v1/systemone`` wire
protocol, so the transport is ``saint.jev_classifier``. What is specific here:

- **The question text is part of the model's input.** The checkpoint directory carries
  ``saint-training.json`` with the exact questions (instructions + option descriptions) it
  was trained with; SAINT sends those verbatim. ``classifier.laya_questions_path`` points at
  it; the file is re-read when its mtime changes (retrain → export → restart the server,
  no SAINT restart needed).
- No API key; the gate reads the answer's max probability (``answer_confidence``).
- Unsure (below ``laya_min_confidence``/``min_confidence``) or unavailable → the LLM
  labeller (``laya_fallthrough = "llm"``), or the embedding head first when
  ``laya_fallthrough = "embedding"``. Laya-labelled rows never train the head
  (``storage.fetch_training_rows``); the LLM labeller stays the reference for drift.

Retrain (about 6 min on the 4080; labels live outside the repo):
    tools/classifier_eval/laya_finetune.py --variant full --questions both --epochs 4 --head-lr 1e-4 --no-cv --final --tag <tag>
    tools/classifier_eval/laya_export_checkpoint.py <tag>.pt <checkpoint-dir> <tag>.json
    systemctl --user restart laya-saint.service
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from saint import jev_classifier as JC
from saint.classifier import VALID_COMPLEXITIES, VALID_DOMAINS, ClassifierResult

CLASSIFIER_USED = "laya"
RETRAIN_FIX = ("retrain Laya: tools/classifier_eval/laya_finetune.py --variant full --questions both "
               "--no-cv --final, laya_export_checkpoint.py, then systemctl --user restart laya-saint")

_cache: dict[str, tuple[float, dict]] = {}


def load_training_meta(path: str | Path) -> dict:
    """The checkpoint's saint-training.json (question text + provenance), mtime-cached."""
    p = Path(path).expanduser()
    mtime = p.stat().st_mtime
    hit = _cache.get(str(p))
    if hit and hit[0] == mtime:
        return hit[1]
    meta = json.loads(p.read_text())
    questions = meta.get("question") if isinstance(meta, dict) and "question" in meta else meta
    if not isinstance(questions, dict):
        raise ValueError(f"{p}: expected a JSON object with the questions")
    for axis, valid in (("domain", VALID_DOMAINS), ("complexity", VALID_COMPLEXITIES)):
        q = questions.get(axis)
        if not isinstance(q, dict) or q.get("type") != "choice" or set((q.get("criteria") or {})) != valid:
            raise ValueError(f"{p}: question {axis!r} must be a 'choice' with criteria {sorted(valid)} "
                             "(a complexity-only checkpoint cannot serve mode='laya')")
    out = {"questions": questions, "trained_at": meta.get("date") if isinstance(meta, dict) else None,
           "trained_on": meta.get("trained_on") if isinstance(meta, dict) else None,
           "base": meta.get("base") if isinstance(meta, dict) else None}
    _cache[str(p)] = (mtime, out)
    return out


def settings_from(classifier_cfg) -> JC.JevSettings:
    """JevSettings for the local server, from ClassifierConfig's laya_* fields."""
    meta = load_training_meta(classifier_cfg.laya_questions_path)
    return JC.JevSettings(
        api_key_env=None, model=classifier_cfg.laya_model, base_url=classifier_cfg.laya_url,
        timeout_s=classifier_cfg.laya_timeout_s, min_confidence=classifier_cfg.laya_min_confidence,
        questions=meta["questions"], classifier_used=CLASSIFIER_USED,
    )


def trained_at(classifier_cfg) -> str | None:
    """ISO timestamp the checkpoint was trained (from saint-training.json) — drift rows start here."""
    d = load_training_meta(classifier_cfg.laya_questions_path).get("trained_at")
    if not d:
        return None
    return d if "T" in d else f"{d}T00:00:00"


async def classify(prompt: str, *, classifier_cfg, min_confidence: float,
                   client: httpx.AsyncClient | None = None) -> ClassifierResult | None:
    """None = defer (unsure); raises JevError when the server is unavailable or answers oddly."""
    return await JC.classify(prompt, settings=settings_from(classifier_cfg),
                             min_confidence=min_confidence, client=client)


async def predict(prompt: str, *, classifier_cfg, client: httpx.AsyncClient | None = None) -> tuple[str, str, float, int]:
    """Ungated answer for offline replay (drift): (domain, complexity, min confidence, latency_ms)."""
    data, latency_ms = await JC.fetch(prompt, settings=settings_from(classifier_cfg), client=client)
    picks = JC.parse_picks(data)
    return picks["domain"][0], picks["complexity"][0], min(picks["domain"][1], picks["complexity"][1]), latency_ms
