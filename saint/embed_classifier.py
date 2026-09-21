"""Embedding-based classifier — a fast CPU alternative to the LLM classifier.

Instead of prompting a generative model for `(domain, complexity)`, we embed the prompt
once (reusing a local embedding backend, e.g. nomic-embed served by johnny) and run two
tiny logistic-regression heads over the vector. That's ~50-100x faster than a 3-4B chat
model on CPU, deterministic, and sidesteps structured-output parsing entirely.

The head is *distilled* from the LLM classifier's own logged predictions (see
`saint classifier train`), so it mimics your existing router; it is only ever a speed
optimization. When it isn't confident enough it returns None, and the router falls back
to the LLM classifier — the cheap head handles the confident majority, the LLM the
ambiguous tail.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from saint.backends import _resolve_api_key, _resolve_model_id
from saint.classifier import ClassifierResult
from saint.config import BackendConfig

log = logging.getLogger(__name__)

DEFAULT_HEAD_PATH = "~/.config/saint/classifier_head.npz"


# --------------------------------------------------------------------------- embedding
# An embedder has a hard context window (nomic-embed: 2048 tokens) and no way to ask it
# to truncate. A character cap can't stand in for it: measured against nomic-embed, prose
# runs ~4.5 chars/token but JSON ~1.3, so any cap safe for the dense case throws away most
# of a prose prompt. Instead we send the text as-is and only shrink what actually overflows.
_CTX_MARKERS = ("maximum context length", "context window", "context_length_exceeded",
                "is longer than the maximum", "reduce the length")
_SHRINK_FLOOR = 256      # chars; below this a row is not worth classifying
_SHRINK_TRIES = 8        # halving from any realistic prompt reaches the floor well inside this


def _is_context_error(exc: BaseException) -> bool:
    """True when a backend refused the input for exceeding its context window."""
    if type(exc).__name__ == "ContextWindowExceededError":
        return True
    return any(m in str(exc).lower() for m in _CTX_MARKERS)


async def _embed_raw(backend: BackendConfig, texts: list[str]) -> np.ndarray:
    import litellm

    resp = await litellm.aembedding(
        model=_resolve_model_id(backend),
        input=list(texts),
        api_base=backend.base_url,
        api_key=_resolve_api_key(backend),
        timeout=backend.timeout_s,
    )
    data = resp["data"] if isinstance(resp, dict) else resp.data
    vecs = [
        np.asarray(d["embedding"] if isinstance(d, dict) else d.embedding, dtype=np.float32)
        for d in data
    ]
    return np.vstack(vecs)


async def _embed_one_shrinking(backend: BackendConfig, text: str, on_shrink=None) -> np.ndarray:
    """Embed one text, halving it until the backend accepts it. Raises if it never fits."""
    t, original = text, len(text)
    for _ in range(_SHRINK_TRIES):
        try:
            vec = await _embed_raw(backend, [t])
            if len(t) < original and on_shrink is not None:
                on_shrink(original, len(t))
            return vec
        except Exception as e:
            if not _is_context_error(e) or len(t) <= _SHRINK_FLOOR:
                raise
            t = t[: max(_SHRINK_FLOOR, len(t) // 2)]
    raise RuntimeError(f"could not fit a {original}-char text into {backend.name}'s context")


async def embed_texts(backend: BackendConfig, texts: list[str], *, on_shrink=None) -> np.ndarray:
    """Embed a batch of texts via the backend's OpenAI-compatible /v1/embeddings. Returns
    an (n, dim) float32 array. Raises on transport/model error (caller decides fallback).

    If the batch is refused for exceeding the embedder's context window, each text is
    embedded individually and any oversize one is halved until it fits — one pathological
    row can't fail the batch. `on_shrink(original_chars, kept_chars)` fires per shrink so
    callers can report how much was dropped."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    try:
        return await _embed_raw(backend, list(texts))
    except Exception as e:
        if not _is_context_error(e):
            raise
    return np.vstack([await _embed_one_shrinking(backend, t, on_shrink) for t in texts])


# --------------------------------------------------------------------------- math
def _normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, 1e-8, None)


def _prep(v: np.ndarray, embed_dim: int) -> np.ndarray:
    """L2-normalize the embedding part only; extra features pass through as they are."""
    if not embed_dim or embed_dim >= v.shape[-1]:
        return _normalize(v)
    return np.concatenate([_normalize(v[..., :embed_dim]), v[..., embed_dim:]], axis=-1)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _fit_softmax(X: np.ndarray, y_idx: np.ndarray, n_classes: int,
                 l2: float = 1.0, lr: float = 0.5, iters: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    """L2-regularized multinomial logistic regression via full-batch gradient descent.
    Tiny head (dim×classes), small data — trains in seconds. Iteration count matters:
    under-converged heads produce soft probabilities that fail the min_confidence gate
    and defer everything to the LLM (400 iters halved coverage at 1200 samples)."""
    n, d = X.shape
    W = np.zeros((d, n_classes), dtype=np.float64)
    b = np.zeros(n_classes, dtype=np.float64)
    if n_classes <= 1:  # degenerate: one label observed — nothing to separate
        return W, b
    Y = np.eye(n_classes)[y_idx]
    for _ in range(iters):
        P = _softmax(X @ W + b)
        gW = X.T @ (P - Y) / n + (l2 / n) * W
        gb = (P - Y).mean(axis=0)
        W -= lr * gW
        b -= lr * gb
    return W, b


def _predict_one(vec: np.ndarray, W: np.ndarray, b: np.ndarray, classes: list[str]) -> tuple[str, float]:
    if len(classes) == 1:
        return classes[0], 1.0
    p = _softmax(vec @ W + b)
    i = int(np.argmax(p))
    return classes[i], float(p[i])


# --------------------------------------------------------------------------- head
@dataclass
class Head:
    """A trained two-axis classifier head (domain + complexity) over embeddings."""

    dim: int
    embed_model: str
    n_samples: int
    trained_at: str
    domain_classes: list[str]
    W_domain: np.ndarray
    b_domain: np.ndarray
    complexity_classes: list[str]
    W_complexity: np.ndarray
    b_complexity: np.ndarray
    embed_dim: int = 0          # leading columns that are the embedding (0 = all of `dim`: legacy heads)
    feature_spec: str = ""      # extra trailing features the head expects (saint.prompt_features.SPEC) or ""

    def predict(self, vec: np.ndarray) -> tuple[str, float, str, float]:
        """(domain, domain_conf, complexity, complexity_conf) for one vector: the embedding,
        followed by the head's extra features when it was trained with any."""
        v = _prep(np.asarray(vec, dtype=np.float64), self.embed_dim)
        dl, dc = _predict_one(v, self.W_domain, self.b_domain, self.domain_classes)
        cl, cc = _predict_one(v, self.W_complexity, self.b_complexity, self.complexity_classes)
        return dl, dc, cl, cc

    def save(self, path: str | Path) -> None:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p, dim=self.dim, embed_model=self.embed_model, n_samples=self.n_samples,
            trained_at=self.trained_at,
            domain_classes=np.array(self.domain_classes), W_domain=self.W_domain, b_domain=self.b_domain,
            complexity_classes=np.array(self.complexity_classes),
            W_complexity=self.W_complexity, b_complexity=self.b_complexity,
            embed_dim=self.embed_dim, feature_spec=self.feature_spec,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Head":
        d = np.load(Path(path).expanduser(), allow_pickle=False)
        return cls(
            dim=int(d["dim"]), embed_model=str(d["embed_model"]), n_samples=int(d["n_samples"]),
            trained_at=str(d["trained_at"]),
            domain_classes=[str(x) for x in d["domain_classes"]],
            W_domain=d["W_domain"], b_domain=d["b_domain"],
            complexity_classes=[str(x) for x in d["complexity_classes"]],
            W_complexity=d["W_complexity"], b_complexity=d["b_complexity"],
            embed_dim=int(d["embed_dim"]) if "embed_dim" in d else 0,
            feature_spec=str(d["feature_spec"]) if "feature_spec" in d else "",
        )


def train_head(embeddings: np.ndarray, domain_labels: list[str], complexity_labels: list[str],
               *, embed_model: str, extras: np.ndarray | None = None, feature_spec: str = "") -> Head:
    """Fit both axes from distilled labels. `embeddings` is (n, dim); labels are parallel lists.
    `extras` (n, k) are optional extra features appended after the normalized embedding."""
    embed_dim = 0 if extras is None else int(np.asarray(embeddings).shape[1])
    X = np.asarray(embeddings, dtype=np.float64)
    X = _normalize(X) if extras is None else np.hstack([_normalize(X), np.asarray(extras, dtype=np.float64)])
    dom_classes = sorted(set(domain_labels))
    cplx_classes = sorted(set(complexity_labels))
    dom_idx = np.array([dom_classes.index(x) for x in domain_labels])
    cplx_idx = np.array([cplx_classes.index(x) for x in complexity_labels])
    Wd, bd = _fit_softmax(X, dom_idx, len(dom_classes))
    Wc, bc = _fit_softmax(X, cplx_idx, len(cplx_classes))
    return Head(
        dim=X.shape[1], embed_model=embed_model, n_samples=X.shape[0],
        trained_at=datetime.now(UTC).isoformat(),
        domain_classes=dom_classes, W_domain=Wd, b_domain=bd,
        complexity_classes=cplx_classes, W_complexity=Wc, b_complexity=bc,
        embed_dim=embed_dim, feature_spec=feature_spec if extras is not None else "",
    )


# --------------------------------------------------------------------------- classify
async def classify(head: Head, embed_backend: BackendConfig, *, prompt: str,
                   min_confidence: float, feature_url: str | None = None) -> ClassifierResult | None:
    """Embed + predict. Returns a ClassifierResult, or None when either axis is below
    `min_confidence` (the signal for the router to fall back to the LLM classifier)."""
    started = time.monotonic()
    X = await embed_texts(embed_backend, [prompt])
    vec = X[0]
    if head.feature_spec:
        from saint import prompt_features as PF
        if not feature_url or head.feature_spec != PF.SPEC:
            return None                      # head needs features we can't supply → LLM classifier
        try:
            vec = np.concatenate([vec, (await PF.extras(feature_url, [prompt]))[0]])
        except Exception as e:  # noqa: BLE001 — sidecar down/slow: abstain, never fail the request
            log.warning("prompt-features sidecar unavailable (%s) — deferring to the LLM classifier", e)
            return None
    dl, dc, cl, cc = head.predict(vec)
    latency_ms = int((time.monotonic() - started) * 1000)
    conf = min(dc, cc)
    if conf < min_confidence:
        return None
    return ClassifierResult(
        domain=dl, complexity=cl,
        reason=f"embedding head · conf {conf:.2f}", latency_ms=latency_ms,
    )
