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

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from saint.backends import _resolve_api_key, _resolve_model_id
from saint.classifier import ClassifierResult
from saint.config import BackendConfig

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

    def predict(self, vec: np.ndarray) -> tuple[str, float, str, float]:
        """(domain, domain_conf, complexity, complexity_conf) for one (already-embedded) vector."""
        v = _normalize(np.asarray(vec, dtype=np.float64))
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
        )


def train_head(embeddings: np.ndarray, domain_labels: list[str], complexity_labels: list[str],
               *, embed_model: str) -> Head:
    """Fit both axes from distilled labels. `embeddings` is (n, dim); labels are parallel lists."""
    X = _normalize(np.asarray(embeddings, dtype=np.float64))
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
    )


# --------------------------------------------------------------------------- classify
async def classify(head: Head, embed_backend: BackendConfig, *, prompt: str,
                   min_confidence: float) -> ClassifierResult | None:
    """Embed + predict. Returns a ClassifierResult, or None when either axis is below
    `min_confidence` (the signal for the router to fall back to the LLM classifier)."""
    started = time.monotonic()
    X = await embed_texts(embed_backend, [prompt])
    dl, dc, cl, cc = head.predict(X[0])
    latency_ms = int((time.monotonic() - started) * 1000)
    conf = min(dc, cc)
    if conf < min_confidence:
        return None
    return ClassifierResult(
        domain=dl, complexity=cl,
        reason=f"embedding head · conf {conf:.2f}", latency_ms=latency_ms,
    )
