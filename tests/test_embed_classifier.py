from __future__ import annotations

import numpy as np
import pytest

from saint import embed_classifier as EC
from saint.config import BackendConfig


def _synth(seed: int = 0, n: int = 200):
    """Linearly-separable synthetic embeddings: domain on dim 0, complexity on dim 1."""
    rng = np.random.default_rng(seed)
    doms, cplxs, X = [], [], []
    for _ in range(n):
        d = rng.choice(["code", "general"])
        c = rng.choice(["trivial", "medium", "hard"])
        v = rng.normal(0, 0.3, 24).astype(np.float32)
        v[0] += 2.0 if d == "code" else -2.0
        v[1] += {"trivial": -2.0, "medium": 0.0, "hard": 2.0}[c]
        doms.append(d)
        cplxs.append(c)
        X.append(v)
    return np.vstack(X), doms, cplxs


def test_train_and_predict_separates():
    X, doms, cplxs = _synth()
    head = EC.train_head(X, doms, cplxs, embed_model="test-embed")
    d_ok = sum(head.predict(v)[0] == d for v, d in zip(X, doms))
    c_ok = sum(head.predict(v)[2] == c for v, c in zip(X, cplxs))
    assert d_ok >= len(doms) - 2       # near-perfect on separable data
    assert c_ok >= len(cplxs) - 5
    assert head.domain_classes == ["code", "general"]
    assert head.complexity_classes == ["hard", "medium", "trivial"]


def test_confidence_is_calibrated():
    X, doms, cplxs = _synth()
    head = EC.train_head(X, doms, cplxs, embed_model="test-embed")
    # a clearly-code/hard point should be confident on both axes
    rng = np.random.default_rng(9)
    v = rng.normal(0, 0.1, 24).astype(np.float32)
    v[0] += 2.0
    v[1] += 2.0
    dl, dc, cl, cc = head.predict(v)
    assert (dl, cl) == ("code", "hard")
    assert dc > 0.7 and cc > 0.6


def test_save_load_roundtrip(tmp_path):
    X, doms, cplxs = _synth()
    head = EC.train_head(X, doms, cplxs, embed_model="test-embed")
    p = tmp_path / "head.npz"
    head.save(p)
    h2 = EC.Head.load(p)
    assert h2.n_samples == head.n_samples
    assert h2.embed_model == "test-embed"
    assert h2.domain_classes == head.domain_classes
    for v in X[:20]:
        assert head.predict(v)[0] == h2.predict(v)[0]
        assert head.predict(v)[2] == h2.predict(v)[2]


def _backend() -> BackendConfig:
    return BackendConfig(name="embed", provider="openai", model="m", api_key_env=None,
                         api_key="x", base_url="http://x/v1", aliases=(), timeout_s=5)


@pytest.mark.asyncio
async def test_classify_defers_below_min_confidence(monkeypatch):
    X, doms, cplxs = _synth()
    head = EC.train_head(X, doms, cplxs, embed_model="test-embed")

    # An ambiguous vector (near the decision boundary) → low confidence → None (defer to LLM).
    async def fake_embed(_backend, _texts):
        return np.zeros((1, 24), dtype=np.float32)

    monkeypatch.setattr(EC, "embed_texts", fake_embed)
    result = await EC.classify(head, _backend(), prompt="anything", min_confidence=0.95)
    assert result is None


@pytest.mark.asyncio
async def test_classify_returns_result_when_confident(monkeypatch):
    X, doms, cplxs = _synth()
    head = EC.train_head(X, doms, cplxs, embed_model="test-embed")

    async def fake_embed(_backend, _texts):
        v = np.zeros((1, 24), dtype=np.float32)
        v[0, 0] = 3.0   # strongly code
        v[0, 1] = -3.0  # strongly trivial
        return v

    monkeypatch.setattr(EC, "embed_texts", fake_embed)
    result = await EC.classify(head, _backend(), prompt="x", min_confidence=0.5)
    assert result is not None
    assert result.domain == "code"
    assert result.complexity == "trivial"
    assert "embedding head" in result.reason


# --------------------------------------------------------------- context-window resilience
class _CtxError(Exception):
    """Stands in for what an OpenAI-compatible embedder returns when input overflows."""

    def __init__(self, n: int):
        super().__init__(
            f"This model's maximum context length is 2048 tokens. However, you requested "
            f"0 output tokens and your prompt contains at least {n} input tokens."
        )


def _fake_raw(char_limit: int, calls: list | None = None):
    """A _embed_raw stand-in that refuses any batch containing an over-long text."""

    async def raw(_backend, texts):
        if calls is not None:
            calls.append(list(texts))
        if any(len(t) > char_limit for t in texts):
            raise _CtxError(max(len(t) for t in texts))
        return np.ones((len(texts), 4), dtype=np.float32)

    return raw


def test_is_context_error_matches_backend_wording():
    assert EC._is_context_error(_CtxError(2049))
    assert not EC._is_context_error(ConnectionError("connection refused"))


@pytest.mark.asyncio
async def test_embed_texts_shrinks_an_oversize_text(monkeypatch):
    monkeypatch.setattr(EC, "_embed_raw", _fake_raw(1000))
    shrunk: list[tuple[int, int]] = []
    X = await EC.embed_texts(_backend(), ["x" * 8000], on_shrink=lambda o, k: shrunk.append((o, k)))
    assert X.shape == (1, 4)
    assert shrunk and shrunk[0][0] == 8000
    assert shrunk[0][1] <= 1000          # halved until the backend accepted it


@pytest.mark.asyncio
async def test_embed_texts_one_bad_row_does_not_fail_the_batch(monkeypatch):
    monkeypatch.setattr(EC, "_embed_raw", _fake_raw(1000))
    texts = ["fits", "x" * 8000, "also fits"]
    X = await EC.embed_texts(_backend(), texts)
    assert X.shape == (3, 4)             # every row still gets a vector


@pytest.mark.asyncio
async def test_embed_texts_passes_short_batches_straight_through(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(EC, "_embed_raw", _fake_raw(1000, calls))
    X = await EC.embed_texts(_backend(), ["a", "b"])
    assert X.shape == (2, 4)
    assert calls == [["a", "b"]]         # no per-text fallback when nothing overflows


@pytest.mark.asyncio
async def test_embed_texts_reraises_non_context_errors(monkeypatch):
    async def raw(_backend, _texts):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(EC, "_embed_raw", raw)
    with pytest.raises(ConnectionError):
        await EC.embed_texts(_backend(), ["anything"])


@pytest.mark.asyncio
async def test_embed_texts_gives_up_below_the_floor(monkeypatch):
    monkeypatch.setattr(EC, "_embed_raw", _fake_raw(10))   # nothing realistic ever fits
    with pytest.raises(_CtxError):        # surfaces the backend's refusal rather than looping
        await EC.embed_texts(_backend(), ["x" * 8000])
