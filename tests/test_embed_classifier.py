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
