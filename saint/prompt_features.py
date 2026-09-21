"""Optional extra inputs for the embedding head: cheap lexical cues computed here, plus task/complexity
probabilities from a local `saint-features` sidecar (tools/prompt_features/server.py). The head records which
spec it was trained with; if the sidecar is unreachable the head abstains and the LLM classifier answers."""
from __future__ import annotations

import math
import re

import httpx
import numpy as np

SPEC = "nvidia32+lex9"
NVIDIA_DIM = 32

_LEX = [re.compile(p, re.I | re.M) for p in (
    r"\b(design|architect\w*|trade-?offs?|approach|strategy|should (we|i)|options?)\b",
    r"\b(review|audit|validate|critique|evaluate|sound\w*|sanity)\b",
    r"\b(plan|roadmap|orchestrat\w*|sub-?agents?|delegate|coordinate|in parallel)\b",
    r"\b(brainstorm|ideas?|what if|explore|consider\w*|think about)\b",
    r"\b(write|implement|fix|debug|refactor|run|add|create|build|update|install)\b",
    r"\b(summar\w+|extract|translate|reformat|compact|list|report)\b",
    r"^\s*\d+[.)] ",
    r"\?",
)]


def lexical(text: str) -> np.ndarray:
    t = text[:4000]
    return np.array([min(len(p.findall(t)), 5) / 5 for p in _LEX] + [min(math.log1p(len(text)) / 10, 1.5)], dtype=np.float64)


async def sidecar(url: str, texts: list[str], timeout_s: float = 2.0) -> np.ndarray:
    async with httpx.AsyncClient(timeout=timeout_s) as c:
        r = await c.post(url.rstrip("/") + "/features", json={"input": texts})
        r.raise_for_status()
        return np.asarray(r.json()["features"], dtype=np.float64)


async def extras(url: str, texts: list[str], timeout_s: float = 2.0) -> np.ndarray:
    """(n, 41) — sidecar probabilities then lexical cues, in SPEC order."""
    out = []
    for i in range(0, len(texts), 16):
        out.append(await sidecar(url, texts[i:i + 16], timeout_s if len(texts) == 1 else 60.0))
    return np.hstack([np.vstack(out), np.vstack([lexical(t) for t in texts])])
