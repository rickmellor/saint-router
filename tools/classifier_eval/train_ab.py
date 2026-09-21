"""A/B the local classifier candidates on rubric-v3 consensus labels (both local labellers agree).

  A  nomic embedding + softmax head            (what SAINT runs today, retrained)
  B  A + NVIDIA task/complexity probs + cheap lexical features
  C  ModernBERT-base fine-tune (end to end)
Run with a torch env:  ~/.venvs/llmc/bin/python train_ab.py relabel-v3.jsonl [--bert]
"""
from __future__ import annotations

import argparse, collections, json, random, re, sys, time
from pathlib import Path

import httpx
import numpy as np
import torch

CX = ["trivial", "medium", "hard"]
LEX = [r"\b(design|architect\w*|trade-?offs?|approach|strategy|should (we|i)|options?)\b", r"\b(review|audit|validate|critique|evaluate|sound\w*|sanity)\b",
       r"\b(plan|roadmap|orchestrat\w*|sub-?agents?|delegate|coordinate|in parallel)\b", r"\b(brainstorm|ideas?|what if|explore|consider\w*|think about)\b",
       r"\b(write|implement|fix|debug|refactor|run|add|create|build|update|install)\b", r"\b(summar\w+|extract|translate|reformat|compact|list|report)\b", r"^\s*\d+[.)] ", r"\?"]


def load(path: str, cap: int = 25) -> list[dict]:
    rows, fam = [], collections.Counter()
    for l in open(path):
        r = json.loads(l); v = [x for x in r["votes"].values() if x[0] != "error"]
        if not v or any(x != v[0] for x in v): continue          # consensus only
        k = r["prompt"].lstrip()[:40]; fam[k] += 1
        if fam[k] > cap: continue                                # near-identical machine prompts must not dominate
        rows.append({"prompt": r["prompt"], "domain": v[0][0], "cx": v[0][1]})
    return rows


def _embed(texts: list[str]) -> np.ndarray:
    out = []
    with httpx.Client(timeout=120) as c:
        model = c.get("http://localhost:8001/v1/models").json()["data"][0]["id"]

        def one(t: str) -> list[float]:                      # halve until it fits the embedder's context, as SAINT does
            while True:
                r = c.post("http://localhost:8001/v1/embeddings", json={"model": model, "input": [t]})
                if r.status_code == 200: return r.json()["data"][0]["embedding"]
                if len(t) < 200: r.raise_for_status()
                t = t[: len(t) // 2]
        for i in range(0, len(texts), 16):
            r = c.post("http://localhost:8001/v1/embeddings", json={"model": model, "input": [t[:6000] for t in texts[i:i + 16]]})
            out += [d["embedding"] for d in r.json()["data"]] if r.status_code == 200 else [one(t[:6000]) for t in texts[i:i + 16]]
    x = np.array(out, dtype=np.float32); return x / np.linalg.norm(x, axis=1, keepdims=True)


def embed(texts: list[str]) -> np.ndarray:
    """Disk-cached (CPU nomic seat is ~5 prompts/s)."""
    import hashlib, pickle
    cp = Path(__file__).parent / "embed-cache.npz.pkl"; cache = pickle.loads(cp.read_bytes()) if cp.exists() else {}
    key = [hashlib.sha1(t.encode()).hexdigest() for t in texts]; miss = [i for i, k in enumerate(key) if k not in cache]
    if miss:
        for i, v in zip(miss, _embed([texts[i] for i in miss])): cache[key[i]] = v
        cp.write_bytes(pickle.dumps(cache))
    return np.stack([cache[k] for k in key])


def lexical(texts: list[str]) -> np.ndarray:
    f = [[min(len(re.findall(p, t[:4000], re.I | re.M)), 5) / 5 for p in LEX] + [min(np.log1p(len(t)) / 10, 1.5)] for t in texts]
    return np.array(f, dtype=np.float32)


def fit_softmax(X, y, n, epochs=400, wd=1e-3):
    X, y = torch.tensor(X), torch.tensor(y); lin = torch.nn.Linear(X.shape[1], n); opt = torch.optim.AdamW(lin.parameters(), lr=0.05, weight_decay=wd)
    w = torch.tensor([len(y) / (n * max((y == k).sum().item(), 1)) for k in range(n)], dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad(); torch.nn.functional.cross_entropy(lin(X), y, weight=w).backward(); opt.step()
    return lambda Z: torch.softmax(lin(torch.tensor(Z)), -1).detach().numpy()


def report(name, prob, y, t0):
    pred = prob.argmax(1); h = CX.index("hard"); tp = ((pred == h) & (y == h)).sum(); fp = ((pred == h) & (y != h)).sum(); fn = ((pred != h) & (y == h)).sum()
    print(f"{name:34s} complexity {np.mean(pred == y):6.1%} | cloud/local routing {np.mean((pred == h) == (y == h)):6.1%} | hard precision {tp / max(tp + fp, 1):5.1%} recall {tp / max(tp + fn, 1):5.1%}"
          f" | local work sent to cloud {fp:3d}  cloud work kept local {fn:3d} | {time.monotonic() - t0:5.1f}s", flush=True)


def cv(n: int, k: int):
    idx = np.arange(n)
    for f in range(k): yield idx[idx % k != f], idx[idx % k == f]


def bert(train, test, ytr, device="cuda"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = "/mnt/data/models/answerdotai/ModernBERT-base"; tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=3, torch_dtype=torch.bfloat16).to(device); opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.01)
    w = torch.tensor([len(ytr) / (3 * max((ytr == k).sum(), 1)) for k in range(3)], dtype=torch.bfloat16, device=device)
    enc = lambda T: tok([t[:6000] for t in T], return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    for ep in range(4):
        idx = list(range(len(train))); random.shuffle(idx); m.train()
        for i in range(0, len(idx), 4):
            b = idx[i:i + 4]; out = m(**enc([train[j] for j in b])).logits
            loss = torch.nn.functional.cross_entropy(out, torch.tensor(ytr[b], device=device), weight=w); loss.backward(); opt.step(); opt.zero_grad()
    m.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(test), 16): P.append(torch.softmax(m(**enc(test[i:i + 16])).logits.float(), -1).cpu().numpy())
    return np.concatenate(P)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("labels", nargs="+"); ap.add_argument("--bert", action="store_true"); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--folds", type=int, default=5); a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed); rows = [r for f in a.labels for r in load(f)]; random.shuffle(rows)
    T = [r["prompt"] for r in rows]; y = np.array([CX.index(r["cx"]) for r in rows]); n = len(rows)
    print(f"{n} prompts: {dict(collections.Counter(r['cx'] for r in rows))}; {a.folds}-fold cross-validation, every prompt scored once", flush=True)
    t0 = time.monotonic(); E = embed(T); sys.path.insert(0, str(Path(__file__).parent)); import nvidia_features as nv
    cfg, tok, net = nv.load("cuda"); F = []
    with torch.no_grad():
        for i in range(0, n, 16):
            enc = tok([t[:6000] for t in T[i:i + 16]], return_tensors="pt", max_length=512, padding=True, truncation=True).to("cuda")
            F.append(np.concatenate([p.float().cpu().numpy() for p in net(enc["input_ids"], enc["attention_mask"])], 1))
    N = np.concatenate(F); del net; torch.cuda.empty_cache(); L = lexical(T)
    for name, X in (("A nomic + head", E), ("  NVIDIA features only", N), ("  nomic + lexical", np.concatenate([E, L], 1)), ("B nomic + NVIDIA + lexical", np.concatenate([E, N, L], 1))):
        P = np.zeros((n, 3), dtype=np.float32)
        for tr, te in cv(n, a.folds): P[te] = fit_softmax(X[tr], y[tr], 3)(X[te])
        report(name, P, y, t0); t0 = time.monotonic()
    if a.bert:
        P = np.zeros((n, 3), dtype=np.float32)
        for tr, te in cv(n, 3): P[te] = bert([T[i] for i in tr], [T[i] for i in te], y[tr])
        report("C ModernBERT-base fine-tune (3-fold)", P, y, t0)


if __name__ == "__main__":
    main()
