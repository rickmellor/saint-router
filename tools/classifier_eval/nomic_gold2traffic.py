# Laya (convaiinnovations/laya) evaluation, 2026-09-26 — see infrastructure/saint/README.md "Laya". Needs: pip install "laya[serve]" in a torch env; the label files live outside the repo.
"""Nomic head (as deployed: embedding [+ lexical]) trained on gold seeds only, tested on real traffic — the same split as laya_ft3 --split gold2traffic."""
import sys, random, collections, time
sys.path.insert(0, "/home/rick/repos/saint-router/tools/classifier_eval")
import numpy as np, torch, train_ab
from train_ab import CX, load, fit_softmax, report, lexical
train_ab._embed.__globals__["httpx"]  # sanity
random.seed(7); torch.manual_seed(7)
rows = [dict(r, src="traffic") for r in load("/home/rick/repos/saint-router/tools/classifier_eval/relabel-v3-final.jsonl")] + [dict(r, src="gold") for r in load("/home/rick/repos/saint-router/tools/classifier_eval/relabel-v3-gold.jsonl")]
random.shuffle(rows); T = [r["prompt"] for r in rows]; y = np.array([CX.index(r["cx"]) for r in rows]); SRC = np.array([r["src"] for r in rows])
tr, te = np.where(SRC == "gold")[0], np.where(SRC == "traffic")[0]; print(f"train gold {len(tr)} test traffic {len(te)}", flush=True)
import hashlib, pickle; from pathlib import Path
cache = pickle.loads(Path("/home/rick/repos/saint-router/tools/classifier_eval/embed-cache.npz.pkl").read_bytes())
miss = [t for t in T if hashlib.sha1(t.encode()).hexdigest() not in cache]; print(f"embed cache misses: {len(miss)}", flush=True)
if miss:
    import re; src = open(train_ab.__file__).read(); train_ab._embed.__globals__["_EMBED_URL"] = "http://localhost:8005"
    # patch URL by re-defining _embed with :8005
    exec(re.sub(r"localhost:8001", "localhost:8005", src[src.find("def _embed"):src.find("def embed(")]), train_ab.__dict__)
E = train_ab.embed(T); L = lexical(T)
for name, X in (("nomic + head", E), ("nomic + lexical (deployed shape, no NVIDIA feats)", np.concatenate([E, L], 1))):
    t0 = time.monotonic(); P = fit_softmax(X[tr], y[tr], 3)(X[te]); report(name + " [gold2traffic]", P, y[te], t0)
    pred = P.argmax(1); ok = (pred == 2) == (y[te] == 2); mp = P.max(1)
    for g in (0.6, 0.7, 0.8, 0.9): m = mp >= g; print(f"      gate maxprob>={g:.1f}: coverage {m.mean():5.1%} acc {ok[m].mean():6.1%}", flush=True)
# reference: 5-fold on everything, same as train_ab
from train_ab import cv
n = len(T); P = np.zeros((n, 3), dtype=np.float32); t0 = time.monotonic()
for a, b in cv(n, 5): P[b] = fit_softmax(E[a], y[a], 3)(E[b])
report("nomic + head [cv5, reference]", P, y, t0)
pred = P.argmax(1); ok = (pred == 2) == (y == 2); mp = P.max(1)
for g in (0.6, 0.7, 0.8, 0.9): m = mp >= g; print(f"      gate maxprob>={g:.1f}: coverage {m.mean():5.1%} acc {ok[m].mean():6.1%}", flush=True)
print("done", flush=True)
