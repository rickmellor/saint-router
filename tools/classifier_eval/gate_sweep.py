"""Coverage vs routing accuracy of the feature-augmented head at different min_confidence gates (5-fold CV).
Below the gate the request defers to the LLM labeller, so what matters is accuracy on what the head KEEPS."""
import collections, random, sys
import httpx, numpy as np, torch
import train_ab as T

random.seed(7); torch.manual_seed(7)
rows = [r for f in sys.argv[1:] for r in T.load(f)]; random.shuffle(rows); P = [r["prompt"] for r in rows]; y = np.array([T.CX.index(r["cx"]) for r in rows]); n = len(rows)
E = T.embed(P); N = np.vstack([np.array(httpx.post("http://127.0.0.1:8005/features", json={"input": P[i:i + 16]}, timeout=60).json()["features"], dtype=np.float32) for i in range(0, n, 16)])
X = np.concatenate([E, N, T.lexical(P)], 1); prob = np.zeros((n, 3), dtype=np.float32)
for tr, te in T.cv(n, 5): prob[te] = T.fit_softmax(X[tr], y[tr], 3)(X[te])
pred, conf, h = prob.argmax(1), prob.max(1), T.CX.index("hard")
for g in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
    k = conf >= g; ok = (pred[k] == h) == (y[k] == h)
    print(f"min_confidence {g:.1f}: head answers {k.mean():6.1%} | routing right on those {ok.mean():6.1%} | local→cloud {int(((pred[k] == h) & (y[k] != h)).sum()):3d}  cloud→local {int(((pred[k] != h) & (y[k] == h)).sum()):3d}  (of {int(k.sum())})")
