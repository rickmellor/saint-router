# Laya (convaiinnovations/laya) evaluation, 2026-09-26 — see infrastructure/saint/README.md "Laya". Needs: pip install "laya[serve]" in a torch env; the label files live outside the repo.
"""Zero-shot Laya + frozen-encoder head on the rubric-v3 A/B set (same rows/metric as train_ab.py)."""
import sys, time, random, collections, json
sys.path.insert(0, "/home/rick/repos/saint-router/tools/classifier_eval")
import numpy as np, torch
torch.set_num_threads(4)
import train_ab
from train_ab import CX, load, cv, fit_softmax, report

random.seed(7); torch.manual_seed(7)
rows = load("/home/rick/repos/saint-router/tools/classifier_eval/relabel-v3-final.jsonl") + load("/home/rick/repos/saint-router/tools/classifier_eval/relabel-v3-gold.jsonl")
random.shuffle(rows)
T = [r["prompt"] for r in rows]; y = np.array([CX.index(r["cx"]) for r in rows]); n = len(rows); H = CX.index("hard")
print(f"{n} prompts: {dict(collections.Counter(r['cx'] for r in rows))}", flush=True)

from laya import Router
router = Router(device="cuda", max_loaded=1)
t0 = time.monotonic(); router.preload(["english"]); print(f"english checkpoint loaded in {time.monotonic()-t0:.1f}s; cuda mem {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

agent = None
for cand in ("_agents", "agents", "_loaded", "_cache"):
    d = getattr(router, cand, None)
    if isinstance(d, dict) and d: agent = list(d.values())[0]; break
if agent is None: agent = router.load("english")
print("agent:", type(agent).__name__, flush=True)

Q_V3 = {"complexity": {"type": "choice",
    "instructions": "Who should do this work: is the user asking for THINKING work (cloud) or MANUAL work (local)? Classify the kind of work, not its difficulty.",
    "criteria": {
        "hard": "thinking work: ideation or brainstorming, designing an architecture or plan, choosing between approaches with trade-offs, reviewing or auditing finished work, orchestrating sub-agents",
        "medium": "manual work (the default): implement, fix, debug, refactor, test, run commands, follow listed steps, read and report, summarize, extract, translate, reformat, explain, answer a question",
        "trivial": "a greeting, acknowledgement like ok or thanks, one-line factual lookup, or a statement with nothing to do"}}}
Q_BIN = {"route": {"type": "choice",
    "instructions": "Should this request go to the cloud frontier model or stay on the strong local fleet? Cloud only for thinking work: ideation, architecture, validation, orchestration. Everything else is local, however difficult.",
    "criteria": {
        "cloud": "brainstorming, designing a system or plan, weighing approaches, reviewing or critiquing work, planning delegated multi-agent work",
        "local": "implementing, debugging, refactoring, running, following steps, summarizing, extracting, explaining, answering, greetings"}}}
Q_JEV = {"complexity": {"type": "choice",
    "instructions": "How demanding is the user's message for an AI assistant to answer well?",
    "criteria": {
        "trivial": "A greeting, acknowledgement, short factual lookup, simple rephrase, or a one-line edit.",
        "medium": "A routine, well-specified task: write or fix a function, explain a concept, summarise a text, answer a multi-part question. A 30B-class local model handles it.",
        "hard": "Needs deep multi-step reasoning, design or architecture trade-offs, subtle debugging across components, long-horizon planning, or expert judgment."}}}

def run(name, Q, key, to_idx, max_len=None, idx=None):
    idx = np.arange(n) if idx is None else idx; TT = [T[i] for i in idx]; yy = y[idx]; m_ = len(idx)
    t0 = time.monotonic(); P = np.zeros((m_, 3), dtype=np.float32); conf = np.zeros(m_); ans_conf = np.zeros(m_)
    outs = []
    for c in range(0, m_, 128):
        outs += agent.predict_batch(TT[c:c+128], Q, batch_size=16, max_len=max_len)
        if c % 512 == 0: print(f"      {name}: {c+128}/{m_} dev={agent.device} {time.monotonic()-t0:.0f}s", flush=True)
    for i, o in enumerate(outs):
        a = o["answers"][key]; pr = a["probabilities"]
        for k, v in pr.items(): P[i, to_idx[k]] += v
        conf[i] = a["confidence"]; ans_conf[i] = a["answer_confidence"]
    report(name, P, yy, t0)
    pred = P.argmax(1); ok = (pred == H) == (yy == H)
    for g in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
        m = ans_conf >= g
        print(f"      gate answer_confidence>={g:.2f}: coverage {m.mean():5.1%}  routing acc on covered {ok[m].mean() if m.any() else float('nan'):6.1%}", flush=True)
    print(f"      pred dist {dict(collections.Counter(CX[p] for p in pred))}  mean conf {conf.mean():.3f}  mean answer_conf {ans_conf.mean():.3f}", flush=True)
    return P, ans_conf

three = {"trivial": 0, "medium": 1, "hard": 2}
rng = np.random.default_rng(7); sub = np.concatenate([rng.choice(np.where(y == k)[0], 200, replace=False) for k in range(3)]); sub.sort()
print(f"subsample for wording variants: {len(sub)} prompts, 200 per class", flush=True)
P_full, C_full = run("Laya ZS rubric-v3 3-way (512, all)", Q_V3, "complexity", three)
run("Laya ZS rubric-v3 3-way (512, all)", Q_V3, "complexity", three)
run("Laya ZS rubric-v3 3-way (1024, all)", Q_V3, "complexity", three, max_len=1024)
run("Laya ZS binary cloud/local (512, all)", Q_BIN, "route", {"cloud": 2, "local": 1})
run("Laya ZS old-Jev difficulty (512, all)", Q_JEV, "complexity", three)
np.savez("laya_zs_full.npz", P=P_full, conf=C_full, y=y)

# single-request latency, warm
lat = []
for t in T[:200]:
    t0 = time.perf_counter(); agent.predict(t, Q_V3); lat.append((time.perf_counter() - t0) * 1000)
lat = np.array(lat); print(f"single-request latency (200 warm, cuda): p50 {np.percentile(lat,50):.1f} ms  p90 {np.percentile(lat,90):.1f}  p99 {np.percentile(lat,99):.1f}", flush=True)

# frozen Laya encoder (ModernBERT-large) mean-pooled embeddings + the same 5-fold softmax head as the nomic head
enc = agent.model.encoder; tok = agent.tok; dev = agent.device
E = []
with torch.no_grad():
    for i in range(0, n, 8):
        b = tok([t[:6000] for t in T[i:i+8]], return_tensors="pt", truncation=True, max_length=512, padding=True).to(dev)
        h = enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state.float()
        m = b["attention_mask"].unsqueeze(-1).float(); E.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
E = np.concatenate(E); E = E / np.linalg.norm(E, axis=1, keepdims=True)
t0 = time.monotonic(); P = np.zeros((n, 3), dtype=np.float32)
for tr, te in cv(n, 5): P[te] = fit_softmax(E[tr], y[tr], 3)(E[te])
report("Laya encoder (frozen) + head 5-fold", P, y, t0)
L = train_ab.lexical(T); t0 = time.monotonic(); P = np.zeros((n, 3), dtype=np.float32); X = np.concatenate([E, L], 1)
for tr, te in cv(n, 5): P[te] = fit_softmax(X[tr], y[tr], 3)(X[te])
report("Laya encoder + lexical 5-fold", P, y, t0)
print("done", flush=True)
