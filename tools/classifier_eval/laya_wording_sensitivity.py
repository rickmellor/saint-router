# Laya (convaiinnovations/laya) evaluation, 2026-09-26 — see infrastructure/saint/README.md "Laya". Needs: pip install "laya[serve]" in a torch env; the label files live outside the repo.
import sys, random, collections, json, time
sys.path.insert(0, "/home/rick/repos/saint-router/tools/classifier_eval")
import numpy as np, torch, laya
from train_ab import CX, load, report
random.seed(7)
rows = load("/home/rick/repos/saint-router/tools/classifier_eval/relabel-v3-final.jsonl") + load("/home/rick/repos/saint-router/tools/classifier_eval/relabel-v3-gold.jsonl")
random.shuffle(rows); T = [r["prompt"] for r in rows]; y = np.array([CX.index(r["cx"]) for r in rows]); n = len(rows); H = 2
a = laya.load("/home/rick/.config/saint/classifier_eval/laya-ft/laya-saint-v3", device="cuda")
Q_TRAIN = json.load(open("/home/rick/.config/saint/classifier_eval/laya-ft/laya-saint-v3/saint-training.json"))["question"]
Q_JEV = {"complexity": {"type": "choice", "instructions": "How demanding is the user's message for an AI assistant to answer well?",
    "criteria": {"trivial": "A greeting, acknowledgement, short factual lookup, simple rephrase, or a one-line edit.",
        "medium": "A routine, well-specified task: write or fix a function, explain a concept, summarise a text, answer a multi-part question. A 30B-class local model handles it.",
        "hard": "Needs deep multi-step reasoning, design or architecture trade-offs, subtle debugging across components, long-horizon planning, or expert judgment."}}}
Q_BARE = {"complexity": {"type": "choice", "instructions": "Classify.", "criteria": {"trivial": "", "medium": "", "hard": ""}}}
Q_SWAP = {"complexity": {"type": "choice", "instructions": Q_TRAIN["complexity"]["instructions"], "criteria": {"hard": Q_TRAIN["complexity"]["criteria"]["hard"], "trivial": Q_TRAIN["complexity"]["criteria"]["trivial"], "medium": Q_TRAIN["complexity"]["criteria"]["medium"]}}}
# rubric tweak without retraining: move 'explain a concept' and 'answer a substantive question' from medium to hard
Q_TWEAK = {"complexity": {"type": "choice", "instructions": Q_TRAIN["complexity"]["instructions"], "criteria": {
    "trivial": Q_TRAIN["complexity"]["criteria"]["trivial"],
    "medium": "manual work: implement, fix, debug, refactor, test, run commands, follow listed steps, read and report, summarize, extract, translate, reformat",
    "hard": "thinking work: ideation or brainstorming, designing an architecture or plan, choosing between approaches with trade-offs, reviewing or auditing finished work, orchestrating sub-agents, explaining a concept, answering a substantive question"}}}
def run(name, Q):
    t0 = time.monotonic(); P = np.zeros((n, 3), dtype=np.float32); outs = []
    with torch.no_grad():
        for c in range(0, n, 128): outs += a.predict_batch(T[c:c+128], Q, batch_size=16)
    for i, o in enumerate(outs):
        for k, v in o["answers"]["complexity"]["probabilities"].items(): P[i, CX.index(k)] = v
    report(name, P, y, t0); pred = P.argmax(1); print(f"      pred dist {dict(collections.Counter(CX[p] for p in pred))}", flush=True); return pred
base = run("fine-tuned, training wording", Q_TRAIN)
p = run("fine-tuned, old-Jev difficulty wording", Q_JEV); print(f"      changed vs training wording: {np.mean(p != base):.1%}")
p = run("fine-tuned, empty descriptions", Q_BARE); print(f"      changed vs training wording: {np.mean(p != base):.1%}")
p = run("fine-tuned, options reordered (hard first)", Q_SWAP); print(f"      changed vs training wording: {np.mean(p != base):.1%}")
p = run("fine-tuned, rubric tweak: explain/answer -> hard", Q_TWEAK); print(f"      changed vs training wording: {np.mean(p != base):.1%}  now hard: {int((p == H).sum())} (was {int((base == H).sum())})")
print("done")
