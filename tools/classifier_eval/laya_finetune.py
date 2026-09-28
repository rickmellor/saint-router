"""Schema-preserving Laya fine-tune: keep the DecisionModel + request-time question, train on rubric-v3 labels with Laya's proper-scoring reward.
Variants: head-only (encoder detached) and full. 3-fold CV through agent.predict_batch, then a final all-data model saved for SAINT."""
import sys, os, time, random, collections, copy, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="head"); ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--head-lr", type=float, default=3e-4); ap.add_argument("--enc-lr", type=float, default=2e-5); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--final", action="store_true"); ap.add_argument("--tag", default="v3"); ap.add_argument("--max-len", type=int, default=512); ap.add_argument("--shuffle-options", action="store_true"); ap.add_argument("--split", default="cv3", help="cv3 | gold2traffic"); ap.add_argument("--head-max-len", type=int, default=192); ap.add_argument("--no-cv", action="store_true"); ap.add_argument("--questions", default="complexity", help="complexity | both (domain+complexity in one forward)")
ARGS = ap.parse_args()
sys.path.insert(0, "/home/rick/repos/saint-router/tools/classifier_eval")
import numpy as np, torch
from train_ab import CX, load, cv, report
from laya.common import collate_items, proper_reward, build_sequence, QTYPES
random.seed(7); torch.manual_seed(7)
rows = [dict(r, src="traffic") for r in load("/home/rick/repos/saint-router/tools/classifier_eval/relabel-v3-final.jsonl")] + [dict(r, src="gold") for r in load("/home/rick/repos/saint-router/tools/classifier_eval/relabel-v3-gold.jsonl")]
random.shuffle(rows); T = [r["prompt"] for r in rows]; y = np.array([CX.index(r["cx"]) for r in rows]); n = len(rows); H = 2; SRC = np.array([r["src"] for r in rows])
DOM = ["code", "general"]; yd = np.array([DOM.index(r["domain"]) for r in rows])
print(f"{n} prompts {dict(collections.Counter(r['cx'] for r in rows))}", flush=True)
from laya import Router
router = Router(device="cuda", max_loaded=1); router.preload(["english"]); agent = router._agents[list(router._agents)[0]]
model, tok, dev = agent.model, agent.tok, agent.device
agent.deaccelerate() if hasattr(agent, "deaccelerate") else None
Q = {"complexity": {"type": "choice",
    "instructions": "Who should do this work: is the user asking for THINKING work (cloud) or MANUAL work (local)? Classify the kind of work, not its difficulty.",
    "criteria": {
        "trivial": "a greeting, acknowledgement like ok or thanks, one-line factual lookup, or a statement with nothing to do",
        "medium": "manual work (the default): implement, fix, debug, refactor, test, run commands, follow listed steps, read and report, summarize, extract, translate, reformat, explain, answer a question",
        "hard": "thinking work: ideation or brainstorming, designing an architecture or plan, choosing between approaches with trade-offs, reviewing or auditing finished work, orchestrating sub-agents"}}}
# criteria keys in CX order (trivial, medium, hard) so option index == label index
Q["domain"] = {"type": "choice", "instructions": "What kind of work is the user's message about?", "criteria": {"code": "source code, shell or ops commands, configuration, repositories, builds, debugging, software tools", "general": "everything else: questions, writing, analysis, planning, math, conversation, summaries, advice with no programming artifact"}}
if ARGS.questions != "both": Q.pop("domain")
internal = {k: agent._to_internal(v) for k, v in Q.items()}; IDS = list(Q)
init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
pad = tok.pad_token_id

def encode(prompts, labels=None, shuffle=False, dlabels=None):
    """One row per (prompt, question). Row order: for each prompt, complexity then domain (if asked)."""
    items = []; lab_pos = []
    for i, p in enumerate(prompts):
        group = []
        for qid in IDS:
            q = internal[qid]; k = len(q["crit"]); order = list(range(k))
            if shuffle: random.shuffle(order)
            seq, markers = build_sequence(tok, p, q, ARGS.max_len, ARGS.head_max_len, option_order=order)
            group.append({"ids": seq, "markers": markers, "qtype": QTYPES["choice"]})
            if labels is not None: lab_pos.append(order.index(int(labels[i] if qid == "complexity" else dlabels[i])))
        items.append(group)
    return collate_items(items, pad), (torch.tensor(lab_pos, device=dev) if labels is not None else None)

def train(prompts, labels, head_only, epochs=None, bs=None, yd_tr=None):
    yd_tr = np.zeros(len(labels), dtype=int) if yd_tr is None else yd_tr
    epochs = ARGS.epochs if epochs is None else epochs; bs = ARGS.bs if bs is None else bs
    model.load_state_dict(init_state); model.to(dev).float(); model.train()
    for p in model.encoder.parameters(): p.requires_grad_(not head_only)
    enc_params = [p for p in model.encoder.parameters()]; head_params = [p for nm, p in model.named_parameters() if not nm.startswith("encoder.")]
    groups = [{"params": head_params, "lr": ARGS.head_lr}] + ([] if head_only else [{"params": enc_params, "lr": ARGS.enc_lr}])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = epochs * ((len(prompts) + bs - 1) // bs); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, steps // 20)) * max(0.0, 1 - s / steps))
    cw = torch.tensor([len(labels) / (3 * max((labels == k).sum(), 1)) for k in range(3)], dtype=torch.float32, device=dev)
    t0 = time.monotonic()
    for ep in range(epochs):
        idx = list(range(len(prompts))); random.shuffle(idx); tot = 0.0; nb = 0
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; batch, lab = encode([prompts[j] for j in b], labels[b], ARGS.shuffle_options, yd_tr[b]); cls = torch.tensor(np.repeat(labels[b], len(IDS)), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev), batch["marker_pos"].to(dev), batch["marker_mask"].to(dev), batch["qtype"].to(dev), detach_encoder=head_only)
            q = torch.softmax(logits.float(), -1); target = torch.nn.functional.one_hot(lab, q.shape[1]).float()
            r = proper_reward(q, target, batch["qtype"].to(dev), batch["marker_mask"].to(dev))
            wrow = cw[cls].clone()
            if len(IDS) > 1: wrow[1::2] = 1.0
            loss = -(r * wrow).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0); opt.step(); sched.step(); opt.zero_grad(); tot += loss.item(); nb += 1
        print(f"    epoch {ep+1}/{epochs} loss {tot / max(nb, 1):.3f} {time.monotonic() - t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
    model.eval()

def evaluate(prompts):
    P = np.zeros((len(prompts), 3), dtype=np.float32); ac = np.zeros(len(prompts)); PD = np.zeros((len(prompts), 2), dtype=np.float32)
    with torch.no_grad():
        outs = []
        for c in range(0, len(prompts), 128): outs += agent.predict_batch(prompts[c:c+128], Q, batch_size=16, max_len=ARGS.max_len, head_max_len=ARGS.head_max_len)
    for i, o in enumerate(outs):
        a = o["answers"]["complexity"]
        for k, v in a["probabilities"].items(): P[i, CX.index(k)] = v
        ac[i] = a["answer_confidence"]
        if "domain" in o["answers"]:
            for k, v in o["answers"]["domain"]["probabilities"].items(): PD[i, DOM.index(k)] = v
    evaluate.PD = PD
    return P, ac

VARIANTS = {"head": [("Laya head-only (encoder frozen) 3-fold", True)], "full": [("Laya full schema-preserving 3-fold", False)], "both": [("Laya head-only (encoder frozen) 3-fold", True), ("Laya full schema-preserving 3-fold", False)]}
print(f"config: {vars(ARGS)}", flush=True)
for name, head_only in ([] if ARGS.no_cv else VARIANTS[ARGS.variant]):
    name = f"{name} ep{ARGS.epochs} hlr{ARGS.head_lr:g}" + ("" if head_only else f" elr{ARGS.enc_lr:g}")
    t0 = time.monotonic(); P = np.zeros((n, 3), dtype=np.float32); AC = np.zeros(n)
    if ARGS.split == "gold2traffic":
        folds = [(np.where(SRC == "gold")[0], np.where(SRC == "traffic")[0])]; EV = folds[0][1]
    else:
        folds = list(cv(n, 3)); EV = np.arange(n)
    for k, (tr, te) in enumerate(folds):
        print(f"{name}: fold {k+1}/{len(folds)} train {len(tr)} test {len(te)}", flush=True); train([T[i] for i in tr], y[tr], head_only, yd_tr=yd[tr])
        PDall = globals().setdefault("PDall", np.zeros((n, 2), dtype=np.float32))
        P[te], AC[te] = evaluate([T[i] for i in te]); PDall[te] = evaluate.PD; pred = P[te].argmax(1); print(f"    fold routing acc {np.mean((pred == H) == (y[te] == H)):.1%}" + (f"  domain acc {np.mean(PDall[te].argmax(1) == yd[te]):.1%}" if len(IDS) > 1 else ""), flush=True)
    report(name + f" [{ARGS.split}]", P[EV], y[EV], t0)
    if len(IDS) > 1: print(f"      domain accuracy {np.mean(PDall[EV].argmax(1) == yd[EV]):.1%}", flush=True)
    pred = P[EV].argmax(1); ok = (pred == H) == (y[EV] == H); mp = P[EV].max(1); AC = AC[EV]
    for g in (0.6, 0.7, 0.8, 0.9):
        m1 = mp >= g; m2 = AC >= g
        print(f"      gate maxprob>={g:.1f}: coverage {m1.mean():5.1%} acc {ok[m1].mean() if m1.any() else float('nan'):6.1%} | answer_confidence>={g:.1f}: coverage {m2.mean():5.1%} acc {ok[m2].mean() if m2.any() else float('nan'):6.1%}", flush=True)
    np.savez(f"laya_ft3_{ARGS.tag}_{'head' if head_only else 'full'}.npz", P=P, AC=AC, y=y)

if not ARGS.final: print("done", flush=True); sys.exit(0)
print("final: training full model on all prompts", flush=True); train(T, y, ARGS.variant == "head", yd_tr=yd)
out = os.path.expanduser("~/.config/saint/classifier_eval/laya-ft"); os.makedirs(out, exist_ok=True)
torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, f"{out}/laya_saint_{ARGS.tag}.pt")
import json; json.dump({"question": Q, "labels": CX, "trained_on": n, "date": "2026-09-26", "base": "convaiinnovations/laya english 55cf4c4"}, open(f"{out}/laya_saint_{ARGS.tag}.json", "w"), indent=1)
print(f"saved {out}/laya_saint_{ARGS.tag}.pt", flush=True); print("done", flush=True)
