#!/usr/bin/env python
"""Seed SAINT's request log with classifier-labeled training rows.

Reads a JSONL dataset of {"prompt", "domain", "complexity"} where domain/complexity
are the *intended* labels, runs each prompt through the real routing pipeline
(decide_route -> the configured LLM classifier), logs each classification to the
request log exactly like `saint explain` does, and reports agreement between the
classifier's labels and the intended ones.

The logged rows carry the CLASSIFIER's labels (that's what `saint classifier train`
distills), so treat the agreement report as a quality gate: if it's poor, fix the
classifier prompt, `saint log clear --yes`, and re-run this script.

Usage:
    python tools/seed_classifier.py tools/seed_prompts.jsonl
        [--config PATH] [--concurrency N] [--limit N] [--no-log] [--out results.jsonl]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from saint.config import load_config
from saint.router import decide_route
from saint.storage import build_log_row, log_request, open_db

DEFAULT_CONFIG = Path("~/.config/saint/config.toml").expanduser()
DOMAINS = ("code", "general")
COMPLEXITIES = ("trivial", "medium", "hard")


async def amain() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="JSONL with prompt/domain/complexity per line")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N prompts")
    ap.add_argument("--no-log", action="store_true", help="Validate only; don't write log rows")
    ap.add_argument("--out", type=Path, default=None, help="Write per-prompt results JSONL here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    items = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]
    conn = None if args.no_log else open_db(Path(cfg.logging.db_path))

    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def classify(item: dict) -> dict:
        nonlocal done
        res = {**item, "got_domain": None, "got_complexity": None, "used": None, "error": None}
        try:
            async with sem:
                decision = await decide_route(
                    cfg=cfg, model_field="saint-explain",
                    messages=[{"role": "user", "content": item["prompt"]}],
                )
            out = decision.classifier_outcome
            if out and out.result:
                res["got_domain"] = out.result.domain
                res["got_complexity"] = out.result.complexity
                res["used"] = out.classifier_used
                if conn is not None:
                    log_request(conn, build_log_row(
                        decision, model_field="saint-explain",
                        backend_latency_ms=None, success=True, error_kind=None,
                        tokens_in=None, tokens_out=None,
                        prompt_storage_mode=cfg.logging.prompt_storage,
                    ))
            else:
                res["error"] = "classifier returned no result"
        except Exception as e:
            res["error"] = f"{type(e).__name__}: {e}"
        done += 1
        if done % 25 == 0 or done == len(items):
            print(f"  {done}/{len(items)}", file=sys.stderr, flush=True)
        return res

    print(f"classifying {len(items)} prompts "
          f"(concurrency={args.concurrency}, logging={'off' if args.no_log else 'on'})…",
          file=sys.stderr, flush=True)
    results = await asyncio.gather(*(classify(it) for it in items))

    if args.out:
        args.out.write_text("".join(json.dumps(r) + "\n" for r in results))

    ok = [r for r in results if r["error"] is None]
    errs = [r for r in results if r["error"] is not None]
    dom_hits = sum(1 for r in ok if r["got_domain"] == r["domain"])
    cpx_hits = sum(1 for r in ok if r["got_complexity"] == r["complexity"])
    both_hits = sum(1 for r in ok if r["got_domain"] == r["domain"]
                    and r["got_complexity"] == r["complexity"])
    fallbacks = sum(1 for r in ok if r["used"] and r["used"] != cfg.classifier.backend)

    print(f"\nclassified: {len(ok)}/{len(results)}   errors: {len(errs)}   "
          f"fallback-classified: {fallbacks}")
    if ok:
        print(f"domain accuracy:     {dom_hits}/{len(ok)} ({100 * dom_hits / len(ok):.0f}%)")
        print(f"complexity accuracy: {cpx_hits}/{len(ok)} ({100 * cpx_hits / len(ok):.0f}%)")
        print(f"both exact:          {both_hits}/{len(ok)} ({100 * both_hits / len(ok):.0f}%)")

    # per-cell breakdown: intended cell -> counter of got cells
    print("\nintended cell        -> classifier said (count)")
    for d in DOMAINS:
        for c in COMPLEXITIES:
            cell = [r for r in ok if r["domain"] == d and r["complexity"] == c]
            if not cell:
                continue
            got = Counter(f"{r['got_domain']},{r['got_complexity']}" for r in cell)
            parts = "  ".join(f"{k}:{v}" for k, v in got.most_common())
            print(f"  {d},{c:<10} n={len(cell):<3} {parts}")

    disagreements = [r for r in ok if r["got_domain"] != r["domain"]
                     or r["got_complexity"] != r["complexity"]]
    if disagreements:
        print(f"\ndisagreements ({len(disagreements)}):")
        for r in disagreements:
            print(f"  [{r['domain']},{r['complexity']} -> {r['got_domain']},{r['got_complexity']}] "
                  f"{r['prompt'][:90]}")
    for r in errs:
        print(f"  ERROR: {r['error']}  {r['prompt'][:70]}", file=sys.stderr)

    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
