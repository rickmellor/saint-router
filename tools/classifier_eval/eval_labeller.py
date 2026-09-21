"""How good and how fast is a backend as the LLM labeller? Uses SAINT's own classify() + live prompt template.
usage: eval_labeller.py <backend-name> [--limit N] [--concurrency 4]"""
from __future__ import annotations

import argparse, asyncio, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import adjudicated, gold, live_policy, route, traffic  # noqa: E402

from saint.classifier import classify, load_prompt_template  # noqa: E402
from saint.config import load_config  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("backend"); ap.add_argument("--limit", type=int); ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=None); ap.add_argument("--template", default=None, help="rubric file (default: the live classifier.prompt)")
    ap.add_argument("--sets", default=None, help="comma list of set names to run"); a = ap.parse_args()
    from pathlib import Path as _P; cfg = load_config(_P("~/.config/saint/config.toml").expanduser()); be = cfg.backends[a.backend]; tpl = load_prompt_template(a.template or cfg.classifier.prompt_template_path)
    pol = live_policy(); cap = cfg.classifier.max_input_chars
    sets = {"gold-curated-300": [r for r in gold() if r["src"] == "seed_prompts.jsonl"],
            "gold-domains-901": [r for r in gold() if r["src"] != "seed_prompts.jsonl"],
            "traffic-haiku-labelled": traffic("cloud-small"),
            "adjudicated-90": adjudicated()}
    if a.sets: sets = {k: v for k, v in sets.items() if k in a.sets.split(",")}
    sem = asyncio.Semaphore(a.concurrency); detail = {}
    for name, rows in sets.items():
        rows = rows[: a.limit] if a.limit else rows
        async def one(r):
            async with sem:
                t0 = time.monotonic()
                try:
                    res = await classify(be, prompt=r["prompt"][:cap], template=tpl)
                    return r, res.domain, res.complexity, time.monotonic() - t0, None
                except Exception as e:  # noqa: BLE001
                    return r, None, None, time.monotonic() - t0, str(e)[:120]
        t0 = time.monotonic(); got = await asyncio.gather(*(one(r) for r in rows)); wall = time.monotonic() - t0
        ok = [g for g in got if g[4] is None]; n = len(ok) or 1
        dom = sum(g[1] == g[0]["domain"] for g in ok) / n; cx = sum(g[2] == g[0]["complexity"] for g in ok) / n
        both = sum(g[1] == g[0]["domain"] and g[2] == g[0]["complexity"] for g in ok) / n
        rt = sum(route(pol, g[1], g[2]) == route(pol, g[0]["domain"], g[0]["complexity"]) for g in ok) / n
        lat = sorted(g[3] for g in ok)
        conf = {}
        for g in ok:
            if g[2] != g[0]["complexity"]:
                k = f"{g[0]['complexity']}->{g[2]}"; conf[k] = conf.get(k, 0) + 1
        print(f"{name:26s} n={len(rows):4d} errors={len(got)-len(ok):3d} | domain {dom:6.1%} complexity {cx:6.1%} both {both:6.1%} ROUTING {rt:6.1%} | "
              f"latency p50 {statistics.median(lat)*1000:5.0f} ms p95 {lat[int(len(lat)*.95)-1]*1000:5.0f} ms | {len(rows)/wall:4.1f} req/s")
        print(f"   complexity confusions (reference->labeller): {dict(sorted(conf.items(), key=lambda x: -x[1]))}")
        detail[name] = [{"prompt": g[0]["prompt"][:200], "ref": [g[0]["domain"], g[0]["complexity"]], "got": [g[1], g[2]], "err": g[4]} for g in got]
    if a.out: Path(a.out).write_text(json.dumps(detail, indent=1))


asyncio.run(main())
