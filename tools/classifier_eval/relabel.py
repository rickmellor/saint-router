"""Relabel every distinct real-traffic prompt with LOCAL labellers under a rubric; keep both votes.

Usage: relabel.py --template tools/classifier_eval/classifier.prompt.v3 --backends local-chat,local-coder --out relabel.jsonl
Resumable: prompts already in --out are skipped. Machine-generated utility prompts are labelled by rule.
"""
from __future__ import annotations

import argparse, asyncio, json, sqlite3, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).parent))
import httpx  # noqa: E402
from saint.classifier import load_prompt_template  # noqa: E402
from saint.config import load_config  # noqa: E402
from common import LOG, gold, label_direct  # noqa: E402

UTILITY = ("Please analyze the following dialogue and generate extremely concise subtopic", "Determine if these two conversation pages are continuous",
           "Update the conversation meta-summary", "Please analyze the latest user-AI conversation below and update the user profile",
           "Please extract user private data", "Compact the following conversation", '{"user_id"')


def is_utility(p: str) -> bool:
    return p.lstrip().startswith(UTILITY)


def prompts() -> list[dict]:
    seeds = {r["prompt"] for r in gold()}; con = sqlite3.connect(f"file:{LOG}?mode=ro", uri=True); out: dict[str, dict] = {}
    for p, d, c, used, be in con.execute("SELECT prompt_content, classifier_domain, classifier_complexity, classifier_used, backend_chosen FROM requests "
                                         "WHERE prompt_content IS NOT NULL ORDER BY id DESC"):
        if p and p.strip() and p not in seeds:
            o = out.setdefault(p, {"prompt": p, "old": [d, c], "old_by": used, "n": 0, "backends": {}})
            o["n"] += 1; o["backends"][be] = o["backends"].get(be, 0) + 1
    return list(out.values())


async def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--template", required=True); ap.add_argument("--backends", default="local-chat,local-coder")
    ap.add_argument("--out", required=True); ap.add_argument("--concurrency", type=int, default=6); ap.add_argument("--limit", type=int); a = ap.parse_args()
    cfg = load_config(Path("~/.config/saint/config.toml").expanduser()); tpl = load_prompt_template(a.template); cap = cfg.classifier.max_input_chars
    names = a.backends.split(","); out = Path(a.out)
    done = {json.loads(l)["prompt"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in prompts() if r["prompt"] not in done][: a.limit]; print(f"{len(done)} done, {len(todo)} to label", flush=True)
    sem = {b: asyncio.Semaphore(a.concurrency) for b in names}; lock = asyncio.Lock(); n = 0
    client = httpx.AsyncClient(timeout=180); seat = {}
    for b in names:
        base = cfg.backends[b].base_url.rstrip("/"); seat[b] = (base, (await client.get(base + "/models")).json()["data"][0]["id"])

    async def one(r: dict) -> None:
        nonlocal n
        if is_utility(r["prompt"]):
            r["votes"] = {"rule": ["general", "medium"]}
        else:
            async def vote(b: str):
                async with sem[b]:
                    try:
                        return b, await label_direct(client, *seat[b], r["prompt"][:cap], tpl)
                    except Exception as e:  # noqa: BLE001
                        return b, ["error", str(e)[:80]]
            r["votes"] = dict(await asyncio.gather(*(vote(b) for b in names)))
        async with lock:
            with out.open("a") as f: f.write(json.dumps(r) + "\n")
            n += 1
            if n % 100 == 0: print(n, flush=True)
    await asyncio.gather(*(one(r) for r in todo))


asyncio.run(main())
