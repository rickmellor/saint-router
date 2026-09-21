"""Shared data loading for the offline classifier evaluation (read-only on the live log)."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG = Path(os.environ.get("SAINT_LOG", "~/.config/saint/log.sqlite")).expanduser()
CELLS = [(d, c) for d in ("code", "general") for c in ("trivial", "medium", "hard")]


def gold() -> list[dict]:
    """Author-intended labels: the curated 300 + the 901 domain prompts."""
    rows = []
    for name in ("seed_prompts.jsonl", "seed_prompts_domains.jsonl"):
        for line in (REPO / "tools" / name).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append({"prompt": r["prompt"], "domain": r["domain"], "complexity": r["complexity"], "src": name})
    return rows


def traffic(labeller: str = "cloud-small") -> list[dict]:
    """Distinct real-traffic prompts with labels from one LLM labeller (newest label wins)."""
    seeds = {r["prompt"] for r in gold()}
    con = sqlite3.connect(f"file:{LOG}?mode=ro", uri=True)
    out: dict[str, dict] = {}
    for p, d, c in con.execute(
            "SELECT prompt_content, classifier_domain, classifier_complexity FROM requests "
            "WHERE prompt_content IS NOT NULL AND classifier_domain IS NOT NULL AND classifier_complexity IS NOT NULL "
            "AND classifier_used = ? ORDER BY id DESC", (labeller,)):
        if p not in out and p not in seeds and p.strip():
            out[p] = {"prompt": p, "domain": d, "complexity": c, "src": labeller}
    return list(out.values())


def route(policy: dict, domain: str, complexity: str) -> str:
    return policy[f"{domain},{complexity}"]


def live_policy() -> dict:
    import tomllib
    cfg = tomllib.loads(Path("~/.config/saint/config.toml").expanduser().read_text())
    return cfg["routing"]["policy"]["normal"]


def adjudicated() -> list[dict]:
    """Real-traffic sample hand-labelled under the kind-of-work rubric (classifier.prompt.v3)."""
    return [json.loads(l) for l in (LOG.parent / "classifier_eval" / "adjudicated.jsonl").read_text().splitlines() if l.strip()]


async def label_direct(client, base_url: str, model: str, prompt: str, template: str, *, max_tokens: int = 160) -> list[str]:
    """One label straight from an OpenAI-compatible seat with thinking OFF (SAINT's classify() leaves it on: ~6 s on Flash-Next)."""
    r = await client.post(f"{base_url}/chat/completions", json={
        "model": model, "messages": [{"role": "user", "content": template.replace("{prompt}", prompt)}],
        "max_tokens": max_tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    r.raise_for_status(); txt = r.json()["choices"][0]["message"]["content"].strip()
    txt = txt[txt.find("{"): txt.rfind("}") + 1]; o = json.loads(txt)
    return [o["domain"], o["complexity"]]
