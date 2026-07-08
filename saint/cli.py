from __future__ import annotations

import os
import socket
from pathlib import Path

import typer
import uvicorn

from saint.config import load_config

app = typer.Typer(no_args_is_help=True, help="saint — localhost OpenAI-compatible router")


def _default_config_path() -> Path:
    return Path(os.path.expanduser("~/.config/saint/config.toml"))


def _load_or_die(config_path: Path | None):
    path = config_path or _default_config_path()
    if not path.exists():
        typer.secho(f"Config not found: {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    try:
        return load_config(path)
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Override [server.host]"),
    port: int | None = typer.Option(None, help="Override [server.port]"),
    config: Path | None = typer.Option(None, help="Path to config.toml"),
):
    """Start the proxy server."""
    cfg = _load_or_die(config)
    bind_host = host or cfg.server.host
    bind_port = port or cfg.server.port

    if bind_host not in ("127.0.0.1", "localhost", "::1"):
        typer.secho(
            f"WARNING: binding to {bind_host} (not loopback). "
            "Anyone on this network may reach the proxy.",
            fg=typer.colors.YELLOW, err=True,
        )

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((bind_host, bind_port))
    except OSError:
        typer.secho(
            f"Port {bind_port} in use; is another `saint serve` running?",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(3) from None
    finally:
        s.close()

    from saint.server import build_app
    app_obj = build_app(cfg, db_path=Path(cfg.logging.db_path))
    uvicorn.run(app_obj, host=bind_host, port=bind_port, log_level="info")


@app.command()
def explain(
    prompt: str = typer.Argument(..., help="Prompt text to classify and route (no backend call)"),
    config: Path | None = typer.Option(None, help="Path to config.toml"),
    test: bool = typer.Option(
        False, "--test", help="Dry run: don't log the classification to the request log."
    ),
):
    """Run the routing pipeline against PROMPT and print the decision.

    The classification is logged to the request log (it's a labeled training example
    for `saint classifier train`) unless --test is given.
    """
    import asyncio

    from saint.explain import format_decision
    from saint.router import decide_route
    from saint.storage import build_log_row, log_request, open_db

    cfg = _load_or_die(config)

    async def _run():
        decision = await decide_route(
            cfg=cfg, model_field="saint-explain",
            messages=[{"role": "user", "content": prompt}],
        )
        print(format_decision(decision, cfg))
        if test:
            print("Not logged (--test).")
            return
        try:
            conn = open_db(Path(cfg.logging.db_path))
            row_id = log_request(conn, build_log_row(
                decision, model_field="saint-explain",
                backend_latency_ms=None, success=True, error_kind=None,
                tokens_in=None, tokens_out=None,
                prompt_storage_mode=cfg.logging.prompt_storage,
            ))
            conn.close()
        except Exception as e:  # logging must never mask the decision output
            typer.secho(f"log write failed: {type(e).__name__}: {e}",
                        fg=typer.colors.YELLOW, err=True)
            return
        note = ""
        if cfg.logging.prompt_storage != "full":
            note = (f" — prompt_storage={cfg.logging.prompt_storage!r}, "
                    "so this row can't be used for classifier training")
        print(f"Logged as row #{row_id}{note}.")

    asyncio.run(_run())


policy_app = typer.Typer(help="Inspect the resolved routing policy.")
config_app = typer.Typer(help="Inspect the resolved configuration.")
app.add_typer(policy_app, name="policy")
app.add_typer(config_app, name="config")


@policy_app.command("show")
def policy_show(config: Path | None = typer.Option(None, help="Path to config.toml")):
    """Print the resolved policy tables."""
    cfg = _load_or_die(config)
    for urgency in ("normal", "urgent", "patient"):
        print(f"[policy.{urgency}]")
        for d in ("code", "general"):
            for c in ("trivial", "medium", "hard"):
                cell = f"{d},{c}"
                target = cfg.routing.policy[urgency][cell]
                print(f'  "{cell:<18}" = "{target}"')
        print()
    print(f"default_urgency    = {cfg.routing.default_urgency}")
    print(f"default_on_failure = {cfg.routing.default_on_failure}")


@config_app.command("show")
def config_show(config: Path | None = typer.Option(None, help="Path to config.toml")):
    """Print the resolved configuration with API keys masked."""
    cfg = _load_or_die(config)
    print(f"[server]\nhost = {cfg.server.host}\nport = {cfg.server.port}\n")
    cloud_providers: set[str] = set()
    for name, b in cfg.backends.items():
        print(f"[backends.{name}]")
        print(f"  provider = {b.provider}")
        print(f"  model    = {b.model}")
        if b.base_url:
            print(f"  base_url = {b.base_url}")
        if b.api_key_env:
            present = "set" if os.environ.get(b.api_key_env) else "MISSING"
            print(f"  api_key  = (from ${b.api_key_env}: {present}) ***")
        elif b.api_key:
            print("  api_key  = ***")
        if b.aliases:
            print(f"  aliases  = {list(b.aliases)}")
        print(f"  timeout_s = {b.timeout_s}")
        print()
        if b.provider == "anthropic":
            cloud_providers.add(b.provider)

    print("[classifier]")
    print(f"  mode              = {cfg.classifier.mode}")
    print(f"  backend           = {cfg.classifier.backend}")
    print(f"  fallback_backend  = {cfg.classifier.fallback_backend}")
    if cfg.classifier.mode == "embedding":
        print(f"  embedding_backend = {cfg.classifier.embedding_backend}")
        print(f"  min_confidence    = {cfg.classifier.min_confidence}")
        print(f"  head_path         = {cfg.classifier.head_path or '(default)'}")
    print(f"  max_input_chars   = {cfg.classifier.max_input_chars}")
    print(f"  timeout_s         = {cfg.classifier.timeout_s}")
    print()
    print("[logging]")
    print(f"  db_path        = {cfg.logging.db_path}")
    print(f"  prompt_storage = {cfg.logging.prompt_storage}")
    print()
    print(f"Backends configured: {', '.join(cfg.backends.keys())}")
    if cloud_providers:
        print(f"Cloud backends present: yes ({', '.join(sorted(cloud_providers))})")
    else:
        print("Cloud backends present: no (offline-only)")


@config_app.command("init")
def config_init(
    path: Path | None = typer.Option(None, "--path", help="Where to write (default ~/.config/saint/config.toml)."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config."),
):
    """Write a starter config from the bundled template."""
    import importlib.resources

    target = path or _default_config_path()
    if target.exists() and not force:
        typer.secho(f"config already exists: {target}  (use --force to overwrite)",
                    fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(1)
    tmpl = importlib.resources.files("saint").joinpath("config.example.toml")
    if not tmpl.is_file():
        typer.secho("bundled config template is missing from the package", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(tmpl.read_text())
    typer.secho(f"✓ wrote {target}", fg=typer.colors.GREEN)
    typer.echo("Next: edit it (backends / API keys / the [johnny] binding), then `saint serve`.")


log_app = typer.Typer(help="Inspect request log.")
app.add_typer(log_app, name="log")


@log_app.command("show")
def log_show(
    limit: int = typer.Option(20, "--limit", "-n"),
    backend: str | None = typer.Option(None, "--backend", "-b"),
    config: Path | None = typer.Option(None),
):
    """Show recent request log rows."""
    from saint.storage import get_recent, open_db

    cfg = _load_or_die(config)
    conn = open_db(Path(cfg.logging.db_path))
    rows = get_recent(conn, limit=limit, backend=backend)
    for r in rows:
        print(
            f"#{r['id']:>4} {r['ts']} {r['request_id'][:8]} "
            f"model={r['model_field']} urgency={r['urgency_used']} "
            f"→ {r['backend_chosen']} ({'OK' if r['success'] else 'FAIL'})"
        )


@log_app.command("id")
def log_id(
    row_id: int = typer.Argument(...),
    config: Path | None = typer.Option(None),
):
    """Show full detail for one log row."""
    from saint.storage import get_by_id, open_db

    cfg = _load_or_die(config)
    conn = open_db(Path(cfg.logging.db_path))
    row = get_by_id(conn, row_id)
    if row is None:
        typer.secho(f"No row with id {row_id}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    for k, v in row.items():
        if k == "prompt_content" and v and len(str(v)) > 500:
            v = str(v)[:500] + "...(truncated)"
        print(f"{k:<32} {v}")


@log_app.command("clear")
def log_clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    config: Path | None = typer.Option(None),
):
    """Delete ALL request log rows (including accumulated classifier training data)."""
    from saint.storage import clear_requests, open_db

    cfg = _load_or_die(config)
    conn = open_db(Path(cfg.logging.db_path))
    (count,) = conn.execute("SELECT COUNT(*) FROM requests").fetchone()
    if count == 0:
        print("Request log is already empty.")
        return
    if not yes:
        typer.confirm(f"Delete all {count} request log rows?", abort=True)
    deleted = clear_requests(conn)
    print(f"Deleted {deleted} rows; ids restart at 1.")


def _effective_prices(b) -> dict | None:
    """Backend pricing (USD/Mtok) with anthropic cache-price derivation (0.1x read,
    1.25x write of price_in). None when no price_in is set (cost columns stay blank)."""
    if b.price_in is None:
        return None
    is_anthropic = b.provider == "anthropic"
    return {
        "in": b.price_in,
        "out": b.price_out if b.price_out is not None else 0.0,
        "cache_read": (b.price_cache_read if b.price_cache_read is not None
                       else (0.1 * b.price_in if is_anthropic else 0.0)),
        "cache_write": (b.price_cache_write if b.price_cache_write is not None
                        else (1.25 * b.price_in if is_anthropic else 0.0)),
    }


@log_app.command("stats")
def log_stats(
    days: float = typer.Option(7.0, "--days", "-d", help="Window in days."),
    config: Path | None = typer.Option(None),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
):
    """Per-backend usage and cost accounting, including the prompt-cache advantage.

    est cost      = uncached_in*price_in + reads*cache_read_price + writes*cache_write_price + out*price_out
    no-cache cost = tokens_in*price_in + out*price_out
    cache adv.    = no-cache - est  (negative when cache writes outweighed reads)
    Backends without price_in show token counts only. Local seats: set prices to 0
    (or leave unset) — their cost is electricity, not tokens."""
    import json as _json
    from datetime import UTC, datetime, timedelta

    from saint.storage import open_db, usage_stats

    cfg = _load_or_die(config)
    conn = open_db(Path(cfg.logging.db_path))
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    rows = usage_stats(conn, since)

    out_rows = []
    for r in rows:
        b = cfg.backends.get(r["backend_chosen"])
        prices = _effective_prices(b) if b else None
        entry = {
            "backend": r["backend_chosen"],
            "requests": r["requests"], "ok": r["ok"], "fail": r["requests"] - r["ok"],
            "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
            "cache_read": r["cache_read"], "cache_write": r["cache_write"],
            "avg_latency_ms": round(r["avg_latency_ms"]) if r["avg_latency_ms"] else None,
            "est_cost": None, "no_cache_cost": None, "cache_advantage": None,
        }
        if prices:
            uncached_in = max(r["tokens_in"] - r["cache_read"] - r["cache_write"], 0)
            est = (uncached_in * prices["in"] + r["cache_read"] * prices["cache_read"]
                   + r["cache_write"] * prices["cache_write"]
                   + r["tokens_out"] * prices["out"]) / 1e6
            nocache = (r["tokens_in"] * prices["in"] + r["tokens_out"] * prices["out"]) / 1e6
            entry["est_cost"] = round(est, 4)
            entry["no_cache_cost"] = round(nocache, 4)
            entry["cache_advantage"] = round(nocache - est, 4)
        out_rows.append(entry)

    priced = [e for e in out_rows if e["est_cost"] is not None]
    totals = {
        "requests": sum(e["requests"] for e in out_rows),
        "est_cost": round(sum(e["est_cost"] for e in priced), 4) if priced else None,
        "cache_advantage": (round(sum(e["cache_advantage"] for e in priced), 4)
                            if priced else None),
    }

    if as_json:
        print(_json.dumps({"window_days": days, "since": since,
                           "backends": out_rows, "totals": totals}, indent=2))
        return

    from rich.console import Console
    from rich.table import Table

    t = Table(title=f"saint usage — last {days:g} day(s)")
    for col in ("backend", "req (ok/fail)", "tok in", "tok out",
                "pc read", "pc write", "avg ms", "est cost", "no-cache", "cache adv."):
        t.add_column(col, justify="right" if col != "backend" else "left")
    for e in out_rows:
        fmt = lambda v: f"${v:.2f}" if v is not None else "—"
        t.add_row(
            e["backend"], f"{e['requests']} ({e['ok']}/{e['fail']})",
            f"{e['tokens_in']:,}", f"{e['tokens_out']:,}",
            f"{e['cache_read']:,}", f"{e['cache_write']:,}",
            str(e["avg_latency_ms"] or "—"),
            fmt(e["est_cost"]), fmt(e["no_cache_cost"]), fmt(e["cache_advantage"]),
        )
    Console().print(t)
    if totals["est_cost"] is not None:
        adv = totals["cache_advantage"]
        direction = "saved" if adv >= 0 else "LOST (writes outweighed reads)"
        typer.secho(f"total est cost ${totals['est_cost']:.2f} — prompt caching "
                    f"{direction} ${abs(adv):.2f}",
                    fg=typer.colors.GREEN if adv >= 0 else typer.colors.YELLOW)


@log_app.command("prune")
def log_prune(
    days: float = typer.Option(30.0, "--days", "-d", help="Delete rows older than this."),
    keep_training: bool = typer.Option(
        True, "--keep-training/--no-keep-training",
        help="Preserve rows the classifier trainer can still use (default: keep)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    config: Path | None = typer.Option(None),
):
    """Age out old request log rows (training-usable rows are kept by default)."""
    from datetime import UTC, datetime, timedelta

    from saint.storage import open_db, prune_requests

    cfg = _load_or_die(config)
    conn = open_db(Path(cfg.logging.db_path))
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    (count,) = conn.execute("SELECT COUNT(*) FROM requests WHERE ts < ?", (cutoff,)).fetchone()
    if count == 0:
        print(f"Nothing older than {days:g} day(s).")
        return
    scope = "non-training rows" if keep_training else "ALL rows (including training data)"
    if not yes:
        typer.confirm(f"Prune {scope} older than {days:g} day(s) "
                      f"(up to {count} candidates)?", abort=True)
    deleted = prune_requests(conn, cutoff, keep_training=keep_training)
    print(f"Pruned {deleted} rows (kept {count - deleted} older rows"
          f"{' as training data' if keep_training else ''}).")


relabel_app = typer.Typer(help="Mark requests with corrected backend.")
app.add_typer(relabel_app, name="relabel")


def _check_backend_defined(cfg, backend: str) -> None:
    if backend not in cfg.backends:
        typer.secho(
            f"Backend '{backend}' is not defined in current config. "
            f"Defined: {sorted(cfg.backends.keys())}",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(1) from None


@relabel_app.command("last")
def relabel_last_cmd(
    backend: str = typer.Argument(..., help="The backend that should have been used"),
    note: str | None = typer.Option(None, "--note"),
    config: Path | None = typer.Option(None),
):
    """Mark the most recent request as 'should have been <backend>'."""
    from saint.storage import open_db
    from saint.storage import relabel_last as _relabel_last

    cfg = _load_or_die(config)
    _check_backend_defined(cfg, backend)
    conn = open_db(Path(cfg.logging.db_path))
    try:
        rid = _relabel_last(conn, backend, note)
        print(f"Relabeled row #{rid} → {backend}")
    except Exception as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


@relabel_app.command("by-id")
def relabel_by_id_cmd(
    row_id: int = typer.Argument(..., metavar="ID"),
    backend: str = typer.Argument(...),
    note: str | None = typer.Option(None, "--note"),
    config: Path | None = typer.Option(None),
):
    """Mark a specific row id as 'should have been <backend>'."""
    from saint.storage import open_db
    from saint.storage import relabel_by_id as _relabel_by_id

    cfg = _load_or_die(config)
    _check_backend_defined(cfg, backend)
    conn = open_db(Path(cfg.logging.db_path))
    try:
        _relabel_by_id(conn, row_id, backend, note)
        print(f"Relabeled row #{row_id} → {backend}")
    except Exception as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


classifier_app = typer.Typer(help="Embedding classifier: train / inspect the head.")
app.add_typer(classifier_app, name="classifier")


@classifier_app.command("train")
def classifier_train(
    config: Path | None = typer.Option(None, help="Path to config.toml"),
    limit: int = typer.Option(5000, "--limit", help="Max recent logged rows to train from."),
    min_samples: int = typer.Option(50, "--min-samples", help="Refuse to train on fewer than this."),
):
    """Distill the embedding head from the request log's (prompt → domain/complexity) labels.

    Reads recent requests with stored full prompts + the LLM classifier's own labels (never
    the head's own — no feedback loop), embeds each prompt via the embedding backend, fits
    the two logistic-regression heads, and saves to classifier.head_path. Needs
    logging.prompt_storage = 'full' and some traffic classified in mode='llm' first.
    """
    import asyncio

    import numpy as np

    from saint import embed_classifier as EC
    from saint.storage import open_db

    cfg = _load_or_die(config)
    eb_name = cfg.classifier.embedding_backend
    if not eb_name or eb_name not in cfg.backends:
        typer.secho("classifier.embedding_backend must be set (and defined) to train.",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    embed_backend = cfg.backends[eb_name]

    from saint.storage import fetch_training_rows

    conn = open_db(Path(cfg.logging.db_path))
    rows = fetch_training_rows(conn, limit)
    seen: set[str] = set()
    prompts: list[str] = []
    doms: list[str] = []
    cplxs: list[str] = []
    for content, dom, cplx in rows:
        if content in seen:  # dedupe; rows are DESC so we keep the most recent label
            continue
        seen.add(content)
        prompts.append(content)
        doms.append(dom)
        cplxs.append(cplx)
    if len(prompts) < min_samples:
        typer.secho(f"only {len(prompts)} usable rows (need ≥{min_samples}). Run more traffic in "
                    "mode='llm' with logging.prompt_storage='full' first.",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.echo(f"embedding {len(prompts)} prompts via '{eb_name}'…")

    async def _embed_all() -> "np.ndarray":
        chunks = []
        batch = 64
        for i in range(0, len(prompts), batch):
            chunks.append(await EC.embed_texts(embed_backend, prompts[i:i + batch]))
            typer.echo(f"  {min(i + batch, len(prompts))}/{len(prompts)}")
        return np.vstack(chunks)

    try:
        X = asyncio.run(_embed_all())
    except Exception as e:
        typer.secho(f"embedding failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None

    head = EC.train_head(X, doms, cplxs, embed_model=str(embed_backend.model))
    out = Path(cfg.classifier.head_path or EC.DEFAULT_HEAD_PATH).expanduser()
    head.save(out)
    d_ok = sum(head.predict(v)[0] == d for v, d in zip(X, doms))
    c_ok = sum(head.predict(v)[2] == c for v, c in zip(X, cplxs))
    typer.secho(f"✓ trained on {len(prompts)} samples → {out}", fg=typer.colors.GREEN)
    typer.echo(f"  train accuracy: domain {d_ok}/{len(prompts)}  complexity {c_ok}/{len(prompts)}")
    typer.echo(f"  domain={head.domain_classes}  complexity={head.complexity_classes}")
    typer.echo("  set classifier.mode = 'embedding' in your config to use it (falls back to the LLM when unsure).")


@classifier_app.command("status")
def classifier_status(
    config: Path | None = typer.Option(None, help="Path to config.toml"),
    drift: bool = typer.Option(
        False, "--drift",
        help="Replay the head against recent LLM-labeled rows (embeds them; a few seconds).",
    ),
    limit: int = typer.Option(200, "--limit", help="Max rows for the --drift comparison."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output (for monitoring jobs)."),
):
    """Show classifier mode, head metadata, live coverage, and (--drift) boundary drift.

    Coverage: how recent traffic split between the head, LLM deferrals, and cache reuse.
    Drift: head predictions vs the LLM's labels on recent deferred rows — the boundary
    region where the head declined. Low routing agreement there, or lots of new labeled
    rows since training, means a retrain is worth it.

    --json emits one object with `healthy` (bool) and `suggestions` (list); a monitoring
    job alerts when healthy is false.
    """
    import json as _json

    from saint import embed_classifier as EC
    from saint.storage import (
        classifier_traffic_mix,
        count_training_rows_since,
        fetch_training_rows,
        open_db,
    )

    say = (lambda *a, **k: None) if as_json else print
    data: dict = {}

    cfg = _load_or_die(config)
    path = Path(cfg.classifier.head_path or EC.DEFAULT_HEAD_PATH).expanduser()
    say(f"mode              = {cfg.classifier.mode}")
    say(f"embedding_backend = {cfg.classifier.embedding_backend}")
    say(f"min_confidence    = {cfg.classifier.min_confidence}")
    say(f"head_path         = {path}")
    data["mode"] = cfg.classifier.mode
    data["min_confidence"] = cfg.classifier.min_confidence
    head = None
    if path.exists():
        head = EC.Head.load(path)
        say(f"trained_at        = {head.trained_at}")
        say(f"n_samples         = {head.n_samples}")
        say(f"embed_model       = {head.embed_model}")
        say(f"dim               = {head.dim}")
        say(f"domain_classes    = {head.domain_classes}")
        say(f"complexity_classes= {head.complexity_classes}")
        data["head"] = {"trained_at": head.trained_at, "n_samples": head.n_samples,
                        "embed_model": head.embed_model, "dim": head.dim}
    else:
        say("head              = (not trained — run `saint classifier train`)")
        data["head"] = None

    conn = open_db(Path(cfg.logging.db_path))
    # Window = routed traffic the CURRENT head has served (since trained_at); explain/
    # seeding rows are excluded so old dataset batches can't taint the stats.
    mix = classifier_traffic_mix(conn, 500, since=head.trained_at if head else None)
    classified = mix["head"] + mix["llm"]
    say()
    window = "routed traffic since head trained" if head else "recent routed traffic"
    say(f"{window} ({sum(mix.values())} classified rows, explain/seed excluded):")
    say(f"  head answered   = {mix['head']}")
    say(f"  llm answered    = {mix['llm']}"
        + ("  (head deferrals)" if cfg.classifier.mode == "embedding" else ""))
    say(f"  cache/inherited = {mix['reused']}")
    data["traffic"] = dict(mix)
    if classified:
        coverage = mix["head"] / classified
        say(f"  head coverage   = {100 * coverage:.0f}% of fresh classifications")
        data["traffic"]["head_coverage"] = round(coverage, 3)
    new_rows = count_training_rows_since(conn, head.trained_at) if head else None
    if new_rows is not None:
        say(f"  new labeled rows since training = {new_rows}")
        data["new_labeled_rows_since_training"] = new_rows

    suggestions: list[str] = []
    if head is None:
        suggestions.append("no head trained — run `saint classifier train`")
    elif classified and mix["head"] / classified < 0.5 and (new_rows or 0) >= 100:
        suggestions.append(
            f"head coverage {100 * mix['head'] / classified:.0f}% with {new_rows} new "
            "labeled rows banked — retrain to grow the confident region"
        )

    if drift and head is not None:
        eb_name = cfg.classifier.embedding_backend
        if not eb_name or eb_name not in cfg.backends:
            _emit_err("classifier.embedding_backend must be set for --drift")
            raise typer.Exit(2)
        import asyncio

        from saint.policy import resolve_policy

        # Deferred/LLM-labeled rows are exactly the trainer's input: the head declined
        # these (or mode was 'llm'), and the LLM's labels are the reference. Only rows
        # AFTER trained_at count — replaying the head over its own training set would
        # flatter it.
        raw = fetch_training_rows(conn, limit=limit * 3, since=head.trained_at)
        seen: set[str] = set()
        rows = []
        for prompt, dom, cplx in raw:
            if prompt in seen:
                continue
            seen.add(prompt)
            rows.append((prompt, dom, cplx))
            if len(rows) >= limit:
                break
        if not rows:
            say("\ndrift: no LLM-labeled rows since training to compare against")
            data["drift"] = None
        else:
            prompts = [r[0] for r in rows]
            small_n = "  (small sample)" if len(rows) < 30 else ""
            say(f"\ndrift check: replaying head over {len(rows)} LLM-labeled rows "
                f"since training…{small_n}")

            async def _embed():
                chunks = []
                for i in range(0, len(prompts), 64):
                    chunks.append(await EC.embed_texts(cfg.backends[eb_name], prompts[i:i + 64]))
                import numpy as np
                return np.vstack(chunks)

            X = asyncio.run(_embed())
            dom_hit = cplx_hit = route_hit = confident = 0
            for k, (_, dom, cplx) in enumerate(rows):
                dl, dc, cl, cc = head.predict(X[k])
                dom_hit += dl == dom
                cplx_hit += cl == cplx
                confident += min(dc, cc) >= cfg.classifier.min_confidence
                want = resolve_policy(cfg.routing.policy, "normal", dom, cplx)
                got = resolve_policy(cfg.routing.policy, "normal", dl, cl)
                route_hit += want == got
            n = len(rows)
            say(f"  domain agreement     = {dom_hit}/{n} ({100 * dom_hit / n:.0f}%)")
            say(f"  complexity agreement = {cplx_hit}/{n} ({100 * cplx_hit / n:.0f}%)")
            say(f"  routing agreement    = {route_hit}/{n} ({100 * route_hit / n:.0f}%)"
                "  (policy.normal destinations)")
            say(f"  head now confident on {confident}/{n} ({100 * confident / n:.0f}%) "
                "of these (they deferred when served)")
            data["drift"] = {
                "sample": n, "small_sample": n < 30,
                "domain_agreement": round(dom_hit / n, 3),
                "complexity_agreement": round(cplx_hit / n, 3),
                "routing_agreement": round(route_hit / n, 3),
                "now_confident": round(confident / n, 3),
            }
            if route_hit / n < 0.9:
                suggestions.append(
                    f"boundary drift: head disagrees with the LLM's routing on "
                    f"{n - route_hit}/{n} recent deferrals — retrain"
                )

    data["suggestions"] = suggestions
    data["healthy"] = not suggestions
    if as_json:
        print(_json.dumps(data, indent=2))
        return
    print()
    if suggestions:
        for s in suggestions:
            typer.secho(f"⚠ {s}", fg=typer.colors.YELLOW)
    else:
        typer.secho("✓ no retrain needed by current heuristics", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
