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
):
    """Run the routing pipeline against PROMPT and print the decision."""
    import asyncio

    from saint.explain import format_decision
    from saint.router import decide_route

    cfg = _load_or_die(config)

    async def _run():
        decision = await decide_route(
            cfg=cfg, model_field="saint-explain",
            messages=[{"role": "user", "content": prompt}],
        )
        print(format_decision(decision, cfg))

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

    conn = open_db(Path(cfg.logging.db_path))
    rows = conn.execute(
        "SELECT prompt_content, classifier_domain, classifier_complexity FROM requests "
        "WHERE prompt_content IS NOT NULL AND classifier_domain IS NOT NULL "
        "AND classifier_complexity IS NOT NULL "
        "AND (classifier_used IS NULL OR classifier_used NOT LIKE '%embed-head%') "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
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
def classifier_status(config: Path | None = typer.Option(None, help="Path to config.toml")):
    """Show the classifier mode and the trained head's metadata (if any)."""
    from saint import embed_classifier as EC

    cfg = _load_or_die(config)
    path = Path(cfg.classifier.head_path or EC.DEFAULT_HEAD_PATH).expanduser()
    print(f"mode              = {cfg.classifier.mode}")
    print(f"embedding_backend = {cfg.classifier.embedding_backend}")
    print(f"min_confidence    = {cfg.classifier.min_confidence}")
    print(f"head_path         = {path}")
    if not path.exists():
        print("head              = (not trained — run `saint classifier train`)")
        return
    h = EC.Head.load(path)
    print(f"trained_at        = {h.trained_at}")
    print(f"n_samples         = {h.n_samples}")
    print(f"embed_model       = {h.embed_model}")
    print(f"dim               = {h.dim}")
    print(f"domain_classes    = {h.domain_classes}")
    print(f"complexity_classes= {h.complexity_classes}")


if __name__ == "__main__":
    app()
