from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Optional

import typer
import uvicorn

from goorouter.config import load_config


app = typer.Typer(no_args_is_help=True, help="goorouter — localhost OpenAI-compatible router")


def _default_config_path() -> Path:
    return Path(os.path.expanduser("~/.goorouter/config.toml"))


def _load_or_die(config_path: Path | None):
    path = config_path or _default_config_path()
    if not path.exists():
        typer.secho(f"Config not found: {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    try:
        return load_config(path)
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(2)


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, help="Override [server.host]"),
    port: Optional[int] = typer.Option(None, help="Override [server.port]"),
    config: Optional[Path] = typer.Option(None, help="Path to config.toml"),
):
    """Start the proxy server."""
    cfg = _load_or_die(config)
    bind_host = host or cfg.server.host
    bind_port = port or cfg.server.port

    if bind_host not in ("127.0.0.1", "localhost", "::1"):
        typer.secho(
            f"WARNING: binding to {bind_host} (not loopback). Anyone on this network may reach the proxy.",
            fg=typer.colors.YELLOW, err=True,
        )

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((bind_host, bind_port))
    except OSError:
        typer.secho(
            f"Port {bind_port} in use; is another `goorouter serve` running?",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(3)
    finally:
        s.close()

    from goorouter.server import build_app
    app_obj = build_app(cfg, db_path=Path(cfg.logging.db_path))
    uvicorn.run(app_obj, host=bind_host, port=bind_port, log_level="info")


@app.command()
def explain(
    prompt: str = typer.Argument(..., help="Prompt text to classify and route (no destination call made)"),
    config: Optional[Path] = typer.Option(None, help="Path to config.toml"),
):
    """Run the routing pipeline against PROMPT and print the decision."""
    import asyncio

    from goorouter.explain import format_decision
    from goorouter.router import decide_route

    cfg = _load_or_die(config)

    async def _run():
        decision = await decide_route(
            cfg=cfg, model_field="goo-explain",
            messages=[{"role": "user", "content": prompt}],
        )
        print(format_decision(decision, cfg))

    asyncio.run(_run())


policy_app = typer.Typer(help="Inspect the resolved routing policy.")
config_app = typer.Typer(help="Inspect the resolved configuration.")
app.add_typer(policy_app, name="policy")
app.add_typer(config_app, name="config")


@policy_app.command("show")
def policy_show(config: Optional[Path] = typer.Option(None, help="Path to config.toml")):
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
def config_show(config: Optional[Path] = typer.Option(None, help="Path to config.toml")):
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
    print(f"  backend          = {cfg.classifier.backend}")
    print(f"  fallback_backend = {cfg.classifier.fallback_backend}")
    print(f"  max_input_chars  = {cfg.classifier.max_input_chars}")
    print(f"  timeout_s        = {cfg.classifier.timeout_s}")
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


log_app = typer.Typer(help="Inspect request log.")
app.add_typer(log_app, name="log")


@log_app.command("show")
def log_show(
    limit: int = typer.Option(20, "--limit", "-n"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
    config: Optional[Path] = typer.Option(None),
):
    """Show recent request log rows."""
    from goorouter.storage import get_recent, open_db

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
    config: Optional[Path] = typer.Option(None),
):
    """Show full detail for one log row."""
    from goorouter.storage import get_by_id, open_db

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
        raise typer.Exit(1)


@relabel_app.command("last")
def relabel_last_cmd(
    backend: str = typer.Argument(..., help="The backend that should have been used"),
    note: Optional[str] = typer.Option(None, "--note"),
    config: Optional[Path] = typer.Option(None),
):
    """Mark the most recent request as 'should have been <backend>'."""
    from goorouter.storage import open_db, relabel_last as _relabel_last

    cfg = _load_or_die(config)
    _check_backend_defined(cfg, backend)
    conn = open_db(Path(cfg.logging.db_path))
    try:
        rid = _relabel_last(conn, backend, note)
        print(f"Relabeled row #{rid} → {backend}")
    except Exception as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@relabel_app.command("by-id")
def relabel_by_id_cmd(
    row_id: int = typer.Argument(..., metavar="ID"),
    backend: str = typer.Argument(...),
    note: Optional[str] = typer.Option(None, "--note"),
    config: Optional[Path] = typer.Option(None),
):
    """Mark a specific row id as 'should have been <backend>'."""
    from goorouter.storage import open_db, relabel_by_id as _relabel_by_id

    cfg = _load_or_die(config)
    _check_backend_defined(cfg, backend)
    conn = open_db(Path(cfg.logging.db_path))
    try:
        _relabel_by_id(conn, row_id, backend, note)
        print(f"Relabeled row #{row_id} → {backend}")
    except Exception as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
