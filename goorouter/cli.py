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


if __name__ == "__main__":
    app()
