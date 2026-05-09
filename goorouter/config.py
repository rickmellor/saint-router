from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PromptStorageMode = Literal["full", "hashed", "none"]
Urgency = Literal["normal", "urgent", "patient"]


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int


@dataclass(frozen=True)
class BackendConfig:
    name: str
    provider: str
    model: str
    api_key_env: str | None
    api_key: str | None
    base_url: str | None
    aliases: tuple[str, ...]
    timeout_s: int


@dataclass(frozen=True)
class ClassifierConfig:
    backend: str
    fallback_backend: str | None
    max_input_chars: int
    timeout_s: int
    prompt_template_path: str | None


@dataclass(frozen=True)
class RoutingConfig:
    default_urgency: Urgency
    default_on_failure: str
    policy: dict[str, dict[str, str]]  # policy[urgency][f"{domain},{complexity}"] = backend_name


@dataclass(frozen=True)
class LoggingConfig:
    db_path: str
    prompt_storage: PromptStorageMode


@dataclass(frozen=True)
class Config:
    server: ServerConfig
    backends: dict[str, BackendConfig]
    classifier: ClassifierConfig
    routing: RoutingConfig
    logging: LoggingConfig


def _expand(value: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    return str(Path(expanded))


def _load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config(path: Path) -> Config:
    raw = _load_toml(path)

    server_raw = raw.get("server", {})
    server = ServerConfig(
        host=server_raw.get("host", "127.0.0.1"),
        port=int(server_raw.get("port", 4000)),
    )

    backends: dict[str, BackendConfig] = {}
    for name, b in raw.get("backends", {}).items():
        backends[name] = BackendConfig(
            name=name,
            provider=b["provider"],
            model=b["model"],
            api_key_env=b.get("api_key_env"),
            api_key=b.get("api_key"),
            base_url=b.get("base_url"),
            aliases=tuple(b.get("aliases", ())),
            timeout_s=int(b.get("timeout_s", 60)),
        )

    cls_raw = raw["classifier"]
    classifier = ClassifierConfig(
        backend=cls_raw["backend"],
        fallback_backend=cls_raw.get("fallback_backend"),
        max_input_chars=int(cls_raw.get("max_input_chars", 8000)),
        timeout_s=int(cls_raw.get("timeout_s", 5)),
        prompt_template_path=(
            _expand(cls_raw["prompt_template_path"])
            if cls_raw.get("prompt_template_path") else None
        ),
    )

    routing_raw = raw["routing"]
    routing = RoutingConfig(
        default_urgency=routing_raw.get("default_urgency", "normal"),
        default_on_failure=routing_raw["default_on_failure"],
        policy={
            urgency: dict(cells)
            for urgency, cells in routing_raw.get("policy", {}).items()
        },
    )

    log_raw = raw["logging"]
    logging_cfg = LoggingConfig(
        db_path=_expand(log_raw["db_path"]),
        prompt_storage=log_raw.get("prompt_storage", "full"),
    )

    return Config(
        server=server,
        backends=backends,
        classifier=classifier,
        routing=routing,
        logging=logging_cfg,
    )
