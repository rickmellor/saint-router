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
    return os.path.expandvars(os.path.expanduser(value))


DOMAINS = ("code", "general")
COMPLEXITIES = ("trivial", "medium", "hard")
URGENCIES = ("normal", "urgent", "patient")
PROMPT_STORAGE_MODES = ("full", "hashed", "none")
ALL_CELLS = tuple(f"{d},{c}" for d in DOMAINS for c in COMPLEXITIES)


def _validate(cfg: Config) -> list[str]:
    errors: list[str] = []
    backend_names = set(cfg.backends.keys())

    if cfg.classifier.backend not in backend_names:
        errors.append(
            f"classifier.backend '{cfg.classifier.backend}' is not defined in [backends]"
        )
    if cfg.classifier.fallback_backend and cfg.classifier.fallback_backend not in backend_names:
        errors.append(
            f"classifier.fallback_backend '{cfg.classifier.fallback_backend}' is not defined in [backends]"
        )
    if cfg.routing.default_on_failure not in backend_names:
        errors.append(
            f"routing.default_on_failure '{cfg.routing.default_on_failure}' is not defined in [backends]"
        )

    if cfg.routing.default_urgency not in URGENCIES:
        errors.append(
            f"routing.default_urgency '{cfg.routing.default_urgency}' must be one of {list(URGENCIES)}"
        )

    if cfg.logging.prompt_storage not in PROMPT_STORAGE_MODES:
        errors.append(
            f"logging.prompt_storage '{cfg.logging.prompt_storage}' must be one of {list(PROMPT_STORAGE_MODES)}"
        )

    for urgency in URGENCIES:
        cells = cfg.routing.policy.get(urgency)
        if cells is None:
            errors.append(f"routing.policy.{urgency} is missing entirely")
            continue
        for cell in ALL_CELLS:
            if cell not in cells:
                errors.append(f"routing.policy.{urgency} missing cell '{cell}'")
            else:
                target = cells[cell]
                if target not in backend_names:
                    errors.append(
                        f"routing.policy.{urgency}['{cell}'] = '{target}' is not defined in [backends]"
                    )

    # Alias collision check: backend names + aliases must be unique, and may not collide with urgency tokens
    seen_aliases: dict[str, str] = {}
    for name, b in cfg.backends.items():
        for alias in (name, *b.aliases):
            if alias in URGENCIES:
                errors.append(
                    f"backend '{name}' alias '{alias}' collides with reserved urgency token"
                )
            existing = seen_aliases.get(alias)
            if existing and existing != name:
                errors.append(
                    f"alias '{alias}' is used by both backends '{existing}' and '{name}'"
                )
            seen_aliases[alias] = name

    return errors


def _load_toml(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError as e:
        raise ValueError(f"Config file not found: {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Failed to parse TOML at {path}: {e}") from e


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

    cfg = Config(
        server=server,
        backends=backends,
        classifier=classifier,
        routing=routing,
        logging=logging_cfg,
    )
    errors = _validate(cfg)
    if errors:
        raise ValueError("Config errors:\n  - " + "\n  - ".join(errors))
    return cfg


def resolve_backend(cfg: Config, token: str) -> BackendConfig | None:
    if token in cfg.backends:
        return cfg.backends[token]
    for b in cfg.backends.values():
        if token in b.aliases:
            return b
    return None
