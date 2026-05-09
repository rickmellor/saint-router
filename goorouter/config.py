from __future__ import annotations

from dataclasses import dataclass
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
