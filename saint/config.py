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
    # provider/model are the static baseline; optional only for johnny_only backends.
    provider: str | None
    model: str | None
    api_key_env: str | None
    api_key: str | None
    base_url: str | None
    aliases: tuple[str, ...]
    timeout_s: int
    # --- optional johnny binding (override overlay; static fields stay the baseline) ---
    johnny_role: str | None = None
    johnny_seat: str | None = None
    johnny_only: bool = False           # no usable static baseline (may omit base_url/model)
    while_loading: str | None = None    # per-backend override of the warm-up target

    @property
    def johnny_bound(self) -> bool:
        return bool(self.johnny_role or self.johnny_seat)

    @property
    def johnny_target(self) -> str | None:
        return self.johnny_role or self.johnny_seat


@dataclass(frozen=True)
class JohnnyConfig:
    transport: str          # "cli" | "http"
    cli_path: str           # CLI transport: the johnny executable
    base_url: str | None    # HTTP transport: johnnyd base url
    resolve_cache_ttl_s: float
    ensure_load: bool       # may SAINT trigger loads, or only observe + fall back?


@dataclass(frozen=True)
class ClassifierConfig:
    backend: str                          # the LLM classifier (also the fallback in embedding mode)
    fallback_backend: str | None
    max_input_chars: int
    timeout_s: int
    prompt_template_path: str | None
    mode: str = "llm"                     # "llm" | "embedding"
    embedding_backend: str | None = None  # backend serving /v1/embeddings (e.g. nomic-embed via johnny)
    head_path: str | None = None          # trained head .npz (default: ~/.config/saint/classifier_head.npz)
    min_confidence: float = 0.6           # below this the embedding head defers to the LLM classifier


@dataclass(frozen=True)
class RoutingConfig:
    default_urgency: Urgency
    default_on_failure: str
    policy: dict[str, dict[str, str]]  # policy[urgency][f"{domain},{complexity}"] = backend_name
    while_loading: str | None = None   # global warm-up target while a bound seat loads


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
    johnny: JohnnyConfig | None = None  # present iff any backend is johnny-bound


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
    fb = cfg.classifier.fallback_backend
    if fb and fb not in backend_names:
        errors.append(
            f"classifier.fallback_backend '{fb}' is not defined in [backends]"
        )
    if cfg.classifier.mode not in ("llm", "embedding"):
        errors.append(f"classifier.mode '{cfg.classifier.mode}' must be 'llm' or 'embedding'")
    if cfg.classifier.mode == "embedding":
        eb = cfg.classifier.embedding_backend
        if not eb:
            errors.append("classifier.embedding_backend is required when classifier.mode = 'embedding'")
        elif eb not in backend_names:
            errors.append(f"classifier.embedding_backend '{eb}' is not defined in [backends]")
    if cfg.routing.default_on_failure not in backend_names:
        errors.append(
            f"routing.default_on_failure '{cfg.routing.default_on_failure}'"
            " is not defined in [backends]"
        )

    if cfg.routing.default_urgency not in URGENCIES:
        errors.append(
            f"routing.default_urgency '{cfg.routing.default_urgency}'"
            f" must be one of {list(URGENCIES)}"
        )

    if cfg.logging.prompt_storage not in PROMPT_STORAGE_MODES:
        errors.append(
            f"logging.prompt_storage '{cfg.logging.prompt_storage}'"
            f" must be one of {list(PROMPT_STORAGE_MODES)}"
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
                        f"routing.policy.{urgency}['{cell}'] = '{target}'"
                        " is not defined in [backends]"
                    )

    # Backend names + aliases must be unique; aliases must not collide with urgency tokens.
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

    # --- static baseline + johnny binding rules ---
    bound = [n for n, b in cfg.backends.items() if b.johnny_bound]
    for name, b in cfg.backends.items():
        if not b.johnny_only and (not b.provider or not b.model):
            errors.append(
                f"backend '{name}' lacks static provider/model (required unless johnny_only = true)"
            )
        if b.johnny_bound and not b.johnny_only and not b.base_url:
            errors.append(
                f"backend '{name}' is johnny-bound but not johnny_only and has no static base_url baseline"
            )
        if b.while_loading and b.while_loading not in backend_names:
            errors.append(
                f"backend '{name}' while_loading '{b.while_loading}' is not defined in [backends]"
            )
    if bound and cfg.johnny is None:
        errors.append("[johnny] block is required when any backend has a johnny binding")
    if cfg.johnny is not None:
        if cfg.johnny.transport not in ("cli", "http"):
            errors.append(f"[johnny].transport '{cfg.johnny.transport}' must be 'cli' or 'http'")
        if cfg.johnny.transport == "http" and not cfg.johnny.base_url:
            errors.append("[johnny].base_url is required when transport = 'http'")
    if cfg.routing.while_loading and cfg.routing.while_loading not in backend_names:
        errors.append(
            f"routing.while_loading '{cfg.routing.while_loading}' is not defined in [backends]"
        )

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
            provider=b.get("provider"),  # optional only for johnny_only (validated below)
            model=b.get("model"),
            api_key_env=b.get("api_key_env"),
            api_key=b.get("api_key"),
            base_url=b.get("base_url"),
            aliases=tuple(b.get("aliases", ())),
            timeout_s=int(b.get("timeout_s", 60)),
            johnny_role=b.get("johnny_role"),
            johnny_seat=b.get("johnny_seat"),
            johnny_only=bool(b.get("johnny_only", False)),
            while_loading=b.get("while_loading"),
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
        mode=cls_raw.get("mode", "llm"),
        embedding_backend=cls_raw.get("embedding_backend"),
        head_path=_expand(cls_raw["head_path"]) if cls_raw.get("head_path") else None,
        min_confidence=float(cls_raw.get("min_confidence", 0.6)),
    )

    routing_raw = raw["routing"]
    routing = RoutingConfig(
        default_urgency=routing_raw.get("default_urgency", "normal"),
        default_on_failure=routing_raw["default_on_failure"],
        policy={
            urgency: dict(cells)
            for urgency, cells in routing_raw.get("policy", {}).items()
        },
        while_loading=routing_raw.get("while_loading"),
    )

    j_raw = raw.get("johnny")
    johnny = None
    if j_raw is not None:
        johnny = JohnnyConfig(
            transport=j_raw.get("transport", "cli"),
            cli_path=j_raw.get("cli_path", "johnny"),
            base_url=j_raw.get("base_url"),
            resolve_cache_ttl_s=float(j_raw.get("resolve_cache_ttl_s", 1.0)),
            ensure_load=bool(j_raw.get("ensure_load", True)),
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
        johnny=johnny,
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
