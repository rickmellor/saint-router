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
    on_error: str | None = None         # dispatch-failure fallback backend (one hop, never chained)
    # --- provider = "bedrock" only (validated) ---
    aws_region: str | None = None       # REQUIRED for bedrock backends
    aws_profile: str | None = None      # AWS profile (credential_process/SSO); omit = default chain
    # --- request shaping (any provider) ---
    drop_params: tuple[str, ...] = ()   # strip these params before dispatch (e.g. models that 400 on temperature)
    default_max_tokens: int | None = None  # injected only when the client omitted max_tokens
    # optional pricing (USD per million tokens) for `saint log stats`; for anthropic
    # backends missing cache prices derive as 0.1x / 1.25x of price_in
    price_in: float | None = None
    price_out: float | None = None
    price_cache_read: float | None = None
    price_cache_write: float | None = None

    @property
    def johnny_bound(self) -> bool:
        return bool(self.johnny_role or self.johnny_seat)

    @property
    def johnny_target(self) -> str | None:
        return self.johnny_role or self.johnny_seat


@dataclass(frozen=True)
class BedrockConfig:
    """SSO credential recovery for bedrock backends (all optional — without this block
    bedrock backends still work via the standard AWS credential chain, but SAINT cannot
    probe or re-prompt SSO on auth failures)."""

    credential_process: str | None = None  # the SSO credential helper binary
    spawn_sso_login: bool = True           # open a browser tab once per auth-failure event
    sso_check_timeout_s: float = 8.0       # --refresh-if-needed probe timeout
    auth_cooldown_s: float = 300.0         # breaker cooldown for classified AUTH failures


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
    ignore_after: tuple[str, ...] = ()    # truncate CLASSIFIER input at the first of these markers
                                          # (client-injected context, e.g. agent memory recall);
                                          # dispatch always forwards the full message untouched


@dataclass(frozen=True)
class RoutingConfig:
    default_urgency: Urgency
    default_on_failure: str
    policy: dict[str, dict[str, str]]  # policy[urgency][f"{domain},{complexity}"] = backend_name
    while_loading: str | None = None   # global warm-up target while a bound seat loads
    multimodal_backend: str | None = None   # image/audio-bearing requests (else default_on_failure)
    embeddings_backend: str | None = None   # serves /v1/embeddings (unset = endpoint disabled)
    retry_same_backend: bool = True    # one immediate retry before the on_error hop
    breaker_failures: int = 3          # consecutive dispatch failures to open the circuit
    breaker_cooldown_s: float = 60.0   # how long an open circuit skips the primary


@dataclass(frozen=True)
class LoggingConfig:
    db_path: str
    prompt_storage: PromptStorageMode


@dataclass(frozen=True)
class CacheConfig:
    """Routing decision caches + provider prompt caching (agent-loop economics).

    Agent clients turn one user prompt into N API calls with the same last user
    message and growing tool history. Without these, every turn re-classifies (flap
    risk at label boundaries) and every cloud turn re-pays for the full prefix.
    """

    turn_cache: bool = True
    turn_ttl_s: float = 300.0            # spans an agent loop; matches Anthropic's 5-min cache
    turn_max_entries: int = 1024
    conversation_affinity: bool = True
    conversation_ttl_s: float = 900.0    # sliding: refreshed each turn; walk-away expires it
    conversation_max_entries: int = 512
    short_follow_up_max_chars: int = 40  # follow-ups shorter than this inherit the conversation's labels
    sticky_conversations: bool = False   # always-inherit (OpenRouter-style); off by default
    anthropic_prompt_caching: bool = True
    prompt_cache_min_chars: int = 4000   # ~1k tokens; below Anthropic's cacheable minimum, skip
    anthropic_cache_ttl: str | None = None   # None = 5m default; "1h" allowed (2x write cost)
    # A client marks per-turn volatile context (live clock, git branch, active file…) by
    # placing this sentinel in the FIRST system message; everything after it is extracted,
    # the marker removed, and the tail relocated to a trailing block on the last user
    # message — i.e. AFTER every cache breakpoint, so it never invalidates the cached
    # prefix (or the local engine's prefix reuse) and never reaches the classifier. Set to
    # "" to disable (marker then passes through as literal text).
    volatile_sentinel: str = "<<<saint:volatile>>>"


@dataclass(frozen=True)
class Config:
    server: ServerConfig
    backends: dict[str, BackendConfig]
    classifier: ClassifierConfig
    routing: RoutingConfig
    logging: LoggingConfig
    johnny: JohnnyConfig | None = None  # present iff any backend is johnny-bound
    cache: CacheConfig = CacheConfig()
    bedrock: BedrockConfig | None = None  # SSO recovery knobs; optional

    @property
    def has_bedrock(self) -> bool:
        return any(b.provider == "bedrock" for b in self.backends.values())


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
    for field in ("multimodal_backend", "embeddings_backend"):
        target = getattr(cfg.routing, field)
        if target and target not in backend_names:
            errors.append(f"routing.{field} '{target}' is not defined in [backends]")
    if cfg.routing.breaker_failures < 1:
        errors.append("routing.breaker_failures must be >= 1")
    if cfg.routing.breaker_cooldown_s <= 0:
        errors.append("routing.breaker_cooldown_s must be > 0")
    for name, b in cfg.backends.items():
        if b.on_error and b.on_error not in backend_names:
            errors.append(f"backend '{name}' on_error '{b.on_error}' is not defined in [backends]")
        if b.on_error == name:
            errors.append(f"backend '{name}' on_error must not point at itself")
        for pf in ("price_in", "price_out", "price_cache_read", "price_cache_write"):
            v = getattr(b, pf)
            if v is not None and v < 0:
                errors.append(f"backend '{name}' {pf} must be >= 0")
        if b.provider == "bedrock":
            if not b.aws_region:
                errors.append(f"backend '{name}' (bedrock) requires aws_region")
            if b.api_key or b.api_key_env:
                errors.append(f"backend '{name}' (bedrock) must not set api_key/api_key_env "
                              "(auth comes from the AWS credential chain)")
            if b.base_url:
                errors.append(f"backend '{name}' (bedrock) must not set base_url")
            if b.johnny_bound:
                errors.append(f"backend '{name}' (bedrock) cannot be johnny-bound")
        else:
            if b.aws_region or b.aws_profile:
                errors.append(f"backend '{name}' sets aws_region/aws_profile but provider "
                              f"is '{b.provider}' (bedrock only)")
        if b.default_max_tokens is not None and b.default_max_tokens < 1:
            errors.append(f"backend '{name}' default_max_tokens must be >= 1")
    if cfg.bedrock is not None:
        if cfg.bedrock.sso_check_timeout_s <= 0:
            errors.append("bedrock.sso_check_timeout_s must be > 0")
        if cfg.bedrock.auth_cooldown_s <= 0:
            errors.append("bedrock.auth_cooldown_s must be > 0")

    # --- cache knobs ---
    c = cfg.cache
    if c.turn_ttl_s <= 0:
        errors.append("cache.turn_ttl_s must be > 0")
    if c.conversation_ttl_s <= 0:
        errors.append("cache.conversation_ttl_s must be > 0")
    if c.turn_max_entries < 1:
        errors.append("cache.turn_max_entries must be >= 1")
    if c.conversation_max_entries < 1:
        errors.append("cache.conversation_max_entries must be >= 1")
    if c.short_follow_up_max_chars < 0:
        errors.append("cache.short_follow_up_max_chars must be >= 0")
    if c.prompt_cache_min_chars < 0:
        errors.append("cache.prompt_cache_min_chars must be >= 0")
    if c.anthropic_cache_ttl not in (None, "1h"):
        errors.append(
            f"cache.anthropic_cache_ttl '{c.anthropic_cache_ttl}' must be '5m' (default) or '1h'"
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
            on_error=b.get("on_error"),
            aws_region=b.get("aws_region"),
            aws_profile=b.get("aws_profile"),
            drop_params=tuple(b.get("drop_params", ())),
            default_max_tokens=(int(b["default_max_tokens"])
                                if b.get("default_max_tokens") is not None else None),
            price_in=float(b["price_in"]) if b.get("price_in") is not None else None,
            price_out=float(b["price_out"]) if b.get("price_out") is not None else None,
            price_cache_read=(float(b["price_cache_read"])
                              if b.get("price_cache_read") is not None else None),
            price_cache_write=(float(b["price_cache_write"])
                               if b.get("price_cache_write") is not None else None),
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
        ignore_after=tuple(cls_raw.get("ignore_after", ())),
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
        multimodal_backend=routing_raw.get("multimodal_backend"),
        embeddings_backend=routing_raw.get("embeddings_backend"),
        retry_same_backend=bool(routing_raw.get("retry_same_backend", True)),
        breaker_failures=int(routing_raw.get("breaker_failures", 3)),
        breaker_cooldown_s=float(routing_raw.get("breaker_cooldown_s", 60.0)),
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

    c_raw = raw.get("cache", {})
    cache_ttl = c_raw.get("anthropic_cache_ttl")
    cache = CacheConfig(
        turn_cache=bool(c_raw.get("turn_cache", True)),
        turn_ttl_s=float(c_raw.get("turn_ttl_s", 300.0)),
        turn_max_entries=int(c_raw.get("turn_max_entries", 1024)),
        conversation_affinity=bool(c_raw.get("conversation_affinity", True)),
        conversation_ttl_s=float(c_raw.get("conversation_ttl_s", 900.0)),
        conversation_max_entries=int(c_raw.get("conversation_max_entries", 512)),
        short_follow_up_max_chars=int(c_raw.get("short_follow_up_max_chars", 40)),
        sticky_conversations=bool(c_raw.get("sticky_conversations", False)),
        anthropic_prompt_caching=bool(c_raw.get("anthropic_prompt_caching", True)),
        prompt_cache_min_chars=int(c_raw.get("prompt_cache_min_chars", 4000)),
        anthropic_cache_ttl=None if cache_ttl in (None, "5m") else cache_ttl,
        volatile_sentinel=str(c_raw.get("volatile_sentinel", "<<<saint:volatile>>>")),
    )

    br_raw = raw.get("bedrock")
    bedrock = None
    if br_raw is not None:
        bedrock = BedrockConfig(
            credential_process=(_expand(br_raw["credential_process"])
                                if br_raw.get("credential_process") else None),
            spawn_sso_login=bool(br_raw.get("spawn_sso_login", True)),
            sso_check_timeout_s=float(br_raw.get("sso_check_timeout_s", 8.0)),
            auth_cooldown_s=float(br_raw.get("auth_cooldown_s", 300.0)),
        )

    cfg = Config(
        server=server,
        backends=backends,
        classifier=classifier,
        routing=routing,
        logging=logging_cfg,
        johnny=johnny,
        cache=cache,
        bedrock=bedrock,
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
