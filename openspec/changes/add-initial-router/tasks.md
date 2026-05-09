# goorouter v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `goorouter` v1: a localhost OpenAI-compatible HTTP proxy that routes chat completions between cloud and local backends via a classifier + policy table, with SQLite logging and a relabel/explain CLI.

**Architecture:** FastAPI server exposes `/v1/chat/completions` and `/v1/models`. A pure-Python orchestrator parses prefixes → classifies (or skips on pin) → looks up policy → forwards via the LiteLLM SDK. SQLite log captures every request. Pure-function modules (config, prefixes, policy) are unit-tested in isolation; I/O modules (backends, classifier, storage, server) get integration tests with mocked HTTP.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, LiteLLM SDK, httpx, typer, tomllib (stdlib), sqlite3 (stdlib), pytest, pytest-asyncio, ruff, mypy.

**Reference spec:** `openspec/changes/add-initial-router/proposal.md` and `design.md`. Capability deltas in `specs/proxy/`, `specs/routing/`, `specs/observability/`.

---

## File Structure

Created during this plan (one responsibility per file):

```
goorouter/
  __init__.py             # version, public re-exports
  __main__.py             # `python -m goorouter` → cli.app()
  config.py               # TOML load, env expansion, validation, typed Config
  prefixes.py             # parse leading !tokens; pure
  policy.py               # (urgency, domain, complexity, config) → backend; pure
  storage.py              # SQLite schema, migrations, log_request, queries, relabel
  backends.py             # backend registry + litellm dispatch (sync + streaming)
  classifier.py           # build prompt, call backend, parse JSON, fallback chain
  router.py               # orchestrator: prefixes → classify → policy → backends
  explain.py              # format RoutingDecision as text
  server.py               # FastAPI app, routes, SSE forwarding
  cli.py                  # typer app; subcommands
  classifier_prompt.txt   # default classifier prompt template
  migrations/0001_init.sql

tests/
  __init__.py
  conftest.py             # shared fixtures (tmp config, mock litellm)
  test_config.py
  test_prefixes.py
  test_policy.py
  test_storage.py
  test_backends.py
  test_classifier.py
  test_router.py
  test_explain.py
  test_server.py
  test_cli.py

config.example.toml
pyproject.toml
README.md
.github/workflows/ci.yml
```

---

## Phase 0 — Project bootstrap

### Task 0.1: Initialize pyproject.toml

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "goorouter"
version = "0.1.0"
description = "Localhost OpenAI-compatible router that picks between cloud and local LLMs per request."
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "Rick Mellor" }]
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "litellm>=1.55",
    "httpx>=0.27",
    "typer>=0.12",
    "anthropic>=0.40",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "ruff>=0.7",
    "mypy>=1.13",
]

[project.scripts]
goorouter = "goorouter.cli:app"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "RUF"]

[tool.mypy]
strict = true
python_version = "3.11"
files = ["goorouter"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "feat: initialize pyproject.toml with deps and tooling"
```

### Task 0.2: Create package skeleton

**Files:**
- Create: `goorouter/__init__.py`
- Create: `goorouter/__main__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`

- [ ] **Step 1: Write package files**

`goorouter/__init__.py`:
```python
"""goorouter — localhost OpenAI-compatible router."""

__version__ = "0.1.0"
```

`goorouter/__main__.py`:
```python
from goorouter.cli import app

if __name__ == "__main__":
    app()
```

`tests/__init__.py`:
```python
```

`tests/conftest.py`:
```python
"""Shared test fixtures."""
```

`.gitignore`:
```
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
dist/
build/
*.egg-info/
.coverage
htmlcov/
*.sqlite
*.sqlite-journal
*.sqlite-wal
*.sqlite-shm
```

- [ ] **Step 2: Install package in editable mode**

```bash
python -m pip install -e ".[dev]"
```

Expected: install completes; `goorouter` console script is registered (will fail to run until cli.py exists, that's fine).

- [ ] **Step 3: Verify skeleton importable**

```bash
python -c "import goorouter; print(goorouter.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 4: Commit**

```bash
git add goorouter tests .gitignore
git commit -m "feat: package skeleton (goorouter, tests, gitignore)"
```

---

## Phase 1 — Configuration

### Task 1.1: Config dataclasses

**Files:**
- Create: `goorouter/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from goorouter.config import BackendConfig, Config, ClassifierConfig, RoutingConfig, ServerConfig, LoggingConfig


def test_config_dataclass_construction():
    cfg = Config(
        server=ServerConfig(host="127.0.0.1", port=4000),
        backends={
            "cloud-large": BackendConfig(
                name="cloud-large",
                provider="anthropic",
                model="claude-opus-4-7",
                api_key_env="ANTHROPIC_API_KEY",
                api_key=None,
                base_url=None,
                aliases=("opus", "claude"),
                timeout_s=120,
            )
        },
        classifier=ClassifierConfig(
            backend="cloud-large",
            fallback_backend=None,
            max_input_chars=8000,
            timeout_s=5,
            prompt_template_path=None,
        ),
        routing=RoutingConfig(
            default_urgency="normal",
            default_on_failure="cloud-large",
            policy={
                "normal": {"code,trivial": "cloud-large"},
                "urgent": {"code,trivial": "cloud-large"},
                "patient": {"code,trivial": "cloud-large"},
            },
        ),
        logging=LoggingConfig(
            db_path="~/.goorouter/log.sqlite",
            prompt_storage="full",
        ),
    )
    assert cfg.backends["cloud-large"].aliases == ("opus", "claude")
    assert cfg.routing.default_on_failure == "cloud-large"
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/test_config.py::test_config_dataclass_construction -v
```

Expected: ImportError — `goorouter.config` does not exist.

- [ ] **Step 3: Implement**

`goorouter/config.py`:
```python
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
```

- [ ] **Step 4: Verify pass**

```bash
pytest tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add goorouter/config.py tests/test_config.py
git commit -m "feat(config): typed config dataclasses"
```

### Task 1.2: TOML loader with path/env expansion

**Files:**
- Modify: `goorouter/config.py` (append `load_config`)
- Modify: `tests/test_config.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:
```python
import os
from pathlib import Path
import pytest

from goorouter.config import load_config

EXAMPLE_TOML = """
[server]
host = "127.0.0.1"
port = 4000

[backends.cloud-large]
provider = "anthropic"
model = "claude-opus-4-7"
api_key_env = "ANTHROPIC_API_KEY"
aliases = ["opus"]
timeout_s = 120

[backends.local-small]
provider = "openai"
base_url = "http://localhost:1234/v1"
model = "qwen2.5-3b-instruct"
api_key = "lm-studio"
aliases = []
timeout_s = 60

[classifier]
backend = "local-small"
max_input_chars = 8000
timeout_s = 5

[routing]
default_urgency = "normal"
default_on_failure = "cloud-large"

[routing.policy.normal]
"code,trivial"    = "local-small"
"code,medium"     = "local-small"
"code,hard"       = "cloud-large"
"general,trivial" = "local-small"
"general,medium"  = "local-small"
"general,hard"    = "cloud-large"

[routing.policy.urgent]
"code,trivial"    = "local-small"
"code,medium"     = "cloud-large"
"code,hard"       = "cloud-large"
"general,trivial" = "cloud-large"
"general,medium"  = "cloud-large"
"general,hard"    = "cloud-large"

[routing.policy.patient]
"code,trivial"    = "local-small"
"code,medium"     = "local-small"
"code,hard"       = "local-small"
"general,trivial" = "local-small"
"general,medium"  = "local-small"
"general,hard"    = "local-small"

[logging]
db_path = "${TEST_HOME}/log.sqlite"
prompt_storage = "full"
"""


def test_load_config_basic(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_HOME", str(tmp_path))
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(EXAMPLE_TOML)
    cfg = load_config(cfg_path)
    assert cfg.server.port == 4000
    assert "cloud-large" in cfg.backends
    assert cfg.backends["cloud-large"].aliases == ("opus",)
    assert cfg.logging.db_path == f"{tmp_path}/log.sqlite"


def test_load_config_tilde_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    monkeypatch.setenv("TEST_HOME", str(tmp_path))
    toml = EXAMPLE_TOML.replace('db_path = "${TEST_HOME}/log.sqlite"', 'db_path = "~/log.sqlite"')
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(toml)
    cfg = load_config(cfg_path)
    assert cfg.logging.db_path == str(Path(tmp_path) / "log.sqlite")
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/test_config.py::test_load_config_basic -v
```

Expected: ImportError on `load_config`.

- [ ] **Step 3: Implement loader**

Append to `goorouter/config.py`:
```python
import os
import tomllib
from pathlib import Path


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


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
```

- [ ] **Step 4: Verify pass**

```bash
pytest tests/test_config.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(config): TOML loader with ~ and \${VAR} expansion"
```

### Task 1.3: Config validation

**Files:**
- Modify: `goorouter/config.py` (add `validate(cfg) -> list[str]` and call it in `load_config`)
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_config.py`:
```python
def test_validate_missing_classifier_backend(tmp_path):
    toml = EXAMPLE_TOML.replace('backend = "local-small"', 'backend = "phantom"')
    p = tmp_path / "c.toml"; p.write_text(toml)
    with pytest.raises(ValueError) as e:
        load_config(p)
    assert "phantom" in str(e.value)


def test_validate_missing_policy_cell(tmp_path):
    # Drop one cell from policy.normal
    toml = EXAMPLE_TOML.replace('"general,hard" = "cloud-large"\n\n[routing.policy.urgent]',
                                 "[routing.policy.urgent]", 1)
    p = tmp_path / "c.toml"; p.write_text(toml)
    with pytest.raises(ValueError) as e:
        load_config(p)
    assert "general,hard" in str(e.value)


def test_validate_undefined_backend_in_policy(tmp_path):
    toml = EXAMPLE_TOML.replace('"code,hard"       = "cloud-large"',
                                 '"code,hard"       = "phantom"', 1)
    p = tmp_path / "c.toml"; p.write_text(toml)
    with pytest.raises(ValueError) as e:
        load_config(p)
    assert "phantom" in str(e.value)


def test_validate_default_on_failure_undefined(tmp_path):
    toml = EXAMPLE_TOML.replace('default_on_failure = "cloud-large"',
                                 'default_on_failure = "phantom"')
    p = tmp_path / "c.toml"; p.write_text(toml)
    with pytest.raises(ValueError) as e:
        load_config(p)
    assert "phantom" in str(e.value)


def test_validate_collects_all_errors(tmp_path):
    toml = EXAMPLE_TOML.replace('backend = "local-small"', 'backend = "phantom1"')
    toml = toml.replace('default_on_failure = "cloud-large"', 'default_on_failure = "phantom2"')
    p = tmp_path / "c.toml"; p.write_text(toml)
    with pytest.raises(ValueError) as e:
        load_config(p)
    msg = str(e.value)
    assert "phantom1" in msg and "phantom2" in msg
```

- [ ] **Step 2: Run, verify failures**

```bash
pytest tests/test_config.py -v
```

Expected: 5 new tests fail (no validation yet).

- [ ] **Step 3: Implement validation**

Append to `goorouter/config.py`:
```python
DOMAINS = ("code", "general")
COMPLEXITIES = ("trivial", "medium", "hard")
URGENCIES = ("normal", "urgent", "patient")
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

    # Alias collision check
    seen_aliases: dict[str, str] = {}
    for name, b in cfg.backends.items():
        for alias in (name, *b.aliases):
            if alias in URGENCIES:
                errors.append(f"backend '{name}' alias '{alias}' collides with reserved urgency token")
            existing = seen_aliases.get(alias)
            if existing and existing != name:
                errors.append(f"alias '{alias}' is used by both backends '{existing}' and '{name}'")
            seen_aliases[alias] = name

    return errors
```

Modify the bottom of `load_config` to call validation:
```python
def load_config(path: Path) -> Config:
    # ... existing parsing code ...
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
```

- [ ] **Step 4: Verify pass**

```bash
pytest tests/test_config.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(config): all-errors-at-once validation"
```

### Task 1.4: Backend lookup by name or alias

**Files:**
- Modify: `goorouter/config.py` (add `resolve_backend`)
- Modify: `tests/test_config.py`

- [ ] **Step 1: Failing test**

Append:
```python
def test_resolve_backend_by_name_or_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_HOME", str(tmp_path))
    p = tmp_path / "c.toml"; p.write_text(EXAMPLE_TOML)
    cfg = load_config(p)
    from goorouter.config import resolve_backend

    assert resolve_backend(cfg, "cloud-large").name == "cloud-large"
    assert resolve_backend(cfg, "opus").name == "cloud-large"
    assert resolve_backend(cfg, "nope") is None
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_config.py::test_resolve_backend_by_name_or_alias -v
```

- [ ] **Step 3: Implement**

Append to `goorouter/config.py`:
```python
def resolve_backend(cfg: Config, token: str) -> BackendConfig | None:
    if token in cfg.backends:
        return cfg.backends[token]
    for b in cfg.backends.values():
        if token in b.aliases:
            return b
    return None
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_config.py -v && git commit -am "feat(config): resolve_backend by name or alias"
```

---

## Phase 2 — Pure logic modules

### Task 2.1: Prefix parser

**Files:**
- Create: `goorouter/prefixes.py`
- Test: `tests/test_prefixes.py`

- [ ] **Step 1: Failing test**

`tests/test_prefixes.py`:
```python
import pytest

from goorouter.prefixes import ParsedPrefixes, UnknownPrefixError, parse_prefixes

URGENCIES = {"urgent", "patient", "normal"}
BACKENDS = {"cloud-large": {"opus", "claude"}, "local-coder": {"coder"}, "local-small": set()}


def test_no_prefix():
    out = parse_prefixes("hello world", URGENCIES, BACKENDS)
    assert out == ParsedPrefixes(urgency=None, pinned_backend=None, stripped="hello world", raw="")


def test_urgency_only():
    out = parse_prefixes("!urgent fix bug", URGENCIES, BACKENDS)
    assert out.urgency == "urgent"
    assert out.pinned_backend is None
    assert out.stripped == "fix bug"
    assert out.raw == "!urgent"


def test_backend_alias():
    out = parse_prefixes("!opus refactor", URGENCIES, BACKENDS)
    assert out.pinned_backend == "cloud-large"
    assert out.stripped == "refactor"


def test_combined():
    out = parse_prefixes("!urgent !opus do this", URGENCIES, BACKENDS)
    assert out.urgency == "urgent"
    assert out.pinned_backend == "cloud-large"
    assert out.stripped == "do this"


def test_last_urgency_wins():
    out = parse_prefixes("!urgent !patient go", URGENCIES, BACKENDS)
    assert out.urgency == "patient"
    assert out.stripped == "go"


def test_leading_whitespace_disables():
    out = parse_prefixes(" !opus please", URGENCIES, BACKENDS)
    assert out.urgency is None
    assert out.pinned_backend is None
    assert out.stripped == " !opus please"


def test_unknown_token_raises():
    with pytest.raises(UnknownPrefixError) as e:
        parse_prefixes("!doesnotexist hi", URGENCIES, BACKENDS)
    assert "doesnotexist" in str(e.value)


def test_case_sensitive():
    with pytest.raises(UnknownPrefixError):
        parse_prefixes("!URGENT hi", URGENCIES, BACKENDS)


def test_empty_string():
    out = parse_prefixes("", URGENCIES, BACKENDS)
    assert out.urgency is None
    assert out.stripped == ""
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_prefixes.py -v
```

Expected: ImportError on `goorouter.prefixes`.

- [ ] **Step 3: Implement**

`goorouter/prefixes.py`:
```python
from __future__ import annotations

from dataclasses import dataclass


class UnknownPrefixError(ValueError):
    """Raised when a !-prefix token doesn't match any urgency or backend."""

    def __init__(self, token: str, known_urgencies: set[str], known_backends: set[str]):
        self.token = token
        super().__init__(
            f"unknown prefix token '{token}'; "
            f"known urgency tokens: {sorted(known_urgencies)}; "
            f"known backend names/aliases: {sorted(known_backends)}"
        )


@dataclass(frozen=True)
class ParsedPrefixes:
    urgency: str | None
    pinned_backend: str | None
    stripped: str
    raw: str


def parse_prefixes(
    content: str,
    urgencies: set[str],
    backends_with_aliases: dict[str, set[str]],
) -> ParsedPrefixes:
    """Parse leading !<token> prefixes. The first '!' must be at index 0.

    Multiple prefixes are space-separated and parsed left-to-right until the
    first non-prefix token. Last urgency wins. Returns the stripped content
    (prefix tokens and any single space after them removed).
    """
    if not content or content[0] != "!":
        return ParsedPrefixes(urgency=None, pinned_backend=None, stripped=content, raw="")

    # Build alias → backend-name lookup
    alias_to_backend: dict[str, str] = {}
    all_known_tokens: set[str] = set()
    for backend_name, aliases in backends_with_aliases.items():
        alias_to_backend[backend_name] = backend_name
        all_known_tokens.add(backend_name)
        for alias in aliases:
            alias_to_backend[alias] = backend_name
            all_known_tokens.add(alias)

    urgency: str | None = None
    pinned: str | None = None
    raw_tokens: list[str] = []
    rest = content
    while rest.startswith("!"):
        # Find end of token: whitespace or end of string
        i = 1
        while i < len(rest) and not rest[i].isspace():
            i += 1
        token = rest[1:i]
        raw_tokens.append("!" + token)
        if token in urgencies:
            urgency = token
        elif token in alias_to_backend:
            # Last backend pin wins (rare, but allowed)
            pinned = alias_to_backend[token]
        else:
            raise UnknownPrefixError(token, urgencies, all_known_tokens)
        # Skip exactly one space if present
        rest = rest[i:].lstrip(" ") if i < len(rest) and rest[i] == " " else rest[i:]

    return ParsedPrefixes(
        urgency=urgency,
        pinned_backend=pinned,
        stripped=rest,
        raw=" ".join(raw_tokens),
    )
```

- [ ] **Step 4: Verify pass**

```bash
pytest tests/test_prefixes.py -v
```

Expected: all 9 pass.

- [ ] **Step 5: Commit**

```bash
git add goorouter/prefixes.py tests/test_prefixes.py
git commit -m "feat(prefixes): parse leading !-prefixes (urgency + backend pin)"
```

### Task 2.2: Policy lookup

**Files:**
- Create: `goorouter/policy.py`
- Test: `tests/test_policy.py`

- [ ] **Step 1: Failing test**

`tests/test_policy.py`:
```python
from goorouter.policy import resolve_policy


POLICY = {
    "normal": {
        "code,trivial": "local-coder",
        "code,medium": "local-coder",
        "code,hard": "cloud-large",
        "general,trivial": "local-small",
        "general,medium": "local-large",
        "general,hard": "cloud-large",
    },
    "urgent": {
        "code,trivial": "local-coder",
        "code,medium": "cloud-small",
        "code,hard": "cloud-large",
        "general,trivial": "cloud-small",
        "general,medium": "cloud-small",
        "general,hard": "cloud-large",
    },
    "patient": {
        "code,trivial": "local-coder",
        "code,medium": "local-coder",
        "code,hard": "local-large",
        "general,trivial": "local-small",
        "general,medium": "local-small",
        "general,hard": "local-large",
    },
}


def test_resolve_normal_code_medium():
    assert resolve_policy(POLICY, "normal", "code", "medium") == "local-coder"


def test_resolve_urgent_general_medium():
    assert resolve_policy(POLICY, "urgent", "general", "medium") == "cloud-small"


def test_resolve_patient_code_hard():
    assert resolve_policy(POLICY, "patient", "code", "hard") == "local-large"
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_policy.py -v
```

- [ ] **Step 3: Implement**

`goorouter/policy.py`:
```python
from __future__ import annotations


def resolve_policy(
    policy: dict[str, dict[str, str]],
    urgency: str,
    domain: str,
    complexity: str,
) -> str:
    """Look up the destination backend name for (urgency, domain, complexity).

    Caller is responsible for having validated the policy table at config-load time.
    """
    return policy[urgency][f"{domain},{complexity}"]
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_policy.py -v && git commit -am "feat(policy): pure lookup helper"
```

---

## Phase 3 — Storage

### Task 3.1: SQLite schema + WAL + migration

**Files:**
- Create: `goorouter/migrations/0001_init.sql`
- Create: `goorouter/storage.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write the migration**

`goorouter/migrations/0001_init.sql`:
```sql
CREATE TABLE IF NOT EXISTS requests (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                              TEXT NOT NULL,
    request_id                      TEXT NOT NULL,
    model_field                     TEXT NOT NULL,
    prefixes_raw                    TEXT,
    pinned_backend                  TEXT,
    urgency_used                    TEXT NOT NULL,
    classifier_used                 TEXT,
    classifier_fallback_reason      TEXT,
    classifier_input_chars          INTEGER,
    classifier_input_truncated_from INTEGER,
    classifier_latency_ms           INTEGER,
    classifier_domain               TEXT,
    classifier_complexity           TEXT,
    classifier_reason               TEXT,
    backend_chosen                  TEXT NOT NULL,
    backend_latency_ms              INTEGER,
    tokens_in                       INTEGER,
    tokens_out                      INTEGER,
    success                         INTEGER NOT NULL,
    error_kind                      TEXT,
    prompt_content                  TEXT,
    prompt_storage_mode             TEXT NOT NULL,
    relabel_backend                 TEXT,
    relabel_ts                      TEXT,
    relabel_note                    TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts      ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_requests_backend ON requests(backend_chosen);
CREATE INDEX IF NOT EXISTS idx_requests_relabel ON requests(relabel_backend) WHERE relabel_backend IS NOT NULL;
```

- [ ] **Step 2: Failing test**

`tests/test_storage.py`:
```python
import sqlite3
from pathlib import Path

import pytest

from goorouter.storage import open_db, schema_version


def test_open_db_creates_schema(tmp_path: Path):
    db = tmp_path / "log.sqlite"
    conn = open_db(db)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='requests'")
    assert cursor.fetchone() is not None
    assert schema_version(conn) == 1


def test_open_db_enables_wal(tmp_path: Path):
    db = tmp_path / "log.sqlite"
    conn = open_db(db)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_open_db_idempotent(tmp_path: Path):
    db = tmp_path / "log.sqlite"
    open_db(db).close()
    # Second open shouldn't error
    conn = open_db(db)
    assert schema_version(conn) == 1
```

- [ ] **Step 3: Run, verify failure**

```bash
pytest tests/test_storage.py -v
```

- [ ] **Step 4: Implement**

`goorouter/storage.py`:
```python
from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path


SCHEMA_VERSION = 1


def _load_migration(name: str) -> str:
    return resources.files("goorouter.migrations").joinpath(name).read_text()


def open_db(path: Path) -> sqlite3.Connection:
    """Open or create the SQLite database. Enables WAL, runs migrations to current version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)  # autocommit; we manage txns explicitly
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _migrate(conn)
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _migrate(conn: sqlite3.Connection) -> None:
    current = schema_version(conn)
    if current < 1:
        conn.executescript(_load_migration("0001_init.sql"))
        conn.execute("PRAGMA user_version = 1")
```

Also create `goorouter/migrations/__init__.py` (empty) so `importlib.resources` finds the package.

```bash
touch goorouter/migrations/__init__.py
```

- [ ] **Step 5: Verify pass + commit**

```bash
pytest tests/test_storage.py -v && git add goorouter/storage.py goorouter/migrations tests/test_storage.py && git commit -m "feat(storage): SQLite schema, WAL, migration runner"
```

### Task 3.2: log_request

**Files:**
- Modify: `goorouter/storage.py`
- Modify: `tests/test_storage.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_storage.py`:
```python
import hashlib
from goorouter.storage import LogRow, log_request


def _row(**overrides) -> LogRow:
    base = LogRow(
        request_id="r-1",
        model_field="goo-auto",
        prefixes_raw=None,
        pinned_backend=None,
        urgency_used="normal",
        classifier_used="local-small",
        classifier_fallback_reason=None,
        classifier_input_chars=200,
        classifier_input_truncated_from=None,
        classifier_latency_ms=180,
        classifier_domain="code",
        classifier_complexity="medium",
        classifier_reason="standard refactor",
        backend_chosen="local-coder",
        backend_latency_ms=4200,
        tokens_in=120,
        tokens_out=480,
        success=True,
        error_kind=None,
        prompt_content="please refactor this",
        prompt_storage_mode="full",
    )
    return base if not overrides else LogRow(**{**base.__dict__, **overrides})


def test_log_request_full(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    row_id = log_request(conn, _row())
    assert row_id == 1
    r = conn.execute("SELECT prompt_content, prompt_storage_mode, success FROM requests").fetchone()
    assert r == ("please refactor this", "full", 1)


def test_log_request_hashed_storage(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(prompt_storage_mode="hashed"))
    r = conn.execute("SELECT prompt_content, prompt_storage_mode FROM requests").fetchone()
    assert len(r[0]) == 64
    assert r[0] == hashlib.sha256(b"please refactor this").hexdigest()
    assert r[1] == "hashed"


def test_log_request_none_storage(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(prompt_storage_mode="none"))
    r = conn.execute("SELECT prompt_content, prompt_storage_mode FROM requests").fetchone()
    assert r == (None, "none")
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_storage.py -v
```

- [ ] **Step 3: Implement**

Append to `goorouter/storage.py`:
```python
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class LogRow:
    request_id: str
    model_field: str
    prefixes_raw: str | None
    pinned_backend: str | None
    urgency_used: str
    classifier_used: str | None
    classifier_fallback_reason: str | None
    classifier_input_chars: int | None
    classifier_input_truncated_from: int | None
    classifier_latency_ms: int | None
    classifier_domain: str | None
    classifier_complexity: str | None
    classifier_reason: str | None
    backend_chosen: str
    backend_latency_ms: int | None
    tokens_in: int | None
    tokens_out: int | None
    success: bool
    error_kind: str | None
    prompt_content: str | None
    prompt_storage_mode: str  # "full" | "hashed" | "none"


def _apply_storage_mode(content: str | None, mode: str) -> str | None:
    if mode == "full":
        return content
    if mode == "hashed":
        return hashlib.sha256((content or "").encode("utf-8")).hexdigest()
    if mode == "none":
        return None
    raise ValueError(f"unknown prompt_storage_mode: {mode}")


def log_request(conn: sqlite3.Connection, row: LogRow) -> int:
    """Insert a request log row. Returns the new id."""
    stored = _apply_storage_mode(row.prompt_content, row.prompt_storage_mode)
    cursor = conn.execute(
        """
        INSERT INTO requests (
            ts, request_id, model_field, prefixes_raw, pinned_backend, urgency_used,
            classifier_used, classifier_fallback_reason, classifier_input_chars,
            classifier_input_truncated_from, classifier_latency_ms, classifier_domain,
            classifier_complexity, classifier_reason, backend_chosen, backend_latency_ms,
            tokens_in, tokens_out, success, error_kind, prompt_content, prompt_storage_mode
        ) VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?
        )
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            row.request_id, row.model_field, row.prefixes_raw, row.pinned_backend,
            row.urgency_used, row.classifier_used, row.classifier_fallback_reason,
            row.classifier_input_chars, row.classifier_input_truncated_from,
            row.classifier_latency_ms, row.classifier_domain, row.classifier_complexity,
            row.classifier_reason, row.backend_chosen, row.backend_latency_ms,
            row.tokens_in, row.tokens_out, 1 if row.success else 0, row.error_kind,
            stored, row.prompt_storage_mode,
        ),
    )
    return int(cursor.lastrowid or 0)
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_storage.py -v && git commit -am "feat(storage): log_request with prompt-storage modes"
```

### Task 3.3: Queries (recent, by id) and relabel

**Files:**
- Modify: `goorouter/storage.py`
- Modify: `tests/test_storage.py`

- [ ] **Step 1: Failing tests**

Append to `tests/test_storage.py`:
```python
from goorouter.storage import get_recent, get_by_id, relabel_last, relabel_by_id, RelabelError


def test_get_recent_orders_by_ts_desc(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(request_id="r-1"))
    log_request(conn, _row(request_id="r-2"))
    log_request(conn, _row(request_id="r-3"))
    rows = get_recent(conn, limit=2)
    ids = [r["request_id"] for r in rows]
    assert ids == ["r-3", "r-2"]


def test_get_recent_filtered_by_backend(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(backend_chosen="local-coder"))
    log_request(conn, _row(backend_chosen="cloud-large"))
    rows = get_recent(conn, limit=10, backend="local-coder")
    assert len(rows) == 1


def test_get_by_id(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    rid = log_request(conn, _row())
    row = get_by_id(conn, rid)
    assert row is not None
    assert row["id"] == rid


def test_relabel_last(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(request_id="a"))
    log_request(conn, _row(request_id="b"))
    relabel_last(conn, "cloud-large", note="should have been bigger")
    r = conn.execute("SELECT request_id, relabel_backend, relabel_note FROM requests ORDER BY id DESC").fetchone()
    assert r == ("b", "cloud-large", "should have been bigger")


def test_relabel_by_id(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    rid = log_request(conn, _row())
    relabel_by_id(conn, rid, "local-coder", note=None)
    r = get_by_id(conn, rid)
    assert r is not None and r["relabel_backend"] == "local-coder"


def test_relabel_no_rows_raises(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    with pytest.raises(RelabelError):
        relabel_last(conn, "cloud-large", note=None)
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_storage.py -v
```

- [ ] **Step 3: Implement**

Append to `goorouter/storage.py`:
```python
class RelabelError(ValueError):
    pass


def _row_to_dict(cursor: sqlite3.Cursor, row: tuple) -> dict:
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


def get_recent(conn: sqlite3.Connection, limit: int = 20, backend: str | None = None) -> list[dict]:
    if backend is None:
        cursor = conn.execute("SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,))
    else:
        cursor = conn.execute(
            "SELECT * FROM requests WHERE backend_chosen = ? ORDER BY id DESC LIMIT ?",
            (backend, limit),
        )
    return [_row_to_dict(cursor, r) for r in cursor.fetchall()]


def get_by_id(conn: sqlite3.Connection, row_id: int) -> dict | None:
    cursor = conn.execute("SELECT * FROM requests WHERE id = ?", (row_id,))
    r = cursor.fetchone()
    return _row_to_dict(cursor, r) if r else None


def _set_relabel(conn: sqlite3.Connection, row_id: int, backend: str, note: str | None) -> None:
    conn.execute(
        "UPDATE requests SET relabel_backend = ?, relabel_ts = ?, relabel_note = ? WHERE id = ?",
        (backend, datetime.now(timezone.utc).isoformat(), note, row_id),
    )


def relabel_last(conn: sqlite3.Connection, backend: str, note: str | None) -> int:
    cursor = conn.execute("SELECT MAX(id) FROM requests")
    last_id = cursor.fetchone()[0]
    if last_id is None:
        raise RelabelError("no rows in requests table to relabel")
    _set_relabel(conn, last_id, backend, note)
    return int(last_id)


def relabel_by_id(conn: sqlite3.Connection, row_id: int, backend: str, note: str | None) -> None:
    if get_by_id(conn, row_id) is None:
        raise RelabelError(f"no row with id {row_id}")
    _set_relabel(conn, row_id, backend, note)
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_storage.py -v && git commit -am "feat(storage): get_recent, get_by_id, relabel_last/by_id"
```

---

## Phase 4 — Backends

### Task 4.1: Backend dispatch via litellm SDK

**Files:**
- Create: `goorouter/backends.py`
- Test: `tests/test_backends.py`

- [ ] **Step 1: Failing test**

`tests/test_backends.py`:
```python
from unittest.mock import AsyncMock, patch

import pytest

from goorouter.backends import call_backend
from goorouter.config import BackendConfig


@pytest.fixture
def cloud_backend() -> BackendConfig:
    return BackendConfig(
        name="cloud-large", provider="anthropic", model="claude-opus-4-7",
        api_key_env="ANTHROPIC_API_KEY", api_key=None, base_url=None,
        aliases=("opus",), timeout_s=120,
    )


@pytest.fixture
def local_backend() -> BackendConfig:
    return BackendConfig(
        name="local-small", provider="openai", model="qwen2.5-3b-instruct",
        api_key_env=None, api_key="lm-studio",
        base_url="http://localhost:1234/v1", aliases=(), timeout_s=60,
    )


async def test_call_anthropic_backend_translates_model(cloud_backend, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    mock = AsyncMock(return_value={"choices": [{"message": {"content": "hi"}}]})
    with patch("goorouter.backends.litellm.acompletion", mock):
        await call_backend(cloud_backend, messages=[{"role": "user", "content": "hi"}], stream=False)
    args, kwargs = mock.call_args
    # litellm anthropic models get the "anthropic/" prefix
    assert kwargs["model"] == "anthropic/claude-opus-4-7"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["timeout"] == 120
    assert kwargs["stream"] is False


async def test_call_openai_compatible_with_base_url(local_backend):
    mock = AsyncMock(return_value={"choices": []})
    with patch("goorouter.backends.litellm.acompletion", mock):
        await call_backend(local_backend, messages=[{"role": "user", "content": "x"}], stream=False)
    args, kwargs = mock.call_args
    # OpenAI-compatible: just the model name (no provider prefix)
    assert kwargs["model"] == "qwen2.5-3b-instruct"
    assert kwargs["api_base"] == "http://localhost:1234/v1"
    assert kwargs["api_key"] == "lm-studio"


async def test_passes_through_tools(cloud_backend, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    mock = AsyncMock(return_value={"choices": []})
    tools = [{"type": "function", "function": {"name": "search"}}]
    with patch("goorouter.backends.litellm.acompletion", mock):
        await call_backend(
            cloud_backend, messages=[{"role": "user", "content": "x"}],
            stream=False, tools=tools, tool_choice="auto",
        )
    kwargs = mock.call_args.kwargs
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_backends.py -v
```

- [ ] **Step 3: Implement**

`goorouter/backends.py`:
```python
from __future__ import annotations

import os
from typing import Any

import litellm

from goorouter.config import BackendConfig


def _resolve_api_key(b: BackendConfig) -> str | None:
    if b.api_key:
        return b.api_key
    if b.api_key_env:
        return os.environ.get(b.api_key_env)
    return None


def _resolve_model_id(b: BackendConfig) -> str:
    """Translate (provider, model) → litellm model id."""
    if b.provider == "anthropic":
        return f"anthropic/{b.model}"
    # OpenAI-compatible: use bare model name; api_base directs the call
    return b.model


async def call_backend(
    backend: BackendConfig,
    *,
    messages: list[dict[str, Any]],
    stream: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    extra_params: dict[str, Any] | None = None,
) -> Any:
    """Dispatch a chat completion to a backend via the LiteLLM SDK.

    For stream=True, returns an async generator of chunks (litellm's CustomStreamWrapper).
    For stream=False, returns the response object.
    """
    kwargs: dict[str, Any] = {
        "model": _resolve_model_id(backend),
        "messages": messages,
        "stream": stream,
        "timeout": backend.timeout_s,
        "api_key": _resolve_api_key(backend),
    }
    if backend.base_url:
        kwargs["api_base"] = backend.base_url
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if extra_params:
        for k, v in extra_params.items():
            kwargs.setdefault(k, v)
    return await litellm.acompletion(**kwargs)
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_backends.py -v && git commit -am "feat(backends): dispatch via litellm SDK with provider translation"
```

---

## Phase 5 — Classifier

### Task 5.1: Default classifier prompt template

**Files:**
- Create: `goorouter/classifier_prompt.txt`

- [ ] **Step 1: Write the prompt**

`goorouter/classifier_prompt.txt`:
```
You are a routing classifier for an LLM router. Given the latest user prompt, classify it on two axes and return JSON only — no prose, no markdown fences.

Schema:
{"domain": "code" | "general", "complexity": "trivial" | "medium" | "hard", "reason": "<one short sentence>"}

Domain:
- "code": writing, debugging, explaining, designing, or reasoning about source code in any language.
- "general": everything else (questions, analysis, planning, prose, reasoning).

Complexity:
- "trivial": one-line answer, simple lookup, obvious format conversion.
- "medium": needs a few sentences or a small code block; standard well-scoped task.
- "hard": multi-step reasoning, novel problem, large refactor, ambiguous spec, deep analysis.

User prompt:
"""
{prompt}
"""

Respond with JSON only.
```

- [ ] **Step 2: Commit**

```bash
git add goorouter/classifier_prompt.txt
git commit -m "feat(classifier): default prompt template"
```

### Task 5.2: Classifier call + JSON parsing

**Files:**
- Create: `goorouter/classifier.py`
- Test: `tests/test_classifier.py`

- [ ] **Step 1: Failing test**

`tests/test_classifier.py`:
```python
import json
from unittest.mock import AsyncMock, patch

import pytest

from goorouter.classifier import (
    ClassifierError,
    ClassifierResult,
    classify,
    load_prompt_template,
)
from goorouter.config import BackendConfig


def _backend() -> BackendConfig:
    return BackendConfig(
        name="local-small", provider="openai", model="qwen2.5-3b-instruct",
        api_key_env=None, api_key="lm-studio",
        base_url="http://localhost:1234/v1", aliases=(), timeout_s=5,
    )


def _response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


async def test_classify_parses_json():
    payload = json.dumps({"domain": "code", "complexity": "medium", "reason": "ok"})
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=_response(payload))):
        result = await classify(_backend(), prompt="refactor X", template=load_prompt_template(None))
    assert result == ClassifierResult(domain="code", complexity="medium", reason="ok",
                                       latency_ms=result.latency_ms)


async def test_classify_strips_code_fences():
    payload = '```json\n{"domain":"general","complexity":"trivial","reason":"hi"}\n```'
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=_response(payload))):
        result = await classify(_backend(), prompt="hi", template=load_prompt_template(None))
    assert result.domain == "general"
    assert result.complexity == "trivial"


async def test_classify_invalid_json_raises():
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=_response("not json"))):
        with pytest.raises(ClassifierError):
            await classify(_backend(), prompt="x", template=load_prompt_template(None))


async def test_classify_invalid_enum_raises():
    payload = json.dumps({"domain": "wat", "complexity": "medium", "reason": "."})
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=_response(payload))):
        with pytest.raises(ClassifierError):
            await classify(_backend(), prompt="x", template=load_prompt_template(None))
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_classifier.py -v
```

- [ ] **Step 3: Implement**

`goorouter/classifier.py`:
```python
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from goorouter.backends import call_backend
from goorouter.config import BackendConfig

VALID_DOMAINS = {"code", "general"}
VALID_COMPLEXITIES = {"trivial", "medium", "hard"}


class ClassifierError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClassifierResult:
    domain: str
    complexity: str
    reason: str
    latency_ms: int


def load_prompt_template(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return resources.files("goorouter").joinpath("classifier_prompt.txt").read_text(encoding="utf-8")


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _extract_json(text: str) -> Any:
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ClassifierError(f"classifier did not return parseable JSON: {text!r}") from e


async def classify(backend: BackendConfig, *, prompt: str, template: str) -> ClassifierResult:
    """Run the classifier on `prompt`. Returns ClassifierResult or raises ClassifierError."""
    rendered = template.replace("{prompt}", prompt)
    started = time.monotonic()
    try:
        response = await call_backend(
            backend,
            messages=[{"role": "user", "content": rendered}],
            stream=False,
        )
    except Exception as e:
        raise ClassifierError(f"classifier call failed: {e}") from e
    latency_ms = int((time.monotonic() - started) * 1000)

    content = response["choices"][0]["message"]["content"] if isinstance(response, dict) else \
              response.choices[0].message.content
    parsed = _extract_json(content)
    if not isinstance(parsed, dict):
        raise ClassifierError(f"classifier JSON not an object: {parsed!r}")
    domain = parsed.get("domain")
    complexity = parsed.get("complexity")
    reason = parsed.get("reason", "")
    if domain not in VALID_DOMAINS:
        raise ClassifierError(f"classifier domain '{domain}' not in {VALID_DOMAINS}")
    if complexity not in VALID_COMPLEXITIES:
        raise ClassifierError(f"classifier complexity '{complexity}' not in {VALID_COMPLEXITIES}")
    return ClassifierResult(
        domain=domain, complexity=complexity, reason=str(reason), latency_ms=latency_ms,
    )
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_classifier.py -v && git commit -am "feat(classifier): call + JSON parse with validation"
```

### Task 5.3: Classifier fallback chain

**Files:**
- Modify: `goorouter/classifier.py`
- Modify: `tests/test_classifier.py`

- [ ] **Step 1: Failing tests**

Append to `tests/test_classifier.py`:
```python
from goorouter.classifier import classify_with_fallback, FallbackOutcome


def _other_backend() -> BackendConfig:
    return BackendConfig(
        name="local-large", provider="openai", model="qwen2.5-32b-instruct",
        api_key_env=None, api_key="lm-studio",
        base_url="http://localhost:1234/v1", aliases=(), timeout_s=180,
    )


async def test_oversize_skips_primary_uses_fallback():
    primary_mock = AsyncMock()  # should not be called
    payload = json.dumps({"domain": "code", "complexity": "medium", "reason": "ok"})
    fb_mock = AsyncMock(return_value=_response(payload))

    async def fake_call(b, **kw):
        return await (primary_mock(b, **kw) if b.name == "local-small" else fb_mock(b, **kw))

    with patch("goorouter.classifier.call_backend", side_effect=fake_call):
        outcome = await classify_with_fallback(
            primary=_backend(), fallback=_other_backend(),
            prompt="x" * 12000, max_input_chars=8000,
            template=load_prompt_template(None),
        )
    assert outcome.result is not None
    assert outcome.fallback_reason == "oversize"
    assert outcome.classifier_used == "local-large"
    assert outcome.input_chars == 12000
    assert outcome.input_truncated_from is None
    primary_mock.assert_not_called()
    fb_mock.assert_called_once()


async def test_primary_error_triggers_fallback():
    payload = json.dumps({"domain": "general", "complexity": "trivial", "reason": "."})

    async def fake_call(b, **kw):
        if b.name == "local-small":
            raise RuntimeError("boom")
        return _response(payload)

    with patch("goorouter.classifier.call_backend", side_effect=fake_call):
        outcome = await classify_with_fallback(
            primary=_backend(), fallback=_other_backend(),
            prompt="hi", max_input_chars=8000,
            template=load_prompt_template(None),
        )
    assert outcome.result is not None
    assert outcome.fallback_reason == "primary_error"
    assert outcome.classifier_used == "local-large"


async def test_no_fallback_oversize_truncates():
    payload = json.dumps({"domain": "code", "complexity": "medium", "reason": "."})
    seen_lengths: list[int] = []

    async def fake_call(b, *, messages, **kw):
        seen_lengths.append(len(messages[0]["content"]))
        return _response(payload)

    with patch("goorouter.classifier.call_backend", side_effect=fake_call):
        outcome = await classify_with_fallback(
            primary=_backend(), fallback=None,
            prompt="x" * 12000, max_input_chars=8000,
            template=load_prompt_template(None),
        )
    assert outcome.fallback_reason is None
    assert outcome.input_chars == 8000
    assert outcome.input_truncated_from == 12000
    # The rendered template includes the (truncated) prompt embedded
    assert seen_lengths[0] > 0


async def test_both_fail_returns_none_result():
    async def fake_call(b, **kw):
        raise RuntimeError("everything down")

    with patch("goorouter.classifier.call_backend", side_effect=fake_call):
        outcome = await classify_with_fallback(
            primary=_backend(), fallback=_other_backend(),
            prompt="x", max_input_chars=8000,
            template=load_prompt_template(None),
        )
    assert outcome.result is None
    assert outcome.fallback_reason == "primary_error"
    assert outcome.classifier_used is None
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_classifier.py -v
```

- [ ] **Step 3: Implement**

Append to `goorouter/classifier.py`:
```python
@dataclass(frozen=True)
class FallbackOutcome:
    result: ClassifierResult | None
    classifier_used: str | None
    fallback_reason: str | None  # "oversize" | "primary_error" | None
    input_chars: int             # chars actually sent to whichever classifier ran
    input_truncated_from: int | None


async def classify_with_fallback(
    *,
    primary: BackendConfig,
    fallback: BackendConfig | None,
    prompt: str,
    max_input_chars: int,
    template: str,
) -> FallbackOutcome:
    original_len = len(prompt)

    # Case 1: oversize and fallback configured → skip primary, full prompt to fallback
    if original_len > max_input_chars and fallback is not None:
        try:
            result = await classify(fallback, prompt=prompt, template=template)
            return FallbackOutcome(
                result=result, classifier_used=fallback.name,
                fallback_reason="oversize", input_chars=original_len,
                input_truncated_from=None,
            )
        except ClassifierError:
            return FallbackOutcome(
                result=None, classifier_used=None,
                fallback_reason="oversize", input_chars=original_len,
                input_truncated_from=None,
            )

    # Case 2: oversize but no fallback → head-truncate, try primary
    if original_len > max_input_chars and fallback is None:
        truncated = prompt[:max_input_chars]
        try:
            result = await classify(primary, prompt=truncated, template=template)
            return FallbackOutcome(
                result=result, classifier_used=primary.name,
                fallback_reason=None, input_chars=max_input_chars,
                input_truncated_from=original_len,
            )
        except ClassifierError:
            return FallbackOutcome(
                result=None, classifier_used=None,
                fallback_reason="primary_error", input_chars=max_input_chars,
                input_truncated_from=original_len,
            )

    # Case 3: fits → primary; fall back on error if configured
    try:
        result = await classify(primary, prompt=prompt, template=template)
        return FallbackOutcome(
            result=result, classifier_used=primary.name,
            fallback_reason=None, input_chars=original_len,
            input_truncated_from=None,
        )
    except ClassifierError:
        if fallback is None:
            return FallbackOutcome(
                result=None, classifier_used=None,
                fallback_reason="primary_error", input_chars=original_len,
                input_truncated_from=None,
            )
        try:
            result = await classify(fallback, prompt=prompt, template=template)
            return FallbackOutcome(
                result=result, classifier_used=fallback.name,
                fallback_reason="primary_error", input_chars=original_len,
                input_truncated_from=None,
            )
        except ClassifierError:
            return FallbackOutcome(
                result=None, classifier_used=None,
                fallback_reason="primary_error", input_chars=original_len,
                input_truncated_from=None,
            )
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_classifier.py -v && git commit -am "feat(classifier): fallback chain for oversize and errors"
```

---

## Phase 6 — Router orchestrator

### Task 6.1: Routing decision (no I/O)

**Files:**
- Create: `goorouter/router.py`
- Test: `tests/test_router.py`

This task isolates the *decision* logic (which backend, why) from the *dispatch* logic (calling that backend). The decision step is pure-ish (it calls the classifier but no destination backend), so it's testable with mocks.

- [ ] **Step 1: Failing tests**

`tests/test_router.py`:
```python
import json
from unittest.mock import AsyncMock, patch

import pytest

from goorouter.config import (
    BackendConfig, Config, ServerConfig, ClassifierConfig, RoutingConfig, LoggingConfig
)
from goorouter.router import RoutingDecision, decide_route


def _cfg() -> Config:
    backends = {
        name: BackendConfig(
            name=name, provider="openai", model=name,
            api_key_env=None, api_key="lm", base_url="http://x", aliases=(), timeout_s=60,
        )
        for name in ("cloud-large", "cloud-small", "local-large", "local-small", "local-coder")
    }
    backends["cloud-large"] = BackendConfig(
        name="cloud-large", provider="anthropic", model="claude-opus-4-7",
        api_key_env="ANTHROPIC_API_KEY", api_key=None, base_url=None,
        aliases=("opus",), timeout_s=120,
    )
    policy = {
        u: {f"{d},{c}": "local-coder" if d == "code" else "local-small"
            for d in ("code", "general") for c in ("trivial", "medium", "hard")}
        for u in ("normal", "urgent", "patient")
    }
    policy["normal"]["code,hard"] = "cloud-large"
    policy["urgent"]["general,medium"] = "cloud-small"
    return Config(
        server=ServerConfig(host="127.0.0.1", port=4000),
        backends=backends,
        classifier=ClassifierConfig(backend="local-small", fallback_backend=None,
                                     max_input_chars=8000, timeout_s=5,
                                     prompt_template_path=None),
        routing=RoutingConfig(default_urgency="normal", default_on_failure="cloud-large",
                              policy=policy),
        logging=LoggingConfig(db_path=":memory:", prompt_storage="full"),
    )


async def test_decide_explain_mode():
    cfg = _cfg()
    decision = await decide_route(
        cfg=cfg, model_field="goo-explain",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert decision.mode == "explain"
    assert decision.backend == "cloud-large"  # would-be-routed backend (general/trivial/normal)


async def test_decide_pinned_by_model_field():
    cfg = _cfg()
    decision = await decide_route(
        cfg=cfg, model_field="goo-cloud-large",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert decision.mode == "dispatch"
    assert decision.backend == "cloud-large"
    assert decision.classifier_result is None
    assert decision.pinned_backend == "cloud-large"


async def test_decide_pinned_by_prefix_overrides_model_field():
    cfg = _cfg()
    decision = await decide_route(
        cfg=cfg, model_field="goo-cloud-large",
        messages=[{"role": "user", "content": "!local-small foo"}],
    )
    assert decision.backend == "local-small"
    assert decision.stripped_last_user == "foo"


async def test_decide_auto_runs_classifier():
    cfg = _cfg()
    payload = json.dumps({"domain": "code", "complexity": "hard", "reason": "novel refactor"})
    response = {"choices": [{"message": {"content": payload}}]}
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=response)):
        decision = await decide_route(
            cfg=cfg, model_field="goo-auto",
            messages=[{"role": "user", "content": "rewrite this thing"}],
        )
    assert decision.backend == "cloud-large"  # policy.normal["code,hard"]
    assert decision.classifier_result is not None
    assert decision.classifier_result.domain == "code"


async def test_decide_urgency_prefix_changes_policy():
    cfg = _cfg()
    payload = json.dumps({"domain": "general", "complexity": "medium", "reason": "."})
    response = {"choices": [{"message": {"content": payload}}]}
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=response)):
        decision = await decide_route(
            cfg=cfg, model_field="goo-auto",
            messages=[{"role": "user", "content": "!urgent help"}],
        )
    # policy.urgent["general,medium"] = cloud-small
    assert decision.backend == "cloud-small"
    assert decision.urgency == "urgent"


async def test_decide_classifier_failure_uses_default_on_failure():
    cfg = _cfg()
    with patch("goorouter.classifier.call_backend",
               AsyncMock(side_effect=RuntimeError("down"))):
        decision = await decide_route(
            cfg=cfg, model_field="goo-auto",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert decision.backend == "cloud-large"  # default_on_failure
    assert decision.classifier_result is None


async def test_decide_unknown_prefix_raises():
    from goorouter.prefixes import UnknownPrefixError
    cfg = _cfg()
    with pytest.raises(UnknownPrefixError):
        await decide_route(
            cfg=cfg, model_field="goo-auto",
            messages=[{"role": "user", "content": "!doesnotexist hi"}],
        )


async def test_decide_multimodal_routes_to_default_on_failure():
    cfg = _cfg()
    decision = await decide_route(
        cfg=cfg, model_field="goo-auto",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]}],
    )
    assert decision.backend == "cloud-large"  # default_on_failure
    assert decision.multimodal is True


async def test_decide_empty_messages_defaults_general_trivial():
    cfg = _cfg()
    decision = await decide_route(cfg=cfg, model_field="goo-auto", messages=[])
    # policy.normal["general,trivial"] = local-small
    assert decision.backend == "local-small"
    assert decision.classifier_result is None
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_router.py -v
```

- [ ] **Step 3: Implement**

`goorouter/router.py`:
```python
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from goorouter.classifier import (
    ClassifierResult, FallbackOutcome, classify_with_fallback, load_prompt_template,
)
from goorouter.config import Config, resolve_backend
from goorouter.policy import resolve_policy
from goorouter.prefixes import ParsedPrefixes, parse_prefixes


GOO_AUTO = "goo-auto"
GOO_EXPLAIN = "goo-explain"
GOO_PREFIX = "goo-"


@dataclass(frozen=True)
class RoutingDecision:
    request_id: str
    mode: Literal["dispatch", "explain"]
    backend: str
    pinned_backend: str | None
    urgency: str
    parsed: ParsedPrefixes
    classifier_outcome: FallbackOutcome | None
    classifier_result: ClassifierResult | None
    multimodal: bool
    model_field: str
    last_user_content_original: str | None
    stripped_last_user: str | None


def _last_user_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m
    return None


def _is_multimodal_content(content: Any) -> bool:
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") not in (None, "text"):
                return True
    return False


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(parts)
    return ""


async def decide_route(
    *,
    cfg: Config,
    model_field: str,
    messages: list[dict[str, Any]],
) -> RoutingDecision:
    request_id = str(uuid.uuid4())

    # Determine if explain mode
    mode: Literal["dispatch", "explain"] = "explain" if model_field == GOO_EXPLAIN else "dispatch"

    # Examine last user message
    last_user = _last_user_message(messages)
    multimodal = bool(last_user and _is_multimodal_content(last_user.get("content")))
    last_text_original = _content_text(last_user.get("content")) if last_user else None

    # Build {urgencies}, {backend → aliases} for prefix parsing
    urgencies = {"urgent", "patient", "normal"}
    backend_alias_map = {b.name: set(b.aliases) for b in cfg.backends.values()}

    parsed = ParsedPrefixes(urgency=None, pinned_backend=None,
                             stripped=last_text_original or "", raw="")
    if last_text_original is not None and not multimodal:
        parsed = parse_prefixes(last_text_original, urgencies, backend_alias_map)

    # Pin precedence: prefix > model field
    pinned_via_model: str | None = None
    if model_field.startswith(GOO_PREFIX) and model_field not in (GOO_AUTO, GOO_EXPLAIN):
        candidate = model_field[len(GOO_PREFIX):]
        if candidate in cfg.backends:
            pinned_via_model = candidate

    pinned = parsed.pinned_backend or pinned_via_model

    # Effective urgency
    urgency = parsed.urgency or cfg.routing.default_urgency

    # Multimodal short-circuit (regardless of pin? we honor pin if set)
    if multimodal and pinned is None:
        return RoutingDecision(
            request_id=request_id, mode=mode,
            backend=cfg.routing.default_on_failure,
            pinned_backend=None, urgency=urgency, parsed=parsed,
            classifier_outcome=None, classifier_result=None,
            multimodal=True, model_field=model_field,
            last_user_content_original=last_text_original,
            stripped_last_user=parsed.stripped if last_text_original else None,
        )

    # Pinned → no classifier
    if pinned is not None:
        return RoutingDecision(
            request_id=request_id, mode=mode, backend=pinned,
            pinned_backend=pinned, urgency=urgency, parsed=parsed,
            classifier_outcome=None, classifier_result=None,
            multimodal=multimodal, model_field=model_field,
            last_user_content_original=last_text_original,
            stripped_last_user=parsed.stripped if last_text_original else None,
        )

    # Empty / no user message: skip classifier, treat as (general, trivial)
    if last_user is None or not parsed.stripped.strip():
        backend = resolve_policy(cfg.routing.policy, urgency, "general", "trivial")
        return RoutingDecision(
            request_id=request_id, mode=mode, backend=backend,
            pinned_backend=None, urgency=urgency, parsed=parsed,
            classifier_outcome=None, classifier_result=None,
            multimodal=multimodal, model_field=model_field,
            last_user_content_original=last_text_original,
            stripped_last_user=parsed.stripped if last_text_original else None,
        )

    # Run classifier (with fallback chain)
    primary = cfg.backends[cfg.classifier.backend]
    fallback = cfg.backends[cfg.classifier.fallback_backend] if cfg.classifier.fallback_backend else None
    template = load_prompt_template(cfg.classifier.prompt_template_path)
    outcome = await classify_with_fallback(
        primary=primary, fallback=fallback,
        prompt=parsed.stripped,
        max_input_chars=cfg.classifier.max_input_chars,
        template=template,
    )

    if outcome.result is None:
        backend = cfg.routing.default_on_failure
    else:
        backend = resolve_policy(cfg.routing.policy, urgency,
                                  outcome.result.domain, outcome.result.complexity)
    return RoutingDecision(
        request_id=request_id, mode=mode, backend=backend,
        pinned_backend=None, urgency=urgency, parsed=parsed,
        classifier_outcome=outcome, classifier_result=outcome.result,
        multimodal=multimodal, model_field=model_field,
        last_user_content_original=last_text_original,
        stripped_last_user=parsed.stripped,
    )
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_router.py -v && git commit -am "feat(router): decide_route orchestrator (no dispatch yet)"
```

### Task 6.2: Substituting stripped content + dispatch

**Files:**
- Modify: `goorouter/router.py` (add `apply_stripping`, `dispatch`)
- Modify: `tests/test_router.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_router.py`:
```python
from goorouter.router import apply_stripping


def test_apply_stripping_replaces_last_user_content():
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "!urgent please"},
    ]
    out = apply_stripping(messages, stripped_last_user="please")
    assert out is not messages  # new list
    assert out[-1]["content"] == "please"
    assert out[-3]["content"] == "earlier"


def test_apply_stripping_no_user_message_unchanged():
    messages = [{"role": "system", "content": "x"}]
    out = apply_stripping(messages, stripped_last_user=None)
    assert out == messages
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_router.py::test_apply_stripping_replaces_last_user_content -v
```

- [ ] **Step 3: Implement**

Append to `goorouter/router.py`:
```python
def apply_stripping(
    messages: list[dict[str, Any]], stripped_last_user: str | None,
) -> list[dict[str, Any]]:
    """Return a new messages list with the last user message's content replaced by `stripped_last_user`."""
    if stripped_last_user is None:
        return list(messages)
    out = [dict(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i]["content"] = stripped_last_user
            break
    return out
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_router.py -v && git commit -am "feat(router): apply_stripping helper for outgoing messages"
```

---

## Phase 7 — Explain mode formatter

### Task 7.1: Format a RoutingDecision as text

**Files:**
- Create: `goorouter/explain.py`
- Test: `tests/test_explain.py`

- [ ] **Step 1: Failing test**

`tests/test_explain.py`:
```python
import json
from unittest.mock import AsyncMock, patch

from goorouter.explain import format_decision
from goorouter.router import decide_route
from tests.test_router import _cfg


async def test_format_decision_explain():
    cfg = _cfg()
    payload = json.dumps({"domain": "code", "complexity": "medium", "reason": "scoped refactor"})
    response = {"choices": [{"message": {"content": payload}}]}
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=response)):
        decision = await decide_route(
            cfg=cfg, model_field="goo-explain",
            messages=[{"role": "user", "content": "!urgent rewrite x"}],
        )
    text = format_decision(decision, cfg)
    assert "Routing decision" in text
    assert "urgent" in text
    assert "code" in text and "medium" in text
    assert "scoped refactor" in text
    # The chosen backend appears
    assert decision.backend in text
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_explain.py -v
```

- [ ] **Step 3: Implement**

`goorouter/explain.py`:
```python
from __future__ import annotations

from goorouter.config import Config
from goorouter.router import RoutingDecision


def format_decision(decision: RoutingDecision, cfg: Config) -> str:
    lines: list[str] = []
    lines.append(f"Routing decision: {decision.backend}")
    backend = cfg.backends.get(decision.backend)
    if backend:
        target = backend.model
        provider = backend.provider
        lines.append(f"  Target: {provider} {target}")
    lines.append("")

    lines.append(f"Prefixes parsed:    {decision.parsed.raw or '(none)'}")
    lines.append(f"Effective urgency:  {decision.urgency}")

    if decision.pinned_backend:
        lines.append(f"Pinned backend:     {decision.pinned_backend} (classifier skipped)")
    elif decision.multimodal:
        lines.append(f"Multimodal content: routed to default_on_failure ({cfg.routing.default_on_failure})")
    elif decision.classifier_outcome is None:
        lines.append("Classifier:         skipped (empty or no user message)")
    else:
        out = decision.classifier_outcome
        lines.append(f"Classifier used:    {out.classifier_used or '(both failed)'}")
        if out.fallback_reason:
            lines.append(f"Fallback reason:    {out.fallback_reason}")
        lines.append(f"Classifier input:   {out.input_chars} chars" + (
            f" (truncated from {out.input_truncated_from})"
            if out.input_truncated_from else ""
        ))
        if out.result:
            lines.append(f"Classifier output:  domain={out.result.domain} "
                          f"complexity={out.result.complexity}")
            lines.append(f"Classifier reason:  {out.result.reason}")
            lines.append(f"Classifier latency: {out.result.latency_ms}ms")
            lines.append(f"Policy lookup:      "
                          f"policy.{decision.urgency}[\"{out.result.domain},{out.result.complexity}\"]"
                          f" = {decision.backend}")
        else:
            lines.append(f"All classifiers failed; using default_on_failure → {decision.backend}")

    lines.append("")
    lines.append("(No tokens consumed at the destination backend.)")
    return "\n".join(lines)
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_explain.py -v && git commit -am "feat(explain): format RoutingDecision as readable text"
```

---

## Phase 8 — HTTP server

### Task 8.1: FastAPI app + /v1/models

**Files:**
- Create: `goorouter/server.py`
- Test: `tests/test_server.py`

- [ ] **Step 1: Failing test**

`tests/test_server.py`:
```python
from fastapi.testclient import TestClient

from goorouter.server import build_app
from tests.test_router import _cfg


def test_v1_models_lists_virtual_and_backends(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    ids = {m["id"] for m in data["data"]}
    expected = {"goo-auto", "goo-explain"} | {f"goo-{name}" for name in cfg.backends}
    assert ids == expected
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_server.py -v
```

- [ ] **Step 3: Implement**

`goorouter/server.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI

from goorouter.config import Config
from goorouter.storage import open_db


def build_app(cfg: Config, *, db_path: Path) -> FastAPI:
    app = FastAPI(title="goorouter", version="0.1.0")
    app.state.cfg = cfg
    app.state.db = open_db(db_path)

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        ids = ["goo-auto", "goo-explain", *(f"goo-{name}" for name in cfg.backends)]
        return {
            "object": "list",
            "data": [
                {"id": mid, "object": "model", "owned_by": "goorouter", "created": 0}
                for mid in ids
            ],
        }

    return app
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_server.py -v && git commit -am "feat(server): FastAPI scaffold + /v1/models"
```

### Task 8.2: /v1/chat/completions non-streaming

**Files:**
- Modify: `goorouter/server.py`
- Modify: `goorouter/router.py` (add `dispatch` helper)
- Modify: `tests/test_server.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_server.py`:
```python
import json
from unittest.mock import AsyncMock, patch


def test_chat_completions_pinned_backend(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    backend_response = {
        "id": "x", "object": "chat.completion", "created": 0,
        "model": "claude-opus-4-7",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    with patch("goorouter.backends.litellm.acompletion",
               AsyncMock(return_value=backend_response)):
        resp = client.post("/v1/chat/completions", json={
            "model": "goo-cloud-large",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "hello"


def test_chat_completions_explain_mode(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    payload = json.dumps({"domain": "code", "complexity": "medium", "reason": "."})
    with patch("goorouter.classifier.call_backend",
               AsyncMock(return_value={"choices": [{"message": {"content": payload}}]})):
        resp = client.post("/v1/chat/completions", json={
            "model": "goo-explain",
            "messages": [{"role": "user", "content": "rewrite x"}],
        })
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert "Routing decision" in content


def test_chat_completions_unknown_prefix_returns_400(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json={
        "model": "goo-auto",
        "messages": [{"role": "user", "content": "!doesnotexist hi"}],
    })
    assert resp.status_code == 400
    assert "doesnotexist" in resp.json()["error"]["message"]
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_server.py -v
```

- [ ] **Step 3: Add `dispatch` helper to router and wire route**

Append to `goorouter/router.py`:
```python
from goorouter.backends import call_backend


async def dispatch_non_streaming(
    cfg: Config, decision: RoutingDecision, messages: list[dict[str, Any]],
    *, tools: list[dict] | None = None, tool_choice: Any = None,
) -> Any:
    out_messages = apply_stripping(messages, decision.stripped_last_user)
    backend = cfg.backends[decision.backend]
    return await call_backend(
        backend, messages=out_messages, stream=False,
        tools=tools, tool_choice=tool_choice,
    )
```

Append to `goorouter/server.py`:
```python
import time
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from goorouter.explain import format_decision
from goorouter.prefixes import UnknownPrefixError
from goorouter.router import (
    GOO_AUTO, GOO_EXPLAIN, decide_route, dispatch_non_streaming,
)
from goorouter.storage import LogRow, log_request


def _explain_response(decision_text: str, model_field: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_field,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": decision_text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def build_app(cfg: Config, *, db_path: Path) -> FastAPI:
    # ... (keep existing code from Task 8.1) ...
    app = FastAPI(title="goorouter", version="0.1.0")
    app.state.cfg = cfg
    app.state.db = open_db(db_path)

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        ids = ["goo-auto", "goo-explain", *(f"goo-{name}" for name in cfg.backends)]
        return {
            "object": "list",
            "data": [
                {"id": mid, "object": "model", "owned_by": "goorouter", "created": 0}
                for mid in ids
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        body = await request.json()
        messages = body.get("messages", [])
        model_field = body.get("model", GOO_AUTO)
        stream = bool(body.get("stream", False))
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")

        try:
            decision = await decide_route(cfg=cfg, model_field=model_field, messages=messages)
        except UnknownPrefixError as e:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(e), "type": "invalid_request_error"}},
            )

        if decision.mode == "explain":
            text = format_decision(decision, cfg)
            return _explain_response(text, model_field)

        if stream:
            raise HTTPException(status_code=501, detail="streaming not implemented in this task")

        started = time.monotonic()
        try:
            response = await dispatch_non_streaming(
                cfg, decision, messages, tools=tools, tool_choice=tool_choice,
            )
            success, error_kind = True, None
        except Exception as e:
            success, error_kind = False, type(e).__name__
            response = None

        backend_latency_ms = int((time.monotonic() - started) * 1000)

        # Log row (synchronous for v1; async/queue is a follow-up enhancement)
        out = decision.classifier_outcome
        usage = (response or {}).get("usage", {}) if isinstance(response, dict) else {}
        log_request(app.state.db, LogRow(
            request_id=decision.request_id,
            model_field=model_field,
            prefixes_raw=decision.parsed.raw or None,
            pinned_backend=decision.pinned_backend,
            urgency_used=decision.urgency,
            classifier_used=out.classifier_used if out else None,
            classifier_fallback_reason=out.fallback_reason if out else None,
            classifier_input_chars=out.input_chars if out else None,
            classifier_input_truncated_from=out.input_truncated_from if out else None,
            classifier_latency_ms=(out.result.latency_ms if out and out.result else None),
            classifier_domain=(out.result.domain if out and out.result else None),
            classifier_complexity=(out.result.complexity if out and out.result else None),
            classifier_reason=(out.result.reason if out and out.result else None),
            backend_chosen=decision.backend,
            backend_latency_ms=backend_latency_ms,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            success=success,
            error_kind=error_kind,
            prompt_content=decision.last_user_content_original,
            prompt_storage_mode=cfg.logging.prompt_storage,
        ))

        if not success:
            raise HTTPException(status_code=502, detail=f"backend error: {error_kind}")
        return response

    return app
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_server.py -v && git commit -am "feat(server): /v1/chat/completions non-streaming + logging"
```

### Task 8.3: Streaming pass-through (SSE)

**Files:**
- Modify: `goorouter/router.py` (add `dispatch_streaming`)
- Modify: `goorouter/server.py` (add streaming branch)
- Modify: `tests/test_server.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_server.py`:
```python
class _FakeStreamChunks:
    """Async iterator yielding fake litellm chunks."""

    def __init__(self, chunks: list[dict]):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def test_chat_completions_streaming(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    chunks = [
        {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
         "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]},
        {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
         "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}]},
        {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    with patch("goorouter.backends.litellm.acompletion",
               AsyncMock(return_value=_FakeStreamChunks(chunks))):
        with client.stream("POST", "/v1/chat/completions", json={
            "model": "goo-cloud-large",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as resp:
            text = "".join(part for part in resp.iter_text())
    assert "data: " in text
    assert "Hel" in text
    assert "data: [DONE]" in text
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_server.py::test_chat_completions_streaming -v
```

- [ ] **Step 3: Implement**

Append to `goorouter/router.py`:
```python
async def dispatch_streaming(
    cfg: Config, decision: RoutingDecision, messages: list[dict[str, Any]],
    *, tools: list[dict] | None = None, tool_choice: Any = None,
) -> Any:
    out_messages = apply_stripping(messages, decision.stripped_last_user)
    backend = cfg.backends[decision.backend]
    return await call_backend(
        backend, messages=out_messages, stream=True,
        tools=tools, tool_choice=tool_choice,
    )
```

Replace the streaming-501 block in `chat_completions` with:
```python
        if stream:
            from fastapi.responses import StreamingResponse

            async def gen():
                started_local = time.monotonic()
                tokens_in_total = 0
                tokens_out_total = 0
                error_kind_local: str | None = None
                try:
                    upstream = await dispatch_streaming(
                        cfg, decision, messages,
                        tools=tools, tool_choice=tool_choice,
                    )
                    async for chunk in upstream:
                        # litellm chunks are dict-like; serialize as SSE
                        if hasattr(chunk, "model_dump"):
                            chunk_dict = chunk.model_dump()
                        elif isinstance(chunk, dict):
                            chunk_dict = chunk
                        else:
                            chunk_dict = dict(chunk)
                        usage = chunk_dict.get("usage") or {}
                        tokens_in_total = max(tokens_in_total, usage.get("prompt_tokens", 0) or 0)
                        tokens_out_total = max(tokens_out_total, usage.get("completion_tokens", 0) or 0)
                        import json as _json
                        yield f"data: {_json.dumps(chunk_dict)}\n\n"
                    yield "data: [DONE]\n\n"
                    success_local = True
                except Exception as e:
                    success_local = False
                    error_kind_local = type(e).__name__
                    import json as _json
                    yield f"data: {_json.dumps({'error': {'type': error_kind_local, 'message': str(e)}})}\n\n"
                finally:
                    out = decision.classifier_outcome
                    log_request(app.state.db, LogRow(
                        request_id=decision.request_id,
                        model_field=model_field,
                        prefixes_raw=decision.parsed.raw or None,
                        pinned_backend=decision.pinned_backend,
                        urgency_used=decision.urgency,
                        classifier_used=out.classifier_used if out else None,
                        classifier_fallback_reason=out.fallback_reason if out else None,
                        classifier_input_chars=out.input_chars if out else None,
                        classifier_input_truncated_from=out.input_truncated_from if out else None,
                        classifier_latency_ms=(out.result.latency_ms if out and out.result else None),
                        classifier_domain=(out.result.domain if out and out.result else None),
                        classifier_complexity=(out.result.complexity if out and out.result else None),
                        classifier_reason=(out.result.reason if out and out.result else None),
                        backend_chosen=decision.backend,
                        backend_latency_ms=int((time.monotonic() - started_local) * 1000),
                        tokens_in=tokens_in_total or None,
                        tokens_out=tokens_out_total or None,
                        success=success_local,
                        error_kind=error_kind_local,
                        prompt_content=decision.last_user_content_original,
                        prompt_storage_mode=cfg.logging.prompt_storage,
                    ))

            return StreamingResponse(gen(), media_type="text/event-stream")
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_server.py -v && git commit -am "feat(server): SSE streaming pass-through with end-of-stream logging"
```

### Task 8.4: Stdout structured request line

**Files:**
- Modify: `goorouter/server.py` (emit summary log)
- Modify: `tests/test_server.py`

- [ ] **Step 1: Failing test**

Append:
```python
def test_stdout_summary_line(tmp_path, capsys):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    backend_response = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    with patch("goorouter.backends.litellm.acompletion", AsyncMock(return_value=backend_response)):
        client.post("/v1/chat/completions", json={
            "model": "goo-cloud-large",
            "messages": [{"role": "user", "content": "hi"}],
        })
    captured = capsys.readouterr()
    assert "[router]" in captured.out
    assert "cloud-large" in captured.out
```

- [ ] **Step 2: Run, verify failure**

- [ ] **Step 3: Implement**

In `goorouter/server.py`, after the non-streaming `log_request(...)` call and after the streaming `gen()` finally-block's `log_request(...)`, add:
```python
        # Stdout summary
        cls_used = decision.classifier_outcome.classifier_used if decision.classifier_outcome else None
        cls_part = (
            f"classified={decision.classifier_result.domain}/{decision.classifier_result.complexity}"
            if decision.classifier_result else (
                "pinned" if decision.pinned_backend else "skipped"
            )
        )
        cls_lat = (decision.classifier_result.latency_ms
                   if decision.classifier_result else None)
        print(
            f"[router] req#{decision.request_id[:8]} model={model_field} "
            f"urgency={decision.urgency} {cls_part} → {decision.backend} "
            f"(cls {cls_lat}ms gen {backend_latency_ms}ms)",
            flush=True,
        )
```

(Place inside both branches — non-stream uses `backend_latency_ms` from above; stream uses the local equivalent. For brevity in this plan, factor into a helper `_emit_summary(decision, model_field, gen_ms)` if duplication bothers you.)

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_server.py -v && git commit -am "feat(server): structured stdout summary per request"
```

---

## Phase 9 — CLI

### Task 9.1: Typer scaffold + `serve`

**Files:**
- Create: `goorouter/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Failing test**

`tests/test_cli.py`:
```python
import subprocess
import sys


def test_cli_help_lists_commands():
    out = subprocess.run([sys.executable, "-m", "goorouter", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    for cmd in ("serve", "explain", "policy", "config", "log", "relabel"):
        assert cmd in out.stdout


def test_cli_serve_help():
    out = subprocess.run([sys.executable, "-m", "goorouter", "serve", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "host" in out.stdout.lower() or "port" in out.stdout.lower()
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_cli.py -v
```

- [ ] **Step 3: Implement**

`goorouter/cli.py`:
```python
from __future__ import annotations

import os
import socket
import sys
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

    # Pre-flight port check
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
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_cli.py -v && git commit -am "feat(cli): typer scaffold + serve command"
```

### Task 9.2: `explain` CLI

**Files:**
- Modify: `goorouter/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_cli.py`:
```python
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch


SAMPLE_CFG = """
[server]
host = "127.0.0.1"
port = 4000

[backends.cloud-large]
provider = "anthropic"
model = "claude-opus-4-7"
api_key_env = "ANTHROPIC_API_KEY"
aliases = ["opus"]
timeout_s = 120

[backends.local-small]
provider = "openai"
base_url = "http://localhost:1234/v1"
model = "qwen2.5-3b-instruct"
api_key = "lm-studio"
timeout_s = 60

[classifier]
backend = "local-small"
max_input_chars = 8000
timeout_s = 5

[routing]
default_urgency = "normal"
default_on_failure = "cloud-large"

[routing.policy.normal]
"code,trivial"    = "local-small"
"code,medium"     = "local-small"
"code,hard"       = "cloud-large"
"general,trivial" = "local-small"
"general,medium"  = "local-small"
"general,hard"    = "cloud-large"

[routing.policy.urgent]
"code,trivial"    = "local-small"
"code,medium"     = "cloud-large"
"code,hard"       = "cloud-large"
"general,trivial" = "cloud-large"
"general,medium"  = "cloud-large"
"general,hard"    = "cloud-large"

[routing.policy.patient]
"code,trivial"    = "local-small"
"code,medium"     = "local-small"
"code,hard"       = "local-small"
"general,trivial" = "local-small"
"general,medium"  = "local-small"
"general,hard"    = "local-small"

[logging]
db_path = "{db}"
prompt_storage = "full"
"""


def test_cli_explain_prints_decision(tmp_path):
    cfg_path = tmp_path / "config.toml"
    db_path = tmp_path / "log.sqlite"
    cfg_path.write_text(SAMPLE_CFG.format(db=db_path.as_posix()))

    payload = json.dumps({"domain": "code", "complexity": "hard", "reason": "novel refactor"})
    response = {"choices": [{"message": {"content": payload}}]}

    # Run CLI in-process for simplicity (using typer's testing helper would also work)
    from goorouter import cli as cli_mod
    from typer.testing import CliRunner
    runner = CliRunner()
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=response)):
        result = runner.invoke(cli_mod.app, ["explain", "rewrite my code", "--config", str(cfg_path)])
    assert result.exit_code == 0
    assert "Routing decision" in result.stdout
    assert "code" in result.stdout and "hard" in result.stdout
```

- [ ] **Step 2: Run, verify failure**

- [ ] **Step 3: Implement**

Append to `goorouter/cli.py`:
```python
import asyncio
from goorouter.explain import format_decision
from goorouter.router import decide_route


@app.command()
def explain(
    prompt: str = typer.Argument(..., help="Prompt text to classify and route (no destination call made)"),
    config: Optional[Path] = typer.Option(None, help="Path to config.toml"),
):
    """Run the routing pipeline against PROMPT and print the decision."""
    cfg = _load_or_die(config)

    async def _run():
        decision = await decide_route(
            cfg=cfg, model_field="goo-explain",
            messages=[{"role": "user", "content": prompt}],
        )
        print(format_decision(decision, cfg))

    asyncio.run(_run())
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_cli.py -v && git commit -am "feat(cli): explain subcommand"
```

### Task 9.3: `policy show` and `config show`

**Files:**
- Modify: `goorouter/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Failing tests**

Append:
```python
def test_cli_policy_show(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(SAMPLE_CFG.format(db=(tmp_path / "log.sqlite").as_posix()))
    from typer.testing import CliRunner
    from goorouter import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["policy", "show", "--config", str(cfg_path)])
    assert result.exit_code == 0
    for urgency in ("normal", "urgent", "patient"):
        assert urgency in result.stdout
    assert "code,hard" in result.stdout


def test_cli_config_show_masks_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-key-do-not-print")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(SAMPLE_CFG.format(db=(tmp_path / "log.sqlite").as_posix()))
    from typer.testing import CliRunner
    from goorouter import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["config", "show", "--config", str(cfg_path)])
    assert result.exit_code == 0
    assert "secret-key-do-not-print" not in result.stdout
    assert "Cloud backends present" in result.stdout
```

- [ ] **Step 2: Run, verify failure**

- [ ] **Step 3: Implement**

Append to `goorouter/cli.py`:
```python
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
            print(f"  api_key  = ***")
        if b.aliases:
            print(f"  aliases  = {list(b.aliases)}")
        print(f"  timeout_s = {b.timeout_s}")
        print()
        if b.provider in ("anthropic",):
            cloud_providers.add(b.provider)

    print(f"[classifier]")
    print(f"  backend          = {cfg.classifier.backend}")
    print(f"  fallback_backend = {cfg.classifier.fallback_backend}")
    print(f"  max_input_chars  = {cfg.classifier.max_input_chars}")
    print(f"  timeout_s        = {cfg.classifier.timeout_s}")
    print()
    print(f"[logging]")
    print(f"  db_path        = {cfg.logging.db_path}")
    print(f"  prompt_storage = {cfg.logging.prompt_storage}")
    print()
    print(f"Backends configured: {', '.join(cfg.backends.keys())}")
    print(f"Cloud backends present: {'yes (' + ', '.join(sorted(cloud_providers)) + ')' if cloud_providers else 'no (offline-only)'}")
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_cli.py -v && git commit -am "feat(cli): policy show, config show with masking + cloud summary"
```

### Task 9.4: `log show` / `log id`

**Files:**
- Modify: `goorouter/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Failing tests**

Append:
```python
def test_cli_log_show_and_id(tmp_path):
    # Pre-populate the db
    from goorouter.storage import open_db, log_request, LogRow
    from tests.test_storage import _row
    db_path = tmp_path / "log.sqlite"
    conn = open_db(db_path)
    log_request(conn, _row(request_id="row-A", backend_chosen="local-coder"))
    log_request(conn, _row(request_id="row-B", backend_chosen="cloud-large"))
    conn.close()

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(SAMPLE_CFG.format(db=db_path.as_posix()))

    from typer.testing import CliRunner
    from goorouter import cli as cli_mod
    runner = CliRunner()

    r1 = runner.invoke(cli_mod.app, ["log", "show", "--config", str(cfg_path)])
    assert r1.exit_code == 0
    assert "row-A" in r1.stdout and "row-B" in r1.stdout

    r2 = runner.invoke(cli_mod.app, ["log", "show", "--backend", "local-coder",
                                      "--config", str(cfg_path)])
    assert r2.exit_code == 0
    assert "row-A" in r2.stdout and "row-B" not in r2.stdout

    r3 = runner.invoke(cli_mod.app, ["log", "id", "1", "--config", str(cfg_path)])
    assert r3.exit_code == 0
    assert "row-A" in r3.stdout
```

- [ ] **Step 2: Run, verify failure**

- [ ] **Step 3: Implement**

Append:
```python
log_app = typer.Typer(help="Inspect request log.")
app.add_typer(log_app, name="log")


@log_app.command("show")
def log_show(
    limit: int = typer.Option(20, "--limit", "-n"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
    config: Optional[Path] = typer.Option(None),
):
    """Show recent request log rows."""
    cfg = _load_or_die(config)
    from goorouter.storage import open_db, get_recent
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
    cfg = _load_or_die(config)
    from goorouter.storage import open_db, get_by_id
    conn = open_db(Path(cfg.logging.db_path))
    row = get_by_id(conn, row_id)
    if row is None:
        typer.secho(f"No row with id {row_id}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    for k, v in row.items():
        if k == "prompt_content" and v and len(str(v)) > 500:
            v = str(v)[:500] + "...(truncated)"
        print(f"{k:<32} {v}")
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_cli.py -v && git commit -am "feat(cli): log show/id"
```

### Task 9.5: `relabel last` / `relabel <id>`

**Files:**
- Modify: `goorouter/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Failing tests**

Append:
```python
def test_cli_relabel_last_and_by_id(tmp_path):
    from goorouter.storage import open_db, log_request, get_by_id
    from tests.test_storage import _row
    db_path = tmp_path / "log.sqlite"
    conn = open_db(db_path)
    log_request(conn, _row(request_id="r1"))
    log_request(conn, _row(request_id="r2"))
    conn.close()

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(SAMPLE_CFG.format(db=db_path.as_posix()))

    from typer.testing import CliRunner
    from goorouter import cli as cli_mod
    runner = CliRunner()

    r = runner.invoke(cli_mod.app, ["relabel", "last", "cloud-large",
                                     "--note", "wrong",
                                     "--config", str(cfg_path)])
    assert r.exit_code == 0

    conn = open_db(db_path)
    last = get_by_id(conn, 2)
    assert last is not None and last["relabel_backend"] == "cloud-large"

    r = runner.invoke(cli_mod.app, ["relabel", "1", "local-small",
                                     "--config", str(cfg_path)])
    assert r.exit_code == 0
    first = get_by_id(conn, 1)
    assert first is not None and first["relabel_backend"] == "local-small"


def test_cli_relabel_undefined_backend_rejected(tmp_path):
    from goorouter.storage import open_db, log_request
    from tests.test_storage import _row
    db_path = tmp_path / "log.sqlite"
    conn = open_db(db_path)
    log_request(conn, _row())
    conn.close()

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(SAMPLE_CFG.format(db=db_path.as_posix()))

    from typer.testing import CliRunner
    from goorouter import cli as cli_mod
    runner = CliRunner()
    r = runner.invoke(cli_mod.app, ["relabel", "last", "phantom",
                                     "--config", str(cfg_path)])
    assert r.exit_code != 0
    assert "phantom" in (r.stdout + (r.stderr or ""))
```

- [ ] **Step 2: Run, verify failure**

- [ ] **Step 3: Implement**

Append:
```python
relabel_app = typer.Typer(help="Mark requests with corrected backend.")
app.add_typer(relabel_app, name="relabel")


def _check_backend_defined(cfg, backend: str) -> None:
    if backend not in cfg.backends:
        typer.secho(
            f"Backend '{backend}' is not defined in current config. Defined: {sorted(cfg.backends.keys())}",
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
    cfg = _load_or_die(config)
    _check_backend_defined(cfg, backend)
    from goorouter.storage import open_db, relabel_last as _relabel_last
    conn = open_db(Path(cfg.logging.db_path))
    try:
        rid = _relabel_last(conn, backend, note)
        print(f"Relabeled row #{rid} → {backend}")
    except Exception as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@relabel_app.command("by-id")
def by_id(
    row_id: int = typer.Argument(..., metavar="ID"),
    backend: str = typer.Argument(...),
    note: Optional[str] = typer.Option(None, "--note"),
    config: Optional[Path] = typer.Option(None),
):
    """Mark a specific row id as 'should have been <backend>'."""
    cfg = _load_or_die(config)
    _check_backend_defined(cfg, backend)
    from goorouter.storage import open_db, relabel_by_id as _relabel_by_id
    conn = open_db(Path(cfg.logging.db_path))
    try:
        _relabel_by_id(conn, row_id, backend, note)
        print(f"Relabeled row #{row_id} → {backend}")
    except Exception as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
```

Note: typer's `add_typer` with both `last` and a numeric default arg requires a tiny shim. The simplest approach is two named commands: `relabel last <backend>` and `relabel by-id <ID> <backend>`. Update tests to match if needed. (If you keep the spec's `relabel <ID> <backend>` form, use a single command with a custom argument parser that detects whether the first positional is "last" vs an integer.)

For v1, use the `last` and `by-id` subcommand pattern to keep typer happy. Update the test invocation:
```python
r = runner.invoke(cli_mod.app, ["relabel", "by-id", "1", "local-small",
                                 "--config", str(cfg_path)])
```

- [ ] **Step 4: Verify pass + commit**

```bash
pytest tests/test_cli.py -v && git commit -am "feat(cli): relabel last and by-id with backend validation"
```

---

## Phase 10 — Docs, examples, CI

### Task 10.1: config.example.toml

**Files:**
- Create: `config.example.toml`

- [ ] **Step 1: Write the file**

`config.example.toml`:
```toml
# goorouter configuration. Copy to ~/.goorouter/config.toml and edit.

[server]
host = "127.0.0.1"
port = 4000

# ---- Backends ----------------------------------------------------------
# Names are arbitrary identifiers. `aliases` give short names you can use
# as !-prefixes (e.g. "!opus") in the latest user message.
# API keys: prefer `api_key_env` (read from the named environment variable);
# `api_key` is allowed but warned for non-LM-Studio providers.

[backends.cloud-large]
provider     = "anthropic"
model        = "claude-opus-4-7"
api_key_env  = "ANTHROPIC_API_KEY"
aliases      = ["opus", "claude"]
timeout_s    = 120

[backends.cloud-small]
provider     = "anthropic"
model        = "claude-haiku-4-5-20251001"
api_key_env  = "ANTHROPIC_API_KEY"
aliases      = ["haiku"]
timeout_s    = 60

[backends.local-large]
provider     = "openai"
base_url     = "http://localhost:1234/v1"
model        = "qwen2.5-32b-instruct"
api_key      = "lm-studio"
aliases      = ["qwen32"]
timeout_s    = 180

[backends.local-small]
provider     = "openai"
base_url     = "http://localhost:1234/v1"
model        = "qwen2.5-3b-instruct"
api_key      = "lm-studio"
aliases      = ["qwen3"]
timeout_s    = 60

[backends.local-coder]
provider     = "openai"
base_url     = "http://localhost:1234/v1"
model        = "qwen2.5-coder-32b-instruct"
api_key      = "lm-studio"
aliases      = ["qwen-coder", "coder"]
timeout_s    = 180

# ---- Classifier --------------------------------------------------------
[classifier]
backend          = "local-small"      # which backend runs the classifier
fallback_backend = "local-large"      # used for oversize input or primary failure (optional)
max_input_chars  = 8000               # primary's input cap (head-truncate if no fallback)
timeout_s        = 5
# prompt_template_path = "~/.goorouter/classifier.prompt"   # optional override

# ---- Routing -----------------------------------------------------------
[routing]
default_urgency    = "normal"
default_on_failure = "cloud-large"    # last-resort backend if classifier and fallback both fail.
                                      # Set to a local backend if running offline-only.

[routing.policy.normal]
"code,trivial"    = "local-coder"
"code,medium"     = "local-coder"
"code,hard"       = "cloud-large"
"general,trivial" = "local-small"
"general,medium"  = "local-large"
"general,hard"    = "cloud-large"

[routing.policy.urgent]
"code,trivial"    = "local-coder"
"code,medium"     = "cloud-small"
"code,hard"       = "cloud-large"
"general,trivial" = "cloud-small"
"general,medium"  = "cloud-small"
"general,hard"    = "cloud-large"

[routing.policy.patient]
"code,trivial"    = "local-coder"
"code,medium"     = "local-coder"
"code,hard"       = "local-large"
"general,trivial" = "local-small"
"general,medium"  = "local-small"
"general,hard"    = "local-large"

# ---- Logging -----------------------------------------------------------
[logging]
db_path        = "~/.goorouter/log.sqlite"
prompt_storage = "full"   # "full" | "hashed" | "none"
                          # full:   readable later; useful for relabeling and future training
                          # hashed: SHA-256 only; deduplication possible without retaining content
                          # none:   metadata only (latency, tokens, decisions)
```

- [ ] **Step 2: Commit**

```bash
git add config.example.toml
git commit -m "docs: config.example.toml matching the design schema"
```

### Task 10.2: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the README**

`README.md`:
```markdown
# goorouter

Localhost OpenAI-compatible router that picks between cloud and local LLM backends per request, based on a classifier and a per-urgency policy table.

## Quickstart

```
uv tool install goorouter         # or: pipx install goorouter / pip install goorouter
mkdir -p ~/.goorouter
cp config.example.toml ~/.goorouter/config.toml
# edit ~/.goorouter/config.toml — point local backends at your LM Studio,
# set ANTHROPIC_API_KEY in your environment for cloud backends
export ANTHROPIC_API_KEY=...      # if using cloud backends
goorouter serve
```

Configure your client (Zed / any OpenAI-compatible client) to use:
- Base URL: `http://127.0.0.1:4000/v1`
- Model: `goo-auto` (or any of `goo-cloud-large`, `goo-local-coder`, etc.)

## How routing works

1. **Per-message prefix** wins: a leading `!opus` / `!urgent` / `!local-small` overrides everything for that one message.
2. **Model name** is next: `model = "goo-cloud-large"` pins a specific backend; `model = "goo-auto"` runs the classifier; `model = "goo-explain"` shows the routing decision without calling the destination model.
3. **Classifier** sees the latest user message and returns `(domain ∈ code|general, complexity ∈ trivial|medium|hard)`.
4. **Policy table** maps `(urgency, domain, complexity) → backend`. Three urgencies (`normal`, `urgent`, `patient`); set per-message via `!urgent` / `!patient` or globally via `default_urgency`.

## Privacy

Your prompts only go to backends you list in `[backends]`. The system honors what you configure; there is no separate "privacy mode."

- For **offline / local-only** routing, define only `local-*` backends and set `default_on_failure` to one of them. The config validator rejects references to undefined backends, so a local-only config has no path that reaches a cloud provider.
- `goorouter config show` prints `Cloud backends present: yes/no` so you can verify at a glance.
- Logging defaults to storing full prompt content (`prompt_storage = "full"`) for personal use. Switch to `"hashed"` (SHA-256 only) or `"none"` (metadata only) in `[logging]` if you don't want prompts on disk.

## CLI

```
goorouter serve                              # start the proxy
goorouter explain "<prompt>"                 # print routing decision (no backend call)
goorouter policy show                        # dump resolved policy tables
goorouter config show                        # dump validated config (api keys masked)
goorouter log show [--limit N] [--backend X] # tail recent requests
goorouter log id <ID>                        # full detail of one request
goorouter relabel last <backend> [--note]    # mark last request as "should have been X"
goorouter relabel by-id <ID> <backend>       # same, by id
```

## Configuration reference

See [`config.example.toml`](./config.example.toml) for the full schema with comments. Every field is documented inline.

The full design lives at [`openspec/changes/add-initial-router/`](./openspec/changes/add-initial-router/).
```

- [ ] **Step 2: Commit**

```bash
git commit -am "docs: README with quickstart, privacy, CLI reference"
```

### Task 10.3: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow**

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.11", "3.12"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - name: Install
        run: python -m pip install -e ".[dev]"
      - name: Lint
        run: ruff check goorouter tests
      - name: Type check
        run: mypy goorouter
      - name: Test
        run: pytest -v
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: matrix tests on {ubuntu, macos, windows} × {3.11, 3.12}"
```

---

## Final checklist

After Phase 10 completes, verify the success criteria from the spec:

- [ ] `uv tool install .` (or `pipx install .`) installs cleanly on Win/Mac/Linux
- [ ] `goorouter serve` starts with the example config and binds 127.0.0.1:4000
- [ ] `/v1/models` returns `goo-auto`, `goo-explain`, plus one entry per backend
- [ ] `model = "goo-auto"` round-trip via httpx works (with real or mocked backends)
- [ ] `model = "goo-cloud-large"` pins correctly
- [ ] `!opus` / `!urgent` per-message prefixes override
- [ ] `model = "goo-explain"` returns a routing breakdown
- [ ] Tool/function-calling pass-through works
- [ ] SQLite log gains one row per request
- [ ] `explain`, `policy show`, `config show`, `log show`, `log id`, `relabel last`, `relabel by-id` work end-to-end
- [ ] CI passes on all 6 cells
- [ ] README documents quickstart, privacy, and CLI
- [ ] `config.example.toml` is committed and matches the schema

When all boxes are checked, this change is ready to be archived (specs deltas merge into `openspec/specs/`).
