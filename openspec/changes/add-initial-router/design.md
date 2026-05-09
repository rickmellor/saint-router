# goorouter v1 — Design

This document captures the technical design behind the [proposal](proposal.md). Sections 1–5 cover the system as a coherent whole; Section 6 enumerates deferred items with the reasoning behind each.

## 1. Architecture & runtime

### Shape

A single Python package, `goorouter`, installable via `uv tool install` / `pipx install` / `pip install`. It exposes one console script (`goorouter`) with subcommands. The main subcommand is `goorouter serve`, which starts a FastAPI app on `127.0.0.1:4000` (configurable). The server speaks the OpenAI Chat Completions API at `/v1/chat/completions` and `/v1/models`. Zed (or any OpenAI-compatible client) is configured to point at `http://127.0.0.1:4000/v1` with any model name; routing is determined per-request by prefix parsing, model field, classifier, and policy.

### Runtime

Foreground process, started from a terminal when needed. Logs to stdout. Exits cleanly on Ctrl-C. No background daemon, no autostart, no system tray. Single-user, localhost-only by default — never binds to `0.0.0.0` unless explicitly set in config (with a loud warning).

### `/v1/models` surface

Returns one virtual model plus one entry per configured backend, all `goo-`-prefixed:

- `goo-auto` — the routed default (classifier + policy decides per request)
- `goo-explain` — runs the routing pipeline up to but not including the backend call; returns the decision as a chat completion (no destination tokens consumed)
- `goo-<backend-name>` — one per `[backends.<name>]` in config; pins routing to that backend, bypassing classifier

The `goo-` prefix is a namespace at the API boundary only; internal config still uses bare names (`cloud-large`, `local-coder`).

### Cross-platform contract

| Concern | v1 decision |
|---|---|
| Install | `uv tool install goorouter` / `pipx install goorouter` / `pip install --user goorouter`. No native installers. |
| Config path | `~/.goorouter/config.toml` on all three OSes. Single canonical path, ignoring platform conventions for simplicity. |
| Data path | `~/.goorouter/log.sqlite`. Same. |
| Console script | `goorouter` on PATH via `pyproject.toml` `[project.scripts]`. Works in cmd, PowerShell, bash, zsh, fish. |
| Python | `>=3.11`. |
| Native deps | None. All deps are pure-Python or have wheels for all three OSes. |
| Path expansion | `~` and `${VAR}` expand on every `*_path` field, cross-platform. |

### What is intentionally not built

Background services, system-tray integration, native installers, multi-user mode, network exposure, GPU management for local models (LM Studio handles its own), telemetry endpoints.

---

## 2. Configuration model

Single TOML file at `~/.goorouter/config.toml`. Validated at startup. All errors are reported together (not just the first), then exit code 2. Created from `config.example.toml` on first run if missing.

```toml
[server]
host = "127.0.0.1"
port = 4000

# ---- Backends ----------------------------------------------------------
# Names are arbitrary identifiers. Aliases let you type short prefixes.
# API keys: prefer `api_key_env` (read from environment); literal `api_key`
# is allowed but warned for non-LM-Studio providers.

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
api_key      = "lm-studio"           # LM Studio ignores this; placeholder required
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
backend          = "local-small"      # references a [backends.*] entry
fallback_backend = "local-large"      # optional: used for oversize input or primary failure
max_input_chars  = 8000               # primary's input cap (head-truncation if no fallback)
timeout_s        = 5                  # tight: classifier should be fast
# prompt_template_path = "~/.goorouter/classifier.prompt"   # optional override

# ---- Routing -----------------------------------------------------------
[routing]
default_urgency    = "normal"
default_on_failure = "cloud-large"    # last-resort backend when classifier + fallback both fail.
                                      # Set to a local backend if running offline / local-only.

# 3 urgency tables × 6 (domain, complexity) cells. All 18 cells must be defined.
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
                          # full:   readable later, useful for relabeling & training
                          # hashed: SHA-256 only, dedup possible, no PII
                          # none:   metadata only (latency, tokens, decisions)
```

### Notable choices

- **API keys are never stored in config by default.** The schema makes `api_key_env` the obvious path; literal `api_key` is supported for non-secret placeholders (LM Studio) and warned for cloud providers.
- **Custom classifier prompt** can be loaded from a file (`prompt_template_path`) — letting you iterate on prompt wording without code changes.
- **Per-backend timeouts**: local 32B models legitimately take 2–3 min on first prompt vs. cloud's <60s expectations.
- **Logging defaults to `full`** because v1 is a personal tool — the most useful default. Privacy modes are explicit choices.
- **Validation enforces referential integrity**: every backend named in policy tables, classifier, or `default_on_failure` must exist under `[backends.*]`.

---

## 3. Request lifecycle & components

### Lifecycle

```
┌──────────────────────────────────────────────────────────────┐
│ Zed (or any OpenAI client)                                   │
│   POST /v1/chat/completions  { model, messages, tools, ... } │
└─────────────────────┬────────────────────────────────────────┘
                      ▼
1. Receive            FastAPI handler parses the body.

2. Model routing      Inspect `model` field:
                      - "goo-explain"      → step 7 (explain mode)
                      - "goo-<backend>"    → pin backend; skip steps 4–5
                      - "goo-auto" / other → continue to step 3

3. Prefix parsing     Look at messages[-1].content (last user msg).
                      Strip leading "!<token>" tokens left-to-right:
                      - !urgent / !patient / !normal → set urgency
                      - !<name>  → resolve via aliases → pin backend
                      Stripped content replaces original message content
                      before forwarding (model never sees "!opus").

4. Classify           If backend NOT pinned by step 2 or 3:
                      build classifier prompt with stripped last-user-msg,
                      call classifier (oversize → fallback if set;
                      else head-truncate to max_input_chars),
                      parse JSON {domain, complexity, reason}.
                      Failure path:
                        primary fail → fallback (if set) → default_on_failure.
                      Mark classifier_used and classifier_fallback_reason in log.

5. Policy lookup      backend = config.policy[urgency][f"{domain},{complexity}"]

6. Forward            Build litellm.acompletion(...) call against chosen backend.
                      stream=true: forward chunks as SSE.
                      stream=false: return full response.

7. (Explain branch)   Skip backend call; format routing decision as a
                      chat completion response. No destination tokens.

8. Log                Write one SQLite row: ts, urgency, prefixes_parsed,
                      classifier output (or null), backend chosen, latency
                      (classifier + backend), tokens, success/error, prompt
                      content per logging.prompt_storage. Async; never blocks.
```

### Edge cases handled at v1

| Case | Behavior |
|---|---|
| Empty `messages` or no user message | Treat as `(general, trivial)`, default urgency. |
| `tools` parameter present | Pass through to backend; classifier sees text only. |
| Multi-turn conversation | Classifier sees only the latest user message. |
| Multimodal content (images) | Out of scope: route to `default_on_failure`, stdout warning, log it. |
| Long prompts (classifier-side) | If `fallback_backend` set: use it. Else head-truncate to `max_input_chars`. |
| Long prompts (destination-side) | Pass through. Backend's native error propagates. |
| Unknown prefix token | 400 to client: `"unknown prefix token 'foo'; defined: [...]"`. |

### Components

```
goorouter/
  __init__.py
  config.py        # TOML load, env expansion, validation → typed Config dataclass
  prefixes.py      # parse leading !tokens; (urgency_override, pinned_backend, stripped_content)
  classifier.py    # build prompt → call classifier backend → ClassifierResult | ClassifierError
  policy.py        # pure: (urgency, domain, complexity, config) → backend_name
  backends.py      # registry; dispatch via litellm.acompletion (sync + streaming)
  router.py        # orchestrator: prefixes → classify → policy → backends; emits log records
  storage.py       # SQLite schema + log_request / get_recent / relabel / query helpers
  explain.py       # format a RoutingDecision as text (used by goo-explain + CLI explain)
  server.py        # FastAPI app: /v1/chat/completions, /v1/models, SSE streaming
  cli.py           # typer: serve, explain, policy show, config show, log, relabel, log show
  __main__.py      # `python -m goorouter` → cli entry
tests/
  test_prefixes.py     # pure unit tests
  test_policy.py       # pure unit tests
  test_config.py       # validation, env expansion, error messages
  test_classifier.py   # uses VCR-style recorded responses for determinism
  test_router.py       # integration: prefix + (mock) classifier + policy + (mock) backend
  test_storage.py      # roundtrip: log → query → relabel
  test_server.py       # full HTTP via httpx.AsyncClient
config.example.toml
pyproject.toml
README.md
```

Each module has one clear job. Pure-function modules (`prefixes`, `policy`) have trivial unit tests. The orchestrator (`router`) is the only file that knows the full sequence. I/O modules (`backends`, `storage`, `server`) are the only ones that touch the network or filesystem.

### Process model

`goorouter serve` is the only long-lived process; it binds the port. All other CLI commands are short-lived and independent of the server. They each read the same config file and SQLite database from disk and work whether or not `serve` is running.

Two pieces of state are shared between processes:

1. **The SQLite database**, with WAL mode (`PRAGMA journal_mode=WAL`) set on first connection. Concurrent reads never block writes; CLI commands and the server can operate against the database simultaneously without lockfiles or PIDs.
2. **The config file**. All processes read it on startup. The running server **does not auto-reload** the config — edits require a `serve` restart to take effect. (The `--watch-config` flag is deferred to v1.5.)

Running `goorouter serve` twice fails to bind the port; v1 prints a friendly message ("Port 4000 in use; is another `goorouter serve` running?") rather than the raw stack trace.

---

## 4. Storage, relabeling, observability

### SQLite schema

Single file at `~/.goorouter/log.sqlite`. One wide table for v1; can be normalized later without migration pain.

```sql
CREATE TABLE requests (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                              TEXT NOT NULL,             -- ISO8601 UTC
    request_id                      TEXT NOT NULL,             -- uuid4 for cross-referencing logs
    model_field                     TEXT NOT NULL,             -- whatever the client sent

    -- Prefix parsing
    prefixes_raw                    TEXT,                      -- e.g. "!urgent !opus" or null
    pinned_backend                  TEXT,                      -- backend pinned by prefix or model field
    urgency_used                    TEXT NOT NULL,             -- normal|urgent|patient

    -- Classifier
    classifier_used                 TEXT,                      -- backend actually called, or null
    classifier_fallback_reason      TEXT,                      -- oversize|primary_error|null
    classifier_input_chars          INTEGER,
    classifier_input_truncated_from INTEGER,                   -- original size if truncated
    classifier_latency_ms           INTEGER,
    classifier_domain               TEXT,                      -- code|general|null
    classifier_complexity           TEXT,                      -- trivial|medium|hard|null
    classifier_reason               TEXT,                      -- one-sentence reason

    -- Final dispatch
    backend_chosen                  TEXT NOT NULL,
    backend_latency_ms              INTEGER,
    tokens_in                       INTEGER,
    tokens_out                      INTEGER,
    success                         INTEGER NOT NULL,          -- 0/1
    error_kind                      TEXT,                      -- timeout|http_4xx|http_5xx|null

    -- Prompt content (subject to logging.prompt_storage)
    prompt_content                  TEXT,                      -- full | sha256 | null
    prompt_storage_mode             TEXT NOT NULL,             -- "full"|"hashed"|"none"

    -- Relabel
    relabel_backend                 TEXT,
    relabel_ts                      TEXT,
    relabel_note                    TEXT
);

CREATE INDEX idx_requests_ts      ON requests(ts);
CREATE INDEX idx_requests_backend ON requests(backend_chosen);
CREATE INDEX idx_requests_relabel ON requests(relabel_backend) WHERE relabel_backend IS NOT NULL;
```

### Logging path

Logging is **fire-and-forget async**. The router builds the log record after the response starts streaming, then queues it for write on a background task. A failed log write logs to stderr but never affects the response.

Schema changes are handled by a tiny migration system. `storage.py` checks `PRAGMA user_version` on startup and runs `0001_init.sql`, `0002_xxx.sql`, etc. as needed. v1 ships `0001_init.sql` only.

### CLI for observability and relabeling

```
goorouter serve                                # start the proxy
goorouter explain "<prompt text>"              # run routing pipeline, print the decision (no backend call)
goorouter policy show                          # dump the resolved policy tables
goorouter config show                          # dump validated config (api keys masked)
goorouter log show [--limit N] [--backend X]   # tail recent requests
goorouter log id <ID>                          # full detail of one request
goorouter relabel last <backend> [--note "."]  # mark most recent request as "should have been <backend>"
goorouter relabel <ID>  <backend> [--note "."] # same, by id
```

`relabel last` is the workhorse: when you notice a wrong routing decision in Zed, one terminal command captures the correction. `last` resolves to `MAX(id)` in the table.

A future training set is: `SELECT prompt_content, COALESCE(relabel_backend, backend_chosen) FROM requests WHERE prompt_storage_mode = 'full' AND success = 1`. v1 doesn't build the training pipeline; it ensures the data shape supports one later.

### Observability beyond the log

- **Server stdout**: one structured line per request, e.g.: `[router] req#1234 model=goo-auto urgency=normal classified=code/medium → cloud-small (cls 287ms, gen 4112ms)`.
- **Truncation / fallback events**: stdout one-liners, e.g. `[router] req#1234: classifier fallback (oversize: 18203 chars > 8000)`.
- **No metrics endpoint, no Prometheus, no OpenTelemetry** for v1.
- **`goo-explain` (API + CLI)** — see Section 5 in the lifecycle, and the `observability/spec.md` requirements for output shape.

### Privacy framing

Privacy is a **configuration decision, not a per-request feature**. If `cloud-large` is in your config, prompts can reach Anthropic by design. The system honors what you configure; it does not invent a "privacy mode."

- `config show` includes a one-line summary: `Cloud backends present: yes (anthropic)` so a user can verify at a glance.
- README ships a `## Privacy` section that explicitly documents: define only backends you trust; for offline-only setups, define only local backends and set `default_on_failure` to one of them; the validator rejects references to undefined backends, so an offline-only config has no cloud fallback path.

---

## 5. Error handling, testing, success criteria

### Error handling (consolidated)

| Failure | Behavior |
|---|---|
| Config invalid at startup | Fail fast, print all errors at once, exit 2. |
| Port already in use | Friendly message, exit 3. |
| Classifier oversize input | Skip primary, use `fallback_backend` if set; else head-truncate & try primary. |
| Classifier timeout / 5xx / malformed JSON | Try fallback once; else `default_on_failure`. |
| Both classifiers fail | Use `default_on_failure`. Log row's `classifier_used = null`. |
| Unknown prefix token | 400 to client with helpful list of defined backends. |
| Backend (chosen destination) errors | Pass through unchanged. **No automatic retry, no automatic re-routing.** |
| Streaming error mid-response | Close SSE cleanly with a final error event. Log row marks failure. |
| Empty `messages` or no user message | Treat as `(general, trivial)`, default urgency. |
| Multimodal content | Out of scope: route to `default_on_failure`, stdout warning. |
| Log write fails | Stderr only. Never affects the response. |

**Principle:** errors in the routing/classifier layer fall back; errors in the destination backend pass through. The router doesn't try to be cleverer than the user — if Opus errors, the user sees the Opus error.

### Testing strategy

| Layer | Test type | Tools |
|---|---|---|
| `prefixes.py`, `policy.py` | Unit (pure functions) | pytest |
| `config.py` | Validation, env expansion, error messages | pytest |
| `classifier.py` | Unit + recorded responses | pytest, VCR-style fixtures |
| `router.py` | Integration with mock classifier + mock backend | pytest-asyncio |
| `storage.py` | Roundtrip log → query → relabel; WAL concurrency smoke | pytest, in-memory SQLite |
| `server.py` | Full HTTP via `httpx.AsyncClient`, including streaming | pytest-asyncio |
| `cli.py` | `subprocess.run` invocations of each command | pytest |
| End-to-end smoke | Manual: real LM Studio + real Anthropic key | docs only |

**CI:** GitHub Actions `{windows-latest, macos-latest, ubuntu-latest} × {python-3.11, python-3.12}`. Lint with `ruff`, type-check with `mypy --strict` on `goorouter/`. Tests must pass on all six cells.

**Coverage targets:** no hard percentage gate. Decision logic (`prefixes`, `policy`, `classifier`, `router`) gets thorough cases; transport layers (`backends`, `server`) get smoke + key paths.

### Success criteria for v1

v1 is "done" when all of these hold:

**Functional**

1. `uv tool install goorouter` (or `pipx install goorouter`) installs cleanly on Windows, macOS, Linux.
2. `goorouter serve` starts with a valid config, binds the port, exits cleanly on Ctrl-C.
3. Zed (or any OpenAI-compatible client) configured at `http://127.0.0.1:4000/v1` can:
   - List models via `/v1/models` and see `goo-auto`, `goo-explain`, plus one `goo-<backend>` per configured backend.
   - Send a chat completion with `model = "goo-auto"` and get a streaming response routed via classifier+policy.
   - Pin via `model = "goo-cloud-large"` and bypass classifier.
   - Override per-message via `!opus` / `!urgent` / etc.
   - Use `model = "goo-explain"` to receive a routing breakdown without any destination call.
4. Tool/function-calling requests pass through to the destination backend and back to the client unchanged.
5. SQLite log gains one row per request including all classifier metadata.
6. CLI commands work: `explain`, `policy show`, `config show`, `log show`, `log id`, `relabel last`, `relabel <id>`.

**Quality**

7. Test suite passes on the 6-cell CI matrix.
8. README has: Quickstart, Privacy section, Configuration reference, CLI reference.
9. `config.example.toml` is committed and matches the schema in this document.

---

## 6. Deferred / Future work

These are intentionally not in v1. Each entry includes the reason and what would unlock building it.

| Item | Why deferred | What would unlock |
|---|---|---|
| **Background-service / autostart** | Single-user dev tool; foreground process is fine and easier to iterate on. Per-OS autostart logic = 3× the surface area. | Once the foreground process is genuinely stable and the user finds themselves restarting it daily, layer service support on top. |
| **`--watch-config` hot reload** | Modest convenience. Easy to forget to restart; easy to add later as additive feature. | Real annoyance with restart cycle during policy tuning. |
| **Multimodal content** | v1 is text-only. Local models often don't accept images; classifier prompt would need significant rework. Routing images to `default_on_failure` with a warning is honest. | When the user actively wants to route image-containing requests; will require classifier prompt redesign and per-backend "supports_images" config. |
| **Per-backend `context_window` validation** | Requires maintaining accurate token-window data per provider/model and a real tokenizer. v1 lets the destination backend's native error propagate. | Pain from blind submissions that fail at 95% completion. |
| **Cost tracking** | First-class candidate for the next change. Promoted to its own change folder at `openspec/changes/add-cost-tracking/` so notes can accumulate. | Real usage data showing where cloud spend lands. |
| **Classifier chains beyond one fallback** | YAGNI: a single fallback covers oversize and one error retry. Chains add config complexity without clear use case. | Field evidence that two-step fallback is insufficient (e.g., specific patterns of cascading failures in the log). |
| **Per-failure-type fallback rules** | Different fallback for oversize vs. error sounds nice but is rarely needed in practice. | Log analysis showing meaningfully different "should fall back to X" patterns for different failure types. |
| **Web UI / dashboard** | Stdout + SQLite + CLI is plenty for a single user. A dashboard adds a frontend stack. | Multi-user need or strong personal preference; outside v1 scope regardless. |
| **Multi-user / non-localhost binding** | Security implications. v1 is single-user, localhost-only. | A clear use case (team router?) which would need auth, rate limiting, audit, and a separate threat model — likely a v2 product. |
| **Trained classifier** | The whole point of v1's logging is to make this *possible* later. Training is its own project (data labeling, model selection, eval, deployment). | A few weeks of v1 use producing relabeled data; a separate change proposal under `openspec/changes/` then takes it from there. |
