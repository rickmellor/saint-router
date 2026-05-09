# Add Initial Router (goorouter v1)

## Intent

Build `goorouter`: a localhost OpenAI-compatible HTTP proxy that routes chat-completion requests between cloud and local LLM backends based on a classifier and a per-urgency policy table.

The motivating problem: the user wants to chat with one tool (Zed editor, Claude Code, etc.) that decides *per request* whether to send work to Anthropic Opus, Anthropic Haiku, or one of several local models loaded in LM Studio — without manually toggling the model in the client. Today every client requires a fixed model pin; routing decisions are forced into the human.

A secondary goal is to **collect labeled data** as a byproduct of normal use, so that a trained routing classifier becomes a future option without requiring a separate data-gathering project.

## Scope

### In scope (v1)

- Cross-platform Python package (Windows, macOS, Linux) installable via `uv tool install` / `pipx`.
- FastAPI app exposing OpenAI-compatible `/v1/chat/completions` and `/v1/models`.
- Single TOML config at `~/.goorouter/config.toml`, validated at startup with all-errors-at-once output.
- Arbitrary number of named backends. The example config defines five: `cloud-large`, `cloud-small`, `local-large`, `local-small`, `local-coder`.
- Routing pipeline:
  1. Parse leading `!`-prefixes from the latest user message (urgency overrides + per-message backend pin).
  2. If `model` field is `goo-<backend>` (or a prefix pinned the backend), bypass the classifier.
  3. If `goo-auto`, invoke the classifier (any user-designated OpenAI-compatible backend), which returns `{domain, complexity}`.
  4. Look up `(urgency, domain, complexity) → backend` in a per-urgency policy table.
  5. Forward to the chosen backend via the LiteLLM SDK; stream the response back as SSE.
- Optional `fallback_backend` for the classifier — used when input exceeds `max_input_chars` or the primary classifier errors.
- `default_on_failure` backend used when classifier and fallback both fail. Defaults to a cloud backend; configurable for offline-only setups.
- `goo-explain` virtual model and `goorouter explain` CLI: report the routing decision for a prompt without calling the destination backend.
- SQLite request log at `~/.goorouter/log.sqlite`. One row per request with prefixes, classifier output, backend chosen, latency, tokens, and prompt content (subject to a privacy-mode setting).
- CLI surface: `serve`, `explain`, `policy show`, `config show`, `log show`, `log id`, `relabel last`, `relabel <id>`.
- Tool/function-calling pass-through to the destination backend.
- Cross-platform CI: `{windows-latest, macos-latest, ubuntu-latest} × {python-3.11, python-3.12}`.

### Out of scope (deferred)

Each item below is intentionally deferred to keep v1 small and shippable. Detailed reasoning per item — including what would unlock building it — is in `design.md` under "Deferred / Future Work."

- Background-service / autostart on any OS.
- `--watch-config` for hot-reload of the config file in the running server.
- Multimodal (image) content in messages; v1 routes such requests to `default_on_failure` and warns.
- Per-backend `context_window` validation prior to dispatch.
- **Cost tracking** — promoted to its own change at [`openspec/changes/add-cost-tracking/`](../add-cost-tracking/).
- Classifier chains beyond a single fallback.
- Per-failure-type fallback rules (different fallback for oversize vs. error).
- Web UI / dashboard.
- Multi-user or non-localhost binding (any non-127.0.0.1 binding requires explicit config + warning).
- Training a custom classifier model. v1 produces the labeled data the eventual training step needs; the training itself is a separate future project.

## Approach

A custom FastAPI server (`server.py`) accepts OpenAI-compatible chat completion requests. A pure-Python orchestrator (`router.py`) sequences: prefix parsing → optional classifier call → policy lookup → forward via the LiteLLM SDK (`litellm.acompletion`). Streaming responses are forwarded as Server-Sent Events without buffering. Every request emits one async-written SQLite log row.

The **LiteLLM SDK is used, not the LiteLLM proxy server**. This gives provider abstraction (Anthropic ↔ OpenAI shape translation, retries, error normalization) without inheriting the proxy's plugin lifecycle, virtual-keys machinery, or runtime cost — features that are dead weight for a single-user dev tool.

Privacy is configuration: users define backends in TOML; the router never reaches outside that list. Local-only setups are fully supported by defining only local backends and pointing `default_on_failure` to one of them.

## Capabilities

This change introduces three capability domains, each with its own delta spec under [`specs/`](specs/):

- **proxy** — HTTP server, OpenAI compatibility, streaming pass-through, model listing.
- **routing** — prefix parsing, classifier invocation, policy lookup, backend dispatch.
- **observability** — SQLite log, relabel CLI, explain mode (API + CLI).
