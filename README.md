# saint

Localhost OpenAI-compatible router that picks between cloud and local LLM backends per request, based on a classifier and a per-urgency policy table.

## Quickstart

```
uv tool install saint-router  # or: pipx install saint-router / pip install saint-router
saint config init             # writes ~/.config/saint/config.toml from the bundled template
# Edit ~/.config/saint/config.toml — point local backends at your LM Studio,
# set ANTHROPIC_API_KEY in your environment for cloud backends.
export ANTHROPIC_API_KEY=...      # if using cloud backends
saint serve
```

Configure your client (Zed / any OpenAI-compatible client) to use:

- Base URL: `http://127.0.0.1:4000/v1`
- Model: `saint-auto` (or any of `saint-cloud-large`, `saint-local-coder`, etc.)

## How routing works

1. **Per-message prefix** wins: a leading `!opus` / `!urgent` / `!local-small` overrides everything for that one message.
2. **Model name** is next: `model = "saint-cloud-large"` pins a specific backend; `model = "saint-auto"` runs the classifier; `model = "saint-explain"` shows the routing decision without calling the destination model.
3. **Classifier** sees the latest user message and returns `(domain ∈ code|general, complexity ∈ trivial|medium|hard)`.
4. **Policy table** maps `(urgency, domain, complexity) → backend`. Three urgencies (`normal`, `urgent`, `patient`); set per-message via `!urgent` / `!patient` or globally via `default_urgency`.

## johnny integration (optional)

saint can route to **johnny-managed local seats** instead of static endpoints. Add a
`johnny_role` (or `johnny_seat`) to a backend plus a `[johnny]` block; when johnny is
reachable and the seat resolves `ready`, johnny's live endpoint + model **override** that
backend's static `base_url`/`model`. Otherwise the backend falls back — `while_loading`
(per-backend → `[routing]` global) → its own **static baseline** → `default_on_failure` —
and **never blocks** on a (multi-minute) load. saint stays fully functional with johnny
absent.

- Integration is over johnny's **CLI** (`johnny resolve`/`up`) or its **HTTP daemon** —
  never a library import, so saint keeps running standalone (and stays LiteLLM-isolated).
- `saint-explain` is **liveness-aware**: it shows the resolved seat, its state/eta, and whether
  the override or the static baseline would serve.
- saint **provides** per-request latency / TTFT / tokens to johnny's telemetry ingest
  spool (best-effort, non-fatal). See the `[johnny]` block in `config.example.toml`.

## Privacy

Your prompts only go to backends you list in `[backends]`. The system honors what you configure; there is no separate "privacy mode."

- For **offline / local-only** routing, define only `local-*` backends and set `default_on_failure` to one of them. The config validator rejects references to undefined backends, so a local-only config has no path that reaches a cloud provider.
- `saint config show` prints `Cloud backends present: yes/no` so you can verify at a glance.
- Logging defaults to storing full prompt content (`prompt_storage = "full"`) for personal use. Switch to `"hashed"` (SHA-256 only) or `"none"` (metadata only) in `[logging]` if you don't want prompts on disk.

## CLI

```
saint serve                              # start the proxy
saint explain "<prompt>"                 # print routing decision (no backend call)
saint policy show                        # dump resolved policy tables
saint config init [--path P] [--force]   # write a starter config from the bundled template
saint config show                        # dump validated config (api keys masked)
saint log show [--limit N] [--backend X] # tail recent requests
saint log id <ID>                        # full detail of one request
saint relabel last <backend> [--note]    # mark last request as "should have been X"
saint relabel by-id <ID> <backend>       # same, by id
```

## Configuration reference

See [`config.example.toml`](./saint/config.example.toml) for the full schema with comments. Every field is documented inline.

The full design lives at [`openspec/changes/add-initial-router/`](./openspec/changes/add-initial-router/).
