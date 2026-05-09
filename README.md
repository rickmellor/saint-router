# goorouter

Localhost OpenAI-compatible router that picks between cloud and local LLM backends per request, based on a classifier and a per-urgency policy table.

## Quickstart

```
uv tool install goorouter         # or: pipx install goorouter / pip install goorouter
mkdir -p ~/.goorouter
cp config.example.toml ~/.goorouter/config.toml
# Edit ~/.goorouter/config.toml — point local backends at your LM Studio,
# set ANTHROPIC_API_KEY in your environment for cloud backends.
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
