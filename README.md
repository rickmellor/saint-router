# saint

Localhost OpenAI-compatible router that picks between cloud and local LLM backends per
request — an embedding classifier labels each prompt, a per-urgency policy table maps the
labels to a backend, and every decision is logged with receipts (tokens, cache activity,
cost). Built to sit in front of [johnny](https://github.com/rickmellor/johnny)-managed
local seats with cloud escalation for the hard stuff.

## Quickstart

```
uv tool install saint-router  # or: pipx install saint-router / pip install saint-router
saint config init             # writes ~/.config/saint/config.toml from the bundled template
# Edit ~/.config/saint/config.toml — point local backends at your OpenAI-compatible
# servers (vLLM, llama.cpp, johnny seats) and set ANTHROPIC_API_KEY for cloud backends.
saint serve
```

Setting `ANTHROPIC_API_KEY` auto-injects the Anthropic ladder — `cloud-small` (Haiku),
`cloud-medium` (Sonnet), `cloud-large` (Opus), `cloud-flagship` (Fable) — each reachable
per-message by an alias (`!haiku`, `!sonnet`, `!opus`, `!fable`). An explicit
`[backends.<name>]` block always overrides the auto entry. Without the key there are no cloud
backends at all, so a policy that references one fails config validation.

Configure your client (any OpenAI-compatible client — pi, hermes, opencode, Zed, …):

- Base URL: `http://127.0.0.1:4000/v1`
- Model: `saint-auto` (or pin with `saint-cloud-large`, `saint-local-coder`, etc.)

Run it as a user service so it survives reboots:

```ini
# ~/.config/systemd/user/saint.service
[Unit]
Description=SAINT LLM router
After=network.target
[Service]
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=-%h/.config/saint/env     # e.g. ANTHROPIC_API_KEY=... (chmod 600)
ExecStart=%h/.local/bin/saint serve
Restart=on-failure
[Install]
WantedBy=default.target
```

## How routing works

1. **Per-message prefix** wins: a leading `!opus` / `@patient` / `!local-coder` overrides
   everything for that one message. `!` and `@` are equivalent sigils — use `@` from
   terminal agents that steal `!` for their own shell escape (pi, opencode). An unknown
   `@token` is treated as ordinary message text (mentions are safe); an unknown `!token`
   is an error (a typo'd directive should fail loud).
2. **Model name** is next: `model = "saint-cloud-large"` pins a backend; `saint-auto`
   runs the classifier; `saint-explain` returns the routing decision without calling the
   destination model.

   > **Only the full `saint-<backend>` id pins.** A bare backend name (`cloud-large`,
   > `local-coder`) is not an error and not a pin — it falls through to the classifier, so a
   > client that thinks it pinned Opus can be quietly served by a local seat. There is no
   > warning; check `x-saint-pinned` on the response, which is absent when nothing pinned.
3. **Classifier** sees the latest user message and returns `(domain ∈ code|general,
   complexity ∈ trivial|medium|hard)`. Agent clients that append memory-recall blocks to
   the user message can list those markers in `classifier.ignore_after` — the classifier
   (and the logged prompt) is truncated at the first marker; dispatch always forwards the
   full message.
4. **Policy table** maps `(urgency, domain, complexity) → backend`. Three urgencies
   (`normal`, `urgent`, `patient`) — set per-message via `@urgent` / `@patient` or
   globally via `default_urgency`.
5. **Multimodal** (image-bearing) requests route to `routing.multimodal_backend`
   (else `default_on_failure`) without classification.

## The classifier

Two-tier: a trained **embedding head** (~30ms — embed the prompt, run two logistic
heads) answers when confident; below `min_confidence` it defers to the **LLM
classifier** (`classifier.backend`), whose labels are logged and become training data.
That's a self-improving loop — the prompts the head is worst at are exactly what
accumulates for the next retrain:

```
saint classifier train              # distill the head from logged LLM labels
saint classifier status             # head metadata + live coverage + rows banked
saint classifier status --drift     # replay the head vs recent LLM labels (boundary drift)
saint classifier status --json      # machine-readable, for monitoring jobs (`healthy` bool)
```

### Drift notification

`saint classifier status --drift` replays the current head against recent LLM-labeled rows that
postdate its training, and by default writes or clears `~/.config/saint/retrain-needed.flag`
(`--no-write-flag` to inspect without touching it). The server reads that flag and emits
`x-saint-retrain` on every response, so a client learns the head has drifted from traffic it is
already making — no polling, no monitoring stack. It self-clears when drift returns in bounds.

Run it on a timer. Two things to know:

- **The flag is not proof the check ran.** It is only rewritten when the command *finishes*, so
  a crashed check leaves the previous state — a failure looks exactly like a clean bill of
  health. Check the exit status of the job, not the flag.
- **A small sample reads optimistically.** Right after a retrain there are few comparable rows;
  agreement on a handful of them means little, and the flag deliberately ignores tiny samples.

### Embedding limits

The embedding backend has a hard context window (nomic-embed: 2048 tokens) and no way to ask it
to truncate. A character cap can't stand in for it — prose runs ~4.5 chars/token and JSON ~1.3,
a 3.5× spread — so any cap safe for dense input discards most of a prose prompt. Instead, text
is sent as-is and only what actually overflows is shrunk, per text, so one pathological row
can't fail a batch. Both training and drift report what they truncated.

The trainer only uses rows labeled by the LLM classifier — never the head's own output
(no self-distillation feedback loop) and never cache-reused labels. To bootstrap a fresh
install, `tools/seed_classifier.py` runs a labeled JSONL dataset (`tools/seed_prompts*.jsonl`,
1,200 prompts included) through the real routing pipeline and reports label agreement.
Run seeding with `classifier.mode = "llm"`; flip to `"embedding"` after training. The
serving process hot-reloads the head on file change — no restart after a retrain.

## Agent-loop economics

Agent clients turn one user prompt into N API calls (same last user message, growing tool
history). The `[cache]` section (defaults on) keeps that affordable:

- **Turn cache** — a given (message, urgency) classifies once; loop turns reuse the
  labels (`classifier_used = "cache"`, 0ms). Also kills label flap: a nondeterministic
  LLM classifier can't re-roll `medium→hard` mid-loop and escalate one arbitrary turn.
- **Conversation affinity** — keyed by `hash(first system msg + first non-system msg)`,
  or an explicit `x-session-id` header / `session_id` body field. Short follow-ups
  ("yes, do that") inherit the conversation's labels instead of classifying out of
  context (`classifier_used = "inherited"`). Sliding TTL: active work keeps affinity
  alive, walking away expires it. `sticky_conversations = true` inherits regardless of
  length (off by default).
- **Anthropic prompt caching** — rolling `cache_control` breakpoints on the system
  message and the last text message, so each loop turn re-reads the previous prefix at
  0.1× price and writes only the delta. OpenAI-provider backends cache implicitly;
  local vLLM seats have their own prefix caching. Cache read/write tokens are logged
  per request and priced in `saint log stats`.

Pins, `saint-explain`, and multimodal requests bypass the routing caches; reused labels
never become classifier training data.

## Reliability

- **Dispatch-failure fallback**: per-backend `on_error` names a one-hop fallback (never
  chained — loop-proof). One immediate same-backend retry first
  (`routing.retry_same_backend`). Streaming fails over only *before* the first chunk
  reaches the client. Log rows record the backend that actually served
  (`state_at_dispatch = "dispatch_fallback"`); the journal shows `decided ⤳ served`.
- **Circuit breaker**: `breaker_failures` consecutive failures open a backend's circuit
  for `breaker_cooldown_s` — while open, dispatch skips straight to its fallback.
- Distinct from `while_loading` (a johnny seat is warming up — serve elsewhere
  meanwhile) and `default_on_failure` (the classifier itself failed).

## johnny integration (optional)

saint can route to **johnny-managed local seats** instead of static endpoints. Add a
`johnny_role` (or `johnny_seat`) to a backend plus a `[johnny]` block; when johnny is
reachable and the seat resolves `ready`, johnny's live endpoint + model **override** that
backend's static `base_url`/`model`. Otherwise the backend serves its **static baseline**
(johnny unreachable) or `while_loading` (seat warming up) — and **never blocks** on a
(multi-minute) load. saint stays fully functional with johnny absent.

- Integration is over johnny's **CLI** (`johnny resolve`/`up`) or its **HTTP daemon** —
  never a library import, so saint keeps running standalone (and stays LiteLLM-isolated).
- `saint-explain` is **liveness-aware**: it shows the resolved seat, its state/eta, and whether
  the override or the static baseline would serve.
- saint **provides** per-request latency / TTFT / tokens to johnny's telemetry ingest
  spool (best-effort, non-fatal). See the `[johnny]` block in `config.example.toml`.

## Discovery — `GET /status`

`/status` returns the full set of routable choices: every backend with its `kind`
(router/local/cloud), live johnny state, model, endpoint and context window for locals, model
/ context / price for cloud tiers, and a cost `rank`. A client can build its whole seat list
from this rather than hardcoding one — dormant seats simply aren't there, and a seat that comes
up mid-session appears without a config change.

## Embeddings

`POST /v1/embeddings` routes to `routing.embeddings_backend` (johnny-resolved like any
bound backend), so embedding traffic shares the same endpoint, failover, and accounting
as chat. Inputs are never stored in the log.

## AWS Bedrock backends + Claude Code

A backend with `provider = "bedrock"` dispatches to a Bedrock inference profile using the
AWS credential chain (no static keys — `aws_profile` carries the corporate
`credential_process`/SSO). SAINT auto-applies a credential patch so short-lived SSO tokens
refresh transparently, classifies AWS auth failures distinctly (force-open the breaker,
spawn the SSO browser login once, recover via a non-interactive refresh probe on the
breaker's half-open trial), and derives Bedrock Claude cache prices like Anthropic's.

`POST /v1/messages` speaks the **Anthropic Messages API**, so Claude Code can route through
SAINT (point `ANTHROPIC_BASE_URL` at it). It uses the same classifier/policy/cache brain
as chat, forwards the client's own `cache_control` untouched, and strips signed thinking
blocks when a turn switches backends (their signatures are model-bound and would otherwise
400 on Bedrock).

See [`docs/corp-deployment.md`](./docs/corp-deployment.md) and
[`docs/examples/corp-bedrock.toml`](./docs/examples/corp-bedrock.toml) for the full corporate
setup. Install with the extra: `uv tool install 'saint-router[bedrock]'`.

## Observability & accounting

Every response carries routing metadata headers — `x-saint-backend`, `x-saint-domain`,
`x-saint-complexity`, `x-saint-classifier`, `x-saint-urgency`, `x-saint-state`,
`x-saint-request-id`, `x-saint-pinned` when a pin was honoured, `x-saint-decided` when a
dispatch fallback changed the server, and `x-saint-retrain` while the classifier drift flag is
set. `curl -si` tells you the whole story without opening the log. Header values are folded to
ASCII, so prose written into the drift flag can't produce an unencodable header.

Every request lands in a SQLite log (`saint log show` / `saint log id <N>`), and the
accounting rolls up with **net savings against a counterfactual**:

```
$ saint log stats --days 7
...
actual est cost   $1.43
all-cloud-large counterfactual (uncached): $9.80
NET SAVINGS $8.38  =  local routing $6.24 + cheaper tiers $0.02 + prompt caching $2.11
```

Local seats are priced too. The `[energy]` block (`price_kwh`, `host_watts`, `gpu_watts` —
all tunable, nothing hardcoded) turns a seat's measured decode rate into $/Mtok of electricity,
so a local seat and a cloud tier can be compared in one unit. The rate is measured from SAINT's
own request log — the p75 of per-response decode rate over substantial responses, which avoids
short replies where fixed TTFT drags the average down.

Give priced backends `price_in` / `price_out` (USD per Mtok; anthropic cache prices
derive as 0.1× / 1.25× of `price_in`). The counterfactual sends all chat traffic to
`--baseline` (default `default_on_failure`) uncached; the three savings components
provably sum to the net. `--json` for monitoring jobs. Cache advantage can be negative
— write-heavy days are reported honestly.

Log lifecycle: `saint log clear` (reset), `saint log prune --days N` (age out old rows;
rows the classifier trainer can still use are preserved by default).

## Privacy

Your prompts only go to backends you list in `[backends]`. The system honors what you
configure; there is no separate "privacy mode."

- For **offline / local-only** routing, define only `local-*` backends and set
  `default_on_failure` to one of them. The config validator rejects references to
  undefined backends, so a local-only config has no path that reaches a cloud provider.
- Mind the policy's cloud cells: `code,hard` escalation ships the conversation —
  including agent tool results — to the cloud backend. `@patient` keeps a single
  request local; the `patient` policy tier keeps everything local.
- `saint config show` prints `Cloud backends present: yes/no` so you can verify at a glance.
- Logging defaults to storing full prompt content (`prompt_storage = "full"`) — that's
  what powers classifier training and relabeling. Switch to `"hashed"` or `"none"` in
  `[logging]` if you don't want prompts on disk. Embedding inputs are never stored.

## CLI

```
saint serve                               # start the proxy
                                          #   endpoints: /v1/chat/completions, /v1/messages
                                          #   (Anthropic/Claude Code), /v1/embeddings
saint explain "<prompt>" [--test]         # print routing decision; logs the classification
                                          #   as training data unless --test
saint policy show                         # dump resolved policy tables
saint config init [--path P] [--force]    # write a starter config from the bundled template
saint config show                         # dump validated config (api keys masked)
saint classifier train [--limit N]        # distill the embedding head from logged labels
saint classifier status [--drift] [--json]# head metadata, coverage, drift, retrain nudges
                          [--limit N]       #   rows compared for --drift (default 200)
                          [--no-write-flag] #   inspect without touching the retrain flag
saint log show [--limit N] [--backend X]  # tail recent requests
saint log id <ID>                         # full detail of one request
saint log stats [--days N] [--baseline B] [--json]   # usage, cost, net savings
saint log prune [--days N] [--no-keep-training]      # age out old rows
saint log clear [--yes]                   # delete ALL log rows
saint relabel last <backend> [--note]     # mark last request as "should have been X"
saint relabel by-id <ID> <backend>        # same, by id
```

## Configuration reference

See [`config.example.toml`](./saint/config.example.toml) for the full schema with
comments. Every field is documented inline.

The v1 design history lives under [`openspec/changes/`](./openspec/changes/).
