# Add cost tracking

> **Status:** superseded (2026-07-08) — implemented as `saint log stats` with a different
> shape than proposed. Prices live per-backend (`price_in`/`price_out`/`price_cache_*`,
> anthropic cache prices derived 0.1x/1.25x) rather than a `[pricing]` section with bundled
> defaults; cost is computed at read time from logged token counts (no `cost_usd_*` columns
> or backfill); cache read/write tokens are logged per request (migration 0003) and priced;
> and the headline metric goes beyond the proposal: NET savings against an all-cloud
> counterfactual, decomposed into local routing + cheaper tiers + prompt caching. Budget
> warnings remain unimplemented. Kept for the ideas not yet built.

## Intent

Per-request cost tracking and reporting. Make the cost of cloud routing visible so the user can tune the policy table and urgency defaults against real spend, not vibes.

This is the most likely "v1.5" change because: (1) cost is the main pressure that motivated routing in the first place, (2) the data needed (per-request `tokens_in`/`tokens_out`) is already captured in v1, and (3) the implementation is mostly additive — new columns, a price table, and a single CLI subcommand.

## Scope

### In scope

- Per-provider price table (input / output USD per million tokens), in config under `[pricing]`, with bundled defaults for known Anthropic and OpenAI-compatible models.
- Cost computed per request from token counts already logged in v1.
- New SQLite columns on the `requests` table: `cost_usd_input`, `cost_usd_output`, `cost_usd_total`. Backfill logic for existing rows.
- CLI: `goorouter cost summary [--days N] [--by backend|urgency|domain|date]`.
- Optional daily/weekly budget with stdout warning when exceeded — non-enforcing.
- Brief per-request cost shown in `goo-explain` ("estimated cost if routed to <backend>: $0.0123").

### Out of scope (for this change)

- Hard budget enforcement (refusing to route when over budget). Separate concern with its own UX questions.
- Cost tracking for local backends. Treated as zero in v1.5 (true cost is power; not user-facing). `cost_usd_*` columns are `NULL` for local backends.
- Real-time cost dashboard or web UI.

## Approach

- Add `[pricing]` section to config with optional overrides; bundle defaults for current Anthropic / OpenAI / OpenRouter models, refreshed at release time.
- Compute cost in the storage layer at log-write time from `tokens_in` / `tokens_out`. If the destination response did not include token counts (some streaming providers omit them mid-stream), record `NULL`.
- New `goorouter cost` command reuses `storage.py` aggregation queries.
- New `0002_add_cost_columns.sql` migration adds the three columns and backfills from token counts using the active price table at migration time.

## Open questions / notes

- **Token counts on streaming responses.** Some OpenAI-compatible providers don't return `usage` for streamed responses. Options: (a) leave cost `NULL`, (b) estimate via a tokenizer (e.g., `tiktoken`), (c) configure per-backend whether to estimate. Probably (a) for v1.5; revisit if it's annoying.
- **Price freshness.** Hard-coded price table will go stale. Consider: a small JSON file under `goorouter/data/prices.json` updated each release, with the option for users to override in config.
- **Cost in `goo-explain`.** Useful but adds latency (we'd need to also know token counts before calling the backend, which means a tokenizer). Can be deferred to v1.6.

## Dependencies

- Requires `add-initial-router` to be archived (so the `requests` table exists with `tokens_in` / `tokens_out`).
- No external dependencies new to v1.
