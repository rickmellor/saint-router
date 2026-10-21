# `GET /usage/v1` — SAINT as a `hybrid-usage/1` provider

A spend report that reads only the cloud bill cannot see the work this router sent to local seats,
and this router cannot see the cloud bill. `GET /usage/v1` lets one report show both: SAINT serves
its own request log as aggregated usage — tokens, per backend, with who pays for each — and a
consumer puts it next to the bill it already reads.

Read-only, one pass over the log, never any prompt or completion text.

**The canonical contract is not in this repo.** It lives in `neurio/hsp-arch-tools`,
`docs/hybrid-inference-usage-contract.md`. This file covers only what SAINT does with it; when the
two disagree, the contract wins. Reference consumer: `bedrock-spend-gui` (`--local URL`).

```
GET http://127.0.0.1:4000/usage/v1?from=2026-10-01&to=2026-10-20&granularity=day&client=claude-code
```

Parameters, response shape and error codes are the contract's (`from`/`to` default to today UTC and
today − 29 days; a window over 366 days, an unknown parameter, a bad date, an unknown granularity
or an empty filter value are 400s). Implementation: `saint/usage.py`, route in `saint/server.py`,
tests in `tests/test_usage.py`.

## Attribution

Four values on every logged row (migration `0005_attribution.sql`):

| column | from |
|---|---|
| `client` | derived from `User-Agent` — the first matching `[clients]` rule name, else the UA's first product token lowercased (`curl/8.5` → `curl`), else `unknown` |
| `origin` | `X-Inference-Origin` — the machine the request came from |
| `session` | `X-Inference-Session` — the orchestrating session that caused it |
| `job_id` | `X-Inference-Job` — one unit of delegated work |

`client` is recorded on **every** request, whether or not it influenced routing; the `[clients]`
rules still only route in `router.decide_route`, and this label never changes a decision. The
`X-Inference-*` headers are for SAINT's log: they are **never forwarded to a backend**
(`backends._provider_kwargs` pins `extra_headers` to `{}`), and they carry no prompt content.
Values are stripped, non-printable-ASCII removed and cut at 128 characters; empty is stored NULL.

Claude Code sends them via `ANTHROPIC_CUSTOM_HEADERS`; `saint-agent` sets them per worker job.

## SAINT specifics

- **`by_backend` is the backend or seat that SERVED the request**, not what routing asked for
  (`usage.served_backend`). `backend_chosen` records the decision, and on two kinds of row that is
  not the hardware that ran it: a **pool** row logs the pool backend (`local-pool`) while one member
  seat answered, and a **spill** row logs the requested backend while another role's seat did the
  work. Both put the actual seat in `johnny_seat`, so SAINT resolves it by the seat id's trailing
  `-<port>` against each backend's `base_url` port — restricted to the pool's members for pool
  rows, and preferring johnny-bound backends otherwise. What it cannot resolve stays under
  `backend_chosen`: a seat id with no port (`gemma31.service`, `saint-features.service#embed`),
  NULL, or a port several backends share. `billing`, `energy_cost_usd` and the `by_backend` key all
  follow the served seat, so pool and spill rows are priced on the machine that burned the watts.
  Filters are unaffected.
- **`by_route`** — SAINT's own breakdown of how the rows were dispatched (a contract extension; the
  contract allows extra fields), over every row matching the filters like `by_client`:
  `pool` (route = the pool backend's name, e.g. `local-pool`), `pinned` (a `!alias` prefix or a
  `saint-<backend>` model id forced it; route = `pinned`) and `routed` (the classifier picked it;
  route = `routed`). Counts are requests and tokens only, like the other non-money breakdowns.
- **`billing` per backend** — `local` when the backend is johnny-bound or its name starts with
  `local` (the same test `saint savings` uses); otherwise from the backend's `provider`:
  `anthropic`→`anthropic`, `bedrock`→`bedrock`, `openai`→`openai`, `vertex_ai`/`vertex`→`vertex`,
  `azure`→`azure`. Anything else, and any backend SAINT cannot resolve at all, is `other`.
  Renamed or aliased backends resolve through `savings.resolve_backend`, so rows written before a
  rename keep their billing instead of degrading to `other`.
- **Tokens** — SAINT logs one prompt-token number that includes cache hits, so
  `input = max(tokens_in − cache_read_tokens − cache_write_tokens, 0)`, clamped at 0. All four keys
  are always present; missing data is `0`, never omitted.
- **Excluded** — `saint-explain` (answers with the routing decision; no dispatch happens) and
  `saint-embeddings` (infra traffic that must never be priced at chat rates), the same exclusions
  `storage.usage_stats` applies.
- **`billed_cost_usd`** — for priced cloud rows, the same formula `saint savings` uses for its
  `real` figure: uncached input at `price_in`, cache reads and writes at their effective prices,
  output at `price_out`. `null` when the backend has no price; local rows are `null`, since local
  hardware carries no per-token bill. Money is shared with `saint/savings.py`, so the savings
  report and this endpoint cannot price the same tokens differently.
- **`energy_cost_usd`** — electricity for local rows: `tokens_out / 1e6 ×` a `$/Mtok` rate derived
  from `[energy]`. Decode rate comes from the backend's `tok_s` (a `johnny bench` figure) or, when
  config is silent, the request log's measured p75; watts from `local_seat_watts`
  (`[energy].host_watts`/`gpu_watts`/`seat_gpus`, or a backend's own `watts`). **Un-costed rather
  than fabricated**: `null` when no decode rate is known, with a `NO_ENERGY_MODEL` note saying so.
  It is an estimate — say so when displaying it.
- **`notes`** — `PARTIAL_PERIOD` when `to` is today or later; `NO_ENERGY_MODEL` when local rows
  exist but none could be priced; `HEADERS_SINCE` naming the date attribution started (or that the
  log holds no attributed rows yet) so a consumer can explain an `origin` filter that returns little.
- **Auth** — if `SAINT_USAGE_TOKEN` is set and non-empty, `/usage/v1` requires
  `Authorization: Bearer <token>` (constant-time compare) and answers 401 `UNAUTHORIZED` without
  one. Unset or empty means open, which is fine on a trusted LAN: origin and session names can
  still reveal machine names and work patterns, so don't expose the endpoint.
- **Errors** — the contract's shape, matching the ccwb spend API so one consumer serves both:
  `UNKNOWN_PARAM` and `INVALID_PARAM` (400), `UNAUTHORIZED` (401), `UNAVAILABLE` (503, when the
  request log can't be read).

## From the command line

The same JSON, straight from the local log, no server needed:

    saint usage --from 2026-10-01 --to 2026-10-20 --granularity day
    saint usage --client claude-code --origin GEN-PW0DXJ6E --session c33f9e30

`--from/--to/--granularity/--client/--origin/--session` mirror the query parameters; a bad
parameter prints the error body and exits 2.

## Conformance

```bash
B=http://127.0.0.1:4000
curl -s "$B/usage/v1?granularity=total" | jq -e '.contract == "hybrid-usage/1"
  and (.totals.tokens_by_type | keys == ["cache_creation","cache_read","input","output"])'
curl -s -o /dev/null -w '%{http_code}\n' "$B/usage/v1?grain=day"        # 400
curl -s "$B/usage/v1?from=2026-10-20&to=2026-10-01" | jq -r .code       # INVALID_PARAM
```
