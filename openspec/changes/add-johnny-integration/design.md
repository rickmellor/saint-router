# Design — johnny Integration

## Context

SAINT v1 dispatches to static backends declared in TOML. The policy grid maps
`(urgency, domain, complexity) → backend_name`; the named backend is assumed live. On a
shared-GPU box running johnny — where seats are loaded on demand and idle ones reaped to
reclaim ~80 W/card — that assumption is false most of the time. johnny exposes liveness
and on-demand loading; this change consumes it.

## Decision 1 — Static config is the baseline; johnny is an override overlay

**Rejected:** a discriminated `kind = static | johnny` where a johnny backend has no static
fields and resolves everything from johnny (so a johnny outage routes to a *different*
backend).

**Chosen:** every backend keeps its static fields as the default and baseline; an optional
`johnny_role`/`johnny_seat` binding overrides the endpoint/model/liveness **only when**
johnny is enabled, reachable, and the seat is `ready`. Otherwise the backend uses its own
static config.

Rationale — this is what keeps the two tools genuinely independent:
- **Graceful degradation is structural, not a special path.** A johnny outage doesn't
  reroute to some other backend; the bound backend simply uses its own static endpoint,
  which may still be serving (e.g. LM Studio is up at `localhost:1234` whether or not
  johnny is tracking it). The static config is a real, usable floor — not a placeholder.
- **Trivial migration.** An existing user adds `johnny_role` + a `[johnny]` block to a
  working backend and changes nothing else. Removing them reverts to pure v1.
- **The override only ever *improves* on the baseline** (corrects a moved port, a
  cross-box seat, a re-induct's renamed model), and silently steps aside when it can't.

`johnny_only = true` is the opt-out for the minority case of a seat with no sensible static
equivalent — it omits the baseline and accepts the reroute-on-outage behavior.

## Decision 2 — Integrate over CLI/HTTP, never as a library

**Rejected:** importing johnny as a Python package and calling its engine in-process.

**Chosen:** a `JohnnyResolver` protocol with two implementations — `CliResolver`
(subprocess `johnny resolve --json`, etc.) for v0, `HttpResolver` (johnnyd) for v1.

Rationale:
- **Standalone operation.** SAINT must keep working with zero johnny installed when
  only static/cloud backends are configured. A hard import breaks that.
- **Dependency isolation.** johnny is deliberately LiteLLM-free; SAINT is built on
  LiteLLM. Co-installing in one venv drags each tool's deps into the other. Separate
  processes keep both `pipx`/`uv tool` installs clean.
- **Failure isolation.** A johnny crash should degrade SAINT to its static baseline,
  not take it down. Process boundary gives that for free.

Cost: subprocess latency per cache-miss on the CLI transport. Mitigated by the TTL cache
(below) and human-paced personal traffic; the HTTP transport removes it for throughput
use. This is the same trade johnny itself makes (stateless CLI core, daemon only when a
feature demands it).

## Decision 3 — `resolve` as the single hot-path primitive

SAINT needs, per dispatch to a bound backend: *where is this role, what model name does
it serve, and is it ready?* Composing that from a full `status --json` fleet dump is
wasteful and over-couples SAINT to johnny's full status shape.

johnny exposes a focused **`resolve <role|seat>`** returning
`{seat, endpoint, model, state ∈ {ready, loading, absent, failed}, eta_s, queue_depth}`.
Read-only, fast, cacheable. SAINT caches results for `resolve_cache_ttl_s` (default
1 s) keyed by role/seat. A `loading` result is cheap to re-poll; a `ready` result is
stable until the seat is reaped (and a reaped-then-cold seat just re-resolves on the next
miss).

`resolve` is kept distinct from the imperative `up --wait` (Decision 4). An optional
future `resolve --ensure` could fuse them for callers wanting one round-trip; not needed
for v1.

## Decision 4 — Never block on a load; fallback order keeps the static floor

Cold start on this hardware is **minutes** (weight load + CUDA/HIP graph capture). A
router that blocks the triggering request on that is unusable. So for a bound backend whose
seat is not `ready`:

- `loading`/`absent` with `ensure_load = true`: fire `johnny up <role>` (idempotent,
  **non-blocking** — returns `{state: loading, eta_s}` immediately), optionally
  `johnny pin` the seat for the load window, then serve **this** request via the fallback
  order below. Subsequent requests resolve `ready`.
- `loading`/`absent` with `ensure_load = false`: do not trigger; serve via the fallback
  order.
- `failed`: serve via the fallback order without a load (don't thrash a crashed seat).

**Fallback order (honors "static is the default" at every step):**
`while_loading` (per-backend → global) → the backend's **own static baseline** (unless
`johnny_only`) → `default_on_failure`.

`while_loading` is a **distinct config** from `default_on_failure` because the two model
different conditions. `default_on_failure` means *the routing machinery broke*.
`while_loading` means *the intended seat is coming up as designed* — an expected,
transient, non-error state that is the common case on a reaped box. The backend's static
baseline sits beneath `while_loading` as the implicit floor: if you set no `while_loading`,
a warming seat's traffic tries the static endpoint (which may be independently live), then
cascades to `default_on_failure`. Folding `while_loading` into `default_on_failure` would
make logs and `explain` lie and force the last-resort target (often cloud) to double as
the warm-up target (often a cheaper local generalist).

**Granularity:** the `while_loading` override lives on the *backend* (the thing that
loads), not the policy cell. The loading property belongs to the seat; attaching it to all
18 cells would be verbose and redundant. No identified case needs per-cell warm-up targets.

## Decision 5 — Classifier-seat handling reuses existing fallback

If SAINT's *classifier* backend is itself johnny-bound, it runs on ~every `saint-auto`
request. A reap → cold-start would tax the first post-idle request with a multi-minute
classifier load — pathological.

Mitigations, all from existing parts:
- **Operational:** pin the classifier seat in johnny (profile `pinned: true` or
  `johnny pin`), or keep the classifier on a cheap static/cloud backend (Haiku-class is
  fast and cheap). Documented as the recommended setup.
- **Mechanical:** treat a non-`ready` classifier resolve as a **classify failure**, which
  takes SAINT's **already-existing** classifier-fallback path
  (`fallback_backend` → `default_on_failure`). No new code path; the existing
  oversize/error fallback simply gains one more trigger.

## Decision 6 — Telemetry: SAINT *provides*, johnny *accepts* (push, not pull)

SAINT already writes one SQLite row per request with `backend_latency_ms`,
`tokens_in/out`, `success`, and classifier latency — and already versions that schema with
migrations. That log stays, for SAINT's own purposes (relabel-for-training, routing
analytics, `log show`). Two new columns aid both SAINT and the push:
- `johnny_seat` — the resolved seat id when the backend was johnny-bound (else NULL).
- `state_at_dispatch` — `johnny_ready` | `static_baseline` | `while_loading` | `fallback`
  (NULL for unbound backends), so the served path is unambiguous.

**Direction: SAINT provides; johnny accepts.** johnny cannot get uniform telemetry by
pulling — vLLM exposes Prometheus, but LM Studio/Ollama do not, and the proxy is the only
place that sees every request to every backend identically. So SAINT maps each
dispatched request to **johnny's normalized ingest schema** (the acceptor owns the
contract; SAINT conforms) and provides it. This generalizes to any future provider and
avoids johnny reverse-engineering SAINT's internal `requests` table.

**Mechanism: a durable spool, daemon-optional.** Because johnny's core is stateless until
its daemon exists, SAINT appends one record per request to a johnny-owned append-only
spool (`$XDG_STATE_HOME/johnny/ingest/`, JSONL); johnny ingests on its reaper/poller tick.
This is *better* than push-to-daemon for intermittent use: if johnny isn't running, records
accumulate and are ingested later rather than dropped. An HTTP POST to johnnyd is an
optional low-latency path once the daemon runs. The write is best-effort and non-fatal —
same discipline as the existing `_safe_log`; a telemetry-provide failure never affects the
response.

**TTFT requires a first-chunk timestamp.** SAINT today measures total stream time
(`backend_latency_ms`). To provide real TTFT it timestamps the first yielded chunk in the
streaming generator (`gen()` in `server.py`); non-streaming requests provide total latency
only (TTFT is not separable there). This is the one new measurement the push needs.

## Decision 7 — Emergent multi-machine, with a reachability caveat

Because a johnny-bound backend resolves its endpoint *from johnny*, and johnny owns
placement across nodes, a backend can resolve to a seat on **another machine** with no
SAINT awareness — SAINT gains cross-box routing for free. The one caveat is that
the endpoint johnny returns must be reachable from SAINT's host. This interacts with
johnny's localhost-default security posture: cross-box seats require explicit LAN exposure
or johnny's agent transport. SAINT does not manage this; it uses whatever endpoint
`resolve` returns and surfaces a connection error like any other backend failure.

## Interaction with the reaper (why this is safe to be aggressive)

SAINT already spills to cloud (or any configured fallback) under pressure. That makes
johnny reaping a local seat **safe from SAINT's perspective**: a request for a
cold seat routes to `while_loading` (or cloud) while the seat warms, instead of failing.
This lets the johnny side run a **more aggressive idle TTL** than a johnny-only deployment
would tolerate — the user-facing cost of a wrongly-reaped seat is one request served by a
fallback plus a background reload, not an error. The two features actively cover for each
other.

## Deferred / future work

- **`resolve --ensure`** fused query+load for one-round-trip callers.
- **Fleet watch/subscribe** instead of TTL-poll, if polling latency matters under load.
- **`suggest-policy`** — advisory grid edits from johnny `--bench` winners; never
  auto-applied.
- **HTTP push** as an optional low-latency ingest path once johnnyd is the common
  transport (the durable spool remains the daemon-free default).
- **Per-urgency `while_loading`** if a real case for urgency-dependent warm-up appears
  (e.g. `!urgent` should warm-up to cloud, `!patient` should wait on local).
