# Add johnny Integration (SAINT ↔ johnny)

## Intent

Teach SAINT (the request router, formerly *goorouter*; PyPI `saint-router`, CLI `saint`,
virtual models `saint-auto`/`saint-explain`/`saint-<backend>`) to route to
**johnny-managed local seats** instead of (only) static, hardcoded backends. johnny is a local-inference environment manager that owns model
lifecycle, placement, and liveness across one or more machines. SAINT predates
johnny; this change capitalizes on johnny's existence to close SAINT's single
biggest structural gap.

The motivating problem: today a SAINT backend is static TOML
(`base_url` + `model`), and the policy grid **assumes the named backend is loaded and
serving**. There is no liveness check anywhere in the dispatch path. When the target
model isn't loaded — the normal state on a multi-model box where GPUs are shared and
idle seats are reaped to save power — the request fails at the backend URL. johnny knows
exactly which seats are up, can load one on demand, and owns the endpoint; SAINT
should ask it rather than assume.

The boundary is deliberate and preserved: SAINT integrates with johnny over its
**CLI (v0) or HTTP daemon (v1) — never as a library import**. This keeps SAINT
**fully functional standalone** (cloud backends and any fixed endpoint need no johnny),
keeps johnny free of SAINT's LiteLLM dependency, and means a johnny outage degrades
SAINT to its existing fallback behavior rather than breaking it.

## Scope

### In scope

- **Static config stays the default and baseline.** Every `[backends.<name>]` keeps its
  v1 static fields (`provider` + `model` + `base_url` + `api_key`). A backend with no
  johnny binding behaves exactly as today. Cloud/Anthropic backends are always static.
- **Optional johnny binding as an override overlay.** A backend MAY add `johnny_role`
  (or `johnny_seat`). When `[johnny]` is enabled, reachable, and the seat resolves
  `ready`, johnny's live endpoint + served-model + liveness **override** the static
  baseline for that dispatch. When johnny is disabled, unreachable, or the seat isn't
  ready, the backend **falls back to its own static config** — never to a different
  backend. An optional `johnny_only = true` declares a backend with no usable static
  baseline (may omit `base_url`/`model`; skips the static-baseline fallback step).
- **`[johnny]` config block.** `transport` (`"cli"` | `"http"`), `cli_path` /
  `base_url`, `resolve_cache_ttl_s`, and `ensure_load` (whether SAINT may *trigger*
  loads or only observe). Absent block + no bound backends ⇒ SAINT behaves exactly as
  it does today.
- **Resolution layer.** A small interface with two implementations (CLI shell-out, HTTP)
  that calls johnny's `resolve <role|seat>` →
  `{seat, endpoint, model, state, eta_s, queue_depth}`, with a short TTL cache. This is
  the only johnny call on the hot path.
- **Liveness-aware dispatch with a static floor.** For a bound backend: `ready` →
  dispatch to the resolved (override) endpoint/model; `loading`/`absent` → (optionally)
  trigger ensure-load and serve via the fallback order **`while_loading` → own static
  baseline → `default_on_failure`**; `failed` → same fallback order without a load.
- **`while_loading` target.** New `[routing] while_loading` (global) with optional
  per-backend override. Semantically distinct from `default_on_failure`: a warming seat
  is a *normal, expected* state on this hardware, not a failure. The backend's own static
  baseline is the implicit floor beneath `while_loading`.
- **Liveness-aware `explain`.** `saint-explain` and `saint explain` report the resolved
  seat, its state, `eta_s`, whether the override or the static baseline would serve, and
  which fallback would fire — so a dry-run reflects reality, not just the policy lookup.
- **Telemetry tagging + provide-to-johnny.** The request log gains `johnny_seat` and
  `state_at_dispatch` (`johnny_ready` | `static_baseline` | `while_loading` | `fallback` |
  NULL) for SAINT's own log. Separately, SAINT **provides** normalized
  per-request latency/tokens to **johnny's ingest schema** (johnny owns the contract) by
  appending to a durable spool (`$XDG_STATE_HOME/johnny/ingest/`) — johnny accepts it as
  `source=proxy`, the uniform cross-backend latency it can't pull from LM Studio/Ollama.
  Best-effort, non-fatal, daemon-optional; HTTP-to-johnnyd is an optional fast path.
  Providing real **TTFT** adds a first-chunk timestamp to SAINT's streaming path.
- **`pin` integration.** When `ensure_load` is enabled and SAINT triggers a load, it
  may `johnny pin` the seat for a short TTL to cover the load window against the reaper.
- Cross-platform behavior unchanged; CLI transport requires `johnny` on PATH only when a
  bound backend is configured.

### Out of scope (deferred)

- **johnny generating the policy grid.** The grid's vocabulary (`domain × complexity`,
  per urgency) stays human-authored — that is SAINT's design intent. A separate,
  advisory `suggest-policy` (fed by johnny `--bench` data) may come later; it never
  auto-edits the grid.
- **Library/in-process coupling to johnny.** Explicitly rejected to preserve standalone
  operation and dependency isolation.
- **SAINT triggering induction/tuning.** SAINT consumes johnny's results; it does
  not drive johnny's tuning pipeline.
- **A watch/streaming subscription to fleet state.** v0/v1 poll via `resolve` (cached).
  A push/subscribe channel can come with johnnyd maturity if polling proves insufficient.
- **Cross-machine reachability management.** johnny owns placement and bind/advertise
  addresses; SAINT simply uses the endpoint johnny returns. Making a cross-box seat
  reachable is johnny's concern, not SAINT's.

## Approach

`config.py` keeps the v1 static backend fields and adds an optional johnny binding
(`johnny_role`/`johnny_seat`, optional `johnny_only`, optional per-backend
`while_loading`) plus a `[johnny]` block. Static fields remain the baseline and stay
required unless `johnny_only`. Config validation gains johnny-specific rules (`[johnny]`
required iff any backend is bound; `johnny_only` permits omitting `base_url`/`model`;
`while_loading`/`default_on_failure` targets must resolve to defined backends).

A new `johnny.py` module defines `JohnnyResolver` (protocol) with `CliResolver` and
`HttpResolver` implementations and a TTL cache. `router.py`'s dispatch path gains a
resolution step before `call_backend` **only for bound backends**: on `ready` it overrides
the endpoint/model with johnny's resolved values; on a non-ready state it serves via the
fallback order `while_loading → static baseline → default_on_failure` (triggering a
non-blocking ensure-load for `loading`/`absent` when `ensure_load`). `backends.py`'s
`call_backend` is unchanged — it still takes a concrete endpoint/model/key; the resolver
supplies the override values when applicable, and the static baseline supplies them
otherwise. The classifier path reuses its **existing** classifier-fallback when its own
backend resolves non-ready (no new mechanism).

Standalone operation is the invariant: if no backend is bound, no resolver is constructed
and no johnny call is ever made. If johnny is configured but unreachable, bound backends
serve via their **own static baseline** (or `while_loading`/`default_on_failure` for
`johnny_only` backends) and a warning is emitted — SAINT keeps serving.

## Capabilities

This change touches three capability domains, each with a delta spec under [`specs/`](specs/):

- **config** — the optional johnny binding, the `[johnny]` block, `while_loading`, and the new
  validation rules.
- **routing** — liveness resolution, non-blocking ensure-load, `while_loading` dispatch,
  classifier-seat handling, standalone/degraded behavior.
- **observability** — `johnny_seat` + `state_at_dispatch` logging and liveness-aware
  `explain`.
