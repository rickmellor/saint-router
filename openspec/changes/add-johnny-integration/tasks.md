# Tasks — johnny Integration

> Per the OpenSpec lifecycle, this checklist is filled in at the **Implementing** stage,
> after the proposal and design are approved. Below is the intended task skeleton; expand
> into a numbered checklist when implementation begins.

## Config layer
- [ ] Discriminated backend model (`StaticBackend` | `JohnnyBackend`) in `config.py`
- [ ] `[johnny]` block parsing (`transport`, `cli_path`/`base_url`, `resolve_cache_ttl_s`, `ensure_load`)
- [ ] `[routing] while_loading` + per-backend `while_loading` override
- [ ] Validation rules (static baseline required unless johnny_only; `[johnny]` required iff any backend bound; target resolution)
- [ ] `config show` surfaces johnny bindings and the `[johnny]` block (keys masked as today)

## Resolver layer
- [ ] `johnny.py`: `JohnnyResolver` protocol + TTL cache
- [ ] `CliResolver` (subprocess `johnny resolve`/`up`/`pin --json`)
- [ ] `HttpResolver` (johnnyd) — behind the same protocol
- [ ] Unreachable-johnny handling → warning + fallback, never a hard failure

## Routing layer
- [ ] Resolution step before dispatch for johnny-bound targets (override on ready)
- [ ] Non-blocking ensure-load + `while_loading` dispatch on `loading`/`absent`
- [ ] `failed`/error → `default_on_failure`
- [ ] Classifier-seat non-ready → existing classifier-fallback path
- [ ] Optional `johnny pin` over the load window when `ensure_load`

## Observability
- [ ] Migration: `johnny_seat` + `state_at_dispatch` columns (schema version bump)
- [ ] Populate both in the log row builder
- [ ] First-chunk timestamp in the streaming generator → TTFT measurement
- [ ] Provide normalized telemetry to johnny's ingest schema (spool append; optional HTTP)
- [ ] Best-effort/non-fatal provide path (stderr on failure, never affects response/log)
- [ ] Liveness-aware `explain` (API + CLI)

## Docs / tests
- [ ] `config.example.toml`: a johnny-bound backend (static + binding) + `[johnny]` block + `while_loading`
- [ ] README: johnny integration section + classifier-seat pin recommendation
- [ ] Tests: resolver (both transports, mocked), cold-seat fallback, standalone no-op,
      johnny-unreachable degradation, classifier-seat fallback, migration round-trip,
      telemetry spool append + non-fatal-on-failure, TTFT first-chunk timing
- [ ] CI unchanged (no new platform matrix; johnny calls mocked in tests)
