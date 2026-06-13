# routing — Delta Spec

## ADDED Requirements

### Requirement: johnny override of the static baseline

For a backend carrying a johnny binding (`johnny_role`/`johnny_seat`), when `[johnny]` is
enabled the router SHALL resolve the seat's live endpoint, served-model name, and
readiness state from johnny at dispatch time and, on `ready`, use them to **override** the
static `base_url`/`model`. Resolution results SHALL be cached for `resolve_cache_ttl_s`.
The static baseline is never discarded; it remains the fallback when the override does not
apply.

#### Scenario: Ready seat overrides the static endpoint
- GIVEN `local-coder` with static `base_url = "http://127.0.0.1:1234/v1"`, `model = "x"`,
  and `johnny_role = "coder"`
- AND johnny resolves `coder` to `{endpoint: "http://127.0.0.1:8002/v1", model: "qwen3-coder", state: "ready"}`
- WHEN a request routes to `local-coder`
- THEN the request is dispatched to `http://127.0.0.1:8002/v1` with model `qwen3-coder`
  (johnny's override, not the static baseline)
- AND `state_at_dispatch` is logged as `johnny_ready`

#### Scenario: Resolution cache avoids per-request johnny calls
- GIVEN `resolve_cache_ttl_s = 1` and a ready bound backend
- WHEN two requests route to the same backend within 1 second
- THEN johnny is resolved at most once for that window

### Requirement: Non-blocking load on a cold seat, static baseline as floor

When a bound backend's seat resolves `loading`/`absent`/`failed`, the router SHALL serve
the current request WITHOUT waiting for a load, using the fallback order: per-backend
`while_loading` -> global `while_loading` -> the backend's own static baseline (unless
`johnny_only`) -> `default_on_failure`. When the state is `loading`/`absent` and
`[johnny] ensure_load` is true, the router SHALL additionally trigger a non-blocking
ensure-load (and MAY pin the seat for the load window). It SHALL NOT trigger a load for a
`failed` state.

#### Scenario: Cold seat triggers load and serves via while_loading
- GIVEN bound `local-coder` resolves `state = "absent"`, `ensure_load = true`
- AND a `while_loading` target of `cloud-small` applies
- WHEN a request routes to `local-coder`
- THEN johnny is asked to load `coder` (non-blocking)
- AND the request is dispatched to `cloud-small`
- AND `state_at_dispatch` is logged as `while_loading`
- AND `backend_chosen` records the intended backend (`local-coder`)

#### Scenario: Cold seat with no while_loading falls to the static baseline
- GIVEN bound `local-coder` with a static `base_url`, no `while_loading` configured,
  resolves `state = "loading"`
- WHEN a request routes to `local-coder`
- THEN the request is served via `local-coder`'s own static `base_url`/`model`
- AND `state_at_dispatch` is logged as `static_baseline`

#### Scenario: ensure_load disabled does not trigger a load
- GIVEN bound `local-coder` resolves `state = "loading"` and `ensure_load = false`
- WHEN a request routes to `local-coder`
- THEN johnny is NOT asked to load the seat
- AND the request is served via the applicable fallback (while_loading / static / default)

#### Scenario: Subsequent request lands on the warmed seat
- GIVEN a prior request triggered a load of `coder`
- AND johnny now resolves `coder` `state = "ready"`
- WHEN a new request routes to `local-coder`
- THEN the request is dispatched to the resolved live endpoint
- AND `state_at_dispatch` is logged as `johnny_ready`

### Requirement: while_loading is distinct from default_on_failure

`while_loading` SHALL be used only for the transient case of a bound seat that is not
ready. It is configured independently of `default_on_failure`, which is reserved for the
terminal case where the chosen serving target itself fails.

#### Scenario: Static floor also fails, cascades to default_on_failure
- GIVEN bound `local-coder` resolves `loading`, no `while_loading` configured, and its
  static `base_url` is also unreachable
- WHEN a request routes to `local-coder`
- THEN the static baseline is attempted and fails
- AND the request is then served via `default_on_failure`
- AND `state_at_dispatch` is logged as `fallback`

### Requirement: Classifier on a bound backend degrades via existing fallback

When the classifier backend carries a johnny binding and resolves to a non-ready state,
the router SHALL treat this as a classifier failure and follow the existing
classifier-fallback path (`fallback_backend`, then `default_on_failure`). It SHALL NOT
block the request waiting for the classifier seat to load.

#### Scenario: Non-ready classifier seat falls back without blocking
- GIVEN the classifier backend is bound and resolves `state = "loading"`
- AND a `fallback_backend` is configured
- WHEN a `saint-auto` request is processed
- THEN the request is not blocked on the classifier seat loading
- AND classification follows the existing fallback path
- AND (if `ensure_load`) the classifier seat is asked to load for future requests

### Requirement: Standalone and degraded operation

When no backend carries a johnny binding (or `[johnny]` is absent), the router SHALL make
no johnny calls and behave identically to the pre-integration router. When a bound backend
is configured but johnny is unreachable, the router SHALL serve affected requests via the
backend's **own static baseline** (unless `johnny_only`, in which case via
`while_loading`/`default_on_failure`) and emit a warning, without failing the request
solely because johnny is unreachable.

#### Scenario: No bindings means no johnny calls
- GIVEN a config with only static backends and no bindings
- WHEN any request is processed
- THEN the resolver is never invoked

#### Scenario: johnny unreachable degrades to the static baseline
- GIVEN bound `local-coder` with a static `base_url`, and a johnny resolver that errors
  (johnny not running)
- WHEN a request routes to `local-coder`
- THEN the request is served via `local-coder`'s own static `base_url`/`model`
- AND `state_at_dispatch` is logged as `static_baseline`
- AND a warning is emitted identifying the unreachable johnny resolver

#### Scenario: johnny_only backend with johnny unreachable uses fallback
- GIVEN bound `managed-vision` with `johnny_only = true` and an unreachable resolver
- WHEN a request routes to `managed-vision`
- THEN the request is served via `while_loading`/`default_on_failure` (no static baseline exists)
- AND a warning is emitted
