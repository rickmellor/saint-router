# observability — Delta Spec

## ADDED Requirements

### Requirement: johnny attribution in the request log

The request log SHALL record, for each request, `johnny_seat` (the resolved seat id when
the chosen backend was johnny-bound and resolved, else NULL) and `state_at_dispatch`, one
of: `johnny_ready` (served via johnny's override endpoint), `static_baseline` (bound
backend served via its own static config because johnny was unreachable or the seat was
not ready and no `while_loading` applied), `while_loading` (served via a `while_loading`
target while the seat warms), `fallback` (served via `default_on_failure`), or NULL (the
backend had no johnny binding). These columns SHALL be added via a forward migration that
increments the schema version; existing rows SHALL remain valid with NULL values.

#### Scenario: Ready override records the seat and johnny_ready
- GIVEN a request dispatched via johnny's override of a bound backend resolving to seat
  `coder@boxA`
- WHEN the row is written
- THEN `johnny_seat` is `coder@boxA` and `state_at_dispatch` is `johnny_ready`

#### Scenario: Static-baseline fallback is distinguishable
- GIVEN a bound backend served via its own static `base_url` because johnny was unreachable
- WHEN the row is written
- THEN `state_at_dispatch` is `static_baseline`

#### Scenario: Deferred request records while_loading
- GIVEN a request served via a `while_loading` target because the intended seat was loading
- WHEN the row is written
- THEN `state_at_dispatch` is `while_loading`
- AND `backend_chosen` records the intended backend, not the `while_loading` target

#### Scenario: Unbound backend leaves johnny fields NULL
- GIVEN a request dispatched to a backend with no johnny binding
- WHEN the row is written
- THEN `johnny_seat` is NULL and `state_at_dispatch` is NULL

### Requirement: Liveness-aware explain

In explain mode (`saint-explain` virtual model and `saint explain` CLI), when the chosen
backend is johnny-bound the breakdown SHALL include the resolved seat, its state, `eta_s`
when loading, whether johnny's override or the static baseline would be used, and the
fallback target that would serve the request if the seat is not ready. Explain mode SHALL
NOT trigger a load.

#### Scenario: Explain reports override vs static baseline
- GIVEN `model = "saint-explain"` and a prompt routing to bound `local-coder`
- AND `local-coder` resolves `state = "ready"` at a johnny endpoint differing from its static `base_url`
- WHEN the request is processed
- THEN the response shows the route to `local-coder`, the resolved seat/state, and that
  johnny's override endpoint would be used (naming it alongside the static baseline)
- AND no load is triggered and the destination backend is not called

#### Scenario: Explain reports a loading seat and its stand-in
- GIVEN `model = "saint-explain"` and a prompt routing to bound `local-coder`
- AND `local-coder` resolves `state = "loading"`, `eta_s = 40`
- WHEN the request is processed
- THEN the response describes the `loading` state and `eta_s`
- AND names the fallback that would serve the request (while_loading / static baseline / default)
- AND no load is triggered

### Requirement: Provide normalized telemetry to johnny

When a request is dispatched to a johnny-bound seat, SAINT SHALL provide a normalized
telemetry record to johnny's ingest in johnny's schema (johnny owns the contract). The
default mechanism SHALL be an append to a durable, append-only spool in johnny's state
directory; an HTTP POST to johnnyd MAY be used when configured. The provide operation SHALL
be best-effort and non-fatal: a failure to provide telemetry SHALL NOT affect the response
or the request's own log row. Records SHALL be tagged so johnny attributes them
`source = proxy`.

#### Scenario: Telemetry record provided on a johnny-bound dispatch
- GIVEN a request dispatched to (or on behalf of) a johnny-bound seat `coder@boxA`
- WHEN the request completes
- THEN a normalized record (seat, latency, tokens, success) is appended to johnny's ingest
- AND the record identifies the source as the proxy

#### Scenario: Ingest failure does not affect the response
- GIVEN johnny's ingest target is unwritable (e.g. johnny not installed, dir missing)
- WHEN a request to a johnny-bound seat completes
- THEN the client still receives the normal response
- AND SAINT's own request-log row is still written
- AND the failure is reported only to stderr

#### Scenario: Unbound backend does not provide johnny telemetry
- GIVEN a request dispatched to a backend with no johnny binding
- WHEN the request completes
- THEN no record is appended to johnny's ingest

### Requirement: Measure TTFT for streaming requests

To provide time-to-first-token, SAINT SHALL timestamp the first streamed chunk of a
streaming response and include the resulting TTFT in the telemetry record. For
non-streaming requests, TTFT is not separable and SHALL be omitted (total latency is still
provided).

#### Scenario: Streaming request records TTFT
- GIVEN a streaming request to a johnny-bound seat
- WHEN the first response chunk is yielded
- THEN the elapsed time to that chunk is captured as TTFT in the telemetry record

#### Scenario: Non-streaming request omits TTFT
- GIVEN a non-streaming request to a johnny-bound seat
- WHEN the request completes
- THEN the telemetry record provides total latency and omits TTFT
