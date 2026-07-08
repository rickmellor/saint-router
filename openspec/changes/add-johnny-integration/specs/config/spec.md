# config — Delta Spec

## ADDED Requirements

### Requirement: Static backend config is the default and baseline

A `[backends.<name>]` entry SHALL retain its v1 static fields (`provider`, `model`,
`base_url`, `api_key`, `timeout`) as the default and baseline. A backend with no johnny
binding SHALL behave exactly as in v1. Static fields are REQUIRED unless the backend sets
`johnny_only = true` (see below). Cloud/Anthropic backends are always static.

#### Scenario: A backend with no johnny binding is unchanged
- GIVEN a backend with `provider`, `model`, `base_url` and no johnny fields
- WHEN config is loaded
- THEN the backend behaves identically to v1 (no johnny involvement at dispatch)

### Requirement: Optional johnny binding as an override overlay

A `[backends.<name>]` entry MAY add a johnny binding via `johnny_role` (or `johnny_seat`).
The binding is an **override overlay**: it does not replace the static fields. When
`[johnny]` is enabled and reachable and the bound seat resolves `ready`, johnny's resolved
endpoint and served-model name override the static `base_url`/`model` for that dispatch.
In all other cases (johnny disabled, unreachable, or the seat not ready) the static
baseline remains in effect per the routing fallback order.

A backend MAY set `johnny_only = true` to declare it has no usable static baseline; such a
backend MAY omit `base_url`/`model`, and its routing skips the static-baseline fallback
step (going `while_loading` -> `default_on_failure`).

#### Scenario: johnny binding does not require dropping static fields
- GIVEN a backend `local-coder` with `base_url`, `model`, AND `johnny_role = "coder"`
- WHEN config is validated
- THEN validation succeeds (static fields coexist with the johnny binding)

#### Scenario: johnny_only backend may omit static endpoint
- GIVEN a backend with `johnny_role = "coder"` and `johnny_only = true` and no `base_url`
- WHEN config is validated
- THEN validation succeeds

#### Scenario: Non-johnny_only backend still requires static fields
- GIVEN a backend with `johnny_role = "coder"`, no `johnny_only`, and no `base_url`
- WHEN config is validated
- THEN validation fails identifying the missing static baseline (`base_url`/`model`)

### Requirement: [johnny] configuration block

A `[johnny]` block SHALL configure resolver `transport` (`"cli"` | `"http"`), the relevant
locator (`cli_path` for CLI, `base_url` for HTTP), `resolve_cache_ttl_s` (default 1), and
`ensure_load` (default true). The block SHALL be required and valid if any backend carries
a johnny binding, and MAY be absent otherwise. When the block is absent (or no backend is
bound), the router SHALL make no johnny calls and behave as v1.

#### Scenario: Bound backend without [johnny] block is rejected
- GIVEN at least one backend with a `johnny_role`/`johnny_seat`
- AND no `[johnny]` block
- WHEN config is validated
- THEN validation fails identifying the missing `[johnny]` block

#### Scenario: [johnny] block absent is fine with no bindings
- GIVEN only static backends with no johnny bindings and no `[johnny]` block
- WHEN config is loaded
- THEN config loads successfully

### Requirement: while_loading target

`[routing]` SHALL support an optional `while_loading` backend name, and a johnny-bound
backend MAY specify a per-backend `while_loading` override. Any `while_loading` value SHALL
resolve to a defined backend. The serving target for a bound backend whose seat is not
ready is resolved in order: the backend's own `while_loading`, else the global
`[routing] while_loading`, else the backend's **own static baseline** (unless
`johnny_only`), else `default_on_failure`.

#### Scenario: while_loading must reference a defined backend
- GIVEN `[routing] while_loading = "nope"` and no backend named `nope`
- WHEN config is validated
- THEN validation fails identifying the undefined `while_loading` target

#### Scenario: Static baseline is the implicit while_loading floor
- GIVEN a johnny-bound backend `local-coder` with a static `base_url`, no per-backend
  `while_loading`, and no global `while_loading`
- WHEN `local-coder`'s seat is loading
- THEN the request is served via `local-coder`'s own static `base_url`/`model`
  (cascading to `default_on_failure` only if that endpoint also fails)

#### Scenario: Per-backend while_loading overrides the static floor
- GIVEN `local-coder` with a static `base_url` AND `while_loading = "cloud-small"`
- WHEN `local-coder`'s seat is loading
- THEN the request is served via `cloud-small`, not the static baseline
