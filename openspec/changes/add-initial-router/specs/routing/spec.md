# routing — Delta Spec

## ADDED Requirements

### Requirement: Prefix parsing in the latest user message

The router SHALL parse leading `!<token>` prefixes from the content of the latest message with `role = "user"`. The first `!` MUST be at character index 0 of the content (no leading whitespace); a leading space disables prefix parsing entirely for that message. Prefix tokens SHALL be matched case-sensitively. Prefixes SHALL be stripped from the content before forwarding to any backend (classifier or destination). Multiple prefixes are permitted, separated by whitespace, parsed left-to-right until the first non-prefix token. If multiple urgency tokens are present, the last one wins. If both an urgency token and a backend token are present, both apply (urgency is set; backend is pinned, making the urgency moot for routing but recorded in the log).

#### Scenario: Urgency prefix recognized and stripped
- GIVEN the latest user message has content `"!urgent fix this bug"`
- WHEN the request is processed
- THEN the effective urgency is `urgent`
- AND the content forwarded to any backend is `"fix this bug"`

#### Scenario: Backend prefix pins routing and is stripped
- GIVEN the latest user message has content `"!opus refactor this function"`
- AND the config defines a backend `cloud-large` with alias `opus`
- WHEN the request is processed
- THEN the chosen backend is `cloud-large`
- AND the classifier is not called
- AND the content forwarded to the backend is `"refactor this function"`

#### Scenario: Multiple prefixes parsed left-to-right
- GIVEN content `"!urgent !opus do this"`
- WHEN the request is processed
- THEN urgency is `urgent`
- AND backend is pinned to `cloud-large` (via alias `opus`)
- AND content forwarded is `"do this"`

#### Scenario: Unknown prefix returns 400
- GIVEN content `"!doesnotexist hello"`
- WHEN the request is processed
- THEN the proxy returns HTTP 400
- AND the response body identifies the unknown token and lists known urgency tokens and configured backend names

#### Scenario: Leading whitespace disables prefix parsing
- GIVEN content `" !opus please summarize"` (leading space)
- WHEN the request is processed
- THEN no prefix is parsed
- AND the content is forwarded to the chosen backend exactly as received (including the leading space)
- AND the auto-routing pipeline runs

#### Scenario: Last urgency token wins
- GIVEN content `"!urgent !patient do this"`
- WHEN the request is processed
- THEN effective urgency is `patient`
- AND content forwarded is `"do this"`

### Requirement: Model field routing

The router SHALL inspect the request's `model` field and route as follows: `goo-explain` enters explain mode; `goo-<backend-name>` (matching a configured backend) pins that backend and bypasses the classifier; any other value (including `goo-auto`) enters the auto-routing pipeline. Per-message prefixes always take precedence over the model field.

#### Scenario: goo-explain skips backend call
- GIVEN `model = "goo-explain"` and any user message
- WHEN the request is processed
- THEN the destination backend is NOT called
- AND the response is a chat completion whose content is the routing breakdown

#### Scenario: goo-<backend> pins routing
- GIVEN `model = "goo-cloud-large"` and a backend `cloud-large` is configured
- WHEN the request is processed (with no overriding prefix)
- THEN the classifier is NOT called
- AND the request is forwarded to `cloud-large`

#### Scenario: Per-message prefix overrides goo-<backend>
- GIVEN `model = "goo-cloud-large"` and the latest user message starts with `"!local-small "`
- WHEN the request is processed
- THEN the request is forwarded to `local-small`, not `cloud-large`

#### Scenario: Unknown model treated as auto
- GIVEN `model = "gpt-4"` (or any unrecognized name)
- WHEN the request is processed
- THEN the auto-routing pipeline runs (classifier + policy)

### Requirement: Classifier invocation and structured output

When the auto-routing pipeline runs and no backend is pinned, the router SHALL call the classifier backend with the (post-prefix-stripping) content of the latest user message and parse the response as JSON with fields `domain` (`"code"|"general"`), `complexity` (`"trivial"|"medium"|"hard"`), and `reason` (string).

#### Scenario: Successful classification
- GIVEN auto routing and a user message about debugging Python code
- WHEN the classifier returns `{"domain":"code","complexity":"medium","reason":"..."}`
- THEN the router proceeds to policy lookup with `(domain=code, complexity=medium, urgency=<effective>)`

#### Scenario: Classifier malformed JSON
- GIVEN the classifier returns text that does not parse as the expected JSON shape
- WHEN the response is received
- THEN the router treats it as a primary failure (per fallback requirements)

### Requirement: Classifier fallback chain

When `[classifier].fallback_backend` is configured, the router SHALL use it in two cases: (1) the classifier input length exceeds `[classifier].max_input_chars`, in which case the primary classifier is skipped entirely and the full prompt is sent to the fallback; (2) the primary classifier fails (timeout, network error, or malformed JSON), in which case the fallback is tried once. If the fallback also fails or no fallback is configured, the router SHALL use the `[routing].default_on_failure` backend.

#### Scenario: Oversize input bypasses primary
- GIVEN `max_input_chars = 8000` and a user message of 12 000 characters
- AND `fallback_backend = "local-large"` is configured
- WHEN the auto-routing pipeline runs
- THEN the primary classifier is NOT called
- AND the fallback classifier is called with the full message
- AND the log row's `classifier_used = "local-large"` and `classifier_fallback_reason = "oversize"`

#### Scenario: Primary error triggers fallback
- GIVEN the primary classifier times out
- AND `fallback_backend = "local-large"` is configured
- WHEN the auto-routing pipeline runs
- THEN the fallback classifier is called
- AND the log row's `classifier_fallback_reason = "primary_error"`

#### Scenario: No fallback, oversize falls back to head truncation
- GIVEN no `fallback_backend` set, `max_input_chars = 8000`, and a 12 000-char message
- WHEN the auto-routing pipeline runs
- THEN the primary classifier is called with the first 8 000 characters
- AND the log row's `classifier_input_truncated_from = 12000`
- AND the log row's `classifier_input_chars = 8000`

#### Scenario: Both classifiers fail
- GIVEN primary and fallback both error or time out
- WHEN the auto-routing pipeline runs
- THEN the request is forwarded to `default_on_failure`
- AND the log row's `classifier_used = null`

### Requirement: Policy lookup with urgency

After classification produces `(domain, complexity)`, the router SHALL look up the destination backend in `[routing.policy.<urgency>]` keyed by `"<domain>,<complexity>"`. The effective urgency is: per-message prefix override if present, else `[routing].default_urgency`.

#### Scenario: Policy lookup uses effective urgency
- GIVEN classifier output `(code, medium)`, no prefix urgency override, `default_urgency = "normal"`, and `[routing.policy.normal] "code,medium" = "local-coder"`
- WHEN policy lookup runs
- THEN the chosen backend is `local-coder`

#### Scenario: Per-message urgency prefix overrides default
- GIVEN classifier output `(general, medium)`, `default_urgency = "normal"`, prefix `!urgent`, and `[routing.policy.urgent] "general,medium" = "cloud-small"`
- WHEN policy lookup runs
- THEN the chosen backend is `cloud-small`

### Requirement: Policy table completeness

The config validator SHALL require all 18 cells of the policy tables (`{normal, urgent, patient} × {code, general} × {trivial, medium, hard}`) to be defined and to reference a backend defined under `[backends.*]`.

#### Scenario: Missing cell rejected at startup
- GIVEN `[routing.policy.urgent]` missing the `"general,hard"` entry
- WHEN `goorouter serve` is invoked
- THEN startup fails with a clear error naming the missing cell
- AND exit code is 2

#### Scenario: Cell references undefined backend rejected
- GIVEN `[routing.policy.normal] "code,hard" = "nonexistent"` and no backend named `nonexistent`
- WHEN `goorouter serve` is invoked
- THEN startup fails with a clear error naming the offending cell and the missing backend

### Requirement: Empty messages and multimodal content

If the request's `messages` array is empty or contains no message with `role = "user"`, the router SHALL treat the request as `(domain=general, complexity=trivial)` with default urgency. If any user-role message contains multimodal content (e.g., images), the router SHALL forward the request to `default_on_failure` and SHALL emit a stdout warning identifying the request id.

#### Scenario: Empty messages
- GIVEN `messages = []`
- WHEN the request is processed
- THEN the router uses `(general, trivial)` with `default_urgency`
- AND classifier is NOT called

#### Scenario: Multimodal content routed to default_on_failure
- GIVEN a user message contains an image content part
- WHEN the request is processed
- THEN the router forwards to `default_on_failure`
- AND a stdout warning is emitted naming the request id
- AND the log row records the request as routed to `default_on_failure`
