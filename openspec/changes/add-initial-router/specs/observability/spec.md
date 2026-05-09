# observability — Delta Spec

## ADDED Requirements

### Requirement: Per-request log row

The router SHALL write exactly one row to the `requests` table in `[logging].db_path` per HTTP request handled (success or failure), capturing: timestamp, request id (uuid4), client `model` field, parsed prefixes, effective urgency, classifier metadata, chosen backend, latencies, token counts, success flag, error kind, and prompt content per `[logging].prompt_storage`. Logging SHALL be asynchronous and SHALL NOT block the response to the client.

#### Scenario: Successful auto-routed request produces a complete row
- GIVEN a successfully completed request that ran the full auto-routing pipeline
- WHEN the log row is written
- THEN the row's `urgency_used`, `classifier_used`, `classifier_domain`, `classifier_complexity`, `backend_chosen`, `success = 1`, and at least one of `tokens_in`/`tokens_out` are populated

#### Scenario: Failed request still produces a row
- GIVEN a request whose chosen backend returned HTTP 500
- WHEN the log row is written
- THEN `success = 0`, `error_kind = "http_5xx"`, and `backend_chosen` is the backend that failed

#### Scenario: Log write failure does not break the request
- GIVEN the SQLite write fails (disk full, locked, etc.)
- WHEN the request has otherwise succeeded
- THEN the client still receives the successful response
- AND a stderr message records the log failure

### Requirement: WAL mode for concurrent access

The storage layer SHALL enable SQLite's WAL journaling mode (`PRAGMA journal_mode=WAL`) on the first connection to the database. CLI read commands and the server's writes MUST be able to operate concurrently without "database is locked" errors at single-user volumes.

#### Scenario: Concurrent read while server writes
- GIVEN `goorouter serve` is running and actively writing log rows
- WHEN a separate process runs `goorouter log show`
- THEN `log show` returns recent rows without error
- AND the server's writes are not blocked

### Requirement: Prompt content storage modes

The router SHALL respect `[logging].prompt_storage` per request and SHALL record which mode was active in the row's `prompt_storage_mode` column. Modes are: `"full"` (store full prompt content), `"hashed"` (store SHA-256 hex of content), `"none"` (store nothing in `prompt_content`).

#### Scenario: Full storage
- GIVEN `prompt_storage = "full"`
- WHEN a request is logged
- THEN `prompt_content` contains the prompt content
- AND `prompt_storage_mode = "full"`

#### Scenario: Hashed storage
- GIVEN `prompt_storage = "hashed"`
- WHEN a request is logged
- THEN `prompt_content` is a 64-character lowercase hex SHA-256
- AND `prompt_storage_mode = "hashed"`

#### Scenario: None storage
- GIVEN `prompt_storage = "none"`
- WHEN a request is logged
- THEN `prompt_content` is `NULL`
- AND `prompt_storage_mode = "none"`

#### Scenario: Mode change preserves history
- GIVEN existing rows written with `prompt_storage_mode = "full"`
- WHEN config is changed to `prompt_storage = "hashed"` and new rows are written
- THEN existing rows still show `"full"` in their `prompt_storage_mode` column
- AND new rows show `"hashed"`

### Requirement: Relabel CLI

The `goorouter relabel` CLI SHALL update an existing row's `relabel_backend`, `relabel_ts`, and (optionally) `relabel_note` columns. `relabel last <backend>` SHALL target the row with the highest `id`. `relabel by-id <id> <backend>` SHALL target the explicit id. The CLI SHALL refuse to set `relabel_backend` to a name not currently defined under `[backends.*]`.

#### Scenario: relabel last
- GIVEN at least one row in the requests table
- WHEN `goorouter relabel last cloud-large` is invoked
- THEN the row with the highest `id` has `relabel_backend = "cloud-large"`, `relabel_ts` set to the current ISO8601 UTC timestamp, and `relabel_note = NULL`

#### Scenario: relabel by id with note
- GIVEN row id 42 exists
- WHEN `goorouter relabel by-id 42 local-coder --note "should have used coder"` is invoked
- THEN row 42's `relabel_backend = "local-coder"` and `relabel_note = "should have used coder"`

#### Scenario: Refuse undefined backend
- GIVEN no backend named `phantom` in current config
- WHEN `goorouter relabel last phantom` is invoked
- THEN the command exits non-zero with a clear error message
- AND no rows are modified

### Requirement: Explain mode (API)

When the request's `model` is `goo-explain`, the router SHALL execute the routing pipeline through policy lookup but SHALL NOT call the destination backend. The response SHALL be a chat completion whose content is a human-readable routing breakdown including: parsed prefixes, effective urgency, classifier backend used (if any), classifier latency, classifier output (`domain`, `complexity`, `reason`), policy lookup, and chosen backend.

#### Scenario: Explain returns breakdown without calling destination
- GIVEN `model = "goo-explain"` and a normal user message
- WHEN the request is processed
- THEN the destination backend named in the policy lookup is NOT called
- AND the response content includes (at minimum) the chosen backend name and the classifier output

### Requirement: Explain CLI

The `goorouter explain "<prompt>"` CLI SHALL produce a textual routing breakdown equivalent in content to the API explain mode for the same prompt and active config.

#### Scenario: CLI explain matches API explain
- GIVEN the same prompt text and active config
- WHEN `goorouter explain "<prompt>"` is run AND a separate API request with `model = "goo-explain"` is sent
- THEN both outputs name the same chosen backend
- AND both report the same classifier output (modulo non-deterministic LLM variation, which the test harness handles via recorded responses)

### Requirement: Log query CLI

`goorouter log show` SHALL print the most recent rows in human-readable form, supporting `--limit N` (default 20) and `--backend <name>` filters. `goorouter log id <ID>` SHALL print full detail for one row.

#### Scenario: log show with default limit
- GIVEN at least 50 rows in the table
- WHEN `goorouter log show` is run with no flags
- THEN exactly 20 rows are printed, ordered by descending `ts`

#### Scenario: log show with backend filter
- GIVEN rows with `backend_chosen` values `cloud-large`, `local-coder`, and `local-small`
- WHEN `goorouter log show --backend local-coder` is run
- THEN only rows with `backend_chosen = "local-coder"` are printed

#### Scenario: log id full detail
- GIVEN row id 42 exists
- WHEN `goorouter log id 42` is run
- THEN every column of row 42 is printed (with `prompt_content` truncated to a reasonable display length if `prompt_storage_mode = "full"`)

### Requirement: Stdout structured request lines

The running server SHALL emit one structured single-line summary to stdout per completed request, naming: request id, model field, effective urgency, classifier output (or "pinned"), chosen backend, classifier latency, and backend latency. Truncation and fallback events SHALL also emit one-line stdout markers.

#### Scenario: Successful auto-routed request produces stdout summary
- GIVEN a successful auto-routed request
- WHEN the response completes
- THEN exactly one stdout line for that request appears containing the request id, the chosen backend, the classifier output, and both latencies

#### Scenario: Fallback fires emits stdout marker
- GIVEN a request triggers oversize fallback
- WHEN the fallback fires
- THEN a stdout line names the request id, the trigger reason (`oversize` with input/limit), and the fallback backend
