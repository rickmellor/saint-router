# proxy — Delta Spec

## ADDED Requirements

### Requirement: Localhost-only HTTP server by default

The proxy SHALL bind to `127.0.0.1` on a configurable port (default `4000`) and SHALL only bind to a non-loopback address when explicitly configured. Any non-loopback bind SHALL emit a stdout warning at startup.

#### Scenario: Default bind
- GIVEN a config with no `[server]` section, or `[server.host]` unset
- WHEN `goorouter serve` is invoked
- THEN the proxy listens on `127.0.0.1:4000`
- AND no warning is emitted

#### Scenario: Non-loopback bind requires explicit config
- GIVEN `[server] host = "0.0.0.0"` in config
- WHEN `goorouter serve` is invoked
- THEN the proxy listens on `0.0.0.0:4000`
- AND a stdout warning is emitted naming the bound address

### Requirement: OpenAI-compatible chat completions

The proxy SHALL accept POST requests at `/v1/chat/completions` with the OpenAI Chat Completions request body shape and SHALL return responses in the matching OpenAI shape (both streaming and non-streaming).

#### Scenario: Non-streaming round-trip
- GIVEN a valid request with `model = "goo-auto"`, `messages = [...]`, `stream = false`
- WHEN the request is processed and the chosen backend returns a completion
- THEN the response body matches OpenAI's non-streaming shape (`choices`, `usage`, etc.)
- AND the HTTP status is 200

#### Scenario: Streaming round-trip
- GIVEN a valid request with `stream = true`
- WHEN the chosen backend streams chunks
- THEN the proxy forwards each chunk as a Server-Sent Event (`data: <json>\n\n`)
- AND the stream ends with `data: [DONE]\n\n`

### Requirement: Models listing

The proxy SHALL expose `/v1/models` returning `goo-auto`, `goo-explain`, and one `goo-<backend>` entry for each backend defined in `[backends]`.

#### Scenario: Listing virtual + configured backends
- GIVEN a config with three backends `cloud-large`, `cloud-small`, `local-coder`
- WHEN a client GETs `/v1/models`
- THEN the response contains exactly five entries: `goo-auto`, `goo-explain`, `goo-cloud-large`, `goo-cloud-small`, `goo-local-coder`

### Requirement: Tool / function calling pass-through

The proxy SHALL forward `tools` and `tool_choice` parameters from the client request to the destination backend without modification, and SHALL return tool-call responses to the client without modification.

#### Scenario: Tools forwarded
- GIVEN a request body with a non-empty `tools` array and `tool_choice = "auto"`
- WHEN the request is forwarded to the chosen backend
- THEN the backend receives `tools` and `tool_choice` exactly as the client sent them

#### Scenario: Tool-call response forwarded
- GIVEN the backend response includes a `tool_calls` field on a message
- WHEN the proxy returns the response to the client
- THEN `tool_calls` is present in the response body unmodified

### Requirement: Single-process port binding

The proxy SHALL exit with a clear, friendly error when the configured port is already in use.

#### Scenario: Port already in use
- GIVEN port 4000 is bound by another process
- WHEN `goorouter serve` is invoked
- THEN `goorouter` exits with code 3
- AND stderr contains a message naming the port and suggesting another `goorouter serve` may be running

### Requirement: SSE forwarding without buffering

The proxy SHALL forward streaming chunks from the destination backend to the client without buffering complete responses in memory.

#### Scenario: Chunks arrive as backend produces them
- GIVEN a streaming request and a slow-streaming backend
- WHEN the backend emits chunk N
- THEN the client receives chunk N before chunk N+1 is generated
- AND the proxy does not hold all chunks in memory before any are sent
