# SAINT on a corporate AWS Bedrock machine

SAINT routes Claude Code and every OpenAI-compatible agent client (pi, opencode, hermes)
through one local endpoint, dispatching to AWS Bedrock Claude tiers with routing, failover,
prompt caching, and full cost/net-savings accounting — under corporate SSO, with no static
keys anywhere.

## How auth works (and why SAINT needs the credential patch)

Corporate Bedrock access chains **Azure AD (Entra) → Cognito identity pool → STS temp
credentials (~1h)**, minted by a `credential_process` helper (e.g.
`~/claude-code-with-bedrock/credential-process`) wired into an AWS profile (`ClaudeCode`)
in `~/.aws/config`. Credentials are keyring-cached; there are no long-lived keys.

The trap: Claude Code also writes *static* temp creds to `~/.aws/credentials` as a cache,
and boto3's default chain prefers that file over `credential_process` — without tracking
its ~1h expiry, so calls silently start 403ing. SAINT applies a credential patch (only
when a `provider = "bedrock"` backend exists) that removes the `shared-credentials-file`
provider, forcing resolution through `credential_process`, whose `Expiration` yields
auto-refreshing `RefreshableCredentials`. You'll see `[saint] bedrock credential patch
applied` in the journal at startup.

## Setup

1. **Copy the template**: `cp docs/examples/corp-bedrock.toml ~/.config/saint/config.toml`,
   then verify the model ids and prices against your account and the Bedrock pricing page.
   Confirm your inference profiles: `aws bedrock list-inference-profiles --region us-east-1`.
2. **Install with the bedrock extra**: `uv tool install 'saint-router[bedrock]'` (pulls boto3).
3. **First SSO login must be interactive** — run it once so the keyring is populated:
   `aws sts get-caller-identity --profile ClaudeCode` (opens the browser flow if stale).
4. **systemd user service** — the keyring lives in the desktop session's D-Bus, so the
   service must run inside a logged-in session:

   ```ini
   # ~/.config/systemd/user/saint.service
   [Unit]
   Description=SAINT LLM router
   After=network.target
   [Service]
   Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin
   # keyring access: the credential_process needs the session bus + display
   PassEnvironment=DISPLAY WAYLAND_DISPLAY XAUTHORITY DBUS_SESSION_BUS_ADDRESS
   ExecStart=%h/.local/bin/saint serve
   Restart=on-failure
   [Install]
   WantedBy=default.target
   ```
   `loginctl enable-linger $USER`, then `systemctl --user enable --now saint`. A service
   started outside a desktop session cannot unlock the keyring — auth will fail loudly
   (see recovery below) until you log in interactively.

## Point Claude Code at SAINT

Edit `~/.claude/settings.json` `env`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:4000",
    "ANTHROPIC_AUTH_TOKEN": "saint-local",
    "ANTHROPIC_MODEL": "fable",
    "ANTHROPIC_SMALL_FAST_MODEL": "haiku"
  }
}
```

- **Remove `CLAUDE_CODE_USE_BEDROCK` and `AWS_PROFILE`.** With `USE_BEDROCK` set, Claude
  Code SigV4-signs straight to AWS and never touches SAINT — the whole setup silently
  does nothing.
- `ANTHROPIC_AUTH_TOKEN` is a required placeholder (Claude Code needs *a* token to skip
  its OAuth path); SAINT binds loopback and ignores it.
- Model vars can be tier aliases (`fable`, `sonnet`, `haiku`) or `auto` to let the
  classifier pick. Because the template aliases the raw inference-profile ids too, an
  unmodified settings.json (`ANTHROPIC_MODEL=global.anthropic.claude-opus-4-8`) already
  works — migrate the names at your leisure.
- **Keep the OTEL block as-is** — Claude Code exports telemetry directly to your collector;
  routing through SAINT doesn't disturb cost-center attribution.

## Verification (in order)

1. `aws sts get-caller-identity --profile ClaudeCode` — proves credential_process + keyring.
2. `systemctl --user start saint`; `journalctl --user -u saint | grep 'credential patch'`.
3. `saint explain "hello"` — routing sanity, no dispatch.
4. Pinned chat: `curl -si localhost:4000/v1/chat/completions -d '{"model":"saint-bedrock-haiku","messages":[{"role":"user","content":"hi"}]}'` — check `x-saint-*` headers.
5. Pinned `/v1/messages`, non-stream then `"stream":true`:
   `curl -si localhost:4000/v1/messages -d '{"model":"haiku","max_tokens":50,"messages":[{"role":"user","content":"hi"}]}'`.
6. Point Claude Code at SAINT; run a session with a tool loop — confirm streaming + tools.
7. Repeat a long-context request; `saint log stats --days 1` shows cache-read tokens and
   the net-savings line.
8. **Auth-failure drill**: let creds expire (or `keyring` delete the cached entry), send a
   request → journal shows one `AUTH failure … SSO re-auth may be required` line and one
   browser tab; after you re-auth, the next request (past `auth_cooldown_s`) recovers with
   no restart.

## Probes to run once on the corp box (they gate nothing in code, but confirm assumptions)

- **A — /v1/messages against Bedrock**: pinned non-stream + stream to a bedrock tier return
  200 and Anthropic-shaped body/SSE.
- **B — /v1/messages against a local seat** (if you add one): a tool round-trip through the
  openai translation path. If it misbehaves, restrict `/v1/messages` routing to bedrock
  tiers (every policy cell is bedrock here, so nothing is lost).
- **C — Bedrock prompt caching**: a >4k-char system prompt written once then read on the
  repeat shows `cache_write_tokens` then `cache_read_tokens` in `saint log id`. If the
  usage field names differ, `_cache_tokens` already falls back to the Converse names.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Claude Code ignores SAINT | `CLAUDE_CODE_USE_BEDROCK` still set — remove it |
| `ExpiredTokenException` / 403 loop | re-auth: `aws sts get-caller-identity --profile ClaudeCode`; SAINT recovers within `auth_cooldown_s` |
| Auth fails only under systemd | service started outside a desktop session — keyring unreachable; log in interactively, `enable-linger` |
| `RuntimeError: ...BaseAWSLLM no longer has _auth_with_aws_profile` | litellm upgraded and moved the private auth API — re-verify `saint/bedrock_auth.py` against the new version |
| Wrong tier / model not found | model id must be a valid inference profile (`aws bedrock list-inference-profiles`); alias must match the client's model string |
| Cert errors on corp network | set `AWS_CA_BUNDLE` / the profile's `ca_bundle` to the corp CA (`/etc/ssl/certs/ca-certificates.crt`) |
