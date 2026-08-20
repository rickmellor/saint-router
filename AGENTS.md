# AGENTS.md — working in this repo

SAINT is a localhost OpenAI-compatible router (FastAPI, `saint/`, port 4000):
each request is classified (domain × complexity) and routed between local
johnny seats and cloud backends (Anthropic, Bedrock). Also serves Anthropic
`/v1/messages`. `docs/corp-deployment.md` covers the Bedrock/corp variant.

## Live editable install + service

- `saint` on PATH is a **pipx editable install of this repo** — code edits are
  live on next process start, no reinstall.
- Runs as the systemd user service `saint.service` (`saint serve`):

      systemctl --user restart saint      # required after config OR code changes
      journalctl --user -u saint          # router decisions + errors

## Runtime config (not in the repo)

- `~/.config/saint/config.toml` — the real config (`saint/config.example.toml`
  is the template). Secrets in `~/.config/saint/env` (systemd EnvironmentFile).
- Request log: `~/.config/saint/log.sqlite`; trained embedding-classifier head:
  `~/.config/saint/classifier_head.npz`.
- `[server].host` stays `127.0.0.1` — every known client is local.

## johnny binding

Backends with `johnny_role` (`coder`/`chat`/`embed`/`classifier`) resolve the
live seat via the `johnny` CLI (results cached ~5 s); the static
`base_url`/`model` in config is the fallback when johnny is absent or the seat
is down. Under johnny profile `coder`, role `coder` maps to the chat seat
(Ornith) via role_aliases — no seat literally holds the coder role, and that's
correct. `while_loading` routes to cloud while a seat warms up.

## Classifier

`mode = "embedding"`: nomic embeddings + trained head, ~ms per request.
Low-confidence defers to `backend` (haiku — its labels also feed retraining);
`fallback_backend` (local 1B) is the offline path. Retrain with
`saint classifier train`.

## Tests

    ~/.local/share/pipx/venvs/saint-router/bin/python -m pytest tests/ -q

2 tests in `test_sso.py` need the `[bedrock]` extra (boto3) and fail with
ModuleNotFoundError without it — expected locally, not a regression.

## Request-routing details worth knowing

- `!alias` prefix in the latest user message forces a backend (e.g. `!opus`).
- `saint-auto` / `saint-explain` / `saint-<backend>` are the exposed model ids.
- `on_error` is one-hop (no chains); `default_on_failure` is the last resort.
- **Volatile sentinel** (`cache.volatile_sentinel`, default `<<<saint:volatile>>>`): a
  client puts non-cacheable per-turn context (live clock, git branch, active file) after
  this marker in its FIRST system message. `_prepare_dispatch` strips the marker and
  relocates the tail to a trailing block on the last user message — after every cache
  breakpoint — so it reaches the model without invalidating the cached prefix or skewing
  the classifier. Helpers `split_volatile`/`append_volatile` in `saint/backends.py`. The
  relocated text becomes user-role; phrase it as context, not a system directive.
