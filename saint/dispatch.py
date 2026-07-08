"""Shared dispatch orchestration: candidate walk + retry + circuit breaker.

Both API surfaces (/v1/chat/completions and /v1/messages) route through
``run_candidates``: resolve each candidate through the johnny binding overlay, try it
(with one optional same-backend retry), record breaker outcomes, and fall through to the
backend's one-hop ``on_error`` fallback. The per-endpoint difference — how a single
attempt is made and what "success" returns — lives in the ``attempt`` closure, which for
streaming endpoints must acquire the stream AND its first chunk (failover is only
possible before anything is flushed to the client).
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from saint.binding import Effective, resolve_for_dispatch
from saint.config import Config
from saint.route_cache import Breaker
from saint.sso import ensure_sso_valid, is_auth_failure, spawn_sso_login


@dataclass
class Served:
    backend: str      # candidate name that actually served
    eff: Effective    # its resolved effective backend (johnny overlay applied)
    value: Any        # whatever `attempt` returned (response | (upstream, first_chunk))


@dataclass
class BedrockRuntime:
    """Per-process SSO recovery state (held on app.state)."""

    spawned_profiles: set[str] = field(default_factory=set)  # one browser tab per expiry event


def dispatch_candidates(cfg: Config, primary: str, breaker: Breaker) -> list[str]:
    """Primary plus its one-hop on_error fallback. The fallback's own on_error is never
    followed (loop-proof). An open breaker skips the primary when a fallback exists."""
    fb = cfg.backends[primary].on_error
    candidates = [primary] + ([fb] if fb and fb != primary else [])
    if len(candidates) > 1 and breaker.is_open(primary):
        print(f"[router] breaker open on '{primary}' — dispatching straight to '{fb}'",
              flush=True)
        return candidates[1:]
    return candidates


async def run_candidates(
    *,
    cfg: Config,
    candidates: list[str],
    rid: str,
    resolver,
    breaker: Breaker,
    attempt: Callable[[Effective], Awaitable[Any]],
    bedrock_state: BedrockRuntime | None = None,
) -> tuple[Served | None, str | None, Effective | None]:
    """Try candidates in order; return (served, last_error_kind, last_effective).

    `attempt(eff)` performs one dispatch try and raises on failure. Success anywhere
    short-circuits. `last_effective` is returned for logging fidelity when everything
    failed (the log row records the last backend we tried).

    Bedrock auth handling: a classified credential failure force-opens the breaker with
    the (longer) auth cooldown, skips the pointless same-backend retry, and spawns the
    SSO browser login once per expiry event. A half-open trial on an auth-flagged bedrock
    candidate is preceded by a non-interactive `ensure_sso_valid` probe so a wedged SSO
    session never burns a real request."""
    error_kind: str | None = None
    eff: Effective | None = None
    for cand in candidates:
        backend_cfg = cfg.backends[cand]
        is_bedrock = backend_cfg.provider == "bedrock"
        sso_cfg = cfg.bedrock if (is_bedrock and cfg.bedrock
                                  and cfg.bedrock.credential_process) else None
        profile = backend_cfg.aws_profile or "default"

        # Half-open trial after an AUTH cooldown: probe credentials first (free).
        if sso_cfg and breaker.is_auth_flagged(cand):
            ok = await ensure_sso_valid(sso_cfg.credential_process, profile,
                                        sso_cfg.sso_check_timeout_s)
            if not ok:
                breaker.open_now(cand, sso_cfg.auth_cooldown_s)  # re-arm, no request burned
                print(f"[router] req#{rid}: SSO still invalid for '{cand}' "
                      f"(profile {profile}) — skipping", file=sys.stderr, flush=True)
                continue
            if bedrock_state is not None:
                bedrock_state.spawned_profiles.discard(profile)  # next expiry may re-prompt

        eff = resolve_for_dispatch(cfg, cand, resolver)
        attempts = 2 if cfg.routing.retry_same_backend else 1
        for i in range(attempts):
            try:
                value = await attempt(eff)
                breaker.record_success(cand)
                return Served(backend=cand, eff=eff, value=value), None, eff
            except Exception as e:
                error_kind = type(e).__name__
                if is_bedrock and cfg.bedrock is not None and is_auth_failure(e):
                    breaker.open_now(cand, cfg.bedrock.auth_cooldown_s)
                    print(f"[router] req#{rid}: AUTH failure on '{cand}' ({error_kind}) — "
                          f"circuit open {cfg.bedrock.auth_cooldown_s:.0f}s; "
                          f"SSO re-auth may be required (profile {profile})",
                          file=sys.stderr, flush=True)
                    if (sso_cfg and cfg.bedrock.spawn_sso_login
                            and bedrock_state is not None
                            and profile not in bedrock_state.spawned_profiles):
                        bedrock_state.spawned_profiles.add(profile)
                        asyncio.create_task(
                            spawn_sso_login(sso_cfg.credential_process, profile))
                    break  # same-backend retry is pointless until creds refresh
                breaker.record_failure(cand)
                print(f"[router] req#{rid}: dispatch to '{cand}' failed "
                      f"({error_kind}) — {'retrying' if i == 0 and attempts == 2 else 'moving on'}",
                      file=sys.stderr, flush=True)
    return None, error_kind, eff
