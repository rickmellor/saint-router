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

from saint.binding import Effective, is_pool, pool_order_with_reasons, resolve_for_dispatch
from saint.capabilities import missing
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


def dispatch_candidates(cfg: Config, primary: str, breaker: Breaker,
                        required: frozenset[str] = frozenset(),
                        pinned: bool = False) -> list[str]:
    """Primary plus its one-hop on_error fallback. The fallback's own on_error is never
    followed (loop-proof). An open breaker skips the primary when a fallback exists.

    A turn carrying non-text content also queues every OTHER backend that could plausibly
    serve it (live johnny caps are decided at resolve time, so any bound backend is a
    candidate; otherwise the static declaration must cover `required`) — otherwise an image
    routed to a text-only seat would 400 instead of reaching the vision seat. A pin is a pin:
    the user named the backend, so we keep the short list and let it fail loudly.

    `local_only` is the hard version of that: the turn may not leave this backend at all, so
    there is nothing else to name — no on_error hop, no capability widening, and an open
    breaker is moot (a pool's own members are its failover). A 503 the client retries is the
    honest outcome when no local member is ready."""
    if getattr(cfg.backends[primary], "local_only", False):
        return [primary]
    fb = cfg.backends[primary].on_error
    candidates = [primary] + ([fb] if fb and fb != primary else [])
    if len(candidates) > 1 and breaker.is_open(primary):
        print(f"[router] breaker open on '{primary}' — dispatching straight to '{fb}'",
              flush=True)
        candidates = candidates[1:]
    if required and not pinned:
        bound, static = [], []
        for name, b in cfg.backends.items():
            if name in candidates:
                continue
            if b.johnny_bound:
                bound.append(name)          # live seat: resolve decides capability
            elif required <= b.static_capabilities:
                static.append(name)
        candidates += bound + static        # a live seat beats a declared one; no duplicates
    return candidates


BILLING_MARKERS = ("credit balance", "billing", "payment required", "insufficient credit", "insufficient fund",
                   "purchase credits", "invalid x-api-key", "authentication_error", "invalid_api_key")


def is_billing_failure(detail: str, status=None) -> bool:
    """A provider refusal that means 'pay or fix credentials' (Anthropic 400 billing, 401/402/403 auth) — never a
    backend health problem, so the router should say so instead of 502."""
    d = (detail or "").lower()
    return status in (401, 402) or any(m in d for m in BILLING_MARKERS)


def _pool_unavailable(pool: str, cfg: Config, reasons: dict[str, str]) -> str:
    """'No local seat is ready' as one line naming every member's problem — the client (and
    `saint log`) needs to see WHICH member and WHY, not just that the pool was empty."""
    members = list(getattr(cfg.backends[pool], "pool", ()) or ())
    listed = [m for m in members if m in reasons] + [m for m in reasons if m not in members]
    return (f"PoolUnavailable: no ready member in {pool} ("
            + "; ".join(f"{m}: {reasons[m]}" for m in listed) + ")")


async def _run_pool(cfg: Config, pool: str, resolver, breaker: Breaker,
                    attempt: Callable[[Effective], Awaitable[Any]], rid: str,
                    required: frozenset[str], conversation_key: str | None,
                    caches, size: tuple[int, int] | None = None
                    ) -> tuple[Served | None, str | None, Effective | None]:
    """Walk a pool once, best member first. Each member gets exactly ONE attempt — the next
    member IS the retry, so a member that just failed is never hammered twice behind a
    same-backend retry.

    The breaker tracks MEMBERS, never the pool name: one dead seat must not blackhole the
    healthy ones, and the pool has no endpoint of its own to be unhealthy. Same 400/422 and
    billing classification as the normal path (a malformed request is not a seat failure)."""
    order, reasons = pool_order_with_reasons(cfg, pool, resolver, required,
                                             conversation_key, caches, size=size)
    error_kind: str | None = None
    last_eff: Effective | None = None
    for eff in order:
        member = eff.pool_member
        if breaker.is_open(member):
            reasons[member] = "breaker open"
            print(f"[router] req#{rid}: pool '{pool}' member '{member}' circuit open — "
                  f"next member", file=sys.stderr, flush=True)
            continue
        last_eff = eff
        try:
            value = await attempt(eff)
        except Exception as e:
            error_kind = type(e).__name__
            detail = " ".join(str(e).split())[:300]
            status = getattr(e, "status_code", None)
            if is_billing_failure(detail, status):
                error_kind = "BillingError: " + detail[:200]
            if not (status in (400, 422) or "BadRequest" in error_kind
                    or "invalid_request_error" in detail):
                breaker.record_failure(member)
            print(f"[router] req#{rid}: pool '{pool}' member '{member}' failed "
                  f"({error_kind}: {detail}) — next member", file=sys.stderr, flush=True)
            continue
        breaker.record_success(member)
        return Served(backend=pool, eff=eff, value=value), None, eff
    if last_eff is None:                    # nothing was even attempted
        return None, _pool_unavailable(pool, cfg, reasons), None
    return None, error_kind, last_eff


async def run_candidates(
    *,
    cfg: Config,
    candidates: list[str],
    rid: str,
    resolver,
    breaker: Breaker,
    attempt: Callable[[Effective], Awaitable[Any]],
    bedrock_state: BedrockRuntime | None = None,
    required: frozenset[str] = frozenset(),
    conversation_key: str | None = None,
    caches=None,
    size: tuple[int, int] | None = None,
) -> tuple[Served | None, str | None, Effective | None]:
    """Try candidates in order; return (served, last_error_kind, last_effective).

    `attempt(eff)` performs one dispatch try and raises on failure. Success anywhere
    short-circuits. `last_effective` is returned for logging fidelity when everything
    failed (the log row records the last backend we tried).

    A candidate whose resolved seat cannot serve `required` is skipped WITHOUT attempting it
    and without touching the breaker — the request is malformed for that backend, not the
    backend unhealthy, and burning it would blackhole a live text-only seat on every image.
    If nothing could be attempted at all, the failure is a CapabilityError (an unfulfillable
    request → 400 upstream, not a 502).

    A candidate that is a POOL expands into its members (see `_run_pool`); `conversation_key`
    and `caches` are what a sticky pool uses to keep a conversation on the member that served
    it, and are ignored by every other candidate.

    `size` = (estimated prompt tokens, requested output tokens) from saint/sizing.py, threaded to
    the spill decision so an oversized turn is not handed to a seat that cannot hold it (or would
    cold-prefill it for a minute). None = no size information, every spill candidate stays eligible.

    Bedrock auth handling: a classified credential failure force-opens the breaker with
    the (longer) auth cooldown, skips the pointless same-backend retry, and spawns the
    SSO browser login once per expiry event. A half-open trial on an auth-flagged bedrock
    candidate is preceded by a non-interactive `ensure_sso_valid` probe so a wedged SSO
    session never burns a real request."""
    error_kind: str | None = None
    eff: Effective | None = None
    skipped: list[tuple[str, str, frozenset[str]]] = []   # (backend, model, caps it lacks)
    attempted = False
    pool_unavailable = False
    for cand in candidates:
        backend_cfg = cfg.backends[cand]
        if is_pool(backend_cfg):
            # A pool is not a backend you can call: it expands to its members here, and the
            # member walk owns its own capability filtering, breaker and error text.
            served, pool_err, pool_eff = await _run_pool(
                cfg, cand, resolver, breaker, attempt, rid, required, conversation_key, caches,
                size)
            if served is not None:
                return served, None, served.eff
            if pool_err is not None:
                error_kind = pool_err
                pool_unavailable = pool_err.startswith("PoolUnavailable")
            if pool_eff is not None:        # a member WAS tried: a real backend error, not a
                attempted = True            # pool-availability one
                eff = pool_eff
            continue
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

        eff = resolve_for_dispatch(cfg, cand, resolver, required, size=size)
        miss = missing(required, eff.capabilities)
        if miss:
            print(f"[router] req#{rid}: skipping '{cand}' ({eff.backend.model}): "
                  f"lacks {','.join(sorted(miss))}", file=sys.stderr, flush=True)
            skipped.append((cand, eff.backend.model, miss))
            continue
        attempts = 2 if cfg.routing.retry_same_backend else 1
        attempted = True
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
                detail = " ".join(str(e).split())[:300]   # the class name alone hid a day of 400s
                status = getattr(e, "status_code", None)
                if is_billing_failure(detail, status):
                    # provider says pay/authenticate: no candidate, retry or breaker will help — surface it as
                    # BillingError:<text> so the endpoint answers 402 instead of a generic 502 (2026-09-23:
                    # "credit balance too low" hid behind 502 for an hour and failed a build job)
                    error_kind = "BillingError: " + detail[:200]
                if status in (400, 422) or "BadRequest" in error_kind or "invalid_request_error" in detail:
                    # the REQUEST was malformed for this backend (shape/param), not the backend
                    # unhealthy — counting it opened the breaker and blackholed a live seat for
                    # 60 s on every client hiccup (2026-08-27). Still move on to the next candidate.
                    pass
                else:
                    breaker.record_failure(cand)
                print(f"[router] req#{rid}: dispatch to '{cand}' failed "
                      f"({error_kind}: {detail}) — {'retrying' if i == 0 and attempts == 2 else 'moving on'}",
                      file=sys.stderr, flush=True)
    if skipped and not attempted and not pool_unavailable:
        # nothing was even tryable: the turn needs a modality no candidate serves. That is a
        # request the deployment cannot answer (400), not a backend outage (502). A pool that
        # had no ready member already said so — keep that, so a warming seat reads as the
        # retryable 503 the client expects rather than a refusal it will not retry.
        error_kind = ("CapabilityError: no backend can serve "
                      + ",".join(sorted(required))
                      + " (" + "; ".join(f"{c}/{m} lacks {','.join(sorted(x))}"
                                         for c, m, x in skipped) + ")")
    return None, error_kind, eff
