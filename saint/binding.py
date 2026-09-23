"""Resolve the effective backend for a dispatch (the johnny override overlay).

Static config is the baseline; a johnny binding is an override that only ever applies
when johnny is enabled, reachable, and the seat is `ready`. Otherwise the fallback order
honors "static is the default" at every step:
    while_loading (per-backend → global) → own static baseline (unless johnny_only) → default_on_failure
`backend_chosen` (logged elsewhere) stays the *intended* backend; `state_at_dispatch`
records what actually served.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace

from saint.config import BackendConfig, Config
from saint.drain import is_draining
from saint.johnny import JohnnyResolver, seat_load


@dataclass(frozen=True)
class Effective:
    backend: BackendConfig          # the backend config call_backend should use
    state_at_dispatch: str | None   # johnny_ready | static_baseline | while_loading | fallback | None
    johnny_seat: str | None
    spilled_from: str | None = None  # the role whose own seat was saturated/not ready/draining when another role's seat served
    drained: str | None = None       # the draining seat this request was steered away from
    seat_load: int | None = None     # requests in flight on the serving seat at dispatch (None = unknown)


def _spill(cfg: Config, b: BackendConfig, resolver: JohnnyResolver, own_seat: str | None, own_load: int | None):
    """The least-loaded READY seat among b.spill's roles, excluding the backend's own seat.
    Returns (Resolution, load) or None. Unknown loads sort last so a readable seat wins."""
    best = None
    for role in b.spill:
        r = resolver.resolve(role)
        if r is None or r.state != "ready" or not r.endpoint or not r.model or r.seat == own_seat:
            continue
        if is_draining(r.seat, r.endpoint, role):
            continue
        load = seat_load(r.endpoint)
        key = (load if load is not None else 10**6, b.spill.index(role))
        if best is None or key < best[0]:
            best = (key, r, load)
    if best is None:
        return None
    _, r, load = best
    if own_load is not None and load is not None and load >= own_load:
        return None                     # nowhere better to go
    return r, load


def _fallback(cfg: Config, b: BackendConfig, seat: str | None) -> Effective:
    wl = b.while_loading or cfg.routing.while_loading
    if wl and wl in cfg.backends:
        return Effective(cfg.backends[wl], "while_loading", seat)
    if not b.johnny_only and b.base_url and b.model:
        return Effective(b, "static_baseline", seat)
    return Effective(cfg.backends[cfg.routing.default_on_failure], "fallback", seat)


def resolve_for_dispatch(cfg: Config, backend_name: str, resolver: JohnnyResolver | None) -> Effective:
    b = cfg.backends[backend_name]
    # Unbound backend (or johnny disabled): pure static, no johnny call ever.
    if not b.johnny_bound or resolver is None or cfg.johnny is None:
        return Effective(b, None, None)

    target = b.johnny_target
    res = resolver.resolve(target)
    if res is None:  # johnny unreachable → degrade to the static baseline (don't reroute)
        print(f"[saint] johnny unreachable resolving '{target}' (backend '{backend_name}') — "
              f"using static baseline", file=sys.stderr)
        # Unlike loading/absent (a seat is warming up — serve from while_loading meanwhile),
        # unreachable says nothing about the seat: the profile's fixed port is still the best
        # guess, so keep the request local instead of rerouting via while_loading (often cloud).
        if not b.johnny_only and b.base_url and b.model:
            return Effective(b, "static_baseline", None)
        return _fallback(cfg, b, None)

    if res.state == "ready" and res.endpoint and res.model:
        # Override the static endpoint/model with johnny's live values. johnny seats are
        # OpenAI-compatible, so default the provider to openai for johnny_only backends.
        if is_draining(res.seat, res.endpoint, target):
            # `saint drain <seat>`: the seat stays up for its in-flight work but takes nothing new — a ready spill seat
            # serves instead, else the normal fallback. Never the draining seat, whatever its load.
            alt = _spill(cfg, b, resolver, res.seat, None) if b.spill else None
            if alt is not None:
                r2, load2 = alt
                print(f"[saint] drain {res.seat}: {target}→{r2.seat} (load {load2})", file=sys.stderr)
                eff = replace(b, base_url=r2.endpoint, model=r2.model, provider=b.provider or "openai")
                return Effective(eff, "johnny_ready", r2.seat, spilled_from=target, seat_load=load2, drained=res.seat)
            print(f"[saint] drain {res.seat}: no spill seat for '{target}' — fallback", file=sys.stderr)
            fb = _fallback(cfg, b, res.seat)
            return replace(fb, drained=res.seat)
        load = seat_load(res.endpoint) if b.spill else None
        if b.spill and load is not None and load >= b.spill_at:
            alt = _spill(cfg, b, resolver, res.seat, load)
            if alt is not None:
                r2, load2 = alt
                print(f"[saint] spill {target}→{r2.seat} (load {load}→{load2})", file=sys.stderr)
                eff = replace(b, base_url=r2.endpoint, model=r2.model, provider=b.provider or "openai")
                return Effective(eff, "johnny_ready", r2.seat, spilled_from=target, seat_load=load2)
        eff = replace(b, base_url=res.endpoint, model=res.model, provider=b.provider or "openai")
        return Effective(eff, "johnny_ready", res.seat, seat_load=load)

    # loading / absent / failed: never block on a load; trigger one (if allowed) and serve via fallback —
    # but a ready spill seat beats the fallback (usually cloud).
    if res.state in ("loading", "absent") and cfg.johnny.ensure_load:
        resolver.ensure_load(target)
    if b.spill:
        alt = _spill(cfg, b, resolver, res.seat, None)
        if alt is not None:
            r2, load2 = alt
            print(f"[saint] spill {target}({res.state})→{r2.seat} (load {load2})", file=sys.stderr)
            eff = replace(b, base_url=r2.endpoint, model=r2.model, provider=b.provider or "openai")
            return Effective(eff, "johnny_ready", r2.seat, spilled_from=target, seat_load=load2)
    return _fallback(cfg, b, res.seat)


def describe_for_explain(cfg: Config, backend_name: str, resolver: JohnnyResolver | None) -> dict | None:
    """Read-only resolution for `explain` (NEVER triggers a load). None if unbound."""
    b = cfg.backends[backend_name]
    if not b.johnny_bound or resolver is None or cfg.johnny is None:
        return None
    target = b.johnny_target
    res = resolver.resolve(target)
    if res is None:
        eff = _fallback(cfg, b, None)
        return {"target": target, "state": "unreachable", "seat": None, "eta_s": None,
                "served_by": eff.backend.name, "state_at_dispatch": eff.state_at_dispatch}
    if res.state == "ready" and res.endpoint and res.model:
        return {"target": target, "state": "ready", "seat": res.seat, "endpoint": res.endpoint,
                "model": res.model, "eta_s": res.eta_s, "served_by": res.seat,
                "state_at_dispatch": "johnny_ready", "static_baseline": b.base_url}
    eff = _fallback(cfg, b, res.seat)
    return {"target": target, "state": res.state, "seat": res.seat, "eta_s": res.eta_s,
            "served_by": eff.backend.name, "state_at_dispatch": eff.state_at_dispatch}
