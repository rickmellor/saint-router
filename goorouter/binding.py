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

from goorouter.config import BackendConfig, Config
from goorouter.johnny import JohnnyResolver


@dataclass(frozen=True)
class Effective:
    backend: BackendConfig          # the backend config call_backend should use
    state_at_dispatch: str | None   # johnny_ready | static_baseline | while_loading | fallback | None
    johnny_seat: str | None


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
        return _fallback(cfg, b, None)

    if res.state == "ready" and res.endpoint and res.model:
        # Override the static endpoint/model with johnny's live values. johnny seats are
        # OpenAI-compatible, so default the provider to openai for johnny_only backends.
        eff = replace(b, base_url=res.endpoint, model=res.model, provider=b.provider or "openai")
        return Effective(eff, "johnny_ready", res.seat)

    # loading / absent / failed: never block on a load; trigger one (if allowed) and serve via fallback.
    if res.state in ("loading", "absent") and cfg.johnny.ensure_load:
        resolver.ensure_load(target)
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
