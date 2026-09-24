"""Cost-savings report: cloud spend vs local energy cost vs what the router saved.

The counterfactual is "what if every request had gone to the cloud baseline backend?"
- Cloud requests are priced at their own $/Mtok (the money actually spent).
- Local requests cost *electricity*: host_watts x $/kWh / (tok/s x 3.6) per Mtok of
  output (the same model /status uses), where tok/s is the measured p75 decode rate.
- Savings = (what local traffic would have cost on the cloud baseline) - (its energy cost).

Everything is derived from the request log + [energy] config; nothing hardcoded and no
running server required.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

# period label -> lookback; None means "all time"
PERIODS: dict[str, timedelta | None] = {
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
    "week": timedelta(days=7),
    "month": timedelta(days=30),
    "year": timedelta(days=365),
    "all": None,
}


def since_for(period: str) -> str:
    """ISO-8601 UTC lower bound for a period label ('all' -> the epoch)."""
    if period not in PERIODS:
        raise ValueError(f"unknown period {period!r}; choose from {', '.join(PERIODS)}")
    delta = PERIODS[period]
    if delta is None:
        return datetime(1970, 1, 1, tzinfo=UTC).isoformat()
    return (datetime.now(UTC) - delta).isoformat()


def local_decode_rates(conn) -> dict[str, float]:
    """Measured p75 decode tok/s per local backend, over substantial responses
    (tokens_out>=128 amortizes TTFT). Mirrors server.py's /status computation so the
    report and the live seat list agree. Backends with <5 samples are omitted (no rate
    -> shown as un-costed rather than a fabricated number)."""
    by_bk: dict[str, list[float]] = {}
    try:
        for bk, tok, lat in conn.execute(
            "SELECT backend_chosen, tokens_out, backend_latency_ms FROM requests "
            "WHERE backend_chosen LIKE 'local%' AND tokens_out>=128 AND backend_latency_ms>0"
        ):
            by_bk.setdefault(bk, []).append(tok / (lat / 1000.0))
    except Exception:
        return {}
    out: dict[str, float] = {}
    for bk, rates in by_bk.items():
        if len(rates) >= 5:
            rates.sort()
            out[bk] = rates[min(len(rates) - 1, int(len(rates) * 0.75))]
    return out


@dataclass
class Row:
    backend: str
    kind: str            # "cloud" | "local" | "other"
    requests: int
    tokens_in: int
    tokens_out: int
    tok_s: float | None
    elec_per_mtok: float | None   # local only
    actual_cost: float            # $ actually spent: cache-aware Anthropic bill (cloud) or power (local)
    cloud_equiv: float | None     # what the baseline cloud backend would have charged (uncached)
    no_cache_cost: float | None = None   # cloud list-price ignoring cache (for the caching breakdown)
    watts: float | None = None           # local: seat draw the energy price is based on
    tok_per_w: float | None = None       # local: tok_s / watts
    source: str | None = None            # local: "bench" (config tok_s from johnny bench) | "log" (request-log p75)
    priced_as: str | None = None         # logged name resolved to another backend (rename / alias)


def _eff_cache_prices(b):
    """(cache_read, cache_write) $/Mtok: explicit config, else Anthropic's 0.1x/1.25x of
    price_in (matches saint log stats), else 0 for non-caching providers."""
    is_ant = getattr(b, "provider", None) in ("anthropic", "bedrock")
    cr = b.price_cache_read if b.price_cache_read is not None else (0.1 * b.price_in if is_ant else 0.0)
    cw = b.price_cache_write if b.price_cache_write is not None else (1.25 * b.price_in if is_ant else 0.0)
    return cr, cw


def is_cloud(b) -> bool:
    """A priced, non-local backend — the only kind that can be a savings counterfactual."""
    return bool(b is not None and b.price_in is not None and not b.johnny_bound and not b.name.startswith("local"))


def pick_baseline(cfg, explicit: str | None = None) -> tuple[str, str]:
    """The counterfactual cloud backend and how it was chosen. A local `default_on_failure`
    (the case since the CPU fallback seat was retired) must never be the baseline: it has no
    cloud price, so every dollar actually spent would show as OVERSPEND. Order: --baseline flag,
    [energy] baseline, default_on_failure if it is cloud, the routing policy's hard tier, the
    priciest cloud backend. Raises ValueError for an undefined name."""
    if explicit:
        name, how = explicit, "flag"
    elif cfg.energy.baseline:
        name, how = cfg.energy.baseline, "config"
    else:
        d = cfg.routing.default_on_failure
        if is_cloud(cfg.backends.get(d)):
            name, how = d, "default_on_failure"
        else:
            urg = getattr(cfg.routing.default_urgency, "value", cfg.routing.default_urgency)
            pol = cfg.routing.policy.get(urg) or next(iter(cfg.routing.policy.values()), {})
            hard = [pol.get(k) for k in ("code,hard", "general,hard")]
            hard = [h for h in hard if is_cloud(cfg.backends.get(h))]
            if hard:
                name, how = hard[0], "policy hard tier"
            else:
                clouds = sorted((b for b in cfg.backends.values() if is_cloud(b)), key=lambda b: -(b.price_in or 0))
                if clouds:
                    name, how = clouds[0].name, "priciest cloud"
                else:
                    name, how = d, "default_on_failure (unpriced)"
    if name not in cfg.backends:
        raise ValueError(f"baseline backend {name!r} is not defined")
    return name, how


def resolve_backend(cfg, logged: str):
    """(backend, priced_as): the config backend for a logged backend_chosen. Names that were
    renamed (cloud-flagship -> cloud-exquisite, 2026-09-23) resolve through aliases, so old rows
    keep a price instead of silently costing $0."""
    b = cfg.backends.get(logged)
    if b is not None:
        return b, None
    tail = logged.split("-", 1)[1] if "-" in logged else logged
    for bb in cfg.backends.values():
        if logged in bb.aliases or tail in bb.aliases:
            return bb, bb.name
    return None, None


def local_seat_watts(cfg, b) -> float:
    """Watts attributed to one local seat: its measured `watts` if configured, else its GPUs at
    full load plus a per-GPU share of the host's non-GPU base — the same formula /status uses.
    Charging every request the whole host (the old behaviour) inflated local $/Mtok 2-3x."""
    en = cfg.energy
    if b is not None and b.watts:
        return float(b.watts)
    n = (b.gpus if (b is not None and b.gpus) else en.seat_gpus)
    locals_ = [x for x in cfg.backends.values() if x.johnny_bound or x.name.startswith("local")]
    total = en.total_gpus or sum((x.gpus or en.seat_gpus) for x in locals_) or n
    host_base = max(0.0, en.host_watts - total * en.gpu_watts)
    return n * en.gpu_watts + host_base * n / total


def compute(conn, cfg, period: str = "day", baseline: str | None = None) -> dict:
    """Build the savings report for a period. Reuses storage.usage_stats for aggregation."""
    from saint.storage import usage_stats

    since = since_for(period)
    baseline, baseline_how = pick_baseline(cfg, baseline)
    bb = cfg.backends.get(baseline)
    base_in = (bb.price_in or 0.0) if bb else 0.0
    base_out = (bb.price_out or 0.0) if bb else 0.0

    en = cfg.energy
    rates = local_decode_rates(conn)
    rows: list[Row] = []
    notes: list[str] = []

    for r in usage_stats(conn, since):
        if r["kind"] == "embed":
            continue                                  # not part of the chat cost counterfactual
        name = r["backend_chosen"]
        b, priced_as = resolve_backend(cfg, name)
        if priced_as:
            notes.append(f"{name} priced as {priced_as} (renamed / alias)")
        tin, tout = r["tokens_in"], r["tokens_out"]
        is_local = bool(b and b.johnny_bound) or name.startswith("local")

        if is_local:
            tok_s, src = ((b.tok_s, "bench") if (b is not None and b.tok_s) else (rates.get(name), "log"))
            watts = local_seat_watts(cfg, b)
            elec = (watts * en.price_kwh / (tok_s * 3.6)) if tok_s else None
            actual = (tout / 1e6 * elec) if elec is not None else 0.0
            rows.append(Row(name, "local", r["requests"], tin, tout, tok_s, elec,
                            actual, (tin * base_in + tout * base_out) / 1e6 if bb else None,
                            watts=watts, tok_per_w=(tok_s / watts if (tok_s and watts) else None),
                            source=(src if tok_s else None), priced_as=priced_as))
        elif b and b.price_in is not None:
            po = b.price_out or 0.0
            crd, cwr = r["cache_read"], r["cache_write"]
            cr_p, cw_p = _eff_cache_prices(b)
            uncached_in = max(tin - crd - cwr, 0)
            real = (uncached_in * b.price_in + crd * cr_p + cwr * cw_p + tout * po) / 1e6  # Anthropic bill
            no_cache = (tin * b.price_in + tout * po) / 1e6                                # list, no caching
            rows.append(Row(name, "cloud", r["requests"], tin, tout, None, None,
                            real, (tin * base_in + tout * base_out) / 1e6 if bb else None, no_cache,
                            priced_as=priced_as))
        else:
            rows.append(Row(name, "other", r["requests"], tin, tout, None, None, 0.0, None))

    cloud_cost = sum(x.actual_cost for x in rows if x.kind == "cloud")
    local_cost = sum(x.actual_cost for x in rows if x.kind == "local")
    local_cloud_equiv = sum((x.cloud_equiv or 0.0) for x in rows if x.kind == "local")
    # what an all-cloud-baseline world would have paid for the LOCAL traffic, minus its
    # energy bill = the savings the local fleet produced
    local_savings = local_cloud_equiv - local_cost
    # split the cloud savings cleanly: cheaper-tier routing (baseline vs own list price,
    # both uncached) and prompt caching (own list price vs what Anthropic actually billed)
    tier_savings = sum((x.cloud_equiv or 0.0) - (x.no_cache_cost or 0.0)
                       for x in rows if x.kind == "cloud" and x.cloud_equiv is not None)
    caching_saved = sum((x.no_cache_cost or 0.0) - x.actual_cost
                        for x in rows if x.kind == "cloud")
    total_actual = cloud_cost + local_cost
    total_if_all_cloud = local_cloud_equiv + sum((x.cloud_equiv or 0.0)
                                                 for x in rows if x.kind == "cloud")
    savings = total_if_all_cloud - total_actual

    return {
        "period": period,
        "since": since,
        "baseline": baseline,
        "baseline_how": baseline_how,
        "notes": notes,
        "energy": {"price_kwh": en.price_kwh, "gpu_watts": en.gpu_watts, "host_watts": en.host_watts},
        "rows": rows,
        "cloud_cost": cloud_cost,
        "local_cost": local_cost,
        "local_cloud_equiv": local_cloud_equiv,
        "local_savings": local_savings,
        "tier_savings": tier_savings,
        "caching_saved": caching_saved,
        "total_actual": total_actual,
        "total_if_all_cloud": total_if_all_cloud,
        "savings": savings,
        "savings_pct": (100.0 * savings / total_if_all_cloud) if total_if_all_cloud else 0.0,
        "requests": sum(x.requests for x in rows),
    }


# ---- rendering: self-contained ANSI (truecolor, degrades on dumb terminals) ----------
_C = {
    "reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
    "cloud": "\033[38;5;110m", "local": "\033[38;5;150m", "save": "\033[38;5;120m",
    "gold": "\033[38;5;222m", "rule": "\033[38;5;240m", "red": "\033[38;5;210m",
    "cyan": "\033[38;5;80m",
}
# 5-row block digits for the headline number (legible: distinct 2/5, clear spacing)
_GLYPH = {
    "0": ("█████", "█   █", "█   █", "█   █", "█████"),
    "1": ("  ██ ", " ███ ", "  ██ ", "  ██ ", "█████"),
    "2": ("█████", "    █", "█████", "█    ", "█████"),
    "3": ("█████", "    █", " ████", "    █", "█████"),
    "4": ("█   █", "█   █", "█████", "    █", "    █"),
    "5": ("█████", "█    ", "█████", "    █", "█████"),
    "6": ("█████", "█    ", "█████", "█   █", "█████"),
    "7": ("█████", "    █", "   █ ", "  █  ", "  █  "),
    "8": ("█████", "█   █", "█████", "█   █", "█████"),
    "9": ("█████", "█   █", "█████", "    █", "█████"),
    "$": (" ███ ", "█ █  ", " ███ ", "  █ █", " ███ "),
    ".": ("     ", "     ", "     ", "     ", "  █  "),
    ",": ("     ", "     ", "     ", "  █  ", " █   "),
    "-": ("     ", "     ", "█████", "     ", "     "),
    " ": ("   ", "   ", "   ", "   ", "   "),
}
_GH = 5  # glyph height


def _bignum(text: str, color: str) -> str:
    rows = [""] * _GH
    for ch in text:
        g = _GLYPH.get(ch, _GLYPH[" "])
        for i in range(_GH):
            rows[i] += g[i] + "  "
    return "\n".join(color + _C["bold"] + r + _C["reset"] for r in rows)


def _bar(frac: float, width: int, color: str) -> str:
    frac = max(0.0, min(1.0, frac))
    fill = round(frac * width)
    return color + "█" * fill + _C["rule"] + "░" * (width - fill) + _C["reset"]


def _money(v: float) -> str:
    return f"${v:,.2f}"


import re as _re
_ANSI = _re.compile(r"\x1b\[[0-9;]*m")


def render(rep: dict, color: bool = True) -> str:
    if not color:
        return _ANSI.sub("", render(rep, color=True))
    def c(k): return _C[k]
    R = c("reset")
    W = 54
    out: list[str] = []
    label = {"hour": "last hour", "day": "last 24 hours", "week": "last 7 days",
             "month": "last 30 days", "year": "last year", "all": "all time"}[rep["period"]]

    out.append("")
    out.append(f"  {c('gold')}{c('bold')}╔{'═' * (W - 2)}╗{R}")
    title = "S.A.I.N.T.   ·   S A V I N G S"   # Semantic Artificial Intelligence Network Terminus
    pad = (W - 2 - len(title)) // 2
    out.append(f"  {c('gold')}{c('bold')}║{R}{' ' * pad}{c('cyan')}{c('bold')}{title}{R}"
               f"{' ' * (W - 2 - pad - len(title))}{c('gold')}{c('bold')}║{R}")
    out.append(f"  {c('gold')}{c('bold')}╚{'═' * (W - 2)}╝{R}")
    out.append(f"  {c('dim')}⚡ {label} · {rep['requests']:,} requests · "
               f"baseline {rep['baseline']} ({rep.get('baseline_how', '')}){R}")
    out.append("")

    peak = max(rep["total_if_all_cloud"], 1e-9)
    out.append(f"  {c('cloud')}cloud{R}  {_bar(rep['cloud_cost'] / peak, 26, c('cloud'))}"
               f"  {c('cloud')}{_money(rep['cloud_cost'])}{R}")
    out.append(f"  {c('local')}local{R}  {_bar(rep['local_cost'] / peak, 26, c('local'))}"
               f"  {c('local')}{_money(rep['local_cost'])}{R} {c('dim')}(energy){R}")
    out.append(f"  {c('save')}saved{R}  {_bar(rep['savings'] / peak, 26, c('save'))}"
               f"  {c('save')}{_money(rep['savings'])}{R}")
    out.append("")
    out.append(f"  {c('dim')}if all cloud {_money(rep['total_if_all_cloud'])}   "
               f"you paid {_money(rep['total_actual'])}{R}")
    out.append("")

    sav = rep["savings"]
    out.append(f"  {c('save') if sav >= 0 else c('red')}{c('bold')}"
               f"{'SAVED' if sav >= 0 else 'OVERSPEND'}{R}")
    out.append(_indent(_bignum(_money(sav), c("save") if sav >= 0 else c("red")), 2))
    out.append(f"     {c('save') if sav >= 0 else c('red')}{c('bold')}"
               f"{rep['savings_pct']:.0f}% saved vs all-cloud{R}")
    out.append(f"     {c('dim')}from  local fleet {_money(rep['local_savings'])}"
               f"  ·  cheaper tiers {_money(rep['tier_savings'])}"
               f"  ·  prompt caching {_money(rep['caching_saved'])}{R}")
    out.append("")

    # per-backend breakdown
    out.append(f"  {c('rule')}{'─' * W}{R}")
    out.append(f"  {c('dim')}{'backend':<22}{'req':>6}{'tok out':>11}"
               f"{'$/Mtok':>9}{'cost':>10}{R}")
    for x in sorted(rep["rows"], key=lambda r: -r.actual_cost):
        kc = c("cloud") if x.kind == "cloud" else c("local") if x.kind == "local" else c("dim")
        rate = (f"{x.elec_per_mtok:.2f}" if x.kind == "local" and x.elec_per_mtok
                else "—" if x.kind == "local" else "")
        tag = " ⚡" if x.kind == "local" else ""
        name = x.backend + tag
        # ⚡ renders as 2 terminal columns but counts as 1 char, so pad by visual width
        pad = " " * max(0, 22 - len(name) - (1 if x.kind == "local" else 0))
        out.append(f"  {kc}{name}{pad}{R}{x.requests:>6}{x.tokens_out:>11,}"
                   f"{c('dim')}{rate:>9}{R}{kc}{_money(x.actual_cost):>10}{R}")
    loc = [x for x in rep["rows"] if x.kind == "local" and x.tok_s]
    if loc:
        en = rep.get("energy") or {}
        out.append("")
        out.append(f"  {c('dim')}energy ${en.get('price_kwh', 0):.2f}/kWh · {en.get('gpu_watts', 0):.0f} W per GPU at load · "
                   f"tok/s [bench] = config tok_s from johnny bench, [log] = request-log p75{R}")
        for x in sorted(loc, key=lambda r: -(r.tok_per_w or 0)):
            out.append(f"  {c('dim')}{x.backend:<22}{x.watts:>6.0f} W{x.tok_s:>8.1f} tok/s [{x.source}]"
                       f"{(x.tok_per_w or 0):>8.3f} tok/W{(x.elec_per_mtok or 0):>8.2f} $/Mtok{R}")
    for n in rep.get("notes") or []:
        out.append(f"  {c('dim')}note: {n}{R}")
    out.append("")
    # Flush-left, so a client that .strip()s the response can't misalign line 1
    # (the margin lived only as a leading 2 spaces per line; relative indent is kept).
    text = "\n".join(out)
    return "\n".join(ln[2:] if ln.startswith("  ") else ln for ln in text.split("\n"))


def _indent(block: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in block.split("\n"))
