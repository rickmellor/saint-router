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
    actual_cost: float            # $ actually spent (cloud) or burned in power (local)
    cloud_equiv: float | None     # what the baseline cloud backend would have charged


def compute(conn, cfg, period: str = "day", baseline: str | None = None) -> dict:
    """Build the savings report for a period. Reuses storage.usage_stats for aggregation."""
    from saint.storage import usage_stats

    since = since_for(period)
    baseline = baseline or cfg.routing.default_on_failure
    bb = cfg.backends.get(baseline)
    base_in = (bb.price_in or 0.0) if bb else 0.0
    base_out = (bb.price_out or 0.0) if bb else 0.0

    en = cfg.energy
    rates = local_decode_rates(conn)
    rows: list[Row] = []

    for r in usage_stats(conn, since):
        if r["kind"] == "embed":
            continue                                  # not part of the chat cost counterfactual
        name = r["backend_chosen"]
        b = cfg.backends.get(name)
        tin, tout = r["tokens_in"], r["tokens_out"]
        is_local = bool(b and b.johnny_bound) or name.startswith("local")

        if is_local:
            tok_s = rates.get(name)
            elec = (en.host_watts * en.price_kwh / (tok_s * 3.6)) if tok_s else None
            actual = (tout / 1e6 * elec) if elec is not None else 0.0
            rows.append(Row(name, "local", r["requests"], tin, tout, tok_s, elec,
                            actual, (tin * base_in + tout * base_out) / 1e6 if bb else None))
        elif b and b.price_in is not None:
            actual = (tin * b.price_in + tout * (b.price_out or 0.0)) / 1e6
            rows.append(Row(name, "cloud", r["requests"], tin, tout, None, None,
                            actual, (tin * base_in + tout * base_out) / 1e6 if bb else None))
        else:
            rows.append(Row(name, "other", r["requests"], tin, tout, None, None, 0.0, None))

    cloud_cost = sum(x.actual_cost for x in rows if x.kind == "cloud")
    local_cost = sum(x.actual_cost for x in rows if x.kind == "local")
    local_cloud_equiv = sum((x.cloud_equiv or 0.0) for x in rows if x.kind == "local")
    # what an all-cloud-baseline world would have paid for the LOCAL traffic, minus its
    # energy bill = the savings the local fleet produced
    local_savings = local_cloud_equiv - local_cost
    # bonus: routing cloud traffic to cheaper-than-baseline tiers
    tier_savings = sum((x.cloud_equiv or 0.0) - x.actual_cost
                       for x in rows if x.kind == "cloud" and x.cloud_equiv is not None)
    total_actual = cloud_cost + local_cost
    total_if_all_cloud = local_cloud_equiv + sum((x.cloud_equiv or 0.0)
                                                 for x in rows if x.kind == "cloud")
    savings = total_if_all_cloud - total_actual

    return {
        "period": period,
        "since": since,
        "baseline": baseline,
        "rows": rows,
        "cloud_cost": cloud_cost,
        "local_cost": local_cost,
        "local_cloud_equiv": local_cloud_equiv,
        "local_savings": local_savings,
        "tier_savings": tier_savings,
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
    title = "S A I N T   ·   S A V I N G S"
    pad = (W - 2 - len(title)) // 2
    out.append(f"  {c('gold')}{c('bold')}║{R}{' ' * pad}{c('cyan')}{c('bold')}{title}{R}"
               f"{' ' * (W - 2 - pad - len(title))}{c('gold')}{c('bold')}║{R}")
    out.append(f"  {c('gold')}{c('bold')}╚{'═' * (W - 2)}╝{R}")
    out.append(f"  {c('dim')}⚡ {label} · {rep['requests']:,} requests · "
               f"baseline {rep['baseline']}{R}")
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
               f"  ·  cheaper tiers {_money(rep['tier_savings'])}{R}")
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
        out.append(f"  {kc}{(x.backend + tag):<22}{R}{x.requests:>6}{x.tokens_out:>11,}"
                   f"{c('dim')}{rate:>9}{R}{kc}{_money(x.actual_cost):>10}{R}")
    out.append("")
    return "\n".join(out)


def _indent(block: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in block.split("\n"))
