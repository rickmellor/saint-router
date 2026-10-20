"""SAINT as a `hybrid-usage/1` provider: attribution, and the `GET /usage/v1` report.

A spend report that reads only the cloud bill cannot see the work this router sent to local
seats, and this router cannot see the cloud bill. The contract (canonical text:
`neurio/hsp-arch-tools` `docs/hybrid-inference-usage-contract.md`) lets a consumer read this
router's own request log and show the two side by side.

Two halves:

* **Attribution** (`client_label`, `attribution`) — recorded on every logged request
  (migration 0005). `client` comes from the `User-Agent`; the rest from the optional
  `X-Inference-Origin/Session/Job` headers. Metadata only: never prompt content, never forwarded
  upstream. `client_label` labels requests for the report and MUST NOT change routing — the
  `[clients]` routing rules stay exactly where they are, in `router.decide_route`.
* **The report** (`query`) — one read-only pass over the `requests` table: the disjoint four-way
  token split, a `billing` value per backend, and the two money columns. Money comes from
  `saint.savings`, so `saint savings` and `/usage/v1` cannot price the same tokens differently.

Nothing here returns prompt or completion text.
"""
from __future__ import annotations

import hmac
import os
import re
import sqlite3
from datetime import UTC, date, datetime, timedelta
from importlib import metadata

from saint import savings as _savings

CONTRACT = "hybrid-usage/1"
PROVIDER_NAME = "saint"
PARAM_ORDER = ("from", "to", "granularity", "client", "origin", "session")
GRANULARITIES = ("day", "total")
DEFAULT_GRANULARITY = "day"
DEFAULT_LOOKBACK_DAYS = 29      # `from` defaults to `to` minus 29 days => a 30-day window
WINDOW_MAX_DAYS = 366           # a longer window is a 400
HEADER_MAX_LEN = 128            # contract §1: longer values are truncated, not rejected
TOKEN_ENV_VAR = "SAINT_USAGE_TOKEN"   # unset/empty ⇒ the endpoint is open (contract §6)
TOKEN_TYPES = ("input", "cache_read", "cache_creation", "output")

# Neither is inference a consumer wants priced: saint-explain answers with the routing decision
# (no dispatch at all) and embeddings are infra traffic that must never be priced at chat rates.
# Same exclusion storage.usage_stats applies.
_EXCLUDED_MODELS = ("saint-explain", "saint-embeddings")

# provider (from [backends.*]) → who bills it. Anything else — and any backend SAINT cannot
# resolve at all — is "other".
_BILLING_PROVIDERS = {
    "anthropic": "anthropic",
    "bedrock": "bedrock",
    "openai": "openai",
    "vertex_ai": "vertex",
    "vertex": "vertex",
    "azure": "azure",
}

# `ts` is stored as ISO-8601 UTC with a "+00:00" offset, so lexicographic comparison against a
# "YYYY-MM-DDT00:00:00" bound is exactly the half-open window we want: "" < "+…" keeps `from`
# 00:00 inclusive, and `to` + 1 day at 00:00 is exclusive. Proven in tests/test_usage.py.
_DAY_START = "T00:00:00"

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_UA_TOKEN = re.compile(r"\s*([^\s/]+)")


# --------------------------------------------------------------- attribution

def _clean_header(value: str | None) -> str | None:
    """A header value as stored: stripped, non-printable-ASCII removed, cut at 128 chars; empty
    becomes NULL. Client metadata is free-form, so a stray control character or an absurd length
    is normalized rather than rejected (contract §1) — and nothing hostile reaches the report."""
    if not value:
        return None
    cleaned = _CONTROL.sub("", str(value)).strip()[:HEADER_MAX_LEN]
    return cleaned or None


def client_label(cfg, user_agent: str | None) -> str:
    """The client label recorded on EVERY request (contract §1: providers MUST record it, whether
    or not it influenced routing).

    The `name` of the first `[clients]` rule whose `user_agent` is a case-insensitive prefix of the
    header (rule order), else the UA's first product token lowercased, cut at `/` or space
    (`curl/8.5.0` → `curl`), else `unknown`."""
    if not user_agent:
        return "unknown"
    ua = user_agent.strip()
    low = ua.lower()
    for rule in getattr(cfg, "clients", ()) or ():
        if rule.user_agent and low.startswith(rule.user_agent.lower()):
            return rule.name
    m = _UA_TOKEN.match(ua)
    return m.group(1).lower() if m else "unknown"


def attribution(headers, cfg) -> dict:
    """The four attribution values for one request. Pass straight into the log row:
    `build_log_row(..., **attribution(request.headers, cfg))`."""
    get = getattr(headers, "get", None)
    ua = get("user-agent") if get else None
    return {
        "client": client_label(cfg, ua),
        "origin": _clean_header(get("x-inference-origin") if get else None),
        "session": _clean_header(get("x-inference-session") if get else None),
        "job_id": _clean_header(get("x-inference-job") if get else None),
    }


# -------------------------------------------------------------------- errors

class _BadParam(ValueError):
    """A 400 from parameter validation. Raised internally; `query` renders the contract body."""

    def __init__(self, message: str, *, code: str = "INVALID_PARAM", param: str | None = None,
                 allowed: list[str] | None = None, hint: str | None = None):
        super().__init__(message)
        self.code, self.param, self.allowed, self.hint = code, param, allowed, hint


def _error(code: str, message: str, **extra) -> tuple[int, dict]:
    """The contract's error shape (same as the ccwb spend API's, so one consumer serves both)."""
    status = {"UNKNOWN_PARAM": 400, "INVALID_PARAM": 400, "UNAUTHORIZED": 401}.get(code, 503)
    body: dict = {"error": message, "code": code}
    body.update({k: v for k, v in extra.items() if v is not None})
    return status, body


# ---------------------------------------------------------------- parameters

def _date_param(raw: str, key: str) -> date:
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        raise _BadParam(f"'{raw}' is not a valid YYYY-MM-DD date", param=key,
                        hint="dates are YYYY-MM-DD, UTC") from None


def _filter_param(raw: str, key: str) -> list[str]:
    """A comma-separated filter → the values to keep. An empty value is INVALID_PARAM: dropping it
    silently would widen the query, the one thing a filter must never do."""
    values = [v.strip() for v in raw.split(",")]
    if any(not v for v in values):
        raise _BadParam(f"'{key}' must be a comma-separated list of non-empty values", param=key,
                        hint="e.g. ?client=claude-code,pi")
    out: list[str] = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


def _parse(params: dict[str, str]) -> dict:
    """Validate everything up front (contract §2: a typo must not silently widen the query)."""
    unknown = sorted(k for k in params if k not in PARAM_ORDER)
    if unknown:
        raise _BadParam("unknown query parameter", code="UNKNOWN_PARAM",
                        param=",".join(unknown), allowed=list(PARAM_ORDER),
                        hint="use only the parameters in allowed")

    # `is not None` throughout: a present-but-empty parameter is a value the client sent, and
    # every one of them is INVALID_PARAM when empty — never silently the default.
    to = _date_param(params["to"], "to") if params.get("to") is not None else _today()
    frm = (_date_param(params["from"], "from") if params.get("from") is not None
           else to - timedelta(days=DEFAULT_LOOKBACK_DAYS))
    if frm > to:
        raise _BadParam("'from' must not be after 'to'", param="from",
                        hint="both dates are inclusive, UTC")
    if (to - frm).days + 1 > WINDOW_MAX_DAYS:
        raise _BadParam(f"window is longer than {WINDOW_MAX_DAYS} days", param="from",
                        hint=f"narrow the window to at most {WINDOW_MAX_DAYS} days")

    granularity = params.get("granularity")
    granularity = DEFAULT_GRANULARITY if granularity is None else granularity.strip()
    if granularity not in GRANULARITIES:
        raise _BadParam(f"'{granularity}' is not a valid granularity", param="granularity",
                        allowed=list(GRANULARITIES), hint="day or total")

    return {
        "from": frm,
        "to": to,
        "lo": frm.isoformat() + _DAY_START,                       # inclusive
        "hi": (to + timedelta(days=1)).isoformat() + _DAY_START,   # exclusive
        "granularity": granularity,
        "filters": {k: (_filter_param(params[k], k) if params.get(k) is not None else None)
                    for k in ("client", "origin", "session")},
    }


# ------------------------------------------------------------- aggregation

def _tokens(tin: int | None, tout: int | None, cr: int | None, cw: int | None) -> dict:
    """The disjoint four-way split (contract §3). SAINT logs one prompt-token number that includes
    cache hits, so both cache figures come off it, clamped at 0. Missing is 0, never omitted."""
    cr, cw = cr or 0, cw or 0
    return {"input": max((tin or 0) - cr - cw, 0), "cache_read": cr,
            "cache_creation": cw, "output": tout or 0}


def _new_agg() -> dict:
    return {"requests": 0, "tokens": dict.fromkeys(TOKEN_TYPES, 0), "energy": [], "billed": []}


def _add(agg: dict, tokens: dict, energy: float | None, billed: float | None) -> None:
    agg["requests"] += 1
    for k, v in tokens.items():
        agg["tokens"][k] += v
    agg["energy"].append(energy)
    agg["billed"].append(billed)


def _money(parts: list[float | None]) -> float | None:
    """Sum of what is known; NULL only when every contributing value was unknown. `0.0` means
    "known to be zero", which is not the same statement."""
    known = [p for p in parts if p is not None]
    return round(sum(known), 4) if known else None


def _entry(agg: dict) -> dict:
    return {"requests": agg["requests"], "tokens_by_type": dict(agg["tokens"]),
            "total_tokens": sum(agg["tokens"].values())}


def _money_entry(agg: dict) -> dict:
    return {**_entry(agg), "energy_cost_usd": _money(agg["energy"]),
            "billed_cost_usd": _money(agg["billed"])}


def _sorted_entries(bucket: dict) -> list:
    return sorted(bucket.items(), key=lambda kv: (-kv[1]["requests"], str(kv[0])))


def _today() -> date:
    """Today, UTC. A seam so `to`'s default and PARTIAL_PERIOD are testable without a clock."""
    return datetime.now(UTC).date()


def _iso(ts: str | None) -> str | None:
    """A logged ts (ISO-8601 +00:00) as the contract's Z-suffixed UTC stamp."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return ts


def provider_version() -> str:
    """The installed package version; "unknown" when running from a source tree with no dist-info
    (an editable checkout, or a test import path)."""
    try:
        return metadata.version("saint-router")
    except Exception:
        return "unknown"


def billing_for(cfg, name: str, cache: dict | None = None) -> tuple[str, object]:
    """(billing, backend) for a logged `backend_chosen`. Renamed/aliased names resolve through
    `savings.resolve_backend`, so rows written before a rename keep their billing instead of
    degrading to `other`. `local` wins over any provider (the test `savings.compute` uses); else
    the provider's biller; unknown provider or backend → `other`."""
    if cache is not None and name in cache:
        return cache[name]
    b, _priced_as = _savings.resolve_backend(cfg, name)
    if _savings.is_local(b, name):
        out = "local", b
    elif b is None:
        out = "other", None
    else:
        out = _BILLING_PROVIDERS.get(str(b.provider), "other"), b
    if cache is not None:
        cache[name] = out
    return out


# -------------------------------------------------------------------- auth

def auth_error(headers, token: str | None = None) -> tuple[int, dict] | None:
    """Contract §6: a provider may be open, or require `Authorization: Bearer <token>`. The token
    comes from the environment (never the URL); unset or empty means open. Constant-time compare —
    the endpoint is on a LAN but a LAN is not nothing."""
    expected = os.environ.get(TOKEN_ENV_VAR) if token is None else token
    if not expected:
        return None
    supplied = headers.get("authorization") if hasattr(headers, "get") else None
    ok = bool(supplied) and hmac.compare_digest(str(supplied), f"Bearer {expected}")
    return None if ok else _error("UNAUTHORIZED", "missing or invalid bearer token",
                                  hint=f"send 'Authorization: Bearer <{TOKEN_ENV_VAR}>'")


# ------------------------------------------------------------------- query

_SELECT = ("SELECT ts, backend_chosen, client, origin, tokens_in, tokens_out, "
           "cache_read_tokens, cache_write_tokens FROM requests ")


def query(conn, cfg, params: dict[str, str]) -> tuple[int, dict]:
    """`GET /usage/v1` → (status, body) for the contract's parameters. Read-only, one pass over the
    window (plus a MIN(ts) for HEADERS_SINCE), and never any prompt text."""
    try:
        win = _parse(params)
    except _BadParam as e:
        return _error(e.code, str(e), param=e.param, allowed=e.allowed, hint=e.hint)

    sql, args = "WHERE ts >= ? AND ts < ? AND backend_chosen IS NOT NULL", [win["lo"], win["hi"]]
    sql += (" AND (model_field IS NULL OR model_field NOT IN ("
            + ",".join("?" * len(_EXCLUDED_MODELS)) + "))")
    args += list(_EXCLUDED_MODELS)
    for key in ("client", "origin", "session"):
        values = win["filters"][key]
        if values:
            # An absent header is NULL, and NULL never appears in an IN list: a request without the
            # header never matches an origin/session filter (contract §2). `client` differs by
            # design — every row is labeled, so a client filter is meaningful on old rows too.
            sql += f" AND {key} IN ({', '.join('?' * len(values))})"
            args += values

    try:
        rows = conn.execute(_SELECT + sql + " ORDER BY id", args).fetchall()
        first_attr = conn.execute(
            "SELECT MIN(ts) FROM requests WHERE client IS NOT NULL").fetchone()[0]
    except sqlite3.Error as e:
        return _error("UNAVAILABLE", f"request log cannot be read right now: {type(e).__name__}: {e}")

    rates = _savings.local_decode_rates(conn)   # log-measured p75 tok/s for seats without `tok_s`
    resolve: dict = {}
    totals, by_day, by_backend = _new_agg(), {}, {}
    by_client, by_origin = {}, {}
    local_rows = local_priced = 0

    for ts, backend, client, origin, tin, tout, cr, cw in rows:
        tokens = _tokens(tin, tout, cr, cw)
        billing, b = billing_for(cfg, backend or "", resolve)
        if billing == "local":
            local_rows += 1
            # electricity for the output tokens; NULL when the seat has no known decode rate
            elec = _savings.local_elec_per_mtok(cfg, b, backend or "", rates)
            energy = (tokens["output"] / 1e6 * elec) if elec is not None else None
            billed = None                       # local hardware carries no per-token bill
            local_priced += energy is not None
        else:
            energy = None                       # not our electricity to estimate
            billed = (_savings.cloud_bill(b, tin, tout, cr, cw)
                      if (b is not None and b.price_in is not None) else None)

        _add(totals, tokens, energy, billed)
        _add(by_day.setdefault((ts or "")[:10], _new_agg()), tokens, energy, billed)
        _add(by_backend.setdefault(backend, _new_agg()), tokens, energy, billed)
        _add(by_client.setdefault(client or "unknown", _new_agg()), tokens, energy, billed)
        _add(by_origin.setdefault(origin, _new_agg()), tokens, energy, billed)

    # keys in the contract's order; by_day only when granularity=day
    body: dict = {
        "contract": CONTRACT,
        "provider": {"name": PROVIDER_NAME, "version": provider_version()},
        "window": {"from": win["from"].isoformat(), "to": win["to"].isoformat(), "timezone": "UTC"},
        "filters": dict(win["filters"]),
        "generated_at": _iso(datetime.now(UTC).isoformat()),
        "data_through": _iso(max((r[0] for r in rows if r[0]), default=None)),
        "currency": "USD",
        "totals": _money_entry(totals),
    }
    if win["granularity"] == "day":
        body["by_day"] = [{"day": day, **_money_entry(agg)} for day, agg in sorted(by_day.items())]
    body["by_backend"] = []
    for name, agg in sorted(by_backend.items(), key=lambda kv: (-kv[1]["requests"], kv[0])):
        billing, b = billing_for(cfg, name, resolve)      # cached, so this is not a second lookup
        body["by_backend"].append({"backend": name, "billing": billing,
                                   # the config model; null for a backend SAINT can't resolve
                                   "model": getattr(b, "model", None), **_money_entry(agg)})
    body["by_client"] = [{"client": key, **_entry(agg)} for key, agg in _sorted_entries(by_client)]
    body["by_origin"] = [{"origin": key, **_entry(agg)} for key, agg in _sorted_entries(by_origin)]
    body["notes"] = _notes(win, rows, first_attr, local_rows, local_priced)
    return 200, body


def _notes(win: dict, rows: list, first_attr: str | None, local_rows: int,
           local_priced: int) -> list[dict]:
    notes: list[dict] = []
    if win["to"] >= _today():
        notes.append({"code": "PARTIAL_PERIOD",
                      "text": "the window includes today, which is still accumulating"})
    if local_rows and not local_priced:
        notes.append({"code": "NO_ENERGY_MODEL",
                      "text": f"{local_rows} local rows exist but no serving seat had a known "
                              "decode rate, so energy_cost_usd would not estimate anything"})
    if first_attr is None:
        notes.append({"code": "HEADERS_SINCE",
                      "text": "this log holds no attributed rows yet — nothing has been logged "
                              "since attribution was added, so client/origin filters are empty"})
    elif win["from"].isoformat() < first_attr[:10]:
        notes.append({"code": "HEADERS_SINCE",
                      "text": f"attribution is only recorded from {first_attr[:10]} "
                              "(earlier rows have no client label)"})
    return notes
