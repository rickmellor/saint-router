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
  `by_backend` is keyed on the backend or seat that SERVED the request (`served_backend`), not on
  what was asked for, and `by_route` groups the same rows by how they were dispatched.

Nothing here returns prompt or completion text.
"""
from __future__ import annotations

import hmac
import os
import re
import sqlite3
from datetime import UTC, date, datetime, timedelta
from importlib import metadata
from urllib.parse import urlsplit

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
# `johnny-Qwen3.8-…-8003` → 8003. Seat ids that carry no port (`gemma31.service`,
# `saint-features.service#embed`) simply don't match, so the row stays under backend_chosen.
_SEAT_PORT = re.compile(r"-(\d+)$")


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


def served_backend(cfg, backend_chosen: str, johnny_seat: str | None,
                   cache: dict | None = None) -> str:
    """The backend that ACTUALLY served a logged request, which is not always `backend_chosen`.

    `backend_chosen` is what routing picked; the seat that ran it is only in `johnny_seat` —
    a pool backend logs `local-pool` while one member's seat answered, and a spill logs the
    requested backend while another role's seat did the work. Both name that seat by id, whose
    trailing `-<port>` is the one field that maps back onto a configured backend's `base_url`.

    Anything this cannot resolve comes back as `backend_chosen` rather than a guess: a seat id
    with no port (`gemma31.service`, `saint-features.service#embed`), NULL, or a port that
    several backends share or none has."""
    key = (backend_chosen, johnny_seat)
    if cache is not None and key in cache:
        return cache[key]

    m = _SEAT_PORT.search(johnny_seat or "")
    port = int(m.group(1)) if m else None
    chosen = cfg.backends.get(backend_chosen)
    out: str | None = None
    if port is None:
        pass                                    # no port in the seat id (or NULL): cannot tell
    elif chosen is not None and not chosen.is_pool and _port_of(chosen.base_url) == port:
        out = backend_chosen                    # the seat WAS the requested backend's own
    else:
        candidates = [b for b in cfg.backends.values()
                      if not b.is_pool and _port_of(b.base_url) == port]
        if chosen is not None and chosen.is_pool:      # a pool row can only be one of its members
            candidates = [b for b in candidates if b.name in chosen.pool]
            if len(candidates) == 1:
                out = candidates[0].name
        if out is None:
            # a spill went to another role's backend, and role backends are johnny-bound; a port
            # several non-johnny backends share stays unresolved rather than guessed
            bound = [b for b in candidates if b.johnny_bound]
            if len(bound) > 1 and chosen is not None and chosen.spill:
                # two seats bound on one port (a live role and a profile-specific one): a spill
                # can only have gone to a role the requested backend lists in `spill`
                bound = [b for b in bound if b.johnny_role in chosen.spill]
            if len(bound) == 1:
                out = bound[0].name
            elif len(candidates) == 1:
                out = candidates[0].name
    if out is None:
        out = backend_chosen
    if cache is not None:
        cache[key] = out
    return out


def _port_of(base_url: str | None) -> int | None:
    """The port of a backend's `base_url`, or None when it has none (cloud) or is unparsable."""
    try:
        return urlsplit(base_url).port
    except ValueError:
        return None


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
           "cache_read_tokens, cache_write_tokens, johnny_seat, pinned_backend FROM requests ")


def query(conn, cfg, params: dict[str, str]) -> tuple[int, dict]:
    """`GET /usage/v1` → (status, body) for the contract's parameters. Read-only, one pass over the
    window (plus a MIN(ts) for HEADERS_SINCE), and never any prompt text."""
    try:
        win = _parse(params)
    except _BadParam as e:
        return _error(e.code, str(e), param=e.param, allowed=e.allowed, hint=e.hint)

    # [logging] usage_since: rows before it predate attribution, so the report starts there
    since = getattr(getattr(cfg, "logging", None), "usage_since", None)
    win["since"] = since if since and since > win["lo"] else None
    lo = win["since"] or win["lo"]
    sql, args = "WHERE ts >= ? AND ts < ? AND backend_chosen IS NOT NULL", [lo, win["hi"]]
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
    seats: dict = {}
    totals, by_day, by_backend = _new_agg(), {}, {}
    by_client, by_origin, by_route = {}, {}, {}
    local_rows = local_priced = 0

    for ts, backend, client, origin, tin, tout, cr, cw, seat, pinned in rows:
        tokens = _tokens(tin, tout, cr, cw)
        # what routing picked, and what actually ran: money and by_backend follow the SERVER, so a
        # pool or spill row is priced on the hardware that burned it
        served = served_backend(cfg, backend or "", seat, seats)
        billing, b = billing_for(cfg, served, resolve)
        if billing == "local":
            local_rows += 1
            # electricity for the output tokens; NULL when the seat has no known decode rate
            elec = _savings.local_elec_per_mtok(cfg, b, served, rates)
            energy = (tokens["output"] / 1e6 * elec) if elec is not None else None
            billed = None                       # local hardware carries no per-token bill
            local_priced += energy is not None
        else:
            energy = None                       # not our electricity to estimate
            billed = (_savings.cloud_bill(b, tin, tout, cr, cw)
                      if (b is not None and b.price_in is not None) else None)

        _add(totals, tokens, energy, billed)
        _add(by_day.setdefault((ts or "")[:10], _new_agg()), tokens, energy, billed)
        _add(by_backend.setdefault(served, _new_agg()), tokens, energy, billed)
        _add(by_client.setdefault(client or "unknown", _new_agg()), tokens, energy, billed)
        _add(by_origin.setdefault(origin, _new_agg()), tokens, energy, billed)
        chosen = cfg.backends.get(backend or "")
        dispatch = ((chosen.name, "pool") if chosen is not None and chosen.is_pool
                    else ("pinned", "pinned") if pinned is not None else ("routed", "routed"))
        _add(by_route.setdefault(dispatch, _new_agg()), tokens, energy, billed)

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
    # SAINT extension (the contract allows extra fields): how the rows were dispatched
    body["by_route"] = [{"route": label, "kind": kind, **_entry(agg)}
                        for (label, kind), agg in _sorted_entries(by_route)]
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
    if win.get("since"):
        # the cut-off replaces HEADERS_SINCE: the unattributed rows are excluded, not just unlabeled
        notes.append({"code": "USAGE_SINCE",
                      "text": f"rows logged before {win['since']} are excluded ([logging] usage_since): "
                              "they predate attribution"})
    elif first_attr is None:
        notes.append({"code": "HEADERS_SINCE",
                      "text": "this log holds no attributed rows yet — nothing has been logged "
                              "since attribution was added, so client/origin filters are empty"})
    elif win["from"].isoformat() < first_attr[:10]:
        notes.append({"code": "HEADERS_SINCE",
                      "text": f"attribution is only recorded from {first_attr[:10]} "
                              "(earlier rows have no client label)"})
    return notes
