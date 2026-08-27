from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from saint.binding import resolve_for_dispatch
from saint.config import Config
from saint.explain import format_decision
from saint.johnny import build_resolver, provide_telemetry
from saint.prefixes import UnknownPrefixError
from saint.router import SAINT_AUTO, decide_route, dispatch_non_streaming
from saint import savings as _savings
from saint.backends import call_embeddings
from saint.dispatch import BedrockRuntime, dispatch_candidates, run_candidates
from saint.route_cache import Breaker, RouteCaches, TTLCache
from saint.storage import LogRow, build_log_row, log_request, open_db

# Chat-completion kwargs we forward to the destination backend, beyond
# `model`/`messages`/`stream`/`tools`/`tool_choice` (which are handled explicitly).
# Anything not in this list is silently dropped. Add to v1.x as needed.
_FORWARDED_PARAMS = (
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "response_format",
    "seed",
    "stream_options",
    "user",
    # vLLM/llama.cpp template knobs (enable_thinking, reasoning_effort …). Forwarded as
    # `extra_body` to OpenAI-compatible backends and dropped for anthropic/bedrock — see
    # backends._shape_request. Without this a client's `enable_thinking: false` never reached
    # the seat, so a 200-token review request came back all reasoning, no content (2026-08-27).
    "chat_template_kwargs",
)


def _explain_response(decision_text: str, model_field: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_field,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": decision_text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


import re as _re

# /savings, /savings week, saint savings month, !savings, @savings … (case-insensitive)
_SAVINGS_RE = _re.compile(
    r"^\s*[/!@]?\s*(?:saint[\s-]+)?savings(?:[\s-]+report)?"
    r"(?:\s+(hour|day|week|month|year|all))?\s*$", _re.IGNORECASE)


def _savings_trigger(model_field: str, messages: list) -> str | None:
    """Return the requested period if this request is a savings-report ask, else None.

    Triggers on the reserved model (saint-savings[:period]) or a last-user-message
    sentinel. Period defaults to 'day'; an unknown period falls back to 'day'."""
    period = None
    if model_field == "saint-savings" or model_field.startswith("saint-savings:"):
        period = model_field.split(":", 1)[1] if ":" in model_field else "day"
    else:
        text = next((m.get("content", "") for m in reversed(messages)
                     if m.get("role") == "user"), "")
        if isinstance(text, str):
            m = _SAVINGS_RE.match(text)
            if m:
                period = (m.group(1) or "day")
    if period is None:
        return None
    period = period.strip().lower()
    return period if period in _savings.PERIODS else "day"


def _safe_log(db, row: LogRow) -> None:
    """Persist a log row. Failures here MUST NOT affect the response — log to stderr only."""
    try:
        log_request(db, row)
    except Exception as e:
        print(
            f"[router] log write failed for req#{row.request_id[:8]}: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr, flush=True,
        )


def _usage_dict(response: Any) -> dict:
    """Usage dict from a dict OR a litellm pydantic ModelResponse (the old isinstance(dict)
    check silently dropped token accounting for real non-streaming responses)."""
    if isinstance(response, dict):
        return response.get("usage") or {}
    if hasattr(response, "model_dump"):
        return (response.model_dump().get("usage")) or {}
    return {}


def _cache_tokens(usage: dict) -> tuple[int | None, int | None]:
    """(cache_read, cache_write) tokens from litellm usage. Anthropic surfaces top-level
    cache_read_input_tokens/cache_creation_input_tokens; OpenAI implicit caching only
    populates prompt_tokens_details.cached_tokens."""
    ptd = usage.get("prompt_tokens_details") or {}
    read = usage.get("cache_read_input_tokens")
    if read is None:
        read = ptd.get("cached_tokens")
    if read is None:
        read = usage.get("cacheReadInputTokens")  # Bedrock Converse naming
    write = usage.get("cache_creation_input_tokens")
    if write is None:
        write = ptd.get("cache_creation_tokens")
    if write is None:
        write = usage.get("cacheWriteInputTokens")  # Bedrock Converse naming
    return read, write


def _provide_telemetry_for(eff, latency_ms, ttft_ms, tokens_in, tokens_out, success):
    """johnny telemetry when the seat actually served (module-level twin of the chat
    endpoint's closure, for other endpoints)."""
    if eff is not None and eff.state_at_dispatch == "johnny_ready" and eff.johnny_seat:
        provide_telemetry({
            "seat": eff.johnny_seat, "ts": int(time.time()),
            "latency_ms": latency_ms, "ttft_ms": ttft_ms,
            "tokens_in": tokens_in, "tokens_out": tokens_out, "success": success,
        })


def _emit_summary(decision, model_field: str, gen_ms: int,
                  cache_read=None, cache_write=None, served=None) -> None:
    out = decision.classifier_outcome
    if decision.classifier_result:
        cls_part = (f"classified={decision.classifier_result.domain}"
                    f"/{decision.classifier_result.complexity}")
        if out is not None and out.classifier_used in ("cache", "inherited"):
            cls_part += f" ({out.classifier_used})"
    else:
        cls_part = "pinned" if decision.pinned_backend else "skipped"
    cls_lat = (
        decision.classifier_result.latency_ms if decision.classifier_result else None
    )
    pc_part = ""
    if cache_read or cache_write:
        pc_part = f", pc r/w {cache_read or 0}/{cache_write or 0}"
    target = decision.backend
    if served and served != decision.backend:
        target = f"{decision.backend} ⤳ {served}"  # dispatch fallback took the hop
    print(
        f"[router] req#{decision.request_id[:8]} model={model_field} "
        f"urgency={decision.urgency} {cls_part} → {target} "
        f"(cls {cls_lat}ms gen {gen_ms}ms{pc_part})",
        flush=True,
    )


def _emit_decision_warnings(decision) -> None:
    """Emit any one-off stdout markers about routing oddities for this request.

    Currently:
    - multimodal content routed to default_on_failure
    - classifier fallback fired (oversize / primary_error)
    """
    rid = decision.request_id[:8]
    if decision.multimodal:
        print(
            f"[router] req#{rid}: multimodal content detected; "
            f"routing to default_on_failure ({decision.backend})",
            flush=True,
        )
    out = decision.classifier_outcome
    if out is not None and out.fallback_reason is not None:
        target = out.classifier_used or "(both failed)"
        if out.fallback_reason == "oversize":
            reason_detail = f"oversize: {out.input_chars} chars"
        else:
            reason_detail = out.fallback_reason
        print(
            f"[router] req#{rid}: classifier fallback ({reason_detail}) → {target}",
            flush=True,
        )


def _extract_forwarded(body: dict[str, Any]) -> dict[str, Any]:
    """Pull through known chat-completion kwargs (excluding handled-explicitly ones)."""
    return {k: body[k] for k in _FORWARDED_PARAMS if k in body}


# Starlette encodes header values as latin-1 and raises on anything outside it. Reasons and
# classifier notes are prose written by humans and other tools, so they carry em dashes,
# smart quotes and middots. Fold them rather than let one character 500 every response.
_HEADER_FOLD = str.maketrans({
    "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00b7": "-",
})


def _header_safe(value: str) -> str:
    """An ASCII-only, single-line rendering of `value`, safe as an HTTP header value."""
    folded = str(value).translate(_HEADER_FOLD).replace("\n", " ").replace("\r", " ")
    return folded.encode("ascii", "ignore").decode("ascii").strip()


_RETRAIN_CACHE: dict[str, object] = {"mtime": None, "reason": None}


def _retrain_reason() -> str | None:
    """The retrain-needed reason if the drift monitor set the flag, else None. mtime-cached so
    it's a cheap stat per request (re-reads only when the flag file changes). Self-clearing:
    when the flag is removed (drift healthy again), this returns None and the header stops."""
    from saint.config import RETRAIN_FLAG_PATH
    path = os.path.expanduser(RETRAIN_FLAG_PATH)
    try:
        mt = os.path.getmtime(path)
    except OSError:
        _RETRAIN_CACHE["mtime"] = None
        _RETRAIN_CACHE["reason"] = None
        return None
    if mt != _RETRAIN_CACHE["mtime"]:
        try:
            reason = (json.load(open(path)).get("reason")
                      or "classifier drift — retrain suggested")
        except Exception:
            reason = "classifier drift — retrain suggested"
        _RETRAIN_CACHE["mtime"] = mt
        _RETRAIN_CACHE["reason"] = str(reason)[:200].replace("\n", " ")
    return _RETRAIN_CACHE["reason"]  # type: ignore[return-value]


def _route_headers(decision, served: str, eff) -> dict[str, str]:
    """Routing metadata surfaced on every response (OpenRouter-style, always on).
    Lets clients see why they got routed without reading the request log."""
    h = {
        "x-saint-request-id": decision.request_id,
        "x-saint-backend": served,
        "x-saint-urgency": decision.urgency,
    }
    retrain = _retrain_reason()
    if retrain:
        h["x-saint-retrain"] = retrain
    if decision.classifier_result:
        h["x-saint-domain"] = decision.classifier_result.domain
        h["x-saint-complexity"] = decision.classifier_result.complexity
    if decision.classifier_outcome and decision.classifier_outcome.classifier_used:
        h["x-saint-classifier"] = decision.classifier_outcome.classifier_used
    if decision.pinned_backend:
        h["x-saint-pinned"] = decision.pinned_backend
    if served != decision.backend:
        h["x-saint-decided"] = decision.backend  # dispatch fallback changed the server
    if eff is not None and eff.state_at_dispatch:
        h["x-saint-state"] = eff.state_at_dispatch
    return {k: _header_safe(v) for k, v in h.items()}




def _seat_maxlen(endpoint: str | None) -> int | None:
    """The seat's ACTUAL served context (vLLM max_model_len) from its /v1/models — the
    launched window, not the model's native maximum. None on any failure."""
    if not endpoint:
        return None
    import json as _json
    import urllib.request as _url
    probe = endpoint.replace("0.0.0.0", "127.0.0.1").rstrip("/")
    try:
        with _url.urlopen(f"{probe}/models", timeout=2) as r:
            data = _json.loads(r.read())
        return (data.get("data") or [{}])[0].get("max_model_len")
    except Exception:
        return None



class _MessageStartDedupe:
    """SSE relay filter: drop every `message_start` event after the first. litellm's
    Anthropic-Messages→chat-completions translation (OpenAI-compatible backends) emits the
    message_start event twice; Anthropic clients treat a second one as a new message.
    Splits on the blank line between SSE events; partial trailing bytes are held until
    the event completes (flush() at end-of-stream)."""

    def __init__(self):
        self._buf = b""
        self._seen_start = False

    def feed(self, chunk) -> bytes:
        data = chunk if isinstance(chunk, bytes) else str(chunk).encode()
        self._buf += data
        out = b""
        while b"\n\n" in self._buf:
            block, self._buf = self._buf.split(b"\n\n", 1)
            block += b"\n\n"
            if b"event: message_start" in block or b'"type": "message_start"' in block:
                if self._seen_start:
                    continue
                self._seen_start = True
            out += block
        return out

    def flush(self) -> bytes:
        tail, self._buf = self._buf, b""
        return tail


def build_app(cfg: Config, *, db_path: Path) -> FastAPI:
    app = FastAPI(title="saint", version="0.1.0")
    app.state.cfg = cfg
    app.state.db = open_db(db_path)
    app.state.resolver = build_resolver(cfg.johnny)  # None unless any backend is johnny-bound
    app.state.route_caches = RouteCaches(
        turns=(TTLCache(cfg.cache.turn_ttl_s, cfg.cache.turn_max_entries)
               if cfg.cache.turn_cache else None),
        conversations=(TTLCache(cfg.cache.conversation_ttl_s, cfg.cache.conversation_max_entries)
                       if cfg.cache.conversation_affinity else None),
    )
    app.state.breaker = Breaker(cfg.routing.breaker_failures, cfg.routing.breaker_cooldown_s)
    from collections import deque as _deque
    app.state.recent_decisions = _deque(maxlen=300)   # per-request routing outcomes for /decisions
    app.state.bedrock_state = BedrockRuntime()
    if cfg.has_bedrock:
        from saint.bedrock_auth import apply_bedrock_auth_patch
        apply_bedrock_auth_patch()

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        ids = ["saint-auto", "saint-explain", *(f"saint-{name}" for name in cfg.backends)]
        return {
            "object": "list",
            "data": [
                {"id": mid, "object": "model", "owned_by": "saint", "created": 0}
                for mid in ids
            ],
        }

    def _remember_decision(decision, *, served, session_id, model_field, latency_ms, usage, api,
                           eff=None, cache_read=None):
        """Keep the routing outcome of a request for clients that can't read response headers
        (Claude Code's status line polls GET /decisions/last?session=…)."""
        try:
            out = decision.classifier_outcome
            res = out.result if out else None
            app.state.recent_decisions.append({
                "ts": time.time(), "api": api, "request_id": decision.request_id[:8],
                "session_id": session_id, "model_field": model_field,
                "backend": served or decision.backend, "routed": decision.backend,
                "fallback": bool(served and served != decision.backend),
                "pinned": decision.pinned_backend, "urgency": decision.urgency,
                "domain": getattr(res, "domain", None), "complexity": getattr(res, "complexity", None),
                "classifier": out.classifier_used if out else None,
                "served_model": (eff.backend.model if eff is not None and getattr(eff, "backend", None)
                                 else (cfg.backends.get(served).model if served and cfg.backends.get(served) else None)),
                "cache_read": cache_read if cache_read is not None else (usage or {}).get("cache_read_input_tokens"),
                "latency_ms": latency_ms,
                "tokens_in": (usage or {}).get("input_tokens") or (usage or {}).get("prompt_tokens"),
                "tokens_out": (usage or {}).get("output_tokens") or (usage or {}).get("completion_tokens"),
            })
        except Exception:
            pass

    @app.get("/decisions/last")
    async def decisions_last(session: str | None = None) -> dict[str, Any]:
        """The newest routing decision, optionally for one client session (substring match on
        the session id SAINT saw — Claude Code's metadata.user_id embeds its session_id)."""
        items = list(app.state.recent_decisions)
        if session:
            for d in reversed(items):
                if d.get("session_id") and session in str(d["session_id"]):
                    return {"match": "session", **d}
        return {"match": "latest", **items[-1]} if items else {"match": "none"}

    @app.get("/decisions/recent")
    async def decisions_recent(n: int = 20) -> list[dict[str, Any]]:
        items = list(app.state.recent_decisions)
        return items[-max(1, min(n, 300)):]

    @app.get("/status")
    async def status() -> dict[str, Any]:
        """The full 'available choices' map for clients (e.g. input): every routable backend
        with its kind, live johnny state, model, context window, price, and a cost rank.
        Local seats are resolved live via johnny; cloud tiers come from config/auto-config."""
        import json as _json
        import subprocess as _sp

        # johnny inventory (best-effort): model -> {context, state} + the active profile
        inv: dict[str, dict[str, Any]] = {}
        profile = None
        all_gpus: set = set()
        try:
            p = _sp.run(["johnny", "status", "--json"], capture_output=True, text=True, timeout=5)
            if p.returncode == 0:
                jd = _json.loads(p.stdout)
                profile = jd.get("profile")
                for s in jd.get("seats", []):
                    all_gpus.update(s.get("gpus") or [])
                    if s.get("model"):
                        inv[s["model"]] = {"context": s.get("native_context"),
                                           "state": s.get("state"), "gpus": s.get("gpus") or []}
        except Exception:
            pass

        # Electricity pricing for local seats: seat_watts × $/kWh ÷ (tok/s × 3.6) = $/Mtok.
        # seat_watts = its GPUs at full load + a per-seat share of the host's non-GPU base.
        # tok/s is measured from SAINT's own request log (johnny-bench perf could override once
        # populated). Inputs live in [energy] config — nothing hardcoded.
        en = cfg.energy
        total_gpus = len(all_gpus) or 1
        host_base = max(0.0, en.host_watts - total_gpus * en.gpu_watts)
        # Sustained throughput per local seat: p75 of the per-response decode rate over
        # SUBSTANTIAL responses (tokens_out>=128 amortizes TTFT; p75 trims queue-slowed
        # outliers) — reflects the decode rate you actually see, not a mean dragged down by
        # short replies. Needs >=5 samples, else the seat shows FREE without a cost.
        measured: dict[str, float] = {}
        try:
            by_bk: dict[str, list] = {}
            for bk, tok, lat in app.state.db.execute(
                "SELECT backend_chosen, tokens_out, backend_latency_ms FROM requests "
                "WHERE backend_chosen LIKE 'local%' AND tokens_out>=128 AND backend_latency_ms>0"):
                by_bk.setdefault(bk, []).append(tok / (lat / 1000.0))
            for bk, rates in by_bk.items():
                if len(rates) >= 5:
                    rates.sort()
                    measured[bk] = rates[min(len(rates) - 1, int(len(rates) * 0.75))]
        except Exception:
            pass

        resolver = app.state.resolver
        choices: list[dict[str, Any]] = [
            {"id": "saint-auto", "kind": "router", "label": "auto", "aliases": [],
             "state": "ready", "context": None, "price_in": None, "price_out": None, "rank": 0},
        ]
        for name, b in cfg.backends.items():
            if name in ("local-embed", "local-classifier"):
                continue                              # infra seats, not chat choices
            e: dict[str, Any] = {"id": f"saint-{name}", "backend": name,
                                 "aliases": list(b.aliases)}
            if b.johnny_bound:
                res = resolver.resolve(b.johnny_target) if resolver else None
                model = res.model if (res and res.model) else b.model
                ep = res.endpoint if res else b.base_url
                ctx = _seat_maxlen(ep)                    # actual served window (max_model_len)
                if ctx is None:                           # fallback: johnny native, then static
                    ctx = inv.get(model, {}).get("context") if model else b.context
                n_gpus = len(inv.get(model, {}).get("gpus") or [])
                tok_s = measured.get(name)                # measured throughput for this seat
                seat_watts = n_gpus * en.gpu_watts + host_base * n_gpus / total_gpus
                elec = round(seat_watts * en.price_kwh / (tok_s * 3.6), 3) if tok_s else None
                e.update(kind="local", role=b.johnny_target, model=model, endpoint=ep,
                         state=(res.state if res else "absent"),
                         eta_s=(res.eta_s if res else None), context=ctx,
                         gpus=n_gpus, tok_s=(round(tok_s, 1) if tok_s else None),
                         price_in=0.0, price_out=0.0, elec_per_mtok=elec, rank=1)
            else:
                e.update(kind=("cloud" if b.provider == "anthropic" else "backend"),
                         model=b.model, state="ready", context=b.context,
                         price_in=b.price_in, price_out=b.price_out,
                         rank=100 + (b.price_in or 0.0))   # cost-ascending for hotkeys
            choices.append(e)
        return {"profile": profile, "choices": choices}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        body = await request.json()
        messages = body.get("messages", [])
        model_field = body.get("model", SAINT_AUTO)
        stream = bool(body.get("stream", False))
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        extra = _extract_forwarded(body)

        # session_id (body field or x-session-id header) keys conversation affinity;
        # it is never forwarded upstream (absent from _FORWARDED_PARAMS).
        session_id = request.headers.get("x-session-id") or body.get("session_id")

        # Reserved savings report — returned as the completion, no routing/LLM/cost (like
        # saint-explain). Two triggers, both a "prompt-time hint" the caller controls:
        #   - model field:  "saint-savings" or "saint-savings:<period>"
        #   - message text:  a line like "/savings", "/savings week", "saint savings month"
        _sv = _savings_trigger(model_field, messages)
        if _sv is not None:
            rep = _savings.compute(app.state.db, cfg, period=_sv)
            return _explain_response(_savings.render(rep, color=True), model_field)

        try:
            from saint.anthropic_api import hoist_routing_directive
            messages = hoist_routing_directive(messages)   # `::coder …` works where `!` is taken
            body["messages"] = messages
            decision = await decide_route(
                cfg=cfg, model_field=model_field, messages=messages,
                caches=app.state.route_caches, session_id=session_id,
            )
        except UnknownPrefixError as e:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(e), "type": "invalid_request_error"}},
            )

        _emit_decision_warnings(decision)

        if decision.mode == "explain":
            text = format_decision(decision, cfg, resolver=app.state.resolver)
            return _explain_response(text, model_field)

        breaker = app.state.breaker
        candidates = dispatch_candidates(cfg, decision.backend, breaker)

        def _provide(eff, latency_ms, ttft_ms, tokens_in, tokens_out, success):
            # Provide telemetry only when the johnny seat actually served the request.
            if eff.state_at_dispatch == "johnny_ready" and eff.johnny_seat:
                provide_telemetry({
                    "seat": eff.johnny_seat, "ts": int(time.time()),
                    "latency_ms": latency_ms, "ttft_ms": ttft_ms,
                    "tokens_in": tokens_in, "tokens_out": tokens_out, "success": success,
                })

        def _log_state(eff, served: str) -> str | None:
            return "dispatch_fallback" if served != decision.backend else eff.state_at_dispatch

        rid = decision.request_id[:8]

        if stream:
            from fastapi.responses import StreamingResponse

            from saint.router import dispatch_streaming

            # Acquire the stream + first chunk inside the attempt: failover is only
            # possible BEFORE anything is flushed to the client. Mid-stream errors keep
            # the existing in-stream error-chunk behavior.
            started = time.monotonic()

            async def _attempt_stream(eff_c):
                upstream_c = await dispatch_streaming(
                    cfg, decision, messages,
                    tools=tools, tool_choice=tool_choice,
                    extra_params=extra or None,
                    effective_backend=eff_c.backend,
                )
                try:
                    first = await upstream_c.__anext__()
                except StopAsyncIteration:
                    first = None  # empty-but-successful stream
                return upstream_c, first

            result, error_kind, _ = await run_candidates(
                cfg=cfg, candidates=candidates, rid=rid,
                resolver=app.state.resolver, breaker=breaker, attempt=_attempt_stream,
                bedrock_state=app.state.bedrock_state,
            )

            if result is None:
                _safe_log(app.state.db, build_log_row(
                    decision=decision, model_field=model_field,
                    backend_latency_ms=int((time.monotonic() - started) * 1000),
                    success=False, error_kind=error_kind,
                    tokens_in=None, tokens_out=None,
                    prompt_storage_mode=cfg.logging.prompt_storage,
                ))
                raise HTTPException(status_code=502, detail=f"backend error: {error_kind}")

            upstream, first_chunk = result.value
            first_chunk_ts = time.monotonic() if first_chunk is not None else None
            eff_final, served_final = result.eff, result.backend

            async def gen():
                tokens_in_total = 0
                tokens_out_total = 0
                cache_read_total = 0
                cache_write_total = 0
                error_kind_local: str | None = None
                success_local = False
                import json as _json

                def _emit(chunk):
                    nonlocal tokens_in_total, tokens_out_total, cache_read_total, cache_write_total
                    if hasattr(chunk, "model_dump"):
                        chunk_dict = chunk.model_dump()
                    elif isinstance(chunk, dict):
                        chunk_dict = chunk
                    else:
                        chunk_dict = dict(chunk)
                    usage = chunk_dict.get("usage") or {}
                    tokens_in_total = max(tokens_in_total, usage.get("prompt_tokens", 0) or 0)
                    tokens_out_total = max(tokens_out_total, usage.get("completion_tokens", 0) or 0)
                    c_read, c_write = _cache_tokens(usage)
                    cache_read_total = max(cache_read_total, c_read or 0)
                    cache_write_total = max(cache_write_total, c_write or 0)
                    return f"data: {_json.dumps(chunk_dict)}\n\n"

                try:
                    if first_chunk is not None:
                        yield _emit(first_chunk)
                        async for chunk in upstream:
                            yield _emit(chunk)
                    yield "data: [DONE]\n\n"
                    success_local = True
                except Exception as e:
                    success_local = False
                    error_kind_local = type(e).__name__
                    import json as _json
                    err_payload = {"error": {"type": error_kind_local, "message": str(e)}}
                    yield f"data: {_json.dumps(err_payload)}\n\n"
                finally:
                    elapsed_ms = int((time.monotonic() - started) * 1000)
                    ttft_ms = int((first_chunk_ts - started) * 1000) if first_chunk_ts else None
                    _safe_log(app.state.db, build_log_row(
                        decision=decision, model_field=model_field,
                        backend_latency_ms=elapsed_ms,
                        success=success_local, error_kind=error_kind_local,
                        tokens_in=tokens_in_total or None,
                        tokens_out=tokens_out_total or None,
                        prompt_storage_mode=cfg.logging.prompt_storage,
                        johnny_seat=eff_final.johnny_seat,
                        state_at_dispatch=_log_state(eff_final, served_final),
                        cache_read_tokens=cache_read_total or None,
                        cache_write_tokens=cache_write_total or None,
                        backend_override=served_final,
                    ))
                    _provide(eff_final, elapsed_ms, ttft_ms, tokens_in_total or None,
                             tokens_out_total or None, success_local)
                    _emit_summary(decision, model_field, elapsed_ms,
                                  cache_read=cache_read_total or None,
                                  cache_write=cache_write_total or None,
                                  served=served_final)

            return StreamingResponse(gen(), media_type="text/event-stream",
                                     headers=_route_headers(decision, served_final, eff_final))

        started = time.monotonic()

        async def _attempt(eff_c):
            return await dispatch_non_streaming(
                cfg, decision, messages, tools=tools, tool_choice=tool_choice,
                extra_params=extra or None, effective_backend=eff_c.backend,
            )

        result, error_kind, last_eff = await run_candidates(
            cfg=cfg, candidates=candidates, rid=rid,
            resolver=app.state.resolver, breaker=breaker, attempt=_attempt,
            bedrock_state=app.state.bedrock_state,
        )
        success = result is not None
        served = result.backend if result else None
        eff = result.eff if result else last_eff
        response = result.value if result else None

        backend_latency_ms = int((time.monotonic() - started) * 1000)

        usage = _usage_dict(response)
        cache_read, cache_write = _cache_tokens(usage)
        _remember_decision(decision, served=(result.backend if result else None),
                           session_id=session_id, model_field=model_field,
                           latency_ms=backend_latency_ms, usage=usage, api="chat",
                           eff=eff, cache_read=cache_read)
        _safe_log(app.state.db, build_log_row(
            decision=decision, model_field=model_field,
            backend_latency_ms=backend_latency_ms,
            success=success, error_kind=None if success else error_kind,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            prompt_storage_mode=cfg.logging.prompt_storage,
            johnny_seat=eff.johnny_seat if eff else None,
            state_at_dispatch=_log_state(eff, served) if served else None,
            cache_read_tokens=cache_read, cache_write_tokens=cache_write,
            backend_override=served,
        ))
        # non-streaming: TTFT not separable -> total latency only
        if eff is not None:
            _provide(eff, backend_latency_ms, None, usage.get("prompt_tokens"),
                     usage.get("completion_tokens"), success)
        _emit_summary(decision, model_field, backend_latency_ms,
                      cache_read=cache_read, cache_write=cache_write, served=served)

        if not success:
            raise HTTPException(status_code=502, detail=f"backend error: {error_kind}")
        payload = response if isinstance(response, dict) else response.model_dump()
        return JSONResponse(content=payload,
                            headers=_route_headers(decision, served, eff))

    @app.post("/v1/messages")
    async def anthropic_messages_endpoint(request: Request) -> Any:
        from saint.anthropic_api import (
            anthropic_error,
            anthropic_usage,
            build_dispatch_params,
            detect_multimodal,
            last_user_text,
            messages_model_field,
            openai_view,
            strip_prefix_from_messages,
        )
        from saint.backends import call_backend_messages

        body = await request.json()
        client_model = body.get("model") or ""
        if body.get("max_tokens") is None:
            return anthropic_error(400, "invalid_request_error", "max_tokens is required")
        from saint.anthropic_api import hoist_routing_directive
        if os.environ.get("SAINT_DEBUG_LAST_USER"):   # shape of the client's last user turn (directive debugging)
            for _m in reversed(body.get("messages", [])):
                if _m.get("role") == "user":
                    _c = _m.get("content")
                    _shape = repr(_c[:300]) if isinstance(_c, str) else [
                        (b.get("type"), (b.get("text") or "")[:120]) for b in _c if isinstance(b, dict)]
                    print(f"[router] last-user shape: {_shape}", file=sys.stderr, flush=True)
                    break
        body["messages"] = hoist_routing_directive(body.get("messages", []))   # `::opus` → `@opus` up front
        stream = bool(body.get("stream", False))

        session_id = (request.headers.get("x-session-id")
                      or (body.get("metadata") or {}).get("user_id"))
        try:
            decision = await decide_route(
                cfg=cfg, model_field=messages_model_field(cfg, client_model),
                messages=openai_view(body),
                caches=app.state.route_caches, session_id=session_id,
                multimodal_override=detect_multimodal(body.get("messages", [])),
            )
        except UnknownPrefixError as e:
            return anthropic_error(400, "invalid_request_error", str(e))

        _emit_decision_warnings(decision)

        if decision.mode == "explain":
            text = format_decision(decision, cfg, resolver=app.state.resolver)
            return JSONResponse(content={
                "id": f"msg_{decision.request_id[:12]}", "type": "message",
                "role": "assistant", "model": client_model,
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            })

        breaker = app.state.breaker
        candidates = dispatch_candidates(cfg, decision.backend, breaker)
        params = build_dispatch_params(body)
        if decision.parsed.raw:  # strip !/@ routing prefixes before the backend sees them
            params["messages"] = strip_prefix_from_messages(params["messages"],
                                                            decision.parsed.raw)
        rid = decision.request_id[:8]

        def _params_for(eff_c):
            # Signed thinking is HMAC-bound to the minting model; strip on a backend switch.
            from saint.thinking import (fold_system_role_messages, sanitize_thinking_blocks,
                                        shape_thinking_param, should_strip, strip_signed_thinking)
            provider = eff_c.backend.provider or ""
            p = shape_thinking_param(params, provider, eff_c.backend.model or "")
            msgs = p["messages"]
            if should_strip(provider, decision.prev_backend, decision.backend):
                msgs = strip_signed_thinking(msgs)
            if provider in ("anthropic", "bedrock"):
                msgs = sanitize_thinking_blocks(msgs)   # empty/unsigned blocks 400 at Anthropic
                msgs = fold_system_role_messages(msgs, provider, eff_c.backend.model or "")
            out_p = p if msgs is p["messages"] else {**p, "messages": msgs}
            if os.environ.get("SAINT_DEBUG_DISPATCH"):   # exact outbound Messages params, for shape bugs
                try:
                    import json as _j
                    with open(f"/tmp/saint-dispatch-{rid}.json", "w") as fh:
                        _j.dump({"backend": eff_c.backend.name, "params": out_p}, fh, default=str)
                except Exception:
                    pass
            return out_p

        def _refresh_affinity(served: str):
            # Record the backend that actually minted this turn's thinking, so later
            # turns (even pinned ones, which decide_route leaves label-less) know the
            # continuity backend for the thinking-signature guard.
            caches = app.state.route_caches
            if not decision.conversation_key or caches.conversations is None:
                return
            from saint.route_cache import ConversationEntry
            entry = caches.conversations.get(decision.conversation_key)
            if entry is None:
                caches.conversations.set(decision.conversation_key,
                                         ConversationEntry(backend=served))
            elif entry.backend != served:
                from dataclasses import replace as _replace
                caches.conversations.set(decision.conversation_key,
                                         _replace(entry, backend=served))

        def _log_state(eff, served: str) -> str | None:
            return "dispatch_fallback" if served != decision.backend else eff.state_at_dispatch

        def _log_messages_row(*, served, eff, latency_ms, success, error_kind, usage,
                              backend_override):
            _remember_decision(decision, served=served, session_id=session_id,
                               model_field=client_model, latency_ms=latency_ms, usage=usage,
                               api="messages", eff=eff)
            _safe_log(app.state.db, build_log_row(
                decision=decision, model_field=client_model or "anthropic-messages",
                backend_latency_ms=latency_ms,
                success=success, error_kind=error_kind,
                tokens_in=usage.get("input_tokens"),
                tokens_out=usage.get("output_tokens"),
                prompt_storage_mode=cfg.logging.prompt_storage,
                johnny_seat=eff.johnny_seat if eff else None,
                state_at_dispatch=_log_state(eff, served) if served else None,
                cache_read_tokens=usage.get("cache_read_input_tokens"),
                cache_write_tokens=usage.get("cache_creation_input_tokens"),
                backend_override=backend_override,
            ))

        started = time.monotonic()

        if stream:
            from fastapi.responses import StreamingResponse

            from saint.anthropic_api import SseUsageTracker

            async def _attempt_stream(eff_c):
                upstream_c = await call_backend_messages(eff_c.backend,
                                                         params=_params_for(eff_c),
                                                         stream=True)
                it = upstream_c.__aiter__()
                try:
                    first = await it.__anext__()
                except StopAsyncIteration:
                    first = None
                return it, first

            result, error_kind, last_eff = await run_candidates(
                cfg=cfg, candidates=candidates, rid=rid,
                resolver=app.state.resolver, breaker=breaker, attempt=_attempt_stream,
                bedrock_state=app.state.bedrock_state,
            )
            if result is None:
                _log_messages_row(served=None, eff=last_eff,
                                  latency_ms=int((time.monotonic() - started) * 1000),
                                  success=False, error_kind=error_kind, usage={},
                                  backend_override=None)
                return anthropic_error(502, "api_error", f"backend error: {error_kind}")

            upstream, first_chunk = result.value
            first_chunk_ts = time.monotonic() if first_chunk is not None else None

            async def gen():
                tracker = SseUsageTracker()
                dedupe = _MessageStartDedupe()   # litellm's messages→completion translation emits
                                                 # message_start twice; Anthropic clients expect one
                success_local = False
                error_kind_local: str | None = None
                try:
                    if first_chunk is not None:
                        tracker.feed(first_chunk)
                        out0 = dedupe.feed(first_chunk)
                        if out0:
                            yield out0
                        async for chunk in upstream:
                            tracker.feed(chunk)
                            out = dedupe.feed(chunk)   # ORIGINAL bytes, minus a repeated message_start
                            if out:
                                yield out
                        tail = dedupe.flush()
                        if tail:
                            yield tail
                    success_local = True
                except Exception as e:
                    success_local = False
                    error_kind_local = type(e).__name__
                    import json as _json
                    err = {"type": "error",
                           "error": {"type": "api_error", "message": str(e)}}
                    yield f"event: error\ndata: {_json.dumps(err)}\n\n".encode()
                finally:
                    elapsed_ms = int((time.monotonic() - started) * 1000)
                    ttft = (int((first_chunk_ts - started) * 1000)
                            if first_chunk_ts else None)
                    if success_local:
                        _refresh_affinity(result.backend)
                    _log_messages_row(served=result.backend, eff=result.eff,
                                      latency_ms=elapsed_ms, success=success_local,
                                      error_kind=error_kind_local, usage=tracker.usage,
                                      backend_override=result.backend)
                    _provide_telemetry_for(result.eff, elapsed_ms, ttft,
                                           tracker.usage.get("input_tokens"),
                                           tracker.usage.get("output_tokens"),
                                           success_local)
                    _emit_summary(decision, client_model or "anthropic-messages",
                                  elapsed_ms,
                                  cache_read=tracker.usage.get("cache_read_input_tokens"),
                                  cache_write=tracker.usage.get("cache_creation_input_tokens"),
                                  served=result.backend)

            return StreamingResponse(
                gen(), media_type="text/event-stream",
                headers=_route_headers(decision, result.backend, result.eff))

        async def _attempt(eff_c):
            return await call_backend_messages(eff_c.backend, params=_params_for(eff_c),
                                               stream=False)

        result, error_kind, last_eff = await run_candidates(
            cfg=cfg, candidates=candidates, rid=rid,
            resolver=app.state.resolver, breaker=breaker, attempt=_attempt,
            bedrock_state=app.state.bedrock_state,
        )
        latency_ms = int((time.monotonic() - started) * 1000)

        if result is None:
            _log_messages_row(served=None, eff=last_eff, latency_ms=latency_ms,
                              success=False, error_kind=error_kind, usage={},
                              backend_override=None)
            return anthropic_error(502, "api_error", f"backend error: {error_kind}")

        usage = anthropic_usage(result.value)
        _refresh_affinity(result.backend)
        _log_messages_row(served=result.backend, eff=result.eff, latency_ms=latency_ms,
                          success=True, error_kind=None, usage=usage,
                          backend_override=result.backend)
        _provide_telemetry_for(result.eff, latency_ms, None, usage.get("input_tokens"),
                               usage.get("output_tokens"), True)
        _emit_summary(decision, client_model or "anthropic-messages", latency_ms,
                      cache_read=usage.get("cache_read_input_tokens"),
                      cache_write=usage.get("cache_creation_input_tokens"),
                      served=result.backend)
        payload = (result.value if isinstance(result.value, dict)
                   else result.value.model_dump())
        return JSONResponse(content=payload,
                            headers=_route_headers(decision, result.backend, result.eff))

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request) -> Any:
        """Anthropic count_tokens for Anthropic-native clients (Claude Code calls it for
        context accounting and 404s otherwise). Counts with the local chat seat's tokenizer
        (vLLM /tokenize) — a close-enough estimate for every backend — and falls back to a
        chars/4 estimate when no local seat answers. Never routes or bills."""
        from saint.anthropic_api import anthropic_error, openai_view
        from saint.config import resolve_backend
        body = await request.json()
        msgs = openai_view(body)
        for t in body.get("tools") or []:      # tool schemas are part of the prompt
            msgs.append({"role": "system", "content": json.dumps(t)[:20000]})
        text_len = sum(len(m["content"]) if isinstance(m.get("content"), str)
                       else len(json.dumps(m.get("content") or "")) for m in msgs)
        estimate = max(1, text_len // 4)
        for name in ("local-chat", "local-coder"):
            b = resolve_backend(cfg, name)
            if b is None or not b.base_url:
                continue
            try:
                import httpx
                url = b.base_url.rstrip("/") + "/tokenize"
                async with httpx.AsyncClient(timeout=10.0) as client:
                    plain = [{"role": m["role"], "content": m["content"] if isinstance(m.get("content"), str)
                              else json.dumps(m.get("content") or "")} for m in msgs]
                    r = await client.post(url, json={"model": b.model, "messages": plain})
                    if r.status_code == 200 and isinstance(r.json().get("count"), int):
                        return JSONResponse(content={"input_tokens": int(r.json()["count"])})
            except Exception:
                continue
        return JSONResponse(content={"input_tokens": estimate})

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Any:
        eb_name = cfg.routing.embeddings_backend
        if not eb_name:
            return JSONResponse(status_code=404, content={"error": {
                "message": "routing.embeddings_backend is not configured",
                "type": "invalid_request_error"}})
        body = await request.json()
        eff = resolve_for_dispatch(cfg, eb_name, app.state.resolver)
        started = time.monotonic()
        try:
            response = await call_embeddings(eff.backend, body.get("input"))
            success, error_kind = True, None
        except Exception as e:
            success, error_kind = False, type(e).__name__
            response = None
        latency_ms = int((time.monotonic() - started) * 1000)
        usage = _usage_dict(response)
        _safe_log(app.state.db, LogRow(
            request_id=str(__import__("uuid").uuid4()), model_field="saint-embeddings",
            prefixes_raw=None, pinned_backend=None, urgency_used="normal",
            classifier_used=None, classifier_fallback_reason=None,
            classifier_input_chars=None, classifier_input_truncated_from=None,
            classifier_latency_ms=None, classifier_domain=None,
            classifier_complexity=None, classifier_reason=None,
            backend_chosen=eb_name, backend_latency_ms=latency_ms,
            tokens_in=usage.get("prompt_tokens"), tokens_out=None,
            success=success, error_kind=error_kind,
            prompt_content=None,  # embedding inputs are bulky and never training data
            prompt_storage_mode="none",
            johnny_seat=eff.johnny_seat, state_at_dispatch=eff.state_at_dispatch,
        ))
        if not success:
            raise HTTPException(status_code=502, detail=f"backend error: {error_kind}")
        payload = response if isinstance(response, dict) else response.model_dump()
        return JSONResponse(content=payload, headers={
            "x-saint-backend": eb_name,
            "x-saint-state": eff.state_at_dispatch or "static",
        })

    return app
