from __future__ import annotations

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
from saint.backends import call_embeddings
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


def _route_headers(decision, served: str, eff) -> dict[str, str]:
    """Routing metadata surfaced on every response (OpenRouter-style, always on).
    Lets clients see why they got routed without reading the request log."""
    h = {
        "x-saint-request-id": decision.request_id,
        "x-saint-backend": served,
        "x-saint-urgency": decision.urgency,
    }
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
    return h


def _dispatch_candidates(cfg: Config, primary: str, breaker) -> list[str]:
    """Primary plus its one-hop on_error fallback. The fallback's own on_error is never
    followed (loop-proof). An open breaker skips the primary when a fallback exists."""
    fb = cfg.backends[primary].on_error
    candidates = [primary] + ([fb] if fb and fb != primary else [])
    if len(candidates) > 1 and breaker.is_open(primary):
        print(f"[router] breaker open on '{primary}' — dispatching straight to '{fb}'",
              flush=True)
        return candidates[1:]
    return candidates


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

        try:
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
        candidates = _dispatch_candidates(cfg, decision.backend, breaker)

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

            # Acquire the stream + first chunk inside the candidate loop: failover is only
            # possible BEFORE anything is flushed to the client. Mid-stream errors keep the
            # existing in-stream error-chunk behavior.
            started = time.monotonic()
            upstream = first_chunk = None
            eff = served = None
            error_kind: str | None = None
            for cand in candidates:
                eff = resolve_for_dispatch(cfg, cand, app.state.resolver)
                attempts = 2 if cfg.routing.retry_same_backend else 1
                for _ in range(attempts):
                    try:
                        upstream = await dispatch_streaming(
                            cfg, decision, messages,
                            tools=tools, tool_choice=tool_choice,
                            extra_params=extra or None,
                            effective_backend=eff.backend,
                        )
                        try:
                            first_chunk = await upstream.__anext__()
                        except StopAsyncIteration:
                            first_chunk = None  # empty-but-successful stream
                        breaker.record_success(cand)
                        served = cand
                        break
                    except Exception as e:
                        error_kind = type(e).__name__
                        breaker.record_failure(cand)
                        print(f"[router] req#{rid}: dispatch to '{cand}' failed "
                              f"({error_kind}) — {'retrying' if _ == 0 and attempts == 2 else 'moving on'}",
                              file=sys.stderr, flush=True)
                        upstream = None
                if served is not None:
                    break

            if served is None:
                _safe_log(app.state.db, build_log_row(
                    decision=decision, model_field=model_field,
                    backend_latency_ms=int((time.monotonic() - started) * 1000),
                    success=False, error_kind=error_kind,
                    tokens_in=None, tokens_out=None,
                    prompt_storage_mode=cfg.logging.prompt_storage,
                ))
                raise HTTPException(status_code=502, detail=f"backend error: {error_kind}")

            first_chunk_ts = time.monotonic() if first_chunk is not None else None
            eff_final, served_final = eff, served

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
        response = None
        eff = served = None
        error_kind: str | None = None
        for cand in candidates:
            eff = resolve_for_dispatch(cfg, cand, app.state.resolver)
            attempts = 2 if cfg.routing.retry_same_backend else 1
            for _ in range(attempts):
                try:
                    response = await dispatch_non_streaming(
                        cfg, decision, messages, tools=tools, tool_choice=tool_choice,
                        extra_params=extra or None, effective_backend=eff.backend,
                    )
                    breaker.record_success(cand)
                    served = cand
                    break
                except Exception as e:
                    error_kind = type(e).__name__
                    breaker.record_failure(cand)
                    print(f"[router] req#{rid}: dispatch to '{cand}' failed "
                          f"({error_kind}) — {'retrying' if _ == 0 and attempts == 2 else 'moving on'}",
                          file=sys.stderr, flush=True)
            if served is not None:
                break
        success = served is not None

        backend_latency_ms = int((time.monotonic() - started) * 1000)

        usage = _usage_dict(response)
        cache_read, cache_write = _cache_tokens(usage)
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
