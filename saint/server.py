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
from saint.route_cache import RouteCaches, TTLCache
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
    write = usage.get("cache_creation_input_tokens")
    if write is None:
        write = ptd.get("cache_creation_tokens")
    return read, write


def _emit_summary(decision, model_field: str, gen_ms: int,
                  cache_read=None, cache_write=None) -> None:
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
    print(
        f"[router] req#{decision.request_id[:8]} model={model_field} "
        f"urgency={decision.urgency} {cls_part} → {decision.backend} "
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

        # johnny override overlay: resolve the chosen backend's live endpoint/liveness.
        # (No-op for unbound backends; degrades to the static baseline if johnny is down.)
        eff = resolve_for_dispatch(cfg, decision.backend, app.state.resolver)

        def _provide(latency_ms, ttft_ms, tokens_in, tokens_out, success):
            # Provide telemetry only when the johnny seat actually served the request.
            if eff.state_at_dispatch == "johnny_ready" and eff.johnny_seat:
                provide_telemetry({
                    "seat": eff.johnny_seat, "ts": int(time.time()),
                    "latency_ms": latency_ms, "ttft_ms": ttft_ms,
                    "tokens_in": tokens_in, "tokens_out": tokens_out, "success": success,
                })

        if stream:
            from fastapi.responses import StreamingResponse

            from saint.router import dispatch_streaming

            async def gen():
                started_local = time.monotonic()
                first_chunk_ts: float | None = None
                tokens_in_total = 0
                tokens_out_total = 0
                cache_read_total = 0
                cache_write_total = 0
                error_kind_local: str | None = None
                success_local = False
                try:
                    upstream = await dispatch_streaming(
                        cfg, decision, messages,
                        tools=tools, tool_choice=tool_choice,
                        extra_params=extra or None,
                        effective_backend=eff.backend,
                    )
                    async for chunk in upstream:
                        if first_chunk_ts is None:
                            first_chunk_ts = time.monotonic()  # TTFT
                        if hasattr(chunk, "model_dump"):
                            chunk_dict = chunk.model_dump()
                        elif isinstance(chunk, dict):
                            chunk_dict = chunk
                        else:
                            chunk_dict = dict(chunk)
                        usage = chunk_dict.get("usage") or {}
                        tokens_in_total = max(
                            tokens_in_total, usage.get("prompt_tokens", 0) or 0
                        )
                        tokens_out_total = max(
                            tokens_out_total, usage.get("completion_tokens", 0) or 0
                        )
                        c_read, c_write = _cache_tokens(usage)
                        cache_read_total = max(cache_read_total, c_read or 0)
                        cache_write_total = max(cache_write_total, c_write or 0)
                        import json as _json
                        yield f"data: {_json.dumps(chunk_dict)}\n\n"
                    yield "data: [DONE]\n\n"
                    success_local = True
                except Exception as e:
                    success_local = False
                    error_kind_local = type(e).__name__
                    import json as _json
                    err_payload = {"error": {"type": error_kind_local, "message": str(e)}}
                    yield f"data: {_json.dumps(err_payload)}\n\n"
                finally:
                    elapsed_ms = int((time.monotonic() - started_local) * 1000)
                    ttft_ms = int((first_chunk_ts - started_local) * 1000) if first_chunk_ts else None
                    _safe_log(app.state.db, build_log_row(
                        decision=decision, model_field=model_field,
                        backend_latency_ms=elapsed_ms,
                        success=success_local, error_kind=error_kind_local,
                        tokens_in=tokens_in_total or None,
                        tokens_out=tokens_out_total or None,
                        prompt_storage_mode=cfg.logging.prompt_storage,
                        johnny_seat=eff.johnny_seat, state_at_dispatch=eff.state_at_dispatch,
                        cache_read_tokens=cache_read_total or None,
                        cache_write_tokens=cache_write_total or None,
                    ))
                    _provide(elapsed_ms, ttft_ms, tokens_in_total or None, tokens_out_total or None, success_local)
                    _emit_summary(decision, model_field, elapsed_ms,
                                  cache_read=cache_read_total or None,
                                  cache_write=cache_write_total or None)

            return StreamingResponse(gen(), media_type="text/event-stream")

        started = time.monotonic()
        try:
            response = await dispatch_non_streaming(
                cfg, decision, messages, tools=tools, tool_choice=tool_choice,
                extra_params=extra or None, effective_backend=eff.backend,
            )
            success, error_kind = True, None
        except Exception as e:
            success, error_kind = False, type(e).__name__
            response = None

        backend_latency_ms = int((time.monotonic() - started) * 1000)

        usage = _usage_dict(response)
        cache_read, cache_write = _cache_tokens(usage)
        _safe_log(app.state.db, build_log_row(
            decision=decision, model_field=model_field,
            backend_latency_ms=backend_latency_ms,
            success=success, error_kind=error_kind,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            prompt_storage_mode=cfg.logging.prompt_storage,
            johnny_seat=eff.johnny_seat, state_at_dispatch=eff.state_at_dispatch,
            cache_read_tokens=cache_read, cache_write_tokens=cache_write,
        ))
        # non-streaming: TTFT not separable -> total latency only
        _provide(backend_latency_ms, None, usage.get("prompt_tokens"), usage.get("completion_tokens"), success)
        _emit_summary(decision, model_field, backend_latency_ms,
                      cache_read=cache_read, cache_write=cache_write)

        if not success:
            raise HTTPException(status_code=502, detail=f"backend error: {error_kind}")
        return response

    return app
