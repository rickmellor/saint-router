from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from goorouter.config import Config
from goorouter.explain import format_decision
from goorouter.prefixes import UnknownPrefixError
from goorouter.router import GOO_AUTO, GOO_EXPLAIN, decide_route, dispatch_non_streaming
from goorouter.storage import LogRow, log_request, open_db


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


def _build_log_row(decision, model_field, backend_latency_ms, success, error_kind,
                    tokens_in, tokens_out, prompt_storage_mode):
    out = decision.classifier_outcome
    return LogRow(
        request_id=decision.request_id,
        model_field=model_field,
        prefixes_raw=decision.parsed.raw or None,
        pinned_backend=decision.pinned_backend,
        urgency_used=decision.urgency,
        classifier_used=out.classifier_used if out else None,
        classifier_fallback_reason=out.fallback_reason if out else None,
        classifier_input_chars=out.input_chars if out else None,
        classifier_input_truncated_from=out.input_truncated_from if out else None,
        classifier_latency_ms=(out.result.latency_ms if out and out.result else None),
        classifier_domain=(out.result.domain if out and out.result else None),
        classifier_complexity=(out.result.complexity if out and out.result else None),
        classifier_reason=(out.result.reason if out and out.result else None),
        backend_chosen=decision.backend,
        backend_latency_ms=backend_latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        success=success,
        error_kind=error_kind,
        prompt_content=decision.last_user_content_original,
        prompt_storage_mode=prompt_storage_mode,
    )


def _emit_summary(decision, model_field: str, gen_ms: int) -> None:
    cls_used = decision.classifier_outcome.classifier_used if decision.classifier_outcome else None
    cls_part = (
        f"classified={decision.classifier_result.domain}/{decision.classifier_result.complexity}"
        if decision.classifier_result else (
            "pinned" if decision.pinned_backend else "skipped"
        )
    )
    cls_lat = (
        decision.classifier_result.latency_ms if decision.classifier_result else None
    )
    print(
        f"[router] req#{decision.request_id[:8]} model={model_field} "
        f"urgency={decision.urgency} {cls_part} → {decision.backend} "
        f"(cls {cls_lat}ms gen {gen_ms}ms)",
        flush=True,
    )


def build_app(cfg: Config, *, db_path: Path) -> FastAPI:
    app = FastAPI(title="goorouter", version="0.1.0")
    app.state.cfg = cfg
    app.state.db = open_db(db_path)

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        ids = ["goo-auto", "goo-explain", *(f"goo-{name}" for name in cfg.backends)]
        return {
            "object": "list",
            "data": [
                {"id": mid, "object": "model", "owned_by": "goorouter", "created": 0}
                for mid in ids
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        body = await request.json()
        messages = body.get("messages", [])
        model_field = body.get("model", GOO_AUTO)
        stream = bool(body.get("stream", False))
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")

        try:
            decision = await decide_route(cfg=cfg, model_field=model_field, messages=messages)
        except UnknownPrefixError as e:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(e), "type": "invalid_request_error"}},
            )

        if decision.mode == "explain":
            text = format_decision(decision, cfg)
            return _explain_response(text, model_field)

        if stream:
            from fastapi.responses import StreamingResponse

            from goorouter.router import dispatch_streaming

            async def gen():
                started_local = time.monotonic()
                tokens_in_total = 0
                tokens_out_total = 0
                error_kind_local: str | None = None
                success_local = False
                try:
                    upstream = await dispatch_streaming(
                        cfg, decision, messages,
                        tools=tools, tool_choice=tool_choice,
                    )
                    async for chunk in upstream:
                        if hasattr(chunk, "model_dump"):
                            chunk_dict = chunk.model_dump()
                        elif isinstance(chunk, dict):
                            chunk_dict = chunk
                        else:
                            chunk_dict = dict(chunk)
                        usage = chunk_dict.get("usage") or {}
                        tokens_in_total = max(tokens_in_total, usage.get("prompt_tokens", 0) or 0)
                        tokens_out_total = max(tokens_out_total, usage.get("completion_tokens", 0) or 0)
                        import json as _json
                        yield f"data: {_json.dumps(chunk_dict)}\n\n"
                    yield "data: [DONE]\n\n"
                    success_local = True
                except Exception as e:
                    success_local = False
                    error_kind_local = type(e).__name__
                    import json as _json
                    yield f"data: {_json.dumps({'error': {'type': error_kind_local, 'message': str(e)}})}\n\n"
                finally:
                    log_request(app.state.db, _build_log_row(
                        decision=decision, model_field=model_field,
                        backend_latency_ms=int((time.monotonic() - started_local) * 1000),
                        success=success_local, error_kind=error_kind_local,
                        tokens_in=tokens_in_total or None,
                        tokens_out=tokens_out_total or None,
                        prompt_storage_mode=cfg.logging.prompt_storage,
                    ))
                    _emit_summary(decision, model_field, int((time.monotonic() - started_local) * 1000))

            return StreamingResponse(gen(), media_type="text/event-stream")

        started = time.monotonic()
        try:
            response = await dispatch_non_streaming(
                cfg, decision, messages, tools=tools, tool_choice=tool_choice,
            )
            success, error_kind = True, None
        except Exception as e:
            success, error_kind = False, type(e).__name__
            response = None

        backend_latency_ms = int((time.monotonic() - started) * 1000)

        usage = (response or {}).get("usage", {}) if isinstance(response, dict) else {}
        log_request(app.state.db, _build_log_row(
            decision=decision, model_field=model_field,
            backend_latency_ms=backend_latency_ms,
            success=success, error_kind=error_kind,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            prompt_storage_mode=cfg.logging.prompt_storage,
        ))
        _emit_summary(decision, model_field, backend_latency_ms)

        if not success:
            raise HTTPException(status_code=502, detail=f"backend error: {error_kind}")
        return response

    return app
