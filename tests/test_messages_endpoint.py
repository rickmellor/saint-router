import json
from dataclasses import replace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from saint.server import build_app
from tests.test_router import _cfg

_ANTHROPIC_OK = {
    "id": "msg_01", "type": "message", "role": "assistant", "model": "m",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 40, "output_tokens": 5,
              "cache_creation_input_tokens": 30, "cache_read_input_tokens": 0},
}

_CLS = json.dumps({"domain": "code", "complexity": "medium", "reason": "."})
_CLS_RESP = {"choices": [{"message": {"content": _CLS}}]}


def _app(tmp_path, cfg=None):
    app = build_app(cfg or _cfg(), db_path=tmp_path / "log.sqlite")
    return app, TestClient(app)


def _row(tmp_path, query):
    import sqlite3
    return sqlite3.connect(tmp_path / "log.sqlite").execute(query).fetchone()


def test_messages_pinned_passthrough_headers_and_log(tmp_path):
    app, client = _app(tmp_path)
    acreate = AsyncMock(return_value=_ANTHROPIC_OK)
    with patch("saint.backends.litellm.anthropic.messages.acreate", acreate):
        resp = client.post("/v1/messages", json={
            "model": "saint-cloud-large", "max_tokens": 100,
            "system": "you are helpful",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"][0]["text"] == "hello"
    assert resp.headers["x-saint-backend"] == "cloud-large"
    kw = acreate.call_args.kwargs
    assert kw["model"] == "anthropic/claude-opus-4-7"
    assert kw["max_tokens"] == 100 and kw["system"] == "you are helpful"
    row = _row(tmp_path, "SELECT model_field, backend_chosen, tokens_in, tokens_out, "
                         "cache_write_tokens, success FROM requests ORDER BY id DESC LIMIT 1")
    assert row == ("saint-cloud-large", "cloud-large", 40, 5, 30, 1)


def test_messages_alias_pins_and_unknown_routes_auto(tmp_path):
    cfg = _cfg()
    backends = dict(cfg.backends)
    backends["cloud-large"] = replace(backends["cloud-large"],
                                      aliases=("opus", "global.anthropic.claude-opus-4-8"))
    cfg = replace(cfg, backends=backends)
    app, client = _app(tmp_path, cfg)
    acreate = AsyncMock(return_value=_ANTHROPIC_OK)
    clf = AsyncMock(return_value=_CLS_RESP)
    with patch("saint.backends.litellm.anthropic.messages.acreate", acreate), \
         patch("saint.classifier.call_backend", clf):
        # raw inference-profile id -> alias pin, no classification
        r1 = client.post("/v1/messages", json={
            "model": "global.anthropic.claude-opus-4-8", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}]})
        # unknown model -> auto (classifier runs)
        r2 = client.post("/v1/messages", json={
            "model": "claude-mystery-9", "max_tokens": 10,
            "messages": [{"role": "user", "content": "write me a widget parser"}]})
    assert r1.headers["x-saint-backend"] == "cloud-large"
    assert clf.call_count == 1
    assert r2.headers["x-saint-backend"] == "local-coder"  # policy code,medium
    assert r2.headers["x-saint-domain"] == "code"


def test_messages_missing_max_tokens_anthropic_400(tmp_path):
    _, client = _app(tmp_path)
    resp = client.post("/v1/messages", json={
        "model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 400
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"


def test_messages_prefixes_parsed_and_stripped(tmp_path):
    app, client = _app(tmp_path)
    acreate = AsyncMock(return_value=_ANTHROPIC_OK)
    with patch("saint.backends.litellm.anthropic.messages.acreate", acreate):
        resp = client.post("/v1/messages", json={
            "model": "auto", "max_tokens": 10,
            "messages": [{"role": "user", "content": "@opus review this design"}]})
    assert resp.headers["x-saint-backend"] == "cloud-large"
    assert resp.headers["x-saint-pinned"] == "cloud-large"
    sent = acreate.call_args.kwargs["messages"]
    assert sent[-1]["content"] == "review this design"  # prefix stripped for the backend


def test_messages_tool_result_turn_is_not_multimodal(tmp_path):
    app, client = _app(tmp_path)
    acreate = AsyncMock(return_value=_ANTHROPIC_OK)
    clf = AsyncMock(return_value=_CLS_RESP)
    with patch("saint.backends.litellm.anthropic.messages.acreate", acreate), \
         patch("saint.classifier.call_backend", clf):
        # first turn establishes conversation labels
        client.post("/v1/messages", json={
            "model": "auto", "max_tokens": 10,
            "system": "agent",
            "messages": [{"role": "user",
                          "content": "refactor the widget parser to stream tokens"}]})
        # agent tool turn: last user message is a tool_result block
        resp = client.post("/v1/messages", json={
            "model": "auto", "max_tokens": 10,
            "system": "agent",
            "messages": [
                {"role": "user", "content": "refactor the widget parser to stream tokens"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "reading"},
                    {"type": "tool_use", "id": "t1", "name": "read", "input": {}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1",
                     "content": [{"type": "text", "text": "def parse(): ..."}]}]},
            ]})
    # NOT routed to default_on_failure/multimodal; inherited the conversation's labels
    assert resp.headers["x-saint-backend"] == "local-coder"
    assert resp.headers["x-saint-classifier"] in ("inherited", "cache")
    assert clf.call_count == 1  # only the first turn classified


def test_messages_image_block_routes_multimodal(tmp_path):
    cfg = _cfg()
    cfg = replace(cfg, routing=replace(cfg.routing, multimodal_backend="local-small"))
    app, client = _app(tmp_path, cfg)
    acreate = AsyncMock(return_value=_ANTHROPIC_OK)
    with patch("saint.backends.litellm.anthropic.messages.acreate", acreate):
        resp = client.post("/v1/messages", json={
            "model": "auto", "max_tokens": 10,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": "x"}}]}]})
    assert resp.headers["x-saint-backend"] == "local-small"


def test_messages_dispatch_failover_and_502_shape(tmp_path):
    cfg = _cfg()
    backends = dict(cfg.backends)
    backends["cloud-large"] = replace(backends["cloud-large"], on_error="local-small")
    cfg = replace(cfg, backends=backends)
    app, client = _app(tmp_path, cfg)
    req = {"model": "saint-cloud-large", "max_tokens": 10,
           "messages": [{"role": "user", "content": "hi"}]}
    with patch("saint.backends.litellm.anthropic.messages.acreate",
               AsyncMock(side_effect=[RuntimeError("boom"), RuntimeError("boom"),
                                      _ANTHROPIC_OK])):
        resp = client.post("/v1/messages", json=req)
    assert resp.status_code == 200
    assert resp.headers["x-saint-backend"] == "local-small"
    assert resp.headers["x-saint-decided"] == "cloud-large"
    with patch("saint.backends.litellm.anthropic.messages.acreate",
               AsyncMock(side_effect=RuntimeError("boom"))):
        resp2 = client.post("/v1/messages", json=req)
    assert resp2.status_code == 502
    assert resp2.json()["error"]["type"] == "api_error"


def test_messages_client_cache_control_passes_through_unmodified(tmp_path):
    app, client = _app(tmp_path)
    big = "You are a meticulous reviewer. " * 400  # over prompt_cache_min_chars
    sys_blocks = [{"type": "text", "text": big, "cache_control": {"type": "ephemeral"}}]
    acreate = AsyncMock(return_value=_ANTHROPIC_OK)
    with patch("saint.backends.litellm.anthropic.messages.acreate", acreate):
        client.post("/v1/messages", json={
            "model": "saint-cloud-large", "max_tokens": 10,
            "system": sys_blocks,
            "messages": [{"role": "user", "content": "go"}]})
    kw = acreate.call_args.kwargs
    assert kw["system"] == sys_blocks          # untouched, single breakpoint
    assert kw["messages"] == [{"role": "user", "content": "go"}]  # no injection


class _FakeByteStream:
    """Async iterator of Anthropic SSE bytes (litellm acreate stream=True shape)."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


_SSE_CHUNKS = [
    b'event: message_start\ndata: {"type":"message_start","message":{"usage":'
    b'{"input_tokens":5000,"cache_creation_input_tokens":300,'
    b'"cache_read_input_tokens":4200,"output_tokens":1}}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta",'
    b'"delta":{"type":"text_delta","text":"hel"}}\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":42}}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
]


def test_messages_streaming_relays_bytes_and_logs_usage(tmp_path):
    app, client = _app(tmp_path)
    with patch("saint.backends.litellm.anthropic.messages.acreate",
               AsyncMock(return_value=_FakeByteStream(_SSE_CHUNKS))):
        with client.stream("POST", "/v1/messages", json={
            "model": "saint-cloud-large", "max_tokens": 100, "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        }) as resp:
            headers = dict(resp.headers)
            raw = b"".join(resp.iter_bytes())
    assert raw == b"".join(_SSE_CHUNKS)     # byte-identical relay
    assert headers["x-saint-backend"] == "cloud-large"
    row = _row(tmp_path, "SELECT tokens_in, tokens_out, cache_read_tokens, "
                         "cache_write_tokens, success FROM requests ORDER BY id DESC LIMIT 1")
    assert row == (5000, 42, 4200, 300, 1)


def test_messages_streaming_failover_before_first_chunk(tmp_path):
    cfg = _cfg()
    backends = dict(cfg.backends)
    backends["cloud-large"] = replace(backends["cloud-large"], on_error="local-small")
    cfg = replace(cfg, backends=backends)
    app, client = _app(tmp_path, cfg)
    with patch("saint.backends.litellm.anthropic.messages.acreate",
               AsyncMock(side_effect=[RuntimeError("boom"), RuntimeError("boom"),
                                      _FakeByteStream(_SSE_CHUNKS)])):
        with client.stream("POST", "/v1/messages", json={
            "model": "saint-cloud-large", "max_tokens": 100, "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        }) as resp:
            headers = dict(resp.headers)
            raw = b"".join(resp.iter_bytes())
    assert b"message_stop" in raw
    assert headers["x-saint-backend"] == "local-small"
    assert headers["x-saint-decided"] == "cloud-large"


def test_sse_tracker_tolerates_garbage():
    from saint.anthropic_api import SseUsageTracker
    t = SseUsageTracker()
    t.feed(b"event: message_start\ndata: {not json}\n\n")
    t.feed(b"random bytes without structure")
    t.feed(b'event: message_delta\ndata: {"type":"message_delta","usage":'
           b'{"output_tokens":7}}\n\n')
    assert t.usage == {"output_tokens": 7}


_THINKING_HISTORY = [
    {"role": "user", "content": "design a rate limiter"},
    {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "considering token bucket", "signature": "sig-abc"},
        {"type": "text", "text": "Here's a design..."}]},
    {"role": "user", "content": "now make it distributed"},
]


def test_messages_thinking_stripped_on_backend_switch(tmp_path):
    # cloud-large has an on_error to local-small, but force a switch via classification:
    # first turn pins nothing; we simulate a served backend differing from history's minter
    cfg = _cfg()
    backends = dict(cfg.backends)
    # two anthropic-ish backends so a switch stays on a signature-validating provider
    backends["cloud-small"] = replace(backends["cloud-small"], on_error=None)
    cfg = replace(cfg, backends=backends)
    app, client = _app(tmp_path, cfg)
    acreate = AsyncMock(return_value=_ANTHROPIC_OK)
    clf = AsyncMock(return_value=_CLS_RESP)
    with patch("saint.backends.litellm.anthropic.messages.acreate", acreate), \
         patch("saint.classifier.call_backend", clf):
        # turn 1 on cloud-small (pinned) — establishes affinity backend = cloud-small
        client.post("/v1/messages", json={
            "model": "saint-cloud-small", "max_tokens": 10, "system": "agent",
            "messages": [{"role": "user", "content": "design a rate limiter"}]})
        # turn 2 same conversation, pinned to a DIFFERENT backend, replaying signed thinking
        client.post("/v1/messages", json={
            "model": "saint-cloud-large", "max_tokens": 10, "system": "agent",
            "messages": _THINKING_HISTORY})
    sent = acreate.call_args.kwargs["messages"]
    # the signed thinking block must be gone (backend switched cloud-small -> cloud-large)
    assistant = [m for m in sent if m["role"] == "assistant"][0]
    assert all(b.get("type") != "thinking" for b in assistant["content"])


def test_messages_thinking_kept_when_backend_stable(tmp_path):
    app, client = _app(tmp_path)
    acreate = AsyncMock(return_value=_ANTHROPIC_OK)
    with patch("saint.backends.litellm.anthropic.messages.acreate", acreate):
        # turn 1 pins cloud-large
        client.post("/v1/messages", json={
            "model": "saint-cloud-large", "max_tokens": 10, "system": "agent",
            "messages": [{"role": "user", "content": "design a rate limiter"}]})
        # turn 2 same backend → thinking preserved
        client.post("/v1/messages", json={
            "model": "saint-cloud-large", "max_tokens": 10, "system": "agent",
            "messages": _THINKING_HISTORY})
    sent = acreate.call_args.kwargs["messages"]
    assistant = [m for m in sent if m["role"] == "assistant"][0]
    assert any(b.get("type") == "thinking" for b in assistant["content"])
