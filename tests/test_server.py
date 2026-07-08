import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from saint.server import build_app
from tests.test_router import _cfg


def test_v1_models_lists_virtual_and_backends(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    ids = {m["id"] for m in data["data"]}
    expected = {"saint-auto", "saint-explain"} | {f"saint-{name}" for name in cfg.backends}
    assert ids == expected


def test_chat_completions_pinned_backend(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    backend_response = {
        "id": "x", "object": "chat.completion", "created": 0,
        "model": "claude-opus-4-7",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    with patch("saint.backends.litellm.acompletion",
               AsyncMock(return_value=backend_response)):
        resp = client.post("/v1/chat/completions", json={
            "model": "saint-cloud-large",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "hello"


def test_chat_completions_explain_mode(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    payload = json.dumps({"domain": "code", "complexity": "medium", "reason": "."})
    with patch("saint.classifier.call_backend",
               AsyncMock(return_value={"choices": [{"message": {"content": payload}}]})):
        resp = client.post("/v1/chat/completions", json={
            "model": "saint-explain",
            "messages": [{"role": "user", "content": "rewrite x"}],
        })
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert "Routing decision" in content


def test_chat_completions_unknown_prefix_returns_400(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json={
        "model": "saint-auto",
        "messages": [{"role": "user", "content": "!doesnotexist hi"}],
    })
    assert resp.status_code == 400
    assert "doesnotexist" in resp.json()["error"]["message"]


class _FakeStreamChunks:
    """Async iterator yielding fake litellm chunks."""

    def __init__(self, chunks: list[dict]):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def test_chat_completions_streaming(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    chunks = [
        {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
         "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]},
        {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
         "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}]},
        {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    with patch("saint.backends.litellm.acompletion",
               AsyncMock(return_value=_FakeStreamChunks(chunks))):
        with client.stream("POST", "/v1/chat/completions", json={
            "model": "saint-cloud-large",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as resp:
            text = "".join(part for part in resp.iter_text())
    assert "data: " in text
    assert "Hel" in text
    assert "data: [DONE]" in text


def test_stdout_summary_line(tmp_path, capsys):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    backend_response = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    with patch("saint.backends.litellm.acompletion",
               AsyncMock(return_value=backend_response)):
        client.post("/v1/chat/completions", json={
            "model": "saint-cloud-large",
            "messages": [{"role": "user", "content": "hi"}],
        })
    captured = capsys.readouterr()
    assert "[router]" in captured.out
    assert "cloud-large" in captured.out


def _sqlite_row(tmp_path, query):
    import sqlite3
    conn = sqlite3.connect(tmp_path / "log.sqlite")
    return conn.execute(query).fetchone()


def test_cache_tokens_logged_non_streaming_dict_and_pydantic(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    usage = {"prompt_tokens": 9000, "completion_tokens": 50,
             "cache_read_input_tokens": 8712, "cache_creation_input_tokens": 130}
    dict_resp = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": usage,
    }

    from pydantic import BaseModel

    class _Pydanticish(BaseModel):  # shaped like litellm's ModelResponse
        choices: list
        usage: dict

    for mock_resp in (dict_resp, _Pydanticish(**dict_resp)):
        with patch("saint.backends.litellm.acompletion", AsyncMock(return_value=mock_resp)):
            client.post("/v1/chat/completions", json={
                "model": "saint-cloud-large",
                "messages": [{"role": "user", "content": "hi"}],
            })
        row = _sqlite_row(tmp_path,
            "SELECT cache_read_tokens, cache_write_tokens, tokens_in FROM requests "
            "ORDER BY id DESC LIMIT 1")
        assert row == (8712, 130, 9000)


def test_cache_tokens_logged_streaming_final_chunk(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 5000, "completion_tokens": 7,
                   "prompt_tokens_details": {"cached_tokens": 4200,
                                             "cache_creation_tokens": 300}}},
    ]
    with patch("saint.backends.litellm.acompletion",
               AsyncMock(return_value=_FakeStreamChunks(chunks))):
        with client.stream("POST", "/v1/chat/completions", json={
            "model": "saint-cloud-large",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as resp:
            "".join(resp.iter_text())
    row = _sqlite_row(tmp_path,
        "SELECT cache_read_tokens, cache_write_tokens FROM requests ORDER BY id DESC LIMIT 1")
    assert row == (4200, 300)


def test_turn_cache_end_to_end(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    payload = json.dumps({"domain": "code", "complexity": "medium", "reason": "."})
    clf = AsyncMock(return_value={"choices": [{"message": {"content": payload}}]})
    backend_response = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    req = {"model": "saint-auto",
           "messages": [{"role": "user", "content": "refactor the widget parser to stream tokens please"}]}
    with patch("saint.classifier.call_backend", clf), \
         patch("saint.backends.litellm.acompletion", AsyncMock(return_value=backend_response)):
        client.post("/v1/chat/completions", json=req)
        client.post("/v1/chat/completions", json=req)
    assert clf.call_count == 1
    row = _sqlite_row(tmp_path,
        "SELECT classifier_used, classifier_latency_ms FROM requests ORDER BY id DESC LIMIT 1")
    assert row == ("cache", 0)


def test_session_id_header_inheritance_end_to_end(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    payload = json.dumps({"domain": "code", "complexity": "hard", "reason": "."})
    clf = AsyncMock(return_value={"choices": [{"message": {"content": payload}}]})
    backend_response = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    hdr = {"x-session-id": "pi-session-1"}
    with patch("saint.classifier.call_backend", clf), \
         patch("saint.backends.litellm.acompletion", AsyncMock(return_value=backend_response)):
        client.post("/v1/chat/completions", headers=hdr, json={
            "model": "saint-auto",
            "messages": [{"role": "user",
                          "content": "explore the saint-router repo and refactor decide_route"}]})
        # different first message, same session id → same conversation; short follow-up inherits
        client.post("/v1/chat/completions", headers=hdr, json={
            "model": "saint-auto",
            "messages": [{"role": "user", "content": "yes, do that"}]})
    assert clf.call_count == 1
    row = _sqlite_row(tmp_path,
        "SELECT classifier_used, backend_chosen FROM requests ORDER BY id DESC LIMIT 1")
    assert row == ("inherited", "cloud-large")  # policy.normal[code,hard]


# --- dispatch failover / breaker / headers / embeddings ---
def _cfg_with_fallback():
    from dataclasses import replace
    cfg = _cfg()
    backends = dict(cfg.backends)
    backends["cloud-large"] = replace(backends["cloud-large"], on_error="local-small")
    return replace(cfg, backends=backends)


_OK = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"}],
       "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


def test_dispatch_failover_non_streaming(tmp_path):
    cfg = _cfg_with_fallback()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    mock = AsyncMock(side_effect=[RuntimeError("boom"), RuntimeError("boom"), _OK])
    with patch("saint.backends.litellm.acompletion", mock):
        resp = client.post("/v1/chat/completions", json={
            "model": "saint-cloud-large",
            "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert mock.call_count == 3  # primary + retry, then fallback
    assert resp.headers["x-saint-backend"] == "local-small"
    assert resp.headers["x-saint-decided"] == "cloud-large"
    row = _sqlite_row(tmp_path,
        "SELECT backend_chosen, state_at_dispatch, success FROM requests ORDER BY id DESC LIMIT 1")
    assert row == ("local-small", "dispatch_fallback", 1)


def test_dispatch_all_candidates_fail_502(tmp_path):
    cfg = _cfg_with_fallback()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    with patch("saint.backends.litellm.acompletion", AsyncMock(side_effect=RuntimeError("boom"))):
        resp = client.post("/v1/chat/completions", json={
            "model": "saint-cloud-large",
            "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 502
    row = _sqlite_row(tmp_path, "SELECT success, error_kind FROM requests ORDER BY id DESC LIMIT 1")
    assert row == (0, "RuntimeError")


def test_breaker_skips_primary_after_threshold(tmp_path):
    from dataclasses import replace
    cfg = _cfg_with_fallback()
    cfg = replace(cfg, routing=replace(cfg.routing, breaker_failures=2, retry_same_backend=True))
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    req = {"model": "saint-cloud-large", "messages": [{"role": "user", "content": "hi"}]}
    # request 1: primary fails twice (breaker opens at 2), fallback serves
    m1 = AsyncMock(side_effect=[RuntimeError("boom"), RuntimeError("boom"), _OK])
    with patch("saint.backends.litellm.acompletion", m1):
        assert client.post("/v1/chat/completions", json=req).status_code == 200
    # request 2: breaker open -> straight to fallback (single call)
    m2 = AsyncMock(return_value=_OK)
    with patch("saint.backends.litellm.acompletion", m2):
        resp = client.post("/v1/chat/completions", json=req)
    assert resp.status_code == 200
    assert m2.call_count == 1
    assert resp.headers["x-saint-backend"] == "local-small"


def test_streaming_failover_before_first_chunk(tmp_path):
    cfg = _cfg_with_fallback()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    mock = AsyncMock(side_effect=[RuntimeError("boom"), RuntimeError("boom"),
                                  _FakeStreamChunks(chunks)])
    with patch("saint.backends.litellm.acompletion", mock):
        with client.stream("POST", "/v1/chat/completions", json={
            "model": "saint-cloud-large",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as resp:
            headers = dict(resp.headers)
            text = "".join(resp.iter_text())
    assert "hi" in text and "data: [DONE]" in text
    assert headers["x-saint-backend"] == "local-small"
    row = _sqlite_row(tmp_path,
        "SELECT backend_chosen, state_at_dispatch FROM requests ORDER BY id DESC LIMIT 1")
    assert row == ("local-small", "dispatch_fallback")


def test_route_headers_on_classified_request(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    payload = json.dumps({"domain": "code", "complexity": "medium", "reason": "."})
    with patch("saint.classifier.call_backend",
               AsyncMock(return_value={"choices": [{"message": {"content": payload}}]})), \
         patch("saint.backends.litellm.acompletion", AsyncMock(return_value=_OK)):
        resp = client.post("/v1/chat/completions", json={
            "model": "saint-auto",
            "messages": [{"role": "user", "content": "write me a widget parser"}]})
    assert resp.headers["x-saint-backend"] == "local-coder"
    assert resp.headers["x-saint-domain"] == "code"
    assert resp.headers["x-saint-complexity"] == "medium"
    assert "x-saint-decided" not in resp.headers  # no fallback hop taken


def test_embeddings_endpoint_routes_and_logs(tmp_path):
    from dataclasses import replace
    cfg = _cfg()
    cfg = replace(cfg, routing=replace(cfg.routing, embeddings_backend="local-small"))
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    emb = {"object": "list", "data": [{"object": "embedding", "index": 0,
                                       "embedding": [0.1, 0.2]}],
           "model": "nomic", "usage": {"prompt_tokens": 5, "total_tokens": 5}}
    with patch("saint.backends.litellm.aembedding", AsyncMock(return_value=emb)) as m:
        resp = client.post("/v1/embeddings", json={"model": "anything", "input": "hello"})
    assert resp.status_code == 200
    assert resp.json()["data"][0]["embedding"] == [0.1, 0.2]
    assert resp.headers["x-saint-backend"] == "local-small"
    assert m.call_args.kwargs["input"] == "hello"
    row = _sqlite_row(tmp_path,
        "SELECT model_field, backend_chosen, tokens_in, prompt_content FROM requests "
        "ORDER BY id DESC LIMIT 1")
    assert row == ("saint-embeddings", "local-small", 5, None)


def test_embeddings_endpoint_unconfigured_404(tmp_path):
    cfg = _cfg()  # embeddings_backend unset
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    resp = client.post("/v1/embeddings", json={"input": "hello"})
    assert resp.status_code == 404
