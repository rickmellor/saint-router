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
