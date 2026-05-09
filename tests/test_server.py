import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from goorouter.server import build_app
from tests.test_router import _cfg


def test_v1_models_lists_virtual_and_backends(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    ids = {m["id"] for m in data["data"]}
    expected = {"goo-auto", "goo-explain"} | {f"goo-{name}" for name in cfg.backends}
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
    with patch("goorouter.backends.litellm.acompletion",
               AsyncMock(return_value=backend_response)):
        resp = client.post("/v1/chat/completions", json={
            "model": "goo-cloud-large",
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
    with patch("goorouter.classifier.call_backend",
               AsyncMock(return_value={"choices": [{"message": {"content": payload}}]})):
        resp = client.post("/v1/chat/completions", json={
            "model": "goo-explain",
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
        "model": "goo-auto",
        "messages": [{"role": "user", "content": "!doesnotexist hi"}],
    })
    assert resp.status_code == 400
    assert "doesnotexist" in resp.json()["error"]["message"]
