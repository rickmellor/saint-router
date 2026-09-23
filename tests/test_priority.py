"""Per-backend vLLM request priority rides in extra_body; clients may override; cloud providers never get it."""
from saint.backends import _shape_request
from saint.config import BackendConfig


def _b(provider="openai", priority=None):
    return BackendConfig(name="b", provider=provider, model="m", api_key_env=None, api_key="x", base_url="http://x/v1",
                         aliases=(), timeout_s=10, priority=priority)


def test_backend_priority_lands_in_extra_body():
    k = _shape_request(_b(priority=10), {"model": "m", "messages": []})
    assert k["extra_body"]["priority"] == 10


def test_client_priority_overrides_backend():
    k = _shape_request(_b(priority=10), {"model": "m", "messages": [], "priority": 0})
    assert k["extra_body"]["priority"] == 0 and "priority" not in k


def test_no_priority_when_unset():
    k = _shape_request(_b(), {"model": "m", "messages": []})
    assert "extra_body" not in k


def test_cloud_never_gets_priority():
    k = _shape_request(_b(provider="anthropic", priority=0), {"model": "m", "messages": [], "priority": 0})
    assert "priority" not in k and "extra_body" not in k


def test_priority_merges_with_chat_template_kwargs():
    k = _shape_request(_b(priority=5), {"model": "m", "messages": [], "chat_template_kwargs": {"enable_thinking": False}})
    assert k["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}, "priority": 5}
