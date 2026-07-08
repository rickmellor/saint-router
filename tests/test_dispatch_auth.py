from dataclasses import replace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from saint.config import BackendConfig, BedrockConfig
from saint.server import build_app
from tests.test_router import _cfg

_OK = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"}],
       "usage": {"prompt_tokens": 3, "completion_tokens": 1}}

AUTH_ERR = RuntimeError("The security token included in the request is expired")


def _bedrock_cfg(auth_cooldown_s=300.0, credential_process="/bin/cred-proc"):
    cfg = _cfg()
    backends = dict(cfg.backends)
    backends["bedrock-sonnet"] = BackendConfig(
        name="bedrock-sonnet", provider="bedrock",
        model="global.anthropic.claude-sonnet-5",
        api_key_env=None, api_key=None, base_url=None, aliases=(), timeout_s=120,
        aws_region="us-east-1", aws_profile="ClaudeCode", on_error="local-small")
    return replace(cfg, backends=backends,
                   bedrock=BedrockConfig(credential_process=credential_process,
                                         auth_cooldown_s=auth_cooldown_s))


def test_auth_failure_opens_breaker_no_retry_and_falls_back(tmp_path):
    cfg = _bedrock_cfg()
    with patch("saint.bedrock_auth.apply_bedrock_auth_patch"):
        app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    mock = AsyncMock(side_effect=[AUTH_ERR, _OK])
    spawn = AsyncMock()
    with patch("saint.backends.litellm.acompletion", mock), \
         patch("saint.dispatch.spawn_sso_login", spawn):
        resp = client.post("/v1/chat/completions", json={
            "model": "saint-bedrock-sonnet",
            "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    # auth failure: NO same-backend retry (call 1 = bedrock, call 2 = fallback)
    assert mock.call_count == 2
    assert resp.headers["x-saint-backend"] == "local-small"
    assert app.state.breaker.is_open("bedrock-sonnet")
    assert app.state.breaker.is_auth_flagged("bedrock-sonnet")
    assert app.state.bedrock_state.spawned_profiles == {"ClaudeCode"}


def test_sso_login_spawned_once_per_expiry_event(tmp_path):
    cfg = _bedrock_cfg(auth_cooldown_s=0.001)  # near-instant half-open for the 2nd request
    with patch("saint.bedrock_auth.apply_bedrock_auth_patch"):
        app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    req = {"model": "saint-bedrock-sonnet", "messages": [{"role": "user", "content": "hi"}]}
    spawn = AsyncMock()
    valid = AsyncMock(return_value=False)   # SSO probe keeps failing
    with patch("saint.backends.litellm.acompletion",
               AsyncMock(side_effect=[AUTH_ERR, _OK, _OK])), \
         patch("saint.dispatch.spawn_sso_login", spawn), \
         patch("saint.dispatch.ensure_sso_valid", valid):
        client.post("/v1/chat/completions", json=req)   # auth fail -> spawn once
        import time
        time.sleep(0.01)                                # cooldown elapses -> half-open
        client.post("/v1/chat/completions", json=req)   # probe fails -> skip, no new spawn
    assert spawn.call_count == 1
    assert valid.call_count == 1


def test_half_open_probe_success_allows_trial(tmp_path):
    cfg = _bedrock_cfg(auth_cooldown_s=0.001)
    with patch("saint.bedrock_auth.apply_bedrock_auth_patch"):
        app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    req = {"model": "saint-bedrock-sonnet", "messages": [{"role": "user", "content": "hi"}]}
    with patch("saint.backends.litellm.acompletion",
               AsyncMock(side_effect=[AUTH_ERR, _OK, _OK])), \
         patch("saint.dispatch.spawn_sso_login", AsyncMock()), \
         patch("saint.dispatch.ensure_sso_valid", AsyncMock(return_value=True)):
        client.post("/v1/chat/completions", json=req)       # fails -> served by fallback
        import time
        time.sleep(0.01)
        resp = client.post("/v1/chat/completions", json=req)  # probe OK -> bedrock serves
    assert resp.headers["x-saint-backend"] == "bedrock-sonnet"
    assert not app.state.breaker.is_auth_flagged("bedrock-sonnet")
    assert app.state.bedrock_state.spawned_profiles == set()


def test_non_auth_failures_keep_existing_semantics(tmp_path):
    cfg = _bedrock_cfg()
    with patch("saint.bedrock_auth.apply_bedrock_auth_patch"):
        app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    mock = AsyncMock(side_effect=[RuntimeError("boom"), RuntimeError("boom"), _OK])
    with patch("saint.backends.litellm.acompletion", mock), \
         patch("saint.dispatch.spawn_sso_login", AsyncMock()) as spawn:
        resp = client.post("/v1/chat/completions", json={
            "model": "saint-bedrock-sonnet",
            "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert mock.call_count == 3            # retry happened (non-auth), then fallback
    spawn.assert_not_called()
    assert not app.state.breaker.is_auth_flagged("bedrock-sonnet")
