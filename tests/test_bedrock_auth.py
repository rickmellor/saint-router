from unittest.mock import MagicMock, patch

import pytest

from saint import bedrock_auth


@pytest.fixture(autouse=True)
def _clean_cache():
    bedrock_auth.reset_session_cache()
    yield
    bedrock_auth.reset_session_cache()


def test_litellm_private_api_still_exists():
    # CI canary: a litellm upgrade that moves this symbol must fail here, not in prod.
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
    assert hasattr(BaseAWSLLM, "_auth_with_aws_profile")


def test_session_constructed_once_and_provider_removed():
    fake_provider = MagicMock()
    fake_bc_session = MagicMock()
    fake_bc_session.get_component.return_value = fake_provider
    fake_boto_session = MagicMock()
    fake_creds = object()
    fake_boto_session.get_credentials.return_value = fake_creds

    with patch("botocore.session.Session", return_value=fake_bc_session) as bc, \
         patch("boto3.Session", return_value=fake_boto_session) as b3:
        creds1, ttl1 = bedrock_auth._auth_via_credential_process(None, "ClaudeCode")
        creds2, ttl2 = bedrock_auth._auth_via_credential_process(None, "ClaudeCode")

    assert bc.call_count == 1 and b3.call_count == 1  # cached after first call
    fake_provider.remove.assert_called_once_with("shared-credentials-file")
    assert creds1 is fake_creds and creds2 is fake_creds
    assert ttl1 is None and ttl2 is None  # litellm iam cache must not double-cache


def test_patch_applied_only_with_bedrock_backend(tmp_path):
    from fastapi.testclient import TestClient  # noqa: F401

    from saint.server import build_app
    from tests.test_router import _cfg

    with patch("saint.bedrock_auth.apply_bedrock_auth_patch") as ap:
        build_app(_cfg(), db_path=tmp_path / "a.sqlite")
    ap.assert_not_called()

    from dataclasses import replace

    from saint.config import BackendConfig
    cfg = _cfg()
    backends = dict(cfg.backends)
    backends["bedrock-sonnet"] = BackendConfig(
        name="bedrock-sonnet", provider="bedrock", model="global.anthropic.claude-sonnet-5",
        api_key_env=None, api_key=None, base_url=None, aliases=(), timeout_s=120,
        aws_region="us-east-1")
    cfg = replace(cfg, backends=backends)
    with patch("saint.bedrock_auth.apply_bedrock_auth_patch") as ap2:
        build_app(cfg, db_path=tmp_path / "b.sqlite")
    ap2.assert_called_once()
