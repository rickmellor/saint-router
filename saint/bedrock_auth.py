"""Bedrock credential-chain fix: force credential_process with refresh semantics.

The problem (observed in the wild on corp SSO deployments): Claude Code writes *static*
temporary credentials to ~/.aws/credentials as a cache. boto3's default chain prefers that
file over the `credential_process` entry in ~/.aws/config — and does not track the cached
creds' ~1h expiry — so Bedrock calls silently start failing with 403s once they lapse,
even though the credential_process could mint fresh ones.

The fix (technique from the litellm-cascade-router reference implementation):

1. Build a botocore session with the ``shared-credentials-file`` provider REMOVED, so
   resolution falls through to ``credential_process`` (from ~/.aws/config). Its output
   includes ``Expiration``, so botocore wraps it in ``RefreshableCredentials`` that
   transparently re-run the helper as tokens near expiry.
2. Cache one boto3.Session per profile at THIS layer, and return ``None`` as the TTL to
   litellm's iam cache. Double-caching at the litellm layer would construct a fresh
   Session (→ fresh credential_process spawn → possibly an SSO browser tab) on every
   litellm-cache expiry instead of letting RefreshableCredentials manage one long-lived
   refresh lifecycle.

This monkeypatches ``BaseAWSLLM._auth_with_aws_profile`` — private litellm API, verified
against litellm 1.91.0. A litellm upgrade must re-verify
``litellm/llms/bedrock/base_aws_llm.py`` (the hasattr guard below turns silent drift into
a loud startup error, and a unit test imports the real symbol so CI catches it first).

Applied only when the config actually defines a bedrock backend (Config.has_bedrock).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# One boto3.Session per AWS profile, so RefreshableCredentials owns the refresh
# lifecycle with a single long-lived credential_process relationship.
_session_cache: dict[str, object] = {}


def _auth_via_credential_process(self, aws_profile_name):  # noqa: ANN001 — litellm method shape
    if aws_profile_name not in _session_cache:
        import boto3
        import botocore.session

        bc_session = botocore.session.Session(profile=aws_profile_name)
        bc_session.get_component("credential_provider").remove("shared-credentials-file")
        _session_cache[aws_profile_name] = boto3.Session(botocore_session=bc_session)
        logger.info(
            "bedrock auth: cached boto3.Session for profile %r "
            "(credential_process, shared-credentials-file skipped)",
            aws_profile_name,
        )
    session = _session_cache[aws_profile_name]
    credentials = session.get_credentials()
    # None TTL: litellm's iam cache must NOT wrap these — RefreshableCredentials
    # handles expiry internally (see module docstring).
    return credentials, None


def apply_bedrock_auth_patch() -> None:
    """Install the credential-chain patch. Raises loudly if litellm's private API moved."""
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    if not hasattr(BaseAWSLLM, "_auth_with_aws_profile"):
        raise RuntimeError(
            "litellm's BaseAWSLLM no longer has _auth_with_aws_profile — the bedrock "
            "credential patch (saint/bedrock_auth.py, verified against litellm 1.91.0) "
            "must be re-verified against this litellm version before bedrock backends "
            "can be used safely."
        )
    BaseAWSLLM._auth_with_aws_profile = _auth_via_credential_process
    logger.info("bedrock auth patch applied: per-profile session cache + credential_process")
    print("[saint] bedrock credential patch applied (credential_process, SSO-refresh-safe)",
          flush=True)


def reset_session_cache() -> None:
    """Test hook."""
    _session_cache.clear()
