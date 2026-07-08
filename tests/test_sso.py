import asyncio
from unittest.mock import AsyncMock, patch

from saint.sso import ensure_sso_valid, is_auth_failure, spawn_sso_login


class _FakeProc:
    def __init__(self, returncode=0, hang=False):
        self.returncode = returncode
        self._hang = hang
        self.killed = False

    async def communicate(self):
        if self._hang:
            await asyncio.sleep(60)
        return b"", b"stderr"

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


async def test_ensure_sso_valid_exit_codes():
    with patch("asyncio.create_subprocess_exec",
               AsyncMock(return_value=_FakeProc(returncode=0))) as m:
        assert await ensure_sso_valid("/bin/cred", "ClaudeCode") is True
    argv = m.call_args.args
    assert "--refresh-if-needed" in argv          # active re-mint, never read-only
    assert "--check-expiration" not in argv
    with patch("asyncio.create_subprocess_exec",
               AsyncMock(return_value=_FakeProc(returncode=1))):
        assert await ensure_sso_valid("/bin/cred", "ClaudeCode") is False


async def test_ensure_sso_valid_timeout_kills_and_fails_closed():
    proc = _FakeProc(hang=True)
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        assert await ensure_sso_valid("/bin/cred", "p", timeout_s=0.05) is False
    assert proc.killed


async def test_ensure_sso_valid_spawn_error_fails_closed():
    with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError)):
        assert await ensure_sso_valid("/missing", "p") is False


async def test_spawn_sso_login_never_raises():
    with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=OSError("denied"))):
        await spawn_sso_login("/missing", "p")  # must not raise


# --- is_auth_failure matrix ---
def test_auth_keywords_and_status():
    assert is_auth_failure(RuntimeError("The security token included in the request is expired"))
    assert is_auth_failure(RuntimeError("UnrecognizedClientException: bad"))
    e = RuntimeError("nope")
    e.status_code = 403
    assert is_auth_failure(e)
    assert not is_auth_failure(RuntimeError("boom"))
    # deliberate divergence from the reference: timeouts are NOT auth failures
    assert not is_auth_failure(RuntimeError("connection timed out"))


def test_auth_cause_chain_and_cycles():
    inner = RuntimeError("Error when retrieving credentials from custom-process")
    outer = RuntimeError("dispatch failed")
    outer.__cause__ = inner
    assert is_auth_failure(outer)
    # cycle safety
    a, b = RuntimeError("x"), RuntimeError("y")
    a.__cause__, b.__cause__ = b, a
    assert not is_auth_failure(a)


def test_botocore_credential_retrieval_error():
    from botocore.exceptions import CredentialRetrievalError
    exc = CredentialRetrievalError(provider="custom-process", error_msg="keyring locked")
    wrapper = RuntimeError("litellm wrapper")
    wrapper.__cause__ = exc
    assert is_auth_failure(wrapper)
