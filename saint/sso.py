"""SSO credential recovery for bedrock backends (technique from litellm-cascade-router).

Two subprocess helpers around the corporate ``credential-process`` binary, plus the
auth-failure classifier that decides when to use them:

- ``ensure_sso_valid``: run ``credential-process --profile X --refresh-if-needed`` —
  non-interactive (built for cron), actively RE-MINTS the short-lived AWS credentials from
  a valid SSO session. Deliberately not a read-only expiry check: after the user
  re-authenticates SSO externally, the cached AWS creds are not repopulated until
  something fetches them, so a read-only check reports "expired" forever. Fail-closed
  (non-zero exit / timeout / spawn error → False); never raises; never opens a browser.
- ``spawn_sso_login``: fire-and-forget ``credential-process --profile X`` which opens the
  SSO browser tab. Called once per auth-failure event; recovery is detected later by the
  breaker's half-open probe running ``ensure_sso_valid``.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

AUTH_FAILURE_STATUS_CODES = {401, 403}
# Substrings (lowercased) that mark a dispatch exception as a credential failure.
# Deliberate divergence from the reference: "timeout"/"deadline" are NOT auth — in saint
# those are ordinary dispatch failures handled by retry/on_error; classifying them as
# auth would impose the long auth cooldown on transient network blips.
AUTH_FAILURE_KEYWORDS = (
    "credential",
    "security token",
    "expired",
    "access denied",
    "not authorized",
    "invalidclienttokenid",
    "signaturedoesnotmatch",
    "unrecognizedclientexception",
    "error when retrieving credentials",
)


def is_auth_failure(exc: Exception) -> bool:
    """True when the exception (or anything in its cause/context chain) looks like an
    AWS credential/auth failure. Cycle-safe; never raises."""
    try:
        from botocore.exceptions import CredentialRetrievalError
    except ImportError:  # boto3 not installed — no bedrock in play
        CredentialRetrievalError = ()  # type: ignore[assignment]

    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if CredentialRetrievalError and isinstance(node, CredentialRetrievalError):
            return True
        status = getattr(node, "status_code", None)
        if status in AUTH_FAILURE_STATUS_CODES:
            return True
        text = str(node).lower()
        if any(k in text for k in AUTH_FAILURE_KEYWORDS):
            return True
        node = node.__cause__ or node.__context__
    return False


async def ensure_sso_valid(credential_process: str, profile: str,
                           timeout_s: float = 8.0) -> bool:
    """Non-interactive credential re-mint probe. True iff exit 0 within the timeout."""
    cmd = [credential_process, "--profile", profile, "--refresh-if-needed"]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (OSError, FileNotFoundError) as exc:
        logger.warning("SSO refresh spawn failed for profile %r: %s", profile, exc)
        return False
    try:
        _out, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning("SSO refresh for profile %r timed out after %.0fs", profile, timeout_s)
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        return False
    if proc.returncode != 0:
        logger.debug("SSO refresh for %r exited %d: %s", profile, proc.returncode,
                     stderr.decode(errors="replace")[:200])
    return proc.returncode == 0


async def spawn_sso_login(credential_process: str, profile: str) -> None:
    """Open the SSO browser tab (detached; never awaited, never raises)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            credential_process, "--profile", profile,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            stdin=asyncio.subprocess.DEVNULL,
        )
        logger.info("Spawned credential-process for profile %r (pid=%s) — SSO browser tab",
                    profile, proc.pid)
    except (OSError, FileNotFoundError) as exc:
        logger.warning("Failed to spawn credential-process for %r: %s. Re-auth manually: "
                       "%s --profile %s", profile, exc, credential_process, profile)
