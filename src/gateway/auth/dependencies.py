"""FastAPI dependency resolving an authenticated team from the `X-Api-Key` header.

Verification flow: validate key format -> extract key_prefix (first 12 chars) ->
query api_keys joined to teams -> compute HMAC-SHA-256(pepper, presented_key) ->
constant-time compare against stored hmac_digest -> check expires_at/revoked_at.

Never logs the presented API key, the pepper, or the computed digest.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from fastapi import Depends, Header, HTTPException, status

from gateway.auth.models import TeamConfig
from gateway.auth.security import extract_key_prefix, validate_key_format, verify_api_key
from gateway.config.database import DatabaseConfig

_UNAUTHORIZED_DETAIL = "Invalid or missing API key"

_SELECT_CANDIDATES_SQL = """
    SELECT ak.hmac_digest, ak.expires_at, ak.revoked_at,
           t.team_id, t.allowed_tiers, t.rate_limit_rpm, t.rate_limit_tpm,
           t.daily_budget_microdollars, t.monthly_budget_microdollars
    FROM api_keys ak
    JOIN teams t ON t.team_id = ak.team_id
    WHERE ak.key_prefix = $1
"""

_db_config: DatabaseConfig | None = None


def _get_pepper() -> bytes:
    """Read the server-side HMAC pepper from the environment. Never log this value."""
    return os.environ["API_KEY_PEPPER"].encode("utf-8")


async def get_db_config() -> DatabaseConfig:
    """Return the process-wide DatabaseConfig, starting its connection pool on first use."""
    global _db_config
    if _db_config is None:
        config = DatabaseConfig()
        await config.startup()
        _db_config = config
    return _db_config


async def reset_db_config() -> None:
    """Shut down and clear the process-wide DatabaseConfig singleton (for test teardown)."""
    global _db_config
    if _db_config is not None:
        await _db_config.shutdown()
        _db_config = None


def _unauthorized() -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED_DETAIL)


async def get_team_config(
    x_api_key: str | None = Header(default=None),
    db: DatabaseConfig = Depends(get_db_config),
) -> TeamConfig:
    """Resolve the authenticated team for an incoming request, or raise 401."""
    if not x_api_key or not validate_key_format(x_api_key):
        raise _unauthorized()

    key_prefix = extract_key_prefix(x_api_key)
    pepper = _get_pepper()

    conn = await db.get_connection()
    try:
        rows = await conn.fetch(_SELECT_CANDIDATES_SQL, key_prefix)
    finally:
        await db.release_connection(conn)

    matched_row = None
    for row in rows:
        if verify_api_key(x_api_key, bytes(row["hmac_digest"]), pepper):
            matched_row = row
            break

    if matched_row is None:
        raise _unauthorized()

    now = datetime.now(UTC)
    if matched_row["revoked_at"] is not None:
        raise _unauthorized()
    if matched_row["expires_at"] is not None and matched_row["expires_at"] < now:
        raise _unauthorized()

    return TeamConfig(
        team_id=str(matched_row["team_id"]),
        allowed_tiers=list(matched_row["allowed_tiers"]),
        rate_limit_rpm=matched_row["rate_limit_rpm"],
        rate_limit_tpm=matched_row["rate_limit_tpm"],
        daily_budget_microdollars=matched_row["daily_budget_microdollars"],
        monthly_budget_microdollars=matched_row["monthly_budget_microdollars"],
    )
