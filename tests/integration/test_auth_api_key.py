"""Integration tests for API-key authentication against a real PostgreSQL instance."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi import Depends, FastAPI

from gateway.auth.dependencies import get_team_config, reset_db_config
from gateway.auth.models import TeamConfig
from gateway.auth.security import compute_hmac_digest, extract_key_prefix, generate_api_key
from gateway.config.database import DatabaseConfig

_test_app = FastAPI()


@_test_app.get("/protected")
async def _protected(team: TeamConfig = Depends(get_team_config)) -> dict[str, object]:
    return {"team_id": team.team_id, "allowed_tiers": team.allowed_tiers}


@pytest.fixture
async def db() -> AsyncIterator[DatabaseConfig]:
    """A DatabaseConfig used to seed/clean up rows independently of the auth dependency."""
    config = DatabaseConfig()
    await config.startup()
    yield config
    await config.shutdown()


@pytest.fixture(autouse=True)
async def _reset_dependency_singleton() -> AsyncIterator[None]:
    """Ensure get_team_config's process-wide DatabaseConfig is torn down after each test."""
    yield
    await reset_db_config()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=_test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class _SeededTeam:
    def __init__(self, team_id: str, api_key: str) -> None:
        self.team_id = team_id
        self.api_key = api_key


async def _seed_team(
    db: DatabaseConfig,
    *,
    expires_at: datetime | None = None,
    revoked_at: datetime | None = None,
) -> _SeededTeam:
    """Insert a team and a matching api_keys row, returning the plaintext key."""
    api_key = generate_api_key()
    pepper = os.environ["API_KEY_PEPPER"].encode("utf-8")
    digest = compute_hmac_digest(pepper, api_key)
    key_prefix = extract_key_prefix(api_key)

    conn = await db.get_connection()
    try:
        team_row = await conn.fetchrow(
            """
            INSERT INTO teams (name, allowed_tiers, rate_limit_rpm, rate_limit_tpm,
                                daily_budget_microdollars, monthly_budget_microdollars)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING team_id
            """,
            f"test-team-{uuid.uuid4()}",
            ["fast", "smart"],
            60,
            100_000,
            1_000_000,
            20_000_000,
        )
        team_id = str(team_row["team_id"])

        await conn.execute(
            """
            INSERT INTO api_keys (team_id, key_prefix, hmac_digest, expires_at, revoked_at)
            VALUES ($1, $2, $3, $4, $5)
            """,
            team_row["team_id"],
            key_prefix,
            digest,
            expires_at,
            revoked_at,
        )
    finally:
        await db.release_connection(conn)

    return _SeededTeam(team_id=team_id, api_key=api_key)


async def _cleanup_team(db: DatabaseConfig, team_id: str) -> None:
    conn = await db.get_connection()
    try:
        await conn.execute("DELETE FROM api_keys WHERE team_id = $1", uuid.UUID(team_id))
        await conn.execute("DELETE FROM teams WHERE team_id = $1", uuid.UUID(team_id))
    finally:
        await db.release_connection(conn)


@pytest.mark.asyncio
async def test_valid_key_returns_200_with_team_config(
    client: httpx.AsyncClient, db: DatabaseConfig
) -> None:
    seeded = await _seed_team(db)
    try:
        response = await client.get("/protected", headers={"X-Api-Key": seeded.api_key})
        assert response.status_code == 200
        body = response.json()
        assert body["team_id"] == seeded.team_id
        assert body["allowed_tiers"] == ["fast", "smart"]
    finally:
        await _cleanup_team(db, seeded.team_id)


@pytest.mark.asyncio
async def test_missing_key_returns_401(client: httpx.AsyncClient) -> None:
    response = await client.get("/protected")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_malformed_key_returns_401(client: httpx.AsyncClient) -> None:
    response = await client.get("/protected", headers={"X-Api-Key": "not-a-valid-key"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_well_formed_but_unknown_key_returns_401(client: httpx.AsyncClient) -> None:
    response = await client.get("/protected", headers={"X-Api-Key": generate_api_key()})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_tampered_key_returns_401(client: httpx.AsyncClient, db: DatabaseConfig) -> None:
    seeded = await _seed_team(db)
    try:
        tampered = seeded.api_key[:-1] + ("a" if seeded.api_key[-1] != "a" else "b")
        response = await client.get("/protected", headers={"X-Api-Key": tampered})
        assert response.status_code == 401
    finally:
        await _cleanup_team(db, seeded.team_id)


@pytest.mark.asyncio
async def test_revoked_key_returns_401(client: httpx.AsyncClient, db: DatabaseConfig) -> None:
    seeded = await _seed_team(db, revoked_at=datetime.now(UTC))
    try:
        response = await client.get("/protected", headers={"X-Api-Key": seeded.api_key})
        assert response.status_code == 401
    finally:
        await _cleanup_team(db, seeded.team_id)


@pytest.mark.asyncio
async def test_expired_key_returns_401(client: httpx.AsyncClient, db: DatabaseConfig) -> None:
    seeded = await _seed_team(db, expires_at=datetime.now(UTC) - timedelta(minutes=1))
    try:
        response = await client.get("/protected", headers={"X-Api-Key": seeded.api_key})
        assert response.status_code == 401
    finally:
        await _cleanup_team(db, seeded.team_id)


@pytest.mark.asyncio
async def test_not_yet_expired_key_returns_200(
    client: httpx.AsyncClient, db: DatabaseConfig
) -> None:
    seeded = await _seed_team(db, expires_at=datetime.now(UTC) + timedelta(hours=1))
    try:
        response = await client.get("/protected", headers={"X-Api-Key": seeded.api_key})
        assert response.status_code == 200
    finally:
        await _cleanup_team(db, seeded.team_id)
