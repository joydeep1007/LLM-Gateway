"""Integration tests for DatabaseConfig and initial migration."""

from __future__ import annotations

import pytest

from gateway.config.database import DatabaseConfig


@pytest.fixture
async def db() -> DatabaseConfig:  # type: ignore[misc]
    """Start DatabaseConfig against the real PostgreSQL instance and tear it down after."""
    config = DatabaseConfig()
    await config.startup()
    yield config  # type: ignore[misc]
    await config.shutdown()


@pytest.mark.asyncio
async def test_migrations_applied(db: DatabaseConfig) -> None:
    """Verify that 001_initial.sql was recorded in schema_migrations."""
    conn = await db.get_connection()
    try:
        row = await conn.fetchrow(
            "SELECT filename FROM schema_migrations WHERE filename = $1",
            "001_initial.sql",
        )
    finally:
        await db.release_connection(conn)

    assert row is not None, "Migration 001_initial.sql was not recorded"


@pytest.mark.asyncio
async def test_teams_table_columns(db: DatabaseConfig) -> None:
    """Verify teams table exists with the expected columns."""
    conn = await db.get_connection()
    try:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'teams'
            ORDER BY ordinal_position
            """
        )
    finally:
        await db.release_connection(conn)

    columns = {row["column_name"]: row for row in rows}
    assert "team_id" in columns, "Missing column: team_id"
    assert "name" in columns, "Missing column: name"
    assert "created_at" in columns, "Missing column: created_at"

    assert columns["name"]["is_nullable"] == "NO"
    assert columns["created_at"]["is_nullable"] == "NO"


@pytest.mark.asyncio
async def test_api_keys_table_columns(db: DatabaseConfig) -> None:
    """Verify api_keys table exists with the expected columns."""
    conn = await db.get_connection()
    try:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'api_keys'
            ORDER BY ordinal_position
            """
        )
    finally:
        await db.release_connection(conn)

    columns = {row["column_name"]: row for row in rows}
    expected = {
        "key_id",
        "team_id",
        "key_prefix",
        "hmac_digest",
        "created_at",
        "expires_at",
        "revoked_at",
    }
    missing = expected - columns.keys()
    assert not missing, f"Missing columns in api_keys: {missing}"

    assert columns["key_id"]["is_nullable"] == "NO"
    assert columns["team_id"]["is_nullable"] == "NO"
    assert columns["key_prefix"]["is_nullable"] == "NO"
    assert columns["hmac_digest"]["is_nullable"] == "NO"
    assert columns["created_at"]["is_nullable"] == "NO"
    # nullable optional columns
    assert columns["expires_at"]["is_nullable"] == "YES"
    assert columns["revoked_at"]["is_nullable"] == "YES"


@pytest.mark.asyncio
async def test_api_keys_prefix_index_exists(db: DatabaseConfig) -> None:
    """Verify the idx_api_keys_prefix index exists."""
    conn = await db.get_connection()
    try:
        row = await conn.fetchrow(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE tablename = 'api_keys' AND indexname = 'idx_api_keys_prefix'
            """
        )
    finally:
        await db.release_connection(conn)

    assert row is not None, "Index idx_api_keys_prefix was not created"


@pytest.mark.asyncio
async def test_migration_idempotent(db: DatabaseConfig) -> None:
    """Running startup() a second time must not raise (migrations already applied)."""
    await db.startup()  # second call — should be a no-op
    await db.shutdown()  # bring pool back to a clean state for the fixture teardown
    # Re-start so the fixture teardown can close properly
    await db.startup()
