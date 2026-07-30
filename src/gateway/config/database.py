"""Database configuration and migration runner."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import asyncpg

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent.parent.parent.parent / "database" / "migrations"

_CREATE_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class DatabaseConfig:
    """Manages the asyncpg connection pool and applies pending migrations."""

    def __init__(self, database_url: str | None = None) -> None:
        self._database_url: str = database_url or os.environ["DATABASE_URL"]
        self._pool: asyncpg.Pool | None = None

    async def startup(self) -> None:
        """Create the connection pool and apply any unapplied migrations."""
        self._pool = await asyncpg.create_pool(self._database_url)
        await self._apply_migrations()

    async def shutdown(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def get_connection(self) -> asyncpg.pool.PoolConnectionProxy:
        """Acquire a connection from the pool.

        The caller is responsible for releasing the connection (use as async context manager).
        """
        if self._pool is None:
            raise RuntimeError("DatabaseConfig.startup() has not been called")
        return await self._pool.acquire()

    async def release_connection(self, conn: asyncpg.pool.PoolConnectionProxy) -> None:
        """Release a connection back to the pool."""
        if self._pool is not None:
            await self._pool.release(conn)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _apply_migrations(self) -> None:
        """Apply all unapplied SQL migration files in sorted order."""
        assert self._pool is not None

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(_CREATE_MIGRATIONS_TABLE)

            rows = await conn.fetch("SELECT filename FROM schema_migrations ORDER BY filename")
            applied: set[str] = {row["filename"] for row in rows}

            migration_files = sorted(f for f in _MIGRATIONS_DIR.iterdir() if f.suffix == ".sql")

            for migration_file in migration_files:
                if migration_file.name in applied:
                    logger.debug("Migration already applied: %s", migration_file.name)
                    continue

                sql = migration_file.read_text(encoding="utf-8")
                logger.info("Applying migration: %s", migration_file.name)

                async with conn.transaction():
                    await conn.execute(sql)
                    await conn.execute(
                        "INSERT INTO schema_migrations (filename) VALUES ($1)",
                        migration_file.name,
                    )

                logger.info("Migration applied: %s", migration_file.name)
