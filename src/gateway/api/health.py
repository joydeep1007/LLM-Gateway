"""Health and readiness endpoints for container and orchestrator probes."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

import asyncpg
import redis.asyncio as redis
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])

ReadinessCheck = Callable[[], Awaitable[None]]


async def _default_redis_ready_check() -> None:
    """Verify Redis can respond to PING."""
    client = redis.from_url(os.environ["REDIS_URL"])
    try:
        pong = await client.ping()
        if pong is not True:
            raise RuntimeError("Redis PING returned a non-OK response")
    finally:
        await client.aclose()


async def _default_db_ready_check() -> None:
    """Verify PostgreSQL can accept a connection and execute a trivial query."""
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        await conn.execute("SELECT 1")
    finally:
        await conn.close()


def get_redis_ready_check() -> ReadinessCheck:
    """Return the Redis readiness check function used by /ready."""
    return _default_redis_ready_check


def get_db_ready_check() -> ReadinessCheck:
    """Return the DB readiness check function used by /ready."""
    return _default_db_ready_check


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness endpoint: always return quickly without dependency checks."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(
    redis_ready_check: ReadinessCheck = Depends(get_redis_ready_check),
    db_ready_check: ReadinessCheck = Depends(get_db_ready_check),
) -> JSONResponse:
    """Readiness endpoint: return ready only when Redis and DB are both reachable."""
    try:
        await redis_ready_check()
        await db_ready_check()
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": str(exc)},
        )

    return JSONResponse(status_code=200, content={"status": "ready"})
