"""Unit tests for health and readiness endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from gateway.api.health import get_db_ready_check, get_redis_ready_check, router


@pytest.fixture
async def app() -> AsyncIterator[FastAPI]:
    """Build a minimal app for health endpoint testing."""
    health_app = FastAPI()
    health_app.include_router(router)
    yield health_app
    health_app.dependency_overrides.clear()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Yield an AsyncClient bound to the in-memory FastAPI app."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_returns_ok_immediately(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_ready_returns_ready_when_redis_and_db_checks_pass(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    redis_mock = AsyncMock()
    redis_mock.ping.return_value = True

    db_mock = AsyncMock()
    db_mock.execute.return_value = "SELECT 1"

    async def redis_check() -> None:
        await redis_mock.ping()

    async def db_check() -> None:
        await db_mock.execute("SELECT 1")

    app.dependency_overrides[get_redis_ready_check] = lambda: redis_check
    app.dependency_overrides[get_db_ready_check] = lambda: db_check

    response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    redis_mock.ping.assert_awaited_once()
    db_mock.execute.assert_awaited_once_with("SELECT 1")


@pytest.mark.asyncio
async def test_ready_returns_503_when_redis_check_fails(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    redis_mock = AsyncMock()
    redis_mock.ping.side_effect = RuntimeError("redis unavailable")

    db_mock = AsyncMock()

    async def redis_check() -> None:
        await redis_mock.ping()

    async def db_check() -> None:
        await db_mock.execute("SELECT 1")

    app.dependency_overrides[get_redis_ready_check] = lambda: redis_check
    app.dependency_overrides[get_db_ready_check] = lambda: db_check

    response = await client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert "redis unavailable" in response.json()["reason"]
    redis_mock.ping.assert_awaited_once()
    db_mock.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_ready_returns_503_when_db_check_fails(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    redis_mock = AsyncMock()
    redis_mock.ping.return_value = True

    db_mock = AsyncMock()
    db_mock.execute.side_effect = RuntimeError("db unavailable")

    async def redis_check() -> None:
        await redis_mock.ping()

    async def db_check() -> None:
        await db_mock.execute("SELECT 1")

    app.dependency_overrides[get_redis_ready_check] = lambda: redis_check
    app.dependency_overrides[get_db_ready_check] = lambda: db_check

    response = await client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert "db unavailable" in response.json()["reason"]
    redis_mock.ping.assert_awaited_once()
    db_mock.execute.assert_awaited_once_with("SELECT 1")
