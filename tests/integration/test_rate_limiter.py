"""Integration tests for RateLimiter (real Redis).

RateLimiter is a thin wrapper around `dual_bucket_lua_script`; these tests verify
the wrapper wires keys/capacities/refill-rates correctly and preserves the
underlying script's guarantee that a rejection never touches the other bucket.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator

import pytest
import redis.asyncio as redis

from gateway.rate_limiting.limiter import RateLimiter


@pytest.fixture
async def redis_client() -> AsyncIterator[redis.Redis]:
    """A real Redis client."""
    client = redis.from_url(os.environ["REDIS_URL"])
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def team_id() -> str:
    """A unique team_id per test to avoid cross-test key collisions."""
    return f"test-{uuid.uuid4().hex}"


async def _read_bucket(client: redis.Redis, key: str) -> float:
    """Read a bucket's raw "available" field directly from Redis."""
    available = await client.hget(key, "available")
    assert available is not None
    return float(available)


@pytest.mark.asyncio
async def test_admits_and_consumes_both_buckets(redis_client: redis.Redis, team_id: str) -> None:
    """A request within both limits is admitted and consumes both buckets."""
    limiter = RateLimiter(redis_client, team_id, limit_rpm=10, limit_tpm=10_000)

    result = await limiter.check_and_consume(estimated_tpm_tokens=5_000)

    assert result.allowed is True
    assert result.retry_after_ms is None
    assert result.remaining_rpm == 9
    assert result.remaining_tpm == 5_000


@pytest.mark.asyncio
async def test_tpm_rejection_does_not_consume_rpm(redis_client: redis.Redis, team_id: str) -> None:
    """Exhausting TPM must reject admission and leave the RPM bucket untouched."""
    limiter = RateLimiter(redis_client, team_id, limit_rpm=10, limit_tpm=1_000)
    rpm_key = f"gateway:rl:{team_id}:rpm"

    # Exhaust TPM.
    first = await limiter.check_and_consume(estimated_tpm_tokens=1_000)
    assert first.allowed is True
    rpm_after_first_call = await _read_bucket(redis_client, rpm_key)

    # This request would fit in RPM but not in TPM.
    result = await limiter.check_and_consume(estimated_tpm_tokens=1)

    assert result.allowed is False
    assert result.retry_after_ms is not None
    assert result.remaining_rpm is None
    assert result.remaining_tpm is None

    rpm_after_rejection = await _read_bucket(redis_client, rpm_key)
    assert rpm_after_rejection == rpm_after_first_call  # RPM untouched by the rejection


@pytest.mark.asyncio
async def test_rpm_rejection_does_not_consume_tpm(redis_client: redis.Redis, team_id: str) -> None:
    """Exhausting RPM must reject admission and leave the TPM bucket untouched."""
    limiter = RateLimiter(redis_client, team_id, limit_rpm=1, limit_tpm=100_000)
    tpm_key = f"gateway:rl:{team_id}:tpm"

    # Exhaust RPM (capacity 1).
    first = await limiter.check_and_consume(estimated_tpm_tokens=1_000)
    assert first.allowed is True
    tpm_after_first_call = await _read_bucket(redis_client, tpm_key)

    # This request would fit in TPM but not in RPM.
    result = await limiter.check_and_consume(estimated_tpm_tokens=1_000)

    assert result.allowed is False
    assert result.retry_after_ms is not None
    assert result.remaining_rpm is None
    assert result.remaining_tpm is None

    tpm_after_rejection = await _read_bucket(redis_client, tpm_key)
    assert tpm_after_rejection == tpm_after_first_call  # TPM untouched by the rejection
