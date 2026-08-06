"""Integration tests for the atomic dual-bucket rate-limit Lua scripts (real Redis).

These tests exercise the actual Lua scripts against a real Redis instance (see
`.env` / `infra/docker-compose.yml` for the `redis` service) rather than a fake
in-memory implementation, since Lua/concurrency correctness cannot be trusted to
a Redis emulator.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import redis.asyncio as redis
from redis.commands.core import AsyncScript

from gateway.domain.models import ChatCompletionRequest, ChatMessage
from gateway.providers.mock import MockLLMProvider
from gateway.rate_limiting.lua_scripts import dual_bucket_lua_script, single_bucket_lua_script


@pytest.fixture
async def redis_client() -> AsyncIterator[redis.Redis]:
    """A real Redis client, flushed after each test to keep buckets isolated."""
    # max_connections raised so the 200-coroutine concurrency test can open enough
    # simultaneous connections against the pool.
    client = redis.from_url(os.environ["REDIS_URL"], max_connections=300)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def team_keys() -> tuple[str, str]:
    """A unique RPM/TPM key pair per test to avoid cross-test interference."""
    team_id = f"test-{uuid.uuid4().hex}"
    return f"gateway:rl:{team_id}:rpm", f"gateway:rl:{team_id}:tpm"


async def _seed_bucket(client: redis.Redis, key: str, available: float, now: float) -> None:
    """Pre-seed a bucket's state directly, bypassing the Lua script."""
    await client.hset(key, mapping={"available": available, "last_refill_ts": now})


async def _read_bucket(client: redis.Redis, key: str) -> tuple[float, float]:
    """Read a bucket's raw (available, last_refill_ts) state directly from Redis."""
    available, last_refill_ts = await client.hmget(key, "available", "last_refill_ts")
    assert available is not None and last_refill_ts is not None
    return float(available), float(last_refill_ts)


@pytest.mark.asyncio
async def test_tpm_reject_does_not_consume_rpm(
    redis_client: redis.Redis, team_keys: tuple[str, str]
) -> None:
    """Filling RPM but exhausting TPM must reject admission without touching RPM."""
    rpm_key, tpm_key = team_keys
    now = time.time()

    await _seed_bucket(redis_client, rpm_key, available=10, now=now)
    await _seed_bucket(redis_client, tpm_key, available=0, now=now)

    script = redis_client.register_script(dual_bucket_lua_script)
    result = await script(
        keys=[rpm_key, tpm_key],
        args=[10, 10 / 60.0, 10_000, 10_000 / 60.0, 100, now],
    )

    assert int(result[0]) == 0
    assert len(result) == 2  # allowed, retry_after_ms (no remaining values on reject)

    rpm_available, _ = await _read_bucket(redis_client, rpm_key)
    tpm_available, _ = await _read_bucket(redis_client, tpm_key)
    assert rpm_available == 10  # untouched: still fully available
    assert tpm_available == 0  # untouched: still exhausted


@pytest.mark.asyncio
async def test_rpm_reject_does_not_consume_tpm(
    redis_client: redis.Redis, team_keys: tuple[str, str]
) -> None:
    """Exhausting RPM but filling TPM must reject admission without touching TPM."""
    rpm_key, tpm_key = team_keys
    now = time.time()

    await _seed_bucket(redis_client, rpm_key, available=0, now=now)
    await _seed_bucket(redis_client, tpm_key, available=10_000, now=now)

    script = redis_client.register_script(dual_bucket_lua_script)
    result = await script(
        keys=[rpm_key, tpm_key],
        args=[10, 10 / 60.0, 10_000, 10_000 / 60.0, 100, now],
    )

    assert int(result[0]) == 0
    assert len(result) == 2

    rpm_available, _ = await _read_bucket(redis_client, rpm_key)
    tpm_available, _ = await _read_bucket(redis_client, tpm_key)
    assert rpm_available == 0  # untouched: still exhausted
    assert tpm_available == 10_000  # untouched: still fully available


@pytest.mark.asyncio
async def test_both_pass_both_consumed(
    redis_client: redis.Redis, team_keys: tuple[str, str]
) -> None:
    """When both buckets have capacity, both are consumed by the correct amounts."""
    rpm_key, tpm_key = team_keys
    now = time.time()

    await _seed_bucket(redis_client, rpm_key, available=10, now=now)
    await _seed_bucket(redis_client, tpm_key, available=10_000, now=now)

    script = redis_client.register_script(dual_bucket_lua_script)
    result = await script(
        keys=[rpm_key, tpm_key],
        args=[10, 10 / 60.0, 10_000, 10_000 / 60.0, 5_000, now],
    )

    assert int(result[0]) == 1
    assert float(result[1]) == 9  # 10 - 1 RPM token
    assert float(result[2]) == 5_000  # 10_000 - 5_000 TPM tokens

    rpm_available, _ = await _read_bucket(redis_client, rpm_key)
    tpm_available, _ = await _read_bucket(redis_client, tpm_key)
    assert rpm_available == 9
    assert tpm_available == 5_000


@pytest.mark.asyncio
async def test_concurrent_200_coroutines(
    redis_client: redis.Redis, team_keys: tuple[str, str]
) -> None:
    """200 concurrent admissions at 2x the RPM cap must admit exactly the cap, no more."""
    rpm_key, tpm_key = team_keys
    rpm_capacity = 100
    now = time.time()  # fixed timestamp shared by every call: refill contributes nothing

    script = redis_client.register_script(dual_bucket_lua_script)

    async def attempt() -> int:
        result = await script(
            keys=[rpm_key, tpm_key],
            args=[rpm_capacity, rpm_capacity / 60.0, 10_000_000, 10_000_000 / 60.0, 1, now],
        )
        return int(result[0])

    results = await asyncio.gather(*(attempt() for _ in range(2 * rpm_capacity)))

    assert sum(results) == rpm_capacity  # exactly the cap admitted, zero slip through


@pytest.mark.asyncio
async def test_conservative_tpm_no_refund(
    redis_client: redis.Redis, team_keys: tuple[str, str]
) -> None:
    """TPM tokens reserved at admission are never written back after execution."""
    rpm_key, tpm_key = team_keys
    now = time.time()

    tpm_capacity = 100_000
    estimated_input_tokens = 1_000
    max_output_tokens = 4_000
    estimated_tpm_tokens = estimated_input_tokens + max_output_tokens  # 5000

    await _seed_bucket(redis_client, rpm_key, available=10, now=now)
    await _seed_bucket(redis_client, tpm_key, available=tpm_capacity, now=now)

    script: AsyncScript = redis_client.register_script(dual_bucket_lua_script)
    call_count = 0

    async def admit(tpm_tokens_requested: int) -> list[Any]:
        nonlocal call_count
        call_count += 1
        return await script(
            keys=[rpm_key, tpm_key],
            args=[10, 10 / 60.0, tpm_capacity, tpm_capacity / 60.0, tpm_tokens_requested, now],
        )

    result = await admit(estimated_tpm_tokens)
    assert int(result[0]) == 1

    # Simulate provider execution: actual usage turns out far lower than the estimate.
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model_tier="fast",
        max_tokens=max_output_tokens,
        temperature=0.0,
        stream=False,
        team_id="test-team",
        request_id=str(uuid.uuid4()),
    )
    provider = MockLLMProvider(response_text=" ".join(["word"] * 500))
    response = await provider.complete(request)
    assert response.output_tokens == 500

    actual_total_tokens = response.input_tokens + response.output_tokens
    assert actual_total_tokens < estimated_tpm_tokens  # confirms the scenario is conservative

    # No second Lua operation touches the TPM bucket after provider execution.
    assert call_count == 1

    tpm_available, _ = await _read_bucket(redis_client, tpm_key)
    assert tpm_available == tpm_capacity - estimated_tpm_tokens  # 95_000, NOT 98_500


@pytest.mark.asyncio
async def test_single_bucket_admits_and_consumes(redis_client: redis.Redis) -> None:
    """single_bucket_lua_script admits and consumes when capacity is sufficient."""
    key = f"test-single-{uuid.uuid4().hex}"
    now = time.time()

    script = redis_client.register_script(single_bucket_lua_script)
    result = await script(keys=[key], args=[10, 10 / 60.0, 3, now])

    assert int(result[0]) == 1
    assert float(result[1]) == 7

    available, _ = await _read_bucket(redis_client, key)
    assert available == 7


@pytest.mark.asyncio
async def test_single_bucket_rejects_without_consuming(redis_client: redis.Redis) -> None:
    """single_bucket_lua_script rejects and leaves the bucket untouched when insufficient."""
    key = f"test-single-{uuid.uuid4().hex}"
    now = time.time()

    await _seed_bucket(redis_client, key, available=2, now=now)

    script = redis_client.register_script(single_bucket_lua_script)
    result = await script(keys=[key], args=[10, 10 / 60.0, 3, now])

    assert int(result[0]) == 0
    assert len(result) == 2

    available, _ = await _read_bucket(redis_client, key)
    assert available == 2  # untouched
