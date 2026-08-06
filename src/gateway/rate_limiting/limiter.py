"""RateLimiter: thin async wrapper around the dual-bucket admission Lua script.

All refill/admission/consumption/retry_after_ms logic lives in
`dual_bucket_lua_script` (see PRD Section 7.4 and Section 8.3). This class only
supplies bucket keys, capacities, refill rates, and the current timestamp to the
script, and parses its reply. No token-bucket math is duplicated here.
"""

from __future__ import annotations

import time

import redis.asyncio as redis
from pydantic import BaseModel
from redis.commands.core import AsyncScript

from gateway.rate_limiting.lua_scripts import dual_bucket_lua_script


class RateLimitResult(BaseModel):
    """Outcome of a rate-limit admission check.

    Attributes:
        allowed: Whether the request was admitted.
        remaining_rpm: Remaining RPM tokens after admission, or None if rejected.
        remaining_tpm: Remaining TPM tokens after admission, or None if rejected.
        retry_after_ms: Milliseconds to wait before retrying, or None if admitted.
    """

    allowed: bool
    remaining_rpm: float | None = None
    remaining_tpm: float | None = None
    retry_after_ms: int | None = None


class RateLimiter:
    """Per-team RPM+TPM admission, backed entirely by `dual_bucket_lua_script`."""

    def __init__(
        self,
        redis_client: redis.Redis,
        team_id: str,
        limit_rpm: int,
        limit_tpm: int,
    ) -> None:
        self._redis = redis_client
        self._team_id = team_id
        self._limit_rpm = limit_rpm
        self._limit_tpm = limit_tpm
        self._rpm_key = f"gateway:rl:{team_id}:rpm"
        self._tpm_key = f"gateway:rl:{team_id}:tpm"
        self._script: AsyncScript = redis_client.register_script(dual_bucket_lua_script)

    async def check_and_consume(self, estimated_tpm_tokens: int) -> RateLimitResult:
        """Atomically admit or reject 1 RPM token + estimated_tpm_tokens TPM tokens.

        `estimated_tpm_tokens` must already be computed by the caller as
        `estimated_input_tokens + max_output_tokens` before calling this method.
        """
        now = time.time()
        result = await self._script(
            keys=[self._rpm_key, self._tpm_key],
            args=[
                self._limit_rpm,
                self._limit_rpm / 60.0,
                self._limit_tpm,
                self._limit_tpm / 60.0,
                estimated_tpm_tokens,
                now,
            ],
        )

        if int(result[0]) != 1:
            return RateLimitResult(allowed=False, retry_after_ms=int(result[1]))

        return RateLimitResult(
            allowed=True,
            remaining_rpm=float(result[1]),
            remaining_tpm=float(result[2]),
        )
