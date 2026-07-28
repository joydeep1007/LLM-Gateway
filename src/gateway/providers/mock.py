"""Mock provider implementation for testing and local development."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from enum import Enum

from gateway.domain.exceptions import (
    InvalidProviderResponseError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from gateway.domain.models import ChatCompletionRequest, ChatCompletionResponse, StreamingChunk
from gateway.providers.base import HealthStatus, LLMProvider, ProviderHealth


class MockMode(Enum):
    """Behavior modes for the mock provider."""

    NORMAL = "NORMAL"
    RATE_LIMITED = "RATE_LIMITED"
    SERVER_ERROR = "SERVER_ERROR"
    TIMEOUT = "TIMEOUT"
    CONNECTION_FAIL = "CONNECTION_FAIL"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    OUTAGE = "OUTAGE"
    RECOVERY = "RECOVERY"


class MockLLMProvider(LLMProvider):
    """Deterministic provider used for tests and local development.

    Attributes:
        mode: Behavior mode (NORMAL, RATE_LIMITED, etc.)
        latency_ms: Simulated latency in milliseconds
        response_text: Text to return in NORMAL mode
        status_code: HTTP status code for SERVER_ERROR mode
        recover_after_calls: Number of failures before recovery in RECOVERY mode
    """

    def __init__(
        self,
        mode: MockMode = MockMode.NORMAL,
        latency_ms: float = 50.0,
        response_text: str = "Mock response.",
        status_code: int = 500,
        recover_after_calls: int = 3,
    ) -> None:
        self.mode = mode
        self.latency_ms = latency_ms
        self.response_text = response_text
        self.status_code = status_code
        self.recover_after_calls = recover_after_calls
        self._call_count = 0

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self._call_count += 1
        await self._check_failure_mode()

        await asyncio.sleep(self.latency_ms / 1000.0)

        input_tokens = self._estimate_input_tokens(request)
        output_tokens = self._estimate_output_tokens(self.response_text)

        return ChatCompletionResponse(
            text=self.response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=self.latency_ms,
            model_id="mock",
            provider="mock",
            finish_reason="stop",
            request_id=request.request_id,
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamingChunk]:
        self._call_count += 1
        await self._check_failure_mode()

        words = self.response_text.split()

        for index, word in enumerate(words):
            await asyncio.sleep(self.latency_ms / 1000.0)
            delta = word if index == 0 else f" {word}"
            finish_reason = "stop" if index == len(words) - 1 else None
            yield StreamingChunk(delta=delta, finish_reason=finish_reason, index=index)

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(status=HealthStatus.HEALTHY, latency_ms=self.latency_ms, error=None)

    def _estimate_input_tokens(self, request: ChatCompletionRequest) -> int:
        token_estimate = sum(len(message.content.split()) for message in request.messages)
        token_estimate += len(request.messages) * 2
        return max(1, token_estimate)

    async def _check_failure_mode(self) -> None:
        """Raise appropriate exception based on the configured mode."""
        if self.mode == MockMode.NORMAL:
            return

        if self.mode == MockMode.RATE_LIMITED:
            raise ProviderRateLimitError(
                retry_after_seconds=2.0,
                message="Mock rate limit exceeded",
                provider="mock",
                status_code=429,
            )

        if self.mode == MockMode.SERVER_ERROR:
            raise ProviderUnavailableError(
                status_code=self.status_code,
                message=f"Mock server error {self.status_code}",
                provider="mock",
            )

        if self.mode == MockMode.TIMEOUT:
            await asyncio.sleep(self.latency_ms / 1000.0)
            raise ProviderTimeoutError(
                message="Mock timeout error",
                provider="mock",
            )

        if self.mode == MockMode.CONNECTION_FAIL:
            raise ProviderError(
                message="Mock connection failure",
                provider="mock",
            )

        if self.mode == MockMode.MALFORMED_RESPONSE:
            raise InvalidProviderResponseError(
                message="Mock malformed response",
                provider="mock",
            )

        if self.mode == MockMode.OUTAGE:
            raise ProviderUnavailableError(
                status_code=503,
                message="Mock outage (503)",
                provider="mock",
            )

        if self.mode == MockMode.RECOVERY:
            if self._call_count <= self.recover_after_calls:
                fail_msg = (
                    f"Mock recovery mode failure "
                    f"(call {self._call_count}/{self.recover_after_calls})"
                )
                raise ProviderUnavailableError(
                    status_code=503,
                    message=fail_msg,
                    provider="mock",
                )
            # After recover_after_calls, fall through to normal behavior

    def _estimate_output_tokens(self, response_text: str) -> int:
        return max(1, len(response_text.split()))
