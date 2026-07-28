"""Mock provider implementation for testing and local development."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from enum import Enum

from gateway.domain.models import ChatCompletionRequest, ChatCompletionResponse, StreamingChunk
from gateway.providers.base import HealthStatus, LLMProvider, ProviderHealth


class MockMode(Enum):
    """Behavior modes for the mock provider."""

    NORMAL = "NORMAL"


class MockLLMProvider(LLMProvider):
    """Deterministic provider used for tests and local development."""

    def __init__(
        self,
        mode: MockMode = MockMode.NORMAL,
        latency_ms: float = 50.0,
        response_text: str = "Mock response.",
    ) -> None:
        self.mode = mode
        self.latency_ms = latency_ms
        self.response_text = response_text

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
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

    def _estimate_output_tokens(self, response_text: str) -> int:
        return max(1, len(response_text.split()))
