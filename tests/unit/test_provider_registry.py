"""Unit tests for the provider registry."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from gateway.domain.models import ChatCompletionRequest, ChatCompletionResponse, StreamingChunk
from gateway.providers.base import HealthStatus, LLMProvider, ProviderHealth
from gateway.providers.registry import ProviderRegistry


class _StubProvider(LLMProvider):
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise NotImplementedError

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamingChunk]:
        raise NotImplementedError

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(status=HealthStatus.HEALTHY, latency_ms=1.0, error=None)


def test_registry_returns_registered_provider() -> None:
    registry = ProviderRegistry()
    provider = _StubProvider()

    registry.register("openai", "gpt-4o-mini", provider)

    assert registry.get("openai", "gpt-4o-mini") is provider


def test_registry_raises_for_unknown_provider() -> None:
    registry = ProviderRegistry()

    with pytest.raises(KeyError, match="Unknown provider/model combination: openai/gpt-4o-mini"):
        registry.get("openai", "gpt-4o-mini")
