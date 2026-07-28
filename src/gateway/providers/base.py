"""Base interfaces for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import Enum

from pydantic import BaseModel

from gateway.domain.models import ChatCompletionRequest, ChatCompletionResponse, StreamingChunk


class HealthStatus(Enum):
    """Health states reported by providers."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"


class ProviderHealth(BaseModel):
    """Health information returned by providers."""

    status: HealthStatus
    latency_ms: float | None = None
    error: str | None = None


class LLMProvider(ABC):
    """Abstract provider interface for chat completions."""

    @abstractmethod
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Return a full completion response for the request."""

    @abstractmethod
    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamingChunk]:
        """Stream completion chunks for the request."""

    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Report the current provider health."""
