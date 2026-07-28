"""Unit tests for the mock provider."""

from __future__ import annotations

import asyncio

import pytest

from gateway.domain.models import ChatCompletionRequest, ChatMessage
from gateway.providers.base import HealthStatus
from gateway.providers.mock import MockLLMProvider


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Say hello."),
        ],
        model_tier="fast",
        max_tokens=16,
        temperature=0.2,
        stream=False,
        team_id="team-123",
        request_id="req-123",
    )


@pytest.mark.asyncio
async def test_complete_returns_mock_response(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    provider = MockLLMProvider(latency_ms=75.0, response_text="Mock response.")

    response = await provider.complete(_request())

    assert calls == [0.075]
    assert response.text == "Mock response."
    assert response.input_tokens > 0
    assert response.output_tokens == 2
    assert response.latency_ms == 75.0
    assert response.model_id == "mock"
    assert response.provider == "mock"
    assert response.finish_reason == "stop"
    assert response.request_id == "req-123"


@pytest.mark.asyncio
async def test_stream_yields_word_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    provider = MockLLMProvider(latency_ms=20.0, response_text="Mock response.")

    chunks = [chunk async for chunk in provider.stream(_request())]

    assert calls == [0.02, 0.02]
    assert [chunk.delta for chunk in chunks] == ["Mock", " response."]
    assert [chunk.index for chunk in chunks] == [0, 1]
    assert [chunk.finish_reason for chunk in chunks] == [None, "stop"]

    health = await provider.health_check()

    assert health.status is HealthStatus.HEALTHY
    assert health.latency_ms == 20.0
    assert health.error is None
