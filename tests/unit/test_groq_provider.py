"""Unit tests for GroqProvider — mock the Groq SDK client."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import groq
import pytest

from gateway.domain.exceptions import (
    InvalidRequestError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from gateway.domain.models import ChatCompletionRequest, ChatMessage
from gateway.providers.groq import GroqProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUEST = ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="hello")],
    model_tier="fast",
    max_tokens=100,
    temperature=0.7,
    stream=False,
    team_id="team-1",
    request_id="req-1",
)


def _make_completion_response(text: str = "hi") -> MagicMock:
    usage = MagicMock(prompt_tokens=5, completion_tokens=3)
    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model = "llama3-8b-8192"
    return resp


def _groq_provider() -> tuple[GroqProvider, MagicMock]:
    """Return (provider, mock_client) with the AsyncGroq client patched."""
    provider = GroqProvider(api_key="test-key")
    mock_client = MagicMock()
    provider._client = mock_client
    return provider, mock_client


# ---------------------------------------------------------------------------
# complete() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_returns_response() -> None:
    provider, mock_client = _groq_provider()
    mock_client.chat.completions.create = AsyncMock(return_value=_make_completion_response("hi"))

    resp = await provider.complete(REQUEST)

    assert resp.text == "hi"
    assert resp.provider == "groq"
    assert resp.request_id == "req-1"
    assert resp.input_tokens == 5
    assert resp.output_tokens == 3


# ---------------------------------------------------------------------------
# complete() — exception translation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_raises_authentication_error() -> None:
    provider, mock_client = _groq_provider()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=groq.AuthenticationError("bad key", response=MagicMock(), body={})
    )

    with pytest.raises(ProviderAuthenticationError):
        await provider.complete(REQUEST)


@pytest.mark.asyncio
async def test_complete_raises_rate_limit_error() -> None:
    provider, mock_client = _groq_provider()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=groq.RateLimitError("rate limited", response=MagicMock(), body={})
    )

    with pytest.raises(ProviderRateLimitError):
        await provider.complete(REQUEST)


@pytest.mark.asyncio
async def test_complete_raises_connection_error() -> None:
    provider, mock_client = _groq_provider()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=groq.APIConnectionError(request=MagicMock())
    )

    with pytest.raises(ProviderConnectionError):
        await provider.complete(REQUEST)


@pytest.mark.asyncio
async def test_complete_raises_unavailable_on_5xx() -> None:
    provider, mock_client = _groq_provider()
    http_resp = MagicMock()
    http_resp.status_code = 500
    mock_client.chat.completions.create = AsyncMock(
        side_effect=groq.APIStatusError("server error", response=http_resp, body={})
    )

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.complete(REQUEST)

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_complete_raises_invalid_request_on_4xx() -> None:
    provider, mock_client = _groq_provider()
    http_resp = MagicMock()
    http_resp.status_code = 400
    mock_client.chat.completions.create = AsyncMock(
        side_effect=groq.APIStatusError("bad request", response=http_resp, body={})
    )

    with pytest.raises(InvalidRequestError):
        await provider.complete(REQUEST)


# ---------------------------------------------------------------------------
# stream() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_yields_chunks() -> None:
    provider, mock_client = _groq_provider()

    async def _fake_stream() -> Any:
        for i, text in enumerate(["hel", "lo"]):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = text
            chunk.choices[0].finish_reason = None if i == 0 else "stop"
            yield chunk

    mock_client.chat.completions.create = AsyncMock(return_value=_fake_stream())

    chunks = []
    async for chunk in provider.stream(REQUEST):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].delta == "hel"
    assert chunks[1].delta == "lo"
    assert chunks[1].finish_reason == "stop"


# ---------------------------------------------------------------------------
# stream() — exception translation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_raises_authentication_error() -> None:
    provider, mock_client = _groq_provider()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=groq.AuthenticationError("bad key", response=MagicMock(), body={})
    )

    with pytest.raises(ProviderAuthenticationError):
        async for _ in provider.stream(REQUEST):
            pass


@pytest.mark.asyncio
async def test_stream_raises_rate_limit_error() -> None:
    provider, mock_client = _groq_provider()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=groq.RateLimitError("rate limited", response=MagicMock(), body={})
    )

    with pytest.raises(ProviderRateLimitError):
        async for _ in provider.stream(REQUEST):
            pass


# ---------------------------------------------------------------------------
# health_check()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_healthy() -> None:
    provider, mock_client = _groq_provider()
    mock_client.models.list = AsyncMock(return_value=MagicMock())

    health = await provider.health_check()

    from gateway.providers.base import HealthStatus

    assert health.status == HealthStatus.HEALTHY
    assert health.latency_ms is not None


@pytest.mark.asyncio
async def test_health_check_down_on_auth_error() -> None:
    provider, mock_client = _groq_provider()
    mock_client.models.list = AsyncMock(
        side_effect=groq.AuthenticationError("bad key", response=MagicMock(), body={})
    )

    health = await provider.health_check()

    from gateway.providers.base import HealthStatus

    assert health.status == HealthStatus.DOWN
    assert health.error is not None
