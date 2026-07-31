"""Unit tests for GeminiProvider — mock the Gemini SDK client."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.domain.exceptions import (
    InvalidRequestError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from gateway.domain.models import ChatCompletionRequest, ChatMessage
from gateway.providers.gemini import GeminiProvider, _translate
from google.genai import errors as gemini_errors

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
    usage = MagicMock(prompt_token_count=5, candidates_token_count=3)
    finish_reason = MagicMock()
    finish_reason.name = "STOP"
    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    resp = MagicMock()
    resp.text = text
    resp.candidates = [candidate]
    resp.usage_metadata = usage
    return resp


def _gemini_provider() -> tuple[GeminiProvider, MagicMock]:
    """Return (provider, mock_client) with the Gemini client patched."""
    with patch("gateway.providers.gemini.genai.Client"):
        provider = GeminiProvider(api_key="test-key")
    mock_client = MagicMock()
    provider._client = mock_client
    return provider, mock_client


def _client_error(code: int, message: str = "test error") -> gemini_errors.ClientError:
    exc = MagicMock(spec=gemini_errors.ClientError)
    exc.code = code  # type: ignore[attr-defined]
    exc.message = message
    return exc


def _server_error(code: int = 500, message: str = "server error") -> gemini_errors.ServerError:
    exc = MagicMock(spec=gemini_errors.ServerError)
    exc.code = code  # type: ignore[attr-defined]
    exc.message = message
    return exc


# ---------------------------------------------------------------------------
# complete() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_returns_response() -> None:
    provider, mock_client = _gemini_provider()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_make_completion_response("hi")
    )

    resp = await provider.complete(REQUEST)

    assert resp.text == "hi"
    assert resp.provider == "gemini"
    assert resp.request_id == "req-1"
    assert resp.input_tokens == 5
    assert resp.output_tokens == 3
    assert resp.finish_reason == "stop"


# ---------------------------------------------------------------------------
# complete() — exception translation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_raises_authentication_error() -> None:
    provider, mock_client = _gemini_provider()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=Exception(401, "bad key")
    )
    with patch(
    "gateway.providers.gemini._translate",
    return_value=ProviderAuthenticationError(
        "bad key",
        provider="gemini",
        model="gemini-2.5-flash",
        status_code=401,
    ),
    ):
        with pytest.raises(ProviderAuthenticationError):
            await provider.complete(REQUEST)


@pytest.mark.asyncio
async def test_complete_raises_rate_limit_error() -> None:
    provider, mock_client = _gemini_provider()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=Exception(429, "rate limited")
    )
    with patch(
        "gateway.providers.gemini._translate",
        return_value=ProviderRateLimitError(
            message="rate limited",
            provider="gemini",
            model="gemini-2.5-flash",
            status_code=429,
        ),
    ):
        with pytest.raises(ProviderRateLimitError):
            await provider.complete(REQUEST)


@pytest.mark.asyncio
async def test_complete_raises_connection_error() -> None:
    provider, mock_client = _gemini_provider()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=OSError("connection refused")
    )

    with pytest.raises(ProviderConnectionError):
        await provider.complete(REQUEST)


@pytest.mark.asyncio
async def test_complete_raises_unavailable_on_5xx() -> None:
    provider, mock_client = _gemini_provider()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=Exception(500, "internal server error")
    )
    with patch(
        "gateway.providers.gemini._translate",
        return_value=ProviderUnavailableError(
            message="internal server error",
            provider="gemini",
            model="gemini-2.5-flash",
            status_code=500,
        ),
    ):
        with pytest.raises(ProviderUnavailableError) as exc_info:
            await provider.complete(REQUEST)

        assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_complete_raises_invalid_request_on_4xx() -> None:
    provider, mock_client = _gemini_provider()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=Exception(400, "bad request")
    )
    
    with patch(
        "gateway.providers.gemini._translate",
        return_value=InvalidRequestError(
            "bad request",
            provider="gemini",
            model="gemini-2.5-flash",
            status_code=400,
        ),
    ):
        with pytest.raises(InvalidRequestError):
            await provider.complete(REQUEST)


# ---------------------------------------------------------------------------
# stream() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_yields_chunks() -> None:
    provider, mock_client = _gemini_provider()

    async def _fake_stream() -> AsyncIterator[Any]:
        for i, text in enumerate(["hel", "lo"]):
            chunk = MagicMock()
            chunk.text = text
            candidate = MagicMock()
            if i == 0:
                candidate.finish_reason = None
            else:
                fr = MagicMock()
                fr.name = "STOP"
                candidate.finish_reason = fr
            chunk.candidates = [candidate]
            yield chunk

    mock_client.aio.models.generate_content_stream = AsyncMock(return_value=_fake_stream())

    chunks = []
    async for chunk in provider.stream(REQUEST):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].delta == "hel"
    assert chunks[0].finish_reason is None
    assert chunks[1].delta == "lo"
    assert chunks[1].finish_reason == "stop"


# ---------------------------------------------------------------------------
# stream() — exception translation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_raises_authentication_error() -> None:
    provider, mock_client = _gemini_provider()
    mock_client.aio.models.generate_content_stream = AsyncMock(
        side_effect=Exception(401, "bad key")
    )
    with patch(
        "gateway.providers.gemini._translate",
        return_value=ProviderAuthenticationError(
            "bad key",
            provider="gemini",
            model="gemini-2.5-flash",
            status_code=401,
        ),
    ):
        with pytest.raises(ProviderAuthenticationError):
            async for _ in provider.stream(REQUEST):
                pass


@pytest.mark.asyncio
async def test_stream_raises_rate_limit_error() -> None:
    provider, mock_client = _gemini_provider()
    mock_client.aio.models.generate_content_stream = AsyncMock(
        side_effect=Exception(429, "rate limited")
    )
    with patch(
        "gateway.providers.gemini._translate",
        return_value=ProviderRateLimitError(
            message="rate limited",
            provider="gemini",
            model="gemini-2.5-flash",
            status_code=429,
        ),
    ):
        with pytest.raises(ProviderRateLimitError):
            async for _ in provider.stream(REQUEST):
                pass


@pytest.mark.asyncio
async def test_stream_raises_connection_error() -> None:
    provider, mock_client = _gemini_provider()
    mock_client.aio.models.generate_content_stream = AsyncMock(
        side_effect=OSError("connection refused")
    )

    with pytest.raises(ProviderConnectionError):
        async for _ in provider.stream(REQUEST):
            pass


@pytest.mark.asyncio
async def test_stream_raises_unavailable_on_5xx() -> None:
    provider, mock_client = _gemini_provider()
    mock_client.aio.models.generate_content_stream = AsyncMock(
        side_effect=Exception(503, "service unavailable")
    )
    with patch(
        "gateway.providers.gemini._translate",
        return_value=ProviderUnavailableError(
            message="service unavailable",
            provider="gemini",
            model="gemini-2.5-flash",
            status_code=503,
        ),
    ):
        with pytest.raises(ProviderUnavailableError):
            async for _ in provider.stream(REQUEST):
                pass


# ---------------------------------------------------------------------------
# health_check()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_healthy() -> None:
    provider, mock_client = _gemini_provider()
    mock_client.aio.models.get = AsyncMock(return_value=MagicMock())

    health = await provider.health_check()

    from gateway.providers.base import HealthStatus

    assert health.status == HealthStatus.HEALTHY
    assert health.latency_ms is not None


@pytest.mark.asyncio
async def test_health_check_down_on_auth_error() -> None:
    provider, mock_client = _gemini_provider()
    mock_client.aio.models.get = AsyncMock(
        side_effect=_client_error(401, "bad key")
    )

    health = await provider.health_check()

    from gateway.providers.base import HealthStatus

    assert health.status == HealthStatus.DOWN
    assert health.error is not None


@pytest.mark.asyncio
async def test_health_check_down_on_server_error() -> None:
    provider, mock_client = _gemini_provider()
    mock_client.aio.models.get = AsyncMock(
        side_effect=_server_error(503, "service unavailable")
    )

    health = await provider.health_check()

    from gateway.providers.base import HealthStatus

    assert health.status == HealthStatus.DOWN
    assert health.error is not None


# ---------------------------------------------------------------------------
# _translate() — exception mapping coverage
# ---------------------------------------------------------------------------


def test_translate_client_error_401() -> None:
    exc = _client_error(401)
    result = _translate(exc, model="gemini-2.5-flash")
    assert isinstance(result, ProviderAuthenticationError)
    assert result.status_code == 401
    assert result.provider == "gemini"


def test_translate_client_error_429() -> None:
    exc = _client_error(429)
    result = _translate(exc, model="gemini-2.5-flash")
    assert isinstance(result, ProviderRateLimitError)
    assert result.status_code == 429


def test_translate_client_error_400() -> None:
    exc = _client_error(400)
    result = _translate(exc, model="gemini-2.5-flash")
    assert isinstance(result, InvalidRequestError)
    assert result.status_code == 400


def test_translate_server_error_500() -> None:
    exc = _server_error(500)
    result = _translate(exc, model="gemini-2.5-flash")
    assert isinstance(result, ProviderUnavailableError)
    assert result.status_code == 500


def test_translate_connection_error() -> None:
    exc = OSError("timed out")
    result = _translate(exc)
    assert isinstance(result, ProviderConnectionError)
    assert result.provider == "gemini"


def test_translate_unknown_exception_passes_through() -> None:
    exc = ValueError("unexpected")
    result = _translate(exc)
    assert result is exc
