"""Unit tests for OpenRouterProvider — mock all HTTP calls with respx."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from gateway.domain.exceptions import (
    InvalidRequestError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from gateway.domain.models import ChatCompletionRequest, ChatMessage
from gateway.providers.base import HealthStatus
from gateway.providers.openrouter import OpenRouterProvider

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_BASE_URL = "https://openrouter.ai/api/v1"

REQUEST = ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="hello")],
    model_tier="fast",
    max_tokens=100,
    temperature=0.7,
    stream=False,
    team_id="team-1",
    request_id="req-1",
)


def _provider(
    model_id: str = "openai/gpt-4o-mini",
    http_referer: str | None = None,
    x_title: str | None = None,
) -> OpenRouterProvider:
    return OpenRouterProvider(
        api_key="test-key",
        model_id=model_id,
        base_url=_BASE_URL,
        http_referer=http_referer,
        x_title=x_title,
    )


def _completion_body(
    text: str = "Hello!",
    model: str = "openai/gpt-4o-mini",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-abc",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _sse_lines(*deltas: str, finish_reason: str = "stop") -> str:
    """Build a minimal SSE response body from text deltas."""
    lines: list[str] = []
    for i, delta in enumerate(deltas):
        is_last = i == len(deltas) - 1
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": delta},
                    "finish_reason": finish_reason if is_last else None,
                }
            ]
        }
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# complete() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_complete_returns_response() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion_body("hi"))
    )

    resp = await provider.complete(REQUEST)

    assert resp.text == "hi"
    assert resp.provider == "openrouter"
    assert resp.request_id == "req-1"
    assert resp.input_tokens == 5
    assert resp.output_tokens == 3
    assert resp.finish_reason == "stop"
    assert resp.model_id == "openai/gpt-4o-mini"


@pytest.mark.asyncio
@respx.mock
async def test_complete_includes_model_id_in_request() -> None:
    provider = _provider(model_id="anthropic/claude-3-haiku")
    route = respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion_body(model="anthropic/claude-3-haiku"))
    )

    await provider.complete(REQUEST)

    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "anthropic/claude-3-haiku"


@pytest.mark.asyncio
@respx.mock
async def test_complete_sends_messages_temperature_max_tokens() -> None:
    provider = _provider()
    route = respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion_body())
    )

    await provider.complete(REQUEST)

    sent = json.loads(route.calls[0].request.content)
    assert sent["messages"] == [{"role": "user", "content": "hello"}]
    assert sent["temperature"] == 0.7
    assert sent["max_tokens"] == 100
    assert sent["stream"] is False


@pytest.mark.asyncio
@respx.mock
async def test_complete_defaults_missing_usage_to_zero() -> None:
    provider = _provider()
    body = _completion_body()
    del body["usage"]
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=body)
    )

    resp = await provider.complete(REQUEST)

    assert resp.input_tokens == 0
    assert resp.output_tokens == 0


@pytest.mark.asyncio
@respx.mock
async def test_complete_uses_model_from_response() -> None:
    provider = _provider(model_id="openai/gpt-4o-mini")
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion_body(model="openai/gpt-4o-mini:detail"))
    )

    resp = await provider.complete(REQUEST)

    assert resp.model_id == "openai/gpt-4o-mini:detail"


# ---------------------------------------------------------------------------
# complete() — exception translation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_complete_raises_auth_error_on_401() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(401, text="Unauthorized")
    )

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        await provider.complete(REQUEST)

    assert exc_info.value.status_code == 401
    assert exc_info.value.provider == "openrouter"


@pytest.mark.asyncio
@respx.mock
async def test_complete_raises_auth_error_on_403() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(403, text="Forbidden")
    )

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        await provider.complete(REQUEST)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@respx.mock
async def test_complete_raises_invalid_request_on_400() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(400, text="Bad Request")
    )

    with pytest.raises(InvalidRequestError) as exc_info:
        await provider.complete(REQUEST)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
@respx.mock
async def test_complete_raises_invalid_request_on_404() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(404, text="Not Found")
    )

    with pytest.raises(InvalidRequestError) as exc_info:
        await provider.complete(REQUEST)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
@respx.mock
async def test_complete_raises_rate_limit_on_429() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(429, text="Too Many Requests")
    )

    with pytest.raises(ProviderRateLimitError) as exc_info:
        await provider.complete(REQUEST)

    assert exc_info.value.status_code == 429
    assert exc_info.value.provider == "openrouter"


@pytest.mark.asyncio
@respx.mock
async def test_complete_raises_unavailable_on_500() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.complete(REQUEST)

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
@respx.mock
async def test_complete_raises_unavailable_on_503() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(503, text="Service Unavailable")
    )

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.complete(REQUEST)

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
@respx.mock
async def test_complete_raises_connection_error_on_timeout() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        side_effect=httpx.ReadTimeout("timed out", request=None)  # type: ignore[arg-type]
    )

    with pytest.raises(ProviderConnectionError) as exc_info:
        await provider.complete(REQUEST)

    assert exc_info.value.provider == "openrouter"


@pytest.mark.asyncio
@respx.mock
async def test_complete_raises_connection_error_on_connect_error() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    with pytest.raises(ProviderConnectionError):
        await provider.complete(REQUEST)


@pytest.mark.asyncio
@respx.mock
async def test_complete_raises_connection_error_on_network_error() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        side_effect=httpx.NetworkError("network down")
    )

    with pytest.raises(ProviderConnectionError):
        await provider.complete(REQUEST)


# ---------------------------------------------------------------------------
# stream() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_stream_yields_chunks() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=_sse_lines("hel", "lo").encode(),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    chunks = []
    async for chunk in provider.stream(REQUEST):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].delta == "hel"
    assert chunks[0].finish_reason is None
    assert chunks[1].delta == "lo"
    assert chunks[1].finish_reason == "stop"


@pytest.mark.asyncio
@respx.mock
async def test_stream_includes_model_id_in_request() -> None:
    provider = _provider(model_id="anthropic/claude-3-haiku")
    route = respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=_sse_lines("ok").encode(),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    async for _ in provider.stream(REQUEST):
        pass

    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "anthropic/claude-3-haiku"
    assert sent["stream"] is True


@pytest.mark.asyncio
@respx.mock
async def test_stream_ignores_keepalive_lines() -> None:
    provider = _provider()
    body = ": keep-alive\n\n" + _sse_lines("hi")
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=body.encode(),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    chunks = []
    async for chunk in provider.stream(REQUEST):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].delta == "hi"


@pytest.mark.asyncio
@respx.mock
async def test_stream_stops_at_done() -> None:
    provider = _provider()
    body = _sse_lines("chunk1", "chunk2") + "\ndata: extra\n"
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=body.encode(),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    chunks = []
    async for chunk in provider.stream(REQUEST):
        chunks.append(chunk)

    assert len(chunks) == 2


# ---------------------------------------------------------------------------
# stream() — exception translation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_stream_raises_auth_error_on_401() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(401, text="Unauthorized")
    )

    with pytest.raises(ProviderAuthenticationError):
        async for _ in provider.stream(REQUEST):
            pass


@pytest.mark.asyncio
@respx.mock
async def test_stream_raises_rate_limit_on_429() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(429, text="Too Many Requests")
    )

    with pytest.raises(ProviderRateLimitError):
        async for _ in provider.stream(REQUEST):
            pass


@pytest.mark.asyncio
@respx.mock
async def test_stream_raises_unavailable_on_500() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(500, text="Server Error")
    )

    with pytest.raises(ProviderUnavailableError):
        async for _ in provider.stream(REQUEST):
            pass


@pytest.mark.asyncio
@respx.mock
async def test_stream_raises_connection_error_on_timeout() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        side_effect=httpx.ReadTimeout("timed out", request=None)  # type: ignore[arg-type]
    )

    with pytest.raises(ProviderConnectionError):
        async for _ in provider.stream(REQUEST):
            pass


@pytest.mark.asyncio
@respx.mock
async def test_stream_raises_connection_error_on_connect_error() -> None:
    provider = _provider()
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        side_effect=httpx.ConnectError("refused")
    )

    with pytest.raises(ProviderConnectionError):
        async for _ in provider.stream(REQUEST):
            pass


# ---------------------------------------------------------------------------
# health_check()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_health_check_healthy() -> None:
    provider = _provider()
    respx.get(f"{_BASE_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    health = await provider.health_check()

    assert health.status == HealthStatus.HEALTHY
    assert health.latency_ms is not None
    assert health.error is None


@pytest.mark.asyncio
@respx.mock
async def test_health_check_down_on_401() -> None:
    provider = _provider()
    respx.get(f"{_BASE_URL}/models").mock(
        return_value=httpx.Response(401, text="Unauthorized")
    )

    health = await provider.health_check()

    assert health.status == HealthStatus.DOWN
    assert health.error is not None
    assert "401" in health.error


@pytest.mark.asyncio
@respx.mock
async def test_health_check_down_on_403() -> None:
    provider = _provider()
    respx.get(f"{_BASE_URL}/models").mock(
        return_value=httpx.Response(403, text="Forbidden")
    )

    health = await provider.health_check()

    assert health.status == HealthStatus.DOWN
    assert health.error is not None


@pytest.mark.asyncio
@respx.mock
async def test_health_check_down_on_500() -> None:
    provider = _provider()
    respx.get(f"{_BASE_URL}/models").mock(
        return_value=httpx.Response(500, text="Error")
    )

    health = await provider.health_check()

    assert health.status == HealthStatus.DOWN
    assert health.error is not None


@pytest.mark.asyncio
@respx.mock
async def test_health_check_down_on_timeout() -> None:
    provider = _provider()
    respx.get(f"{_BASE_URL}/models").mock(
        side_effect=httpx.ReadTimeout("timed out", request=None)  # type: ignore[arg-type]
    )

    health = await provider.health_check()

    assert health.status == HealthStatus.DOWN
    assert health.error is not None


@pytest.mark.asyncio
@respx.mock
async def test_health_check_down_on_connect_error() -> None:
    provider = _provider()
    respx.get(f"{_BASE_URL}/models").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    health = await provider.health_check()

    assert health.status == HealthStatus.DOWN
    assert health.error is not None


# ---------------------------------------------------------------------------
# Optional headers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_optional_headers_sent_when_configured() -> None:
    provider = _provider(
        http_referer="https://myapp.example.com",
        x_title="My LLM App",
    )
    route = respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion_body())
    )

    await provider.complete(REQUEST)

    req_headers = route.calls[0].request.headers
    assert req_headers["HTTP-Referer"] == "https://myapp.example.com"
    assert req_headers["X-Title"] == "My LLM App"


@pytest.mark.asyncio
@respx.mock
async def test_optional_headers_absent_when_not_configured() -> None:
    provider = _provider()
    route = respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion_body())
    )

    await provider.complete(REQUEST)

    req_headers = route.calls[0].request.headers
    assert "HTTP-Referer" not in req_headers
    assert "X-Title" not in req_headers


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_does_not_raise() -> None:
    provider = _provider()
    await provider.close()


# ---------------------------------------------------------------------------
# Exception fields — provider and model preserved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_exception_preserves_provider_and_model() -> None:
    model = "anthropic/claude-3-haiku"
    provider = _provider(model_id=model)
    respx.post(f"{_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(429, text="Rate limit")
    )

    with pytest.raises(ProviderRateLimitError) as exc_info:
        await provider.complete(REQUEST)

    assert exc_info.value.provider == "openrouter"
    assert exc_info.value.model == model
