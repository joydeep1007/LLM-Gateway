"""Integration tests for POST /v1/chat/completions."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx
import pytest

from gateway.api.completions import _sse_stream
from gateway.domain.exceptions import ProviderUnavailableError
from gateway.domain.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    StreamingChunk,
)
from gateway.main import app
from gateway.providers.base import HealthStatus, LLMProvider, ProviderHealth
from gateway.providers.mock import MockLLMProvider, MockMode


class _MidStreamFailureProvider(LLMProvider):
    """Test double that yields one chunk then raises, to exercise mid-stream failure handling."""

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise NotImplementedError

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamingChunk]:
        yield StreamingChunk(delta="Hello", finish_reason=None, index=0)
        raise ProviderUnavailableError(
            status_code=503, message="mid-stream outage", provider="mock"
        )

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(status=HealthStatus.HEALTHY)


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """Run the app's lifespan and yield an httpx.AsyncClient bound to it via ASGI transport."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_chat_completions_returns_200_with_expected_shape(
    client: httpx.AsyncClient,
) -> None:
    """A valid request returns 200 with a ChatCompletionResponse-shaped JSON body."""
    payload = {
        "messages": [{"role": "user", "content": "Hello there"}],
        "model": "fast",
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False,
    }

    resp = await client.post("/v1/chat/completions", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "Mock response."
    assert body["provider"] == "mock"
    assert body["model_id"] == "mock"
    assert body["finish_reason"] == "stop"
    assert body["input_tokens"] >= 1
    assert body["output_tokens"] >= 1
    assert body["latency_ms"] >= 0
    assert isinstance(body["request_id"], str) and body["request_id"]


@pytest.mark.asyncio
async def test_chat_completions_sets_expected_headers(client: httpx.AsyncClient) -> None:
    """The response includes X-Provider-Used, X-Request-ID, and X-Attempt-ID headers."""
    payload = {
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "fast",
        "max_tokens": 50,
        "temperature": 0.5,
        "stream": False,
    }

    resp = await client.post("/v1/chat/completions", json=payload)

    assert resp.status_code == 200
    assert resp.headers["x-provider-used"] == "mock"
    assert resp.headers["x-request-id"] == resp.json()["request_id"]
    assert resp.headers["x-attempt-id"]


@pytest.mark.asyncio
async def test_stream_endpoint_returns_sse_content_type_and_ordered_chunks(
    client: httpx.AsyncClient,
) -> None:
    """The HTTP response is SSE-typed and carries chunks in order over the full stream."""
    app.state.provider_registry.register(
        "mock",
        "mock",
        MockLLMProvider(mode=MockMode.NORMAL, response_text="one two three four", latency_ms=5.0),
    )
    payload = {
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "fast",
        "max_tokens": 50,
        "temperature": 0.5,
        "stream": True,
    }

    data_lines: list[str] = []
    async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())

    assert len(data_lines) == 4
    chunks = [json.loads(line) for line in data_lines]
    assert [c["delta"] for c in chunks] == ["one", " two", " three", " four"]
    assert chunks[-1]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_sse_stream_yields_chunks_incrementally_not_all_at_once() -> None:
    """The SSE generator yields each chunk as the provider produces it, not buffered upfront.

    httpx's ASGITransport drives the whole app to completion before returning a response, so
    incremental delivery must be verified directly against the generator used by the endpoint.
    """
    provider = MockLLMProvider(
        mode=MockMode.NORMAL, response_text="one two three four", latency_ms=30.0
    )
    domain_request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hi")],
        model_tier="fast",
        max_tokens=50,
        temperature=0.5,
        stream=True,
        team_id="stub-team",
        request_id="req-1",
    )

    arrival_times: list[float] = []
    deltas: list[str] = []
    async for line in _sse_stream(provider, domain_request, "req-1", "attempt-1"):
        arrival_times.append(time.monotonic())
        payload = json.loads(line.removeprefix("data:").strip())
        deltas.append(payload["delta"])

    assert deltas == ["one", " two", " three", " four"]

    gaps = [t2 - t1 for t1, t2 in zip(arrival_times, arrival_times[1:])]
    assert all(gap >= 0.02 for gap in gaps)


@pytest.mark.asyncio
async def test_stream_mid_stream_failure_emits_error_event_without_500(
    client: httpx.AsyncClient,
) -> None:
    """A ProviderError raised after the first chunk yields an SSE error event, not a 500."""
    app.state.provider_registry.register("mock", "mock", _MidStreamFailureProvider())
    payload = {
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "fast",
        "max_tokens": 50,
        "temperature": 0.5,
        "stream": True,
    }

    raw = ""
    async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        async for text in resp.aiter_text():
            raw += text

    events = [event for event in raw.split("\n\n") if event.strip()]
    assert len(events) == 2

    first_data_line = events[0].strip()
    assert first_data_line.startswith("data:")
    first_chunk = json.loads(first_data_line[len("data:") :].strip())
    assert first_chunk["delta"] == "Hello"

    error_event_lines = events[1].splitlines()
    assert any(line.strip() == "event: error" for line in error_event_lines)
    error_data_line = next(line for line in error_event_lines if line.startswith("data:"))
    error_payload = json.loads(error_data_line[len("data:") :].strip())
    assert error_payload["error"]["type"] == "ProviderUnavailableError"
    assert error_payload["error"]["provider"] == "mock"
