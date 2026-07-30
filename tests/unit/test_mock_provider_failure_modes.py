"""Unit tests for MockLLMProvider failure modes."""

from __future__ import annotations

import pytest

from gateway.domain.exceptions import (
    InvalidProviderResponseError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from gateway.domain.models import ChatCompletionRequest, ChatMessage
from gateway.providers.mock import MockLLMProvider, MockMode


@pytest.fixture
def sample_request() -> ChatCompletionRequest:
    """Create a sample chat completion request for testing."""
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello, world!")],
        model_tier="fast",
        max_tokens=100,
        temperature=0.7,
        stream=False,
        team_id="test-team",
        request_id="test-request-123",
    )


class TestRateLimitedMode:
    """Tests for RATE_LIMITED mode."""

    @pytest.mark.asyncio
    async def test_complete_raises_rate_limit_error(
        self, sample_request: ChatCompletionRequest
    ) -> None:
        provider = MockLLMProvider(mode=MockMode.RATE_LIMITED)

        with pytest.raises(ProviderRateLimitError) as exc_info:
            await provider.complete(sample_request)

        assert exc_info.value.retry_after_seconds == 2.0
        assert exc_info.value.status_code == 429
        assert exc_info.value.provider == "mock"
        assert "rate limit" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_stream_raises_rate_limit_error(
        self, sample_request: ChatCompletionRequest
    ) -> None:
        provider = MockLLMProvider(mode=MockMode.RATE_LIMITED)

        with pytest.raises(ProviderRateLimitError) as exc_info:
            async for _ in provider.stream(sample_request):
                pass

        assert exc_info.value.retry_after_seconds == 2.0


class TestServerErrorMode:
    """Tests for SERVER_ERROR mode."""

    @pytest.mark.asyncio
    async def test_complete_raises_unavailable_error_default_500(
        self, sample_request: ChatCompletionRequest
    ) -> None:
        provider = MockLLMProvider(mode=MockMode.SERVER_ERROR)

        with pytest.raises(ProviderUnavailableError) as exc_info:
            await provider.complete(sample_request)

        assert exc_info.value.status_code == 500
        assert exc_info.value.provider == "mock"

    @pytest.mark.asyncio
    async def test_complete_raises_unavailable_error_custom_status(
        self, sample_request: ChatCompletionRequest
    ) -> None:
        provider = MockLLMProvider(mode=MockMode.SERVER_ERROR, status_code=502)

        with pytest.raises(ProviderUnavailableError) as exc_info:
            await provider.complete(sample_request)

        assert exc_info.value.status_code == 502
        assert "502" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_stream_raises_unavailable_error(
        self, sample_request: ChatCompletionRequest
    ) -> None:
        provider = MockLLMProvider(mode=MockMode.SERVER_ERROR, status_code=503)

        with pytest.raises(ProviderUnavailableError) as exc_info:
            async for _ in provider.stream(sample_request):
                pass

        assert exc_info.value.status_code == 503


class TestTimeoutMode:
    """Tests for TIMEOUT mode."""

    @pytest.mark.asyncio
    async def test_complete_raises_timeout_error(
        self, sample_request: ChatCompletionRequest
    ) -> None:
        provider = MockLLMProvider(mode=MockMode.TIMEOUT, latency_ms=10.0)

        with pytest.raises(ProviderTimeoutError) as exc_info:
            await provider.complete(sample_request)

        assert exc_info.value.provider == "mock"
        assert "timeout" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_stream_raises_timeout_error(self, sample_request: ChatCompletionRequest) -> None:
        provider = MockLLMProvider(mode=MockMode.TIMEOUT)

        with pytest.raises(ProviderTimeoutError):
            async for _ in provider.stream(sample_request):
                pass

    @pytest.mark.asyncio
    async def test_timeout_mode_simulates_delay(
        self, sample_request: ChatCompletionRequest
    ) -> None:
        """Verify that TIMEOUT mode actually waits before raising the error."""
        import time

        provider = MockLLMProvider(mode=MockMode.TIMEOUT, latency_ms=50.0)
        start = time.time()

        with pytest.raises(ProviderTimeoutError):
            await provider.complete(sample_request)

        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms >= 40.0  # Allow some margin


class TestConnectionFailMode:
    """Tests for CONNECTION_FAIL mode."""

    @pytest.mark.asyncio
    async def test_complete_raises_provider_error(
        self, sample_request: ChatCompletionRequest
    ) -> None:
        provider = MockLLMProvider(mode=MockMode.CONNECTION_FAIL)

        with pytest.raises(ProviderError) as exc_info:
            await provider.complete(sample_request)

        assert exc_info.value.provider == "mock"
        assert "connection" in exc_info.value.message.lower()
        # CONNECTION_FAIL should NOT be ProviderUnavailableError or ProviderTimeoutError
        assert not isinstance(exc_info.value, ProviderUnavailableError)
        assert not isinstance(exc_info.value, ProviderTimeoutError)

    @pytest.mark.asyncio
    async def test_stream_raises_provider_error(
        self, sample_request: ChatCompletionRequest
    ) -> None:
        provider = MockLLMProvider(mode=MockMode.CONNECTION_FAIL)

        with pytest.raises(ProviderError):
            async for _ in provider.stream(sample_request):
                pass


class TestMalformedResponseMode:
    """Tests for MALFORMED_RESPONSE mode."""

    @pytest.mark.asyncio
    async def test_complete_raises_invalid_response_error(
        self, sample_request: ChatCompletionRequest
    ) -> None:
        provider = MockLLMProvider(mode=MockMode.MALFORMED_RESPONSE)

        with pytest.raises(InvalidProviderResponseError) as exc_info:
            await provider.complete(sample_request)

        assert exc_info.value.provider == "mock"
        assert "malformed" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_stream_raises_invalid_response_error(
        self, sample_request: ChatCompletionRequest
    ) -> None:
        provider = MockLLMProvider(mode=MockMode.MALFORMED_RESPONSE)

        with pytest.raises(InvalidProviderResponseError):
            async for _ in provider.stream(sample_request):
                pass


class TestOutageMode:
    """Tests for OUTAGE mode."""

    @pytest.mark.asyncio
    async def test_complete_always_raises_503(self, sample_request: ChatCompletionRequest) -> None:
        provider = MockLLMProvider(mode=MockMode.OUTAGE)

        # Should consistently fail with 503
        for _ in range(3):
            with pytest.raises(ProviderUnavailableError) as exc_info:
                await provider.complete(sample_request)

            assert exc_info.value.status_code == 503
            assert "outage" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_stream_always_raises_503(self, sample_request: ChatCompletionRequest) -> None:
        provider = MockLLMProvider(mode=MockMode.OUTAGE)

        with pytest.raises(ProviderUnavailableError) as exc_info:
            async for _ in provider.stream(sample_request):
                pass

        assert exc_info.value.status_code == 503


class TestRecoveryMode:
    """Tests for RECOVERY mode."""

    @pytest.mark.asyncio
    async def test_complete_fails_then_recovers_default(
        self, sample_request: ChatCompletionRequest
    ) -> None:
        provider = MockLLMProvider(mode=MockMode.RECOVERY)

        # First 3 calls should fail (default recover_after_calls=3)
        for call_num in range(1, 4):
            with pytest.raises(ProviderUnavailableError) as exc_info:
                await provider.complete(sample_request)

            assert exc_info.value.status_code == 503
            assert f"call {call_num}/3" in exc_info.value.message

        # Fourth call should succeed
        response = await provider.complete(sample_request)
        assert response.text == "Mock response."
        assert response.provider == "mock"

        # Subsequent calls should also succeed
        response2 = await provider.complete(sample_request)
        assert response2.text == "Mock response."

    @pytest.mark.asyncio
    async def test_complete_fails_then_recovers_custom_count(
        self, sample_request: ChatCompletionRequest
    ) -> None:
        provider = MockLLMProvider(mode=MockMode.RECOVERY, recover_after_calls=2)

        # First 2 calls should fail
        for call_num in range(1, 3):
            with pytest.raises(ProviderUnavailableError) as exc_info:
                await provider.complete(sample_request)

            assert f"call {call_num}/2" in exc_info.value.message

        # Third call should succeed
        response = await provider.complete(sample_request)
        assert response.text == "Mock response."

    @pytest.mark.asyncio
    async def test_stream_fails_then_recovers(self, sample_request: ChatCompletionRequest) -> None:
        provider = MockLLMProvider(mode=MockMode.RECOVERY, recover_after_calls=1)

        # First call should fail
        with pytest.raises(ProviderUnavailableError):
            async for _ in provider.stream(sample_request):
                pass

        # Second call should succeed
        chunks = []
        async for chunk in provider.stream(sample_request):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert chunks[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_recovery_mode_zero_failures(self, sample_request: ChatCompletionRequest) -> None:
        """Test RECOVERY mode with recover_after_calls=0 (immediate success)."""
        provider = MockLLMProvider(mode=MockMode.RECOVERY, recover_after_calls=0)

        # First call should succeed
        response = await provider.complete(sample_request)
        assert response.text == "Mock response."


class TestNormalModeUnchanged:
    """Verify that NORMAL mode behavior is preserved."""

    @pytest.mark.asyncio
    async def test_complete_returns_response(self, sample_request: ChatCompletionRequest) -> None:
        provider = MockLLMProvider(mode=MockMode.NORMAL)

        response = await provider.complete(sample_request)

        assert response.text == "Mock response."
        assert response.provider == "mock"
        assert response.model_id == "mock"
        assert response.finish_reason == "stop"
        assert response.input_tokens > 0
        assert response.output_tokens > 0

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, sample_request: ChatCompletionRequest) -> None:
        provider = MockLLMProvider(mode=MockMode.NORMAL, response_text="Hello world")

        chunks = []
        async for chunk in provider.stream(sample_request):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " world"
        assert chunks[1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_custom_response_text(self, sample_request: ChatCompletionRequest) -> None:
        provider = MockLLMProvider(mode=MockMode.NORMAL, response_text="Custom test response here")

        response = await provider.complete(sample_request)
        assert response.text == "Custom test response here"


class TestHealthCheckWithFailureModes:
    """Test health_check behavior with different modes."""

    @pytest.mark.asyncio
    async def test_health_check_succeeds_in_normal_mode(self) -> None:
        provider = MockLLMProvider(mode=MockMode.NORMAL, latency_ms=25.0)
        health = await provider.health_check()

        assert health.status.value == "HEALTHY"
        assert health.latency_ms == 25.0
        assert health.error is None

    @pytest.mark.asyncio
    async def test_health_check_succeeds_in_failure_modes(self) -> None:
        """health_check should succeed regardless of mode (it doesn't call _check_failure_mode)."""
        for mode in [
            MockMode.RATE_LIMITED,
            MockMode.SERVER_ERROR,
            MockMode.TIMEOUT,
            MockMode.CONNECTION_FAIL,
            MockMode.MALFORMED_RESPONSE,
            MockMode.OUTAGE,
        ]:
            provider = MockLLMProvider(mode=mode)
            health = await provider.health_check()
            assert health.status.value == "HEALTHY"
