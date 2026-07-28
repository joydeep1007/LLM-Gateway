"""Unit tests for domain models."""

import pytest
from pydantic import ValidationError

from gateway.domain.models import (
    AttemptOutcome,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelConfig,
    ProviderAttempt,
    StreamingChunk,
    Usage,
)


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_valid_construction(self) -> None:
        """Test creating a valid ChatMessage."""
        msg = ChatMessage(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"

    def test_all_valid_roles(self) -> None:
        """Test all valid role values."""
        for role in ["system", "user", "assistant"]:
            msg = ChatMessage(role=role, content="test")  # type: ignore[arg-type]
            assert msg.role == role

    def test_invalid_role(self) -> None:
        """Test that invalid role values are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role="invalid", content="test")  # type: ignore[arg-type]
        assert "role" in str(exc_info.value)

    def test_empty_content(self) -> None:
        """Test that empty content is allowed."""
        msg = ChatMessage(role="user", content="")
        assert msg.content == ""


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest model."""

    def test_valid_construction(self) -> None:
        """Test creating a valid ChatCompletionRequest."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello!"),
        ]
        req = ChatCompletionRequest(
            messages=messages,
            model_tier="fast",
            max_tokens=100,
            temperature=0.7,
            stream=False,
            team_id="team-123",
            request_id="req-456",
        )
        assert len(req.messages) == 2
        assert req.model_tier == "fast"
        assert req.max_tokens == 100
        assert req.temperature == 0.7
        assert req.stream is False
        assert req.team_id == "team-123"
        assert req.request_id == "req-456"

    def test_max_tokens_must_be_positive(self) -> None:
        """Test that max_tokens must be greater than 0."""
        messages = [ChatMessage(role="user", content="test")]
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                messages=messages,
                model_tier="fast",
                max_tokens=0,
                temperature=0.5,
                stream=False,
                team_id="team-123",
                request_id="req-456",
            )
        assert "max_tokens" in str(exc_info.value)

    def test_temperature_bounds(self) -> None:
        """Test temperature validation bounds."""
        messages = [ChatMessage(role="user", content="test")]

        # Valid temperature at lower bound
        req = ChatCompletionRequest(
            messages=messages,
            model_tier="fast",
            max_tokens=100,
            temperature=0.0,
            stream=False,
            team_id="team-123",
            request_id="req-456",
        )
        assert req.temperature == 0.0

        # Valid temperature at upper bound
        req = ChatCompletionRequest(
            messages=messages,
            model_tier="fast",
            max_tokens=100,
            temperature=2.0,
            stream=False,
            team_id="team-123",
            request_id="req-456",
        )
        assert req.temperature == 2.0

        # Invalid temperature too low
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=messages,
                model_tier="fast",
                max_tokens=100,
                temperature=-0.1,
                stream=False,
                team_id="team-123",
                request_id="req-456",
            )

        # Invalid temperature too high
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=messages,
                model_tier="fast",
                max_tokens=100,
                temperature=2.1,
                stream=False,
                team_id="team-123",
                request_id="req-456",
            )

    def test_empty_messages_list(self) -> None:
        """Test that empty messages list is allowed (validation is application-level)."""
        req = ChatCompletionRequest(
            messages=[],
            model_tier="fast",
            max_tokens=100,
            temperature=0.5,
            stream=False,
            team_id="team-123",
            request_id="req-456",
        )
        assert len(req.messages) == 0


class TestChatCompletionResponse:
    """Tests for ChatCompletionResponse model."""

    def test_valid_construction(self) -> None:
        """Test creating a valid ChatCompletionResponse."""
        resp = ChatCompletionResponse(
            text="Hello, how can I help you?",
            input_tokens=10,
            output_tokens=8,
            latency_ms=245.5,
            model_id="gpt-3.5-turbo",
            provider="openai",
            finish_reason="stop",
            request_id="req-456",
        )
        assert resp.text == "Hello, how can I help you?"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 8
        assert resp.latency_ms == 245.5
        assert resp.model_id == "gpt-3.5-turbo"
        assert resp.provider == "openai"
        assert resp.finish_reason == "stop"
        assert resp.request_id == "req-456"

    def test_negative_tokens_rejected(self) -> None:
        """Test that negative token counts are rejected."""
        with pytest.raises(ValidationError):
            ChatCompletionResponse(
                text="test",
                input_tokens=-1,
                output_tokens=8,
                latency_ms=100.0,
                model_id="test",
                provider="test",
                finish_reason="stop",
                request_id="req-456",
            )

        with pytest.raises(ValidationError):
            ChatCompletionResponse(
                text="test",
                input_tokens=10,
                output_tokens=-1,
                latency_ms=100.0,
                model_id="test",
                provider="test",
                finish_reason="stop",
                request_id="req-456",
            )

    def test_negative_latency_rejected(self) -> None:
        """Test that negative latency is rejected."""
        with pytest.raises(ValidationError):
            ChatCompletionResponse(
                text="test",
                input_tokens=10,
                output_tokens=8,
                latency_ms=-1.0,
                model_id="test",
                provider="test",
                finish_reason="stop",
                request_id="req-456",
            )

    def test_zero_values_allowed(self) -> None:
        """Test that zero tokens and latency are allowed."""
        resp = ChatCompletionResponse(
            text="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            model_id="test",
            provider="test",
            finish_reason="stop",
            request_id="req-456",
        )
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.latency_ms == 0.0


class TestUsage:
    """Tests for Usage model."""

    def test_valid_construction(self) -> None:
        """Test creating a valid Usage."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cost_microdollars=1500,
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cost_microdollars == 1500

    def test_zero_values_allowed(self) -> None:
        """Test that zero values are allowed."""
        usage = Usage(
            input_tokens=0,
            output_tokens=0,
            cost_microdollars=0,
        )
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cost_microdollars == 0

    def test_negative_values_rejected(self) -> None:
        """Test that negative values are rejected."""
        with pytest.raises(ValidationError):
            Usage(input_tokens=-1, output_tokens=50, cost_microdollars=1500)

        with pytest.raises(ValidationError):
            Usage(input_tokens=100, output_tokens=-1, cost_microdollars=1500)

        with pytest.raises(ValidationError):
            Usage(input_tokens=100, output_tokens=50, cost_microdollars=-1)


class TestStreamingChunk:
    """Tests for StreamingChunk model."""

    def test_valid_construction_with_finish_reason(self) -> None:
        """Test creating a StreamingChunk with finish_reason."""
        chunk = StreamingChunk(
            delta="Hello",
            finish_reason="stop",
            index=5,
        )
        assert chunk.delta == "Hello"
        assert chunk.finish_reason == "stop"
        assert chunk.index == 5

    def test_valid_construction_without_finish_reason(self) -> None:
        """Test creating a StreamingChunk without finish_reason."""
        chunk = StreamingChunk(
            delta="Hello",
            index=0,
        )
        assert chunk.delta == "Hello"
        assert chunk.finish_reason is None
        assert chunk.index == 0

    def test_empty_delta_allowed(self) -> None:
        """Test that empty delta is allowed."""
        chunk = StreamingChunk(delta="", finish_reason=None, index=0)
        assert chunk.delta == ""

    def test_negative_index_rejected(self) -> None:
        """Test that negative index is rejected."""
        with pytest.raises(ValidationError):
            StreamingChunk(delta="test", finish_reason=None, index=-1)


class TestModelConfig:
    """Tests for ModelConfig model."""

    def test_valid_construction(self) -> None:
        """Test creating a valid ModelConfig."""
        config = ModelConfig(
            provider="openai",
            model_id="gpt-3.5-turbo",
            quality_tier="fast",
            max_tokens=4096,
            cost_per_input_token_microdollars=10,
            cost_per_output_token_microdollars=30,
        )
        assert config.provider == "openai"
        assert config.model_id == "gpt-3.5-turbo"
        assert config.quality_tier == "fast"
        assert config.max_tokens == 4096
        assert config.cost_per_input_token_microdollars == 10
        assert config.cost_per_output_token_microdollars == 30

    def test_max_tokens_must_be_positive(self) -> None:
        """Test that max_tokens must be greater than 0."""
        with pytest.raises(ValidationError):
            ModelConfig(
                provider="openai",
                model_id="test",
                quality_tier="fast",
                max_tokens=0,
                cost_per_input_token_microdollars=10,
                cost_per_output_token_microdollars=30,
            )

    def test_negative_costs_rejected(self) -> None:
        """Test that negative costs are rejected."""
        with pytest.raises(ValidationError):
            ModelConfig(
                provider="openai",
                model_id="test",
                quality_tier="fast",
                max_tokens=1000,
                cost_per_input_token_microdollars=-1,
                cost_per_output_token_microdollars=30,
            )

        with pytest.raises(ValidationError):
            ModelConfig(
                provider="openai",
                model_id="test",
                quality_tier="fast",
                max_tokens=1000,
                cost_per_input_token_microdollars=10,
                cost_per_output_token_microdollars=-1,
            )

    def test_zero_cost_allowed(self) -> None:
        """Test that zero cost is allowed (e.g., for mock providers)."""
        config = ModelConfig(
            provider="mock",
            model_id="mock-model",
            quality_tier="fast",
            max_tokens=1000,
            cost_per_input_token_microdollars=0,
            cost_per_output_token_microdollars=0,
        )
        assert config.cost_per_input_token_microdollars == 0
        assert config.cost_per_output_token_microdollars == 0


class TestAttemptOutcome:
    """Tests for AttemptOutcome enum."""

    def test_enum_values(self) -> None:
        """Test that all expected enum values exist."""
        assert AttemptOutcome.SUCCESS == "SUCCESS"
        assert AttemptOutcome.RETRYABLE_FAILURE == "RETRYABLE_FAILURE"
        assert AttemptOutcome.NON_RETRYABLE_FAILURE == "NON_RETRYABLE_FAILURE"
        assert AttemptOutcome.STREAMING_PARTIAL == "STREAMING_PARTIAL"

    def test_enum_membership(self) -> None:
        """Test enum membership checks."""
        assert "SUCCESS" == AttemptOutcome.SUCCESS.value
        assert AttemptOutcome.SUCCESS in AttemptOutcome


class TestProviderAttempt:
    """Tests for ProviderAttempt model."""

    def test_valid_construction_without_result(self) -> None:
        """Test creating a ProviderAttempt before execution."""
        attempt = ProviderAttempt(
            attempt_id="attempt-789",
            request_id="req-456",
            attempt_number=1,
            provider="openai",
            model="gpt-3.5-turbo",
            estimated_cost_microdollars=1000,
        )
        assert attempt.attempt_id == "attempt-789"
        assert attempt.request_id == "req-456"
        assert attempt.attempt_number == 1
        assert attempt.provider == "openai"
        assert attempt.model == "gpt-3.5-turbo"
        assert attempt.estimated_cost_microdollars == 1000
        assert attempt.actual_cost_microdollars is None
        assert attempt.outcome is None

    def test_valid_construction_with_result(self) -> None:
        """Test creating a ProviderAttempt after execution."""
        attempt = ProviderAttempt(
            attempt_id="attempt-789",
            request_id="req-456",
            attempt_number=2,
            provider="openai",
            model="gpt-3.5-turbo",
            estimated_cost_microdollars=1000,
            actual_cost_microdollars=950,
            outcome=AttemptOutcome.SUCCESS,
        )
        assert attempt.actual_cost_microdollars == 950
        assert attempt.outcome == AttemptOutcome.SUCCESS

    def test_attempt_number_must_be_positive(self) -> None:
        """Test that attempt_number must be greater than 0."""
        with pytest.raises(ValidationError):
            ProviderAttempt(
                attempt_id="attempt-789",
                request_id="req-456",
                attempt_number=0,
                provider="openai",
                model="test",
                estimated_cost_microdollars=1000,
            )

    def test_negative_costs_rejected(self) -> None:
        """Test that negative costs are rejected."""
        with pytest.raises(ValidationError):
            ProviderAttempt(
                attempt_id="attempt-789",
                request_id="req-456",
                attempt_number=1,
                provider="openai",
                model="test",
                estimated_cost_microdollars=-1,
            )

        with pytest.raises(ValidationError):
            ProviderAttempt(
                attempt_id="attempt-789",
                request_id="req-456",
                attempt_number=1,
                provider="openai",
                model="test",
                estimated_cost_microdollars=1000,
                actual_cost_microdollars=-1,
            )

    def test_zero_cost_allowed(self) -> None:
        """Test that zero cost is allowed."""
        attempt = ProviderAttempt(
            attempt_id="attempt-789",
            request_id="req-456",
            attempt_number=1,
            provider="mock",
            model="mock-model",
            estimated_cost_microdollars=0,
            actual_cost_microdollars=0,
        )
        assert attempt.estimated_cost_microdollars == 0
        assert attempt.actual_cost_microdollars == 0

    def test_all_outcomes(self) -> None:
        """Test that all outcome values can be assigned."""
        for outcome in AttemptOutcome:
            attempt = ProviderAttempt(
                attempt_id="attempt-789",
                request_id="req-456",
                attempt_number=1,
                provider="openai",
                model="test",
                estimated_cost_microdollars=1000,
                outcome=outcome,
            )
            assert attempt.outcome == outcome
