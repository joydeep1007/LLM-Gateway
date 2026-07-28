"""Domain models for LLM Gateway.

This module defines the core domain models used throughout the gateway application.
All models use Pydantic v2 for validation and serialization.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Attributes:
        role: The role of the message sender (system, user, or assistant).
        content: The text content of the message.
    """

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Request for a chat completion from the gateway.

    Attributes:
        messages: List of conversation messages.
        model_tier: Logical model tier (e.g., "fast", "smart", "premium").
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 to 2.0).
        stream: Whether to stream the response.
        team_id: ID of the requesting team.
        request_id: Unique identifier for this request.
    """

    messages: list[ChatMessage]
    model_tier: str
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=2.0)
    stream: bool
    team_id: str
    request_id: str


class ChatCompletionResponse(BaseModel):
    """Response from a chat completion request.

    Attributes:
        text: The generated completion text.
        input_tokens: Number of tokens in the input.
        output_tokens: Number of tokens in the output.
        latency_ms: Request latency in milliseconds.
        model_id: Specific model that generated the response.
        provider: Provider that served the request.
        finish_reason: Reason the generation finished (e.g., "stop", "length").
        request_id: Request identifier from the original request.
    """

    text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    model_id: str
    provider: str
    finish_reason: str
    request_id: str


class Usage(BaseModel):
    """Token usage and cost information.

    Attributes:
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
        cost_microdollars: Total cost in microdollars (millionths of a dollar).
    """

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_microdollars: int = Field(ge=0)


class StreamingChunk(BaseModel):
    """A chunk of a streaming response.

    Attributes:
        delta: The incremental text content in this chunk.
        finish_reason: Reason the stream finished, or None if ongoing.
        index: Sequential index of this chunk in the stream.
    """

    delta: str
    finish_reason: str | None = None
    index: int = Field(ge=0)


class ModelConfig(BaseModel):
    """Configuration for a specific provider model.

    Attributes:
        provider: Provider name (e.g., "openai", "anthropic").
        model_id: Provider-specific model identifier.
        quality_tier: Logical quality tier this model maps to.
        max_tokens: Maximum tokens this model supports.
        cost_per_input_token_microdollars: Cost per input token in microdollars.
        cost_per_output_token_microdollars: Cost per output token in microdollars.
    """

    provider: str
    model_id: str
    quality_tier: str
    max_tokens: int = Field(gt=0)
    cost_per_input_token_microdollars: int = Field(ge=0)
    cost_per_output_token_microdollars: int = Field(ge=0)


class AttemptOutcome(StrEnum):
    """Outcome of a provider attempt.

    Attributes:
        SUCCESS: Request completed successfully.
        RETRYABLE_FAILURE: Request failed but can be retried.
        NON_RETRYABLE_FAILURE: Request failed and should not be retried.
        STREAMING_PARTIAL: Streaming request partially completed before failure.
    """

    SUCCESS = "SUCCESS"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    NON_RETRYABLE_FAILURE = "NON_RETRYABLE_FAILURE"
    STREAMING_PARTIAL = "STREAMING_PARTIAL"


class ProviderAttempt(BaseModel):
    """Record of an attempt to call a provider.

    Each retry or fallback creates a new attempt with a unique attempt_id.

    Attributes:
        attempt_id: Unique identifier for this specific attempt.
        request_id: Identifier of the parent gateway request.
        attempt_number: Sequential attempt number (1-indexed).
        provider: Provider name for this attempt.
        model: Model identifier for this attempt.
        estimated_cost_microdollars: Estimated cost before execution.
        actual_cost_microdollars: Actual cost after execution, or None if unknown.
        outcome: Result of the attempt, or None if not yet completed.
    """

    attempt_id: str
    request_id: str
    attempt_number: int = Field(gt=0)
    provider: str
    model: str
    estimated_cost_microdollars: int = Field(ge=0)
    actual_cost_microdollars: int | None = Field(default=None, ge=0)
    outcome: AttemptOutcome | None = None
