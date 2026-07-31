"""Groq provider implementation."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any, cast

import groq
from groq import AsyncGroq, AsyncStream
from groq.types.chat import ChatCompletionMessageParam
from groq.types.chat.chat_completion import ChatCompletion as GroqChatCompletion
from groq.types.chat.chat_completion_chunk import ChatCompletionChunk

from gateway.domain.exceptions import (
    InvalidRequestError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from gateway.domain.models import ChatCompletionRequest, ChatCompletionResponse, StreamingChunk
from gateway.providers.base import HealthStatus, LLMProvider, ProviderHealth

_PROVIDER = "groq"


def _translate(exc: Exception, model: str | None = None) -> Exception:
    """Map Groq SDK exceptions to gateway domain exceptions."""
    if isinstance(exc, groq.AuthenticationError):
        return ProviderAuthenticationError(
            str(exc), provider=_PROVIDER, model=model, status_code=401
        )
    if isinstance(exc, groq.RateLimitError):
        return ProviderRateLimitError(
            message=str(exc), provider=_PROVIDER, model=model, status_code=429
        )
    if isinstance(exc, groq.APIConnectionError):
        return ProviderConnectionError(str(exc), provider=_PROVIDER, model=model)
    if isinstance(exc, groq.APIStatusError):
        status = exc.status_code
        if status >= 500:
            return ProviderUnavailableError(
                status_code=status, message=str(exc), provider=_PROVIDER, model=model
            )
        return InvalidRequestError(str(exc), provider=_PROVIDER, model=model, status_code=status)
    return exc


class GroqProvider(LLMProvider):
    """LLM provider backed by the Groq inference API."""

    def __init__(self, api_key: str, model_id: str = "llama-3.3-70b-versatile") -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model_id = model_id

    def _kwargs(self, request: ChatCompletionRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": cast(
                list[ChatCompletionMessageParam],
                [{"role": m.role, "content": m.content} for m in request.messages],
            ),
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        return kwargs

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        start = time.monotonic()
        try:
            resp = cast(
                GroqChatCompletion,
                await self._client.chat.completions.create(**self._kwargs(request), stream=False),
            )
        except Exception as exc:
            raise _translate(exc, self._model_id) from exc

        latency_ms = (time.monotonic() - start) * 1000.0
        choice = resp.choices[0]
        usage = resp.usage
        return ChatCompletionResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            model_id=resp.model,
            provider=_PROVIDER,
            finish_reason=choice.finish_reason or "stop",
            request_id=request.request_id,
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamingChunk]:
        try:
            sdk_stream = cast(
                AsyncStream[ChatCompletionChunk],
                await self._client.chat.completions.create(**self._kwargs(request), stream=True),
            )
        except Exception as exc:
            raise _translate(exc, self._model_id) from exc

        index = 0
        try:
            async for chunk in sdk_stream:
                choice = chunk.choices[0] if chunk.choices else None
                delta_text = (
                    choice.delta.content if choice and choice.delta and choice.delta.content else ""
                )
                finish_reason = choice.finish_reason if choice else None
                yield StreamingChunk(delta=delta_text, finish_reason=finish_reason, index=index)
                index += 1
        except Exception as exc:
            raise _translate(exc, self._model_id) from exc

    async def health_check(self) -> ProviderHealth:
        start = time.monotonic()
        try:
            await self._client.models.list()
        except groq.AuthenticationError as exc:
            return ProviderHealth(status=HealthStatus.DOWN, error=str(exc))
        except Exception as exc:
            return ProviderHealth(status=HealthStatus.DOWN, error=str(exc))
        latency_ms = (time.monotonic() - start) * 1000.0
        return ProviderHealth(status=HealthStatus.HEALTHY, latency_ms=latency_ms)
