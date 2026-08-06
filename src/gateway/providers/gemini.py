"""Gemini provider implementation."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import errors as gemini_errors
from google.genai import types as gemini_types

from gateway.domain.exceptions import (
    InvalidRequestError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from gateway.domain.models import ChatCompletionRequest, ChatCompletionResponse, StreamingChunk
from gateway.providers.base import HealthStatus, LLMProvider, ProviderHealth

_PROVIDER = "gemini"
_ROLE_MAP: dict[str, str] = {"assistant": "model"}


def _translate(exc: Exception, model: str | None = None) -> Exception:
    """Map Gemini SDK exceptions to gateway domain exceptions."""
    if isinstance(exc, gemini_errors.ClientError):
        code = getattr(exc, "code", 400) or 400
        if code == 401:
            return ProviderAuthenticationError(
                str(exc), provider=_PROVIDER, model=model, status_code=401
            )
        if code == 429:
            return ProviderRateLimitError(
                message=str(exc), provider=_PROVIDER, model=model, status_code=429
            )
        return InvalidRequestError(str(exc), provider=_PROVIDER, model=model, status_code=code)
    if isinstance(exc, gemini_errors.ServerError):
        code = getattr(exc, "code", 500) or 500
        return ProviderUnavailableError(
            status_code=code, message=str(exc), provider=_PROVIDER, model=model
        )
    if isinstance(exc, (OSError, ConnectionError)):
        return ProviderConnectionError(str(exc), provider=_PROVIDER, model=model)
    return exc


class GeminiProvider(LLMProvider):
    """LLM provider backed by the Google Gemini API."""

    def __init__(self, api_key: str, model_id: str = "gemini-2.5-flash") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_id = model_id

    def _contents(self, request: ChatCompletionRequest) -> list[gemini_types.Content]:
        return [
            gemini_types.Content(
                role=_ROLE_MAP.get(m.role, m.role),
                parts=[gemini_types.Part(text=m.content)],
            )
            for m in request.messages
        ]

    def _config(self, request: ChatCompletionRequest) -> gemini_types.GenerateContentConfig:
        kwargs: dict[str, Any] = {}
        if request.max_tokens is not None:
            kwargs["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        return gemini_types.GenerateContentConfig(**kwargs)

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        start = time.monotonic()
        try:
            resp = await self._client.aio.models.generate_content(
                model=self._model_id,
                contents=self._contents(request),
                config=self._config(request),
            )
        except Exception as exc:
            raise _translate(exc, self._model_id) from exc

        latency_ms = (time.monotonic() - start) * 1000.0
        candidate = resp.candidates[0] if resp.candidates else None
        usage = resp.usage_metadata
        finish_reason = (
            candidate.finish_reason.name.lower()
            if candidate and candidate.finish_reason
            else "stop"
        )
        return ChatCompletionResponse(
            text=resp.text or "",
            input_tokens=int(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=int(usage.candidates_token_count or 0) if usage else 0,
            latency_ms=latency_ms,
            model_id=self._model_id,
            provider=_PROVIDER,
            finish_reason=finish_reason,
            request_id=request.request_id,
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamingChunk]:
        try:
            sdk_stream: AsyncIterator[gemini_types.GenerateContentResponse] = (
                await self._client.aio.models.generate_content_stream(
                    model=self._model_id,
                    contents=self._contents(request),
                    config=self._config(request),
                )
            )
        except Exception as exc:
            raise _translate(exc, self._model_id) from exc

        index = 0
        try:
            async for chunk in sdk_stream:
                candidate = chunk.candidates[0] if chunk.candidates else None
                finish_reason: str | None = (
                    candidate.finish_reason.name.lower()
                    if candidate and candidate.finish_reason
                    else None
                )
                yield StreamingChunk(
                    delta=chunk.text or "",
                    finish_reason=finish_reason,
                    index=index,
                )
                index += 1
        except Exception as exc:
            raise _translate(exc, self._model_id) from exc

    async def health_check(self) -> ProviderHealth:
        start = time.monotonic()
        try:
            await self._client.aio.models.get(model=self._model_id)
        except gemini_errors.ClientError as exc:
            return ProviderHealth(status=HealthStatus.DOWN, error=str(exc))
        except Exception as exc:
            return ProviderHealth(status=HealthStatus.DOWN, error=str(exc))
        latency_ms = (time.monotonic() - start) * 1000.0
        return ProviderHealth(status=HealthStatus.HEALTHY, latency_ms=latency_ms)
