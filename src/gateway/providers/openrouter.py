"""OpenRouter provider implementation."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from gateway.domain.exceptions import (
    InvalidRequestError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from gateway.domain.models import ChatCompletionRequest, ChatCompletionResponse, StreamingChunk
from gateway.providers.base import HealthStatus, LLMProvider, ProviderHealth

_PROVIDER = "openrouter"
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _translate(exc: Exception, model: str | None = None) -> Exception:
    """Map httpx exceptions and HTTP status codes to gateway domain exceptions."""
    if isinstance(exc, httpx.TimeoutException):
        return ProviderConnectionError(str(exc), provider=_PROVIDER, model=model)
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, httpx.ReadError)):
        return ProviderConnectionError(str(exc), provider=_PROVIDER, model=model)
    return exc


def _translate_status(status_code: int, message: str, model: str | None = None) -> Exception:
    """Map an HTTP status code to a gateway domain exception."""
    if status_code in (401, 403):
        return ProviderAuthenticationError(
            message, provider=_PROVIDER, model=model, status_code=status_code
        )
    if status_code == 429:
        return ProviderRateLimitError(
            message=message, provider=_PROVIDER, model=model, status_code=status_code
        )
    if status_code >= 500:
        return ProviderUnavailableError(
            status_code=status_code, message=message, provider=_PROVIDER, model=model
        )
    # 400, 404, and other 4xx
    return InvalidRequestError(
        message, provider=_PROVIDER, model=model, status_code=status_code
    )


class OpenRouterProvider(LLMProvider):
    """LLM provider backed by the OpenRouter REST API."""

    def __init__(
        self,
        api_key: str,
        model_id: str = "openai/gpt-4o-mini",
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 60.0,
        http_referer: str | None = None,
        x_title: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")

        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if x_title:
            headers["X-Title"] = x_title

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
        )

    def _build_payload(self, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        return payload

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        start = time.monotonic()
        try:
            response = await self._client.post(
                "/chat/completions",
                json=self._build_payload(request, stream=False),
            )
        except Exception as exc:
            raise _translate(exc, self._model_id) from exc

        if response.status_code != 200:
            raise _translate_status(
                response.status_code, response.text, self._model_id
            )

        latency_ms = (time.monotonic() - start) * 1000.0
        data = response.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return ChatCompletionResponse(
            text=choice["message"].get("content") or "",
            input_tokens=usage.get("prompt_tokens") or 0,
            output_tokens=usage.get("completion_tokens") or 0,
            latency_ms=latency_ms,
            model_id=data.get("model") or self._model_id,
            provider=_PROVIDER,
            finish_reason=choice.get("finish_reason") or "stop",
            request_id=request.request_id,
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamingChunk]:
        payload = self._build_payload(request, stream=True)
        index = 0
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise _translate_status(
                        response.status_code, response.text, self._model_id
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[len("data:"):].strip()
                    if payload_str == "[DONE]":
                        break
                    if not payload_str:
                        continue
                    try:
                        chunk_data = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk_data.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    delta_text: str = delta.get("content") or ""
                    finish_reason: str | None = choice.get("finish_reason")
                    yield StreamingChunk(delta=delta_text, finish_reason=finish_reason, index=index)
                    index += 1
        except (ProviderAuthenticationError, ProviderRateLimitError,
                ProviderUnavailableError, InvalidRequestError, ProviderConnectionError):
            raise
        except Exception as exc:
            raise _translate(exc, self._model_id) from exc

    async def health_check(self) -> ProviderHealth:
        start = time.monotonic()
        try:
            response = await self._client.get("/models")
        except httpx.TimeoutException as exc:
            return ProviderHealth(status=HealthStatus.DOWN, error=str(exc))
        except Exception as exc:
            return ProviderHealth(status=HealthStatus.DOWN, error=str(exc))

        if response.status_code in (401, 403):
            return ProviderHealth(
                status=HealthStatus.DOWN,
                error=f"Authentication failed (HTTP {response.status_code})",
            )
        if response.status_code >= 500:
            return ProviderHealth(
                status=HealthStatus.DOWN,
                error=f"Server error (HTTP {response.status_code})",
            )

        latency_ms = (time.monotonic() - start) * 1000.0
        return ProviderHealth(status=HealthStatus.HEALTHY, latency_ms=latency_ms)

    async def close(self) -> None:
        """Close the shared AsyncClient."""
        await self._client.aclose()
