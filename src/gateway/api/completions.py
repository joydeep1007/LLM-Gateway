"""POST /v1/chat/completions endpoint."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from gateway.domain.exceptions import ProviderError
from gateway.domain.models import ChatCompletionRequest, ChatMessage
from gateway.providers.base import LLMProvider
from gateway.providers.registry import ProviderRegistry

router = APIRouter(prefix="/v1", tags=["completions"])
logger = structlog.get_logger(__name__)


class ChatCompletionAPIRequest(BaseModel):
    """OpenAI-compatible request body for /v1/chat/completions."""

    messages: list[ChatMessage]
    model: str
    max_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    stream: bool = False


async def _stub_authenticate() -> str:
    """Placeholder auth dependency that always allows the request.

    Returns a fixed team_id until real API-key authentication is implemented.
    """
    return "stub-team"


async def _sse_stream(
    provider: LLMProvider,
    domain_request: ChatCompletionRequest,
    request_id: str,
    attempt_id: str,
) -> AsyncIterator[str]:
    """Yield SSE `data:` lines for each chunk, closing cleanly with an error event on failure."""
    output_tokens = 0
    chunks_sent = 0
    try:
        async for chunk in provider.stream(domain_request):
            chunks_sent += 1
            output_tokens += len(chunk.delta.split())
            yield f"data: {chunk.model_dump_json()}\n\n"
    except ProviderError as exc:
        logger.warning(
            "chat_completion.stream_failed",
            request_id=request_id,
            attempt_id=attempt_id,
            provider=exc.provider,
            error_type=type(exc).__name__,
            chunks_sent=chunks_sent,
        )
        error_payload = {
            "error": {
                "message": exc.message,
                "type": type(exc).__name__,
                "provider": exc.provider,
            }
        }
        yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
        return

    logger.info(
        "chat_completion.stream_completed",
        request_id=request_id,
        attempt_id=attempt_id,
        output_tokens=output_tokens,
        chunks_sent=chunks_sent,
    )


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    body: ChatCompletionAPIRequest, request: Request
) -> JSONResponse | StreamingResponse:
    """Serve a chat completion via the mock provider (no auth/routing yet)."""
    team_id = await _stub_authenticate()
    request_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())

    domain_request = ChatCompletionRequest(
        messages=body.messages,
        model_tier=body.model,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        stream=body.stream,
        team_id=team_id,
        request_id=request_id,
    )

    registry: ProviderRegistry = request.app.state.provider_registry
    provider = registry.get("mock", "mock")

    if body.stream:
        return StreamingResponse(
            _sse_stream(provider, domain_request, request_id, attempt_id),
            media_type="text/event-stream",
            headers={
                "X-Provider-Used": "mock",
                "X-Request-ID": request_id,
                "X-Attempt-ID": attempt_id,
            },
        )

    response = await provider.complete(domain_request)

    return JSONResponse(
        content=response.model_dump(),
        headers={
            "X-Provider-Used": response.provider,
            "X-Request-ID": request_id,
            "X-Attempt-ID": attempt_id,
        },
    )
