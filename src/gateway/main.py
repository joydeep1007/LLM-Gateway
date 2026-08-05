"""FastAPI application entry point for the LLM Gateway."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from gateway.api.completions import router as completions_router
from gateway.providers.mock import MockLLMProvider, MockMode
from gateway.providers.registry import ProviderRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize shared resources on startup and release them on shutdown."""
    registry = ProviderRegistry()
    registry.register("mock", "mock", MockLLMProvider(mode=MockMode.NORMAL))
    app.state.provider_registry = registry
    yield


app = FastAPI(title="LLM Gateway", lifespan=lifespan)
app.include_router(completions_router)
