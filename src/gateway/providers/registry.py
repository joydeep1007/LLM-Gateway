"""Registry for provider implementations."""

from __future__ import annotations

from gateway.providers.base import LLMProvider


class ProviderRegistry:
    """Map provider/model pairs to provider instances."""

    def __init__(self) -> None:
        self._providers: dict[tuple[str, str], LLMProvider] = {}

    def register(self, provider: str, model_id: str, implementation: LLMProvider) -> None:
        self._providers[(provider, model_id)] = implementation

    def get(self, provider: str, model_id: str) -> LLMProvider:
        try:
            return self._providers[(provider, model_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown provider/model combination: {provider}/{model_id}") from exc
