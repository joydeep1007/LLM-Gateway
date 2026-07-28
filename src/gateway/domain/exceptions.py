"""Domain-specific exceptions for provider interactions.

This module defines provider-related exception types used by the gateway
to classify retryable vs non-retryable failures and to carry provider
context such as provider name, model id and HTTP status codes.
"""
from __future__ import annotations


class ProviderError(Exception):
    """Base error for provider-related failures.

    Attributes:
        message: Human-readable error message.
        provider: Optional provider name (e.g. "openai").
        model: Optional provider model identifier.
        status_code: Optional provider HTTP/status code.
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code


class ProviderTimeoutError(ProviderError):
    """Retryable error indicating the provider request timed out."""


class ProviderRateLimitError(ProviderError):
    """Retryable error indicating the provider rejected the request due to rate limits.

    Attributes:
        retry_after_seconds: Optional suggested delay before retrying.
    """

    def __init__(
        self,
        retry_after_seconds: float | None = None,
        message: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
    ) -> None:
        msg = message or "Provider rate limit exceeded"
        super().__init__(msg, provider=provider, model=model, status_code=status_code)
        self.retry_after_seconds = retry_after_seconds


class ProviderUnavailableError(ProviderError):
    """Retryable error indicating the provider is unavailable (eg. 5xx).

    Attributes:
        status_code: HTTP or provider status code indicating unavailability.
    """

    def __init__(
        self,
        status_code: int,
        message: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        msg = message or f"Provider unavailable (status_code={status_code})"
        super().__init__(msg, provider=provider, model=model, status_code=status_code)
        self.status_code = status_code


class ProviderAuthenticationError(ProviderError):
    """Non-retryable error indicating authentication/authorization failure."""


class InvalidRequestError(ProviderError):
    """Non-retryable error indicating an invalid request to the provider."""


class InvalidProviderResponseError(ProviderError):
    """Non-retryable error for unexpected or malformed provider responses."""


def is_retryable(exc: ProviderError) -> bool:
    """Return True if the given provider error is considered retryable.

    Retryable errors are currently:
    - ProviderTimeoutError
    - ProviderRateLimitError
    - ProviderUnavailableError
    """

    return isinstance(
        exc, (ProviderTimeoutError, ProviderRateLimitError, ProviderUnavailableError)
    )
