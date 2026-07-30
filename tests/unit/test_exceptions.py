"""Unit tests for provider-related exceptions."""

from gateway.domain import exceptions


def test_provider_error_attributes() -> None:
    err = exceptions.ProviderError(
        "something went wrong",
        provider="openai",
        model="gpt-x",
        status_code=502,
    )
    assert isinstance(err, Exception)
    assert err.message == "something went wrong"
    assert err.provider == "openai"
    assert err.model == "gpt-x"
    assert err.status_code == 502


def test_base_not_retryable() -> None:
    err = exceptions.ProviderError("base")
    assert exceptions.is_retryable(err) is False


def test_timeout_is_retryable() -> None:
    err = exceptions.ProviderTimeoutError("timeout")
    assert isinstance(err, exceptions.ProviderTimeoutError)
    assert exceptions.is_retryable(err) is True


def test_rate_limit_error() -> None:
    err = exceptions.ProviderRateLimitError(retry_after_seconds=1.5)
    assert isinstance(err, exceptions.ProviderRateLimitError)
    assert err.retry_after_seconds == 1.5
    assert exceptions.is_retryable(err) is True


def test_rate_limit_none_retry_after() -> None:
    err = exceptions.ProviderRateLimitError(retry_after_seconds=None)
    assert err.retry_after_seconds is None
    assert exceptions.is_retryable(err) is True


def test_unavailable_error() -> None:
    err = exceptions.ProviderUnavailableError(status_code=503)
    assert isinstance(err, exceptions.ProviderUnavailableError)
    assert err.status_code == 503
    assert exceptions.is_retryable(err) is True


def test_non_retryable_errors() -> None:
    for cls in (
        exceptions.ProviderAuthenticationError,
        exceptions.InvalidRequestError,
        exceptions.InvalidProviderResponseError,
    ):
        err = cls("nope")
        assert exceptions.is_retryable(err) is False
