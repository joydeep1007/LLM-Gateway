"""Unit tests for API key generation, format validation, and HMAC verification."""

from __future__ import annotations

from gateway.auth.security import (
    KEY_PREFIX,
    KEY_PREFIX_LENGTH,
    compute_hmac_digest,
    extract_key_prefix,
    generate_api_key,
    validate_key_format,
    verify_api_key,
)


class TestGenerateApiKey:
    def test_has_expected_prefix(self) -> None:
        key = generate_api_key()
        assert key.startswith(KEY_PREFIX)

    def test_has_at_least_128_bits_of_entropy(self) -> None:
        key = generate_api_key()
        random_part = key[len(KEY_PREFIX) :]
        assert len(random_part) >= 22

    def test_generates_unique_keys(self) -> None:
        keys = {generate_api_key() for _ in range(100)}
        assert len(keys) == 100

    def test_generated_key_passes_format_validation(self) -> None:
        assert validate_key_format(generate_api_key()) is True


class TestValidateKeyFormat:
    def test_valid_key_passes(self) -> None:
        assert validate_key_format("llmgw_" + "a" * 22) is True

    def test_valid_key_with_url_safe_chars_passes(self) -> None:
        assert validate_key_format("llmgw_" + "A1-_" * 6) is True

    def test_missing_prefix_fails(self) -> None:
        assert validate_key_format("a" * 28) is False

    def test_wrong_prefix_fails(self) -> None:
        assert validate_key_format("otherpre_" + "a" * 22) is False

    def test_too_short_random_part_fails(self) -> None:
        assert validate_key_format("llmgw_" + "a" * 10) is False

    def test_empty_string_fails(self) -> None:
        assert validate_key_format("") is False

    def test_invalid_characters_fail(self) -> None:
        assert validate_key_format("llmgw_" + "a" * 21 + "!") is False

    def test_longer_random_part_still_passes(self) -> None:
        assert validate_key_format("llmgw_" + "a" * 40) is True


class TestExtractKeyPrefix:
    def test_returns_first_twelve_chars(self) -> None:
        key = "llmgw_" + "a" * 22
        prefix = extract_key_prefix(key)
        assert prefix == key[:KEY_PREFIX_LENGTH]
        assert len(prefix) == KEY_PREFIX_LENGTH

    def test_prefix_is_stable_for_same_key(self) -> None:
        key = generate_api_key()
        assert extract_key_prefix(key) == extract_key_prefix(key)


class TestComputeHmacDigest:
    def test_deterministic_for_same_inputs(self) -> None:
        pepper = b"test-pepper"
        key = "llmgw_" + "a" * 22
        assert compute_hmac_digest(pepper, key) == compute_hmac_digest(pepper, key)

    def test_different_keys_produce_different_digests(self) -> None:
        pepper = b"test-pepper"
        digest1 = compute_hmac_digest(pepper, generate_api_key())
        digest2 = compute_hmac_digest(pepper, generate_api_key())
        assert digest1 != digest2

    def test_different_peppers_produce_different_digests(self) -> None:
        key = generate_api_key()
        digest1 = compute_hmac_digest(b"pepper-one", key)
        digest2 = compute_hmac_digest(b"pepper-two", key)
        assert digest1 != digest2

    def test_digest_is_32_bytes(self) -> None:
        digest = compute_hmac_digest(b"test-pepper", generate_api_key())
        assert len(digest) == 32


class TestVerifyApiKey:
    def test_correct_key_and_pepper_verifies(self) -> None:
        pepper = b"test-pepper"
        key = generate_api_key()
        digest = compute_hmac_digest(pepper, key)
        assert verify_api_key(key, digest, pepper) is True

    def test_wrong_key_fails_verification(self) -> None:
        pepper = b"test-pepper"
        digest = compute_hmac_digest(pepper, generate_api_key())
        assert verify_api_key(generate_api_key(), digest, pepper) is False

    def test_wrong_pepper_fails_verification(self) -> None:
        key = generate_api_key()
        digest = compute_hmac_digest(b"correct-pepper", key)
        assert verify_api_key(key, digest, b"wrong-pepper") is False

    def test_tampered_digest_fails_verification(self) -> None:
        pepper = b"test-pepper"
        key = generate_api_key()
        digest = bytearray(compute_hmac_digest(pepper, key))
        digest[0] ^= 0xFF
        assert verify_api_key(key, bytes(digest), pepper) is False
