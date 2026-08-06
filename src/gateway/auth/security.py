"""API key generation and HMAC-SHA-256 verification.

Key format: `llmgw_` prefix followed by >=22 URL-safe base64 characters
(>=128 bits of entropy). Keys are never stored in plaintext: only an
HMAC-SHA-256 digest (keyed by a server-side pepper) is persisted, alongside
the first 12 characters of the key as a lookup prefix.
"""

from __future__ import annotations

import hmac
import re
import secrets
from hashlib import sha256

KEY_PREFIX = "llmgw_"
KEY_PREFIX_LENGTH = 12
_RANDOM_BYTES = 16  # 128 bits of entropy

_KEY_FORMAT_RE = re.compile(rf"^{KEY_PREFIX}[A-Za-z0-9_-]{{22,}}$")


def generate_api_key() -> str:
    """Generate a new API key: `llmgw_` + >=22 URL-safe base64 chars (>=128 bits entropy)."""
    return f"{KEY_PREFIX}{secrets.token_urlsafe(_RANDOM_BYTES)}"


def validate_key_format(api_key: str) -> bool:
    """Return True if api_key matches the expected `llmgw_<base64url>` format."""
    return bool(_KEY_FORMAT_RE.match(api_key))


def extract_key_prefix(api_key: str) -> str:
    """Return the first KEY_PREFIX_LENGTH characters of api_key, used as a DB lookup key."""
    return api_key[:KEY_PREFIX_LENGTH]


def compute_hmac_digest(pepper: bytes, api_key: str) -> bytes:
    """Compute the HMAC-SHA-256 digest of api_key, keyed by the server-side pepper."""
    return hmac.new(pepper, api_key.encode("utf-8"), sha256).digest()


def verify_api_key(api_key: str, stored_digest: bytes, pepper: bytes) -> bool:
    """Constant-time comparison of api_key's HMAC digest against stored_digest."""
    computed = compute_hmac_digest(pepper, api_key)
    return hmac.compare_digest(computed, stored_digest)
