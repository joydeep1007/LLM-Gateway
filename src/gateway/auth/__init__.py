"""Authentication and authorization."""

from gateway.auth.dependencies import get_db_config, get_team_config, reset_db_config
from gateway.auth.models import TeamConfig
from gateway.auth.security import (
    compute_hmac_digest,
    extract_key_prefix,
    generate_api_key,
    validate_key_format,
    verify_api_key,
)

__all__ = [
    "TeamConfig",
    "get_team_config",
    "get_db_config",
    "reset_db_config",
    "compute_hmac_digest",
    "extract_key_prefix",
    "generate_api_key",
    "validate_key_format",
    "verify_api_key",
]
