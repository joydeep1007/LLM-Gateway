-- Migration 001: initial schema
-- teams
CREATE TABLE IF NOT EXISTS teams (
    team_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- api_keys
CREATE TABLE IF NOT EXISTS api_keys (
    key_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id     UUID NOT NULL REFERENCES teams(team_id),
    key_prefix  VARCHAR(12) NOT NULL,
    hmac_digest BYTEA NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ,
    revoked_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);
