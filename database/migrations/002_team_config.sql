-- Migration 002: per-team rate limit and budget configuration
ALTER TABLE teams
    ADD COLUMN IF NOT EXISTS allowed_tiers TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    ADD COLUMN IF NOT EXISTS rate_limit_rpm INTEGER NOT NULL DEFAULT 60,
    ADD COLUMN IF NOT EXISTS rate_limit_tpm INTEGER NOT NULL DEFAULT 100000,
    ADD COLUMN IF NOT EXISTS daily_budget_microdollars BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS monthly_budget_microdollars BIGINT NOT NULL DEFAULT 0;
