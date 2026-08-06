"""Pydantic models for authenticated team configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TeamConfig(BaseModel):
    """Per-team configuration resolved from an authenticated API key.

    Attributes:
        team_id: Unique identifier of the team owning the API key.
        allowed_tiers: Logical model tiers this team is permitted to use.
        rate_limit_rpm: Requests-per-minute limit for this team.
        rate_limit_tpm: Tokens-per-minute limit for this team.
        daily_budget_microdollars: Maximum spend per day, in microdollars.
        monthly_budget_microdollars: Maximum spend per month, in microdollars.
    """

    team_id: str
    allowed_tiers: list[str]
    rate_limit_rpm: int = Field(gt=0)
    rate_limit_tpm: int = Field(gt=0)
    daily_budget_microdollars: int = Field(ge=0)
    monthly_budget_microdollars: int = Field(ge=0)
