"""Unit tests for the TeamConfig model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gateway.auth.models import TeamConfig


def _make_team_config(**overrides: object) -> TeamConfig:
    defaults: dict[str, object] = {
        "team_id": "11111111-1111-1111-1111-111111111111",
        "allowed_tiers": ["fast", "smart"],
        "rate_limit_rpm": 60,
        "rate_limit_tpm": 100_000,
        "daily_budget_microdollars": 1_000_000,
        "monthly_budget_microdollars": 20_000_000,
    }
    defaults.update(overrides)
    return TeamConfig.model_validate(defaults)


class TestTeamConfig:
    def test_valid_config_round_trips(self) -> None:
        config = _make_team_config()
        assert config.team_id == "11111111-1111-1111-1111-111111111111"
        assert config.allowed_tiers == ["fast", "smart"]
        assert config.rate_limit_rpm == 60
        assert config.rate_limit_tpm == 100_000
        assert config.daily_budget_microdollars == 1_000_000
        assert config.monthly_budget_microdollars == 20_000_000

    def test_rate_limit_rpm_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _make_team_config(rate_limit_rpm=0)

    def test_rate_limit_tpm_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _make_team_config(rate_limit_tpm=0)

    def test_daily_budget_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            _make_team_config(daily_budget_microdollars=-1)

    def test_monthly_budget_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            _make_team_config(monthly_budget_microdollars=-1)

    def test_daily_budget_zero_is_allowed(self) -> None:
        config = _make_team_config(daily_budget_microdollars=0)
        assert config.daily_budget_microdollars == 0

    def test_empty_allowed_tiers_is_valid(self) -> None:
        config = _make_team_config(allowed_tiers=[])
        assert config.allowed_tiers == []
