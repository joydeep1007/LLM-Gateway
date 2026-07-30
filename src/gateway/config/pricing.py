"""Pricing configuration loader.

Loads config/pricing.yaml, keeping pricing values as Decimal, and provides
helpers to build a domain ModelConfig (with microdollar pricing) and to
estimate the conservative pre-request cost of a provider call.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from gateway.domain.models import ModelConfig

_PRICING_FILE = Path(__file__).parent.parent.parent.parent / "config" / "pricing.yaml"

_MICRODOLLARS_PER_DOLLAR = Decimal(1_000_000)


class ModelPricing(BaseModel):
    """Raw pricing entry for a single provider model, as read from YAML."""

    quality_tier: str
    max_tokens: int = Field(gt=0)
    cost_per_input_token: Decimal
    cost_per_output_token: Decimal


class PricingConfig(BaseModel):
    """Pricing configuration for all provider models, keyed by provider then model_id."""

    providers: dict[str, dict[str, ModelPricing]]

    @classmethod
    def load(cls, path: Path | None = None) -> PricingConfig:
        """Load pricing configuration from a YAML file (defaults to config/pricing.yaml)."""
        pricing_path = path or _PRICING_FILE
        with pricing_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    def get_pricing(self, provider: str, model_id: str) -> ModelPricing:
        """Return the raw pricing entry for a provider/model pair."""
        try:
            return self.providers[provider][model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown provider/model combination: {provider}/{model_id}") from exc


def _to_microdollars(value: Decimal) -> int:
    """Convert a Decimal dollar amount to integer microdollars, rounded half-up."""
    return int((value * _MICRODOLLARS_PER_DOLLAR).to_integral_value(rounding=ROUND_HALF_UP))


def get_model_config(
    provider: str, model_id: str, config: PricingConfig | None = None
) -> ModelConfig:
    """Build a ModelConfig for provider/model_id with pricing converted to microdollars."""
    pricing_config = config or PricingConfig.load()
    pricing = pricing_config.get_pricing(provider, model_id)
    return ModelConfig(
        provider=provider,
        model_id=model_id,
        quality_tier=pricing.quality_tier,
        max_tokens=pricing.max_tokens,
        cost_per_input_token_microdollars=_to_microdollars(pricing.cost_per_input_token),
        cost_per_output_token_microdollars=_to_microdollars(pricing.cost_per_output_token),
    )


def cost_estimate_microdollars(
    model_config: ModelConfig, input_tokens: int, max_output_tokens: int
) -> int:
    """Conservative pre-request cost estimate in microdollars.

    Computed as (input_tokens * cost_per_input) + (max_output_tokens * cost_per_output).
    """
    return (
        input_tokens * model_config.cost_per_input_token_microdollars
        + max_output_tokens * model_config.cost_per_output_token_microdollars
    )
