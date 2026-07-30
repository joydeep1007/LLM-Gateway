"""Unit tests for pricing configuration loading and cost estimation."""

from decimal import Decimal

import pytest

from gateway.config.pricing import (
    ModelPricing,
    PricingConfig,
    cost_estimate_microdollars,
    get_model_config,
)
from gateway.domain.models import ModelConfig


class TestPricingConfigLoad:
    """Tests for PricingConfig.load() against the real config/pricing.yaml."""

    def test_loads_default_pricing_file(self) -> None:
        config = PricingConfig.load()
        assert "groq" in config.providers
        assert "gemini" in config.providers
        assert "openrouter" in config.providers

    def test_values_are_decimal_not_float(self) -> None:
        config = PricingConfig.load()
        pricing = config.get_pricing("groq", "llama-3.3-70b-versatile")
        assert isinstance(pricing.cost_per_input_token, Decimal)
        assert isinstance(pricing.cost_per_output_token, Decimal)
        assert pricing.cost_per_input_token == Decimal("0.000005")
        assert pricing.cost_per_output_token == Decimal("0.000015")

    def test_unknown_provider_raises_key_error(self) -> None:
        config = PricingConfig.load()
        with pytest.raises(KeyError):
            config.get_pricing("unknown-provider", "some-model")

    def test_unknown_model_raises_key_error(self) -> None:
        config = PricingConfig.load()
        with pytest.raises(KeyError):
            config.get_pricing("groq", "unknown-model")


class TestGetModelConfig:
    """Tests for get_model_config() microdollar conversion."""

    def test_groq_model_exact_conversion(self) -> None:
        model_config = get_model_config("groq", "llama-3.3-70b-versatile")
        assert isinstance(model_config, ModelConfig)
        assert model_config.provider == "groq"
        assert model_config.model_id == "llama-3.3-70b-versatile"
        assert model_config.quality_tier == "smart"
        assert model_config.max_tokens == 8192
        # 0.000005 * 1_000_000 = 5, 0.000015 * 1_000_000 = 15
        assert model_config.cost_per_input_token_microdollars == 5
        assert model_config.cost_per_output_token_microdollars == 15

    def test_gemini_model_rounds_half_up(self) -> None:
        model_config = get_model_config("gemini", "gemini-2.5-flash")
        # 0.0000015 * 1_000_000 = 1.5 -> rounds to 2
        assert model_config.cost_per_input_token_microdollars == 2
        # 0.0000060 * 1_000_000 = 6
        assert model_config.cost_per_output_token_microdollars == 6

    def test_openrouter_model(self) -> None:
        model_config = get_model_config("openrouter", "openrouter-free")
        assert model_config.cost_per_input_token_microdollars == 2
        assert model_config.cost_per_output_token_microdollars == 6

    def test_unknown_model_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_model_config("groq", "unknown-model")

    def test_uses_provided_config_instance(self) -> None:
        config = PricingConfig(
            providers={
                "custom": {
                    "custom-model": ModelPricing(
                        quality_tier="fast",
                        max_tokens=1024,
                        cost_per_input_token=Decimal("0.000002"),
                        cost_per_output_token=Decimal("0.000004"),
                    )
                }
            }
        )
        model_config = get_model_config("custom", "custom-model", config=config)
        assert model_config.cost_per_input_token_microdollars == 2
        assert model_config.cost_per_output_token_microdollars == 4


class TestCostEstimateMicrodollars:
    """Tests for cost_estimate_microdollars()."""

    def test_known_values(self) -> None:
        model_config = get_model_config("groq", "llama-3.3-70b-versatile")
        # (1000 * 5) + (500 * 15) = 5000 + 7500 = 12500
        result = cost_estimate_microdollars(model_config, input_tokens=1000, max_output_tokens=500)
        assert result == 12500

    def test_zero_tokens_returns_zero(self) -> None:
        model_config = get_model_config("gemini", "gemini-2.5-flash")
        result = cost_estimate_microdollars(model_config, input_tokens=0, max_output_tokens=0)
        assert result == 0

    def test_conservative_estimate_uses_max_output_tokens(self) -> None:
        model_config = get_model_config("openrouter", "openrouter-free")
        # (100 * 2) + (50 * 6) = 200 + 300 = 500
        result = cost_estimate_microdollars(model_config, input_tokens=100, max_output_tokens=50)
        assert result == 500
