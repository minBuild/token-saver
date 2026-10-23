import pytest

from token_saver.policy import Policy
from token_saver.pricing import (
    PRICES,
    UnknownModelError,
    estimate_cost,
    estimate_tokens,
    jev_cost,
    price_for,
    register_price,
)


def test_every_model_in_the_default_policy_has_a_price():
    for tier in Policy.default().tiers:
        assert tier.model in PRICES


def test_cost_is_list_price_times_tokens():
    assert estimate_cost("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.00)
    assert estimate_cost("claude-opus-5-5", 0, 1_000_000) == pytest.approx(20.00)
    assert estimate_cost("claude-haiku-5-5", 2_000, 800) == pytest.approx(0.0002 + 0.0004)


def test_the_ladder_gets_cheaper_as_it_goes_down():
    ladder = ["claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
    costs = [estimate_cost(model, 1_000, 1_000) for model in ladder]
    assert costs == sorted(costs, reverse=True)


def test_jev_is_a_rounding_error_next_to_the_calls_it_routes():
    assert jev_cost(200) == pytest.approx(0.0000084)
    # Cheaper than even the cheapest tier, and under 1% of a short Sonnet call.
    assert jev_cost(200) < estimate_cost("claude-haiku-5-5", 200, 1)
    assert jev_cost(200) < estimate_cost("claude-sonnet-5-5", 200, 300) / 100


def test_unknown_models_can_be_registered():
    with pytest.raises(UnknownModelError, match="register_price"):
        price_for("claude-imaginary-9")
    register_price("claude-imaginary-9", 1.0, 2.0)
    try:
        assert estimate_cost("claude-imaginary-9", 1_000_000, 1_000_000) == pytest.approx(3.0)
    finally:
        del PRICES["claude-imaginary-9"]


def test_token_estimate_is_rough_but_sane():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("가" * 100) == 50
    assert estimate_tokens("Hello world, " * 10) > estimate_tokens("Hello world")
