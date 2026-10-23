"""Claude list prices and a deliberately simple cost estimate.

Prices are USD per million tokens on the Claude API, copied from Anthropic's pricing page on
:data:`PRICES_AS_OF`. The estimate prices input and output tokens only: cached reads are priced
as plain input, which overstates rather than understates the bill. Thinking tokens are billed
as output, so a higher effort level costs more in practice than this estimate shows; the only
way to know real spend is ``usage`` on a real response, which ``token-saver run`` reports.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICES_AS_OF = "2026-10-06"

#: Jev bills input tokens only; output tokens are free.
JEV_INPUT_PRICE = 0.042


@dataclass(frozen=True)
class Price:
    input: float
    """USD per million input tokens."""
    output: float
    """USD per million output tokens."""


PRICES: dict[str, Price] = {
    "claude-fable-5-1": Price(10.00, 50.00),
    "claude-fable-5": Price(10.00, 50.00),
    "claude-opus-5-5": Price(4.00, 20.00),
    "claude-opus-5": Price(5.00, 25.00),
    "claude-opus-4-8": Price(5.00, 25.00),
    "claude-opus-4-7": Price(5.00, 25.00),
    "claude-opus-4-6": Price(5.00, 25.00),
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "claude-sonnet-5": Price(2.00, 10.00),
    "claude-sonnet-4-6": Price(3.00, 15.00),
    "claude-haiku-5-5": Price(0.10, 0.50),  # prompts up to 100K tokens
    "claude-haiku-4-5": Price(1.00, 5.00),
}


class UnknownModelError(KeyError):
    """No list price is known for this model id."""


def price_for(model: str) -> Price:
    try:
        return PRICES[model]
    except KeyError:
        raise UnknownModelError(
            f"no price known for {model!r}; add it with token_saver.pricing.register_price()"
        ) from None


def register_price(model: str, input_per_mtok: float, output_per_mtok: float) -> None:
    """Teach the estimator about a model that is not in the built-in table."""
    PRICES[model] = Price(input_per_mtok, output_per_mtok)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD for one request, from list prices."""
    price = price_for(model)
    return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


def jev_cost(input_tokens: int) -> float:
    """USD for one Jev classification."""
    return input_tokens * JEV_INPUT_PRICE / 1_000_000


def estimate_tokens(text: str) -> int:
    """A rough token count for sizing estimates, not for billing.

    English runs at about four characters per token; Korean and other non-ASCII scripts are
    counted at two characters per token, which is on the generous side for current tokenizers.
    """
    ascii_chars = sum(1 for ch in text if ch.isascii())
    other_chars = len(text) - ascii_chars
    return max(1, round(ascii_chars / 4 + other_chars / 2))
