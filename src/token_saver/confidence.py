"""Confidence formulas published by TypeSafe, so local classifiers report comparable numbers.

Jev returns a ``confidence`` on every choice and score answer. These are the published
formulas (https://docs.typesafe.ai/confidence), reimplemented so that the offline heuristic
classifier and the tests produce values on the same 0..1 scale as the real model.
"""

from __future__ import annotations

from collections.abc import Sequence


def choice_confidence(probabilities: Sequence[float]) -> float:
    """How far the top probability sits above an even split across ``n`` options.

    Only the top probability counts: ``(0.6, 0.3, 0.1)`` and ``(0.6, 0.2, 0.2)`` both give 0.4.
    """
    n = len(probabilities)
    if n < 2:
        return 1.0
    return (max(probabilities) - 1 / n) / (1 - 1 / n)


def score_confidence(probabilities: Sequence[float]) -> float:
    """Confidence for ordered levels, where probability far from the top level costs more.

    ``probabilities`` must be ordered by level. With three levels, ``(0, 0.5, 0.5)`` gives 0.25
    and ``(0.5, 0, 0.5)`` gives 0: mass on a neighbouring level is a mild disagreement, mass on
    the far end is a real one.
    """
    n = len(probabilities)
    if n < 2:
        return 1.0
    top = max(range(n), key=lambda i: probabilities[i])
    spread = sum(p * abs(i - top) for i, p in enumerate(probabilities))
    even_spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / even_spread)


def noul_confidence(probability: float) -> float:
    """``|2p - 1|``: 0 when the model is torn, 1 when it is sure either way."""
    return abs(2 * probability - 1)
