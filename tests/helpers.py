"""Builders for System One responses shaped exactly like the API returns them."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from typesafe_sdk import SystemOneResponse

from token_saver.signals import DIFFICULTY_LEVELS, TASK_TYPES


def make_raw_response(
    *,
    task_type: str = "coding",
    task_type_probability: float = 0.8,
    difficulty_probabilities: Sequence[float] = (0.0, 0.1, 0.6, 0.25, 0.05),
    deep_reasoning: float = 0.4,
    sensitive: float | None = 0.02,
    model: str = "jev-1.13.0",
    input_tokens: int = 180,
) -> dict[str, Any]:
    """Build a response body the way the API sends it: string keys, raw floats, usage block."""
    others = [name for name in TASK_TYPES if name != task_type]
    rest = (1.0 - task_type_probability) / len(others)
    task_probabilities = {task_type: task_type_probability, **{name: rest for name in others}}
    n = len(task_probabilities)
    task_confidence = (task_type_probability - 1 / n) / (1 - 1 / n)

    probs = list(difficulty_probabilities)
    score = sum(i * p for i, p in enumerate(probs))
    top = probs.index(max(probs))
    spread = sum(p * abs(i - top) for i, p in enumerate(probs))
    even = sum(abs(i - (len(probs) - 1) / 2) for i in range(len(probs))) / len(probs)
    difficulty_confidence = max(0.0, 1 - spread / even)

    answers: dict[str, Any] = {
        "task_type": {
            "type": "choice",
            "choice": task_type,
            "confidence": task_confidence,
            "probabilities": task_probabilities,
        },
        "difficulty": {
            "type": "score",
            "score": score,
            "confidence": difficulty_confidence,
            "legend": {str(i): DIFFICULTY_LEVELS[i] for i in range(len(probs))},
            "probabilities": {str(i): p for i, p in enumerate(probs)},
        },
        "deep_reasoning": {"type": "noul", "noul": deep_reasoning},
    }
    if sensitive is not None:
        answers["sensitive"] = {"type": "noul", "noul": sensitive}
    return {
        "model": model,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        "answers": answers,
    }


def parse_raw_response(raw: dict[str, Any]) -> SystemOneResponse:
    """Parse the way the SDK does, from JSON text, so integer level keys are coerced."""
    return SystemOneResponse.model_validate_json(json.dumps(raw))


def make_response(**kwargs: Any) -> SystemOneResponse:
    return parse_raw_response(make_raw_response(**kwargs))
