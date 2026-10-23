"""The questions token-saver asks about a prompt, and the typed signals it gets back.

Every classifier (Jev, the offline heuristic, recorded fixtures) answers the same four
questions, so the routing policy never needs to know where the signals came from:

* ``task_type``       - a ``choice``: what kind of work is being asked for
* ``difficulty``      - a ``score``: how hard it is to answer well, on five described levels
* ``deep_reasoning``  - a ``noul``: would a careful expert need several steps of thought
* ``sensitive``       - a ``noul``: does the prompt carry secrets or personal data (flag only)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
)

QUESTION_TASK_TYPE = "task_type"
QUESTION_DIFFICULTY = "difficulty"
QUESTION_DEEP_REASONING = "deep_reasoning"
QUESTION_SENSITIVE = "sensitive"

# Each option gets a one-line description, as the TypeSafe docs recommend: the model sees the
# names and the descriptions, so they should separate neighbouring options from each other.
TASK_TYPES: dict[str, str] = {
    "chitchat": "Greetings, thanks, small talk, or a one-line reaction with no task to do",
    "factual": "A short factual or how-to question with a well-known answer",
    "rewrite": "Editing, translating, summarizing, or reformatting text the user provides",
    "coding": "Writing, fixing, reviewing, or explaining code",
    "analysis": "Reasoning through a problem: planning, comparing options, math, multi-step logic",
    "creative": "Original writing such as stories, poems, or marketing copy",
    "agentic": "A long task that needs many steps, tools, or files to finish",
}

# Ordered from easiest to hardest. Levels describe situations rather than degrees, because a
# score model matches each level description on its own and never sees the level numbers.
DIFFICULTY_LEVELS: tuple[str, ...] = (
    "Trivial: one sentence from common knowledge answers it, no reasoning needed",
    "Simple: one clear step, such as a short rewrite, a small code snippet, or a definition",
    "Moderate: a few steps of reasoning, or a medium piece of code with some design choices",
    "Hard: careful multi-step reasoning, debugging across components, or real trade-offs",
    "Expert: research-level depth, large system design, or subtle correctness where a mistake "
    "is costly",
)

MAX_DIFFICULTY_LEVEL = len(DIFFICULTY_LEVELS) - 1


def build_questions() -> dict[str, Choice | Score | Noul]:
    """The System One question set, sent in a single Jev call (questions run in parallel)."""
    return {
        QUESTION_TASK_TYPE: Choice(
            instructions="What kind of work does the user ask the AI assistant to do?",
            criteria=dict(TASK_TYPES),
        ),
        QUESTION_DIFFICULTY: Score(
            instructions="How hard is it to answer this request well?",
            criteria=list(DIFFICULTY_LEVELS),
        ),
        QUESTION_DEEP_REASONING: Noul(
            instructions=(
                "Would a careful expert need to think through several steps before answering "
                "this well?"
            ),
        ),
        QUESTION_SENSITIVE: Noul(
            instructions=(
                "Does the message contain secrets, credentials, personal data, or confidential "
                "business information?"
            ),
        ),
    }


def build_state(prompt: str) -> dict[str, str]:
    """Wrap the prompt in a named field, which the TypeSafe docs recommend over bare strings."""
    return {"user_message": prompt}


class SignalError(ValueError):
    """A classifier response did not contain the answers the policy needs."""


@dataclass(frozen=True)
class Signals:
    """What a classifier concluded about one prompt."""

    task_type: str
    task_type_confidence: float
    task_type_probabilities: dict[str, float]
    difficulty: float
    """Probability-weighted difficulty level, ``0.0 .. MAX_DIFFICULTY_LEVEL``, may be fractional."""
    difficulty_confidence: float
    difficulty_probabilities: tuple[float, ...]
    """One probability per level, ordered from easiest to hardest."""
    deep_reasoning: float
    """Probability that answering well needs several steps of thought."""
    sensitive: float
    """Probability that the prompt contains secrets or personal data. Informational only."""
    source: str
    """Which classifier produced this: ``jev``, ``heuristic`` or ``fixture``."""
    model: str | None = None
    """The versioned model that answered, e.g. ``jev-1.13.0``, when a model was involved."""
    latency_ms: float | None = None
    input_tokens: int | None = None
    fallback_reason: str | None = None
    """Set when the preferred classifier failed and a fallback produced these signals."""

    @property
    def difficulty_level(self) -> int:
        """The nearest discrete level, rounding half up so ties escalate rather than relax."""
        return max(0, min(MAX_DIFFICULTY_LEVEL, math.floor(self.difficulty + 0.5)))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["difficulty_level"] = self.difficulty_level
        data["difficulty_probabilities"] = list(self.difficulty_probabilities)
        return data


A = TypeVar("A", ChoiceAnswer, ScoreAnswer, NoulAnswer)


def _expect(answers: Mapping[str, object], name: str, kind: type[A]) -> A:
    answer = answers.get(name)
    if not isinstance(answer, kind):
        found = type(answer).__name__ if answer is not None else "nothing"
        raise SignalError(f"question {name!r}: expected a {kind.__name__}, got {found}")
    return answer


def signals_from_response(
    response: SystemOneResponse,
    *,
    source: str,
    latency_ms: float | None = None,
) -> Signals:
    """Turn a raw System One response into :class:`Signals`, validating the answer types."""
    answers = response.answers
    task = _expect(answers, QUESTION_TASK_TYPE, ChoiceAnswer)
    difficulty = _expect(answers, QUESTION_DIFFICULTY, ScoreAnswer)
    reasoning = _expect(answers, QUESTION_DEEP_REASONING, NoulAnswer)
    sensitive = answers.get(QUESTION_SENSITIVE)

    levels = sorted(difficulty.probabilities)
    if levels != list(range(len(levels))):
        raise SignalError(f"difficulty levels are not contiguous from 0: {levels}")

    return Signals(
        task_type=task.choice,
        task_type_confidence=task.confidence,
        task_type_probabilities=dict(task.probabilities),
        difficulty=difficulty.score,
        difficulty_confidence=difficulty.confidence,
        difficulty_probabilities=tuple(difficulty.probabilities[level] for level in levels),
        deep_reasoning=reasoning.noul,
        sensitive=sensitive.noul if isinstance(sensitive, NoulAnswer) else 0.0,
        source=source,
        model=response.model,
        latency_ms=latency_ms,
        input_tokens=response.usage.input_tokens,
    )
