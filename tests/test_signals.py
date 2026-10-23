import pytest
from typesafe_sdk import Choice, Noul, Score

from tests.helpers import parse_raw_response
from token_saver.signals import (
    DIFFICULTY_LEVELS,
    TASK_TYPES,
    SignalError,
    Signals,
    build_questions,
    build_state,
    signals_from_response,
)


def test_question_set_covers_every_signal_the_policy_uses():
    questions = build_questions()
    assert set(questions) == {"task_type", "difficulty", "deep_reasoning", "sensitive"}
    assert isinstance(questions["task_type"], Choice)
    assert set(questions["task_type"].criteria) == set(TASK_TYPES)
    assert isinstance(questions["difficulty"], Score)
    assert list(questions["difficulty"].criteria) == list(DIFFICULTY_LEVELS)
    assert isinstance(questions["deep_reasoning"], Noul)
    assert isinstance(questions["sensitive"], Noul)


def test_state_wraps_the_prompt_in_a_named_field():
    assert build_state("hello") == {"user_message": "hello"}


def test_signals_are_parsed_from_a_system_one_response(response_factory):
    response = response_factory(
        task_type="analysis",
        task_type_probability=0.7,
        difficulty_probabilities=(0.0, 0.1, 0.6, 0.25, 0.05),
        deep_reasoning=0.66,
        sensitive=0.03,
        input_tokens=211,
    )

    signals = signals_from_response(response, source="jev", latency_ms=131.5)

    assert signals.task_type == "analysis"
    assert signals.task_type_probabilities["analysis"] == pytest.approx(0.7)
    assert signals.difficulty == pytest.approx(0.1 + 1.2 + 0.75 + 0.2)
    assert signals.difficulty_level == 2
    assert signals.difficulty_probabilities == (0.0, 0.1, 0.6, 0.25, 0.05)
    assert signals.deep_reasoning == pytest.approx(0.66)
    assert signals.sensitive == pytest.approx(0.03)
    assert signals.source == "jev"
    assert signals.model == "jev-1.13.0"
    assert signals.latency_ms == 131.5
    assert signals.input_tokens == 211


def test_sensitive_is_optional_and_defaults_to_zero(response_factory):
    signals = signals_from_response(response_factory(sensitive=None), source="fixture")
    assert signals.sensitive == 0.0


def test_missing_or_mistyped_answers_raise_signal_error(raw_response):
    raw = raw_response()
    del raw["answers"]["difficulty"]
    with pytest.raises(SignalError, match="difficulty"):
        signals_from_response(parse_raw_response(raw), source="jev")

    raw = raw_response()
    raw["answers"]["deep_reasoning"] = raw["answers"]["task_type"]
    with pytest.raises(SignalError, match="deep_reasoning"):
        signals_from_response(parse_raw_response(raw), source="jev")


@pytest.mark.parametrize(
    ("difficulty", "level"),
    [(0.0, 0), (0.49, 0), (0.5, 1), (2.5, 3), (3.9, 4), (4.0, 4), (9.0, 4), (-1.0, 0)],
)
def test_difficulty_level_rounds_half_up_and_clamps(difficulty, level):
    signals = Signals(
        task_type="factual",
        task_type_confidence=1.0,
        task_type_probabilities={"factual": 1.0},
        difficulty=difficulty,
        difficulty_confidence=1.0,
        difficulty_probabilities=(1.0, 0.0, 0.0, 0.0, 0.0),
        deep_reasoning=0.0,
        sensitive=0.0,
        source="test",
    )
    assert signals.difficulty_level == level


def test_to_dict_is_json_friendly(response_factory):
    signals = signals_from_response(response_factory(), source="jev")
    data = signals.to_dict()
    assert data["difficulty_level"] == signals.difficulty_level
    assert data["difficulty_probabilities"] == list(signals.difficulty_probabilities)
    assert data["source"] == "jev"
