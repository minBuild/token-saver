import pytest

from token_saver.classifiers.heuristic import HeuristicClassifier
from token_saver.signals import TASK_TYPES

classifier = HeuristicClassifier()

TRACEBACK = """I get this error when running my script:
Traceback (most recent call last):
  File "app.py", line 3, in <module>
    import requests
ModuleNotFoundError: No module named 'requests'
"""

ARCHITECTURE = (
    "Design a multi-region active-active PostgreSQL architecture for a payments system. "
    "Compare synchronous vs asynchronous replication, analyze the trade-offs for consistency "
    "and latency, and recommend a strategy with a step by step migration plan."
)


@pytest.mark.parametrize(
    ("prompt", "task_type"),
    [
        ("hi", "chitchat"),
        ("안녕! 오늘 기분 어때?", "chitchat"),
        ("What is the capital of France?", "factual"),
        ("Which database should I use for a chat app?", "factual"),
        ("Translate this to Korean: good morning, team", "rewrite"),
        ("이 문장을 영어로 번역해줘: 오늘 회의는 3시에 시작합니다", "rewrite"),
        (TRACEBACK, "coding"),
        ("로그인 API에서 500 에러가 나는 버그를 고쳐줘", "coding"),
        (ARCHITECTURE, "analysis"),
        ("Write a short poem about autumn rain", "creative"),
        ("Migrate the entire repository from Flask to FastAPI and deploy it", "agentic"),
    ],
)
def test_task_type_for_representative_prompts(prompt, task_type):
    assert classifier.classify(prompt).task_type == task_type


def test_short_greetings_are_trivial_and_need_no_deep_reasoning():
    signals = classifier.classify("hi")
    assert signals.difficulty_level == 0
    assert signals.deep_reasoning < 0.1


def test_architecture_questions_are_hard_and_need_deep_reasoning():
    signals = classifier.classify(ARCHITECTURE)
    assert signals.difficulty_level >= 3
    assert signals.deep_reasoning > 0.7


def test_whole_word_matching_keeps_short_keywords_from_firing_inside_words():
    # "hi" inside "Which" and "this" must not make a real question look like small talk.
    signals = classifier.classify("Which option fits this use case best?")
    assert signals.task_type != "chitchat"


def test_secrets_and_personal_data_are_flagged():
    assert (
        classifier.classify("My key is sk-abcdefghijklmnop1234, why does auth fail?").sensitive
        > 0.8
    )
    assert classifier.classify("Email me at jane.doe@example.com about the invoice").sensitive > 0.8
    assert classifier.classify("비밀번호를 잊어버렸는데 어떻게 재설정해?").sensitive > 0.8
    assert classifier.classify("What is the capital of France?").sensitive < 0.2


def test_signals_are_well_formed():
    signals = classifier.classify(ARCHITECTURE)
    assert set(signals.task_type_probabilities) == set(TASK_TYPES)
    assert sum(signals.task_type_probabilities.values()) == pytest.approx(1.0)
    assert sum(signals.difficulty_probabilities) == pytest.approx(1.0)
    assert 0.0 <= signals.task_type_confidence <= 1.0
    assert 0.0 <= signals.difficulty_confidence <= 1.0
    assert signals.source == "heuristic"
    assert signals.model is None
    assert signals.latency_ms is not None


def test_classification_is_deterministic():
    first = classifier.classify(ARCHITECTURE)
    second = classifier.classify(ARCHITECTURE)
    assert first.task_type_probabilities == second.task_type_probabilities
    assert first.difficulty == second.difficulty


def test_empty_prompts_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        classifier.classify("  \n ")
