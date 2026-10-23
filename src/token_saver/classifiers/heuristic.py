"""An offline, dependency-free stand-in for Jev.

It keeps token-saver routing when Jev is unreachable, when there is no API key, and in tests.
It is deliberately simple and transparent: word counts, code markers and short keyword lists in
English and Korean. It is not meant to be accurate, only reasonable and predictable, and it
reports confidence through the same formulas Jev uses so the policy can treat it the same way.
"""

from __future__ import annotations

import math
import re
import time

from token_saver.confidence import choice_confidence, score_confidence
from token_saver.signals import DIFFICULTY_LEVELS, TASK_TYPES, Signals

_CODE_MARKERS = re.compile(
    r"```|\bdef \w+\(|\bclass \w+[:(]|\bimport \w+|\bfunction\b|=>|\bSELECT\b.+\bFROM\b"
    r"|Traceback \(most recent call last\)|\w+Error:|\w+Exception\b|stack ?trace",
    re.IGNORECASE,
)

# Each hit nudges the task-type weights. Lists are short on purpose: this is a fallback.
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "chitchat": (
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good night",
        "lol",
        "haha",
        "안녕",
        "고마워",
        "고맙",
        "감사",
        "ㅎㅎ",
        "ㅋㅋ",
        "반가워",
        "잘자",
        "수고",
    ),
    "factual": (
        "what is",
        "what's",
        "who is",
        "who was",
        "when did",
        "when was",
        "how many",
        "how far",
        "define",
        "meaning of",
        "capital of",
        "뭐야",
        "무엇",
        "누구",
        "언제",
        "몇",
        "뜻이",
    ),
    "rewrite": (
        "translate",
        "summarize",
        "summarise",
        "rewrite",
        "proofread",
        "paraphrase",
        "shorten",
        "fix the grammar",
        "reformat",
        "번역",
        "요약",
        "다듬",
        "교정",
        "고쳐 줘",
        "고쳐줘",
        "바꿔 줘",
        "바꿔줘",
        "줄여",
        "정리해",
    ),
    "coding": (
        "code",
        "function",
        "bug",
        "error",
        "implement",
        "refactor",
        "unit test",
        "regex",
        "sql",
        "python",
        "javascript",
        "typescript",
        "rust",
        "golang",
        "api",
        "compile",
        "script",
        "구현",
        "함수",
        "버그",
        "에러",
        "리팩토링",
        "코드",
        "스크립트",
        "컴파일",
        "쿼리",
    ),
    "analysis": (
        "why",
        "prove",
        "compare",
        "trade-off",
        "tradeoff",
        "pros and cons",
        "design",
        "architecture",
        "optimize",
        "optimise",
        "root cause",
        "strategy",
        "analyze",
        "analyse",
        "evaluate",
        "step by step",
        "estimate",
        "왜",
        "증명",
        "비교",
        "장단점",
        "설계",
        "아키텍처",
        "최적화",
        "원인",
        "전략",
        "분석",
        "평가",
        "추정",
    ),
    "creative": (
        "story",
        "poem",
        "slogan",
        "tagline",
        "lyrics",
        "screenplay",
        "haiku",
        "novel",
        "소설",
        "슬로건",
        "카피",
        "가사",
        "시나리오",
        "동화",
        "시를",
    ),
    "agentic": (
        "repository",
        "repo",
        "codebase",
        "migrate",
        "set up",
        "pipeline",
        "deploy",
        "end-to-end",
        "entire",
        "whole project",
        "ci/cd",
        "전체",
        "마이그레이션",
        "배포",
        "파이프라인",
        "처음부터 끝까지",
        "프로젝트를",
    ),
}


def _compile(keyword: str) -> re.Pattern[str]:
    # ASCII keywords match whole words only ("hi" must not fire inside "which"); Korean
    # keywords match as substrings because particles attach to the word without a space.
    pattern = re.escape(keyword)
    if keyword.isascii():
        pattern = rf"\b{pattern}\b"
    return re.compile(pattern)


_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    task: tuple(_compile(keyword) for keyword in keywords) for task, keywords in _KEYWORDS.items()
}

_SENSITIVE = re.compile(
    r"sk-[A-Za-z0-9_-]{8,}|AKIA[0-9A-Z]{12,}|ghp_[A-Za-z0-9]{20,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\bpassword\b|\bpasswd\b|비밀번호|주민(?:등록)?번호|\b\d{3}-\d{2}-\d{4}\b"
    r"|\b(?:\d[ -]?){13,16}\b|[\w.+-]+@[\w-]+\.[\w.]+|\b01[016789]-?\d{3,4}-?\d{4}\b",
    re.IGNORECASE,
)

_LEVELS = len(DIFFICULTY_LEVELS)


class HeuristicClassifier:
    name = "heuristic"

    def classify(self, prompt: str) -> Signals:
        started = time.perf_counter()
        text = prompt.strip()
        if not text:
            raise ValueError("prompt must not be empty")
        lower = text.lower()
        words = len(text.split())

        hits = {
            task: sum(1 for pattern in patterns if pattern.search(lower))
            for task, patterns in _PATTERNS.items()
        }
        has_code = _CODE_MARKERS.search(text) is not None
        if has_code:
            hits["coding"] += 2

        task_probabilities = self._task_probabilities(hits, words)
        task_type = max(task_probabilities, key=lambda t: task_probabilities[t])

        center = self._difficulty_center(task_type, hits, words, has_code)
        difficulty_probabilities = self._split(center)
        difficulty = sum(i * p for i, p in enumerate(difficulty_probabilities))

        return Signals(
            task_type=task_type,
            task_type_confidence=choice_confidence(list(task_probabilities.values())),
            task_type_probabilities=task_probabilities,
            difficulty=difficulty,
            difficulty_confidence=score_confidence(difficulty_probabilities),
            difficulty_probabilities=difficulty_probabilities,
            deep_reasoning=_sigmoid((center - 2.0) * 2.0),
            sensitive=0.9 if _SENSITIVE.search(text) else 0.05,
            source=self.name,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _task_probabilities(hits: dict[str, int], words: int) -> dict[str, float]:
        weights = {task: 1.0 + 3.0 * hits[task] for task in TASK_TYPES}
        other_hits = any(count for task, count in hits.items() if task != "chitchat")
        if words > 12 or other_hits:
            weights["chitchat"] *= 0.25  # small talk is short and asks for nothing
        elif hits["chitchat"]:
            weights["chitchat"] *= 3.0
        if not any(hits.values()):
            # Nothing recognisable: short prompts are probably questions, long ones need thought.
            weights["factual" if words <= 25 else "analysis"] += 2.0
        total = sum(weights.values())
        return {task: weight / total for task, weight in weights.items()}

    @staticmethod
    def _difficulty_center(task_type: str, hits: dict[str, int], words: int, code: bool) -> float:
        # Length is the strongest cheap signal: people write more when they ask for more.
        if words <= 8:
            center = 0.3
        elif words <= 25:
            center = 1.0
        elif words <= 50:
            center = 1.8
        elif words <= 100:
            center = 2.4
        else:
            center = 3.0
        center += min(1.5, 0.6 * hits["analysis"])
        if hits["agentic"]:
            center += 1.0
        if code or (hits["coding"] and words > 15):
            center += 0.5
        if task_type == "chitchat":
            center = min(center, 0.3)
        elif task_type == "factual":
            center = min(center, 1.2)
        return max(0.0, min(float(_LEVELS - 1), center))

    @staticmethod
    def _split(center: float) -> tuple[float, ...]:
        """Put the mass on the two levels around ``center`` so the weighted mean equals it.

        A bell curve clipped at the ends would drag the mean towards the middle level; the
        two-point split keeps "0.3" meaning "almost certainly trivial".
        """
        low = min(int(center), _LEVELS - 2)
        fraction = center - low
        probabilities = [0.0] * _LEVELS
        probabilities[low] = 1.0 - fraction
        probabilities[low + 1] = fraction
        return tuple(probabilities)


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))
