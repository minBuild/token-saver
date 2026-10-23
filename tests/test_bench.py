from collections import Counter
from pathlib import Path

import pytest

from tests.helpers import make_signals
from token_saver.bench import BenchCase, load_cases, run_bench, summarize
from token_saver.classifiers import ClassifierError
from token_saver.router import Router
from token_saver.signals import Signals

REPO_DATASET = Path(__file__).resolve().parents[1] / "bench" / "prompts.jsonl"


class ScriptedClassifier:
    """Answers each prompt with a preset difficulty, keyed by prompt text."""

    name = "scripted"

    def __init__(self, script: dict[str, float]) -> None:
        self.script = script

    def classify(self, prompt: str) -> Signals:
        if prompt not in self.script:
            raise ClassifierError("not in script")
        return make_signals(task_type="creative", difficulty=self.script[prompt], source="scripted")


def test_load_cases_validates_the_file(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '{"id": "a", "prompt": "hi", "expected": "trivial", "lang": "en"}\n'
        "\n"
        '{"id": "b", "prompt": "design a database", "expected": "hard"}\n',
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert [c.id for c in cases] == ["a", "b"]
    assert cases[0].lang == "en" and cases[1].lang == ""

    path.write_text('{"id": "a", "prompt": "hi", "expected": "legendary"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not in the policy"):
        load_cases(path)
    path.write_text('{"id": "a", "prompt": "hi"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing expected"):
        load_cases(path)
    path.write_text('{"id": "a", "prompt": "hi", "expected": "trivial"}\n' * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        load_cases(path)
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_cases(path)
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no prompts"):
        load_cases(path)


def test_report_scores_agreement_and_prices_the_routing():
    cases = [
        BenchCase("t", "hi", "trivial"),
        BenchCase("s", "translate", "simple"),
        BenchCase("h", "design", "hard"),
        BenchCase("x", "prove", "expert"),
    ]
    classifier = ScriptedClassifier({"hi": 0.0, "translate": 1.0, "design": 2.0, "prove": 4.0})
    report = run_bench(cases, Router(classifier), output_tokens=1000)

    assert report.size == 4
    assert [row.decision.tier for row in report.rows] == ["trivial", "simple", "moderate", "expert"]
    assert report.exact_rate == pytest.approx(0.75)
    assert report.within_one_rate == pytest.approx(1.0)
    assert report.baseline_cost > report.routed_cost > 0
    assert report.classifier_cost == 0.0  # the scripted classifier reports no tokens
    assert 0 < report.savings < 1
    assert report.sources == Counter({"scripted": 4})
    assert report.fallbacks == 0
    assert report.confusion()["hard"] == Counter({"moderate": 1})

    data = report.to_dict()
    assert data["rows"][2]["routed"] == "moderate" and data["rows"][2]["expected"] == "hard"

    markdown = report.to_markdown()
    assert "| Exact tier match | 75% (3/4) |" in markdown
    assert "| h | hard | moderate (+1) |" in markdown
    assert "| expected \\ routed | trivial | simple | moderate | hard | expert |" in markdown


def test_fallbacks_are_counted_and_jev_tokens_are_priced():
    cases = [BenchCase("a", "unknown prompt", "trivial")]
    report = run_bench(cases, Router(ScriptedClassifier({})))
    assert report.fallbacks == 1
    assert report.sources == Counter({"heuristic": 1})
    assert report.mean_latency_ms is not None


def test_the_repository_prompt_set_is_balanced_and_well_formed():
    cases = load_cases(REPO_DATASET)
    assert len(cases) == 50
    assert summarize(cases) == Counter(
        {"trivial": 10, "simple": 10, "moderate": 10, "hard": 10, "expert": 10}
    )
    assert {c.lang for c in cases} == {"en", "ko"}
    assert all(c.prompt.strip() for c in cases)


def test_the_heuristic_is_a_reasonable_fallback_on_the_repository_prompt_set():
    # A loose floor: the heuristic is a fallback, not the product. Jev results are recorded
    # separately once an API key is available.
    report = run_bench(load_cases(REPO_DATASET), Router())
    assert report.within_one_rate >= 0.7
    assert report.savings > 0
