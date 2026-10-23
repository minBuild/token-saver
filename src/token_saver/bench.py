"""Run the router over a labelled prompt set and report agreement, latency and estimated cost.

The prompt set is a JSON Lines file; each line has ``id``, ``prompt``, ``expected`` (a tier
name from the policy) and optionally ``lang``. The report compares what the router picked with
what a human labelled, and prices both against a fixed baseline such as "everything goes to
Opus at xhigh", which is what a default Claude setup effectively does.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from token_saver.policy import Decision, Effort, Policy
from token_saver.pricing import estimate_cost, estimate_tokens, jev_cost
from token_saver.router import Router

DEFAULT_BASELINE_MODEL = "claude-opus-5-5"
DEFAULT_BASELINE_EFFORT: Effort = "xhigh"
DEFAULT_OUTPUT_TOKENS = 800


@dataclass(frozen=True)
class BenchCase:
    id: str
    prompt: str
    expected: str
    lang: str = ""


def load_cases(path: Path, policy: Policy | None = None) -> list[BenchCase]:
    """Read a JSON Lines prompt set, checking ids are unique and expected tiers exist."""
    tier_names = {tier.name for tier in (policy or Policy.default()).tiers}
    cases: list[BenchCase] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: not valid JSON ({exc.msg})") from None
        missing = {"id", "prompt", "expected"} - set(data)
        if missing:
            raise ValueError(f"{path}:{number}: missing {', '.join(sorted(missing))}")
        if data["expected"] not in tier_names:
            raise ValueError(
                f"{path}:{number}: expected tier {data['expected']!r} is not in the policy"
            )
        if data["id"] in seen:
            raise ValueError(f"{path}:{number}: duplicate id {data['id']!r}")
        seen.add(data["id"])
        cases.append(
            BenchCase(
                id=str(data["id"]),
                prompt=str(data["prompt"]),
                expected=str(data["expected"]),
                lang=str(data.get("lang", "")),
            )
        )
    if not cases:
        raise ValueError(f"{path}: no prompts found")
    return cases


@dataclass(frozen=True)
class BenchRow:
    case: BenchCase
    decision: Decision
    expected_index: int
    input_tokens: int
    routed_cost: float
    baseline_cost: float
    classifier_cost: float

    @property
    def exact(self) -> bool:
        return self.decision.tier == self.case.expected

    @property
    def distance(self) -> int:
        return abs(self.decision.tier_index - self.expected_index)


@dataclass(frozen=True)
class BenchReport:
    rows: tuple[BenchRow, ...]
    policy: Policy
    baseline_model: str
    baseline_effort: Effort
    output_tokens: int

    @property
    def size(self) -> int:
        return len(self.rows)

    @property
    def exact_rate(self) -> float:
        return sum(row.exact for row in self.rows) / self.size

    @property
    def within_one_rate(self) -> float:
        return sum(row.distance <= 1 for row in self.rows) / self.size

    @property
    def routed_cost(self) -> float:
        return sum(row.routed_cost for row in self.rows)

    @property
    def baseline_cost(self) -> float:
        return sum(row.baseline_cost for row in self.rows)

    @property
    def classifier_cost(self) -> float:
        return sum(row.classifier_cost for row in self.rows)

    @property
    def savings(self) -> float:
        """Share of the baseline bill saved, after paying for the classifier."""
        if self.baseline_cost == 0:
            return 0.0
        return 1 - (self.routed_cost + self.classifier_cost) / self.baseline_cost

    @property
    def mean_latency_ms(self) -> float | None:
        latencies = [
            row.decision.signals.latency_ms
            for row in self.rows
            if row.decision.signals.latency_ms is not None
        ]
        return statistics.fmean(latencies) if latencies else None

    @property
    def sources(self) -> Counter[str]:
        return Counter(row.decision.signals.source for row in self.rows)

    @property
    def fallbacks(self) -> int:
        return sum(row.decision.signals.fallback_reason is not None for row in self.rows)

    def confusion(self) -> dict[str, Counter[str]]:
        """Routed-tier counts for each expected tier, in policy order."""
        names = [tier.name for tier in self.policy.tiers]
        table: dict[str, Counter[str]] = {name: Counter() for name in names}
        for row in self.rows:
            table[row.case.expected][row.decision.tier] += 1
        return table

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "exact_rate": self.exact_rate,
            "within_one_rate": self.within_one_rate,
            "routed_cost": self.routed_cost,
            "baseline_cost": self.baseline_cost,
            "classifier_cost": self.classifier_cost,
            "savings": self.savings,
            "mean_latency_ms": self.mean_latency_ms,
            "sources": dict(self.sources),
            "fallbacks": self.fallbacks,
            "baseline": {"model": self.baseline_model, "effort": self.baseline_effort},
            "output_tokens": self.output_tokens,
            "rows": [
                {
                    "id": row.case.id,
                    "lang": row.case.lang,
                    "expected": row.case.expected,
                    "routed": row.decision.tier,
                    "model": row.decision.model,
                    "effort": row.decision.effort,
                    "source": row.decision.signals.source,
                    "difficulty": row.decision.signals.difficulty,
                    "task_type": row.decision.signals.task_type,
                    "routed_cost": row.routed_cost,
                    "baseline_cost": row.baseline_cost,
                }
                for row in self.rows
            ],
        }

    def to_markdown(self) -> str:
        sources = ", ".join(f"{name} x{count}" for name, count in sorted(self.sources.items()))
        latency = f"{self.mean_latency_ms:.1f} ms" if self.mean_latency_ms is not None else "n/a"
        exact = sum(row.exact for row in self.rows)
        within = sum(row.distance <= 1 for row in self.rows)
        lines = [
            f"Prompts: {self.size} | Classifier: {sources} | "
            f"Baseline: {self.baseline_model} @ {self.baseline_effort} | "
            f"Assumed output: {self.output_tokens} tokens per request",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Exact tier match | {self.exact_rate:.0%} ({exact}/{self.size}) |",
            f"| Within one tier | {self.within_one_rate:.0%} ({within}/{self.size}) |",
            f"| Estimated Claude spend, routed | ${self.routed_cost:.4f} |",
            f"| Estimated Claude spend, baseline | ${self.baseline_cost:.4f} |",
            f"| Classifier spend | ${self.classifier_cost:.6f} |",
            f"| Savings vs baseline | {self.savings:.1%} |",
            f"| Mean classifier latency | {latency} |",
            f"| Fallbacks to the heuristic | {self.fallbacks} |",
            "",
        ]
        names = [tier.name for tier in self.policy.tiers]
        lines.append("| expected \\ routed | " + " | ".join(names) + " |")
        lines.append("|---|" + "---|" * len(names))
        for expected, counts in self.confusion().items():
            lines.append(
                f"| {expected} | " + " | ".join(str(counts.get(name, 0)) for name in names) + " |"
            )
        lines.append("")
        lines.append("| id | expected | routed | model | effort | task type | difficulty |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in self.rows:
            mark = "" if row.exact else (" (+1)" if row.distance == 1 else f" (+{row.distance})")
            lines.append(
                f"| {row.case.id} | {row.case.expected} | {row.decision.tier}{mark} | "
                f"{row.decision.model} | {row.decision.effort} | "
                f"{row.decision.signals.task_type} | {row.decision.signals.difficulty:.2f} |"
            )
        return "\n".join(lines) + "\n"


def run_bench(
    cases: Iterable[BenchCase],
    router: Router,
    *,
    baseline_model: str = DEFAULT_BASELINE_MODEL,
    baseline_effort: Effort = DEFAULT_BASELINE_EFFORT,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
) -> BenchReport:
    """Route every case and price the outcome against sending everything to the baseline."""
    policy = router.policy
    rows: list[BenchRow] = []
    for case in cases:
        decision = router.route(case.prompt)
        input_tokens = estimate_tokens(case.prompt)
        classifier_tokens = decision.signals.input_tokens
        rows.append(
            BenchRow(
                case=case,
                decision=decision,
                expected_index=policy.tier_index(case.expected),
                input_tokens=input_tokens,
                routed_cost=estimate_cost(decision.model, input_tokens, output_tokens),
                baseline_cost=estimate_cost(baseline_model, input_tokens, output_tokens),
                classifier_cost=jev_cost(classifier_tokens) if classifier_tokens else 0.0,
            )
        )
    return BenchReport(
        rows=tuple(rows),
        policy=policy,
        baseline_model=baseline_model,
        baseline_effort=baseline_effort,
        output_tokens=output_tokens,
    )


def summarize(cases: Sequence[BenchCase]) -> Counter[str]:
    """How many prompts the set holds per expected tier."""
    return Counter(case.expected for case in cases)
