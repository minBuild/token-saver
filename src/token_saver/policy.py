"""The routing policy: a ladder of (model, effort) tiers and the rules that pick one.

The policy is plain YAML so it can be tuned without touching code; ``policies/default.yaml``
ships with the package and documents every knob. :meth:`Policy.decide` is pure: the same
signals always give the same :class:`Decision`, with a human-readable trail of reasons.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from token_saver.signals import MAX_DIFFICULTY_LEVEL, TASK_TYPES, Signals

Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORTS: tuple[Effort, ...] = ("low", "medium", "high", "xhigh", "max")

DEFAULT_POLICY_RESOURCE = "policies/default.yaml"


class PolicyError(ValueError):
    """The policy file is malformed or references something that does not exist."""


@dataclass(frozen=True)
class Tier:
    name: str
    model: str
    effort: Effort
    note: str = ""


@dataclass(frozen=True)
class LowConfidenceRule:
    below: float
    escalate_by: int = 1


@dataclass(frozen=True)
class DeepReasoningRule:
    above: float
    min_tier: str


@dataclass(frozen=True)
class Decision:
    """Where a prompt should go, and why."""

    model: str
    effort: Effort
    tier: str
    tier_index: int
    reasons: tuple[str, ...]
    signals: Signals

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "effort": self.effort,
            "tier": self.tier,
            "tier_index": self.tier_index,
            "reasons": list(self.reasons),
            "signals": self.signals.to_dict(),
        }


@dataclass(frozen=True)
class Policy:
    tiers: tuple[Tier, ...]
    difficulty_levels: tuple[str, ...]
    """Tier name for each difficulty level, index 0 being the easiest."""
    low_confidence: LowConfidenceRule | None = None
    deep_reasoning: DeepReasoningRule | None = None
    floors: Mapping[str, str] = field(default_factory=dict)
    """Minimum tier per task type, applied when the task type is confidently known."""
    ceilings: Mapping[str, str] = field(default_factory=dict)
    """Maximum tier per task type, applied last so it wins over every other rule."""
    task_type_min_confidence: float = 0.5
    version: int = 1

    # -- loading -------------------------------------------------------------------------

    @classmethod
    def default(cls) -> Policy:
        text = resources.files("token_saver").joinpath(DEFAULT_POLICY_RESOURCE).read_text("utf-8")
        return cls.from_yaml(text)

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_yaml(cls, text: str) -> Policy:
        data = yaml.safe_load(text)
        if not isinstance(data, Mapping):
            raise PolicyError("policy must be a mapping at the top level")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Policy:
        _reject_unknown_keys(data, {"version", "tiers", "difficulty", "rules"}, "policy")
        version = data.get("version", 1)
        if not isinstance(version, int):
            raise PolicyError(f"version must be an integer, got {version!r}")

        tiers = _parse_tiers(data.get("tiers"))
        names = {tier.name for tier in tiers}

        difficulty = data.get("difficulty") or {}
        _reject_unknown_keys(difficulty, {"levels"}, "difficulty")
        levels = difficulty.get("levels")
        if not isinstance(levels, list) or len(levels) != MAX_DIFFICULTY_LEVEL + 1:
            raise PolicyError(
                f"difficulty.levels must list exactly {MAX_DIFFICULTY_LEVEL + 1} tier names, "
                f"one per difficulty level"
            )
        for level in levels:
            _require_tier(level, names, "difficulty.levels")

        rules = data.get("rules") or {}
        _reject_unknown_keys(
            rules,
            {"low_confidence", "deep_reasoning", "floors", "ceilings", "task_type_min_confidence"},
            "rules",
        )
        low_confidence = None
        if (raw := rules.get("low_confidence")) is not None:
            _reject_unknown_keys(raw, {"below", "escalate_by"}, "rules.low_confidence")
            escalate_by = raw.get("escalate_by", 1)
            if not isinstance(escalate_by, int) or escalate_by < 1:
                raise PolicyError("rules.low_confidence.escalate_by must be a positive integer")
            low_confidence = LowConfidenceRule(
                below=_probability(raw.get("below"), "rules.low_confidence.below"),
                escalate_by=escalate_by,
            )
        deep_reasoning = None
        if (raw := rules.get("deep_reasoning")) is not None:
            _reject_unknown_keys(raw, {"above", "min_tier"}, "rules.deep_reasoning")
            min_tier = raw.get("min_tier")
            _require_tier(min_tier, names, "rules.deep_reasoning.min_tier")
            deep_reasoning = DeepReasoningRule(
                above=_probability(raw.get("above"), "rules.deep_reasoning.above"),
                min_tier=min_tier,
            )

        return cls(
            tiers=tiers,
            difficulty_levels=tuple(levels),
            low_confidence=low_confidence,
            deep_reasoning=deep_reasoning,
            floors=_parse_bounds(rules.get("floors"), names, "rules.floors"),
            ceilings=_parse_bounds(rules.get("ceilings"), names, "rules.ceilings"),
            task_type_min_confidence=_probability(
                rules.get("task_type_min_confidence", 0.5), "rules.task_type_min_confidence"
            ),
            version=version,
        )

    # -- deciding ------------------------------------------------------------------------

    def tier_index(self, name: str) -> int:
        for index, tier in enumerate(self.tiers):
            if tier.name == name:
                return index
        raise PolicyError(f"unknown tier {name!r}")

    def decide(self, signals: Signals) -> Decision:
        """Pick a tier for these signals. Rules apply in a fixed order, each leaving a reason."""
        reasons: list[str] = []
        if signals.fallback_reason:
            reasons.append(f"classifier fallback to {signals.source}: {signals.fallback_reason}")

        level = signals.difficulty_level
        index = self.tier_index(self.difficulty_levels[level])
        reasons.append(
            f"difficulty {signals.difficulty:.2f} rounds to level {level} "
            f"-> {self.tiers[index].name}"
        )

        if self.low_confidence and signals.difficulty_confidence < self.low_confidence.below:
            escalated = min(index + self.low_confidence.escalate_by, len(self.tiers) - 1)
            if escalated != index:
                reasons.append(
                    f"difficulty confidence {signals.difficulty_confidence:.2f} < "
                    f"{self.low_confidence.below} -> escalated to {self.tiers[escalated].name}"
                )
                index = escalated

        if self.deep_reasoning and signals.deep_reasoning > self.deep_reasoning.above:
            minimum = self.tier_index(self.deep_reasoning.min_tier)
            if minimum > index:
                reasons.append(
                    f"deep reasoning {signals.deep_reasoning:.2f} > {self.deep_reasoning.above} "
                    f"-> at least {self.deep_reasoning.min_tier}"
                )
                index = minimum

        if signals.task_type_confidence >= self.task_type_min_confidence:
            floor = self.floors.get(signals.task_type)
            if floor is not None and self.tier_index(floor) > index:
                reasons.append(f"{signals.task_type} floor -> {floor}")
                index = self.tier_index(floor)
            ceiling = self.ceilings.get(signals.task_type)
            if ceiling is not None and self.tier_index(ceiling) < index:
                reasons.append(f"{signals.task_type} ceiling -> {ceiling}")
                index = self.tier_index(ceiling)
        elif signals.task_type in self.floors or signals.task_type in self.ceilings:
            reasons.append(
                f"task type {signals.task_type} confidence {signals.task_type_confidence:.2f} < "
                f"{self.task_type_min_confidence} -> floors and ceilings skipped"
            )

        tier = self.tiers[index]
        return Decision(
            model=tier.model,
            effort=tier.effort,
            tier=tier.name,
            tier_index=index,
            reasons=tuple(reasons),
            signals=signals,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tiers": [
                {"name": t.name, "model": t.model, "effort": t.effort, "note": t.note}
                for t in self.tiers
            ],
            "difficulty": {"levels": list(self.difficulty_levels)},
            "rules": {
                "low_confidence": (
                    {
                        "below": self.low_confidence.below,
                        "escalate_by": self.low_confidence.escalate_by,
                    }
                    if self.low_confidence
                    else None
                ),
                "deep_reasoning": (
                    {"above": self.deep_reasoning.above, "min_tier": self.deep_reasoning.min_tier}
                    if self.deep_reasoning
                    else None
                ),
                "task_type_min_confidence": self.task_type_min_confidence,
                "floors": dict(self.floors),
                "ceilings": dict(self.ceilings),
            },
        }


# -- parsing helpers ---------------------------------------------------------------------


def _reject_unknown_keys(data: Any, allowed: set[str], where: str) -> None:
    if not isinstance(data, Mapping):
        raise PolicyError(f"{where} must be a mapping")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise PolicyError(f"{where}: unknown key(s) {', '.join(map(repr, unknown))}")


def _parse_tiers(raw: Any) -> tuple[Tier, ...]:
    if not isinstance(raw, list) or not raw:
        raise PolicyError("tiers must be a non-empty list")
    tiers: list[Tier] = []
    seen: set[str] = set()
    for position, item in enumerate(raw):
        _reject_unknown_keys(item, {"name", "model", "effort", "note"}, f"tiers[{position}]")
        name, model, effort = item.get("name"), item.get("model"), item.get("effort")
        if not isinstance(name, str) or not name:
            raise PolicyError(f"tiers[{position}]: name must be a non-empty string")
        if name in seen:
            raise PolicyError(f"tiers[{position}]: duplicate tier name {name!r}")
        if not isinstance(model, str) or not model:
            raise PolicyError(f"tier {name!r}: model must be a non-empty string")
        if effort not in EFFORTS:
            raise PolicyError(f"tier {name!r}: effort must be one of {', '.join(EFFORTS)}")
        seen.add(name)
        tiers.append(
            Tier(
                name=name,
                model=model,
                effort=cast(Effort, effort),
                note=str(item.get("note", "") or ""),
            )
        )
    return tuple(tiers)


def _require_tier(name: Any, names: set[str], where: str) -> None:
    if not isinstance(name, str) or name not in names:
        raise PolicyError(f"{where}: {name!r} is not a defined tier")


def _probability(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
        raise PolicyError(f"{where} must be a number between 0 and 1")
    return float(value)


def _parse_bounds(raw: Any, names: set[str], where: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise PolicyError(f"{where} must be a mapping of task type to tier name")
    bounds: dict[str, str] = {}
    for task_type, tier in raw.items():
        if task_type not in TASK_TYPES:
            raise PolicyError(f"{where}: {task_type!r} is not a task type")
        _require_tier(tier, names, f"{where}.{task_type}")
        bounds[task_type] = tier
    return bounds
