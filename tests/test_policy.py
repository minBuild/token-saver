import json

import pytest

from tests.helpers import make_signals
from token_saver.policy import EFFORTS, Policy, PolicyError

MINIMAL = """
tiers:
  - {name: cheap, model: claude-haiku-5-5, effort: medium}
  - {name: mid, model: claude-sonnet-5-5, effort: high}
  - {name: top, model: claude-opus-5-5, effort: xhigh}
difficulty:
  levels: [cheap, cheap, mid, top, top]
"""


def test_default_policy_is_the_documented_ladder():
    policy = Policy.default()
    assert [t.name for t in policy.tiers] == ["trivial", "simple", "moderate", "hard", "expert"]
    assert [(t.model, t.effort) for t in policy.tiers] == [
        ("claude-haiku-5-5", "high"),
        ("claude-sonnet-5-5", "high"),
        ("claude-sonnet-5-5", "xhigh"),
        ("claude-opus-5-5", "xhigh"),
        ("claude-fable-5-1", "xhigh"),
    ]
    assert policy.difficulty_levels == ("trivial", "simple", "moderate", "hard", "expert")
    assert policy.low_confidence is not None and policy.low_confidence.below == 0.35
    assert policy.deep_reasoning is not None and policy.deep_reasoning.min_tier == "hard"
    assert policy.floors == {"coding": "simple", "analysis": "moderate", "agentic": "hard"}
    assert policy.ceilings == {"chitchat": "trivial", "factual": "moderate"}
    assert all(t.effort in EFFORTS for t in policy.tiers)


@pytest.mark.parametrize(
    ("difficulty", "tier"),
    [(0.1, "trivial"), (1.2, "simple"), (2.4, "moderate"), (2.5, "hard"), (3.8, "expert")],
)
def test_difficulty_level_picks_the_tier_when_nothing_else_fires(difficulty, tier):
    decision = Policy.default().decide(make_signals(task_type="creative", difficulty=difficulty))
    assert decision.tier == tier
    assert decision.reasons[0].startswith(f"difficulty {difficulty:.2f} rounds to level")


def test_low_confidence_escalates_one_tier_and_is_capped_at_the_top():
    policy = Policy.default()
    unsure = policy.decide(
        make_signals(task_type="creative", difficulty=1.0, difficulty_confidence=0.2)
    )
    assert unsure.tier == "moderate"
    assert any("escalated to moderate" in r for r in unsure.reasons)

    already_top = policy.decide(
        make_signals(task_type="creative", difficulty=4.0, difficulty_confidence=0.2)
    )
    assert already_top.tier == "expert"
    assert not any("escalated" in r for r in already_top.reasons)


def test_deep_reasoning_raises_to_at_least_hard():
    policy = Policy.default()
    decision = policy.decide(make_signals(task_type="creative", difficulty=1.0, deep_reasoning=0.9))
    assert decision.tier == "hard"
    assert any("deep reasoning 0.90 > 0.7" in r for r in decision.reasons)

    # Already above the minimum: nothing changes.
    decision = policy.decide(make_signals(task_type="creative", difficulty=4.0, deep_reasoning=0.9))
    assert decision.tier == "expert"


def test_task_type_floors_and_ceilings_bound_the_pick():
    policy = Policy.default()
    coding = policy.decide(make_signals(task_type="coding", difficulty=0.0))
    assert coding.tier == "simple"
    assert "coding floor -> simple" in coding.reasons

    chitchat = policy.decide(make_signals(task_type="chitchat", difficulty=3.0))
    assert chitchat.tier == "trivial"
    assert "chitchat ceiling -> trivial" in chitchat.reasons


def test_bounds_are_skipped_when_the_task_type_is_uncertain():
    policy = Policy.default()
    decision = policy.decide(
        make_signals(task_type="coding", task_type_confidence=0.3, difficulty=0.0)
    )
    assert decision.tier == "trivial"
    assert any("floors and ceilings skipped" in r for r in decision.reasons)


def test_rules_apply_in_order_and_ceilings_win_last():
    # Deep reasoning pushes small talk to `hard`, then the chitchat ceiling pulls it back.
    decision = Policy.default().decide(
        make_signals(task_type="chitchat", difficulty=0.0, deep_reasoning=0.95)
    )
    assert decision.tier == "trivial"
    assert [r.split(" -> ")[-1] for r in decision.reasons] == [
        "trivial",
        "at least hard",
        "trivial",
    ]


def test_fallback_reason_leads_the_explanation():
    signals = make_signals(source="heuristic", fallback_reason="Jev request failed: timeout")
    decision = Policy.default().decide(signals)
    assert decision.reasons[0] == "classifier fallback to heuristic: Jev request failed: timeout"


def test_decision_and_policy_serialise_to_json():
    policy = Policy.default()
    decision = policy.decide(make_signals())
    data = json.loads(json.dumps(decision.to_dict()))
    assert data["model"] == decision.model
    assert data["signals"]["source"] == "test"
    assert json.loads(json.dumps(policy.to_dict()))["tiers"][0]["name"] == "trivial"


def test_minimal_policy_from_yaml_and_from_file(tmp_path):
    policy = Policy.from_yaml(MINIMAL)
    assert policy.low_confidence is None and policy.deep_reasoning is None
    assert policy.floors == {} and policy.ceilings == {}
    assert policy.decide(make_signals(difficulty=1.0)).model == "claude-haiku-5-5"

    path = tmp_path / "policy.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    assert Policy.load(path).tiers == policy.tiers


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("tiers: []\ndifficulty: {levels: []}", "non-empty list"),
        (MINIMAL.replace("effort: medium", "effort: turbo"), "effort must be one of"),
        (
            MINIMAL.replace("levels: [cheap, cheap, mid, top, top]", "levels: [cheap, top]"),
            "exactly 5",
        ),
        (
            MINIMAL.replace(
                "levels: [cheap, cheap, mid, top, top]", "levels: [cheap, cheap, mid, top, nope]"
            ),
            "not a defined tier",
        ),
        (MINIMAL + "rules:\n  floors: {cooking: top}\n", "not a task type"),
        (MINIMAL + "rules:\n  low_confidence: {below: 7}\n", "between 0 and 1"),
        (
            MINIMAL + "rules:\n  deep_reasoning: {above: 0.5, min_tier: nope}\n",
            "not a defined tier",
        ),
        (MINIMAL + "rulez: {}\n", "unknown key"),
        (MINIMAL.replace("name: mid", "name: cheap"), "duplicate tier name"),
        ("- just\n- a list\n", "mapping at the top level"),
    ],
)
def test_malformed_policies_fail_with_a_pointed_message(yaml_text, message):
    with pytest.raises(PolicyError, match=message):
        Policy.from_yaml(yaml_text)
