import logging

import pytest

from tests.helpers import make_signals
from token_saver import router as router_module
from token_saver.classifiers import (
    ClassifierError,
    FixtureStore,
    HeuristicClassifier,
    JevClassifier,
)
from token_saver.policy import Policy
from token_saver.router import Router, route
from token_saver.signals import Signals


class StubClassifier:
    name = "stub"

    def __init__(self, signals: Signals | None = None, error: str | None = None) -> None:
        self.signals = signals
        self.error = error
        self.calls = 0
        self.closed = False

    def classify(self, prompt: str) -> Signals:
        self.calls += 1
        if self.error:
            raise ClassifierError(self.error)
        assert self.signals is not None
        return self.signals

    def close(self) -> None:
        self.closed = True


def test_routes_with_the_preferred_classifier():
    stub = StubClassifier(make_signals(task_type="analysis", difficulty=3.0, source="stub"))
    decision = Router(stub).route("Compare the two designs")
    assert decision.tier == "hard"
    assert decision.model == "claude-opus-5-5"
    assert decision.signals.source == "stub"
    assert decision.signals.fallback_reason is None


def test_falls_back_to_the_heuristic_and_explains_why(caplog):
    stub = StubClassifier(error="Jev request failed: timed out")
    with caplog.at_level(logging.WARNING, logger="token_saver.router"):
        decision = Router(stub).route("hi")
    assert decision.signals.source == "heuristic"
    assert decision.signals.fallback_reason == "Jev request failed: timed out"
    assert decision.reasons[0].startswith("classifier fallback to heuristic")
    assert decision.tier == "trivial"
    assert "falling back to heuristic" in caplog.text


def test_strict_routers_raise_instead_of_falling_back():
    stub = StubClassifier(error="boom")
    with pytest.raises(ClassifierError, match="boom"):
        Router(stub, strict=True).route("hi")


def test_a_router_without_a_classifier_uses_the_fallback_only():
    custom = StubClassifier(make_signals(task_type="creative", difficulty=0.0, source="custom"))
    assert Router(None, fallback=custom).route("hi").signals.source == "custom"
    assert isinstance(Router().fallback, HeuristicClassifier)


def test_a_custom_policy_is_honoured():
    policy = Policy.from_yaml(
        """
        tiers:
          - {name: only, model: claude-sonnet-5-5, effort: low}
        difficulty: {levels: [only, only, only, only, only]}
        """
    )
    decision = Router(StubClassifier(make_signals(difficulty=4.0)), policy).route("anything")
    assert (decision.model, decision.effort) == ("claude-sonnet-5-5", "low")


def test_close_saves_the_store_and_closes_the_classifier(tmp_path):
    path = tmp_path / "jev.json"
    store = FixtureStore(path)
    store.put("hello", {"model": "jev-1.13.0", "usage": {}, "answers": {}})
    stub = StubClassifier(make_signals())
    with Router(stub, store=store) as router:
        router.route("hello")
    assert path.exists()
    assert stub.closed


def test_from_env_picks_the_heuristic_without_an_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    router = Router.from_env()
    assert router.classifier is None
    assert router.route("hi").signals.source == "heuristic"


def test_from_env_picks_jev_with_an_api_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    router = Router.from_env(model="jev-1.13.0")
    assert isinstance(router.classifier, JevClassifier)
    assert router.classifier.model == "jev-1.13.0"
    router.close()


def test_from_env_can_force_jev_or_fixtures(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    recording = Router.from_env(classifier="jev", fixtures=tmp_path / "jev.json", record=True)
    assert isinstance(recording.classifier, JevClassifier)
    assert recording.store is not None and recording.classifier.store is recording.store
    recording.close()

    replaying = Router.from_env(classifier="fixture", fixtures=tmp_path / "jev.json")
    assert replaying.classifier is not None and replaying.classifier.name == "fixture"

    with pytest.raises(ValueError, match="needs a fixtures file"):
        Router.from_env(classifier="fixture")
    with pytest.raises(ValueError, match="recording needs a fixtures file"):
        Router.from_env(classifier="jev", record=True)
    with pytest.raises(ValueError, match="classifier must be one of"):
        Router.from_env(classifier="oracle")


def test_module_level_route_uses_one_shared_router(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(router_module, "_default_router", None)
    first = route("hi")
    assert first.tier == "trivial"
    assert router_module.default_router() is router_module.default_router()
