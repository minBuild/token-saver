"""Classifiers turn a prompt into :class:`~token_saver.signals.Signals`.

* :class:`JevClassifier` asks Jev, TypeSafe's System One model (the real thing).
* :class:`FixtureClassifier` replays recorded Jev responses for tests and offline benchmarks.
"""

from token_saver.classifiers.base import Classifier, ClassifierError
from token_saver.classifiers.fixture import FixtureClassifier, FixtureStore, fixture_key
from token_saver.classifiers.jev import JevClassifier

__all__ = [
    "Classifier",
    "ClassifierError",
    "FixtureClassifier",
    "FixtureStore",
    "JevClassifier",
    "fixture_key",
]
