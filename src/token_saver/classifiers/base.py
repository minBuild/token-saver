from __future__ import annotations

from typing import Protocol, runtime_checkable

from token_saver.signals import Signals


class ClassifierError(RuntimeError):
    """The classifier could not produce signals: network, auth, a bad response, no fixture."""


@runtime_checkable
class Classifier(Protocol):
    """Anything that can judge a prompt. Implementations must be safe to call repeatedly."""

    name: str

    def classify(self, prompt: str) -> Signals: ...
