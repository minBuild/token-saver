"""Wire a classifier and a policy together, with a safe fallback when the classifier fails."""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path
from types import TracebackType

from typesafe_sdk.constants import API_KEY_ENV, DEFAULT_MODEL

from token_saver.classifiers import (
    Classifier,
    ClassifierError,
    FixtureClassifier,
    FixtureStore,
    HeuristicClassifier,
    JevClassifier,
)
from token_saver.classifiers.jev import DEFAULT_TIMEOUT_SECONDS
from token_saver.policy import Decision, Policy
from token_saver.signals import Signals

log = logging.getLogger(__name__)

CLASSIFIER_CHOICES = ("auto", "jev", "heuristic", "fixture")


class Router:
    """Classify a prompt, then let the policy decide. The public entry point of the package.

    ``classifier`` is the preferred source of signals (usually Jev). When it raises
    :class:`ClassifierError`, the ``fallback`` answers instead and the decision says so, unless
    ``strict`` is set, in which case the error propagates. The fallback defaults to the offline
    heuristic, so a router without any classifier still routes.
    """

    def __init__(
        self,
        classifier: Classifier | None = None,
        policy: Policy | None = None,
        *,
        fallback: Classifier | None = None,
        strict: bool = False,
        store: FixtureStore | None = None,
    ) -> None:
        self.classifier = classifier
        self.policy = policy if policy is not None else Policy.default()
        self.fallback = fallback if fallback is not None else HeuristicClassifier()
        self.strict = strict
        self.store = store

    @classmethod
    def from_env(
        cls,
        policy: Policy | None = None,
        *,
        classifier: str = "auto",
        fixtures: Path | None = None,
        record: bool = False,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        strict: bool = False,
    ) -> Router:
        """Build a router from the environment.

        ``classifier`` is one of ``auto`` (Jev when ``TYPESAFE_API_KEY`` is set, otherwise the
        heuristic), ``jev``, ``heuristic`` or ``fixture``. ``fixtures`` names a
        :class:`FixtureStore` file: the ``fixture`` classifier replays it, and Jev records into
        it when ``record`` is set (the file is written on :meth:`close`).
        """
        if classifier not in CLASSIFIER_CHOICES:
            raise ValueError(f"classifier must be one of {', '.join(CLASSIFIER_CHOICES)}")
        if classifier == "auto":
            classifier = "jev" if os.environ.get(API_KEY_ENV) else "heuristic"
            if classifier == "heuristic":
                log.info("%s is not set; routing with the offline heuristic", API_KEY_ENV)

        store = FixtureStore.load(fixtures) if fixtures is not None and record else None
        if classifier == "jev":
            chosen: Classifier | None = JevClassifier(model=model, timeout=timeout, store=store)
        elif classifier == "fixture":
            if fixtures is None:
                raise ValueError("the fixture classifier needs a fixtures file")
            chosen = FixtureClassifier(fixtures)
        else:
            chosen = None
        return cls(chosen, policy, strict=strict, store=store)

    def route(self, prompt: str) -> Decision:
        return self.policy.decide(self.classify(prompt))

    def classify(self, prompt: str) -> Signals:
        if self.classifier is None:
            return self.fallback.classify(prompt)
        try:
            return self.classifier.classify(prompt)
        except ClassifierError as exc:
            if self.strict:
                raise
            log.warning(
                "%s failed (%s); falling back to %s", self.classifier.name, exc, self.fallback.name
            )
            return replace(self.fallback.classify(prompt), fallback_reason=str(exc))

    def close(self) -> None:
        """Persist recorded fixtures and release the classifier's connections."""
        if self.store is not None:
            self.store.save()
        closer = getattr(self.classifier, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> Router:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


_default_router: Router | None = None


def default_router() -> Router:
    """A process-wide router built from the environment, created on first use."""
    global _default_router
    if _default_router is None:
        _default_router = Router.from_env()
    return _default_router


def route(prompt: str) -> Decision:
    """Route one prompt with the default policy and whatever classifier the environment allows."""
    return default_router().route(prompt)
