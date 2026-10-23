"""Classify prompts with Jev, TypeSafe's System One model."""

from __future__ import annotations

import time
from types import TracebackType

from typesafe_sdk import SystemOneResponse, TypeSafeClient, TypeSafeError
from typesafe_sdk.constants import DEFAULT_MODEL

from token_saver.classifiers.base import ClassifierError
from token_saver.classifiers.fixture import FixtureStore
from token_saver.signals import (
    SignalError,
    Signals,
    build_questions,
    build_state,
    signals_from_response,
)

#: Jev answers in roughly 130 ms at p50; anything slower than this is not worth waiting for,
#: because the fallback heuristic is free. LiteLLM's auto-router uses the same 3 s default.
DEFAULT_TIMEOUT_SECONDS = 3.0


class JevClassifier:
    """Ask Jev the token-saver question set and parse the answers into :class:`Signals`.

    Pass ``store`` to record every live response into a :class:`FixtureStore`. Prompts already
    in the store are answered from it without a network call, which keeps repeated benchmark
    runs free and lets the recorded file be replayed later by :class:`FixtureClassifier`.
    """

    name = "jev"

    def __init__(
        self,
        client: TypeSafeClient | None = None,
        *,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        store: FixtureStore | None = None,
    ) -> None:
        self.model = model
        self.store = store
        self._owns_client = client is None
        self._client = (
            client if client is not None else TypeSafeClient(model=model, timeout=timeout)
        )

    def classify(self, prompt: str) -> Signals:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        if self.store is not None:
            recorded = self.store.get(prompt)
            if recorded is not None:
                return recorded.to_signals(source=self.name)

        response, latency_ms = self.ask(prompt)
        try:
            signals = signals_from_response(response, source=self.name, latency_ms=latency_ms)
        except SignalError as exc:
            raise ClassifierError(f"Jev returned an unexpected answer set: {exc}") from exc

        if self.store is not None:
            self.store.put(prompt, response, latency_ms=latency_ms)
        return signals

    def ask(self, prompt: str) -> tuple[SystemOneResponse, float]:
        """Send one System One request and return the raw response with its latency in ms."""
        started = time.perf_counter()
        try:
            response = self._client.system_one(
                build_state(prompt), build_questions(), model=self.model
            )
        except (TypeSafeError, OSError) as exc:
            raise ClassifierError(f"Jev request failed: {exc}") from exc
        return response, (time.perf_counter() - started) * 1000

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> JevClassifier:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
