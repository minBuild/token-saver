"""Record and replay Jev responses, for tests, demos and offline benchmark runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from typesafe_sdk import SystemOneResponse

from token_saver.classifiers.base import ClassifierError
from token_saver.signals import SignalError, Signals, signals_from_response

FIXTURE_FORMAT = 1


def fixture_key(prompt: str) -> str:
    """A short, stable id for a prompt. Whitespace runs are collapsed before hashing."""
    normalized = " ".join(prompt.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RecordedResponse:
    prompt: str
    response: dict[str, Any]
    """The raw System One response body, exactly as the API returned it."""
    recorded_at: str
    latency_ms: float | None = None

    def to_signals(self, *, source: str = "fixture") -> Signals:
        # Parse from JSON text, like the SDK does, so integer level keys are coerced.
        parsed = SystemOneResponse.model_validate_json(json.dumps(self.response))
        return signals_from_response(parsed, source=source, latency_ms=self.latency_ms)


class FixtureStore:
    """A JSON file mapping prompt keys to recorded responses. Human-readable and diff-friendly."""

    def __init__(self, path: Path, entries: dict[str, RecordedResponse] | None = None) -> None:
        self.path = path
        self._entries: dict[str, RecordedResponse] = dict(entries or {})

    @classmethod
    def load(cls, path: Path) -> FixtureStore:
        """Load a store, or start an empty one if the file does not exist yet."""
        if not path.exists():
            return cls(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != FIXTURE_FORMAT:
            raise ClassifierError(f"{path}: unsupported fixture format {data.get('format')!r}")
        entries = {
            key: RecordedResponse(
                prompt=entry["prompt"],
                response=entry["response"],
                recorded_at=entry["recorded_at"],
                latency_ms=entry.get("latency_ms"),
            )
            for key, entry in data["entries"].items()
        }
        return cls(path, entries)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": FIXTURE_FORMAT,
            "entries": {
                key: {
                    "prompt": entry.prompt,
                    "recorded_at": entry.recorded_at,
                    "latency_ms": entry.latency_ms,
                    "response": entry.response,
                }
                for key, entry in sorted(self._entries.items())
            },
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def get(self, prompt: str) -> RecordedResponse | None:
        return self._entries.get(fixture_key(prompt))

    def put(
        self,
        prompt: str,
        response: SystemOneResponse | dict[str, Any],
        *,
        latency_ms: float | None = None,
    ) -> RecordedResponse:
        raw = (
            response.model_dump(mode="json")
            if isinstance(response, SystemOneResponse)
            else response
        )
        entry = RecordedResponse(
            prompt=prompt,
            response=raw,
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            latency_ms=latency_ms,
        )
        self._entries[fixture_key(prompt)] = entry
        return entry

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, prompt: object) -> bool:
        return isinstance(prompt, str) and fixture_key(prompt) in self._entries


class FixtureClassifier:
    """Answer only from recorded responses. Never touches the network."""

    name = "fixture"

    def __init__(self, store: FixtureStore | Path) -> None:
        self.store = store if isinstance(store, FixtureStore) else FixtureStore.load(store)

    def classify(self, prompt: str) -> Signals:
        recorded = self.store.get(prompt)
        if recorded is None:
            raise ClassifierError(
                f"no recorded Jev response for this prompt (key {fixture_key(prompt)}) in "
                f"{self.store.path}; record one with `token-saver bench --record`"
            )
        try:
            return recorded.to_signals(source=self.name)
        except (SignalError, ValueError) as exc:
            raise ClassifierError(
                f"recorded response for {fixture_key(prompt)} is invalid: {exc}"
            ) from exc
