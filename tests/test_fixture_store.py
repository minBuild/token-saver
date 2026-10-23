import json

import pytest

from tests.helpers import make_raw_response, make_response
from token_saver.classifiers import ClassifierError, FixtureClassifier, FixtureStore, fixture_key


def test_fixture_key_is_short_stable_and_whitespace_insensitive():
    assert fixture_key("a  b\n c") == fixture_key("a b c")
    assert fixture_key("a b c") != fixture_key("a b d")
    assert len(fixture_key("anything")) == 16


def test_store_round_trips_through_json(tmp_path):
    path = tmp_path / "fixtures" / "jev.json"
    store = FixtureStore(path)
    store.put("이 문장을 요약해줘", make_raw_response(task_type="rewrite"), latency_ms=120.5)
    store.save()

    text = path.read_text(encoding="utf-8")
    assert "이 문장을 요약해줘" in text  # stored readable, not as \u escapes
    assert json.loads(text)["format"] == 1

    loaded = FixtureStore.load(path)
    assert len(loaded) == 1
    entry = loaded.get("이 문장을 요약해줘")
    assert entry is not None
    assert entry.prompt == "이 문장을 요약해줘"
    assert entry.latency_ms == 120.5
    assert entry.recorded_at.endswith("+00:00")
    assert entry.to_signals().task_type == "rewrite"


def test_store_accepts_sdk_response_objects():
    store = FixtureStore(path=None)  # type: ignore[arg-type]
    entry = store.put("hello", make_response(task_type="factual"))
    # Dumped in JSON mode, so the file can be written and read back by the SDK parser.
    assert list(entry.response["answers"]["difficulty"]["probabilities"]) == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]
    assert entry.to_signals().task_type == "factual"


def test_loading_a_missing_file_gives_an_empty_store(tmp_path):
    store = FixtureStore.load(tmp_path / "missing.json")
    assert len(store) == 0
    assert "anything" not in store


def test_loading_an_unknown_format_fails_loudly(tmp_path):
    path = tmp_path / "jev.json"
    path.write_text(json.dumps({"format": 99, "entries": {}}), encoding="utf-8")
    with pytest.raises(ClassifierError, match="unsupported fixture format"):
        FixtureStore.load(path)


def test_fixture_classifier_refuses_unknown_prompts(tmp_path):
    classifier = FixtureClassifier(tmp_path / "jev.json")
    with pytest.raises(ClassifierError, match="no recorded Jev response"):
        classifier.classify("never seen before")


def test_fixture_classifier_rejects_corrupt_entries(tmp_path):
    store = FixtureStore(tmp_path / "jev.json")
    raw = make_raw_response()
    del raw["answers"]["deep_reasoning"]
    store.put("hello", raw)
    with pytest.raises(ClassifierError, match="is invalid"):
        FixtureClassifier(store).classify("hello")
