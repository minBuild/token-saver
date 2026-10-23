import json

import httpx2
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

from tests.helpers import make_raw_response
from token_saver.classifiers import ClassifierError, FixtureClassifier, FixtureStore, JevClassifier


def make_client(handler) -> TypeSafeClient:
    """A real SDK client whose HTTP layer is replaced by ``handler``: no network, no retries."""
    return TypeSafeClient(
        api_key="test-key",
        transport=httpx2.MockTransport(handler),
        retry=RetryPolicy(max_retries=0),
        compression=False,
    )


def test_sends_the_question_set_and_parses_the_answers():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=make_raw_response(task_type="coding"))

    classifier = JevClassifier(make_client(handler))
    signals = classifier.classify("Write a function that reverses a string")

    assert seen["url"].endswith("/v1/systemone")
    assert seen["authorization"] == "Bearer test-key"
    body = seen["body"]
    assert body["model"] == "jev-latest"
    assert body["state"] == {"user_message": "Write a function that reverses a string"}
    assert {name: q["type"] for name, q in body["questions"].items()} == {
        "task_type": "choice",
        "difficulty": "score",
        "deep_reasoning": "noul",
        "sensitive": "noul",
    }
    assert signals.task_type == "coding"
    assert signals.source == "jev"
    assert signals.model == "jev-1.13.0"
    assert signals.input_tokens == 180
    assert signals.latency_ms is not None and signals.latency_ms >= 0


def test_a_pinned_model_version_is_sent_as_is():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["model"] = json.loads(request.content)["model"]
        return httpx2.Response(200, json=make_raw_response(model="jev-1.13.0"))

    JevClassifier(make_client(handler), model="jev-1.13.0").classify("hello there")
    assert seen["model"] == "jev-1.13.0"


def test_api_errors_become_classifier_errors():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"detail": "Invalid API key"})

    with pytest.raises(ClassifierError, match="Jev request failed"):
        JevClassifier(make_client(handler)).classify("hello there")


def test_connection_errors_become_classifier_errors():
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused")

    with pytest.raises(ClassifierError, match="Jev request failed"):
        JevClassifier(make_client(handler)).classify("hello there")


def test_an_unexpected_answer_set_is_rejected():
    def handler(request: httpx2.Request) -> httpx2.Response:
        raw = make_raw_response()
        del raw["answers"]["difficulty"]
        return httpx2.Response(200, json=raw)

    with pytest.raises(ClassifierError, match="unexpected answer set"):
        JevClassifier(make_client(handler)).classify("hello there")


def test_a_missing_api_key_is_reported_as_a_classifier_error(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ClassifierError, match="TYPESAFE_API_KEY"):
        JevClassifier()


def test_empty_prompts_are_rejected_before_any_request():
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no request expected")

    with pytest.raises(ValueError, match="empty"):
        JevClassifier(make_client(handler)).classify("   ")


def test_store_records_live_answers_and_replays_them_without_the_network(tmp_path):
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(200, json=make_raw_response(task_type="rewrite"))

    path = tmp_path / "jev.json"
    store = FixtureStore(path)
    classifier = JevClassifier(make_client(handler), store=store)

    first = classifier.classify("Summarize this paragraph")
    second = classifier.classify("Summarize   this paragraph")  # whitespace-insensitive key

    assert calls == 1
    assert "Summarize this paragraph" in store
    assert first.task_type == second.task_type == "rewrite"
    assert second.source == "jev"

    store.save()
    replayed = FixtureClassifier(path).classify("Summarize this paragraph")
    assert replayed.task_type == "rewrite"
    assert replayed.source == "fixture"
    assert replayed.latency_ms == first.latency_ms


def test_context_manager_closes_an_owned_client_only():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=make_raw_response())

    shared = make_client(handler)
    with JevClassifier(shared) as classifier:
        classifier.classify("hello there")
    # The injected client is still usable after the classifier is closed.
    JevClassifier(shared).classify("hello again")
