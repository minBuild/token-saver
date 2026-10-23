from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from tests.helpers import make_signals
from token_saver.executor import FALLBACK_BETA, ClaudeExecutor, ExecutorError
from token_saver.policy import Policy


class FakeStream:
    def __init__(self, chunks, message):
        self._chunks = chunks
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    @property
    def text_stream(self):
        yield from self._chunks

    def get_final_message(self):
        return self._message


class FakeMessagesAPI:
    def __init__(self, chunks, message, error=None):
        self.chunks = chunks
        self.message = message
        self.error = error
        self.calls: list[dict] = []

    def stream(self, **params):
        self.calls.append(params)
        if self.error is not None:
            raise self.error
        return FakeStream(self.chunks, self.message)


def fake_message(*, text="answer", model="claude-sonnet-5-5", stop_reason="end_turn", cache=0):
    return SimpleNamespace(
        model=model,
        stop_reason=stop_reason,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(input_tokens=120, output_tokens=40, cache_read_input_tokens=cache),
    )


def fake_client(chunks=("ans", "wer"), message=None, error=None):
    message = message or fake_message()
    plain = FakeMessagesAPI(chunks, message, error)
    beta = FakeMessagesAPI(chunks, message, error)
    return SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta)), plain, beta


def decide(difficulty: float):
    return Policy.default().decide(make_signals(task_type="creative", difficulty=difficulty))


def test_runs_with_the_decided_model_and_effort_and_streams_text():
    client, plain, beta = fake_client()
    seen: list[str] = []
    decision = decide(2.0)  # sonnet @ xhigh
    result = ClaudeExecutor(client, max_tokens=2048, system="Be brief.").run(
        "Explain locking", decision, on_text=seen.append
    )

    assert seen == ["ans", "wer"]
    assert result.text == "answer"
    assert result.model == "claude-sonnet-5-5"
    assert result.decision is decision
    assert (result.input_tokens, result.output_tokens) == (120, 40)
    assert result.cost_usd == pytest.approx((120 * 2 + 40 * 10) / 1_000_000)
    assert not result.refused

    assert plain.calls == []
    (params,) = beta.calls
    assert params["model"] == "claude-sonnet-5-5"
    assert params["output_config"] == {"effort": "xhigh"}
    assert params["max_tokens"] == 2048
    assert params["system"] == "Be brief."
    assert params["messages"] == [{"role": "user", "content": "Explain locking"}]
    assert params["betas"] == [FALLBACK_BETA]
    assert params["fallbacks"] == "default"
    assert "thinking" not in params


def test_models_without_server_fallbacks_use_the_plain_messages_api():
    client, plain, beta = fake_client(message=fake_message(model="claude-haiku-5-5"))
    ClaudeExecutor(client).run("hi", decide(0.0))
    assert beta.calls == []
    (params,) = plain.calls
    assert params["model"] == "claude-haiku-5-5"
    assert "fallbacks" not in params and "betas" not in params


def test_server_fallbacks_can_be_switched_off():
    client, plain, beta = fake_client()
    ClaudeExecutor(client, server_fallbacks=False).run("hi", decide(2.0))
    assert beta.calls == [] and len(plain.calls) == 1


def test_the_answering_model_is_priced_and_refusals_are_flagged():
    message = fake_message(model="claude-opus-4-8", stop_reason="refusal", text="", cache=1000)
    client, _, _ = fake_client(chunks=(), message=message)
    result = ClaudeExecutor(client).run("hi", decide(4.0))  # fable -> fell back to opus 4.8
    assert result.model == "claude-opus-4-8"
    assert result.refused
    assert result.cache_read_tokens == 1000
    assert result.cost_usd == pytest.approx((1120 * 5 + 40 * 25) / 1_000_000)
    assert result.to_dict()["refused"] is True


def test_unknown_models_have_no_cost_instead_of_crashing():
    client, _, _ = fake_client(message=fake_message(model="claude-mystery-7"))
    assert ClaudeExecutor(client).run("hi", decide(2.0)).cost_usd is None


def test_sdk_errors_become_executor_errors_and_others_propagate():
    response = httpx2.Response(
        401, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    error = anthropic.AuthenticationError("bad key", response=response, body=None)
    client, _, _ = fake_client(error=error)
    with pytest.raises(ExecutorError, match="Claude request failed"):
        ClaudeExecutor(client).run("hi", decide(2.0))

    client, _, _ = fake_client(error=ZeroDivisionError("not ours"))
    with pytest.raises(ZeroDivisionError):
        ClaudeExecutor(client).run("hi", decide(2.0))


def test_a_missing_api_key_is_reported_clearly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(anthropic, "Anthropic", _raise_missing_key)
    with pytest.raises(ExecutorError, match="api_key"):
        ClaudeExecutor()


def _raise_missing_key(*args, **kwargs):
    raise anthropic.AnthropicError("The api_key client option must be set")
