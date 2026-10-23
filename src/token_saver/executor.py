"""Send a routed prompt to Claude with the chosen model and effort, and report what it cost.

The Anthropic SDK is an optional dependency (``pip install "token-saver[claude]"``): routing
never needs it, only running does. Thinking is left at the API default (adaptive on every
current model) and depth is controlled through ``output_config.effort``, which is exactly the
knob the policy decides.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from token_saver.policy import Decision
from token_saver.pricing import UnknownModelError, estimate_cost

#: Models whose safety classifiers may decline a request. For these the API can re-run the
#: request on a substitute model inside the same call when asked to (`fallbacks: "default"`).
FALLBACK_CAPABLE_MODELS = frozenset(
    {"claude-fable-5-1", "claude-fable-5", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)
FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_MAX_TOKENS = 16_000


class ExecutorError(RuntimeError):
    """Claude could not be called: missing SDK, missing credentials, or an API error."""


@dataclass(frozen=True)
class RunResult:
    text: str
    model: str
    """The model that answered. Differs from the decision after a server-side fallback."""
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: float | None
    """List-price cost of this call, or None when the answering model has no known price."""
    decision: Decision

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "refused": self.refused,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cost_usd": self.cost_usd,
            "decision": self.decision.to_dict(),
        }


class ClaudeExecutor:
    """Run prompts through the Anthropic SDK using a :class:`Decision`.

    ``client`` is an ``anthropic.Anthropic`` instance; when omitted one is built from the
    environment (``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile).
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system: str | None = None,
        server_fallbacks: bool = True,
    ) -> None:
        self.max_tokens = max_tokens
        self.system = system
        self.server_fallbacks = server_fallbacks
        self._client = client if client is not None else _default_client()

    def run(
        self,
        prompt: str,
        decision: Decision,
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> RunResult:
        """Stream one answer. ``on_text`` receives each text delta as it arrives."""
        params: dict[str, Any] = {
            "model": decision.model,
            "max_tokens": self.max_tokens,
            "output_config": {"effort": decision.effort},
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system:
            params["system"] = self.system

        api = self._client.messages
        if self.server_fallbacks and decision.model in FALLBACK_CAPABLE_MODELS:
            api = self._client.beta.messages
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"

        try:
            with api.stream(**params) as stream:
                for delta in stream.text_stream:
                    if on_text is not None:
                        on_text(delta)
                message = stream.get_final_message()
        except Exception as exc:
            if _is_anthropic_error(exc):
                raise ExecutorError(f"Claude request failed: {exc}") from exc
            raise

        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        usage = message.usage
        cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
        try:
            cost: float | None = estimate_cost(
                message.model, usage.input_tokens + cache_read, usage.output_tokens
            )
        except UnknownModelError:
            cost = None
        return RunResult(
            text=text,
            model=message.model,
            stop_reason=message.stop_reason,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=cache_read,
            cost_usd=cost,
            decision=decision,
        )


def _default_client() -> Any:
    try:
        import anthropic
    except ImportError:
        raise ExecutorError(
            "the anthropic package is not installed; run: pip install 'token-saver[claude]'"
        ) from None
    try:
        return anthropic.Anthropic()
    except anthropic.AnthropicError as exc:
        raise ExecutorError(str(exc)) from exc


def _is_anthropic_error(exc: BaseException) -> bool:
    try:
        import anthropic
    except ImportError:  # pragma: no cover - the SDK produced the exception, so it is there
        return False
    return isinstance(exc, anthropic.AnthropicError)
