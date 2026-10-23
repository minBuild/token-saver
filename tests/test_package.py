import token_saver


def test_the_package_root_exposes_the_public_api():
    assert token_saver.__version__
    assert token_saver.route is token_saver.router.route
    assert token_saver.Router is token_saver.router.Router
    assert {"Decision", "Policy", "Signals", "Tier"} <= set(token_saver.__all__)


def test_the_module_docstring_example_holds(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(token_saver.router, "_default_router", None)
    decision = token_saver.route(
        "Design a rate limiter for a public API across 12 regions "
        "and compare token bucket vs sliding window"
    )
    assert (decision.model, decision.effort, decision.tier) == ("claude-opus-5-5", "xhigh", "hard")
