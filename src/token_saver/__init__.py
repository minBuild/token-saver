"""token-saver: pick the cheapest Claude model and effort level for each prompt.

>>> from token_saver import route
>>> decision = route(
...     "Design a rate limiter for a public API across 12 regions "
...     "and compare token bucket vs sliding window"
... )
>>> decision.model, decision.effort, decision.tier
('claude-opus-5-5', 'xhigh', 'hard')
"""

from token_saver.policy import Decision, Policy, Tier
from token_saver.router import Router, route
from token_saver.signals import Signals

__version__ = "0.1.0"

__all__ = ["Decision", "Policy", "Router", "Signals", "Tier", "__version__", "route"]
