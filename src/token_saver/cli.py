"""The ``token-saver`` command line: route one prompt, benchmark a prompt set, show the policy."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from typesafe_sdk.constants import DEFAULT_MODEL

from token_saver import __version__
from token_saver.bench import (
    DEFAULT_BASELINE_EFFORT,
    DEFAULT_BASELINE_MODEL,
    DEFAULT_OUTPUT_TOKENS,
    load_cases,
    run_bench,
)
from token_saver.classifiers import ClassifierError
from token_saver.classifiers.jev import DEFAULT_TIMEOUT_SECONDS
from token_saver.policy import EFFORTS, Decision, Effort, Policy, PolicyError
from token_saver.pricing import UnknownModelError
from token_saver.router import CLASSIFIER_CHOICES, Router

DEFAULT_DATASET = Path("bench/prompts.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="token-saver",
        description="Pick the cheapest Claude model and effort level that can handle a prompt.",
    )
    parser.add_argument("--version", action="version", version=f"token-saver {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log classifier fallbacks")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--policy", type=Path, metavar="FILE", help="routing policy YAML")
    common.add_argument(
        "--classifier",
        choices=CLASSIFIER_CHOICES,
        default="auto",
        help="auto picks jev when TYPESAFE_API_KEY is set, else heuristic (default: auto)",
    )
    common.add_argument("--fixtures", type=Path, metavar="FILE", help="recorded Jev responses")
    common.add_argument(
        "--record", action="store_true", help="record live Jev answers into --fixtures"
    )
    common.add_argument("--model", default=DEFAULT_MODEL, help="Jev model or alias")
    common.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Jev timeout in seconds"
    )
    common.add_argument(
        "--strict", action="store_true", help="fail instead of falling back to the heuristic"
    )
    common.add_argument("--json", action="store_true", help="print JSON instead of text")

    subparsers = parser.add_subparsers(dest="command", required=True)

    route = subparsers.add_parser("route", parents=[common], help="decide model and effort")
    route.add_argument("prompt", help="the prompt, or - to read it from stdin")

    bench = subparsers.add_parser("bench", parents=[common], help="score a labelled prompt set")
    bench.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="JSON Lines file")
    bench.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    bench.add_argument("--baseline-effort", choices=EFFORTS, default=DEFAULT_BASELINE_EFFORT)
    bench.add_argument(
        "--output-tokens",
        type=int,
        default=DEFAULT_OUTPUT_TOKENS,
        help="assumed output tokens per request for the cost estimate",
    )
    bench.add_argument("--markdown", action="store_true", help="print a Markdown report")
    bench.add_argument("--out", type=Path, metavar="FILE", help="also write the report here")

    policy = subparsers.add_parser("policy", help="show the routing ladder")
    policy.add_argument("--policy", type=Path, metavar="FILE", help="routing policy YAML")
    policy.add_argument("--json", action="store_true", help="print JSON instead of text")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.verbose else logging.ERROR, format="%(message)s"
    )
    _tolerate_console_encoding()
    try:
        if args.command == "route":
            return _route(args)
        if args.command == "bench":
            return _bench(args)
        return _policy(args)
    except (ClassifierError, PolicyError, UnknownModelError, ValueError, OSError) as exc:
        print(f"token-saver: {exc}", file=sys.stderr)
        return 1


def _load_policy(path: Path | None) -> Policy:
    return Policy.load(path) if path is not None else Policy.default()


def _build_router(args: argparse.Namespace) -> Router:
    return Router.from_env(
        _load_policy(args.policy),
        classifier=args.classifier,
        fixtures=args.fixtures,
        record=args.record,
        model=args.model,
        timeout=args.timeout,
        strict=args.strict,
    )


def _route(args: argparse.Namespace) -> int:
    prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
    if not prompt.strip():
        raise ValueError("prompt is empty")
    with _build_router(args) as router:
        decision = router.route(prompt)
    if args.json:
        print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_decision(decision))
    return 0


def _bench(args: argparse.Namespace) -> int:
    policy = _load_policy(args.policy)
    if not args.dataset.exists():
        raise ValueError(f"dataset not found: {args.dataset}")
    cases = load_cases(args.dataset, policy)
    with _build_router(args) as router:
        report = run_bench(
            cases,
            router,
            baseline_model=args.baseline_model,
            baseline_effort=cast(Effort, args.baseline_effort),
            output_tokens=args.output_tokens,
        )
    if args.json:
        text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    elif args.markdown:
        text = report.to_markdown()
    else:
        text = format_report_summary(report.to_dict())
    print(text, end="" if text.endswith("\n") else "\n")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return 0


def _policy(args: argparse.Namespace) -> int:
    policy = _load_policy(args.policy)
    if args.json:
        print(json.dumps(policy.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(format_policy(policy))
    return 0


# -- formatting --------------------------------------------------------------------------


def format_decision(decision: Decision) -> str:
    s = decision.signals
    source = s.source
    if s.model:
        source += f" {s.model}"
    if s.latency_ms is not None:
        source += f"  {s.latency_ms:.0f} ms"
    if s.input_tokens:
        source += f"  {s.input_tokens} tokens"
    lines = [
        f"{decision.model} @ {decision.effort}  (tier: {decision.tier})",
        f"  task        {s.task_type}  (confidence {s.task_type_confidence:.2f})",
        f"  difficulty  {s.difficulty:.2f} -> level {s.difficulty_level}  "
        f"(confidence {s.difficulty_confidence:.2f})",
        f"  reasoning   {s.deep_reasoning:.2f}   sensitive {s.sensitive:.2f}",
        f"  source      {source}",
    ]
    for index, reason in enumerate(decision.reasons):
        lines.append(f"  {'why' if index == 0 else '   '}         {reason}")
    return "\n".join(lines)


def format_policy(policy: Policy) -> str:
    width = max(len(tier.name) for tier in policy.tiers)
    lines = [f"version {policy.version}", ""]
    for level, name in enumerate(policy.difficulty_levels):
        tier = policy.tiers[policy.tier_index(name)]
        note = f"  {tier.note}" if tier.note else ""
        lines.append(f"  level {level}  {name:<{width}}  {tier.model} @ {tier.effort}{note}")
    lines.append("")
    if policy.low_confidence:
        lines.append(
            f"  low confidence  < {policy.low_confidence.below}: escalate "
            f"{policy.low_confidence.escalate_by} tier(s)"
        )
    if policy.deep_reasoning:
        lines.append(
            f"  deep reasoning  > {policy.deep_reasoning.above}: at least "
            f"{policy.deep_reasoning.min_tier}"
        )
    if policy.floors:
        lines.append(
            "  floors          " + ", ".join(f"{k} >= {v}" for k, v in policy.floors.items())
        )
    if policy.ceilings:
        lines.append(
            "  ceilings        " + ", ".join(f"{k} <= {v}" for k, v in policy.ceilings.items())
        )
    lines.append(f"  bounds apply when task type confidence >= {policy.task_type_min_confidence}")
    return "\n".join(lines)


def format_report_summary(report: dict[str, Any]) -> str:
    latency = report["mean_latency_ms"]
    sources = ", ".join(f"{k} x{v}" for k, v in sorted(report["sources"].items()))
    return "\n".join(
        [
            f"prompts            {report['size']}  ({sources})",
            f"exact tier match   {report['exact_rate']:.0%}",
            f"within one tier    {report['within_one_rate']:.0%}",
            f"claude spend       ${report['routed_cost']:.4f} routed vs "
            f"${report['baseline_cost']:.4f} baseline "
            f"({report['baseline']['model']} @ {report['baseline']['effort']})",
            f"classifier spend   ${report['classifier_cost']:.6f}",
            f"savings            {report['savings']:.1%}",
            f"classifier latency {latency:.1f} ms"
            if latency is not None
            else "classifier latency n/a",
            f"fallbacks          {report['fallbacks']}",
        ]
    )


def _tolerate_console_encoding() -> None:
    """Keep Korean prompts from crashing a legacy Windows console: replace, don't raise."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="replace")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
