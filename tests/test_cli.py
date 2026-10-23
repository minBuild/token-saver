import io
import json

import pytest

from token_saver.cli import main

HEURISTIC = ["--classifier", "heuristic"]


def test_route_prints_a_readable_decision(capsys, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert main(["route", "hi", *HEURISTIC]) == 0
    out = capsys.readouterr().out
    assert out.startswith("claude-haiku-5-5 @ high  (tier: trivial)")
    assert "task        chitchat" in out
    assert "source      heuristic" in out
    assert "why         difficulty" in out


def test_route_can_emit_json_and_read_stdin(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("안녕! 오늘 기분 어때?"))
    assert main(["route", "-", "--json", *HEURISTIC]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["tier"] == "trivial"
    assert data["signals"]["task_type"] == "chitchat"
    assert data["signals"]["source"] == "heuristic"


def test_route_rejects_an_empty_prompt(capsys):
    assert main(["route", "   ", *HEURISTIC]) == 1
    assert "prompt is empty" in capsys.readouterr().err


def test_route_honours_a_custom_policy_file(tmp_path, capsys):
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "tiers:\n"
        "  - {name: only, model: claude-sonnet-5-5, effort: low}\n"
        "difficulty: {levels: [only, only, only, only, only]}\n",
        encoding="utf-8",
    )
    assert main(["route", "anything at all", "--policy", str(policy), *HEURISTIC]) == 0
    assert capsys.readouterr().out.startswith("claude-sonnet-5-5 @ low  (tier: only)")


def test_broken_policy_files_fail_with_exit_code_1(tmp_path, capsys):
    policy = tmp_path / "policy.yaml"
    policy.write_text("tiers: []\n", encoding="utf-8")
    assert main(["route", "hi", "--policy", str(policy), *HEURISTIC]) == 1
    assert "non-empty list" in capsys.readouterr().err


def test_jev_without_a_key_fails_clearly(monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert main(["route", "hi", "--classifier", "jev"]) == 1
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err


def test_policy_command_shows_the_ladder(capsys):
    assert main(["policy"]) == 0
    out = capsys.readouterr().out
    assert "level 0  trivial   claude-haiku-5-5 @ high" in out
    assert "level 4  expert    claude-fable-5-1 @ xhigh" in out
    assert "deep reasoning  > 0.7: at least hard" in out

    assert main(["policy", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tiers"][0]["name"] == "trivial"


def test_bench_reports_in_three_formats(tmp_path, capsys):
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text(
        '{"id": "a", "prompt": "hi", "expected": "trivial"}\n'
        '{"id": "b", "prompt": "Prove that the sum of the first n odd numbers is n^2 and '
        'compare two proof strategies", "expected": "hard"}\n',
        encoding="utf-8",
    )
    base = ["bench", "--dataset", str(dataset), *HEURISTIC]

    assert main(base) == 0
    summary = capsys.readouterr().out
    assert "prompts            2  (heuristic x2)" in summary
    assert "savings" in summary

    out = tmp_path / "report.md"
    assert main([*base, "--markdown", "--out", str(out)]) == 0
    markdown = capsys.readouterr().out
    assert "| Exact tier match |" in markdown
    assert out.read_text(encoding="utf-8") == markdown

    assert main([*base, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["size"] == 2 and len(data["rows"]) == 2


def test_bench_with_a_missing_dataset_fails_clearly(tmp_path, capsys):
    assert main(["bench", "--dataset", str(tmp_path / "nope.jsonl"), *HEURISTIC]) == 1
    assert "dataset not found" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["route"], ["route", "hi", "--classifier", "oracle"]])
def test_usage_errors_exit_with_argparse_code(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


class FakeExecutor:
    """Stands in for ClaudeExecutor: no SDK, no network."""

    refused = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, prompt, decision, *, on_text=None):
        from token_saver.executor import RunResult

        for delta in ("Hel", "lo"):
            if on_text is not None:
                on_text(delta)
        return RunResult(
            text="Hello",
            model=decision.model,
            stop_reason="refusal" if self.refused else "end_turn",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cost_usd=0.00012,
            decision=decision,
        )


def test_run_streams_the_answer_and_reports_usage_on_stderr(monkeypatch, capsys):
    monkeypatch.setattr("token_saver.cli.ClaudeExecutor", FakeExecutor)
    assert main(["run", "hi", "--max-tokens", "512", *HEURISTIC]) == 0
    captured = capsys.readouterr()
    assert captured.out == "Hello\n"
    assert (
        "[token-saver] claude-haiku-5-5 @ high  (tier: trivial, classifier: heuristic)"
        in captured.err
    )
    assert "answered by claude-haiku-5-5  in 10 / out 5 tokens  cost $0.0001" in captured.err


def test_run_can_print_json_and_flags_refusals(monkeypatch, capsys):
    class Refusing(FakeExecutor):
        refused = True

    monkeypatch.setattr("token_saver.cli.ClaudeExecutor", Refusing)
    assert main(["run", "hi", "--json", "--no-fallbacks", *HEURISTIC]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["refused"] is True
    assert data["decision"]["tier"] == "trivial"


def test_run_without_the_anthropic_sdk_or_key_fails_clearly(monkeypatch, capsys):
    from token_saver.executor import ExecutorError

    def missing(**kwargs):
        raise ExecutorError("The api_key client option must be set")

    monkeypatch.setattr("token_saver.cli.ClaudeExecutor", missing)
    assert main(["run", "hi", *HEURISTIC]) == 1
    assert "api_key" in capsys.readouterr().err
