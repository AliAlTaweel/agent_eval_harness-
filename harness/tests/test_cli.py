import json

from click.testing import CliRunner

from harness.aggregator import AggregateReport
from harness.cli import cli


def test_gate_command_passes_with_no_prior_history(tmp_path, monkeypatch):
    testcases_dir = tmp_path / "testcases"
    case_dir = testcases_dir / "clean_diff"
    case_dir.mkdir(parents=True)
    (case_dir / "diff.patch").write_text("+ def add(a, b): return a + b")
    (case_dir / "expected.json").write_text(
        json.dumps({"verdict": "approve", "expected_tool_agents": ["security_review"], "min_security_issues": 0})
    )

    history_path = tmp_path / "history.json"
    report_out = tmp_path / "report.html"

    def fake_run_case(diff_text):
        from datetime import datetime, timezone

        from trace_schema import RunTrace

        now = datetime.now(timezone.utc)
        trace = RunTrace(
            run_id="r1",
            input={"diff": diff_text},
            steps=[{"agent_name": "security_review", "action": "analyze_diff", "started_at": now.isoformat(), "ended_at": now.isoformat(), "tool_calls": [], "output": None, "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0}],
            final_output={"verdict": "approve", "security_issues": [], "style_issues": [], "test_coverage_gap": False},
            started_at=now,
            ended_at=now,
            total_tokens=2,
            total_cost_usd=0.0,
        )
        return trace

    monkeypatch.setattr("harness.cli._run_case", lambda diff_text: fake_run_case(diff_text))
    monkeypatch.setattr(
        "harness.cli.HallucinationJudge",
        lambda *a, **k: type("J", (), {"judge": staticmethod(lambda **kw: __import__("harness.judge", fromlist=["JudgeVerdict"]).JudgeVerdict(hallucinated=False, reasoning=""))})(),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "gate",
            "--testcases-dir",
            str(testcases_dir),
            "--history-path",
            str(history_path),
            "--report-out",
            str(report_out),
            "--commit-sha",
            "abc123",
        ],
    )

    assert result.exit_code == 0, result.output
    assert report_out.exists()
    assert history_path.exists()
