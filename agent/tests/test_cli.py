import json

from click.testing import CliRunner

from agent.cli import cli
from agent.schemas import Issue, Verdict


def test_review_command_writes_trace_and_comment(tmp_path, monkeypatch):
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text("+ query = f'SELECT {x}'")

    trace_out = tmp_path / "trace.json"
    comment_out = tmp_path / "comment.md"

    canned_verdict = Verdict(
        security_issues=[Issue(file="a.py", line=1, description="sqli", severity="critical")],
        style_issues=[],
        test_coverage_gap=False,
        verdict="request_changes",
    )

    def fake_run_review(diff, client):
        from datetime import datetime, timezone

        from trace_schema import RunTrace

        now = datetime.now(timezone.utc)
        trace = RunTrace(
            run_id="test-run",
            input={"diff": diff},
            steps=[],
            final_output=canned_verdict.model_dump(),
            started_at=now,
            ended_at=now,
            total_tokens=0,
            total_cost_usd=0.0,
        )
        return canned_verdict, trace

    monkeypatch.setattr("agent.cli.run_review", fake_run_review)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "review",
            "--diff-file",
            str(diff_file),
            "--trace-out",
            str(trace_out),
            "--comment-out",
            str(comment_out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert trace_out.exists()
    assert comment_out.exists()

    trace_data = json.loads(trace_out.read_text())
    assert trace_data["run_id"] == "test-run"

    comment_text = comment_out.read_text()
    assert "sqli" in comment_text

    output_json = json.loads(result.output)
    assert output_json["verdict"] == "request_changes"
