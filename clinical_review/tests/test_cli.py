import json

from click.testing import CliRunner

from clinical_review.cli import cli
from clinical_review.schemas import Finding, Verdict


def test_review_command_writes_trace_and_comment(tmp_path, monkeypatch):
    note_file = tmp_path / "note.txt"
    note_file.write_text("CHIEF COMPLAINT: cough.\nASSESSMENT: viral illness.\nPLAN: rest.")

    trace_out = tmp_path / "trace.json"
    comment_out = tmp_path / "comment.md"

    canned_verdict = Verdict(
        completeness_issues=[Finding(element="Review of Systems", description="missing", severity="medium")],
        compliance_issues=[],
        coding_clarity_issues=[],
        verdict="needs_revision",
    )

    def fake_run_review(note_text, encounter_type, client):
        from datetime import datetime, timezone

        from trace_schema import RunTrace

        now = datetime.now(timezone.utc)
        trace = RunTrace(
            run_id="test-run",
            input={"note_text": note_text, "encounter_type": encounter_type},
            steps=[],
            final_output=canned_verdict.model_dump(),
            started_at=now,
            ended_at=now,
            total_tokens=0,
            total_cost_usd=0.0,
        )
        return canned_verdict, trace

    monkeypatch.setattr("clinical_review.cli.run_review", fake_run_review)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "review",
            "--note-file",
            str(note_file),
            "--encounter-type",
            "new_patient",
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
    assert "Review of Systems" in comment_text

    output_json = json.loads(result.output)
    assert output_json["verdict"] == "needs_revision"


def test_review_command_rejects_invalid_encounter_type(tmp_path):
    note_file = tmp_path / "note.txt"
    note_file.write_text("some note")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["review", "--note-file", str(note_file), "--encounter-type", "annual_physical"],
    )

    assert result.exit_code != 0
