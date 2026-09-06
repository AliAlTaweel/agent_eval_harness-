import json
from datetime import datetime, timezone
from pathlib import Path

import click

from trace_schema import RunTrace

from harness.adapters.clinical_review import ClinicalReviewExpected, ClinicalReviewScorer
from harness.aggregator import aggregate
from harness.gate import check_gate
from harness.history import HistoryEntry, append_history, load_history
from harness.judge import HallucinationJudge
from harness.report import render_html_report


@click.group()
def cli():
    pass


def _run_clinical_case(note_text: str, encounter_type: str) -> RunTrace:
    try:
        from clinical_review.graph import ReviewFailedError, run_review
        from clinical_review.llm import OllamaClient
    except ImportError:
        raise click.ClickException(
            "The 'clinical_review' package is required to run 'harness gate-clinical-review' — "
            "install it as part of this uv workspace (uv sync from the repo root)."
        ) from None

    try:
        _, trace = run_review(note_text=note_text, encounter_type=encounter_type, client=OllamaClient())
    except ReviewFailedError as exc:
        click.echo(f"WARNING: case failed with error: {exc}", err=True)
        return exc.trace
    return trace


@cli.command(name="gate-clinical-review")
@click.option("--testcases-dir", required=True, type=click.Path(exists=True))
@click.option("--history-path", required=True, type=click.Path())
@click.option("--report-out", required=True, type=click.Path())
@click.option("--commit-sha", required=True)
def gate_clinical_review(testcases_dir, history_path, report_out, commit_sha):
    judge = HallucinationJudge()
    scorer = ClinicalReviewScorer(judge=judge)

    results = []
    per_case = []
    for case_dir in sorted(Path(testcases_dir).iterdir()):
        if not case_dir.is_dir():
            continue
        input_data = json.loads((case_dir / "input.json").read_text())
        expected_data = json.loads((case_dir / "expected.json").read_text())
        expected = ClinicalReviewExpected.model_validate(expected_data)

        trace = _run_clinical_case(input_data["note_text"], input_data["encounter_type"])
        result = scorer.score(trace, expected)
        results.append(result)
        per_case.append(
            {
                "name": case_dir.name,
                "task_success": result.task_success,
                "hallucination_rate": result.hallucination_rate,
                "latency_seconds": result.latency_seconds,
            }
        )

    current = aggregate(results)
    history = load_history(history_path)
    gate_result = check_gate(current, history)

    entry = HistoryEntry(commit_sha=commit_sha, timestamp=datetime.now(timezone.utc).isoformat(), report=current)
    append_history(history_path, entry)

    html = render_html_report(current, load_history(history_path), per_case=per_case)
    Path(report_out).write_text(html)

    click.echo(gate_result.reason)
    if not gate_result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
