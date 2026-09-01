import json
from datetime import datetime, timezone
from pathlib import Path

import click

from trace_schema import RunTrace

from harness.adapters.pr_review import PrReviewExpected, PrReviewScorer
from harness.aggregator import aggregate
from harness.gate import check_gate
from harness.history import HistoryEntry, append_history, load_history
from harness.judge import HallucinationJudge
from harness.report import render_html_report


def _run_case(diff_text: str) -> RunTrace:
    try:
        from agent.graph import ReviewFailedError, run_review
        from agent.llm import OllamaClient
    except ImportError:
        raise click.ClickException(
            "The 'agent' package is required to run 'harness gate' — install it as part of "
            "this uv workspace (uv sync from the repo root)."
        ) from None

    try:
        _, trace = run_review(diff=diff_text, client=OllamaClient())
    except ReviewFailedError as exc:
        click.echo(f"WARNING: case failed with error: {exc}", err=True)
        return exc.trace
    return trace


@click.group()
def cli():
    pass


@cli.command()
@click.option("--testcases-dir", required=True, type=click.Path(exists=True))
@click.option("--history-path", required=True, type=click.Path())
@click.option("--report-out", required=True, type=click.Path())
@click.option("--commit-sha", required=True)
def gate(testcases_dir, history_path, report_out, commit_sha):
    judge = HallucinationJudge()
    scorer = PrReviewScorer(judge=judge)

    results = []
    per_case = []
    for case_dir in sorted(Path(testcases_dir).iterdir()):
        if not case_dir.is_dir():
            continue
        diff_text = (case_dir / "diff.patch").read_text()
        expected_data = json.loads((case_dir / "expected.json").read_text())
        expected = PrReviewExpected.model_validate(expected_data)

        trace = _run_case(diff_text)
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
