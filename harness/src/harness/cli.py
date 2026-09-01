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
    from agent.graph import ReviewFailedError, run_review
    from agent.llm import OllamaClient

    try:
        _, trace = run_review(diff=diff_text, client=OllamaClient())
    except ReviewFailedError as exc:
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
    for case_dir in sorted(Path(testcases_dir).iterdir()):
        if not case_dir.is_dir():
            continue
        diff_text = (case_dir / "diff.patch").read_text()
        expected_data = json.loads((case_dir / "expected.json").read_text())
        expected = PrReviewExpected.model_validate(expected_data)

        trace = _run_case(diff_text)
        results.append(scorer.score(trace, expected))

    current = aggregate(results)
    history = load_history(history_path)
    gate_result = check_gate(current, history)

    entry = HistoryEntry(commit_sha=commit_sha, timestamp=datetime.now(timezone.utc).isoformat(), report=current)
    append_history(history_path, entry)

    html = render_html_report(current, load_history(history_path))
    Path(report_out).write_text(html)

    click.echo(gate_result.reason)
    if not gate_result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
