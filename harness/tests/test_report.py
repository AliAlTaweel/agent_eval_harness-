from harness.aggregator import AggregateReport
from harness.history import HistoryEntry
from harness.report import render_html_report


def test_render_html_report_includes_summary_numbers():
    current = AggregateReport(pass_rate=0.75, avg_hallucination_rate=0.1, avg_cost_usd=0.0, avg_latency_seconds=3.2, total_cases=8)
    history = [
        HistoryEntry(commit_sha="a1", timestamp="2026-08-30T00:00:00Z", report=AggregateReport(pass_rate=0.5, avg_hallucination_rate=0.2, avg_cost_usd=0.0, avg_latency_seconds=4.0, total_cases=8))
    ]

    html = render_html_report(current, history)

    assert "<html" in html.lower()
    assert "0.75" in html
    assert "a1" in html
    assert "0.5" in html


def test_render_html_report_handles_empty_history():
    current = AggregateReport(pass_rate=1.0, avg_hallucination_rate=0.0, avg_cost_usd=0.0, avg_latency_seconds=1.0, total_cases=3)
    html = render_html_report(current, [])
    assert "1.0" in html
