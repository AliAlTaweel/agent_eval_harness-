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


def test_render_html_report_renders_per_case_table_when_given():
    current = AggregateReport(pass_rate=0.5, avg_hallucination_rate=0.1, avg_cost_usd=0.0, avg_latency_seconds=2.0, total_cases=2)
    per_case = [
        {"name": "clean_diff", "task_success": True, "hallucination_rate": 0.0, "latency_seconds": 1.5},
        {"name": "sql_injection", "task_success": False, "hallucination_rate": 0.5, "latency_seconds": 2.5},
    ]

    html = render_html_report(current, [], per_case=per_case)

    assert "Per-Case Results" in html
    assert "clean_diff" in html
    assert "sql_injection" in html
    assert "pass" in html
    assert "fail" in html


def test_render_html_report_omits_per_case_section_when_not_given():
    current = AggregateReport(pass_rate=1.0, avg_hallucination_rate=0.0, avg_cost_usd=0.0, avg_latency_seconds=1.0, total_cases=3)
    html = render_html_report(current, [])
    assert "Per-Case Results" not in html
