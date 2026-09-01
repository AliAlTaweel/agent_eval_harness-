from harness.aggregator import AggregateReport
from harness.gate import check_gate
from harness.history import HistoryEntry


def _report(pass_rate: float) -> AggregateReport:
    return AggregateReport(pass_rate=pass_rate, avg_hallucination_rate=0.0, avg_cost_usd=0.0, avg_latency_seconds=1.0, total_cases=4)


def test_gate_passes_when_no_history():
    result = check_gate(_report(0.5), [])
    assert result.passed is True


def test_gate_passes_when_pass_rate_holds_or_improves():
    history = [HistoryEntry(commit_sha="a", timestamp="t", report=_report(0.5))]
    result = check_gate(_report(0.75), history)
    assert result.passed is True


def test_gate_fails_on_regression():
    history = [HistoryEntry(commit_sha="a", timestamp="t", report=_report(0.75))]
    result = check_gate(_report(0.5), history)
    assert result.passed is False
    assert "0.5" in result.reason
    assert "0.75" in result.reason
