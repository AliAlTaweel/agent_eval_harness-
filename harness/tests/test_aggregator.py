from harness.aggregator import aggregate
from harness.scorer import ScoreResult


def test_aggregate_computes_pass_rate_and_averages():
    results = [
        ScoreResult(task_success=True, tool_call_correctness=1.0, hallucination_rate=0.0, latency_seconds=2.0, cost_usd=0.0),
        ScoreResult(task_success=False, tool_call_correctness=0.5, hallucination_rate=0.5, latency_seconds=4.0, cost_usd=0.0),
    ]

    report = aggregate(results)

    assert report.total_cases == 2
    assert report.pass_rate == 0.5
    assert report.avg_hallucination_rate == 0.25
    assert report.avg_latency_seconds == 3.0
    assert report.avg_cost_usd == 0.0


def test_aggregate_empty_results():
    report = aggregate([])
    assert report.total_cases == 0
    assert report.pass_rate == 0.0
