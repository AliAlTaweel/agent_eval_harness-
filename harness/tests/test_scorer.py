from datetime import datetime, timedelta, timezone

from trace_schema import RunTrace

from harness.scorer import ScoreResult, latency_seconds


def _trace(seconds: float) -> RunTrace:
    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=seconds)
    return RunTrace(
        run_id="r1",
        input=None,
        steps=[],
        final_output=None,
        started_at=start,
        ended_at=end,
        total_tokens=0,
        total_cost_usd=0.0,
    )


def test_latency_seconds_computes_duration():
    trace = _trace(2.5)
    assert abs(latency_seconds(trace) - 2.5) < 0.01


def test_score_result_defaults_details_to_empty_dict():
    result = ScoreResult(
        task_success=True,
        tool_call_correctness=1.0,
        hallucination_rate=0.0,
        latency_seconds=1.0,
        cost_usd=0.0,
    )
    assert result.details == {}
