# harness/tests/adapters/test_pr_review.py
from datetime import datetime, timedelta, timezone

from trace_schema import RunTrace, Step

from harness.adapters.pr_review import PrReviewExpected, PrReviewScorer
from harness.judge import FakeHallucinationJudge, JudgeVerdict


def _trace(verdict: str, security_issues: list[dict], agent_names: list[str], diff: str = "the diff") -> RunTrace:
    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=3)
    steps = [
        Step(agent_name=name, action="analyze_diff", started_at=start, ended_at=end, tokens_in=10, tokens_out=5)
        for name in agent_names
    ]
    return RunTrace(
        run_id="r1",
        input={"diff": diff},
        steps=steps,
        final_output={
            "verdict": verdict,
            "security_issues": security_issues,
            "style_issues": [],
            "test_coverage_gap": False,
        },
        started_at=start,
        ended_at=end,
        total_tokens=sum(s.tokens_in + s.tokens_out for s in steps),
        total_cost_usd=0.0,
    )


def test_task_success_matches_expected_verdict():
    trace = _trace("request_changes", [], ["security_review", "style_review", "test_coverage", "merge"])
    expected = PrReviewExpected(
        verdict="request_changes", expected_tool_agents=["security_review", "style_review", "test_coverage", "merge"]
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = PrReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is True
    assert result.tool_call_correctness == 1.0


def test_task_success_false_on_verdict_mismatch():
    trace = _trace("approve", [], ["security_review", "style_review", "test_coverage", "merge"])
    expected = PrReviewExpected(
        verdict="request_changes", expected_tool_agents=["security_review", "style_review", "test_coverage", "merge"]
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = PrReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is False


def test_tool_call_correctness_partial_when_agent_missing():
    trace = _trace("approve", [], ["security_review", "style_review"])
    expected = PrReviewExpected(
        verdict="approve", expected_tool_agents=["security_review", "style_review", "test_coverage", "merge"]
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = PrReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.tool_call_correctness == 0.5


def test_hallucination_rate_uses_judge_per_security_issue():
    trace = _trace(
        "request_changes",
        [{"file": "a.py", "line": 1, "description": "fabricated issue", "severity": "high"}],
        ["security_review", "style_review", "test_coverage", "merge"],
    )
    expected = PrReviewExpected(
        verdict="request_changes", expected_tool_agents=["security_review", "style_review", "test_coverage", "merge"]
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=True, reasoning="not in diff"))
    scorer = PrReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.hallucination_rate == 1.0


def test_hallucination_rate_zero_when_no_issues_reported():
    trace = _trace("approve", [], ["security_review", "style_review", "test_coverage", "merge"])
    expected = PrReviewExpected(
        verdict="approve", expected_tool_agents=["security_review", "style_review", "test_coverage", "merge"]
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=True, reasoning="unused"))
    scorer = PrReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.hallucination_rate == 0.0
