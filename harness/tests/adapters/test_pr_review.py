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


def test_task_success_false_when_min_security_issues_not_met():
    # Correct verdict but the agent didn't actually surface the real
    # vulnerability (e.g. it hallucinated some unrelated issue instead).
    trace = _trace(
        "request_changes",
        [{"file": "a.py", "line": 1, "description": "unrelated nit", "severity": "low"}],
        ["security_review", "style_review", "test_coverage", "merge"],
    )
    expected = PrReviewExpected(
        verdict="request_changes",
        expected_tool_agents=["security_review", "style_review", "test_coverage", "merge"],
        min_security_issues=2,
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = PrReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is False


def test_task_success_unaffected_when_min_security_issues_is_zero():
    trace = _trace("approve", [], ["security_review", "style_review", "test_coverage", "merge"])
    expected = PrReviewExpected(
        verdict="approve",
        expected_tool_agents=["security_review", "style_review", "test_coverage", "merge"],
        min_security_issues=0,
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = PrReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is True


def test_task_success_checks_expected_test_coverage_gap_when_set():
    start_kwargs = dict(
        verdict="request_changes",
        security_issues=[],
        agent_names=["security_review", "style_review", "test_coverage", "merge"],
    )
    trace = _trace(**start_kwargs)
    # trace's final_output always sets test_coverage_gap=False (see _trace helper)
    expected = PrReviewExpected(
        verdict="request_changes",
        expected_tool_agents=["security_review", "style_review", "test_coverage", "merge"],
        expected_test_coverage_gap=True,
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = PrReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is False


def test_task_success_ignores_test_coverage_gap_when_not_set():
    trace = _trace("approve", [], ["security_review", "style_review", "test_coverage", "merge"])
    expected = PrReviewExpected(
        verdict="approve",
        expected_tool_agents=["security_review", "style_review", "test_coverage", "merge"],
        expected_test_coverage_gap=None,
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = PrReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is True


def test_crashed_trace_fails_task_success_and_flags_crash_in_details():
    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=1)
    trace = RunTrace(
        run_id="r1",
        input={"diff": "the diff"},
        steps=[],
        final_output={"error": "boom"},
        started_at=start,
        ended_at=end,
        total_tokens=0,
        total_cost_usd=0.0,
    )
    expected = PrReviewExpected(
        verdict="approve", expected_tool_agents=["security_review", "style_review", "test_coverage", "merge"]
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=True, reasoning="unused"))
    scorer = PrReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is False
    assert result.details["crashed"] is True
    assert result.details["error"] == "boom"
