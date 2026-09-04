from datetime import datetime, timedelta, timezone

from trace_schema import RunTrace, Step

from harness.adapters.clinical_review import ClinicalReviewExpected, ClinicalReviewScorer
from harness.judge import FakeHallucinationJudge, JudgeVerdict


def _trace(verdict: str, completeness_issues: list[dict], compliance_issues: list[dict], agent_names: list[str], note_text: str = "the note") -> RunTrace:
    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=4)
    steps = [
        Step(agent_name=name, action="analyze_note", started_at=start, ended_at=end, tokens_in=10, tokens_out=5)
        for name in agent_names
    ]
    return RunTrace(
        run_id="r1",
        input={"note_text": note_text, "encounter_type": "new_patient"},
        steps=steps,
        final_output={
            "verdict": verdict,
            "completeness_issues": completeness_issues,
            "compliance_issues": compliance_issues,
            "coding_clarity_issues": [],
        },
        started_at=start,
        ended_at=end,
        total_tokens=sum(s.tokens_in + s.tokens_out for s in steps),
        total_cost_usd=0.0,
    )


def test_task_success_matches_expected_verdict():
    trace = _trace("needs_revision", [], [], ["completeness_review", "compliance_review", "coding_clarity_review", "merge"])
    expected = ClinicalReviewExpected(
        verdict="needs_revision",
        expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is True
    assert result.tool_call_correctness == 1.0


def test_task_success_false_on_verdict_mismatch():
    trace = _trace("approve", [], [], ["completeness_review", "compliance_review", "coding_clarity_review", "merge"])
    expected = ClinicalReviewExpected(
        verdict="needs_revision",
        expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is False


def test_task_success_false_when_min_completeness_issues_not_met():
    trace = _trace(
        "needs_revision",
        [{"element": "Assessment", "description": "vague wording", "severity": "low"}],
        [],
        ["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    expected = ClinicalReviewExpected(
        verdict="needs_revision",
        expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
        min_completeness_issues=2,
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is False


def test_task_success_false_when_min_compliance_issues_not_met():
    trace = _trace(
        "needs_revision",
        [],
        [],
        ["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    expected = ClinicalReviewExpected(
        verdict="needs_revision",
        expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
        min_compliance_issues=1,
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is False


def test_tool_call_correctness_partial_when_agent_missing():
    trace = _trace("approve", [], [], ["completeness_review", "compliance_review"])
    expected = ClinicalReviewExpected(
        verdict="approve",
        expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.tool_call_correctness == 0.5


def test_hallucination_rate_uses_judge_across_compliance_and_coding_clarity_issues():
    trace = _trace(
        "needs_revision",
        [],
        [{"element": "Compliance", "description": "fabricated compliance gap", "severity": "medium"}],
        ["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    expected = ClinicalReviewExpected(
        verdict="needs_revision",
        expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=True, reasoning="not in note"))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.hallucination_rate == 1.0


def test_hallucination_rate_ignores_completeness_issues():
    trace = _trace(
        "needs_revision",
        [{"element": "ROS", "description": "fabricated gap", "severity": "medium"}],
        [],
        ["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    expected = ClinicalReviewExpected(
        verdict="needs_revision",
        expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=True, reasoning="always hallucinated"))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.hallucination_rate == 0.0


def test_hallucination_rate_zero_when_no_issues_reported():
    trace = _trace("approve", [], [], ["completeness_review", "compliance_review", "coding_clarity_review", "merge"])
    expected = ClinicalReviewExpected(
        verdict="approve",
        expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=True, reasoning="unused"))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.hallucination_rate == 0.0


def test_crashed_trace_fails_task_success_and_flags_crash_in_details():
    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=1)
    trace = RunTrace(
        run_id="r1",
        input={"note_text": "the note", "encounter_type": "new_patient"},
        steps=[],
        final_output={"error": "boom"},
        started_at=start,
        ended_at=end,
        total_tokens=0,
        total_cost_usd=0.0,
    )
    expected = ClinicalReviewExpected(
        verdict="approve", expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"]
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=True, reasoning="unused"))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is False
    assert result.details["crashed"] is True
    assert result.details["error"] == "boom"
