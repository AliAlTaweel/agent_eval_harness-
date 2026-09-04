from typing import Literal

from pydantic import BaseModel

from trace_schema import RunTrace

from harness.judge import HallucinationJudge
from harness.scorer import ScoreResult, latency_seconds


class ClinicalReviewExpected(BaseModel):
    verdict: Literal["approve", "needs_revision"]
    expected_check_agents: list[str]
    min_completeness_issues: int = 0
    min_compliance_issues: int = 0


class ClinicalReviewScorer:
    def __init__(self, judge: HallucinationJudge):
        self._judge = judge

    def score(self, trace: RunTrace, expected: ClinicalReviewExpected) -> ScoreResult:
        final = trace.final_output or {}

        if final.get("error"):
            return ScoreResult(
                task_success=False,
                tool_call_correctness=0.0,
                hallucination_rate=0.0,
                latency_seconds=latency_seconds(trace),
                cost_usd=trace.total_cost_usd,
                details={"crashed": True, "error": final["error"]},
            )

        completeness_issues = final.get("completeness_issues", [])
        compliance_issues = final.get("compliance_issues", [])
        coding_clarity_issues = final.get("coding_clarity_issues", [])

        task_success = final.get("verdict") == expected.verdict
        if expected.min_completeness_issues > 0:
            task_success = task_success and len(completeness_issues) >= expected.min_completeness_issues
        if expected.min_compliance_issues > 0:
            task_success = task_success and len(compliance_issues) >= expected.min_compliance_issues

        actual_agents = {s.agent_name for s in trace.steps}
        expected_agents = set(expected.expected_check_agents)
        tool_call_correctness = (
            len(expected_agents & actual_agents) / len(expected_agents) if expected_agents else 1.0
        )

        note_text = (trace.input or {}).get("note_text", "")
        all_issues = completeness_issues + compliance_issues + coding_clarity_issues
        if all_issues:
            hallucinated_count = 0
            for issue in all_issues:
                verdict = self._judge.judge(source_material=note_text, claim=issue.get("description", ""))
                if verdict.hallucinated:
                    hallucinated_count += 1
            hallucination_rate = hallucinated_count / len(all_issues)
        else:
            hallucination_rate = 0.0

        return ScoreResult(
            task_success=task_success,
            tool_call_correctness=tool_call_correctness,
            hallucination_rate=hallucination_rate,
            latency_seconds=latency_seconds(trace),
            cost_usd=trace.total_cost_usd,
            details={"actual_verdict": final.get("verdict")},
        )
