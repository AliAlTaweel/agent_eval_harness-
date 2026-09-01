# harness/src/harness/adapters/pr_review.py
from typing import Literal

from pydantic import BaseModel

from trace_schema import RunTrace

from harness.judge import HallucinationJudge
from harness.scorer import ScoreResult, latency_seconds


class PrReviewExpected(BaseModel):
    verdict: Literal["approve", "request_changes"]
    expected_tool_agents: list[str]
    min_security_issues: int = 0


class PrReviewScorer:
    def __init__(self, judge: HallucinationJudge):
        self._judge = judge

    def score(self, trace: RunTrace, expected: PrReviewExpected) -> ScoreResult:
        final = trace.final_output or {}
        task_success = final.get("verdict") == expected.verdict

        actual_agents = {s.agent_name for s in trace.steps}
        expected_agents = set(expected.expected_tool_agents)
        tool_call_correctness = (
            len(expected_agents & actual_agents) / len(expected_agents) if expected_agents else 1.0
        )

        security_issues = final.get("security_issues", [])
        diff = (trace.input or {}).get("diff", "")
        if security_issues:
            hallucinated_count = 0
            for issue in security_issues:
                verdict = self._judge.judge(source_material=diff, claim=issue.get("description", ""))
                if verdict.hallucinated:
                    hallucinated_count += 1
            hallucination_rate = hallucinated_count / len(security_issues)
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
