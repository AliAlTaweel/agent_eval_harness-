from pydantic import BaseModel

from harness.aggregator import AggregateReport
from harness.history import HistoryEntry


class GateResult(BaseModel):
    passed: bool
    reason: str


def check_gate(current: AggregateReport, history: list[HistoryEntry]) -> GateResult:
    if not history:
        return GateResult(passed=True, reason="no baseline yet; recording first result")

    baseline = history[-1].report
    if current.pass_rate >= baseline.pass_rate:
        return GateResult(
            passed=True,
            reason=f"pass rate {current.pass_rate} >= baseline {baseline.pass_rate}",
        )
    return GateResult(
        passed=False,
        reason=f"pass rate regressed: {current.pass_rate} < baseline {baseline.pass_rate}",
    )
