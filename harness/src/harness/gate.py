from pydantic import BaseModel

from harness.aggregator import AggregateReport
from harness.history import HistoryEntry


class GateResult(BaseModel):
    passed: bool
    reason: str


def check_gate(current: AggregateReport, history: list[HistoryEntry]) -> GateResult:
    if not history:
        return GateResult(passed=True, reason="no baseline yet; recording first result")

    best_prior_pass_rate = max(entry.report.pass_rate for entry in history)
    if current.pass_rate >= best_prior_pass_rate:
        return GateResult(
            passed=True,
            reason=f"pass rate {current.pass_rate} >= best prior pass rate {best_prior_pass_rate}",
        )
    return GateResult(
        passed=False,
        reason=f"pass rate regressed: {current.pass_rate} < best prior pass rate {best_prior_pass_rate}",
    )
