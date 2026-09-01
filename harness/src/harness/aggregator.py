from pydantic import BaseModel

from harness.scorer import ScoreResult


class AggregateReport(BaseModel):
    pass_rate: float
    avg_hallucination_rate: float
    avg_cost_usd: float
    avg_latency_seconds: float
    total_cases: int


def aggregate(results: list[ScoreResult]) -> AggregateReport:
    if not results:
        return AggregateReport(
            pass_rate=0.0, avg_hallucination_rate=0.0, avg_cost_usd=0.0, avg_latency_seconds=0.0, total_cases=0
        )
    n = len(results)
    return AggregateReport(
        pass_rate=sum(1 for r in results if r.task_success) / n,
        avg_hallucination_rate=sum(r.hallucination_rate for r in results) / n,
        avg_cost_usd=sum(r.cost_usd for r in results) / n,
        avg_latency_seconds=sum(r.latency_seconds for r in results) / n,
        total_cases=n,
    )
