from typing import Any, Protocol

from pydantic import BaseModel

from trace_schema import RunTrace


class ScoreResult(BaseModel):
    task_success: bool
    tool_call_correctness: float
    hallucination_rate: float
    latency_seconds: float
    cost_usd: float
    details: dict[str, Any] = {}


class Scorer(Protocol):
    def score(self, trace: RunTrace, expected: Any) -> ScoreResult: ...


def latency_seconds(trace: RunTrace) -> float:
    return (trace.ended_at - trace.started_at).total_seconds()
