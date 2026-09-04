import uuid
from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from trace_schema import RunTrace

from clinical_review.nodes.coding_clarity import review_coding_clarity
from clinical_review.nodes.compliance import review_compliance
from clinical_review.nodes.completeness import review_completeness
from clinical_review.nodes.merge import merge_findings
from clinical_review.schemas import (
    CodingClarityFindings,
    CompletenessFindings,
    ComplianceFindings,
    EncounterType,
    Verdict,
)
from clinical_review.tracing import TraceRecorder


class ReviewState(TypedDict, total=False):
    note_text: str
    encounter_type: EncounterType
    completeness: CompletenessFindings
    compliance: ComplianceFindings
    coding_clarity: CodingClarityFindings
    verdict: Verdict


def _build_graph(client, recorder: TraceRecorder):
    graph = StateGraph(ReviewState)

    graph.add_node(
        "completeness_review",
        lambda s: {"completeness": review_completeness(s["note_text"], s["encounter_type"], client, recorder)},
    )
    graph.add_node(
        "compliance_review",
        lambda s: {"compliance": review_compliance(s["note_text"], client, recorder)},
    )
    graph.add_node(
        "coding_clarity_review",
        lambda s: {"coding_clarity": review_coding_clarity(s["note_text"], client, recorder)},
    )
    graph.add_node(
        "merge",
        lambda s: {
            "verdict": merge_findings(s["completeness"], s["compliance"], s["coding_clarity"], client, recorder)
        },
    )

    graph.add_edge(START, "completeness_review")
    graph.add_edge(START, "compliance_review")
    graph.add_edge(START, "coding_clarity_review")
    graph.add_edge("completeness_review", "merge")
    graph.add_edge("compliance_review", "merge")
    graph.add_edge("coding_clarity_review", "merge")
    graph.add_edge("merge", END)

    return graph.compile()


class ReviewFailedError(RuntimeError):
    """Raised when a review node fails; carries whatever trace was captured before the failure."""

    def __init__(self, cause: BaseException, trace: RunTrace):
        super().__init__(str(cause))
        self.trace = trace


def _build_trace(
    note_text: str,
    encounter_type: EncounterType,
    recorder: TraceRecorder,
    started_at: datetime,
    final_output: dict | None,
) -> RunTrace:
    ended_at = datetime.now(timezone.utc)
    steps = sorted(recorder.steps, key=lambda s: s.started_at)
    total_tokens = sum(s.tokens_in + s.tokens_out for s in steps)

    return RunTrace(
        run_id=str(uuid.uuid4()),
        input={"note_text": note_text, "encounter_type": encounter_type},
        steps=steps,
        final_output=final_output,
        started_at=started_at,
        ended_at=ended_at,
        total_tokens=total_tokens,
        total_cost_usd=sum(s.cost_usd for s in steps),
    )


def run_review(note_text: str, encounter_type: EncounterType, client) -> tuple[Verdict, RunTrace]:
    recorder = TraceRecorder()
    compiled = _build_graph(client, recorder)

    started_at = datetime.now(timezone.utc)
    try:
        final_state = compiled.invoke({"note_text": note_text, "encounter_type": encounter_type})
    except Exception as exc:
        partial_trace = _build_trace(note_text, encounter_type, recorder, started_at, {"error": str(exc)})
        raise ReviewFailedError(exc, partial_trace) from exc

    verdict: Verdict = final_state["verdict"]
    trace = _build_trace(note_text, encounter_type, recorder, started_at, verdict.model_dump(mode="json"))
    return verdict, trace
