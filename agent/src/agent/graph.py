# agent/src/agent/graph.py
import uuid
from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from trace_schema import RunTrace

from agent.nodes.merge import merge_findings
from agent.nodes.security import review_security
from agent.nodes.style import review_style
from agent.nodes.test_coverage import review_test_coverage
from agent.schemas import SecurityFindings, StyleFindings, TestCoverageFindings, Verdict
from agent.tracing import TraceRecorder


class ReviewState(TypedDict, total=False):
    diff: str
    security: SecurityFindings
    style: StyleFindings
    coverage: TestCoverageFindings
    verdict: Verdict


def _build_graph(client, recorder: TraceRecorder):
    graph = StateGraph(ReviewState)

    graph.add_node("security_review", lambda s: {"security": review_security(s["diff"], client, recorder)})
    graph.add_node("style_review", lambda s: {"style": review_style(s["diff"], client, recorder)})
    graph.add_node("test_coverage", lambda s: {"coverage": review_test_coverage(s["diff"], client, recorder)})
    graph.add_node(
        "merge",
        lambda s: {
            "verdict": merge_findings(s["security"], s["style"], s["coverage"], client, recorder)
        },
    )

    graph.add_edge(START, "security_review")
    graph.add_edge(START, "style_review")
    graph.add_edge(START, "test_coverage")
    graph.add_edge("security_review", "merge")
    graph.add_edge("style_review", "merge")
    graph.add_edge("test_coverage", "merge")
    graph.add_edge("merge", END)

    return graph.compile()


def run_review(diff: str, client) -> tuple[Verdict, RunTrace]:
    recorder = TraceRecorder()
    compiled = _build_graph(client, recorder)

    started_at = datetime.now(timezone.utc)
    final_state = compiled.invoke({"diff": diff})
    ended_at = datetime.now(timezone.utc)

    verdict: Verdict = final_state["verdict"]
    steps = recorder.steps
    total_tokens = sum(s.tokens_in + s.tokens_out for s in steps)

    trace = RunTrace(
        run_id=str(uuid.uuid4()),
        input={"diff": diff},
        steps=steps,
        final_output=verdict.model_dump(),
        started_at=started_at,
        ended_at=ended_at,
        total_tokens=total_tokens,
        total_cost_usd=sum(s.cost_usd for s in steps),
    )
    return verdict, trace
