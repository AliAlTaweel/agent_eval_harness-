# agent/tests/test_graph.py
import pytest
from trace_schema import RunTrace

from agent.graph import ReviewFailedError, run_review
from agent.llm import LlmUsage
from agent.schemas import Issue, SecurityFindings, StyleFindings, TestCoverageFindings, Verdict


class SequencedFakeClient:
    """Returns a canned response keyed by the requested response_model.

    The three review nodes (security, style, coverage) run concurrently under
    LangGraph's executor, so dispatching by call order is a race. Dispatching
    by response_model type is deterministic regardless of which node's thread
    calls .chat() first.
    """

    def __init__(self, responses: dict):
        self._responses = dict(responses)

    def chat(self, system, user, response_model):
        response = self._responses[response_model]
        return response, LlmUsage(tokens_in=10, tokens_out=5)


class FailingClient:
    """Raises for one response_model, returns canned responses for the rest."""

    def __init__(self, responses: dict, fail_on: type):
        self._responses = dict(responses)
        self._fail_on = fail_on

    def chat(self, system, user, response_model):
        if response_model is self._fail_on:
            raise RuntimeError("simulated node failure")
        response = self._responses[response_model]
        return response, LlmUsage(tokens_in=10, tokens_out=5)


def test_run_review_produces_verdict_and_trace():
    security = SecurityFindings(issues=[Issue(file="a.py", line=1, description="sqli", severity="critical")])
    style = StyleFindings(issues=[])
    coverage = TestCoverageFindings(has_gap=True, explanation="no tests")
    merge_llm_output = Verdict(security_issues=[], style_issues=[], test_coverage_gap=False, verdict="request_changes")

    client = SequencedFakeClient(
        {
            SecurityFindings: security,
            StyleFindings: style,
            TestCoverageFindings: coverage,
            Verdict: merge_llm_output,
        }
    )

    verdict, trace = run_review(diff="+ query = f'SELECT {x}'", client=client)

    assert isinstance(verdict, Verdict)
    assert verdict.verdict == "request_changes"
    assert verdict.test_coverage_gap is True
    assert len(verdict.security_issues) == 1

    assert isinstance(trace, RunTrace)
    assert len(trace.steps) == 4
    agent_names = {s.agent_name for s in trace.steps}
    assert agent_names == {"security_review", "style_review", "test_coverage", "merge"}
    assert trace.final_output["verdict"] == "request_changes"
    assert trace.total_tokens == sum(s.tokens_in + s.tokens_out for s in trace.steps)

    # steps are ordered deterministically by start time, regardless of which
    # node's thread happened to finish first
    started_ats = [s.started_at for s in trace.steps]
    assert started_ats == sorted(started_ats)
    assert trace.steps[-1].agent_name == "merge"


def test_run_review_captures_partial_trace_on_node_failure():
    security = SecurityFindings(issues=[])
    style = StyleFindings(issues=[])
    coverage = TestCoverageFindings(has_gap=False, explanation="fine")

    client = FailingClient(
        {
            SecurityFindings: security,
            StyleFindings: style,
            TestCoverageFindings: coverage,
        },
        fail_on=TestCoverageFindings,
    )

    with pytest.raises(ReviewFailedError) as exc_info:
        run_review(diff="+ x = 1", client=client)

    trace = exc_info.value.trace
    assert isinstance(trace, RunTrace)
    # security_review and style_review completed before the failure; merge
    # never ran because test_coverage raised.
    agent_names = {s.agent_name for s in trace.steps}
    assert agent_names == {"security_review", "style_review"}
    assert "simulated node failure" in trace.final_output["error"]
