# agent/tests/test_graph.py
from trace_schema import RunTrace

from agent.graph import run_review
from agent.llm import FakeOllamaClient
from agent.schemas import Issue, SecurityFindings, StyleFindings, TestCoverageFindings, Verdict


class SequencedFakeClient:
    """Returns a different canned response per call, in call order."""

    def __init__(self, responses: list):
        self._responses = list(responses)

    def chat(self, system, user, response_model):
        from agent.llm import LlmUsage

        response = self._responses.pop(0)
        return response, LlmUsage(tokens_in=10, tokens_out=5)


def test_run_review_produces_verdict_and_trace():
    security = SecurityFindings(issues=[Issue(file="a.py", line=1, description="sqli", severity="critical")])
    style = StyleFindings(issues=[])
    coverage = TestCoverageFindings(has_gap=True, explanation="no tests")
    merge_llm_output = Verdict(security_issues=[], style_issues=[], test_coverage_gap=False, verdict="request_changes")

    client = SequencedFakeClient([security, style, coverage, merge_llm_output])

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
