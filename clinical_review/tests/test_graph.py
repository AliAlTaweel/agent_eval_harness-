import pytest
from trace_schema import RunTrace

from clinical_review.graph import ReviewFailedError, run_review
from clinical_review.llm import LlmUsage
from clinical_review.schemas import (
    CodingClarityFindings,
    CompletenessFindings,
    ComplianceFindings,
    Finding,
    Verdict,
)


class SequencedFakeClient:
    """Returns a canned response keyed by the requested response_model.

    The three review nodes run concurrently under LangGraph's executor, so
    dispatching by call order is a race. Dispatching by response_model type
    is deterministic regardless of which node's thread calls .chat() first.
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
    completeness = CompletenessFindings(
        issues=[Finding(element="Review of Systems", description="missing", severity="medium")]
    )
    compliance = ComplianceFindings(issues=[])
    coding_clarity = CodingClarityFindings(issues=[])
    merge_llm_output = Verdict(
        completeness_issues=[], compliance_issues=[], coding_clarity_issues=[], verdict="needs_revision"
    )

    client = SequencedFakeClient(
        {
            CompletenessFindings: completeness,
            ComplianceFindings: compliance,
            CodingClarityFindings: coding_clarity,
            Verdict: merge_llm_output,
        }
    )

    verdict, trace = run_review(note_text="CHIEF COMPLAINT: cough.", encounter_type="new_patient", client=client)

    assert isinstance(verdict, Verdict)
    assert verdict.verdict == "needs_revision"
    assert len(verdict.completeness_issues) == 1

    assert isinstance(trace, RunTrace)
    assert len(trace.steps) == 4
    agent_names = {s.agent_name for s in trace.steps}
    assert agent_names == {"completeness_review", "compliance_review", "coding_clarity_review", "merge"}
    assert trace.final_output["verdict"] == "needs_revision"
    assert trace.total_tokens == sum(s.tokens_in + s.tokens_out for s in trace.steps)
    assert trace.input == {"note_text": "CHIEF COMPLAINT: cough.", "encounter_type": "new_patient"}

    started_ats = [s.started_at for s in trace.steps]
    assert started_ats == sorted(started_ats)
    assert trace.steps[-1].agent_name == "merge"


def test_run_review_captures_partial_trace_on_node_failure():
    completeness = CompletenessFindings(issues=[])
    compliance = ComplianceFindings(issues=[])
    coding_clarity = CodingClarityFindings(issues=[])

    client = FailingClient(
        {
            CompletenessFindings: completeness,
            ComplianceFindings: compliance,
            CodingClarityFindings: coding_clarity,
        },
        fail_on=CodingClarityFindings,
    )

    with pytest.raises(ReviewFailedError) as exc_info:
        run_review(note_text="ASSESSMENT: fine.", encounter_type="follow_up", client=client)

    trace = exc_info.value.trace
    assert isinstance(trace, RunTrace)
    agent_names = {s.agent_name for s in trace.steps}
    assert "coding_clarity_review" not in agent_names
    assert "simulated node failure" in trace.final_output["error"]
