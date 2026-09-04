from clinical_review.llm import FakeOllamaClient
from clinical_review.nodes.compliance import review_compliance
from clinical_review.schemas import ComplianceFindings, Finding
from clinical_review.tracing import TraceRecorder


def test_review_compliance_returns_findings_and_records_step():
    canned = ComplianceFindings(
        issues=[
            Finding(
                element="Assessment/Plan",
                description="plan escalates therapy but exam and vitals show well-controlled findings",
                severity="high",
            )
        ]
    )
    client = FakeOllamaClient(response=canned, tokens_in=80, tokens_out=25)
    recorder = TraceRecorder()

    result = review_compliance(
        note_text="VITALS: BP 122/76.\nASSESSMENT: poorly controlled hypertension.\nPLAN: escalate therapy.",
        client=client,
        recorder=recorder,
    )

    assert result == canned
    assert len(recorder.steps) == 1
    assert recorder.steps[0].agent_name == "compliance_review"
