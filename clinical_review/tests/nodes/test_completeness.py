from clinical_review.llm import FakeOllamaClient
from clinical_review.nodes.completeness import review_completeness
from clinical_review.schemas import CompletenessFindings, Finding
from clinical_review.tracing import TraceRecorder


def test_review_completeness_returns_findings_and_records_step():
    canned = CompletenessFindings(
        issues=[Finding(element="Review of Systems", description="not documented", severity="medium")]
    )
    client = FakeOllamaClient(response=canned, tokens_in=100, tokens_out=30)
    recorder = TraceRecorder()

    result = review_completeness(
        note_text="CHIEF COMPLAINT: cough.\nASSESSMENT: viral illness.\nPLAN: rest.",
        encounter_type="new_patient",
        client=client,
        recorder=recorder,
    )

    assert result == canned
    assert len(recorder.steps) == 1
    assert recorder.steps[0].agent_name == "completeness_review"


def test_review_completeness_uses_follow_up_required_elements():
    canned = CompletenessFindings(issues=[])
    client = FakeOllamaClient(response=canned)
    recorder = TraceRecorder()

    review_completeness(
        note_text="REASON FOR VISIT: follow-up.\nASSESSMENT: stable.\nPLAN: continue current regimen.",
        encounter_type="follow_up",
        client=client,
        recorder=recorder,
    )

    assert len(recorder.steps) == 1
