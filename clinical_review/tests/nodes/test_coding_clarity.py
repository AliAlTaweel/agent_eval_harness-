from clinical_review.llm import FakeOllamaClient
from clinical_review.nodes.coding_clarity import review_coding_clarity
from clinical_review.schemas import CodingClarityFindings, Finding
from clinical_review.tracing import TraceRecorder


def test_review_coding_clarity_returns_findings_and_records_step():
    canned = CodingClarityFindings(
        issues=[Finding(element="Assessment", description="vague, no stated working diagnosis", severity="low")]
    )
    client = FakeOllamaClient(response=canned, tokens_in=60, tokens_out=20)
    recorder = TraceRecorder()

    result = review_coding_clarity(
        note_text="ASSESSMENT: probably fine.\nPLAN: reassure.",
        client=client,
        recorder=recorder,
    )

    assert result == canned
    assert len(recorder.steps) == 1
    assert recorder.steps[0].agent_name == "coding_clarity_review"
