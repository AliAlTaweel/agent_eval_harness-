from clinical_review.schemas import ComplianceFindings
from clinical_review.tracing import TraceRecorder

SYSTEM_PROMPT = (
    "You are reviewing a clinical note for documentation-compliance issues. "
    "Identify cases where the Assessment or Plan is not supported by any "
    "documented finding in the note, or where the exam describes abnormal "
    "findings but no vital signs are recorded. Only report issues clearly "
    "supported by the note content. Respond with structured findings. This "
    "is an illustrative documentation-quality check, not a real "
    "coding/compliance authority."
)


def review_compliance(note_text: str, client, recorder: TraceRecorder) -> ComplianceFindings:
    return recorder.record(
        "compliance_review",
        "analyze_note",
        lambda: client.chat(system=SYSTEM_PROMPT, user=note_text, response_model=ComplianceFindings),
    )
