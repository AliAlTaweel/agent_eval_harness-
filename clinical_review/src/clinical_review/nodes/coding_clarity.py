from clinical_review.schemas import CodingClarityFindings
from clinical_review.tracing import TraceRecorder

SYSTEM_PROMPT = (
    "You are reviewing a clinical note for coding clarity. Identify cases "
    "where the Assessment uses vague, non-specific language that would not "
    "map to a clear, billable diagnosis (for example, phrases like "
    "'probably fine' or 'likely nothing serious' without a stated working "
    "diagnosis). Only report issues clearly supported by the note content. "
    "Respond with structured findings. This is an illustrative "
    "documentation-quality check, not a real coding/compliance authority."
)


def review_coding_clarity(note_text: str, client, recorder: TraceRecorder) -> CodingClarityFindings:
    return recorder.record(
        "coding_clarity_review",
        "analyze_note",
        lambda: client.chat(system=SYSTEM_PROMPT, user=note_text, response_model=CodingClarityFindings),
    )
