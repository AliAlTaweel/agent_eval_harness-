from clinical_review.schemas import CompletenessFindings, EncounterType
from clinical_review.tracing import TraceRecorder

REQUIRED_ELEMENTS: dict[str, list[str]] = {
    "new_patient": [
        "Chief Complaint",
        "History of Present Illness",
        "Review of Systems",
        "Past Medical History",
        "Physical Exam findings",
        "Assessment",
        "Plan",
    ],
    "follow_up": [
        "Reason for visit or interval history",
        "Relevant Physical Exam findings",
        "Assessment",
        "Plan",
    ],
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are reviewing a clinical note for documentation completeness, for "
    "an encounter of type '{encounter_type}'. A complete note of this type "
    "should document: {required}. Identify which of these elements are "
    "missing or clearly inadequate in the note. Only report elements you "
    "cannot find evidence of in the note text. Respond with structured "
    "findings. This is an illustrative documentation-quality check, not a "
    "real coding/compliance authority."
)


def review_completeness(
    note_text: str, encounter_type: EncounterType, client, recorder: TraceRecorder
) -> CompletenessFindings:
    required = ", ".join(REQUIRED_ELEMENTS[encounter_type])
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(encounter_type=encounter_type, required=required)
    return recorder.record(
        "completeness_review",
        "analyze_note",
        lambda: client.chat(system=system_prompt, user=note_text, response_model=CompletenessFindings),
    )
