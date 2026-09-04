from clinical_review.schemas import (
    CodingClarityFindings,
    CompletenessFindings,
    ComplianceFindings,
    Verdict,
)
from clinical_review.tracing import TraceRecorder

SYSTEM_PROMPT = (
    "You are the supervising reviewer for a clinical note's documentation "
    "quality. Given a summary of completeness, compliance, and "
    "coding-clarity findings, decide the final verdict: 'approve' if there "
    "are no compliance issues and no missing required elements, otherwise "
    "'needs_revision'. Respond with a verdict field only."
)


def _summarize(
    completeness: CompletenessFindings, compliance: ComplianceFindings, coding_clarity: CodingClarityFindings
) -> str:
    lines = [
        f"Completeness issues: {[i.description for i in completeness.issues]}",
        f"Compliance issues: {[i.description for i in compliance.issues]}",
        f"Coding clarity issues: {[i.description for i in coding_clarity.issues]}",
    ]
    return "\n".join(lines)


def merge_findings(
    completeness: CompletenessFindings,
    compliance: ComplianceFindings,
    coding_clarity: CodingClarityFindings,
    client,
    recorder: TraceRecorder,
) -> Verdict:
    summary = _summarize(completeness, compliance, coding_clarity)

    def call():
        parsed, usage = client.chat(system=SYSTEM_PROMPT, user=summary, response_model=Verdict)
        return parsed, usage

    llm_verdict: Verdict = recorder.record("merge", "decide_verdict", call)

    return Verdict(
        completeness_issues=completeness.issues,
        compliance_issues=compliance.issues,
        coding_clarity_issues=coding_clarity.issues,
        verdict=llm_verdict.verdict,
    )
