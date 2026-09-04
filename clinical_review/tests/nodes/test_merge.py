from clinical_review.llm import FakeOllamaClient
from clinical_review.nodes.merge import merge_findings
from clinical_review.schemas import (
    CodingClarityFindings,
    CompletenessFindings,
    ComplianceFindings,
    Finding,
    Verdict,
)
from clinical_review.tracing import TraceRecorder


def test_merge_findings_combines_inputs_and_uses_llm_verdict():
    completeness = CompletenessFindings(
        issues=[Finding(element="Review of Systems", description="missing", severity="medium")]
    )
    compliance = ComplianceFindings(issues=[])
    coding_clarity = CodingClarityFindings(issues=[])
    canned_verdict = Verdict(
        completeness_issues=[], compliance_issues=[], coding_clarity_issues=[], verdict="needs_revision"
    )
    client = FakeOllamaClient(response=canned_verdict, tokens_in=50, tokens_out=5)
    recorder = TraceRecorder()

    result = merge_findings(
        completeness=completeness, compliance=compliance, coding_clarity=coding_clarity, client=client, recorder=recorder
    )

    assert result.verdict == "needs_revision"
    assert result.completeness_issues == completeness.issues
    assert result.compliance_issues == compliance.issues
    assert result.coding_clarity_issues == coding_clarity.issues
    assert recorder.steps[0].agent_name == "merge"


def test_merge_findings_approves_when_no_issues():
    completeness = CompletenessFindings(issues=[])
    compliance = ComplianceFindings(issues=[])
    coding_clarity = CodingClarityFindings(issues=[])
    canned_verdict = Verdict(
        completeness_issues=[], compliance_issues=[], coding_clarity_issues=[], verdict="approve"
    )
    client = FakeOllamaClient(response=canned_verdict)
    recorder = TraceRecorder()

    result = merge_findings(
        completeness=completeness, compliance=compliance, coding_clarity=coding_clarity, client=client, recorder=recorder
    )

    assert result.verdict == "approve"
