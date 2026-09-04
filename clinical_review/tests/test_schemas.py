from clinical_review.schemas import (
    CodingClarityFindings,
    CompletenessFindings,
    ComplianceFindings,
    Finding,
    Verdict,
)


def test_finding_requires_severity_enum():
    finding = Finding(element="Review of Systems", description="not documented", severity="medium")
    assert finding.severity == "medium"


def test_verdict_shape():
    v = Verdict(
        completeness_issues=[Finding(element="ROS", description="missing", severity="medium")],
        compliance_issues=[],
        coding_clarity_issues=[],
        verdict="needs_revision",
    )
    assert v.verdict == "needs_revision"
    assert len(v.completeness_issues) == 1


def test_findings_models_hold_issue_lists():
    completeness = CompletenessFindings(issues=[])
    compliance = ComplianceFindings(issues=[])
    coding_clarity = CodingClarityFindings(issues=[])
    assert completeness.issues == []
    assert compliance.issues == []
    assert coding_clarity.issues == []
