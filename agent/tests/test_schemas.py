from agent.schemas import Issue, SecurityFindings, StyleFindings, TestCoverageFindings, Verdict


def test_issue_requires_severity_enum():
    issue = Issue(file="app.py", line=10, description="SQL injection", severity="critical")
    assert issue.severity == "critical"


def test_verdict_shape():
    v = Verdict(
        security_issues=[Issue(file="a.py", line=1, description="x", severity="low")],
        style_issues=[],
        test_coverage_gap=True,
        verdict="request_changes",
    )
    assert v.verdict == "request_changes"
    assert len(v.security_issues) == 1


def test_findings_models_hold_issue_lists():
    sec = SecurityFindings(issues=[])
    style = StyleFindings(issues=[])
    cov = TestCoverageFindings(has_gap=False, explanation="tests present")
    assert sec.issues == []
    assert style.issues == []
    assert cov.has_gap is False
