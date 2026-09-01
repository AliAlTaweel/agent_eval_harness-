from agent.llm import FakeOllamaClient
from agent.nodes.merge import merge_findings
from agent.schemas import Issue, SecurityFindings, StyleFindings, TestCoverageFindings, Verdict
from agent.tracing import TraceRecorder


def test_merge_findings_combines_inputs_and_uses_llm_verdict():
    security = SecurityFindings(issues=[Issue(file="a.py", line=1, description="sqli", severity="critical")])
    style = StyleFindings(issues=[])
    coverage = TestCoverageFindings(has_gap=False, explanation="covered")
    canned_verdict = Verdict(
        security_issues=[], style_issues=[], test_coverage_gap=False, verdict="request_changes"
    )
    client = FakeOllamaClient(response=canned_verdict, tokens_in=50, tokens_out=5)
    recorder = TraceRecorder()

    result = merge_findings(security=security, style=style, coverage=coverage, client=client, recorder=recorder)

    assert result.verdict == "request_changes"
    assert result.security_issues == security.issues
    assert result.style_issues == style.issues
    assert result.test_coverage_gap is False
    assert recorder.steps[0].agent_name == "merge"


def test_merge_findings_approves_when_no_issues():
    security = SecurityFindings(issues=[])
    style = StyleFindings(issues=[])
    coverage = TestCoverageFindings(has_gap=False, explanation="covered")
    canned_verdict = Verdict(security_issues=[], style_issues=[], test_coverage_gap=False, verdict="approve")
    client = FakeOllamaClient(response=canned_verdict)
    recorder = TraceRecorder()

    result = merge_findings(security=security, style=style, coverage=coverage, client=client, recorder=recorder)

    assert result.verdict == "approve"
