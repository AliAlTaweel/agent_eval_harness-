from agent.schemas import SecurityFindings, StyleFindings, TestCoverageFindings, Verdict
from agent.tracing import TraceRecorder

SYSTEM_PROMPT = (
    "You are the supervising critic for a pull request review. Given a "
    "summary of security, style, and test-coverage findings, decide the "
    "final verdict: 'approve' if there are no critical/high security "
    "issues and no test coverage gap, otherwise 'request_changes'. "
    "Respond with a verdict field only."
)


def _summarize(security: SecurityFindings, style: StyleFindings, coverage: TestCoverageFindings) -> str:
    lines = [
        f"Security issues: {[ (i.severity, i.description) for i in security.issues ]}",
        f"Style issues: {[ i.description for i in style.issues ]}",
        f"Test coverage gap: {coverage.has_gap} ({coverage.explanation})",
    ]
    return "\n".join(lines)


def merge_findings(
    security: SecurityFindings,
    style: StyleFindings,
    coverage: TestCoverageFindings,
    client,
    recorder: TraceRecorder,
) -> Verdict:
    summary = _summarize(security, style, coverage)

    def call():
        parsed, usage = client.chat(system=SYSTEM_PROMPT, user=summary, response_model=Verdict)
        return parsed, usage

    llm_verdict: Verdict = recorder.record("merge", "decide_verdict", call)

    return Verdict(
        security_issues=security.issues,
        style_issues=style.issues,
        test_coverage_gap=coverage.has_gap,
        verdict=llm_verdict.verdict,
    )
