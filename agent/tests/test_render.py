from agent.render import render_comment
from agent.schemas import Issue, Verdict


def test_render_comment_includes_verdict_and_issues():
    verdict = Verdict(
        security_issues=[Issue(file="app.py", line=12, description="SQL injection", severity="critical")],
        style_issues=[],
        test_coverage_gap=True,
        verdict="request_changes",
    )
    comment = render_comment(verdict)

    assert "request_changes".upper() in comment.upper() or "Request Changes" in comment
    assert "app.py" in comment
    assert "SQL injection" in comment
    assert "test coverage" in comment.lower()


def test_render_comment_clean_approve():
    verdict = Verdict(security_issues=[], style_issues=[], test_coverage_gap=False, verdict="approve")
    comment = render_comment(verdict)
    assert "approve" in comment.lower()
