from clinical_review.render import render_comment
from clinical_review.schemas import Finding, Verdict


def test_render_comment_includes_verdict_and_issues():
    verdict = Verdict(
        completeness_issues=[Finding(element="Review of Systems", description="not documented", severity="medium")],
        compliance_issues=[],
        coding_clarity_issues=[],
        verdict="needs_revision",
    )
    comment = render_comment(verdict)

    assert "Needs Revision" in comment
    assert "Review of Systems" in comment
    assert "not documented" in comment
    assert "illustrative demo agent" in comment.lower()


def test_render_comment_clean_approve():
    verdict = Verdict(
        completeness_issues=[], compliance_issues=[], coding_clarity_issues=[], verdict="approve"
    )
    comment = render_comment(verdict)
    assert "Approve" in comment
