from jinja2 import Template

from agent.schemas import Verdict

TEMPLATE = Template(
    """\
## PR Review: {{ "✅ Approve" if verdict.verdict == "approve" else "🚫 Request Changes" }}

{% if verdict.security_issues %}
### Security Issues
{% for issue in verdict.security_issues %}
- **[{{ issue.severity }}]** `{{ issue.file }}{% if issue.line %}:{{ issue.line }}{% endif %}` — {{ issue.description }}
{% endfor %}
{% else %}
### Security Issues
None found.
{% endif %}

{% if verdict.style_issues %}
### Style Issues
{% for issue in verdict.style_issues %}
- **[{{ issue.severity }}]** `{{ issue.file }}{% if issue.line %}:{{ issue.line }}{% endif %}` — {{ issue.description }}
{% endfor %}
{% else %}
### Style Issues
None found.
{% endif %}

### Test Coverage
{{ "⚠️ Gap detected — new/changed logic is missing tests." if verdict.test_coverage_gap else "✅ No gap detected." }}
"""
)


def render_comment(verdict: Verdict) -> str:
    return TEMPLATE.render(verdict=verdict)
