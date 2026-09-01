from jinja2 import Template

from harness.aggregator import AggregateReport
from harness.history import HistoryEntry

TEMPLATE = Template(
    """\
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Eval Report</title></head>
<body>
  <h1>Eval Harness Report</h1>
  <h2>Current Run</h2>
  <table border="1" cellpadding="4">
    <tr><th>Pass rate</th><td>{{ current.pass_rate }}</td></tr>
    <tr><th>Avg hallucination rate</th><td>{{ current.avg_hallucination_rate }}</td></tr>
    <tr><th>Avg cost (USD)</th><td>{{ current.avg_cost_usd }}</td></tr>
    <tr><th>Avg latency (s)</th><td>{{ current.avg_latency_seconds }}</td></tr>
    <tr><th>Total cases</th><td>{{ current.total_cases }}</td></tr>
  </table>

  {% if per_case %}
  <h2>Per-Case Results</h2>
  <table border="1" cellpadding="4">
    <tr><th>Case</th><th>Result</th><th>Hallucination rate</th><th>Latency (s)</th></tr>
    {% for case in per_case %}
    <tr>
      <td>{{ case.name }}</td>
      <td>{{ "pass" if case.task_success else "fail" }}</td>
      <td>{{ case.hallucination_rate }}</td>
      <td>{{ case.latency_seconds }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  <h2>Trend</h2>
  {% if history %}
  <table border="1" cellpadding="4">
    <tr><th>Commit</th><th>Timestamp</th><th>Pass rate</th><th>Hallucination rate</th></tr>
    {% for entry in history %}
    <tr>
      <td>{{ entry.commit_sha }}</td>
      <td>{{ entry.timestamp }}</td>
      <td>{{ entry.report.pass_rate }}</td>
      <td>{{ entry.report.avg_hallucination_rate }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <p>No history yet.</p>
  {% endif %}
</body>
</html>
"""
)


def render_html_report(
    current: AggregateReport,
    history: list[HistoryEntry],
    per_case: list[dict] | None = None,
) -> str:
    return TEMPLATE.render(current=current, history=history, per_case=per_case)
