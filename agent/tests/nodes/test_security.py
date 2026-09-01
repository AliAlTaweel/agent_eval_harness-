from agent.llm import FakeOllamaClient
from agent.nodes.security import review_security
from agent.schemas import Issue, SecurityFindings
from agent.tracing import TraceRecorder


def test_review_security_returns_findings_and_records_step():
    canned = SecurityFindings(
        issues=[Issue(file="app.py", line=42, description="SQL injection via string concat", severity="critical")]
    )
    client = FakeOllamaClient(response=canned, tokens_in=100, tokens_out=30)
    recorder = TraceRecorder()

    result = review_security(diff="- old\n+ query = f'SELECT * FROM x WHERE id={id}'", client=client, recorder=recorder)

    assert result == canned
    assert len(recorder.steps) == 1
    assert recorder.steps[0].agent_name == "security_review"
