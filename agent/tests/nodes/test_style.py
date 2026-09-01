from agent.llm import FakeOllamaClient
from agent.nodes.style import review_style
from agent.schemas import StyleFindings
from agent.tracing import TraceRecorder


def test_review_style_returns_findings_and_records_step():
    canned = StyleFindings(issues=[])
    client = FakeOllamaClient(response=canned, tokens_in=80, tokens_out=10)
    recorder = TraceRecorder()

    result = review_style(diff="+ x=1", client=client, recorder=recorder)

    assert result == canned
    assert recorder.steps[0].agent_name == "style_review"
