from agent.llm import FakeOllamaClient
from agent.nodes.test_coverage import review_test_coverage
from agent.schemas import TestCoverageFindings
from agent.tracing import TraceRecorder


def test_review_test_coverage_returns_findings_and_records_step():
    canned = TestCoverageFindings(has_gap=True, explanation="no new tests for changed function")
    client = FakeOllamaClient(response=canned, tokens_in=90, tokens_out=20)
    recorder = TraceRecorder()

    result = review_test_coverage(diff="+ def foo(): ...", client=client, recorder=recorder)

    assert result == canned
    assert recorder.steps[0].agent_name == "test_coverage"
