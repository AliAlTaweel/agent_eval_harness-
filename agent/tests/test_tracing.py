from trace_schema import ToolCall

from agent.llm import LlmUsage
from agent.schemas import Issue, SecurityFindings
from agent.tracing import TraceRecorder


def test_record_appends_step_with_timing_and_usage():
    recorder = TraceRecorder()

    def fake_call():
        return {"ok": True}, LlmUsage(tokens_in=5, tokens_out=2)

    output = recorder.record("security_review", "analyze_diff", fake_call)

    assert output == {"ok": True}
    assert len(recorder.steps) == 1
    step = recorder.steps[0]
    assert step.agent_name == "security_review"
    assert step.action == "analyze_diff"
    assert step.tokens_in == 5
    assert step.tokens_out == 2
    assert step.output == {"ok": True}
    assert step.cost_usd == 0.0
    assert step.started_at <= step.ended_at


def test_record_stores_tool_calls():
    recorder = TraceRecorder()

    def fake_call():
        return "result", LlmUsage(tokens_in=1, tokens_out=1)

    tool_calls = [ToolCall(name="grep", args={"pattern": "eval("}, result=[])]
    recorder.record("security_review", "search", fake_call, tool_calls=tool_calls)

    assert recorder.steps[0].tool_calls == tool_calls


def test_record_serializes_pydantic_model_output_as_dict():
    recorder = TraceRecorder()

    findings = SecurityFindings(
        issues=[Issue(file="a.py", line=1, description="sqli", severity="critical")]
    )

    def fake_call():
        return findings, LlmUsage(tokens_in=3, tokens_out=4)

    output = recorder.record("security_review", "analyze_diff", fake_call)

    assert output is findings
    step = recorder.steps[0]
    assert isinstance(step.output, dict)
    assert step.output == findings.model_dump(mode="json")
