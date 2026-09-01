from datetime import datetime, timezone

from trace_schema.models import ToolCall, Step


def test_tool_call_round_trip():
    tc = ToolCall(name="search_diff", args={"query": "SELECT"}, result=["line 12"])
    data = tc.model_dump()
    restored = ToolCall.model_validate(data)
    assert restored == tc


def test_step_round_trip():
    now = datetime.now(timezone.utc)
    step = Step(
        agent_name="security_review",
        action="analyze_diff",
        tool_calls=[ToolCall(name="grep", args={"pattern": "eval("}, result=[])],
        output={"issues": []},
        started_at=now,
        ended_at=now,
        tokens_in=120,
        tokens_out=45,
        cost_usd=0.0,
    )
    data = step.model_dump()
    restored = Step.model_validate(data)
    assert restored == step
    assert restored.tool_calls[0].name == "grep"
