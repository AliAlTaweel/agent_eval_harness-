from typing import Any

from pydantic import AwareDatetime, BaseModel, Field

SCHEMA_VERSION = "1.0.0"


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    result: Any = None


class Step(BaseModel):
    agent_name: str
    action: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    output: Any = None
    started_at: AwareDatetime
    ended_at: AwareDatetime
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class RunTrace(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    input: Any
    steps: list[Step] = Field(default_factory=list)
    final_output: Any = None
    started_at: AwareDatetime
    ended_at: AwareDatetime
    total_tokens: int = 0
    total_cost_usd: float = 0.0
