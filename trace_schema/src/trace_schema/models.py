from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    result: Any = None


class Step(BaseModel):
    agent_name: str
    action: str
    tool_calls: list[ToolCall] = []
    output: Any = None
    started_at: datetime
    ended_at: datetime
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
