from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel
from trace_schema import Step, ToolCall

from clinical_review.llm import LlmUsage


class TraceRecorder:
    def __init__(self):
        self._steps: list[Step] = []

    @property
    def steps(self) -> list[Step]:
        return self._steps

    def record(
        self,
        agent_name: str,
        action: str,
        fn: Callable[[], tuple[Any, LlmUsage]],
        tool_calls: list[ToolCall] | None = None,
    ) -> Any:
        started_at = datetime.now(timezone.utc)
        output, usage = fn()
        ended_at = datetime.now(timezone.utc)
        if isinstance(output, BaseModel):
            recorded_output = output.model_dump(mode="json")
        elif isinstance(output, (dict, list, str, int, float, bool, type(None))):
            recorded_output = output
        else:
            recorded_output = str(output)
        self._steps.append(
            Step(
                agent_name=agent_name,
                action=action,
                tool_calls=tool_calls or [],
                output=recorded_output,
                started_at=started_at,
                ended_at=ended_at,
                tokens_in=usage.tokens_in,
                tokens_out=usage.tokens_out,
                cost_usd=0.0,
            )
        )
        return output
