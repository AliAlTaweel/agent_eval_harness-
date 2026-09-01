from agent.schemas import StyleFindings
from agent.tracing import TraceRecorder

SYSTEM_PROMPT = (
    "You are a code style reviewer for a pull request diff. Identify style "
    "issues: naming, formatting, dead code, overly long functions. Only "
    "report issues clearly supported by the diff content."
)


def review_style(diff: str, client, recorder: TraceRecorder) -> StyleFindings:
    return recorder.record(
        "style_review",
        "analyze_diff",
        lambda: client.chat(system=SYSTEM_PROMPT, user=diff, response_model=StyleFindings),
    )
