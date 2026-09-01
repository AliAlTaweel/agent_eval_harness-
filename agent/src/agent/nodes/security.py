from agent.schemas import SecurityFindings
from agent.tracing import TraceRecorder

SYSTEM_PROMPT = (
    "You are a security reviewer for a pull request diff. Identify security "
    "vulnerabilities such as SQL injection, XSS, hardcoded secrets, unsafe "
    "deserialization, and command injection. Only report issues clearly "
    "supported by the diff content. Respond with structured findings."
)


def review_security(diff: str, client, recorder: TraceRecorder) -> SecurityFindings:
    return recorder.record(
        "security_review",
        "analyze_diff",
        lambda: client.chat(system=SYSTEM_PROMPT, user=diff, response_model=SecurityFindings),
    )
