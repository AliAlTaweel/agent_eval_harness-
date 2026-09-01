from agent.schemas import TestCoverageFindings
from agent.tracing import TraceRecorder

SYSTEM_PROMPT = (
    "You review a pull request diff for test coverage. Determine whether "
    "new or changed logic lacks corresponding test changes in the same "
    "diff. Respond with has_gap and a short explanation."
)


def review_test_coverage(diff: str, client, recorder: TraceRecorder) -> TestCoverageFindings:
    return recorder.record(
        "test_coverage",
        "analyze_diff",
        lambda: client.chat(system=SYSTEM_PROMPT, user=diff, response_model=TestCoverageFindings),
    )
