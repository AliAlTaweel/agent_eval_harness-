import json
from pathlib import Path

import streamlit as st

from harness.adapters.pr_review import PrReviewExpected, PrReviewScorer
from harness.judge import HallucinationJudge

try:
    from agent.graph import ReviewFailedError, run_review
    from agent.llm import OllamaClient
    from agent.render import render_comment
except ImportError:
    st.error(
        "The 'agent' package is required to run this app — install it as part of "
        "this uv workspace (`uv sync` from the repo root)."
    )
    st.stop()

st.set_page_config(page_title="PR Review Agent + Harness", layout="wide")
st.title("PR Review Agent + Harness")

model = st.sidebar.text_input("Ollama model", value="qwen2.5:7b")
mode = st.sidebar.radio(
    "Diff source",
    ["Upload your own diff", "Score against known testcase"],
)

TESTCASES_DIR = Path(__file__).resolve().parents[3] / "testcases"

diff_text: str | None = None
expected: PrReviewExpected | None = None

if mode == "Upload your own diff":
    uploaded = st.file_uploader("Upload a diff/patch file", type=["diff", "patch", "txt"])
    if uploaded is not None:
        diff_text = uploaded.read().decode("utf-8")
        st.caption(f"{uploaded.name} — {len(diff_text.splitlines())} lines")
else:
    case_names = sorted(p.name for p in TESTCASES_DIR.iterdir() if p.is_dir()) if TESTCASES_DIR.exists() else []
    if not case_names:
        st.error(f"No testcases found under {TESTCASES_DIR}")
    else:
        case_name = st.selectbox("Testcase", case_names)
        case_dir = TESTCASES_DIR / case_name
        diff_text = (case_dir / "diff.patch").read_text()
        expected = PrReviewExpected.model_validate(json.loads((case_dir / "expected.json").read_text()))
        st.caption(f"{case_name} — {len(diff_text.splitlines())} lines")
        with st.expander("Expected result"):
            st.json(expected.model_dump())

if diff_text is not None and st.button("Run Review", type="primary"):
    client = OllamaClient(model=model)
    with st.spinner("Running multi-agent review..."):
        try:
            verdict, trace = run_review(diff=diff_text, client=client)
        except ReviewFailedError as exc:
            st.error(f"Review failed: {exc}")
            st.json(exc.trace.model_dump(mode="json"))
            st.stop()

    st.markdown(render_comment(verdict))

    with st.expander("Verdict JSON"):
        st.json(verdict.model_dump(mode="json"))

    with st.expander("Trace stats"):
        st.metric("Total tokens", trace.total_tokens)
        st.metric("Total cost (USD)", f"{trace.total_cost_usd:.4f}")
        latency = (trace.ended_at - trace.started_at).total_seconds()
        st.metric("Latency (s)", f"{latency:.2f}")

    with st.expander("Full trace JSON"):
        st.json(trace.model_dump(mode="json"))

    st.divider()
    st.subheader("Harness score")

    judge = HallucinationJudge(model=model)

    if expected is not None:
        scorer = PrReviewScorer(judge=judge)
        with st.spinner("Scoring against expected result..."):
            result = scorer.score(trace, expected)

        cols = st.columns(4)
        cols[0].metric("Task success", "✅ Pass" if result.task_success else "❌ Fail")
        cols[1].metric("Tool call correctness", f"{result.tool_call_correctness:.2f}")
        cols[2].metric("Hallucination rate", f"{result.hallucination_rate:.2f}")
        cols[3].metric("Latency (s)", f"{result.latency_seconds:.2f}")

        with st.expander("Score details"):
            st.json(result.model_dump())
    else:
        st.caption(
            "No ground truth for an uploaded diff, so this only runs the hallucination "
            "judge against the flagged security issues — task success and tool-call "
            "correctness require a known expected verdict (use 'Score against known "
            "testcase' for the full harness score)."
        )
        security_issues = (trace.final_output or {}).get("security_issues", [])
        if security_issues:
            hallucinated = 0
            with st.spinner("Running hallucination judge..."):
                for issue in security_issues:
                    verdict_j = judge.judge(source_material=diff_text, claim=issue.get("description", ""))
                    hallucinated += verdict_j.hallucinated
            hallucination_rate = hallucinated / len(security_issues)
            st.metric("Hallucination rate (security issues)", f"{hallucination_rate:.2f}")
        else:
            st.caption("No security issues flagged — nothing to check for hallucination.")
