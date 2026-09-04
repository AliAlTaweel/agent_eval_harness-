# clinical_review/src/clinical_review/streamlit_app.py
import streamlit as st

from clinical_review.graph import ReviewFailedError, run_review
from clinical_review.llm import OllamaClient
from clinical_review.render import render_comment

st.set_page_config(page_title="Clinical Note Review Agent", layout="wide")
st.title("Clinical Note Review Agent")
st.caption("Illustrative demo agent. Not a clinical compliance or coding authority.")

model = st.sidebar.text_input("Ollama model", value="qwen2.5:7b")

encounter_type = st.selectbox("Encounter type", ["new_patient", "follow_up"])
note_text = st.text_area("Clinical note text", height=300)

if note_text.strip() and st.button("Run Review", type="primary"):
    client = OllamaClient(model=model)
    with st.spinner("Running multi-agent review..."):
        try:
            verdict, trace = run_review(note_text=note_text, encounter_type=encounter_type, client=client)
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
