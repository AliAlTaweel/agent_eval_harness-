import pytest

from clinical_review.graph import run_review
from clinical_review.llm import OllamaClient
from clinical_review.schemas import Verdict


@pytest.mark.integration
def test_run_review_against_real_ollama_flags_incomplete_note():
    note_text = """\
CHIEF COMPLAINT: Sore throat for 3 days.

ASSESSMENT: Likely viral pharyngitis.
"""
    client = OllamaClient(model="qwen2.5:7b")

    verdict, trace = run_review(note_text=note_text, encounter_type="new_patient", client=client)

    assert isinstance(verdict, Verdict)
    assert len(trace.steps) == 4
