import json
from pathlib import Path

from harness.adapters.clinical_review import ClinicalReviewExpected

TESTCASES_DIR = Path(__file__).resolve().parents[2] / "clinical_testcases"


def test_every_testcase_has_input_and_valid_expected_json():
    case_dirs = [d for d in TESTCASES_DIR.iterdir() if d.is_dir()]
    assert len(case_dirs) >= 4, "expected at least 4 synthetic clinical test cases"

    for case_dir in case_dirs:
        input_path = case_dir / "input.json"
        expected_path = case_dir / "expected.json"
        assert input_path.exists(), f"missing input.json in {case_dir}"
        assert expected_path.exists(), f"missing expected.json in {case_dir}"

        input_data = json.loads(input_path.read_text())
        assert "note_text" in input_data
        assert input_data["encounter_type"] in ("new_patient", "follow_up")

        expected_data = json.loads(expected_path.read_text())
        ClinicalReviewExpected.model_validate(expected_data)
