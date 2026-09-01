import json
from pathlib import Path

from harness.adapters.pr_review import PrReviewExpected

TESTCASES_DIR = Path(__file__).resolve().parents[2] / "testcases"


def test_every_testcase_has_diff_and_valid_expected_json():
    case_dirs = [d for d in TESTCASES_DIR.iterdir() if d.is_dir()]
    assert len(case_dirs) >= 6, "expected at least 6 synthetic test cases"

    for case_dir in case_dirs:
        diff_path = case_dir / "diff.patch"
        expected_path = case_dir / "expected.json"
        assert diff_path.exists(), f"missing diff.patch in {case_dir}"
        assert expected_path.exists(), f"missing expected.json in {case_dir}"

        data = json.loads(expected_path.read_text())
        PrReviewExpected.model_validate(data)
