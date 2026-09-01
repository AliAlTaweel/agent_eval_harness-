import json
import subprocess
from pathlib import Path

from trace_schema.export_schema import run_trace_json_schema

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_entry_point_prints_valid_schema_json():
    result = subprocess.run(
        ["uv", "run", "python", "-m", "trace_schema.export_schema"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    schema = json.loads(result.stdout)
    assert schema["title"] == "RunTrace"


def test_run_trace_json_schema_has_expected_top_level_keys():
    schema = run_trace_json_schema()
    assert schema["title"] == "RunTrace"
    assert "run_id" in schema["properties"]
    assert "steps" in schema["properties"]


def test_top_level_package_reexports_models():
    from trace_schema import RunTrace, Step, ToolCall

    assert RunTrace.__name__ == "RunTrace"
    assert Step.__name__ == "Step"
    assert ToolCall.__name__ == "ToolCall"
