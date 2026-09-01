import json

from trace_schema import RunTrace

from harness.collector import load_trace


def test_load_trace_reads_and_validates_json(tmp_path):
    now = "2026-09-01T00:00:00Z"
    trace_data = {
        "schema_version": "1.0.0",
        "run_id": "run-1",
        "input": {"diff": "..."},
        "steps": [],
        "final_output": {"verdict": "approve"},
        "started_at": now,
        "ended_at": now,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
    }
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace_data))

    trace = load_trace(trace_path)

    assert isinstance(trace, RunTrace)
    assert trace.run_id == "run-1"
