from trace_schema.export_schema import run_trace_json_schema


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
