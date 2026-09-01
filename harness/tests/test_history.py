from harness.aggregator import AggregateReport
from harness.history import HistoryEntry, append_history, load_history


def test_load_history_missing_file_returns_empty_list(tmp_path):
    path = tmp_path / "history.json"
    assert load_history(path) == []


def test_append_and_load_history_round_trip(tmp_path):
    path = tmp_path / "history.json"
    report = AggregateReport(pass_rate=0.8, avg_hallucination_rate=0.1, avg_cost_usd=0.0, avg_latency_seconds=3.0, total_cases=5)
    entry = HistoryEntry(commit_sha="abc123", timestamp="2026-09-01T00:00:00Z", report=report)

    append_history(path, entry)
    loaded = load_history(path)

    assert len(loaded) == 1
    assert loaded[0].commit_sha == "abc123"
    assert loaded[0].report.pass_rate == 0.8

    entry2 = HistoryEntry(commit_sha="def456", timestamp="2026-09-02T00:00:00Z", report=report)
    append_history(path, entry2)
    loaded2 = load_history(path)
    assert len(loaded2) == 2
