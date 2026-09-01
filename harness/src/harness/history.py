import json
from pathlib import Path

from pydantic import BaseModel

from harness.aggregator import AggregateReport


class HistoryEntry(BaseModel):
    commit_sha: str
    timestamp: str
    report: AggregateReport


def load_history(path: str | Path) -> list[HistoryEntry]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return [HistoryEntry.model_validate(item) for item in data]


def append_history(path: str | Path, entry: HistoryEntry) -> None:
    entries = load_history(path)
    entries.append(entry)
    Path(path).write_text(json.dumps([e.model_dump() for e in entries], indent=2))
