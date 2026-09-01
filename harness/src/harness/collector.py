import json
from pathlib import Path

from trace_schema import RunTrace


def load_trace(path: str | Path) -> RunTrace:
    data = json.loads(Path(path).read_text())
    return RunTrace.model_validate(data)
