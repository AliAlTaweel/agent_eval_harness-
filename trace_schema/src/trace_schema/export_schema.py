import json
from typing import Any

from trace_schema.models import RunTrace


def run_trace_json_schema() -> dict[str, Any]:
    return RunTrace.model_json_schema()


if __name__ == "__main__":
    print(json.dumps(run_trace_json_schema(), indent=2))
