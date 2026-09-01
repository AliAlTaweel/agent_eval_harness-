# Trace Schema (Phase 0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `trace_schema` package — the shared pydantic contract for agent run traces that both `agent/` and `harness/` depend on, and nothing else.

**Architecture:** A single `uv` workspace root plus one member package, `trace_schema`, exposing pydantic models (`ToolCall`, `Step`, `RunTrace`) and a script to export them as JSON Schema. No dependency on LangGraph, Ollama, or anything agent/harness-specific.

**Tech Stack:** Python 3.12, `uv`, `pydantic` v2, `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-01-pr-review-eval-design.md`

## Global Constraints

- Python 3.12 only.
- `uv` workspace layout; each package has its own `pyproject.toml` and `src/<package>/` layout.
- `trace_schema` must have zero dependency on `agent` or `harness` — it is a leaf package.
- `schema_version` field starts at `"1.0.0"`.
- Timestamps are timezone-aware UTC (`datetime` with `tzinfo=timezone.utc`).

---

### Task 1: Workspace root + `trace_schema` package scaffold

**Files:**
- Create: `pyproject.toml` (workspace root)
- Create: `trace_schema/pyproject.toml`
- Create: `trace_schema/src/trace_schema/__init__.py`
- Create: `trace_schema/tests/__init__.py`
- Create: `.gitignore`
- Create: `.python-version`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: an installable `trace_schema` package importable as `import trace_schema`, registered as a `uv` workspace member.

- [ ] **Step 1: Create `.python-version`**

```
3.12
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
*.egg-info/
dist/
build/
.env
```

- [ ] **Step 3: Create the workspace root `pyproject.toml`**

```toml
[project]
name = "code-review-workspace"
version = "0.0.0"
requires-python = ">=3.12"
dependencies = []

[tool.uv.workspace]
members = ["trace_schema", "agent", "harness"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 4: Create `trace_schema/pyproject.toml`**

```toml
[project]
name = "trace-schema"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.6",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/trace_schema"]
```

- [ ] **Step 5: Create empty package files**

`trace_schema/src/trace_schema/__init__.py`:
```python
```

`trace_schema/tests/__init__.py`:
```python
```

- [ ] **Step 6: Sync the workspace and verify the package installs**

Run: `cd /Users/alial-taweel/projects/code_review && uv sync`
Expected: completes without error, creates `.venv/` at the workspace root.

Run: `uv run python -c "import trace_schema; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore .python-version trace_schema/
git commit -m "chore: scaffold uv workspace and trace_schema package"
```

---

### Task 2: `ToolCall` and `Step` models

**Files:**
- Create: `trace_schema/src/trace_schema/models.py`
- Test: `trace_schema/tests/test_models.py`

**Interfaces:**
- Consumes: `pydantic.BaseModel`.
- Produces:
  - `class ToolCall(BaseModel)`: fields `name: str`, `args: dict[str, Any]`, `result: Any`.
  - `class Step(BaseModel)`: fields `agent_name: str`, `action: str`, `tool_calls: list[ToolCall]`, `output: Any`, `started_at: datetime`, `ended_at: datetime`, `tokens_in: int`, `tokens_out: int`, `cost_usd: float`.
  - Both importable as `from trace_schema.models import ToolCall, Step`.

- [ ] **Step 1: Write the failing test**

```python
# trace_schema/tests/test_models.py
from datetime import datetime, timezone

from trace_schema.models import ToolCall, Step


def test_tool_call_round_trip():
    tc = ToolCall(name="search_diff", args={"query": "SELECT"}, result=["line 12"])
    data = tc.model_dump()
    restored = ToolCall.model_validate(data)
    assert restored == tc


def test_step_round_trip():
    now = datetime.now(timezone.utc)
    step = Step(
        agent_name="security_review",
        action="analyze_diff",
        tool_calls=[ToolCall(name="grep", args={"pattern": "eval("}, result=[])],
        output={"issues": []},
        started_at=now,
        ended_at=now,
        tokens_in=120,
        tokens_out=45,
        cost_usd=0.0,
    )
    data = step.model_dump()
    restored = Step.model_validate(data)
    assert restored == step
    assert restored.tool_calls[0].name == "grep"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest trace_schema/tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trace_schema.models'`

- [ ] **Step 3: Write the implementation**

```python
# trace_schema/src/trace_schema/models.py
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    result: Any = None


class Step(BaseModel):
    agent_name: str
    action: str
    tool_calls: list[ToolCall] = []
    output: Any = None
    started_at: datetime
    ended_at: datetime
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest trace_schema/tests/test_models.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add trace_schema/src/trace_schema/models.py trace_schema/tests/test_models.py
git commit -m "feat(trace_schema): add ToolCall and Step models"
```

---

### Task 3: `RunTrace` model

**Files:**
- Modify: `trace_schema/src/trace_schema/models.py`
- Test: `trace_schema/tests/test_models.py`

**Interfaces:**
- Consumes: `Step` from Task 2.
- Produces: `class RunTrace(BaseModel)` with fields `schema_version: str = "1.0.0"`, `run_id: str`, `input: Any`, `steps: list[Step]`, `final_output: Any`, `started_at: datetime`, `ended_at: datetime`, `total_tokens: int`, `total_cost_usd: float`. Importable as `from trace_schema.models import RunTrace`.

- [ ] **Step 1: Write the failing test**

```python
# append to trace_schema/tests/test_models.py
from trace_schema.models import RunTrace


def test_run_trace_round_trip():
    now = datetime.now(timezone.utc)
    step = Step(
        agent_name="security_review",
        action="analyze_diff",
        started_at=now,
        ended_at=now,
    )
    trace = RunTrace(
        run_id="run-123",
        input={"diff": "..."},
        steps=[step],
        final_output={"verdict": "approve"},
        started_at=now,
        ended_at=now,
        total_tokens=200,
        total_cost_usd=0.0,
    )
    data = trace.model_dump()
    restored = RunTrace.model_validate(data)
    assert restored == trace
    assert restored.schema_version == "1.0.0"


def test_run_trace_defaults_schema_version():
    now = datetime.now(timezone.utc)
    trace = RunTrace(
        run_id="run-1",
        input=None,
        steps=[],
        final_output=None,
        started_at=now,
        ended_at=now,
        total_tokens=0,
        total_cost_usd=0.0,
    )
    assert trace.schema_version == "1.0.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest trace_schema/tests/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'RunTrace'`

- [ ] **Step 3: Write the implementation**

Append to `trace_schema/src/trace_schema/models.py`:

```python
class RunTrace(BaseModel):
    schema_version: str = "1.0.0"
    run_id: str
    input: Any
    steps: list[Step] = []
    final_output: Any = None
    started_at: datetime
    ended_at: datetime
    total_tokens: int = 0
    total_cost_usd: float = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest trace_schema/tests/test_models.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add trace_schema/src/trace_schema/models.py trace_schema/tests/test_models.py
git commit -m "feat(trace_schema): add RunTrace model"
```

---

### Task 4: Public exports + JSON Schema export script

**Files:**
- Modify: `trace_schema/src/trace_schema/__init__.py`
- Create: `trace_schema/src/trace_schema/export_schema.py`
- Test: `trace_schema/tests/test_export_schema.py`

**Interfaces:**
- Consumes: `RunTrace` from Task 3.
- Produces:
  - `from trace_schema import RunTrace, Step, ToolCall` (top-level re-export).
  - `trace_schema.export_schema.run_trace_json_schema() -> dict` returning `RunTrace.model_json_schema()`.
  - CLI entry point `uv run python -m trace_schema.export_schema > schema.json` prints the schema as JSON to stdout.

- [ ] **Step 1: Write the failing test**

```python
# trace_schema/tests/test_export_schema.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest trace_schema/tests/test_export_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trace_schema.export_schema'`

- [ ] **Step 3: Write the implementation**

```python
# trace_schema/src/trace_schema/__init__.py
from trace_schema.models import RunTrace, Step, ToolCall

__all__ = ["RunTrace", "Step", "ToolCall"]
```

```python
# trace_schema/src/trace_schema/export_schema.py
import json

from trace_schema.models import RunTrace


def run_trace_json_schema() -> dict:
    return RunTrace.model_json_schema()


if __name__ == "__main__":
    print(json.dumps(run_trace_json_schema(), indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest trace_schema/tests/test_export_schema.py -v`
Expected: 2 passed

Run: `uv run python -m trace_schema.export_schema | head -5`
Expected: prints valid JSON starting with `{`.

- [ ] **Step 5: Commit**

```bash
git add trace_schema/src/trace_schema/__init__.py trace_schema/src/trace_schema/export_schema.py trace_schema/tests/test_export_schema.py
git commit -m "feat(trace_schema): export public API and JSON Schema script"
```

---

## Definition of Done

- [ ] `uv run pytest trace_schema/` passes with all tests green.
- [ ] `uv run python -m trace_schema.export_schema` prints a valid JSON Schema.
- [ ] `trace_schema/pyproject.toml` has no dependency on `agent` or `harness`.
- [ ] All commits made; `git log --oneline` shows one commit per task.
