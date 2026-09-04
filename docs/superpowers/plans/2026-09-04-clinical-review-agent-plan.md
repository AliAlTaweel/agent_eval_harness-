# Clinical Documentation Review Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `clinical_review/` — a second, independent multi-agent review package (completeness/compliance/coding-clarity checks + merge verdict on a clinical note) — plus a harness adapter and synthetic test suite, proving the existing `trace_schema`/`harness` design generalizes to a new domain without modification.

**Architecture:** `clinical_review/` is a new `uv` workspace member, structurally identical in shape to `agent/` (own `llm.py`, `tracing.py`, `schemas.py`, `nodes/`, `graph.py`, `render.py`, `cli.py`) but with its own domain-specific prompts and schemas. It depends only on `trace_schema`, never on `agent` or `harness`. `harness/adapters/clinical_review.py` and a new `harness gate-clinical-review` CLI command are the sole integration points, mirroring the existing `agent`/`pr_review` integration exactly.

**Tech Stack:** Python 3.12, `uv`, LangGraph, Ollama (`qwen2.5:7b`), pydantic v2, Jinja2, click, Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-clinical-review-agent-design.md`

## Global Constraints

- Python 3.12, new `uv` workspace member `clinical_review`, sibling to `agent`/`harness`.
- `clinical_review` depends only on `trace_schema` — never imports from `agent` or `harness`.
- `harness` may depend on `clinical_review` only via its public `run_review()` function, guarded by the same try/except `ImportError` → `click.ClickException` pattern already used for `agent` in `harness/cli.py`.
- Local Ollama, model `qwen2.5:7b`.
- Every run produces a `trace_schema.RunTrace`.
- All review-rule prompts are explicitly illustrative example rules, not a real clinical/coding/compliance authority — this disclaimer must appear in the rendered comment footer and the README.
- No real patient data anywhere in the repo — all note text in tests/fixtures is fabricated.
- The existing `harness gate` command (PR-review) must not be modified — the new command is additive only (`gate-clinical-review`).

---

### Task 1: Workspace scaffold + `clinical_review` package skeleton

**Files:**
- Modify: `pyproject.toml` (root) — add `clinical_review` to workspace members and pytest testpaths
- Create: `clinical_review/pyproject.toml`
- Create: `clinical_review/src/clinical_review/__init__.py`
- Create: `clinical_review/tests/__init__.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: an installable `clinical_review` package importable as `import clinical_review`, registered as a `uv` workspace member.

- [ ] **Step 1: Modify the workspace root `pyproject.toml`**

Change the `[tool.uv.workspace]` and `[tool.pytest.ini_options]` sections to:

```toml
[tool.uv.workspace]
members = ["trace_schema", "agent", "harness", "clinical_review"]

[tool.pytest.ini_options]
testpaths = ["trace_schema/tests", "agent/tests", "harness/tests", "clinical_review/tests"]
addopts = "--import-mode=importlib"
markers = ["integration: requires a running local Ollama"]
```

- [ ] **Step 2: Create `clinical_review/pyproject.toml`**

```toml
[project]
name = "clinical-review"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "trace-schema",
    "langgraph>=0.2",
    "ollama>=0.3",
    "pydantic>=2.6",
    "jinja2>=3.1",
    "click>=8.1",
]

[project.optional-dependencies]
ui = [
    "streamlit>=1.38",
]

[project.scripts]
clinical-review = "clinical_review.cli:cli"

[tool.uv.sources]
trace-schema = { workspace = true }

[dependency-groups]
dev = [
    "pytest>=8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/clinical_review"]

[tool.pytest.ini_options]
markers = ["integration: requires a running local Ollama"]
```

- [ ] **Step 3: Create empty package files**

`clinical_review/src/clinical_review/__init__.py`:
```python
```

`clinical_review/tests/__init__.py`:
```python
```

- [ ] **Step 4: Sync the workspace and verify the package installs**

Run: `cd /Users/alial-taweel/projects/code_review && uv sync`
Expected: completes without error.

Run: `uv run python -c "import clinical_review; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml clinical_review/pyproject.toml clinical_review/src clinical_review/tests
git commit -m "chore: scaffold clinical_review workspace package"
```

---

### Task 2: `llm.py` + `tracing.py`

**Files:**
- Create: `clinical_review/src/clinical_review/llm.py`
- Create: `clinical_review/src/clinical_review/tracing.py`
- Test: `clinical_review/tests/test_llm.py`
- Test: `clinical_review/tests/test_tracing.py`

**Interfaces:**
- Consumes: `trace_schema.Step`, `trace_schema.ToolCall`.
- Produces:
  - `class LlmUsage` (dataclass: `tokens_in: int`, `tokens_out: int`).
  - `class OllamaClient`: `__init__(self, model: str = "qwen2.5:7b")`, `chat(self, system: str, user: str, response_model: type[BaseModel]) -> tuple[BaseModel, LlmUsage]`.
  - `class FakeOllamaClient`: `__init__(self, response: BaseModel, tokens_in: int = 0, tokens_out: int = 0)`, same `chat` signature, returns the canned response.
  - `class TraceRecorder`: `steps` property (`list[Step]`), `record(self, agent_name: str, action: str, fn: Callable[[], tuple[Any, LlmUsage]], tool_calls: list[ToolCall] | None = None) -> Any`.
  - Importable as `from clinical_review.llm import LlmUsage, OllamaClient, FakeOllamaClient` and `from clinical_review.tracing import TraceRecorder`.

- [ ] **Step 1: Write the failing tests**

```python
# clinical_review/tests/test_llm.py
from pydantic import BaseModel

from clinical_review.llm import FakeOllamaClient


class Greeting(BaseModel):
    text: str


def test_fake_client_returns_canned_response_and_usage():
    canned = Greeting(text="hello")
    client = FakeOllamaClient(response=canned, tokens_in=10, tokens_out=3)

    result, usage = client.chat(system="be nice", user="say hi", response_model=Greeting)

    assert result == canned
    assert usage.tokens_in == 10
    assert usage.tokens_out == 3
```

```python
# clinical_review/tests/test_tracing.py
from pydantic import BaseModel

from trace_schema import ToolCall

from clinical_review.llm import LlmUsage
from clinical_review.tracing import TraceRecorder


class Findings(BaseModel):
    issues: list[str] = []


def test_record_appends_step_with_timing_and_usage():
    recorder = TraceRecorder()

    def fake_call():
        return {"ok": True}, LlmUsage(tokens_in=5, tokens_out=2)

    output = recorder.record("completeness_review", "analyze_note", fake_call)

    assert output == {"ok": True}
    assert len(recorder.steps) == 1
    step = recorder.steps[0]
    assert step.agent_name == "completeness_review"
    assert step.action == "analyze_note"
    assert step.tokens_in == 5
    assert step.tokens_out == 2
    assert step.output == {"ok": True}
    assert step.cost_usd == 0.0
    assert step.started_at <= step.ended_at


def test_record_stores_tool_calls():
    recorder = TraceRecorder()

    def fake_call():
        return "result", LlmUsage(tokens_in=1, tokens_out=1)

    tool_calls = [ToolCall(name="grep", args={"pattern": "vitals"}, result=[])]
    recorder.record("completeness_review", "search", fake_call, tool_calls=tool_calls)

    assert recorder.steps[0].tool_calls == tool_calls


def test_record_serializes_pydantic_model_output_as_dict():
    recorder = TraceRecorder()
    findings = Findings(issues=["missing ROS"])

    def fake_call():
        return findings, LlmUsage(tokens_in=3, tokens_out=4)

    output = recorder.record("completeness_review", "analyze_note", fake_call)

    assert output is findings
    step = recorder.steps[0]
    assert isinstance(step.output, dict)
    assert step.output == findings.model_dump(mode="json")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest clinical_review/tests/test_llm.py clinical_review/tests/test_tracing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clinical_review.llm'`

- [ ] **Step 3: Write the implementation**

```python
# clinical_review/src/clinical_review/llm.py
from dataclasses import dataclass

import ollama
from pydantic import BaseModel


@dataclass
class LlmUsage:
    tokens_in: int
    tokens_out: int


class OllamaClient:
    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model

    def chat(
        self, system: str, user: str, response_model: type[BaseModel]
    ) -> tuple[BaseModel, LlmUsage]:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=response_model.model_json_schema(),
        )
        parsed = response_model.model_validate_json(response["message"]["content"])
        usage = LlmUsage(
            tokens_in=response.prompt_eval_count or 0,
            tokens_out=response.eval_count or 0,
        )
        return parsed, usage


class FakeOllamaClient:
    def __init__(self, response: BaseModel, tokens_in: int = 0, tokens_out: int = 0):
        self._response = response
        self._usage = LlmUsage(tokens_in=tokens_in, tokens_out=tokens_out)

    def chat(
        self, system: str, user: str, response_model: type[BaseModel]
    ) -> tuple[BaseModel, LlmUsage]:
        return self._response, self._usage
```

```python
# clinical_review/src/clinical_review/tracing.py
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel
from trace_schema import Step, ToolCall

from clinical_review.llm import LlmUsage


class TraceRecorder:
    def __init__(self):
        self._steps: list[Step] = []

    @property
    def steps(self) -> list[Step]:
        return self._steps

    def record(
        self,
        agent_name: str,
        action: str,
        fn: Callable[[], tuple[Any, LlmUsage]],
        tool_calls: list[ToolCall] | None = None,
    ) -> Any:
        started_at = datetime.now(timezone.utc)
        output, usage = fn()
        ended_at = datetime.now(timezone.utc)
        if isinstance(output, BaseModel):
            recorded_output = output.model_dump(mode="json")
        elif isinstance(output, (dict, list, str, int, float, bool, type(None))):
            recorded_output = output
        else:
            recorded_output = str(output)
        self._steps.append(
            Step(
                agent_name=agent_name,
                action=action,
                tool_calls=tool_calls or [],
                output=recorded_output,
                started_at=started_at,
                ended_at=ended_at,
                tokens_in=usage.tokens_in,
                tokens_out=usage.tokens_out,
                cost_usd=0.0,
            )
        )
        return output
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest clinical_review/tests/test_llm.py clinical_review/tests/test_tracing.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add clinical_review/src/clinical_review/llm.py clinical_review/src/clinical_review/tracing.py clinical_review/tests/test_llm.py clinical_review/tests/test_tracing.py
git commit -m "feat(clinical_review): add OllamaClient and TraceRecorder"
```

---

### Task 3: `schemas.py`

**Files:**
- Create: `clinical_review/src/clinical_review/schemas.py`
- Test: `clinical_review/tests/test_schemas.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `EncounterType = Literal["new_patient", "follow_up"]`
  - `class Finding(BaseModel)`: `element: str`, `description: str`, `severity: Literal["low", "medium", "high", "critical"]`.
  - `class CompletenessFindings(BaseModel)`: `issues: list[Finding] = []`.
  - `class ComplianceFindings(BaseModel)`: `issues: list[Finding] = []`.
  - `class CodingClarityFindings(BaseModel)`: `issues: list[Finding] = []`.
  - `class Verdict(BaseModel)`: `completeness_issues: list[Finding] = []`, `compliance_issues: list[Finding] = []`, `coding_clarity_issues: list[Finding] = []`, `verdict: Literal["approve", "needs_revision"]`.
  - Importable as `from clinical_review.schemas import EncounterType, Finding, CompletenessFindings, ComplianceFindings, CodingClarityFindings, Verdict`.

- [ ] **Step 1: Write the failing test**

```python
# clinical_review/tests/test_schemas.py
from clinical_review.schemas import (
    CodingClarityFindings,
    CompletenessFindings,
    ComplianceFindings,
    Finding,
    Verdict,
)


def test_finding_requires_severity_enum():
    finding = Finding(element="Review of Systems", description="not documented", severity="medium")
    assert finding.severity == "medium"


def test_verdict_shape():
    v = Verdict(
        completeness_issues=[Finding(element="ROS", description="missing", severity="medium")],
        compliance_issues=[],
        coding_clarity_issues=[],
        verdict="needs_revision",
    )
    assert v.verdict == "needs_revision"
    assert len(v.completeness_issues) == 1


def test_findings_models_hold_issue_lists():
    completeness = CompletenessFindings(issues=[])
    compliance = ComplianceFindings(issues=[])
    coding_clarity = CodingClarityFindings(issues=[])
    assert completeness.issues == []
    assert compliance.issues == []
    assert coding_clarity.issues == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest clinical_review/tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clinical_review.schemas'`

- [ ] **Step 3: Write the implementation**

```python
# clinical_review/src/clinical_review/schemas.py
from typing import Literal

from pydantic import BaseModel

EncounterType = Literal["new_patient", "follow_up"]


class Finding(BaseModel):
    element: str
    description: str
    severity: Literal["low", "medium", "high", "critical"]


class CompletenessFindings(BaseModel):
    issues: list[Finding] = []


class ComplianceFindings(BaseModel):
    issues: list[Finding] = []


class CodingClarityFindings(BaseModel):
    issues: list[Finding] = []


class Verdict(BaseModel):
    completeness_issues: list[Finding] = []
    compliance_issues: list[Finding] = []
    coding_clarity_issues: list[Finding] = []
    verdict: Literal["approve", "needs_revision"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest clinical_review/tests/test_schemas.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add clinical_review/src/clinical_review/schemas.py clinical_review/tests/test_schemas.py
git commit -m "feat(clinical_review): add Finding/Verdict schemas"
```

---

### Task 4: Completeness review node

**Files:**
- Create: `clinical_review/src/clinical_review/nodes/__init__.py`
- Create: `clinical_review/src/clinical_review/nodes/completeness.py`
- Test: `clinical_review/tests/nodes/__init__.py`
- Test: `clinical_review/tests/nodes/test_completeness.py`

**Interfaces:**
- Consumes: `CompletenessFindings` from Task 3, `TraceRecorder` from Task 2, `EncounterType` from Task 3.
- Produces: `review_completeness(note_text: str, encounter_type: EncounterType, client, recorder: TraceRecorder) -> CompletenessFindings`, importable as `from clinical_review.nodes.completeness import review_completeness`. `REQUIRED_ELEMENTS: dict[str, list[str]]` module-level constant, keyed by `"new_patient"` / `"follow_up"`.

- [ ] **Step 1: Write the failing test**

```python
# clinical_review/tests/nodes/__init__.py
```

```python
# clinical_review/tests/nodes/test_completeness.py
from clinical_review.llm import FakeOllamaClient
from clinical_review.nodes.completeness import review_completeness
from clinical_review.schemas import CompletenessFindings, Finding
from clinical_review.tracing import TraceRecorder


def test_review_completeness_returns_findings_and_records_step():
    canned = CompletenessFindings(
        issues=[Finding(element="Review of Systems", description="not documented", severity="medium")]
    )
    client = FakeOllamaClient(response=canned, tokens_in=100, tokens_out=30)
    recorder = TraceRecorder()

    result = review_completeness(
        note_text="CHIEF COMPLAINT: cough.\nASSESSMENT: viral illness.\nPLAN: rest.",
        encounter_type="new_patient",
        client=client,
        recorder=recorder,
    )

    assert result == canned
    assert len(recorder.steps) == 1
    assert recorder.steps[0].agent_name == "completeness_review"


def test_review_completeness_uses_follow_up_required_elements():
    canned = CompletenessFindings(issues=[])
    client = FakeOllamaClient(response=canned)
    recorder = TraceRecorder()

    review_completeness(
        note_text="REASON FOR VISIT: follow-up.\nASSESSMENT: stable.\nPLAN: continue current regimen.",
        encounter_type="follow_up",
        client=client,
        recorder=recorder,
    )

    assert len(recorder.steps) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest clinical_review/tests/nodes/test_completeness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clinical_review.nodes'`

- [ ] **Step 3: Write the implementation**

```python
# clinical_review/src/clinical_review/nodes/__init__.py
```

```python
# clinical_review/src/clinical_review/nodes/completeness.py
from clinical_review.schemas import CompletenessFindings, EncounterType
from clinical_review.tracing import TraceRecorder

REQUIRED_ELEMENTS: dict[str, list[str]] = {
    "new_patient": [
        "Chief Complaint",
        "History of Present Illness",
        "Review of Systems",
        "Past Medical History",
        "Physical Exam findings",
        "Assessment",
        "Plan",
    ],
    "follow_up": [
        "Reason for visit or interval history",
        "Relevant Physical Exam findings",
        "Assessment",
        "Plan",
    ],
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are reviewing a clinical note for documentation completeness, for "
    "an encounter of type '{encounter_type}'. A complete note of this type "
    "should document: {required}. Identify which of these elements are "
    "missing or clearly inadequate in the note. Only report elements you "
    "cannot find evidence of in the note text. Respond with structured "
    "findings. This is an illustrative documentation-quality check, not a "
    "real coding/compliance authority."
)


def review_completeness(
    note_text: str, encounter_type: EncounterType, client, recorder: TraceRecorder
) -> CompletenessFindings:
    required = ", ".join(REQUIRED_ELEMENTS[encounter_type])
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(encounter_type=encounter_type, required=required)
    return recorder.record(
        "completeness_review",
        "analyze_note",
        lambda: client.chat(system=system_prompt, user=note_text, response_model=CompletenessFindings),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest clinical_review/tests/nodes/test_completeness.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add clinical_review/src/clinical_review/nodes/__init__.py clinical_review/src/clinical_review/nodes/completeness.py clinical_review/tests/nodes/__init__.py clinical_review/tests/nodes/test_completeness.py
git commit -m "feat(clinical_review): add completeness review node"
```

---

### Task 5: Compliance review node

**Files:**
- Create: `clinical_review/src/clinical_review/nodes/compliance.py`
- Test: `clinical_review/tests/nodes/test_compliance.py`

**Interfaces:**
- Consumes: `ComplianceFindings` from Task 3, `TraceRecorder` from Task 2.
- Produces: `review_compliance(note_text: str, client, recorder: TraceRecorder) -> ComplianceFindings`, importable as `from clinical_review.nodes.compliance import review_compliance`.

- [ ] **Step 1: Write the failing test**

```python
# clinical_review/tests/nodes/test_compliance.py
from clinical_review.llm import FakeOllamaClient
from clinical_review.nodes.compliance import review_compliance
from clinical_review.schemas import ComplianceFindings, Finding
from clinical_review.tracing import TraceRecorder


def test_review_compliance_returns_findings_and_records_step():
    canned = ComplianceFindings(
        issues=[
            Finding(
                element="Assessment/Plan",
                description="plan escalates therapy but exam and vitals show well-controlled findings",
                severity="high",
            )
        ]
    )
    client = FakeOllamaClient(response=canned, tokens_in=80, tokens_out=25)
    recorder = TraceRecorder()

    result = review_compliance(
        note_text="VITALS: BP 122/76.\nASSESSMENT: poorly controlled hypertension.\nPLAN: escalate therapy.",
        client=client,
        recorder=recorder,
    )

    assert result == canned
    assert len(recorder.steps) == 1
    assert recorder.steps[0].agent_name == "compliance_review"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest clinical_review/tests/nodes/test_compliance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clinical_review.nodes.compliance'`

- [ ] **Step 3: Write the implementation**

```python
# clinical_review/src/clinical_review/nodes/compliance.py
from clinical_review.schemas import ComplianceFindings
from clinical_review.tracing import TraceRecorder

SYSTEM_PROMPT = (
    "You are reviewing a clinical note for documentation-compliance issues. "
    "Identify cases where the Assessment or Plan is not supported by any "
    "documented finding in the note, or where the exam describes abnormal "
    "findings but no vital signs are recorded. Only report issues clearly "
    "supported by the note content. Respond with structured findings. This "
    "is an illustrative documentation-quality check, not a real "
    "coding/compliance authority."
)


def review_compliance(note_text: str, client, recorder: TraceRecorder) -> ComplianceFindings:
    return recorder.record(
        "compliance_review",
        "analyze_note",
        lambda: client.chat(system=SYSTEM_PROMPT, user=note_text, response_model=ComplianceFindings),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest clinical_review/tests/nodes/test_compliance.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add clinical_review/src/clinical_review/nodes/compliance.py clinical_review/tests/nodes/test_compliance.py
git commit -m "feat(clinical_review): add compliance review node"
```

---

### Task 6: Coding-clarity review node

**Files:**
- Create: `clinical_review/src/clinical_review/nodes/coding_clarity.py`
- Test: `clinical_review/tests/nodes/test_coding_clarity.py`

**Interfaces:**
- Consumes: `CodingClarityFindings` from Task 3, `TraceRecorder` from Task 2.
- Produces: `review_coding_clarity(note_text: str, client, recorder: TraceRecorder) -> CodingClarityFindings`, importable as `from clinical_review.nodes.coding_clarity import review_coding_clarity`.

- [ ] **Step 1: Write the failing test**

```python
# clinical_review/tests/nodes/test_coding_clarity.py
from clinical_review.llm import FakeOllamaClient
from clinical_review.nodes.coding_clarity import review_coding_clarity
from clinical_review.schemas import CodingClarityFindings, Finding
from clinical_review.tracing import TraceRecorder


def test_review_coding_clarity_returns_findings_and_records_step():
    canned = CodingClarityFindings(
        issues=[Finding(element="Assessment", description="vague, no stated working diagnosis", severity="low")]
    )
    client = FakeOllamaClient(response=canned, tokens_in=60, tokens_out=20)
    recorder = TraceRecorder()

    result = review_coding_clarity(
        note_text="ASSESSMENT: probably fine.\nPLAN: reassure.",
        client=client,
        recorder=recorder,
    )

    assert result == canned
    assert len(recorder.steps) == 1
    assert recorder.steps[0].agent_name == "coding_clarity_review"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest clinical_review/tests/nodes/test_coding_clarity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clinical_review.nodes.coding_clarity'`

- [ ] **Step 3: Write the implementation**

```python
# clinical_review/src/clinical_review/nodes/coding_clarity.py
from clinical_review.schemas import CodingClarityFindings
from clinical_review.tracing import TraceRecorder

SYSTEM_PROMPT = (
    "You are reviewing a clinical note for coding clarity. Identify cases "
    "where the Assessment uses vague, non-specific language that would not "
    "map to a clear, billable diagnosis (for example, phrases like "
    "'probably fine' or 'likely nothing serious' without a stated working "
    "diagnosis). Only report issues clearly supported by the note content. "
    "Respond with structured findings. This is an illustrative "
    "documentation-quality check, not a real coding/compliance authority."
)


def review_coding_clarity(note_text: str, client, recorder: TraceRecorder) -> CodingClarityFindings:
    return recorder.record(
        "coding_clarity_review",
        "analyze_note",
        lambda: client.chat(system=SYSTEM_PROMPT, user=note_text, response_model=CodingClarityFindings),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest clinical_review/tests/nodes/test_coding_clarity.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add clinical_review/src/clinical_review/nodes/coding_clarity.py clinical_review/tests/nodes/test_coding_clarity.py
git commit -m "feat(clinical_review): add coding-clarity review node"
```

---

### Task 7: Merge node

**Files:**
- Create: `clinical_review/src/clinical_review/nodes/merge.py`
- Test: `clinical_review/tests/nodes/test_merge.py`

**Interfaces:**
- Consumes: `CompletenessFindings`, `ComplianceFindings`, `CodingClarityFindings`, `Verdict` from Task 3, `TraceRecorder` from Task 2.
- Produces: `merge_findings(completeness: CompletenessFindings, compliance: ComplianceFindings, coding_clarity: CodingClarityFindings, client, recorder: TraceRecorder) -> Verdict`, importable as `from clinical_review.nodes.merge import merge_findings`. Deterministically carries through the three issue lists from inputs; only the `verdict` field is trusted from the LLM's response.

- [ ] **Step 1: Write the failing test**

```python
# clinical_review/tests/nodes/test_merge.py
from clinical_review.llm import FakeOllamaClient
from clinical_review.nodes.merge import merge_findings
from clinical_review.schemas import (
    CodingClarityFindings,
    CompletenessFindings,
    ComplianceFindings,
    Finding,
    Verdict,
)
from clinical_review.tracing import TraceRecorder


def test_merge_findings_combines_inputs_and_uses_llm_verdict():
    completeness = CompletenessFindings(
        issues=[Finding(element="Review of Systems", description="missing", severity="medium")]
    )
    compliance = ComplianceFindings(issues=[])
    coding_clarity = CodingClarityFindings(issues=[])
    canned_verdict = Verdict(
        completeness_issues=[], compliance_issues=[], coding_clarity_issues=[], verdict="needs_revision"
    )
    client = FakeOllamaClient(response=canned_verdict, tokens_in=50, tokens_out=5)
    recorder = TraceRecorder()

    result = merge_findings(
        completeness=completeness, compliance=compliance, coding_clarity=coding_clarity, client=client, recorder=recorder
    )

    assert result.verdict == "needs_revision"
    assert result.completeness_issues == completeness.issues
    assert result.compliance_issues == compliance.issues
    assert result.coding_clarity_issues == coding_clarity.issues
    assert recorder.steps[0].agent_name == "merge"


def test_merge_findings_approves_when_no_issues():
    completeness = CompletenessFindings(issues=[])
    compliance = ComplianceFindings(issues=[])
    coding_clarity = CodingClarityFindings(issues=[])
    canned_verdict = Verdict(
        completeness_issues=[], compliance_issues=[], coding_clarity_issues=[], verdict="approve"
    )
    client = FakeOllamaClient(response=canned_verdict)
    recorder = TraceRecorder()

    result = merge_findings(
        completeness=completeness, compliance=compliance, coding_clarity=coding_clarity, client=client, recorder=recorder
    )

    assert result.verdict == "approve"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest clinical_review/tests/nodes/test_merge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clinical_review.nodes.merge'`

- [ ] **Step 3: Write the implementation**

```python
# clinical_review/src/clinical_review/nodes/merge.py
from clinical_review.schemas import (
    CodingClarityFindings,
    CompletenessFindings,
    ComplianceFindings,
    Verdict,
)
from clinical_review.tracing import TraceRecorder

SYSTEM_PROMPT = (
    "You are the supervising reviewer for a clinical note's documentation "
    "quality. Given a summary of completeness, compliance, and "
    "coding-clarity findings, decide the final verdict: 'approve' if there "
    "are no compliance issues and no missing required elements, otherwise "
    "'needs_revision'. Respond with a verdict field only."
)


def _summarize(
    completeness: CompletenessFindings, compliance: ComplianceFindings, coding_clarity: CodingClarityFindings
) -> str:
    lines = [
        f"Completeness issues: {[i.description for i in completeness.issues]}",
        f"Compliance issues: {[i.description for i in compliance.issues]}",
        f"Coding clarity issues: {[i.description for i in coding_clarity.issues]}",
    ]
    return "\n".join(lines)


def merge_findings(
    completeness: CompletenessFindings,
    compliance: ComplianceFindings,
    coding_clarity: CodingClarityFindings,
    client,
    recorder: TraceRecorder,
) -> Verdict:
    summary = _summarize(completeness, compliance, coding_clarity)

    def call():
        parsed, usage = client.chat(system=SYSTEM_PROMPT, user=summary, response_model=Verdict)
        return parsed, usage

    llm_verdict: Verdict = recorder.record("merge", "decide_verdict", call)

    return Verdict(
        completeness_issues=completeness.issues,
        compliance_issues=compliance.issues,
        coding_clarity_issues=coding_clarity.issues,
        verdict=llm_verdict.verdict,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest clinical_review/tests/nodes/test_merge.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add clinical_review/src/clinical_review/nodes/merge.py clinical_review/tests/nodes/test_merge.py
git commit -m "feat(clinical_review): add merge node"
```

---

### Task 8: Graph wiring + `run_review`

**Files:**
- Create: `clinical_review/src/clinical_review/graph.py`
- Test: `clinical_review/tests/test_graph.py`

**Interfaces:**
- Consumes: `review_completeness` (Task 4), `review_compliance` (Task 5), `review_coding_clarity` (Task 6), `merge_findings` (Task 7), `TraceRecorder` (Task 2), `CompletenessFindings`/`ComplianceFindings`/`CodingClarityFindings`/`Verdict`/`EncounterType` (Task 3), `trace_schema.RunTrace`.
- Produces:
  - `class ReviewFailedError(RuntimeError)`: `__init__(self, cause: BaseException, trace: RunTrace)`, exposes `.trace`.
  - `run_review(note_text: str, encounter_type: EncounterType, client) -> tuple[Verdict, RunTrace]`. Raises `ReviewFailedError` on node failure, carrying a partial trace. `trace.input == {"note_text": note_text, "encounter_type": encounter_type}`.

**IMPORTANT — concurrency:** the three review nodes run in parallel under LangGraph's executor, same as `agent/graph.py`. The fake test client below MUST dispatch by `response_model` type, not call order — the original `agent` project found via jitter testing that order-based fakes fail 83% of the time against a genuinely concurrent pipeline. Do not write an order-based fake for this task.

- [ ] **Step 1: Write the failing tests**

```python
# clinical_review/tests/test_graph.py
import pytest
from trace_schema import RunTrace

from clinical_review.graph import ReviewFailedError, run_review
from clinical_review.llm import LlmUsage
from clinical_review.schemas import (
    CodingClarityFindings,
    CompletenessFindings,
    ComplianceFindings,
    Finding,
    Verdict,
)


class SequencedFakeClient:
    """Returns a canned response keyed by the requested response_model.

    The three review nodes run concurrently under LangGraph's executor, so
    dispatching by call order is a race. Dispatching by response_model type
    is deterministic regardless of which node's thread calls .chat() first.
    """

    def __init__(self, responses: dict):
        self._responses = dict(responses)

    def chat(self, system, user, response_model):
        response = self._responses[response_model]
        return response, LlmUsage(tokens_in=10, tokens_out=5)


class FailingClient:
    """Raises for one response_model, returns canned responses for the rest."""

    def __init__(self, responses: dict, fail_on: type):
        self._responses = dict(responses)
        self._fail_on = fail_on

    def chat(self, system, user, response_model):
        if response_model is self._fail_on:
            raise RuntimeError("simulated node failure")
        response = self._responses[response_model]
        return response, LlmUsage(tokens_in=10, tokens_out=5)


def test_run_review_produces_verdict_and_trace():
    completeness = CompletenessFindings(
        issues=[Finding(element="Review of Systems", description="missing", severity="medium")]
    )
    compliance = ComplianceFindings(issues=[])
    coding_clarity = CodingClarityFindings(issues=[])
    merge_llm_output = Verdict(
        completeness_issues=[], compliance_issues=[], coding_clarity_issues=[], verdict="needs_revision"
    )

    client = SequencedFakeClient(
        {
            CompletenessFindings: completeness,
            ComplianceFindings: compliance,
            CodingClarityFindings: coding_clarity,
            Verdict: merge_llm_output,
        }
    )

    verdict, trace = run_review(note_text="CHIEF COMPLAINT: cough.", encounter_type="new_patient", client=client)

    assert isinstance(verdict, Verdict)
    assert verdict.verdict == "needs_revision"
    assert len(verdict.completeness_issues) == 1

    assert isinstance(trace, RunTrace)
    assert len(trace.steps) == 4
    agent_names = {s.agent_name for s in trace.steps}
    assert agent_names == {"completeness_review", "compliance_review", "coding_clarity_review", "merge"}
    assert trace.final_output["verdict"] == "needs_revision"
    assert trace.total_tokens == sum(s.tokens_in + s.tokens_out for s in trace.steps)
    assert trace.input == {"note_text": "CHIEF COMPLAINT: cough.", "encounter_type": "new_patient"}

    started_ats = [s.started_at for s in trace.steps]
    assert started_ats == sorted(started_ats)
    assert trace.steps[-1].agent_name == "merge"


def test_run_review_captures_partial_trace_on_node_failure():
    completeness = CompletenessFindings(issues=[])
    compliance = ComplianceFindings(issues=[])
    coding_clarity = CodingClarityFindings(issues=[])

    client = FailingClient(
        {
            CompletenessFindings: completeness,
            ComplianceFindings: compliance,
            CodingClarityFindings: coding_clarity,
        },
        fail_on=CodingClarityFindings,
    )

    with pytest.raises(ReviewFailedError) as exc_info:
        run_review(note_text="ASSESSMENT: fine.", encounter_type="follow_up", client=client)

    trace = exc_info.value.trace
    assert isinstance(trace, RunTrace)
    agent_names = {s.agent_name for s in trace.steps}
    assert agent_names == {"completeness_review", "compliance_review"}
    assert "simulated node failure" in trace.final_output["error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest clinical_review/tests/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clinical_review.graph'`

- [ ] **Step 3: Write the implementation**

```python
# clinical_review/src/clinical_review/graph.py
import uuid
from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from trace_schema import RunTrace

from clinical_review.nodes.coding_clarity import review_coding_clarity
from clinical_review.nodes.compliance import review_compliance
from clinical_review.nodes.completeness import review_completeness
from clinical_review.nodes.merge import merge_findings
from clinical_review.schemas import (
    CodingClarityFindings,
    CompletenessFindings,
    ComplianceFindings,
    EncounterType,
    Verdict,
)
from clinical_review.tracing import TraceRecorder


class ReviewState(TypedDict, total=False):
    note_text: str
    encounter_type: EncounterType
    completeness: CompletenessFindings
    compliance: ComplianceFindings
    coding_clarity: CodingClarityFindings
    verdict: Verdict


def _build_graph(client, recorder: TraceRecorder):
    graph = StateGraph(ReviewState)

    graph.add_node(
        "completeness_review",
        lambda s: {"completeness": review_completeness(s["note_text"], s["encounter_type"], client, recorder)},
    )
    graph.add_node(
        "compliance_review",
        lambda s: {"compliance": review_compliance(s["note_text"], client, recorder)},
    )
    graph.add_node(
        "coding_clarity_review",
        lambda s: {"coding_clarity": review_coding_clarity(s["note_text"], client, recorder)},
    )
    graph.add_node(
        "merge",
        lambda s: {
            "verdict": merge_findings(s["completeness"], s["compliance"], s["coding_clarity"], client, recorder)
        },
    )

    graph.add_edge(START, "completeness_review")
    graph.add_edge(START, "compliance_review")
    graph.add_edge(START, "coding_clarity_review")
    graph.add_edge("completeness_review", "merge")
    graph.add_edge("compliance_review", "merge")
    graph.add_edge("coding_clarity_review", "merge")
    graph.add_edge("merge", END)

    return graph.compile()


class ReviewFailedError(RuntimeError):
    """Raised when a review node fails; carries whatever trace was captured before the failure."""

    def __init__(self, cause: BaseException, trace: RunTrace):
        super().__init__(str(cause))
        self.trace = trace


def _build_trace(
    note_text: str,
    encounter_type: EncounterType,
    recorder: TraceRecorder,
    started_at: datetime,
    final_output: dict | None,
) -> RunTrace:
    ended_at = datetime.now(timezone.utc)
    steps = sorted(recorder.steps, key=lambda s: s.started_at)
    total_tokens = sum(s.tokens_in + s.tokens_out for s in steps)

    return RunTrace(
        run_id=str(uuid.uuid4()),
        input={"note_text": note_text, "encounter_type": encounter_type},
        steps=steps,
        final_output=final_output,
        started_at=started_at,
        ended_at=ended_at,
        total_tokens=total_tokens,
        total_cost_usd=sum(s.cost_usd for s in steps),
    )


def run_review(note_text: str, encounter_type: EncounterType, client) -> tuple[Verdict, RunTrace]:
    recorder = TraceRecorder()
    compiled = _build_graph(client, recorder)

    started_at = datetime.now(timezone.utc)
    try:
        final_state = compiled.invoke({"note_text": note_text, "encounter_type": encounter_type})
    except Exception as exc:
        partial_trace = _build_trace(note_text, encounter_type, recorder, started_at, {"error": str(exc)})
        raise ReviewFailedError(exc, partial_trace) from exc

    verdict: Verdict = final_state["verdict"]
    trace = _build_trace(note_text, encounter_type, recorder, started_at, verdict.model_dump(mode="json"))
    return verdict, trace
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest clinical_review/tests/test_graph.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add clinical_review/src/clinical_review/graph.py clinical_review/tests/test_graph.py
git commit -m "feat(clinical_review): wire LangGraph pipeline and run_review"
```

---

### Task 9: `render.py`

**Files:**
- Create: `clinical_review/src/clinical_review/render.py`
- Test: `clinical_review/tests/test_render.py`

**Interfaces:**
- Consumes: `Verdict` from Task 3.
- Produces: `render_comment(verdict: Verdict) -> str`, importable as `from clinical_review.render import render_comment`.

- [ ] **Step 1: Write the failing test**

```python
# clinical_review/tests/test_render.py
from clinical_review.render import render_comment
from clinical_review.schemas import Finding, Verdict


def test_render_comment_includes_verdict_and_issues():
    verdict = Verdict(
        completeness_issues=[Finding(element="Review of Systems", description="not documented", severity="medium")],
        compliance_issues=[],
        coding_clarity_issues=[],
        verdict="needs_revision",
    )
    comment = render_comment(verdict)

    assert "Needs Revision" in comment
    assert "Review of Systems" in comment
    assert "not documented" in comment
    assert "illustrative demo agent" in comment.lower()


def test_render_comment_clean_approve():
    verdict = Verdict(
        completeness_issues=[], compliance_issues=[], coding_clarity_issues=[], verdict="approve"
    )
    comment = render_comment(verdict)
    assert "Approve" in comment
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest clinical_review/tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clinical_review.render'`

- [ ] **Step 3: Write the implementation**

```python
# clinical_review/src/clinical_review/render.py
from jinja2 import Template

from clinical_review.schemas import Verdict

TEMPLATE = Template(
    """\
## Clinical Note Review: {{ "✅ Approve" if verdict.verdict == "approve" else "🚫 Needs Revision" }}

{% if verdict.completeness_issues %}
### Completeness Issues
{% for issue in verdict.completeness_issues %}
- **[{{ issue.severity }}]** `{{ issue.element }}` — {{ issue.description }}
{% endfor %}
{% else %}
### Completeness Issues
None found.
{% endif %}

{% if verdict.compliance_issues %}
### Compliance Issues
{% for issue in verdict.compliance_issues %}
- **[{{ issue.severity }}]** `{{ issue.element }}` — {{ issue.description }}
{% endfor %}
{% else %}
### Compliance Issues
None found.
{% endif %}

{% if verdict.coding_clarity_issues %}
### Coding Clarity Issues
{% for issue in verdict.coding_clarity_issues %}
- **[{{ issue.severity }}]** `{{ issue.element }}` — {{ issue.description }}
{% endfor %}
{% else %}
### Coding Clarity Issues
None found.
{% endif %}

---
*Generated by an illustrative demo agent. Not a clinical compliance or coding authority.*
"""
)


def render_comment(verdict: Verdict) -> str:
    return TEMPLATE.render(verdict=verdict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest clinical_review/tests/test_render.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add clinical_review/src/clinical_review/render.py clinical_review/tests/test_render.py
git commit -m "feat(clinical_review): add markdown comment rendering"
```

---

### Task 10: CLI

**Files:**
- Create: `clinical_review/src/clinical_review/cli.py`
- Test: `clinical_review/tests/test_cli.py`

**Interfaces:**
- Consumes: `run_review`/`ReviewFailedError` (Task 8), `OllamaClient` (Task 2), `render_comment` (Task 9).
- Produces: `clinical-review review --note-file <path> --encounter-type {new_patient,follow_up} --trace-out trace.json --comment-out comment.md --model qwen2.5:7b` CLI command, registered via `[project.scripts]` in Task 1's `pyproject.toml`.

- [ ] **Step 1: Write the failing test**

```python
# clinical_review/tests/test_cli.py
import json

from click.testing import CliRunner

from clinical_review.cli import cli
from clinical_review.schemas import Finding, Verdict


def test_review_command_writes_trace_and_comment(tmp_path, monkeypatch):
    note_file = tmp_path / "note.txt"
    note_file.write_text("CHIEF COMPLAINT: cough.\nASSESSMENT: viral illness.\nPLAN: rest.")

    trace_out = tmp_path / "trace.json"
    comment_out = tmp_path / "comment.md"

    canned_verdict = Verdict(
        completeness_issues=[Finding(element="Review of Systems", description="missing", severity="medium")],
        compliance_issues=[],
        coding_clarity_issues=[],
        verdict="needs_revision",
    )

    def fake_run_review(note_text, encounter_type, client):
        from datetime import datetime, timezone

        from trace_schema import RunTrace

        now = datetime.now(timezone.utc)
        trace = RunTrace(
            run_id="test-run",
            input={"note_text": note_text, "encounter_type": encounter_type},
            steps=[],
            final_output=canned_verdict.model_dump(),
            started_at=now,
            ended_at=now,
            total_tokens=0,
            total_cost_usd=0.0,
        )
        return canned_verdict, trace

    monkeypatch.setattr("clinical_review.cli.run_review", fake_run_review)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "review",
            "--note-file",
            str(note_file),
            "--encounter-type",
            "new_patient",
            "--trace-out",
            str(trace_out),
            "--comment-out",
            str(comment_out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert trace_out.exists()
    assert comment_out.exists()

    trace_data = json.loads(trace_out.read_text())
    assert trace_data["run_id"] == "test-run"

    comment_text = comment_out.read_text()
    assert "Review of Systems" in comment_text

    output_json = json.loads(result.output)
    assert output_json["verdict"] == "needs_revision"


def test_review_command_rejects_invalid_encounter_type(tmp_path):
    note_file = tmp_path / "note.txt"
    note_file.write_text("some note")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["review", "--note-file", str(note_file), "--encounter-type", "annual_physical"],
    )

    assert result.exit_code != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest clinical_review/tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clinical_review.cli'`

- [ ] **Step 3: Write the implementation**

```python
# clinical_review/src/clinical_review/cli.py
import json
import sys

import click

from clinical_review.graph import ReviewFailedError, run_review
from clinical_review.llm import OllamaClient
from clinical_review.render import render_comment


@click.group()
def cli():
    pass


@cli.command()
@click.option("--note-file", required=True, type=click.Path(exists=True), help="Path to a clinical note text file.")
@click.option(
    "--encounter-type",
    required=True,
    type=click.Choice(["new_patient", "follow_up"]),
    help="Encounter type — determines which documentation elements are required.",
)
@click.option("--trace-out", default="trace.json", type=click.Path(), help="Where to write the RunTrace JSON.")
@click.option(
    "--comment-out", default="comment.md", type=click.Path(), help="Where to write the markdown review comment."
)
@click.option("--model", default="qwen2.5:7b", help="Ollama model to use.")
def review(note_file, encounter_type, trace_out, comment_out, model):
    with open(note_file) as f:
        note_text = f.read()
    client = OllamaClient(model=model)

    try:
        verdict, trace = run_review(note_text=note_text, encounter_type=encounter_type, client=client)
    except ReviewFailedError as exc:
        with open(trace_out, "w") as f:
            json.dump(exc.trace.model_dump(mode="json"), f, indent=2)
        click.echo(f"Review failed: {exc}", err=True)
        sys.exit(1)

    with open(trace_out, "w") as f:
        json.dump(trace.model_dump(mode="json"), f, indent=2)

    with open(comment_out, "w") as f:
        f.write(render_comment(verdict))

    click.echo(json.dumps(verdict.model_dump(), indent=2))


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest clinical_review/tests/test_cli.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add clinical_review/src/clinical_review/cli.py clinical_review/tests/test_cli.py
git commit -m "feat(clinical_review): add review CLI command"
```

---

### Task 11: Streamlit UI

**Files:**
- Create: `clinical_review/src/clinical_review/streamlit_app.py`

**Interfaces:**
- Consumes: `run_review`/`ReviewFailedError` (Task 8), `OllamaClient` (Task 2), `render_comment` (Task 9).
- Produces: a runnable Streamlit page with a note-text textarea, an encounter-type dropdown, and a Run Review button. No automated test (Streamlit apps aren't unit-testable the way CLI/library code is) — verified manually in Step 3 below, same as the original PR agent's UI.

- [ ] **Step 1: Write the implementation**

```python
# clinical_review/src/clinical_review/streamlit_app.py
import streamlit as st

from clinical_review.graph import ReviewFailedError, run_review
from clinical_review.llm import OllamaClient
from clinical_review.render import render_comment

st.set_page_config(page_title="Clinical Note Review Agent", layout="wide")
st.title("Clinical Note Review Agent")
st.caption("Illustrative demo agent. Not a clinical compliance or coding authority.")

model = st.sidebar.text_input("Ollama model", value="qwen2.5:7b")

encounter_type = st.selectbox("Encounter type", ["new_patient", "follow_up"])
note_text = st.text_area("Clinical note text", height=300)

if note_text.strip() and st.button("Run Review", type="primary"):
    client = OllamaClient(model=model)
    with st.spinner("Running multi-agent review..."):
        try:
            verdict, trace = run_review(note_text=note_text, encounter_type=encounter_type, client=client)
        except ReviewFailedError as exc:
            st.error(f"Review failed: {exc}")
            st.json(exc.trace.model_dump(mode="json"))
            st.stop()

    st.markdown(render_comment(verdict))

    with st.expander("Verdict JSON"):
        st.json(verdict.model_dump(mode="json"))

    with st.expander("Trace stats"):
        st.metric("Total tokens", trace.total_tokens)
        st.metric("Total cost (USD)", f"{trace.total_cost_usd:.4f}")
        latency = (trace.ended_at - trace.started_at).total_seconds()
        st.metric("Latency (s)", f"{latency:.2f}")

    with st.expander("Full trace JSON"):
        st.json(trace.model_dump(mode="json"))
```

- [ ] **Step 2: Syntax-check the file**

Run: `uv run python -c "import ast; ast.parse(open('clinical_review/src/clinical_review/streamlit_app.py').read())" && echo "syntax ok"`
Expected: prints `syntax ok`

- [ ] **Step 3: Manually verify in a browser**

Run: `uv sync --extra ui` then `uv run streamlit run clinical_review/src/clinical_review/streamlit_app.py`

Expected: page loads, entering note text + selecting an encounter type + clicking "Run Review" produces a rendered comment and the three expandable JSON sections, matching the CLI's output for the same input.

- [ ] **Step 4: Commit**

```bash
git add clinical_review/src/clinical_review/streamlit_app.py
git commit -m "feat(clinical_review): add Streamlit UI"
```

---

### Task 12: Harness adapter

**Files:**
- Create: `harness/src/harness/adapters/clinical_review.py`
- Test: `harness/tests/adapters/test_clinical_review.py`

**Interfaces:**
- Consumes: `trace_schema.RunTrace`, `harness.judge.HallucinationJudge`/`FakeHallucinationJudge`, `harness.scorer.ScoreResult`/`latency_seconds`.
- Produces:
  - `class ClinicalReviewExpected(BaseModel)`: `verdict: Literal["approve", "needs_revision"]`, `expected_check_agents: list[str]`, `min_completeness_issues: int = 0`, `min_compliance_issues: int = 0`.
  - `class ClinicalReviewScorer`: `__init__(self, judge: HallucinationJudge)`, `score(self, trace: RunTrace, expected: ClinicalReviewExpected) -> ScoreResult`.
  - Importable as `from harness.adapters.clinical_review import ClinicalReviewExpected, ClinicalReviewScorer`. This file must have zero imports from `clinical_review` or `agent` (same constraint `pr_review.py` already satisfies) — everything it needs is expressed in plain dicts pulled from `trace.final_output`/`trace.input`.

- [ ] **Step 1: Write the failing tests**

```python
# harness/tests/adapters/test_clinical_review.py
from datetime import datetime, timedelta, timezone

from trace_schema import RunTrace, Step

from harness.adapters.clinical_review import ClinicalReviewExpected, ClinicalReviewScorer
from harness.judge import FakeHallucinationJudge, JudgeVerdict


def _trace(verdict: str, completeness_issues: list[dict], compliance_issues: list[dict], agent_names: list[str], note_text: str = "the note") -> RunTrace:
    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=4)
    steps = [
        Step(agent_name=name, action="analyze_note", started_at=start, ended_at=end, tokens_in=10, tokens_out=5)
        for name in agent_names
    ]
    return RunTrace(
        run_id="r1",
        input={"note_text": note_text, "encounter_type": "new_patient"},
        steps=steps,
        final_output={
            "verdict": verdict,
            "completeness_issues": completeness_issues,
            "compliance_issues": compliance_issues,
            "coding_clarity_issues": [],
        },
        started_at=start,
        ended_at=end,
        total_tokens=sum(s.tokens_in + s.tokens_out for s in steps),
        total_cost_usd=0.0,
    )


def test_task_success_matches_expected_verdict():
    trace = _trace("needs_revision", [], [], ["completeness_review", "compliance_review", "coding_clarity_review", "merge"])
    expected = ClinicalReviewExpected(
        verdict="needs_revision",
        expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is True
    assert result.tool_call_correctness == 1.0


def test_task_success_false_on_verdict_mismatch():
    trace = _trace("approve", [], [], ["completeness_review", "compliance_review", "coding_clarity_review", "merge"])
    expected = ClinicalReviewExpected(
        verdict="needs_revision",
        expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is False


def test_task_success_false_when_min_completeness_issues_not_met():
    trace = _trace(
        "needs_revision",
        [{"element": "Assessment", "description": "vague wording", "severity": "low"}],
        [],
        ["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    expected = ClinicalReviewExpected(
        verdict="needs_revision",
        expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
        min_completeness_issues=2,
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is False


def test_task_success_false_when_min_compliance_issues_not_met():
    trace = _trace(
        "needs_revision",
        [],
        [],
        ["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    expected = ClinicalReviewExpected(
        verdict="needs_revision",
        expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
        min_compliance_issues=1,
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is False


def test_tool_call_correctness_partial_when_agent_missing():
    trace = _trace("approve", [], [], ["completeness_review", "compliance_review"])
    expected = ClinicalReviewExpected(
        verdict="approve",
        expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.tool_call_correctness == 0.5


def test_hallucination_rate_uses_judge_across_all_issue_types():
    trace = _trace(
        "needs_revision",
        [{"element": "ROS", "description": "fabricated gap", "severity": "medium"}],
        [],
        ["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    expected = ClinicalReviewExpected(
        verdict="needs_revision",
        expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=True, reasoning="not in note"))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.hallucination_rate == 1.0


def test_hallucination_rate_zero_when_no_issues_reported():
    trace = _trace("approve", [], [], ["completeness_review", "compliance_review", "coding_clarity_review", "merge"])
    expected = ClinicalReviewExpected(
        verdict="approve",
        expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=True, reasoning="unused"))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.hallucination_rate == 0.0


def test_crashed_trace_fails_task_success_and_flags_crash_in_details():
    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=1)
    trace = RunTrace(
        run_id="r1",
        input={"note_text": "the note", "encounter_type": "new_patient"},
        steps=[],
        final_output={"error": "boom"},
        started_at=start,
        ended_at=end,
        total_tokens=0,
        total_cost_usd=0.0,
    )
    expected = ClinicalReviewExpected(
        verdict="approve", expected_check_agents=["completeness_review", "compliance_review", "coding_clarity_review", "merge"]
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=True, reasoning="unused"))
    scorer = ClinicalReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is False
    assert result.details["crashed"] is True
    assert result.details["error"] == "boom"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest harness/tests/adapters/test_clinical_review.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.adapters.clinical_review'`

- [ ] **Step 3: Write the implementation**

```python
# harness/src/harness/adapters/clinical_review.py
from typing import Literal

from pydantic import BaseModel

from trace_schema import RunTrace

from harness.judge import HallucinationJudge
from harness.scorer import ScoreResult, latency_seconds


class ClinicalReviewExpected(BaseModel):
    verdict: Literal["approve", "needs_revision"]
    expected_check_agents: list[str]
    min_completeness_issues: int = 0
    min_compliance_issues: int = 0


class ClinicalReviewScorer:
    def __init__(self, judge: HallucinationJudge):
        self._judge = judge

    def score(self, trace: RunTrace, expected: ClinicalReviewExpected) -> ScoreResult:
        final = trace.final_output or {}

        if final.get("error"):
            return ScoreResult(
                task_success=False,
                tool_call_correctness=0.0,
                hallucination_rate=0.0,
                latency_seconds=latency_seconds(trace),
                cost_usd=trace.total_cost_usd,
                details={"crashed": True, "error": final["error"]},
            )

        completeness_issues = final.get("completeness_issues", [])
        compliance_issues = final.get("compliance_issues", [])
        coding_clarity_issues = final.get("coding_clarity_issues", [])

        task_success = final.get("verdict") == expected.verdict
        if expected.min_completeness_issues > 0:
            task_success = task_success and len(completeness_issues) >= expected.min_completeness_issues
        if expected.min_compliance_issues > 0:
            task_success = task_success and len(compliance_issues) >= expected.min_compliance_issues

        actual_agents = {s.agent_name for s in trace.steps}
        expected_agents = set(expected.expected_check_agents)
        tool_call_correctness = (
            len(expected_agents & actual_agents) / len(expected_agents) if expected_agents else 1.0
        )

        note_text = (trace.input or {}).get("note_text", "")
        all_issues = completeness_issues + compliance_issues + coding_clarity_issues
        if all_issues:
            hallucinated_count = 0
            for issue in all_issues:
                verdict = self._judge.judge(source_material=note_text, claim=issue.get("description", ""))
                if verdict.hallucinated:
                    hallucinated_count += 1
            hallucination_rate = hallucinated_count / len(all_issues)
        else:
            hallucination_rate = 0.0

        return ScoreResult(
            task_success=task_success,
            tool_call_correctness=tool_call_correctness,
            hallucination_rate=hallucination_rate,
            latency_seconds=latency_seconds(trace),
            cost_usd=trace.total_cost_usd,
            details={"actual_verdict": final.get("verdict")},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest harness/tests/adapters/test_clinical_review.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add harness/src/harness/adapters/clinical_review.py harness/tests/adapters/test_clinical_review.py
git commit -m "feat(harness): add clinical_review scoring adapter"
```

---

### Task 13: `harness gate-clinical-review` CLI command

**Files:**
- Modify: `harness/src/harness/cli.py`
- Test: `harness/tests/test_cli.py`

**Interfaces:**
- Consumes: `ClinicalReviewExpected`/`ClinicalReviewScorer` (Task 12), `clinical_review.graph.run_review`/`ReviewFailedError` and `clinical_review.llm.OllamaClient` (Tasks 2, 8) imported at call time via try/except `ImportError`, `harness.aggregator.aggregate`, `harness.history.{HistoryEntry, load_history, append_history}`, `harness.gate.check_gate`, `harness.report.render_html_report` (all pre-existing, unchanged).
- Produces: a new `gate-clinical-review` command on the existing `harness` CLI group: `harness gate-clinical-review --testcases-dir <dir> --history-path <path> --report-out <path> --commit-sha <sha>`. The pre-existing `gate` command and `_run_case` function are not modified.

- [ ] **Step 1: Write the failing test**

Append to `harness/tests/test_cli.py`:

```python
def test_gate_clinical_review_command_passes_with_no_prior_history(tmp_path, monkeypatch):
    testcases_dir = tmp_path / "clinical_testcases"
    case_dir = testcases_dir / "clean_new_patient"
    case_dir.mkdir(parents=True)
    (case_dir / "input.json").write_text(
        json.dumps({"note_text": "CHIEF COMPLAINT: cough.\nASSESSMENT: viral illness.\nPLAN: rest.", "encounter_type": "new_patient"})
    )
    (case_dir / "expected.json").write_text(
        json.dumps(
            {
                "verdict": "approve",
                "expected_check_agents": ["completeness_review"],
                "min_completeness_issues": 0,
                "min_compliance_issues": 0,
            }
        )
    )

    history_path = tmp_path / "clinical_history.json"
    report_out = tmp_path / "clinical_report.html"

    def fake_run_clinical_case(note_text, encounter_type):
        from datetime import datetime, timezone

        from trace_schema import RunTrace

        now = datetime.now(timezone.utc)
        return RunTrace(
            run_id="r1",
            input={"note_text": note_text, "encounter_type": encounter_type},
            steps=[
                {
                    "agent_name": "completeness_review",
                    "action": "analyze_note",
                    "started_at": now.isoformat(),
                    "ended_at": now.isoformat(),
                    "tool_calls": [],
                    "output": None,
                    "tokens_in": 1,
                    "tokens_out": 1,
                    "cost_usd": 0.0,
                }
            ],
            final_output={
                "verdict": "approve",
                "completeness_issues": [],
                "compliance_issues": [],
                "coding_clarity_issues": [],
            },
            started_at=now,
            ended_at=now,
            total_tokens=2,
            total_cost_usd=0.0,
        )

    monkeypatch.setattr("harness.cli._run_clinical_case", lambda note_text, encounter_type: fake_run_clinical_case(note_text, encounter_type))
    monkeypatch.setattr(
        "harness.cli.HallucinationJudge",
        lambda *a, **k: type(
            "J", (), {"judge": staticmethod(lambda **kw: __import__("harness.judge", fromlist=["JudgeVerdict"]).JudgeVerdict(hallucinated=False, reasoning=""))}
        )(),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "gate-clinical-review",
            "--testcases-dir",
            str(testcases_dir),
            "--history-path",
            str(history_path),
            "--report-out",
            str(report_out),
            "--commit-sha",
            "abc123",
        ],
    )

    assert result.exit_code == 0, result.output
    assert report_out.exists()
    assert history_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest harness/tests/test_cli.py -v -k clinical_review`
Expected: FAIL with `AssertionError` (no such command `gate-clinical-review`) or a click usage error.

- [ ] **Step 3: Write the implementation**

Add to `harness/src/harness/cli.py`, after the existing `_run_case` function and before the existing `@cli.command()` for `gate` (imports go at the top alongside the existing ones):

```python
from harness.adapters.clinical_review import ClinicalReviewExpected, ClinicalReviewScorer


def _run_clinical_case(note_text: str, encounter_type: str) -> RunTrace:
    try:
        from clinical_review.graph import ReviewFailedError, run_review
        from clinical_review.llm import OllamaClient
    except ImportError:
        raise click.ClickException(
            "The 'clinical_review' package is required to run 'harness gate-clinical-review' — "
            "install it as part of this uv workspace (uv sync from the repo root)."
        ) from None

    try:
        _, trace = run_review(note_text=note_text, encounter_type=encounter_type, client=OllamaClient())
    except ReviewFailedError as exc:
        click.echo(f"WARNING: case failed with error: {exc}", err=True)
        return exc.trace
    return trace


@cli.command(name="gate-clinical-review")
@click.option("--testcases-dir", required=True, type=click.Path(exists=True))
@click.option("--history-path", required=True, type=click.Path())
@click.option("--report-out", required=True, type=click.Path())
@click.option("--commit-sha", required=True)
def gate_clinical_review(testcases_dir, history_path, report_out, commit_sha):
    judge = HallucinationJudge()
    scorer = ClinicalReviewScorer(judge=judge)

    results = []
    per_case = []
    for case_dir in sorted(Path(testcases_dir).iterdir()):
        if not case_dir.is_dir():
            continue
        input_data = json.loads((case_dir / "input.json").read_text())
        expected_data = json.loads((case_dir / "expected.json").read_text())
        expected = ClinicalReviewExpected.model_validate(expected_data)

        trace = _run_clinical_case(input_data["note_text"], input_data["encounter_type"])
        result = scorer.score(trace, expected)
        results.append(result)
        per_case.append(
            {
                "name": case_dir.name,
                "task_success": result.task_success,
                "hallucination_rate": result.hallucination_rate,
                "latency_seconds": result.latency_seconds,
            }
        )

    current = aggregate(results)
    history = load_history(history_path)
    gate_result = check_gate(current, history)

    entry = HistoryEntry(commit_sha=commit_sha, timestamp=datetime.now(timezone.utc).isoformat(), report=current)
    append_history(history_path, entry)

    html = render_html_report(current, load_history(history_path), per_case=per_case)
    Path(report_out).write_text(html)

    click.echo(gate_result.reason)
    if not gate_result.passed:
        raise SystemExit(1)
```

Note: `cli`, `Path`, `json`, `datetime`, `timezone`, `click`, `RunTrace`, `aggregate`, `HistoryEntry`, `load_history`, `append_history`, `check_gate`, `render_html_report`, `HallucinationJudge` are all already imported/defined earlier in `harness/cli.py` — only the `from harness.adapters.clinical_review import ...` line is a new import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest harness/tests/test_cli.py -v`
Expected: all tests pass, including both the pre-existing `gate` test and the new `gate-clinical-review` test.

- [ ] **Step 5: Commit**

```bash
git add harness/src/harness/cli.py harness/tests/test_cli.py
git commit -m "feat(harness): add gate-clinical-review CLI command"
```

---

### Task 14: `clinical_testcases/` synthetic fixtures

**Files:**
- Create: `clinical_testcases/clean_new_patient/input.json`
- Create: `clinical_testcases/clean_new_patient/expected.json`
- Create: `clinical_testcases/incomplete_new_patient/input.json`
- Create: `clinical_testcases/incomplete_new_patient/expected.json`
- Create: `clinical_testcases/clean_follow_up/input.json`
- Create: `clinical_testcases/clean_follow_up/expected.json`
- Create: `clinical_testcases/mismatched_follow_up/input.json`
- Create: `clinical_testcases/mismatched_follow_up/expected.json`
- Test: `harness/tests/test_clinical_testcases_shape.py`

**Interfaces:**
- Consumes: `ClinicalReviewExpected` (Task 12).
- Produces: 4 fixture directories, each with `input.json` (`{"note_text": str, "encounter_type": "new_patient"|"follow_up"}`) and `expected.json` (validates against `ClinicalReviewExpected`).

- [ ] **Step 1: Write the failing test**

```python
# harness/tests/test_clinical_testcases_shape.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest harness/tests/test_clinical_testcases_shape.py -v`
Expected: FAIL — `clinical_testcases/` does not exist yet.

- [ ] **Step 3: Write the fixtures**

`clinical_testcases/clean_new_patient/input.json`:
```json
{
  "encounter_type": "new_patient",
  "note_text": "CHIEF COMPLAINT: Sore throat and low-grade fever for 3 days.\n\nHISTORY OF PRESENT ILLNESS: 29-year-old presents with a 3-day history of sore throat, low-grade fever, and mild difficulty swallowing. Denies cough or shortness of breath.\n\nREVIEW OF SYSTEMS: Negative for cough, chest pain, rash, or joint pain. Positive for sore throat and fatigue.\n\nPAST MEDICAL HISTORY: No chronic conditions. No prior surgeries.\n\nVITALS: BP 118/74, HR 82, Temp 100.2F, RR 14, SpO2 99% on room air.\n\nPHYSICAL EXAM: Oropharynx erythematous with mild tonsillar swelling, no exudate. No cervical lymphadenopathy. Lungs clear to auscultation bilaterally.\n\nASSESSMENT: Acute pharyngitis, likely viral.\n\nPLAN: Supportive care with fluids and rest. Return if symptoms worsen or fever persists beyond 5 days."
}
```

`clinical_testcases/clean_new_patient/expected.json`:
```json
{
  "verdict": "approve",
  "expected_check_agents": ["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
  "min_completeness_issues": 0,
  "min_compliance_issues": 0
}
```

`clinical_testcases/incomplete_new_patient/input.json`:
```json
{
  "encounter_type": "new_patient",
  "note_text": "CHIEF COMPLAINT: Sore throat and low-grade fever for 3 days.\n\nHISTORY OF PRESENT ILLNESS: 29-year-old presents with a 3-day history of sore throat, low-grade fever, and mild difficulty swallowing. Denies cough or shortness of breath.\n\nVITALS: BP 118/74, HR 82, Temp 100.2F, RR 14, SpO2 99% on room air.\n\nPHYSICAL EXAM: Oropharynx erythematous with mild tonsillar swelling, no exudate. No cervical lymphadenopathy. Lungs clear to auscultation bilaterally.\n\nASSESSMENT: Acute pharyngitis, likely viral.\n\nPLAN: Supportive care with fluids and rest. Return if symptoms worsen or fever persists beyond 5 days."
}
```

`clinical_testcases/incomplete_new_patient/expected.json`:
```json
{
  "verdict": "needs_revision",
  "expected_check_agents": ["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
  "min_completeness_issues": 2,
  "min_compliance_issues": 0
}
```

`clinical_testcases/clean_follow_up/input.json`:
```json
{
  "encounter_type": "follow_up",
  "note_text": "REASON FOR VISIT: Follow-up for hypertension management, 6 weeks after starting lisinopril 10mg daily.\n\nINTERVAL HISTORY: Patient reports good medication tolerance, no dizziness, no cough, no swelling. Adherent to daily dosing. Home blood pressure readings averaging 128/80.\n\nVITALS: BP 126/78, HR 74, Temp 98.2F, RR 14.\n\nPHYSICAL EXAM: No lower extremity edema. Heart regular rate and rhythm, no murmurs. Lungs clear bilaterally.\n\nASSESSMENT: Hypertension, well-controlled on current regimen.\n\nPLAN: Continue lisinopril 10mg daily. Recheck basic metabolic panel in 3 months. Follow up in 3 months or sooner if symptoms arise."
}
```

`clinical_testcases/clean_follow_up/expected.json`:
```json
{
  "verdict": "approve",
  "expected_check_agents": ["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
  "min_completeness_issues": 0,
  "min_compliance_issues": 0
}
```

`clinical_testcases/mismatched_follow_up/input.json`:
```json
{
  "encounter_type": "follow_up",
  "note_text": "REASON FOR VISIT: Follow-up for hypertension management, 6 weeks after starting lisinopril 10mg daily.\n\nINTERVAL HISTORY: Patient reports good medication tolerance, no dizziness, no cough, no swelling. Adherent to daily dosing.\n\nVITALS: BP 122/76, HR 72, Temp 98.2F, RR 14.\n\nPHYSICAL EXAM: No lower extremity edema. Heart regular rate and rhythm, no murmurs. Lungs clear bilaterally. No abnormal findings noted.\n\nASSESSMENT: Poorly controlled hypertension requiring urgent escalation of therapy.\n\nPLAN: Add second antihypertensive agent and schedule urgent cardiology referral."
}
```

`clinical_testcases/mismatched_follow_up/expected.json`:
```json
{
  "verdict": "needs_revision",
  "expected_check_agents": ["completeness_review", "compliance_review", "coding_clarity_review", "merge"],
  "min_completeness_issues": 0,
  "min_compliance_issues": 1
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest harness/tests/test_clinical_testcases_shape.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add clinical_testcases/ harness/tests/test_clinical_testcases_shape.py
git commit -m "test(clinical_review): add synthetic clinical note test suite"
```

---

### Task 15: Real end-to-end run + README

**Files:**
- Create: `docs/results/clinical_report.html` (output of Step 2 below, committed as evidence)
- Create: `docs/results/clinical_history.json` (output of Step 2 below, committed as evidence)
- Modify: `README.md`
- Modify: `.gitignore` (mirror the existing PR-review pattern for the new report/history filenames)

**Interfaces:**
- Consumes: everything from Tasks 1–14.
- Produces: no new code — a real recorded run against live Ollama, plus documentation.

- [ ] **Step 1: Sync the full workspace**

Run: `uv sync --extra ui`
Expected: completes without error; `clinical_review`, `agent`, `harness`, `trace_schema` all importable.

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -q -m "not integration"`
Expected: all tests pass (existing + all new `clinical_review`/`harness` tests from Tasks 1–14), zero failures.

- [ ] **Step 3: Run a real `harness gate-clinical-review` against live Ollama**

Run:
```bash
ollama pull qwen2.5:7b
uv run harness gate-clinical-review \
  --testcases-dir clinical_testcases \
  --history-path docs/results/clinical_history.json \
  --report-out docs/results/clinical_report.html \
  --commit-sha "$(git rev-parse HEAD)"
```
Expected: completes, prints a gate reason (pass/fail against no-prior-history baseline), and writes both output files with real, non-placeholder numbers.

- [ ] **Step 4: Update `.gitignore`**

Add two lines (mirroring the existing `/history.json` / `/report.html` pattern for uncommitted local runs — the `docs/results/` copies are committed intentionally, so this only ignores stray root-level runs):
```
/clinical_history.json
/clinical_report.html
```

- [ ] **Step 5: Update `README.md`**

Add a new section after the existing "Running the eval harness" section:

```markdown
### Clinical documentation review agent (`clinical_review/`)

A second, independent agent proving the `trace_schema`/`harness` design
generalizes across domains without modification. Reviews a clinical note's
*documentation quality* — completeness, compliance, coding clarity — not
its medical correctness.

**This is an illustrative portfolio demo, not a clinical tool.** Its review
rules are example documentation-quality checks, not a real coding/compliance
authority, and it must never be used for actual patient care or clinical
decision-making.

```bash
uv run clinical-review review --note-file <path> --encounter-type {new_patient,follow_up}
```

Streamlit UI:
```bash
uv sync --extra ui
uv run streamlit run clinical_review/src/clinical_review/streamlit_app.py
```

Eval harness (separate from the PR-review gate — a different domain's pass
rate is a different number):
```bash
uv run harness gate-clinical-review --testcases-dir clinical_testcases \
  --history-path docs/results/clinical_history.json \
  --report-out docs/results/clinical_report.html --commit-sha "$(git rev-parse HEAD)"
```

See `docs/results/clinical_report.html` for an example of the harness's
output on this second domain.
```

- [ ] **Step 6: Commit**

```bash
git add docs/results/clinical_report.html docs/results/clinical_history.json README.md .gitignore
git commit -m "docs: add real clinical_review harness run and README section"
```

---

## Definition of Done

- [ ] `uv run pytest -q -m "not integration"` passes with all tests green, including every `clinical_review/tests/` and new `harness/tests/` file.
- [ ] `clinical_review/pyproject.toml` has no dependency on `agent` or `harness`.
- [ ] `grep -r "from agent" clinical_review/src harness/src/harness/adapters/clinical_review.py` returns nothing, and `grep -r "from clinical_review" agent/src` returns nothing.
- [ ] `harness gate` (PR-review, pre-existing) still passes unmodified — `gate-clinical-review` is additive only.
- [ ] A real `harness gate-clinical-review` run against live Ollama is committed under `docs/results/`.
- [ ] README documents the new package, its CLI, its Streamlit UI, and the illustrative-rules disclaimer.
- [ ] All commits made; `git log --oneline` shows one commit per task.
