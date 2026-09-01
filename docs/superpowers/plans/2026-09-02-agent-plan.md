# PR Review Agent (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `agent` package — a LangGraph multi-agent PR reviewer (security / style / test-coverage / merge) that emits a `trace_schema.RunTrace` for every run, exposed as both a CLI and a composite GitHub Action.

**Architecture:** A LangGraph graph with a `supervisor` entry node fanning out in parallel to three independent review nodes, each calling a local Ollama model (`qwen2.5:7b`) for structured JSON output; a `merge` node combines their findings into a final verdict. A shared `tracing.py` helper wraps every LLM call so trace-writing isn't duplicated per node. The CLI is the primary tested entry point; `action.yml` wraps it for GitHub Actions.

**Tech Stack:** Python 3.12, `uv`, LangGraph, `ollama` Python client, `pydantic` v2, `jinja2`, `pytest`, `click` (CLI).

**Spec:** `docs/superpowers/specs/2026-09-01-pr-review-eval-design.md`

**Depends on:** `docs/superpowers/plans/2026-09-01-trace-schema-plan.md` (must be complete — `trace_schema` package must be importable).

## Global Constraints

- Python 3.12 only, part of the same `uv` workspace as `trace_schema`.
- `agent` depends on `trace_schema` as a workspace path dependency; never vendors it.
- Model: Ollama `qwen2.5:7b`, called via `ollama.chat(..., format=<pydantic model>.model_json_schema())` for structured output.
- Every LLM call must go through `agent.tracing.traced_call(...)` — no node calls Ollama directly.
- No live Ollama calls in unit tests — all node unit tests use a stub/fake client. Exactly one integration test (marked `@pytest.mark.integration`) exercises the full graph against a real local Ollama.
- `qwen2.5:7b` must already be pulled locally (`ollama pull qwen2.5:7b`) before running the integration test — this plan does not manage model downloads.

---

### Task 1: `agent` package scaffold + Ollama client wrapper

**Files:**
- Create: `agent/pyproject.toml`
- Create: `agent/src/agent/__init__.py`
- Create: `agent/src/agent/llm.py`
- Create: `agent/tests/__init__.py`
- Test: `agent/tests/test_llm.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `agent.llm.OllamaClient` with method `chat(system: str, user: str, response_model: type[pydantic.BaseModel]) -> tuple[pydantic.BaseModel, LlmUsage]`, where `LlmUsage` is a dataclass `{tokens_in: int, tokens_out: int}`. Also `agent.llm.FakeOllamaClient` (test double) with the same interface, constructed with a canned response.

- [ ] **Step 1: Create `agent/pyproject.toml`**

```toml
[project]
name = "agent"
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
packages = ["src/agent"]

[tool.pytest.ini_options]
markers = ["integration: requires a running local Ollama"]
```

- [ ] **Step 2: Create empty package files**

`agent/src/agent/__init__.py`:
```python
```

`agent/tests/__init__.py`:
```python
```

- [ ] **Step 3: Write the failing test**

```python
# agent/tests/test_llm.py
from pydantic import BaseModel

from agent.llm import FakeOllamaClient


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

- [ ] **Step 4: Run test to verify it fails**

Run: `uv sync && uv run pytest agent/tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.llm'`

- [ ] **Step 5: Write the implementation**

```python
# agent/src/agent/llm.py
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
            tokens_in=response.get("prompt_eval_count", 0),
            tokens_out=response.get("eval_count", 0),
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

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest agent/tests/test_llm.py -v`
Expected: 1 passed

- [ ] **Step 7: Commit**

```bash
git add agent/pyproject.toml agent/src/agent/__init__.py agent/src/agent/llm.py agent/tests/__init__.py agent/tests/test_llm.py
git commit -m "chore(agent): scaffold package and Ollama client wrapper"
```

---

### Task 2: Shared tracing helper

**Files:**
- Create: `agent/src/agent/tracing.py`
- Test: `agent/tests/test_tracing.py`

**Interfaces:**
- Consumes: `trace_schema.Step`, `trace_schema.ToolCall`; `agent.llm.LlmUsage` from Task 1.
- Produces: `agent.tracing.TraceRecorder` class with:
  - `TraceRecorder()` — holds an internal `list[Step]`.
  - `record(agent_name: str, action: str, fn: Callable[[], tuple[Any, LlmUsage]], tool_calls: list[ToolCall] | None = None) -> Any` — times `fn()`, appends a `Step` capturing timing/usage/output/tool_calls, returns `fn()`'s first element (the parsed output).
  - `.steps -> list[Step]` property returning the recorded steps.
  - Cost is computed as `0.0` for now (local model, no billing) — `cost_usd=0.0` on every `Step`.

- [ ] **Step 1: Write the failing test**

```python
# agent/tests/test_tracing.py
from trace_schema import ToolCall

from agent.llm import LlmUsage
from agent.tracing import TraceRecorder


def test_record_appends_step_with_timing_and_usage():
    recorder = TraceRecorder()

    def fake_call():
        return {"ok": True}, LlmUsage(tokens_in=5, tokens_out=2)

    output = recorder.record("security_review", "analyze_diff", fake_call)

    assert output == {"ok": True}
    assert len(recorder.steps) == 1
    step = recorder.steps[0]
    assert step.agent_name == "security_review"
    assert step.action == "analyze_diff"
    assert step.tokens_in == 5
    assert step.tokens_out == 2
    assert step.output == {"ok": True}
    assert step.cost_usd == 0.0
    assert step.started_at <= step.ended_at


def test_record_stores_tool_calls():
    recorder = TraceRecorder()

    def fake_call():
        return "result", LlmUsage(tokens_in=1, tokens_out=1)

    tool_calls = [ToolCall(name="grep", args={"pattern": "eval("}, result=[])]
    recorder.record("security_review", "search", fake_call, tool_calls=tool_calls)

    assert recorder.steps[0].tool_calls == tool_calls
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest agent/tests/test_tracing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.tracing'`

- [ ] **Step 3: Write the implementation**

```python
# agent/src/agent/tracing.py
from datetime import datetime, timezone
from typing import Any, Callable

from trace_schema import Step, ToolCall

from agent.llm import LlmUsage


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
        self._steps.append(
            Step(
                agent_name=agent_name,
                action=action,
                tool_calls=tool_calls or [],
                output=output if isinstance(output, (dict, list, str, int, float, bool, type(None))) else str(output),
                started_at=started_at,
                ended_at=ended_at,
                tokens_in=usage.tokens_in,
                tokens_out=usage.tokens_out,
                cost_usd=0.0,
            )
        )
        return output
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest agent/tests/test_tracing.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add agent/src/agent/tracing.py agent/tests/test_tracing.py
git commit -m "feat(agent): add shared TraceRecorder helper"
```

---

### Task 3: Findings and verdict models

**Files:**
- Create: `agent/src/agent/schemas.py`
- Test: `agent/tests/test_schemas.py`

**Interfaces:**
- Consumes: `pydantic.BaseModel`.
- Produces:
  - `class Issue(BaseModel)`: `file: str`, `line: int | None`, `description: str`, `severity: Literal["low", "medium", "high", "critical"]`.
  - `class SecurityFindings(BaseModel)`: `issues: list[Issue]`.
  - `class StyleFindings(BaseModel)`: `issues: list[Issue]`.
  - `class TestCoverageFindings(BaseModel)`: `has_gap: bool`, `explanation: str`.
  - `class Verdict(BaseModel)`: `security_issues: list[Issue]`, `style_issues: list[Issue]`, `test_coverage_gap: bool`, `verdict: Literal["approve", "request_changes"]`.

- [ ] **Step 1: Write the failing test**

```python
# agent/tests/test_schemas.py
from agent.schemas import Issue, SecurityFindings, StyleFindings, TestCoverageFindings, Verdict


def test_issue_requires_severity_enum():
    issue = Issue(file="app.py", line=10, description="SQL injection", severity="critical")
    assert issue.severity == "critical"


def test_verdict_shape():
    v = Verdict(
        security_issues=[Issue(file="a.py", line=1, description="x", severity="low")],
        style_issues=[],
        test_coverage_gap=True,
        verdict="request_changes",
    )
    assert v.verdict == "request_changes"
    assert len(v.security_issues) == 1


def test_findings_models_hold_issue_lists():
    sec = SecurityFindings(issues=[])
    style = StyleFindings(issues=[])
    cov = TestCoverageFindings(has_gap=False, explanation="tests present")
    assert sec.issues == []
    assert style.issues == []
    assert cov.has_gap is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest agent/tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.schemas'`

- [ ] **Step 3: Write the implementation**

```python
# agent/src/agent/schemas.py
from typing import Literal

from pydantic import BaseModel


class Issue(BaseModel):
    file: str
    line: int | None = None
    description: str
    severity: Literal["low", "medium", "high", "critical"]


class SecurityFindings(BaseModel):
    issues: list[Issue] = []


class StyleFindings(BaseModel):
    issues: list[Issue] = []


class TestCoverageFindings(BaseModel):
    has_gap: bool
    explanation: str = ""


class Verdict(BaseModel):
    security_issues: list[Issue] = []
    style_issues: list[Issue] = []
    test_coverage_gap: bool
    verdict: Literal["approve", "request_changes"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest agent/tests/test_schemas.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add agent/src/agent/schemas.py agent/tests/test_schemas.py
git commit -m "feat(agent): add findings and verdict pydantic schemas"
```

---

### Task 4: Review nodes (security, style, test_coverage)

**Files:**
- Create: `agent/src/agent/nodes/__init__.py`
- Create: `agent/src/agent/nodes/security.py`
- Create: `agent/src/agent/nodes/style.py`
- Create: `agent/src/agent/nodes/test_coverage.py`
- Test: `agent/tests/nodes/__init__.py`
- Test: `agent/tests/nodes/test_security.py`
- Test: `agent/tests/nodes/test_style.py`
- Test: `agent/tests/nodes/test_test_coverage.py`

**Interfaces:**
- Consumes: `agent.llm.OllamaClient`-shaped client (duck-typed `.chat(system, user, response_model)`), `agent.tracing.TraceRecorder` from Task 2, `agent.schemas.SecurityFindings` / `StyleFindings` / `TestCoverageFindings` from Task 3.
- Produces:
  - `agent.nodes.security.review_security(diff: str, client, recorder: TraceRecorder) -> SecurityFindings`
  - `agent.nodes.style.review_style(diff: str, client, recorder: TraceRecorder) -> StyleFindings`
  - `agent.nodes.test_coverage.review_test_coverage(diff: str, client, recorder: TraceRecorder) -> TestCoverageFindings`
  - Each function calls `recorder.record(<agent_name>, "analyze_diff", lambda: client.chat(system=..., user=diff, response_model=<Findings type>))`.

- [ ] **Step 1: Create empty test package file**

`agent/tests/nodes/__init__.py`:
```python
```

- [ ] **Step 2: Write the failing tests**

```python
# agent/tests/nodes/test_security.py
from agent.llm import FakeOllamaClient
from agent.nodes.security import review_security
from agent.schemas import Issue, SecurityFindings
from agent.tracing import TraceRecorder


def test_review_security_returns_findings_and_records_step():
    canned = SecurityFindings(
        issues=[Issue(file="app.py", line=42, description="SQL injection via string concat", severity="critical")]
    )
    client = FakeOllamaClient(response=canned, tokens_in=100, tokens_out=30)
    recorder = TraceRecorder()

    result = review_security(diff="- old\n+ query = f'SELECT * FROM x WHERE id={id}'", client=client, recorder=recorder)

    assert result == canned
    assert len(recorder.steps) == 1
    assert recorder.steps[0].agent_name == "security_review"
```

```python
# agent/tests/nodes/test_style.py
from agent.llm import FakeOllamaClient
from agent.nodes.style import review_style
from agent.schemas import StyleFindings
from agent.tracing import TraceRecorder


def test_review_style_returns_findings_and_records_step():
    canned = StyleFindings(issues=[])
    client = FakeOllamaClient(response=canned, tokens_in=80, tokens_out=10)
    recorder = TraceRecorder()

    result = review_style(diff="+ x=1", client=client, recorder=recorder)

    assert result == canned
    assert recorder.steps[0].agent_name == "style_review"
```

```python
# agent/tests/nodes/test_test_coverage.py
from agent.llm import FakeOllamaClient
from agent.nodes.test_coverage import review_test_coverage
from agent.schemas import TestCoverageFindings
from agent.tracing import TraceRecorder


def test_review_test_coverage_returns_findings_and_records_step():
    canned = TestCoverageFindings(has_gap=True, explanation="no new tests for changed function")
    client = FakeOllamaClient(response=canned, tokens_in=90, tokens_out=20)
    recorder = TraceRecorder()

    result = review_test_coverage(diff="+ def foo(): ...", client=client, recorder=recorder)

    assert result == canned
    assert recorder.steps[0].agent_name == "test_coverage"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest agent/tests/nodes/ -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.nodes'`

- [ ] **Step 4: Write the implementations**

```python
# agent/src/agent/nodes/__init__.py
```

```python
# agent/src/agent/nodes/security.py
from agent.schemas import SecurityFindings
from agent.tracing import TraceRecorder

SYSTEM_PROMPT = (
    "You are a security reviewer for a pull request diff. Identify security "
    "vulnerabilities such as SQL injection, XSS, hardcoded secrets, unsafe "
    "deserialization, and command injection. Only report issues clearly "
    "supported by the diff content. Respond with structured findings."
)


def review_security(diff: str, client, recorder: TraceRecorder) -> SecurityFindings:
    return recorder.record(
        "security_review",
        "analyze_diff",
        lambda: client.chat(system=SYSTEM_PROMPT, user=diff, response_model=SecurityFindings),
    )
```

```python
# agent/src/agent/nodes/style.py
from agent.schemas import StyleFindings
from agent.tracing import TraceRecorder

SYSTEM_PROMPT = (
    "You are a code style reviewer for a pull request diff. Identify style "
    "issues: naming, formatting, dead code, overly long functions. Only "
    "report issues clearly supported by the diff content."
)


def review_style(diff: str, client, recorder: TraceRecorder) -> StyleFindings:
    return recorder.record(
        "style_review",
        "analyze_diff",
        lambda: client.chat(system=SYSTEM_PROMPT, user=diff, response_model=StyleFindings),
    )
```

```python
# agent/src/agent/nodes/test_coverage.py
from agent.schemas import TestCoverageFindings
from agent.tracing import TraceRecorder

SYSTEM_PROMPT = (
    "You review a pull request diff for test coverage. Determine whether "
    "new or changed logic lacks corresponding test changes in the same "
    "diff. Respond with has_gap and a short explanation."
)


def review_test_coverage(diff: str, client, recorder: TraceRecorder) -> TestCoverageFindings:
    return recorder.record(
        "test_coverage",
        "analyze_diff",
        lambda: client.chat(system=SYSTEM_PROMPT, user=diff, response_model=TestCoverageFindings),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest agent/tests/nodes/ -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add agent/src/agent/nodes/ agent/tests/nodes/
git commit -m "feat(agent): add security, style, test_coverage review nodes"
```

---

### Task 5: Merge (critic) node

**Files:**
- Create: `agent/src/agent/nodes/merge.py`
- Test: `agent/tests/nodes/test_merge.py`

**Interfaces:**
- Consumes: `SecurityFindings`, `StyleFindings`, `TestCoverageFindings` (Task 3), `Verdict` (Task 3), `TraceRecorder` (Task 2), an `OllamaClient`-shaped client.
- Produces: `agent.nodes.merge.merge_findings(security: SecurityFindings, style: StyleFindings, coverage: TestCoverageFindings, client, recorder: TraceRecorder) -> Verdict`. The critic LLM call decides `verdict` ("approve" vs "request_changes"); `security_issues`/`style_issues`/`test_coverage_gap` are carried through deterministically from the inputs (not re-derived by the LLM), so the merge node's LLM call only needs to output `{"verdict": "..."}`. To keep `Verdict` as the single response schema, the merge node passes a summarized prompt and validates only the `verdict` field from the LLM response, then constructs the full `Verdict` itself.

- [ ] **Step 1: Write the failing test**

```python
# agent/tests/nodes/test_merge.py
from agent.llm import FakeOllamaClient
from agent.nodes.merge import merge_findings
from agent.schemas import Issue, SecurityFindings, StyleFindings, TestCoverageFindings, Verdict
from agent.tracing import TraceRecorder


def test_merge_findings_combines_inputs_and_uses_llm_verdict():
    security = SecurityFindings(issues=[Issue(file="a.py", line=1, description="sqli", severity="critical")])
    style = StyleFindings(issues=[])
    coverage = TestCoverageFindings(has_gap=False, explanation="covered")
    canned_verdict = Verdict(
        security_issues=[], style_issues=[], test_coverage_gap=False, verdict="request_changes"
    )
    client = FakeOllamaClient(response=canned_verdict, tokens_in=50, tokens_out=5)
    recorder = TraceRecorder()

    result = merge_findings(security=security, style=style, coverage=coverage, client=client, recorder=recorder)

    assert result.verdict == "request_changes"
    assert result.security_issues == security.issues
    assert result.style_issues == style.issues
    assert result.test_coverage_gap is False
    assert recorder.steps[0].agent_name == "merge"


def test_merge_findings_approves_when_no_issues():
    security = SecurityFindings(issues=[])
    style = StyleFindings(issues=[])
    coverage = TestCoverageFindings(has_gap=False, explanation="covered")
    canned_verdict = Verdict(security_issues=[], style_issues=[], test_coverage_gap=False, verdict="approve")
    client = FakeOllamaClient(response=canned_verdict)
    recorder = TraceRecorder()

    result = merge_findings(security=security, style=style, coverage=coverage, client=client, recorder=recorder)

    assert result.verdict == "approve"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest agent/tests/nodes/test_merge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.nodes.merge'`

- [ ] **Step 3: Write the implementation**

```python
# agent/src/agent/nodes/merge.py
from agent.schemas import SecurityFindings, StyleFindings, TestCoverageFindings, Verdict
from agent.tracing import TraceRecorder

SYSTEM_PROMPT = (
    "You are the supervising critic for a pull request review. Given a "
    "summary of security, style, and test-coverage findings, decide the "
    "final verdict: 'approve' if there are no critical/high security "
    "issues and no test coverage gap, otherwise 'request_changes'. "
    "Respond with a verdict field only."
)


def _summarize(security: SecurityFindings, style: StyleFindings, coverage: TestCoverageFindings) -> str:
    lines = [
        f"Security issues: {[ (i.severity, i.description) for i in security.issues ]}",
        f"Style issues: {[ i.description for i in style.issues ]}",
        f"Test coverage gap: {coverage.has_gap} ({coverage.explanation})",
    ]
    return "\n".join(lines)


def merge_findings(
    security: SecurityFindings,
    style: StyleFindings,
    coverage: TestCoverageFindings,
    client,
    recorder: TraceRecorder,
) -> Verdict:
    summary = _summarize(security, style, coverage)

    def call():
        parsed, usage = client.chat(system=SYSTEM_PROMPT, user=summary, response_model=Verdict)
        return parsed, usage

    llm_verdict: Verdict = recorder.record("merge", "decide_verdict", call)

    return Verdict(
        security_issues=security.issues,
        style_issues=style.issues,
        test_coverage_gap=coverage.has_gap,
        verdict=llm_verdict.verdict,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest agent/tests/nodes/test_merge.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add agent/src/agent/nodes/merge.py agent/tests/nodes/test_merge.py
git commit -m "feat(agent): add merge/critic node"
```

---

### Task 6: LangGraph graph wiring

**Files:**
- Create: `agent/src/agent/graph.py`
- Test: `agent/tests/test_graph.py`

**Interfaces:**
- Consumes: `review_security`, `review_style`, `review_test_coverage` (Task 4), `merge_findings` (Task 5), `TraceRecorder` (Task 2), an `OllamaClient`-shaped client.
- Produces: `agent.graph.run_review(diff: str, client) -> tuple[Verdict, RunTrace]` — builds and invokes the LangGraph graph, returns the final `Verdict` plus a fully populated `trace_schema.RunTrace` (via `TraceRecorder.steps`, wrapped with `run_id`, `input`, `final_output`, `started_at`/`ended_at`, `total_tokens`, `total_cost_usd` computed from the steps).

- [ ] **Step 1: Write the failing test**

```python
# agent/tests/test_graph.py
from trace_schema import RunTrace

from agent.graph import run_review
from agent.llm import FakeOllamaClient
from agent.schemas import Issue, SecurityFindings, StyleFindings, TestCoverageFindings, Verdict


class SequencedFakeClient:
    """Returns a different canned response per call, in call order."""

    def __init__(self, responses: list):
        self._responses = list(responses)

    def chat(self, system, user, response_model):
        from agent.llm import LlmUsage

        response = self._responses.pop(0)
        return response, LlmUsage(tokens_in=10, tokens_out=5)


def test_run_review_produces_verdict_and_trace():
    security = SecurityFindings(issues=[Issue(file="a.py", line=1, description="sqli", severity="critical")])
    style = StyleFindings(issues=[])
    coverage = TestCoverageFindings(has_gap=True, explanation="no tests")
    merge_llm_output = Verdict(security_issues=[], style_issues=[], test_coverage_gap=False, verdict="request_changes")

    client = SequencedFakeClient([security, style, coverage, merge_llm_output])

    verdict, trace = run_review(diff="+ query = f'SELECT {x}'", client=client)

    assert isinstance(verdict, Verdict)
    assert verdict.verdict == "request_changes"
    assert verdict.test_coverage_gap is True
    assert len(verdict.security_issues) == 1

    assert isinstance(trace, RunTrace)
    assert len(trace.steps) == 4
    agent_names = {s.agent_name for s in trace.steps}
    assert agent_names == {"security_review", "style_review", "test_coverage", "merge"}
    assert trace.final_output["verdict"] == "request_changes"
    assert trace.total_tokens == sum(s.tokens_in + s.tokens_out for s in trace.steps)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest agent/tests/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.graph'`

- [ ] **Step 3: Write the implementation**

```python
# agent/src/agent/graph.py
import uuid
from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from trace_schema import RunTrace

from agent.nodes.merge import merge_findings
from agent.nodes.security import review_security
from agent.nodes.style import review_style
from agent.nodes.test_coverage import review_test_coverage
from agent.schemas import SecurityFindings, StyleFindings, TestCoverageFindings, Verdict
from agent.tracing import TraceRecorder


class ReviewState(TypedDict, total=False):
    diff: str
    security: SecurityFindings
    style: StyleFindings
    coverage: TestCoverageFindings
    verdict: Verdict


def _build_graph(client, recorder: TraceRecorder):
    graph = StateGraph(ReviewState)

    graph.add_node("security_review", lambda s: {"security": review_security(s["diff"], client, recorder)})
    graph.add_node("style_review", lambda s: {"style": review_style(s["diff"], client, recorder)})
    graph.add_node("test_coverage", lambda s: {"coverage": review_test_coverage(s["diff"], client, recorder)})
    graph.add_node(
        "merge",
        lambda s: {
            "verdict": merge_findings(s["security"], s["style"], s["coverage"], client, recorder)
        },
    )

    graph.add_edge(START, "security_review")
    graph.add_edge(START, "style_review")
    graph.add_edge(START, "test_coverage")
    graph.add_edge("security_review", "merge")
    graph.add_edge("style_review", "merge")
    graph.add_edge("test_coverage", "merge")
    graph.add_edge("merge", END)

    return graph.compile()


def run_review(diff: str, client) -> tuple[Verdict, RunTrace]:
    recorder = TraceRecorder()
    compiled = _build_graph(client, recorder)

    started_at = datetime.now(timezone.utc)
    final_state = compiled.invoke({"diff": diff})
    ended_at = datetime.now(timezone.utc)

    verdict: Verdict = final_state["verdict"]
    steps = recorder.steps
    total_tokens = sum(s.tokens_in + s.tokens_out for s in steps)

    trace = RunTrace(
        run_id=str(uuid.uuid4()),
        input={"diff": diff},
        steps=steps,
        final_output=verdict.model_dump(),
        started_at=started_at,
        ended_at=ended_at,
        total_tokens=total_tokens,
        total_cost_usd=sum(s.cost_usd for s in steps),
    )
    return verdict, trace
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest agent/tests/test_graph.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add agent/src/agent/graph.py agent/tests/test_graph.py
git commit -m "feat(agent): wire LangGraph graph and produce RunTrace"
```

---

### Task 7: Markdown rendering

**Files:**
- Create: `agent/src/agent/render.py`
- Test: `agent/tests/test_render.py`

**Interfaces:**
- Consumes: `Verdict` from Task 3.
- Produces: `agent.render.render_comment(verdict: Verdict) -> str` returning a markdown string.

- [ ] **Step 1: Write the failing test**

```python
# agent/tests/test_render.py
from agent.render import render_comment
from agent.schemas import Issue, Verdict


def test_render_comment_includes_verdict_and_issues():
    verdict = Verdict(
        security_issues=[Issue(file="app.py", line=12, description="SQL injection", severity="critical")],
        style_issues=[],
        test_coverage_gap=True,
        verdict="request_changes",
    )
    comment = render_comment(verdict)

    assert "request_changes".upper() in comment.upper() or "Request Changes" in comment
    assert "app.py" in comment
    assert "SQL injection" in comment
    assert "test coverage" in comment.lower()


def test_render_comment_clean_approve():
    verdict = Verdict(security_issues=[], style_issues=[], test_coverage_gap=False, verdict="approve")
    comment = render_comment(verdict)
    assert "approve" in comment.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest agent/tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.render'`

- [ ] **Step 3: Write the implementation**

```python
# agent/src/agent/render.py
from jinja2 import Template

from agent.schemas import Verdict

TEMPLATE = Template(
    """\
## PR Review: {{ "✅ Approve" if verdict.verdict == "approve" else "🚫 Request Changes" }}

{% if verdict.security_issues %}
### Security Issues
{% for issue in verdict.security_issues %}
- **[{{ issue.severity }}]** `{{ issue.file }}{% if issue.line %}:{{ issue.line }}{% endif %}` — {{ issue.description }}
{% endfor %}
{% else %}
### Security Issues
None found.
{% endif %}

{% if verdict.style_issues %}
### Style Issues
{% for issue in verdict.style_issues %}
- **[{{ issue.severity }}]** `{{ issue.file }}{% if issue.line %}:{{ issue.line }}{% endif %}` — {{ issue.description }}
{% endfor %}
{% else %}
### Style Issues
None found.
{% endif %}

### Test Coverage
{{ "⚠️ Gap detected — new/changed logic is missing tests." if verdict.test_coverage_gap else "✅ No gap detected." }}
"""
)


def render_comment(verdict: Verdict) -> str:
    return TEMPLATE.render(verdict=verdict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest agent/tests/test_render.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add agent/src/agent/render.py agent/tests/test_render.py
git commit -m "feat(agent): render verdict as markdown PR comment"
```

---

### Task 8: CLI

**Files:**
- Create: `agent/src/agent/cli.py`
- Test: `agent/tests/test_cli.py`

**Interfaces:**
- Consumes: `run_review` (Task 6), `render_comment` (Task 7), `OllamaClient` (Task 1).
- Produces: a `click` command group `agent review` with:
  - `--diff-file PATH` — read diff from file.
  - `--trace-out PATH` (default `trace.json`) — write the `RunTrace` JSON here.
  - `--comment-out PATH` (default `comment.md`) — write the rendered markdown here.
  - `--model TEXT` (default `qwen2.5:7b`).
  - Registered as a console script `agent` in `agent/pyproject.toml` (`[project.scripts] agent = "agent.cli:cli"`).
  - Prints the verdict JSON to stdout on success; exits 0 on `approve`, exits 0 on `request_changes` too (the Action step, not the CLI exit code, decides CI pass/fail — this CLI's job is only to review and report).

- [ ] **Step 1: Write the failing test**

```python
# agent/tests/test_cli.py
import json

from click.testing import CliRunner

from agent.cli import cli
from agent.schemas import Issue, Verdict


def test_review_command_writes_trace_and_comment(tmp_path, monkeypatch):
    diff_file = tmp_path / "sample.diff"
    diff_file.write_text("+ query = f'SELECT {x}'")

    trace_out = tmp_path / "trace.json"
    comment_out = tmp_path / "comment.md"

    canned_verdict = Verdict(
        security_issues=[Issue(file="a.py", line=1, description="sqli", severity="critical")],
        style_issues=[],
        test_coverage_gap=False,
        verdict="request_changes",
    )

    def fake_run_review(diff, client):
        from datetime import datetime, timezone

        from trace_schema import RunTrace

        now = datetime.now(timezone.utc)
        trace = RunTrace(
            run_id="test-run",
            input={"diff": diff},
            steps=[],
            final_output=canned_verdict.model_dump(),
            started_at=now,
            ended_at=now,
            total_tokens=0,
            total_cost_usd=0.0,
        )
        return canned_verdict, trace

    monkeypatch.setattr("agent.cli.run_review", fake_run_review)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "review",
            "--diff-file",
            str(diff_file),
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
    assert "sqli" in comment_text

    output_json = json.loads(result.output)
    assert output_json["verdict"] == "request_changes"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest agent/tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.cli'`

- [ ] **Step 3: Write the implementation**

```python
# agent/src/agent/cli.py
import json

import click

from agent.graph import run_review
from agent.llm import OllamaClient
from agent.render import render_comment


@click.group()
def cli():
    pass


@cli.command()
@click.option("--diff-file", required=True, type=click.Path(exists=True), help="Path to a diff/patch file.")
@click.option("--trace-out", default="trace.json", type=click.Path(), help="Where to write the RunTrace JSON.")
@click.option("--comment-out", default="comment.md", type=click.Path(), help="Where to write the markdown PR comment.")
@click.option("--model", default="qwen2.5:7b", help="Ollama model to use.")
def review(diff_file, trace_out, comment_out, model):
    diff_text = open(diff_file).read()
    client = OllamaClient(model=model)

    verdict, trace = run_review(diff=diff_text, client=client)

    with open(trace_out, "w") as f:
        json.dump(trace.model_dump(mode="json"), f, indent=2)

    with open(comment_out, "w") as f:
        f.write(render_comment(verdict))

    click.echo(json.dumps(verdict.model_dump(), indent=2))


if __name__ == "__main__":
    cli()
```

Add the console script entry to `agent/pyproject.toml`:

```toml
[project.scripts]
agent = "agent.cli:cli"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv sync && uv run pytest agent/tests/test_cli.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add agent/src/agent/cli.py agent/pyproject.toml agent/tests/test_cli.py
git commit -m "feat(agent): add review CLI command"
```

---

### Task 9: Integration test against real Ollama

**Files:**
- Create: `agent/tests/test_integration.py`

**Interfaces:**
- Consumes: `run_review` (Task 6), `OllamaClient` (Task 1). No new production code.
- Produces: one test marked `@pytest.mark.integration` that is skipped by default and only runs with `-m integration`.

- [ ] **Step 1: Write the test**

```python
# agent/tests/test_integration.py
import pytest

from agent.graph import run_review
from agent.llm import OllamaClient


@pytest.mark.integration
def test_run_review_against_real_ollama_flags_sql_injection():
    diff = """\
--- a/app.py
+++ b/app.py
@@ -1,3 +1,3 @@
 def get_user(user_id):
-    return db.execute("SELECT * FROM users WHERE id = ?", [user_id])
+    return db.execute(f"SELECT * FROM users WHERE id = {user_id}")
"""
    client = OllamaClient(model="qwen2.5:7b")

    verdict, trace = run_review(diff=diff, client=client)

    assert verdict.verdict == "request_changes"
    assert len(verdict.security_issues) >= 1
    assert len(trace.steps) == 4
```

- [ ] **Step 2: Run it explicitly (not part of default test run)**

Run: `ollama pull qwen2.5:7b` (if not already pulled), then:
Run: `uv run pytest agent/tests/test_integration.py -v -m integration`
Expected: PASS (verdict is `request_changes`, at least one security issue reported). If it fails on `verdict.verdict`, inspect the model's actual output — local 7B models can be inconsistent; adjust the `SYSTEM_PROMPT` in `agent/src/agent/nodes/merge.py` if needed, not the test's intent.

- [ ] **Step 3: Confirm default test run still skips it**

Run: `uv run pytest agent/ -v`
Expected: integration test shown as `deselected` or not collected under `-m "not integration"`; confirm by running `uv run pytest agent/ -v -m "not integration"` and seeing all other tests pass.

- [ ] **Step 4: Commit**

```bash
git add agent/tests/test_integration.py
git commit -m "test(agent): add integration test against real Ollama"
```

---

### Task 10: GitHub Action wrapper

**Files:**
- Create: `agent/action.yml`
- Create: `.github/workflows/pr-review.yml`

**Interfaces:**
- Consumes: `agent` console script (Task 8).
- Produces: a composite Action at `agent/action.yml` that checks out the PR, computes the diff against the base ref, runs `agent review`, posts the rendered comment via the GitHub API using `GITHUB_TOKEN`, and uploads `trace.json` as a build artifact. `.github/workflows/pr-review.yml` triggers it on `pull_request`, running on `self-hosted`.

- [ ] **Step 1: Write `agent/action.yml`**

```yaml
name: "PR Review Agent"
description: "Runs the multi-agent PR review bot against the current PR diff."
runs:
  using: "composite"
  steps:
    - name: Compute diff against base ref
      shell: bash
      run: |
        git fetch origin "${{ github.event.pull_request.base.ref }}" --depth=1
        git diff "origin/${{ github.event.pull_request.base.ref }}"...HEAD > pr.diff

    - name: Install uv
      uses: astral-sh/setup-uv@v3

    - name: Sync workspace
      shell: bash
      run: uv sync

    - name: Run review
      shell: bash
      run: uv run agent review --diff-file pr.diff --trace-out trace.json --comment-out comment.md

    - name: Post PR comment
      shell: bash
      env:
        GH_TOKEN: ${{ github.token }}
      run: gh pr comment "${{ github.event.pull_request.number }}" --body-file comment.md

    - name: Upload trace artifact
      uses: actions/upload-artifact@v4
      with:
        name: pr-review-trace
        path: trace.json
```

- [ ] **Step 2: Write `.github/workflows/pr-review.yml`**

```yaml
name: PR Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  review:
    runs-on: self-hosted
    permissions:
      pull-requests: write
      contents: read
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: ./agent
```

- [ ] **Step 3: Validate YAML syntax**

Run: `python -c "import yaml; yaml.safe_load(open('agent/action.yml')); yaml.safe_load(open('.github/workflows/pr-review.yml')); print('valid')"`
Expected: prints `valid`.

- [ ] **Step 4: Commit**

```bash
git add agent/action.yml .github/workflows/pr-review.yml
git commit -m "feat(agent): add composite GitHub Action and PR review workflow"
```

---

## Definition of Done

- [ ] `uv run pytest agent/ -m "not integration"` passes with all tests green.
- [ ] `uv run pytest agent/ -m integration` passes against a real local Ollama with `qwen2.5:7b` pulled.
- [ ] `uv run agent review --diff-file <some diff>` produces `trace.json` (valid against `trace_schema.RunTrace`) and `comment.md`.
- [ ] `agent/action.yml` and `.github/workflows/pr-review.yml` are valid YAML.
- [ ] No file under `agent/` imports anything from a future `harness` package.
