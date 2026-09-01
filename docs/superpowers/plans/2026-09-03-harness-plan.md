# Eval Harness (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `harness` package — a generic, agent-agnostic eval harness (collector, pluggable scorer, gate, report) that scores `trace_schema.RunTrace` runs, plus a `pr_review` adapter and synthetic test-case suite that dogfood it against the `agent` package.

**Architecture:** `harness` core (`collector`, `scorer` protocol, `gate`, `aggregator`, `report`, `history`) depends only on `trace_schema`. `harness/adapters/pr_review.py` is the sole place PR-review-specific scoring logic lives, implementing the `Scorer` protocol by depending on `agent.schemas.Verdict` (a read-only import of types, not behavior) and `agent.graph.run_review` (to execute the agent under test). Everything else in `harness/` never imports `agent`.

**Tech Stack:** Python 3.12, `uv`, `pydantic` v2, `ollama` (for the LLM-judge), `jinja2` (HTML report), `click` (CLI), `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-01-pr-review-eval-design.md`

**Depends on:**
- `docs/superpowers/plans/2026-09-01-trace-schema-plan.md` (complete)
- `docs/superpowers/plans/2026-09-02-agent-plan.md` (complete — needed for the `pr_review` adapter and the synthetic test suite; harness *core* tasks 1-7 only need `trace_schema`)

## Global Constraints

- Python 3.12, same `uv` workspace.
- `harness` core modules (`collector.py`, `scorer.py`, `gate.py`, `aggregator.py`, `report.py`, `history.py`) import only `trace_schema`, the standard library, and generic third-party libs (`pydantic`, `jinja2`, `click`). They must never import `agent`.
- Only `harness/src/harness/adapters/pr_review.py` and its test may import from `agent`.
- Judge model: Ollama `qwen2.5:7b`, same as the agent — reuses `agent.llm.OllamaClient`'s pattern but harness core defines its own tiny client interface so it doesn't depend on `agent.llm` either (duplication here is intentional: it's the decoupling boundary, not laziness).
- Regression gate: fail (non-zero exit) if pass rate drops below the last recorded baseline pass rate in `history.json`.

---

### Task 1: `harness` package scaffold + collector

**Files:**
- Create: `harness/pyproject.toml`
- Create: `harness/src/harness/__init__.py`
- Create: `harness/src/harness/collector.py`
- Create: `harness/tests/__init__.py`
- Test: `harness/tests/test_collector.py`

**Interfaces:**
- Consumes: `trace_schema.RunTrace`.
- Produces: `harness.collector.load_trace(path: str | Path) -> RunTrace` — reads a JSON file and validates it as a `RunTrace`.

- [ ] **Step 1: Create `harness/pyproject.toml`**

```toml
[project]
name = "harness"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "trace-schema",
    "pydantic>=2.6",
    "jinja2>=3.1",
    "click>=8.1",
    "ollama>=0.3",
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
packages = ["src/harness"]

[project.scripts]
harness = "harness.cli:cli"
```

- [ ] **Step 2: Create empty package files**

`harness/src/harness/__init__.py`:
```python
```

`harness/tests/__init__.py`:
```python
```

- [ ] **Step 3: Write the failing test**

```python
# harness/tests/test_collector.py
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv sync && uv run pytest harness/tests/test_collector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.collector'`

- [ ] **Step 5: Write the implementation**

```python
# harness/src/harness/collector.py
import json
from pathlib import Path

from trace_schema import RunTrace


def load_trace(path: str | Path) -> RunTrace:
    data = json.loads(Path(path).read_text())
    return RunTrace.model_validate(data)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest harness/tests/test_collector.py -v`
Expected: 1 passed

- [ ] **Step 7: Commit**

```bash
git add harness/pyproject.toml harness/src/harness/__init__.py harness/src/harness/collector.py harness/tests/__init__.py harness/tests/test_collector.py
git commit -m "chore(harness): scaffold package and add trace collector"
```

---

### Task 2: `Scorer` protocol and `ScoreResult`

**Files:**
- Create: `harness/src/harness/scorer.py`
- Test: `harness/tests/test_scorer.py`

**Interfaces:**
- Consumes: `trace_schema.RunTrace`.
- Produces:
  - `class ScoreResult(BaseModel)`: `task_success: bool`, `tool_call_correctness: float` (0.0-1.0), `hallucination_rate: float` (0.0-1.0), `latency_seconds: float`, `cost_usd: float`, `details: dict[str, Any] = {}`.
  - `class Scorer(Protocol)`: `def score(self, trace: RunTrace, expected: Any) -> ScoreResult: ...`
  - `def latency_seconds(trace: RunTrace) -> float` — helper, `(trace.ended_at - trace.started_at).total_seconds()`.

- [ ] **Step 1: Write the failing test**

```python
# harness/tests/test_scorer.py
from datetime import datetime, timedelta, timezone

from trace_schema import RunTrace

from harness.scorer import ScoreResult, latency_seconds


def _trace(seconds: float) -> RunTrace:
    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=seconds)
    return RunTrace(
        run_id="r1",
        input=None,
        steps=[],
        final_output=None,
        started_at=start,
        ended_at=end,
        total_tokens=0,
        total_cost_usd=0.0,
    )


def test_latency_seconds_computes_duration():
    trace = _trace(2.5)
    assert abs(latency_seconds(trace) - 2.5) < 0.01


def test_score_result_defaults_details_to_empty_dict():
    result = ScoreResult(
        task_success=True,
        tool_call_correctness=1.0,
        hallucination_rate=0.0,
        latency_seconds=1.0,
        cost_usd=0.0,
    )
    assert result.details == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest harness/tests/test_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.scorer'`

- [ ] **Step 3: Write the implementation**

```python
# harness/src/harness/scorer.py
from typing import Any, Protocol

from pydantic import BaseModel

from trace_schema import RunTrace


class ScoreResult(BaseModel):
    task_success: bool
    tool_call_correctness: float
    hallucination_rate: float
    latency_seconds: float
    cost_usd: float
    details: dict[str, Any] = {}


class Scorer(Protocol):
    def score(self, trace: RunTrace, expected: Any) -> ScoreResult: ...


def latency_seconds(trace: RunTrace) -> float:
    return (trace.ended_at - trace.started_at).total_seconds()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest harness/tests/test_scorer.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add harness/src/harness/scorer.py harness/tests/test_scorer.py
git commit -m "feat(harness): add Scorer protocol and ScoreResult"
```

---

### Task 3: LLM-judge for hallucination scoring

**Files:**
- Create: `harness/src/harness/judge.py`
- Test: `harness/tests/test_judge.py`

**Interfaces:**
- Consumes: `trace_schema.RunTrace`, `ollama`.
- Produces:
  - `class JudgeVerdict(BaseModel)`: `hallucinated: bool`, `reasoning: str`.
  - `class HallucinationJudge`: `__init__(self, model: str = "qwen2.5:7b")`; method `judge(self, source_material: str, claim: str) -> JudgeVerdict` — asks the LLM whether `claim` is supported by `source_material`, using structured output.
  - `class FakeHallucinationJudge`: same interface, constructed with a canned `JudgeVerdict`, for tests.

- [ ] **Step 1: Write the failing test**

```python
# harness/tests/test_judge.py
from harness.judge import FakeHallucinationJudge, JudgeVerdict


def test_fake_judge_returns_canned_verdict():
    canned = JudgeVerdict(hallucinated=True, reasoning="claim not present in diff")
    judge = FakeHallucinationJudge(response=canned)

    result = judge.judge(source_material="diff text", claim="uses eval() unsafely")

    assert result == canned
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest harness/tests/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.judge'`

- [ ] **Step 3: Write the implementation**

```python
# harness/src/harness/judge.py
import ollama
from pydantic import BaseModel

JUDGE_SYSTEM_PROMPT = (
    "You are an evaluator checking whether a claim made by an AI code "
    "reviewer is actually supported by the provided source material (a "
    "diff or tool output). Respond with hallucinated=true if the claim "
    "describes something not present in or not supported by the source "
    "material, otherwise hallucinated=false. Give brief reasoning."
)


class JudgeVerdict(BaseModel):
    hallucinated: bool
    reasoning: str


class HallucinationJudge:
    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model

    def judge(self, source_material: str, claim: str) -> JudgeVerdict:
        user_content = f"SOURCE MATERIAL:\n{source_material}\n\nCLAIM:\n{claim}"
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            format=JudgeVerdict.model_json_schema(),
        )
        return JudgeVerdict.model_validate_json(response["message"]["content"])


class FakeHallucinationJudge:
    def __init__(self, response: JudgeVerdict):
        self._response = response

    def judge(self, source_material: str, claim: str) -> JudgeVerdict:
        return self._response
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest harness/tests/test_judge.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add harness/src/harness/judge.py harness/tests/test_judge.py
git commit -m "feat(harness): add LLM-as-judge hallucination checker"
```

---

### Task 4: `pr_review` adapter

**Files:**
- Create: `harness/src/harness/adapters/__init__.py`
- Create: `harness/src/harness/adapters/pr_review.py`
- Test: `harness/tests/adapters/__init__.py`
- Test: `harness/tests/adapters/test_pr_review.py`

**Interfaces:**
- Consumes: `Scorer`, `ScoreResult`, `latency_seconds` (Task 2), `HallucinationJudge`/`FakeHallucinationJudge`/`JudgeVerdict` (Task 3), `trace_schema.RunTrace`. Imports `agent.schemas.Verdict` for typing the expected/actual verdict shape only.
- Produces:
  - `class PrReviewExpected(BaseModel)`: `verdict: Literal["approve", "request_changes"]`, `expected_tool_agents: list[str]` (expected `agent_name` sequence, unordered-set compared since the graph fans out in parallel), `min_security_issues: int = 0`.
  - `class PrReviewScorer` implementing `Scorer`: `__init__(self, judge)`. `.score(trace, expected: PrReviewExpected) -> ScoreResult`:
    - `task_success`: `trace.final_output["verdict"] == expected.verdict`.
    - `tool_call_correctness`: fraction of `expected.expected_tool_agents` present in `{s.agent_name for s in trace.steps}`.
    - `hallucination_rate`: for each security issue in `trace.final_output["security_issues"]`, ask the judge whether it's supported by `trace.input["diff"]`; fraction judged `hallucinated=True` (0.0 if no issues reported).
    - `latency_seconds`: via `latency_seconds(trace)`.
    - `cost_usd`: `trace.total_cost_usd`.

- [ ] **Step 1: Create empty test package file**

`harness/tests/adapters/__init__.py`:
```python
```

- [ ] **Step 2: Write the failing test**

```python
# harness/tests/adapters/test_pr_review.py
from datetime import datetime, timedelta, timezone

from trace_schema import RunTrace, Step

from harness.adapters.pr_review import PrReviewExpected, PrReviewScorer
from harness.judge import FakeHallucinationJudge, JudgeVerdict


def _trace(verdict: str, security_issues: list[dict], agent_names: list[str], diff: str = "the diff") -> RunTrace:
    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=3)
    steps = [
        Step(agent_name=name, action="analyze_diff", started_at=start, ended_at=end, tokens_in=10, tokens_out=5)
        for name in agent_names
    ]
    return RunTrace(
        run_id="r1",
        input={"diff": diff},
        steps=steps,
        final_output={
            "verdict": verdict,
            "security_issues": security_issues,
            "style_issues": [],
            "test_coverage_gap": False,
        },
        started_at=start,
        ended_at=end,
        total_tokens=sum(s.tokens_in + s.tokens_out for s in steps),
        total_cost_usd=0.0,
    )


def test_task_success_matches_expected_verdict():
    trace = _trace("request_changes", [], ["security_review", "style_review", "test_coverage", "merge"])
    expected = PrReviewExpected(
        verdict="request_changes", expected_tool_agents=["security_review", "style_review", "test_coverage", "merge"]
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = PrReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is True
    assert result.tool_call_correctness == 1.0


def test_task_success_false_on_verdict_mismatch():
    trace = _trace("approve", [], ["security_review", "style_review", "test_coverage", "merge"])
    expected = PrReviewExpected(
        verdict="request_changes", expected_tool_agents=["security_review", "style_review", "test_coverage", "merge"]
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = PrReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.task_success is False


def test_tool_call_correctness_partial_when_agent_missing():
    trace = _trace("approve", [], ["security_review", "style_review"])
    expected = PrReviewExpected(
        verdict="approve", expected_tool_agents=["security_review", "style_review", "test_coverage", "merge"]
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=False, reasoning=""))
    scorer = PrReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.tool_call_correctness == 0.5


def test_hallucination_rate_uses_judge_per_security_issue():
    trace = _trace(
        "request_changes",
        [{"file": "a.py", "line": 1, "description": "fabricated issue", "severity": "high"}],
        ["security_review", "style_review", "test_coverage", "merge"],
    )
    expected = PrReviewExpected(
        verdict="request_changes", expected_tool_agents=["security_review", "style_review", "test_coverage", "merge"]
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=True, reasoning="not in diff"))
    scorer = PrReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.hallucination_rate == 1.0


def test_hallucination_rate_zero_when_no_issues_reported():
    trace = _trace("approve", [], ["security_review", "style_review", "test_coverage", "merge"])
    expected = PrReviewExpected(
        verdict="approve", expected_tool_agents=["security_review", "style_review", "test_coverage", "merge"]
    )
    judge = FakeHallucinationJudge(response=JudgeVerdict(hallucinated=True, reasoning="unused"))
    scorer = PrReviewScorer(judge=judge)

    result = scorer.score(trace, expected)

    assert result.hallucination_rate == 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest harness/tests/adapters/test_pr_review.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.adapters'`

- [ ] **Step 4: Write the implementation**

```python
# harness/src/harness/adapters/__init__.py
```

```python
# harness/src/harness/adapters/pr_review.py
from typing import Literal

from pydantic import BaseModel

from trace_schema import RunTrace

from harness.judge import HallucinationJudge
from harness.scorer import ScoreResult, latency_seconds


class PrReviewExpected(BaseModel):
    verdict: Literal["approve", "request_changes"]
    expected_tool_agents: list[str]
    min_security_issues: int = 0


class PrReviewScorer:
    def __init__(self, judge: HallucinationJudge):
        self._judge = judge

    def score(self, trace: RunTrace, expected: PrReviewExpected) -> ScoreResult:
        final = trace.final_output or {}
        task_success = final.get("verdict") == expected.verdict

        actual_agents = {s.agent_name for s in trace.steps}
        expected_agents = set(expected.expected_tool_agents)
        tool_call_correctness = (
            len(expected_agents & actual_agents) / len(expected_agents) if expected_agents else 1.0
        )

        security_issues = final.get("security_issues", [])
        diff = (trace.input or {}).get("diff", "")
        if security_issues:
            hallucinated_count = 0
            for issue in security_issues:
                verdict = self._judge.judge(source_material=diff, claim=issue.get("description", ""))
                if verdict.hallucinated:
                    hallucinated_count += 1
            hallucination_rate = hallucinated_count / len(security_issues)
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

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest harness/tests/adapters/test_pr_review.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add harness/src/harness/adapters/ harness/tests/adapters/
git commit -m "feat(harness): add pr_review Scorer adapter"
```

---

### Task 5: Aggregator

**Files:**
- Create: `harness/src/harness/aggregator.py`
- Test: `harness/tests/test_aggregator.py`

**Interfaces:**
- Consumes: `harness.scorer.ScoreResult` (Task 2).
- Produces: `class AggregateReport(BaseModel)`: `pass_rate: float`, `avg_hallucination_rate: float`, `avg_cost_usd: float`, `avg_latency_seconds: float`, `total_cases: int`. `def aggregate(results: list[ScoreResult]) -> AggregateReport`.

- [ ] **Step 1: Write the failing test**

```python
# harness/tests/test_aggregator.py
from harness.aggregator import aggregate
from harness.scorer import ScoreResult


def test_aggregate_computes_pass_rate_and_averages():
    results = [
        ScoreResult(task_success=True, tool_call_correctness=1.0, hallucination_rate=0.0, latency_seconds=2.0, cost_usd=0.0),
        ScoreResult(task_success=False, tool_call_correctness=0.5, hallucination_rate=0.5, latency_seconds=4.0, cost_usd=0.0),
    ]

    report = aggregate(results)

    assert report.total_cases == 2
    assert report.pass_rate == 0.5
    assert report.avg_hallucination_rate == 0.25
    assert report.avg_latency_seconds == 3.0
    assert report.avg_cost_usd == 0.0


def test_aggregate_empty_results():
    report = aggregate([])
    assert report.total_cases == 0
    assert report.pass_rate == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest harness/tests/test_aggregator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.aggregator'`

- [ ] **Step 3: Write the implementation**

```python
# harness/src/harness/aggregator.py
from pydantic import BaseModel

from harness.scorer import ScoreResult


class AggregateReport(BaseModel):
    pass_rate: float
    avg_hallucination_rate: float
    avg_cost_usd: float
    avg_latency_seconds: float
    total_cases: int


def aggregate(results: list[ScoreResult]) -> AggregateReport:
    if not results:
        return AggregateReport(
            pass_rate=0.0, avg_hallucination_rate=0.0, avg_cost_usd=0.0, avg_latency_seconds=0.0, total_cases=0
        )
    n = len(results)
    return AggregateReport(
        pass_rate=sum(1 for r in results if r.task_success) / n,
        avg_hallucination_rate=sum(r.hallucination_rate for r in results) / n,
        avg_cost_usd=sum(r.cost_usd for r in results) / n,
        avg_latency_seconds=sum(r.latency_seconds for r in results) / n,
        total_cases=n,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest harness/tests/test_aggregator.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add harness/src/harness/aggregator.py harness/tests/test_aggregator.py
git commit -m "feat(harness): add result aggregator"
```

---

### Task 6: History store + gate

**Files:**
- Create: `harness/src/harness/history.py`
- Create: `harness/src/harness/gate.py`
- Test: `harness/tests/test_history.py`
- Test: `harness/tests/test_gate.py`

**Interfaces:**
- Consumes: `AggregateReport` (Task 5).
- Produces:
  - `history.py`: `class HistoryEntry(BaseModel)`: `commit_sha: str`, `timestamp: str`, `report: AggregateReport`. `def load_history(path) -> list[HistoryEntry]` (returns `[]` if file missing). `def append_history(path, entry: HistoryEntry) -> None` (appends and writes back as a JSON list).
  - `gate.py`: `class GateResult(BaseModel)`: `passed: bool`, `reason: str`. `def check_gate(current: AggregateReport, history: list[HistoryEntry]) -> GateResult` — passes if `history` is empty (no baseline yet) or `current.pass_rate >= history[-1].report.pass_rate`; otherwise fails with a reason string naming both rates.

- [ ] **Step 1: Write the failing tests**

```python
# harness/tests/test_history.py
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
```

```python
# harness/tests/test_gate.py
from harness.aggregator import AggregateReport
from harness.gate import check_gate
from harness.history import HistoryEntry


def _report(pass_rate: float) -> AggregateReport:
    return AggregateReport(pass_rate=pass_rate, avg_hallucination_rate=0.0, avg_cost_usd=0.0, avg_latency_seconds=1.0, total_cases=4)


def test_gate_passes_when_no_history():
    result = check_gate(_report(0.5), [])
    assert result.passed is True


def test_gate_passes_when_pass_rate_holds_or_improves():
    history = [HistoryEntry(commit_sha="a", timestamp="t", report=_report(0.5))]
    result = check_gate(_report(0.75), history)
    assert result.passed is True


def test_gate_fails_on_regression():
    history = [HistoryEntry(commit_sha="a", timestamp="t", report=_report(0.75))]
    result = check_gate(_report(0.5), history)
    assert result.passed is False
    assert "0.5" in result.reason
    assert "0.75" in result.reason
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest harness/tests/test_history.py harness/tests/test_gate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementations**

```python
# harness/src/harness/history.py
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
```

```python
# harness/src/harness/gate.py
from pydantic import BaseModel

from harness.aggregator import AggregateReport
from harness.history import HistoryEntry


class GateResult(BaseModel):
    passed: bool
    reason: str


def check_gate(current: AggregateReport, history: list[HistoryEntry]) -> GateResult:
    if not history:
        return GateResult(passed=True, reason="no baseline yet; recording first result")

    baseline = history[-1].report
    if current.pass_rate >= baseline.pass_rate:
        return GateResult(
            passed=True,
            reason=f"pass rate {current.pass_rate} >= baseline {baseline.pass_rate}",
        )
    return GateResult(
        passed=False,
        reason=f"pass rate regressed: {current.pass_rate} < baseline {baseline.pass_rate}",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest harness/tests/test_history.py harness/tests/test_gate.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add harness/src/harness/history.py harness/src/harness/gate.py harness/tests/test_history.py harness/tests/test_gate.py
git commit -m "feat(harness): add history store and regression gate"
```

---

### Task 7: HTML report

**Files:**
- Create: `harness/src/harness/report.py`
- Test: `harness/tests/test_report.py`

**Interfaces:**
- Consumes: `AggregateReport` (Task 5), `HistoryEntry` (Task 6).
- Produces: `def render_html_report(current: AggregateReport, history: list[HistoryEntry]) -> str` — returns a self-contained HTML string with a summary table (pass rate, hallucination rate, avg cost, avg latency, total cases) and a simple trend table/inline-SVG sparkline of `pass_rate` over `history` entries (oldest to newest, including current as the last point).

- [ ] **Step 1: Write the failing test**

```python
# harness/tests/test_report.py
from harness.aggregator import AggregateReport
from harness.history import HistoryEntry
from harness.report import render_html_report


def test_render_html_report_includes_summary_numbers():
    current = AggregateReport(pass_rate=0.75, avg_hallucination_rate=0.1, avg_cost_usd=0.0, avg_latency_seconds=3.2, total_cases=8)
    history = [
        HistoryEntry(commit_sha="a1", timestamp="2026-08-30T00:00:00Z", report=AggregateReport(pass_rate=0.5, avg_hallucination_rate=0.2, avg_cost_usd=0.0, avg_latency_seconds=4.0, total_cases=8))
    ]

    html = render_html_report(current, history)

    assert "<html" in html.lower()
    assert "0.75" in html
    assert "a1" in html
    assert "0.5" in html


def test_render_html_report_handles_empty_history():
    current = AggregateReport(pass_rate=1.0, avg_hallucination_rate=0.0, avg_cost_usd=0.0, avg_latency_seconds=1.0, total_cases=3)
    html = render_html_report(current, [])
    assert "1.0" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest harness/tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.report'`

- [ ] **Step 3: Write the implementation**

```python
# harness/src/harness/report.py
from jinja2 import Template

from harness.aggregator import AggregateReport
from harness.history import HistoryEntry

TEMPLATE = Template(
    """\
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Eval Report</title></head>
<body>
  <h1>Eval Harness Report</h1>
  <h2>Current Run</h2>
  <table border="1" cellpadding="4">
    <tr><th>Pass rate</th><td>{{ current.pass_rate }}</td></tr>
    <tr><th>Avg hallucination rate</th><td>{{ current.avg_hallucination_rate }}</td></tr>
    <tr><th>Avg cost (USD)</th><td>{{ current.avg_cost_usd }}</td></tr>
    <tr><th>Avg latency (s)</th><td>{{ current.avg_latency_seconds }}</td></tr>
    <tr><th>Total cases</th><td>{{ current.total_cases }}</td></tr>
  </table>

  <h2>Trend</h2>
  {% if history %}
  <table border="1" cellpadding="4">
    <tr><th>Commit</th><th>Timestamp</th><th>Pass rate</th><th>Hallucination rate</th></tr>
    {% for entry in history %}
    <tr>
      <td>{{ entry.commit_sha }}</td>
      <td>{{ entry.timestamp }}</td>
      <td>{{ entry.report.pass_rate }}</td>
      <td>{{ entry.report.avg_hallucination_rate }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <p>No history yet.</p>
  {% endif %}
</body>
</html>
"""
)


def render_html_report(current: AggregateReport, history: list[HistoryEntry]) -> str:
    return TEMPLATE.render(current=current, history=history)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest harness/tests/test_report.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add harness/src/harness/report.py harness/tests/test_report.py
git commit -m "feat(harness): add static HTML report generator"
```

---

### Task 8: Synthetic test-case suite

**Files:**
- Create: `testcases/sql_injection/diff.patch`
- Create: `testcases/sql_injection/expected.json`
- Create: `testcases/xss/diff.patch`
- Create: `testcases/xss/expected.json`
- Create: `testcases/hardcoded_secret/diff.patch`
- Create: `testcases/hardcoded_secret/expected.json`
- Create: `testcases/clean_diff/diff.patch`
- Create: `testcases/clean_diff/expected.json`
- Create: `testcases/missing_tests/diff.patch`
- Create: `testcases/missing_tests/expected.json`
- Create: `testcases/style_only/diff.patch`
- Create: `testcases/style_only/expected.json`
- Test: `harness/tests/test_testcases_shape.py`

**Interfaces:**
- Consumes: `PrReviewExpected` (Task 4) shape for validation only.
- Produces: 6 test-case directories under `testcases/`, each with `diff.patch` (raw unified diff text) and `expected.json` (validates as `PrReviewExpected`).

- [ ] **Step 1: Write `testcases/sql_injection/diff.patch`**

```diff
--- a/app/db.py
+++ b/app/db.py
@@ -10,7 +10,7 @@ class UserRepo:
     def get_user(self, user_id):
-        return self.conn.execute("SELECT * FROM users WHERE id = ?", [user_id])
+        return self.conn.execute(f"SELECT * FROM users WHERE id = {user_id}")
```

`testcases/sql_injection/expected.json`:
```json
{
  "verdict": "request_changes",
  "expected_tool_agents": ["security_review", "style_review", "test_coverage", "merge"],
  "min_security_issues": 1
}
```

- [ ] **Step 2: Write `testcases/xss/diff.patch`**

```diff
--- a/app/templates.py
+++ b/app/templates.py
@@ -5,6 +5,6 @@ def render_comment(comment_text):
-    return f"<div>{escape(comment_text)}</div>"
+    return f"<div>{comment_text}</div>"
```

`testcases/xss/expected.json`:
```json
{
  "verdict": "request_changes",
  "expected_tool_agents": ["security_review", "style_review", "test_coverage", "merge"],
  "min_security_issues": 1
}
```

- [ ] **Step 3: Write `testcases/hardcoded_secret/diff.patch`**

```diff
--- a/app/config.py
+++ b/app/config.py
@@ -1,4 +1,5 @@
 import os
 
 API_KEY = os.environ.get("API_KEY")
+STRIPE_SECRET_KEY = "REDACTED_FAKE_STRIPE_KEY_FOR_TESTING_NOT_REAL"
```

`testcases/hardcoded_secret/expected.json`:
```json
{
  "verdict": "request_changes",
  "expected_tool_agents": ["security_review", "style_review", "test_coverage", "merge"],
  "min_security_issues": 1
}
```

- [ ] **Step 4: Write `testcases/clean_diff/diff.patch`**

```diff
--- a/app/utils.py
+++ b/app/utils.py
@@ -1,3 +1,6 @@
 def add(a, b):
     return a + b
+
+def subtract(a, b):
+    return a - b
--- a/tests/test_utils.py
+++ b/tests/test_utils.py
@@ -1,3 +1,7 @@
 def test_add():
     assert add(2, 3) == 5
+
+
+def test_subtract():
+    assert subtract(5, 3) == 2
```

`testcases/clean_diff/expected.json`:
```json
{
  "verdict": "approve",
  "expected_tool_agents": ["security_review", "style_review", "test_coverage", "merge"],
  "min_security_issues": 0
}
```

- [ ] **Step 5: Write `testcases/missing_tests/diff.patch`**

```diff
--- a/app/pricing.py
+++ b/app/pricing.py
@@ -1,3 +1,9 @@
 def base_price(item):
     return item.price
+
+
+def discounted_price(item, discount_percent):
+    if discount_percent < 0 or discount_percent > 100:
+        raise ValueError("invalid discount")
+    return item.price * (1 - discount_percent / 100)
```

`testcases/missing_tests/expected.json`:
```json
{
  "verdict": "request_changes",
  "expected_tool_agents": ["security_review", "style_review", "test_coverage", "merge"],
  "min_security_issues": 0
}
```

- [ ] **Step 6: Write `testcases/style_only/diff.patch`**

```diff
--- a/app/format.py
+++ b/app/format.py
@@ -1,4 +1,4 @@
-def formatName(first,last):
-  return first+" "+last
+def format_name(first, last):
+    return first + " " + last
```

`testcases/style_only/expected.json`:
```json
{
  "verdict": "approve",
  "expected_tool_agents": ["security_review", "style_review", "test_coverage", "merge"],
  "min_security_issues": 0
}
```

- [ ] **Step 7: Write a shape-validation test**

```python
# harness/tests/test_testcases_shape.py
import json
from pathlib import Path

from harness.adapters.pr_review import PrReviewExpected

TESTCASES_DIR = Path(__file__).resolve().parents[2] / "testcases"


def test_every_testcase_has_diff_and_valid_expected_json():
    case_dirs = [d for d in TESTCASES_DIR.iterdir() if d.is_dir()]
    assert len(case_dirs) >= 6, "expected at least 6 synthetic test cases"

    for case_dir in case_dirs:
        diff_path = case_dir / "diff.patch"
        expected_path = case_dir / "expected.json"
        assert diff_path.exists(), f"missing diff.patch in {case_dir}"
        assert expected_path.exists(), f"missing expected.json in {case_dir}"

        data = json.loads(expected_path.read_text())
        PrReviewExpected.model_validate(data)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest harness/tests/test_testcases_shape.py -v`
Expected: 1 passed

- [ ] **Step 9: Commit**

```bash
git add testcases/ harness/tests/test_testcases_shape.py
git commit -m "test: add synthetic PR diff test-case suite"
```

---

### Task 9: `harness` CLI (`run`, `gate`, `report`)

**Files:**
- Create: `harness/src/harness/cli.py`
- Test: `harness/tests/test_cli.py`

**Interfaces:**
- Consumes: `agent.graph.run_review`, `agent.llm.OllamaClient` (only imported inside the CLI's `run` command, which is explicitly the harness's one integration point with a concrete agent — kept separate from `harness` core); `PrReviewScorer`/`PrReviewExpected` (Task 4); `aggregate` (Task 5); `load_history`/`append_history`/`HistoryEntry` (Task 6); `check_gate` (Task 6); `render_html_report` (Task 7).
- Produces: `harness gate --testcases-dir testcases/ --history-path history.json --report-out report.html --commit-sha <sha>` — for each case dir: runs the agent on `diff.patch`, scores with `PrReviewScorer`, aggregates, appends to history, writes `report.html`, prints the `GateResult` reason, exits 1 if `not passed`.

- [ ] **Step 1: Write the failing test**

```python
# harness/tests/test_cli.py
import json

from click.testing import CliRunner

from harness.aggregator import AggregateReport
from harness.cli import cli


def test_gate_command_passes_with_no_prior_history(tmp_path, monkeypatch):
    testcases_dir = tmp_path / "testcases"
    case_dir = testcases_dir / "clean_diff"
    case_dir.mkdir(parents=True)
    (case_dir / "diff.patch").write_text("+ def add(a, b): return a + b")
    (case_dir / "expected.json").write_text(
        json.dumps({"verdict": "approve", "expected_tool_agents": ["security_review"], "min_security_issues": 0})
    )

    history_path = tmp_path / "history.json"
    report_out = tmp_path / "report.html"

    def fake_run_case(diff_text):
        from datetime import datetime, timezone

        from trace_schema import RunTrace

        now = datetime.now(timezone.utc)
        trace = RunTrace(
            run_id="r1",
            input={"diff": diff_text},
            steps=[{"agent_name": "security_review", "action": "analyze_diff", "started_at": now.isoformat(), "ended_at": now.isoformat(), "tool_calls": [], "output": None, "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0}],
            final_output={"verdict": "approve", "security_issues": [], "style_issues": [], "test_coverage_gap": False},
            started_at=now,
            ended_at=now,
            total_tokens=2,
            total_cost_usd=0.0,
        )
        return trace

    monkeypatch.setattr("harness.cli._run_case", lambda diff_text: fake_run_case(diff_text))
    monkeypatch.setattr(
        "harness.cli.HallucinationJudge",
        lambda *a, **k: type("J", (), {"judge": staticmethod(lambda **kw: __import__("harness.judge", fromlist=["JudgeVerdict"]).JudgeVerdict(hallucinated=False, reasoning=""))})(),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "gate",
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

Run: `uv run pytest harness/tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.cli'`

- [ ] **Step 3: Write the implementation**

```python
# harness/src/harness/cli.py
import json
from datetime import datetime, timezone
from pathlib import Path

import click

from trace_schema import RunTrace

from harness.adapters.pr_review import PrReviewExpected, PrReviewScorer
from harness.aggregator import aggregate
from harness.gate import check_gate
from harness.history import HistoryEntry, append_history, load_history
from harness.judge import HallucinationJudge
from harness.report import render_html_report


def _run_case(diff_text: str) -> RunTrace:
    from agent.graph import run_review
    from agent.llm import OllamaClient

    _, trace = run_review(diff=diff_text, client=OllamaClient())
    return trace


@click.group()
def cli():
    pass


@cli.command()
@click.option("--testcases-dir", required=True, type=click.Path(exists=True))
@click.option("--history-path", required=True, type=click.Path())
@click.option("--report-out", required=True, type=click.Path())
@click.option("--commit-sha", required=True)
def gate(testcases_dir, history_path, report_out, commit_sha):
    judge = HallucinationJudge()
    scorer = PrReviewScorer(judge=judge)

    results = []
    for case_dir in sorted(Path(testcases_dir).iterdir()):
        if not case_dir.is_dir():
            continue
        diff_text = (case_dir / "diff.patch").read_text()
        expected_data = json.loads((case_dir / "expected.json").read_text())
        expected = PrReviewExpected.model_validate(expected_data)

        trace = _run_case(diff_text)
        results.append(scorer.score(trace, expected))

    current = aggregate(results)
    history = load_history(history_path)
    gate_result = check_gate(current, history)

    entry = HistoryEntry(commit_sha=commit_sha, timestamp=datetime.now(timezone.utc).isoformat(), report=current)
    append_history(history_path, entry)

    html = render_html_report(current, load_history(history_path))
    Path(report_out).write_text(html)

    click.echo(gate_result.reason)
    if not gate_result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv sync && uv run pytest harness/tests/test_cli.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add harness/src/harness/cli.py harness/tests/test_cli.py
git commit -m "feat(harness): add gate CLI command wiring collector/scorer/gate/report"
```

---

### Task 10: CI workflow gating on the harness (dogfooding)

**Files:**
- Create: `.github/workflows/eval-gate.yml`

**Interfaces:**
- Consumes: `harness` console script (Task 9).
- Produces: a workflow triggered on `push` and `pull_request`, running on `self-hosted`, that runs `harness gate` against `testcases/`, uploads `report.html` as an artifact, and fails the job if the gate fails (via the CLI's `SystemExit(1)`).

- [ ] **Step 1: Write `.github/workflows/eval-gate.yml`**

```yaml
name: Eval Gate

on:
  push:
    branches: [main]
  pull_request:

jobs:
  eval:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Sync workspace
        run: uv sync

      - name: Run eval gate
        run: |
          uv run harness gate \
            --testcases-dir testcases \
            --history-path history.json \
            --report-out report.html \
            --commit-sha "${{ github.sha }}"

      - name: Upload eval report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-report
          path: |
            report.html
            history.json
```

- [ ] **Step 2: Validate YAML syntax**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/eval-gate.yml')); print('valid')"`
Expected: prints `valid`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/eval-gate.yml
git commit -m "ci: add eval-gate workflow dogfooding the harness on this repo"
```

---

### Task 11: Generate a real results report

**Files:**
- Modify: none (this task runs commands, not code changes)
- Create: `docs/results/report.html` (copied from a real run)
- Create: `docs/results/history.json` (copied from a real run)

**Interfaces:**
- Consumes: the full working `harness gate` command from Task 9/10.
- Produces: a committed, real (non-placeholder) results report generated by actually running the harness against `testcases/` with a live Ollama.

- [ ] **Step 1: Run the harness for real**

Run: `ollama pull qwen2.5:7b` (if not already), then:
Run: `uv run harness gate --testcases-dir testcases --history-path history.json --report-out report.html --commit-sha "$(git rev-parse HEAD)"`
Expected: exits 0 or 1 depending on real scores against the 6 synthetic cases — record whichever it is, don't force a particular outcome.

- [ ] **Step 2: Copy the real output into `docs/results/`**

Run: `mkdir -p docs/results && cp report.html docs/results/report.html && cp history.json docs/results/history.json`

- [ ] **Step 3: Commit**

```bash
git add docs/results/report.html docs/results/history.json
git commit -m "docs: add real eval harness results report"
```

---

### Task 12: README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: a top-level `README.md` explaining the architecture, the decoupling rationale, how to run each part, and how CI dogfoods the harness against the agent.

- [ ] **Step 1: Write `README.md`**

```markdown
# PR Review Bot + Eval Harness

Two decoupled projects sharing one contract.

## Why decoupled

`trace_schema/` defines what an agent "run" looks like — input, steps
(agent name, action, tool calls, output), final output, timestamps,
token/cost usage. `agent/` (the PR review bot) and `harness/` (the eval
harness) both depend on `trace_schema/`, but never on each other's
internals. `harness/` core (`collector.py`, `scorer.py`, `gate.py`,
`aggregator.py`, `report.py`, `history.py`) only imports `trace_schema` —
it has no idea what a "PR review" or a "security issue" is. The one place
PR-review-specific scoring logic lives is `harness/adapters/pr_review.py`,
which implements the harness's `Scorer` protocol. Swapping in a
completely different agent means writing a new adapter — the harness
core doesn't change.

## Layout

- `trace_schema/` — the shared contract (pydantic models + JSON Schema export).
- `agent/` — LangGraph multi-agent PR reviewer (security / style /
  test-coverage / merge), CLI, and a composite GitHub Action.
- `harness/` — generic eval harness: trace collector, pluggable
  `Scorer` protocol, pass/fail gate, HTML report/dashboard, and the
  `pr_review` adapter + CLI.
- `testcases/` — synthetic PR diffs with known expected outcomes (SQL
  injection, XSS, hardcoded secret, clean diff, missing tests, style-only).
- `docs/results/` — a real report generated by running the harness
  against `testcases/`.

## Running locally

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com)
with `qwen2.5:7b` pulled (`ollama pull qwen2.5:7b`).

```bash
uv sync

# Review a diff
uv run agent review --diff-file some.diff --trace-out trace.json --comment-out comment.md

# Run the eval harness against the synthetic test suite
uv run harness gate --testcases-dir testcases --history-path history.json \
  --report-out report.html --commit-sha "$(git rev-parse HEAD)"
```

## CI (dogfooding)

Two workflows, both requiring a self-hosted runner with Ollama installed
(GitHub-hosted runners can't reach a local Ollama instance):

- `.github/workflows/pr-review.yml` — runs `agent/`'s composite Action on
  incoming PRs to this repo and posts a review comment.
- `.github/workflows/eval-gate.yml` — runs `harness gate` against
  `testcases/` on every push/PR, using this repo's own `agent/` code.
  This is the harness gating the agent's own repo: if a change to `agent/`
  regresses the pass rate on the synthetic test suite, the build fails.

## Known limitations

- LLM is local-only (`qwen2.5:7b` via Ollama) — no hosted-runner CI path
  today; both workflows require a self-hosted runner.
- The test-case suite is synthetic (6 hand-written diffs), not sourced
  from real historical PRs.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add top-level README explaining architecture and decoupling"
```

---

## Definition of Done

- [ ] `uv run pytest harness/ -v` passes with all tests green.
- [ ] No file under `harness/src/harness/collector.py`, `scorer.py`, `gate.py`, `aggregator.py`, `report.py`, or `history.py` imports `agent`.
- [ ] `uv run harness gate --testcases-dir testcases --history-path <tmp>/history.json --report-out <tmp>/report.html --commit-sha test` runs end-to-end against a real local Ollama and produces a report.
- [ ] `.github/workflows/eval-gate.yml` is valid YAML and gates on regression.
- [ ] `docs/results/report.html` contains real numbers from an actual run, not placeholders.
- [ ] `README.md` exists at the repo root and explains the architecture and decoupling rationale.
