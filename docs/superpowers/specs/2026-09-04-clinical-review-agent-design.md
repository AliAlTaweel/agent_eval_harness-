# Clinical Documentation Review Agent — Design

**Status:** Approved for planning
**Date:** 2026-09-04

## Goal

Port the existing PR-review architecture to a second, unrelated domain —
clinical documentation review — to prove the decoupled `trace_schema` /
`harness` design generalizes across agents without modification. Build a new
`clinical_review/` agent package that reviews a clinical note's
**documentation quality** (completeness, compliance, coding clarity), not
its medical correctness. This is a portfolio/demo project, not a clinical
tool: no diagnosis, no patient-facing output, no claim of regulatory
compliance.

**Explicit non-goal:** this agent does not evaluate whether the documented
clinical reasoning is medically correct, and must never be described or used
as a clinical decision-support tool. Its review rules are illustrative
examples of documentation-quality checks, not a real coding/compliance
authority (no CMS/AMA guideline citations, no legal claim of accuracy).

## Why a separate package, not a mode on `agent/`

`agent/` is scoped to one domain (PR review): its prompts, schemas, and
findings shape (`file`/`line`) are specific to source diffs. Bolting a
second, unrelated domain onto it as a flag would mix two domains' prompts
and types in one codebase and blur the "one package, one responsibility"
boundary the rest of this project follows. `clinical_review/` is a new,
independent workspace member — structurally a sibling of `agent/`, not a
variant of it. Both depend only on `trace_schema` and never on each other.

## Repo layout changes

```
trace_schema/            (unchanged)
agent/                   (unchanged)
clinical_review/         (new — sibling to agent/)
  pyproject.toml
  src/clinical_review/
    __init__.py
    llm.py               (OllamaClient — same shape as agent/llm.py, own copy;
                           clinical_review must not import from agent/)
    schemas.py
    tracing.py            (TraceRecorder — same shape as agent/tracing.py)
    nodes/
      completeness.py
      compliance.py
      coding_clarity.py
      merge.py
    graph.py
    render.py
    cli.py
    streamlit_app.py
  tests/
harness/
  src/harness/
    adapters/
      pr_review.py         (unchanged)
      clinical_review.py   (new)
    cli.py                 (adds `gate-clinical-review` command; `gate`
                             command unchanged)
testcases/                 (unchanged — PR-review fixtures)
clinical_testcases/         (new — clinical-review fixtures)
  clean_new_patient/
    input.json
    expected.json
  incomplete_new_patient/
    input.json
    expected.json
  clean_follow_up/
    input.json
    expected.json
  mismatched_follow_up/
    input.json
    expected.json
```

`clinical_review/llm.py` and `tracing.py` duplicate `agent/llm.py` and
`agent/tracing.py` rather than sharing code — both packages are meant to be
independently deployable/readable units, and the duplication is a handful of
small, stable functions (`OllamaClient.chat`, `TraceRecorder.record`), not
business logic. If a third agent is ever added, this is the point at which
extracting a shared (optional, non-`trace_schema`) helper package becomes
worth it — not before (YAGNI).

## Global Constraints

- Python 3.12, `uv` workspace member (same as `agent/`, `harness/`).
- `clinical_review` depends only on `trace_schema` — never on `agent` or
  `harness`.
- `harness` may depend on `clinical_review` only via its public API
  (`run_review`), the same sanctioned pattern already used for `agent` in
  `harness/cli.py`.
- Local Ollama, model `qwen2.5:7b` — same as `agent/`.
- Every run produces a `trace_schema.RunTrace`, identical contract to the PR
  agent.
- All review rules encoded in prompts are explicitly illustrative example
  rules, not a real clinical/coding/compliance authority. This must be
  stated in the package README and in the rendered comment's footer.
- No real patient data anywhere in the repo — all note text in tests/docs is
  fabricated.

## Components

### Input & schemas (`clinical_review/schemas.py`)

```python
from typing import Literal
from pydantic import BaseModel

EncounterType = Literal["new_patient", "follow_up"]

class Finding(BaseModel):
    element: str
    description: str
    severity: Literal["low", "medium", "high", "critical"]

class CompletenessFindings(BaseModel):
    issues: list[Finding]

class ComplianceFindings(BaseModel):
    issues: list[Finding]

class CodingClarityFindings(BaseModel):
    issues: list[Finding]

class Verdict(BaseModel):
    completeness_issues: list[Finding]
    compliance_issues: list[Finding]
    coding_clarity_issues: list[Finding]
    verdict: Literal["approve", "needs_revision"]
```

`Finding.element` replaces the PR agent's `file`/`line` — it names the
missing/problematic documentation element (e.g. `"Review of Systems"`)
since there is no source-code location to point at.

### Graph (`clinical_review/graph.py`)

Same fan-out/fan-in `StateGraph` shape as `agent/graph.py`:

```python
class ReviewState(TypedDict, total=False):
    note_text: str
    encounter_type: EncounterType
    completeness: CompletenessFindings
    compliance: ComplianceFindings
    coding_clarity: CodingClarityFindings
    verdict: Verdict
```

`START` fans out to `completeness_review`, `compliance_review`,
`coding_clarity_review` (parallel), all fan in to `merge`, then `END`.

`run_review(note_text: str, encounter_type: EncounterType, client) ->
tuple[Verdict, RunTrace]` — same signature pattern as `agent.graph.run_review`
plus the extra `encounter_type` parameter. Raises `ReviewFailedError(cause,
trace)` on node failure, carrying a partial trace — identical mechanism to
`agent.graph.ReviewFailedError`. `trace.input` is
`{"note_text": note_text, "encounter_type": encounter_type}`.

### Specialist nodes (illustrative rule sets)

Each node is a thin wrapper (`recorder.record(<name>, "analyze_note", ...)`),
same pattern as `agent/nodes/*.py`. Prompts encode a **small, explicitly
illustrative** rule set:

**Completeness** (`nodes/completeness.py`) — required elements differ by
`encounter_type`:
- `new_patient`: Chief Complaint, History of Present Illness, Review of
  Systems, Past Medical History, Physical Exam findings, Assessment, Plan.
- `follow_up`: Reason for visit / interval history, relevant exam findings,
  Assessment, Plan.

**Compliance** (`nodes/compliance.py`) — example checks: Assessment/Plan not
supported by any documented finding; exam claims abnormal findings but no
vitals recorded when vitals would be expected.

**Coding clarity** (`nodes/coding_clarity.py`) — example check: Assessment
uses vague, non-specific language that would not map to a billable
diagnosis (e.g. "probably fine" vs. a stated working diagnosis).

### Merge (`nodes/merge.py`)

Deterministically carries through `completeness_issues`, `compliance_issues`,
`coding_clarity_issues` from the three worker outputs — identical pattern to
`agent/nodes/merge.py`. Only the `verdict` field (`approve` /
`needs_revision`) is trusted from the LLM's merge call.

### Render (`clinical_review/render.py`)

Jinja2 template, structurally identical to `agent/render.py`'s PR comment,
producing a "Clinical Note Review" markdown summary. Must include a fixed
footer: *"Generated by an illustrative demo agent. Not a clinical
compliance or coding authority."*

### CLI (`clinical_review/cli.py`)

```
clinical-review review --note-file <path> --encounter-type {new_patient,follow_up} \
  --trace-out trace.json --comment-out comment.md --model qwen2.5:7b
```
Same structure as `agent/cli.py`'s `review` command.

### Streamlit UI (`clinical_review/streamlit_app.py`)

Agent-only (no harness scoring — matches the PR agent's original
pre-harness-integration UI). Textarea for note text, a `st.selectbox` for
encounter type, "Run Review" button, then the rendered comment + expandable
Verdict JSON / trace stats / full trace JSON — same layout as the harness's
existing "Upload your own diff" mode.

### Harness adapter (`harness/adapters/clinical_review.py`)

```python
class ClinicalReviewExpected(BaseModel):
    verdict: Literal["approve", "needs_revision"]
    expected_check_agents: list[str]
    min_completeness_issues: int = 0
    min_compliance_issues: int = 0

class ClinicalReviewScorer:
    def __init__(self, judge: HallucinationJudge): ...
    def score(self, trace: RunTrace, expected: ClinicalReviewExpected) -> ScoreResult: ...
```

Mirrors `PrReviewScorer`: deterministic `task_success` check against
`expected.verdict` (+ optional `min_completeness_issues` /
`min_compliance_issues` thresholds), `tool_call_correctness` from comparing
`expected_check_agents` against `{s.agent_name for s in trace.steps}`,
`hallucination_rate` via the same domain-agnostic `HallucinationJudge`
(source material = `trace.input["note_text"]`, claim = each finding's
`description`), and the same `final.get("error")` crash-detection path as
`PrReviewScorer`.

### Harness CLI (`harness/cli.py`)

New command, additive only — the existing `gate` command is not modified,
to avoid any regression risk to the already-verified PR-review gate:

```
harness gate-clinical-review --testcases-dir clinical_testcases \
  --history-path docs/results/clinical_history.json \
  --report-out docs/results/clinical_report.html --commit-sha <sha>
```

Reads each `clinical_testcases/*/input.json` (`{"note_text": ..., "encounter_type": ...}`)
+ `expected.json`, calls `clinical_review.graph.run_review` (imported with
the same try/except ImportError → `click.ClickException` pattern
`_run_case` already uses for `agent`), scores with `ClinicalReviewScorer`,
and writes an independent report/history file (kept separate from the
PR-review ones — mixing two domains' pass rates into one number would be
meaningless).

### Testcases (`clinical_testcases/`)

Four fabricated notes with known ground truth:
1. `clean_new_patient/` — all required elements present → `approve`.
2. `incomplete_new_patient/` — missing ROS and PMH → `needs_revision`,
   `min_completeness_issues: 2`.
3. `clean_follow_up/` — complete follow-up note → `approve`.
4. `mismatched_follow_up/` — Assessment/Plan not supported by documented
   findings → `needs_revision`, `min_compliance_issues: 1`.

## Testing strategy

Same TDD approach as the original `agent`/`harness` plans: unit tests per
node/schema with fake LLM clients dispatching by `response_model` type (the
same pattern that fixed the concurrency race in `agent/tests/test_graph.py`
— `clinical_review/tests/test_graph.py` must use the same type-dispatch fake
client from the start, not order-based, since the graph is equally
concurrent). One integration test gated behind `-m integration` (real
Ollama). Harness adapter gets unit tests with `FakeHallucinationJudge`,
mirroring `harness/tests/adapters/test_pr_review.py`.

## Deliverables checklist

- [ ] `clinical_review/` package: schemas, nodes, graph, render, CLI, tests.
- [ ] `clinical_review/streamlit_app.py`.
- [ ] `harness/adapters/clinical_review.py` + tests.
- [ ] `harness gate-clinical-review` CLI command.
- [ ] `clinical_testcases/` with 4 fixtures.
- [ ] A real `harness gate-clinical-review` run against live Ollama, with
      committed `docs/results/clinical_report.html` /
      `clinical_history.json` (real numbers, not placeholders).
- [ ] README updated: new package documented, illustrative-rules disclaimer
      stated clearly, Streamlit UI instructions added.
