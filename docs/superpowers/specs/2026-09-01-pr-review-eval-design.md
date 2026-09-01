# PR Review Bot + Eval Harness — Design

Date: 2026-09-01
Status: Approved for implementation planning

## Goal

A portfolio project with two genuinely decoupled halves:

1. A multi-agent GitHub PR code-review bot (LangGraph).
2. A generic, agent-agnostic eval harness that scores agent runs.

The only thing the two halves share is a trace schema. The harness must never
import or know about the agent's internals; the agent must never import the
harness. This is the central design constraint and the reason the project is
built in the order below.

## Stack decisions

- **Language/runtime:** Python 3.12.
- **Package manager:** `uv`, using a workspace with three member packages.
- **Orchestration:** LangGraph.
- **LLM:** Local via Ollama, model `qwen2.5:7b` — used for the three review
  agents, the supervisor/critic, and the LLM-as-judge scorer. Chosen over
  llama3.1:8b for stronger structured-output/tool-calling reliability at this
  size, which matters because every agent returns structured JSON findings
  and the harness parses them.
- **CI:** GitHub Actions on a **self-hosted runner** (the developer's
  machine, with Ollama installed) — both the PR-review Action and the
  harness-gating job run there. This is a known limitation to call out in
  the README: the Action only runs while the runner is registered and
  online. Portable/hosted-runner execution (installing Ollama in-job, or
  swapping to a hosted LLM provider) is explicitly out of scope for now.
- **Dashboard:** Static, self-contained HTML report (tables + a trend chart),
  generated per run and also emitted as a CI job artifact.

## Repo layout

Monorepo, uv workspace, `src/` layout per package:

```
code_review/
├── pyproject.toml              # workspace root
├── trace_schema/
│   ├── pyproject.toml
│   ├── src/trace_schema/
│   │   ├── __init__.py
│   │   └── models.py           # pydantic RunTrace / Step / ToolCall
│   └── tests/
├── agent/
│   ├── pyproject.toml
│   ├── action.yml               # composite GitHub Action
│   ├── src/agent/
│   │   ├── graph.py             # LangGraph definition
│   │   ├── nodes/               # supervisor, security, style, test_coverage, merge
│   │   ├── tracing.py           # shared Step-recording helper
│   │   ├── render.py            # verdict JSON -> markdown PR comment
│   │   └── cli.py                # `agent review --diff-file / --pr-context`
│   └── tests/
├── harness/
│   ├── pyproject.toml
│   ├── src/harness/
│   │   ├── collector.py         # loads RunTrace JSON
│   │   ├── scorer.py             # Scorer protocol
│   │   ├── gate.py               # pass/fail vs baseline
│   │   ├── report.py             # static HTML report generation
│   │   ├── history.py            # trend-over-commits JSON store
│   │   └── adapters/
│   │       └── pr_review.py      # the only PR-review-aware code in harness/
│   └── tests/
├── testcases/                    # synthetic diffs + expected.json per case
├── docs/
└── .github/workflows/
    ├── pr-review.yml              # runs the agent on incoming PRs
    └── eval-gate.yml              # runs harness against testcases/, gates on regression
```

## Phase 0 — `trace_schema/`

Pydantic models, single source of truth, JSON Schema exported from them
(not hand-maintained separately):

```python
class ToolCall(BaseModel):
    name: str
    args: dict
    result: Any

class Step(BaseModel):
    agent_name: str
    action: str
    tool_calls: list[ToolCall]
    output: Any
    started_at: datetime
    ended_at: datetime
    tokens_in: int
    tokens_out: int
    cost_usd: float

class RunTrace(BaseModel):
    schema_version: str
    run_id: str
    input: Any
    steps: list[Step]
    final_output: Any
    started_at: datetime
    ended_at: datetime
    total_tokens: int
    total_cost_usd: float
```

`schema_version` is bumped on breaking changes; both `agent/` and
`harness/` depend on this package as a normal path dependency inside the
workspace, never vendor or copy it.

## Phase 1 — `agent/`

**Graph:** `supervisor` fans out in parallel to `security_review`,
`style_review`, `test_coverage`; each is an independent LangGraph node that
reviews the diff and returns a structured findings object (pydantic-typed
LLM output via Ollama). A `merge` node (the critic) combines the three
outputs into the final verdict.

**Structured output:**
```json
{
  "security_issues": [...],
  "style_issues": [...],
  "test_coverage_gap": true,
  "verdict": "approve" | "request_changes"
}
```
plus a markdown PR comment rendered from that JSON via a template.

**Tracing:** one shared helper (`tracing.py`) wraps each node's LLM call,
timing it and appending a `Step` (with tool calls, token counts, cost) to
the in-flight `RunTrace`. Agents call this helper; they don't hand-roll
trace-writing themselves.

**Entry points:**
- CLI: `agent review --diff-file path/to.diff` or `--pr-context` (reads
  `GITHUB_*` env vars set by Actions). This is the primary tested surface.
- `action.yml`: composite Action, triggered on `pull_request`, checks out
  the PR, computes the diff (`git diff` against the base ref), calls the
  CLI, posts the rendered comment via `GITHUB_TOKEN`, and uploads the
  trace as a job artifact.

## Phase 2 — `harness/`

**Core stays generic.** `collector.py` loads any `RunTrace` JSON.
`scorer.py` defines:

```python
class Scorer(Protocol):
    def score(self, trace: RunTrace, expected: Any) -> ScoreResult: ...
```

`ScoreResult` carries the five dimensions: task success (bool/score),
tool-call correctness (right tool, right args, right order — diffed against
an expected call sequence), hallucination rate (judge-scored: does the
output claim things not supported by the diff/tool outputs), latency,
cost. `gate.py` and `report.py` operate only on `ScoreResult` + `RunTrace`
— no PR-review vocabulary anywhere in `harness/` core.

**LLM-as-judge:** a small Ollama-backed judge call (same `qwen2.5:7b`)
used specifically for the hallucination-rate dimension and for any
free-text task-success grading where exact-match isn't possible; deterministic
comparisons (verdict match, tool-call sequence match) don't need the judge.

**Adapter:** `harness/adapters/pr_review.py` implements `Scorer` for this
agent — it knows about `verdict`, `security_issues`, etc., and is the *only*
place that PR-review-specific knowledge lives outside `agent/`.

**Test-case suite** (`testcases/`): ~6–8 synthetic diffs, each a directory
with `diff.patch` + `expected.json` (expected verdict, expected findings
categories, expected tool-call sequence). Covers: SQL injection, XSS,
hardcoded secret, clean diff (expect approve), diff with no tests added
(expect `test_coverage_gap: true`), style-only violations, a diff mixing a
real bug with noise (tests hallucination resistance).

**Gate:** `harness gate --testcases testcases/ --baseline history.json`
runs the agent (or replays saved traces, for faster CI) against every case,
scores each with the adapter, aggregates pass rate / hallucination rate /
avg cost / avg latency, compares pass rate to the last recorded baseline in
`history.json`, exits non-zero on regression. `report.py` renders the same
aggregate + the historical trend into a static HTML file.

## CI (dogfooding)

Two workflows, both on the self-hosted runner:
- `pr-review.yml`: runs `agent/`'s Action on PRs to this repo.
- `eval-gate.yml`: on push/PR, runs `harness gate` against `testcases/`
  using the current `agent/` code, updates `history.json`, publishes the
  HTML report as a build artifact, fails the build on regression. This is
  the harness gating the agent's own repo, satisfying the dogfooding
  deliverable.

## Testing strategy

- `trace_schema/`: schema validation round-trip tests.
- `agent/`: unit tests per node with a fake/stubbed LLM client (no live
  Ollama calls in unit tests); one slow/integration test that runs the full
  graph against Ollama on a small fixture diff.
- `harness/`: unit tests for `collector`, `gate`, `aggregator` against
  hand-written fixture traces (not real agent output) — this is what
  proves the harness has no hidden dependency on `agent/`. The
  `pr_review` adapter gets its own tests separately.

## Deliverables checklist (from the request)

- [ ] `trace_schema/`, `agent/`, `harness/` as decoupled packages.
- [ ] README explaining the architecture and the decoupling rationale.
- [ ] `.github/workflows/` showing the harness gating the agent's repo.
- [ ] `harness report` output with real numbers from the synthetic test
      suite, checked into `docs/` or linked from the README (not
      placeholders).
