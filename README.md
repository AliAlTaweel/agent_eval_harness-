# pr-review-eval

A decoupled PR-review agent and (future) eval harness, sharing a common
`trace_schema` contract so review runs can be recorded, replayed, and scored
independently of the agent implementation.

This repo currently contains:

- `trace_schema/` — the shared `RunTrace`/`Step`/`ToolCall` pydantic contract
  that any review run (agent or harness) is recorded against.
- `agent/` — a LangGraph multi-agent PR reviewer. It runs security, style,
  and test-coverage review nodes against a diff, merges their findings into a
  verdict via a local Ollama model, and emits a `trace_schema.RunTrace` plus
  a markdown PR comment.

The eval harness (`harness/`) that will replay and score traces against the
`trace_schema` contract is not built yet — this README will grow a harness
section when that lands.

## Running locally

```bash
uv sync
ollama pull qwen2.5:7b
uv run agent review --diff-file <path-to-diff-file>
```

This writes `trace.json` (a `RunTrace`) and `comment.md` (the PR comment) to
the current directory by default; override with `--trace-out` /
`--comment-out`. Run `uv run pytest` from the repo root to run the full test
suite across `trace_schema/tests/` and `agent/tests/` (pass
`-m "not integration"` to skip the test that requires a running local
Ollama).

## Known limitation: self-hosted runner required

`.github/workflows/pr-review.yml` runs on `runs-on: self-hosted` because the
review step shells out to a local Ollama instance (`qwen2.5:7b`). **This
workflow will not work on GitHub-hosted runners** — they cannot reach a
local Ollama instance, and there is no hosted Ollama equivalent wired up.
To use the workflow, register a self-hosted runner that has Ollama installed
and the `qwen2.5:7b` model already pulled.
