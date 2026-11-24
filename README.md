# claude-plan-execute

Automated plan-review-implement workflow for Claude Code. Reads a task file, then for each task:

1. **Plan** — Claude produces a detailed implementation plan.
2. **Review** — A second Claude reviews it. If rejected, a third revises it. Loop until approved.
3. **Implement** — Claude implements the approved plan, runs tests, and commits.

> This runner follows [Claude Autopilot](https://github.com/RachidChabane/cc-mvp-scaffold).
> The idea reached its final form in
> [Agent Factory](https://github.com/RachidChabane/agent-factory).

## Install

```bash
# Core package:
pip install -e .

# With dashboard (FastAPI + uvicorn):
pip install -e '.[dashboard]'

# Full dev stack (adds pytest):
pip install -e '.[dev]'
```

The CLI entry points are not yet declared in `pyproject.toml`. Symlink them onto your PATH manually:

```bash
ln -s "$(pwd)/bin/claude-plan-execute" ~/.local/bin/claude-plan-execute
ln -s "$(pwd)/bin/claude-workflow-dashboard" ~/.local/bin/claude-workflow-dashboard
```

## Usage

### CLI

```bash
# From any project with a tasks.yaml:
claude-plan-execute --list
claude-plan-execute --task 1 --skip-permissions
claude-plan-execute --dry-run

# Point at a different project:
claude-plan-execute --dir /path/to/project --tasks docs/tasks.yaml
```

### Dashboard

```bash
claude-workflow-dashboard --dir /path/to/project --tasks tasks.yaml
# Open http://localhost:8420
```

Configure model, effort, review rounds, and permissions in the sidebar. Click Run All or run individual tasks.

## Features

### v0.2 (baseline)
- **Phases** — group tasks and set phase-level defaults (model, effort, task class)
- **Task classes** — per-class model/effort configuration via `defaults.task_class_models`
- **Caps** — hard limits on phase duration, review rounds, implement turns, and total attempts
- **Gate registry** — pluggable post-implement gates: `tests`, `lint`, `invariant-grep`,
  `test-critique`, `security-auditor`, `untranslated-strings`; shell and Claude-agent gate types
- **Persona file** — `docs/persona.md` auto-injected into every prompt as persona context
- **Decision log** — `docs/decisions/` files surfaced as context during implement
- **Failure records** — per-attempt structured records stored in `state.json`
- **Resolved config** — `plans/task-{id}/resolved-config.json` shows the exact config used
- **Dashboard** — web UI at `http://localhost:8420`

### v0.3 (current)
- **Services** — declarative lifecycle for dev servers and databases (`services:` block);
  HTTP/TCP/shell ready-checks; restart from dashboard
- **Branch per task** — automatic git branch + optional GitHub PR/merge per task
  (`branch_per_task: true`, `auto_pr: true`, `auto_merge: true`)
- **Human checkpoints** — `kind: human` gate in `invariants.yaml`; Approve/Reject via
  dashboard or `claude-plan-execute approve/reject` CLI subcommands
- **Structured findings** — agent gates emit a JSON appendix
  (`<!-- BEGIN_FINDINGS_JSON -->`) for machine-readable cross-task analysis
- **Aggregation panel** — `aggregations.yaml` defines regex or JSON-field rollups
  across all tasks; results surface in the dashboard
- **WIP-commit resume** — if an implement is interrupted, progress.md tracks remaining
  steps; the runner resumes from the last incomplete step on restart
- **Diff precompute** — `{diff_path}` template variable available in all custom
  gate prompt templates; runner writes the task diff to a temp file before gate execution
- **Token observability** — per-phase token counts appended to
  `plans/task-{id}/usage.jsonl`; per-task summary in `state.json`
- **Rate-limit retry** — automatic exponential backoff on 429 errors
  (configurable via `caps.max_rate_limit_retries`)
- **`--explain TASK_ID`** — dry-run dump of resolved model, gates, when-evaluations,
  and caps for any task without running it
- **`touches:<glob>` gate when-filter** — activate gates only when a task's
  `key_files` match a pathspec pattern
- **Per-task `gates_extra`** — additive gate list at phase or task level without
  overriding project defaults
- **Cross-task memory** — per-project memory directory (`~/.claude/projects/.../memory/`)
  with type-aware lifecycle pruning and citation-count tracking

## Migration

- **v1 → v2**: see [docs/migration-v2.md](docs/migration-v2.md)
- **v2 → v0.3**: see [docs/migration-v3.md](docs/migration-v3.md)
  (all new primitives are opt-in; no v0.2.0 task file requires changes)

## Recipes

Step-by-step guides for common patterns:

| Recipe | Covers |
|--------|--------|
| [services.md](docs/recipes/services.md) | Postgres, Redis, dev-server lifecycle |
| [branch-per-task.md](docs/recipes/branch-per-task.md) | Enabling branch_per_task, prerequisites |
| [human-checkpoint.md](docs/recipes/human-checkpoint.md) | Registering human gates, Approve/Reject UX |
| [structured-findings.md](docs/recipes/structured-findings.md) | JSON appendix from agent gates |
| [aggregation.md](docs/recipes/aggregation.md) | Writing aggregations.yaml (regex vs JSON field) |
| [resume-on-context-fill.md](docs/recipes/resume-on-context-fill.md) | WIP-commit + progress.md convention |
| [rate-limit-and-tokens.md](docs/recipes/rate-limit-and-tokens.md) | Retry tuning, reading usage.jsonl |
| [custom-agent-gates.md](docs/recipes/custom-agent-gates.md) | Adding custom gates via invariants.yaml |
| [multi-suite-tests.md](docs/recipes/multi-suite-tests.md) | Multiple test suites as separate shell gates |
| [persona-vs-claude-md.md](docs/recipes/persona-vs-claude-md.md) | Prompt layer precedence |
| [migration-via-task-implementer.md](docs/recipes/migration-via-task-implementer.md) | DB migration pattern with services |
| [cross-task-memory.md](docs/recipes/cross-task-memory.md) | Memory lifecycle, citations, prune pass |

## Task File Format

```yaml
version: 2

defaults:
  model: opus
  effort: max
  plans_dir: plans
  test_command: "pytest -q"
  branch_per_task: false          # set true to enable per-task branches + PRs
  caps:
    max_minutes_per_phase: 45
    max_review_rounds: 3
    max_implement_turns: 200
    max_attempts: 3
    max_rate_limit_retries: 3     # v0.3: exponential backoff retry count
  task_class_models:
    schema: { model: opus,   effort: max }
    ui:     { model: sonnet, effort: medium }

services:                         # v0.3: declare dev servers / DBs
  - name: postgres
    command: "docker compose up -d db"
    ready_check: { kind: tcp, port: 5432 }
    lifetime: run

phases:
  - id: foundation
    title: "Foundation"
    defaults: { task_class: schema }

tasks:
  - id: 1
    phase: foundation
    title: "Short descriptive title"
    description: |
      Detailed description of what to implement.
    key_files:
      - src/foo.py
    commit_message: "feat: add foo"
    gates_extra: ["security-auditor"]   # v0.3: append gates beyond defaults
```

Place as `tasks.yaml` in the project root (or `docs/tasks.yaml`). Use `--tasks` for custom paths.

## CLI Reference

```
claude-plan-execute [options]

  --tasks FILE          Task file path (default: tasks.yaml)
  --dir DIR             Working directory (default: cwd)
  --task ID             Run a specific task
  --skip-to PHASE       Skip to: plan, review, implement
  --dry-run             Print prompts without executing
  --list                List tasks with status
  --status              Show detailed execution state
  --reset [TASK_ID]     Reset one or all tasks
  --model MODEL         Override model (opus, sonnet, haiku)
  --effort EFFORT       Override effort (low, medium, high, max)
  --skip-permissions    Pass --dangerously-skip-permissions to Claude
  --verbose             Pass --verbose to Claude
  --no-color            Disable colored output
  --explain TASK_ID     Show resolved config for a task without running it

Subcommands:
  approve TASK_ID GATE  Write approval sentinel for a human-checkpoint gate
  reject TASK_ID GATE   Write rejection sentinel for a human-checkpoint gate
```

## State & Resume

Progress is tracked in `{plans_dir}/state.json`. If interrupted (Ctrl+C), the script saves state and prints a resume command. On restart, completed tasks are skipped. The state file is versioned (`schema_version: 3`); v1 and v2 state files self-heal on first access.

## Requirements

- Python 3.11+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)
- Runtime deps installed via `pip install -e .` (PyYAML, Pydantic ≥ 2.0)
- Dashboard deps: `pip install -e '.[dashboard]'` (FastAPI, uvicorn)

## License

Copyright (C) 2026 Rachid Chabane.

[AGPL-3.0](LICENSE): use it, fork it and change it; a version you distribute or offer
to others over a network stays open under the same license, with its notices.
