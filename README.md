# claude-plan-execute

Automated plan-review-implement workflow for Claude Code. Reads a task file, then for each task:

1. **Plan** — Claude produces a detailed implementation plan.
2. **Review** — A second Claude reviews it. If rejected, a third revises it. Loop until approved.
3. **Implement** — Claude implements the approved plan, runs tests, and commits.

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

## v2 Features

- **Phases** — group tasks and set phase-level defaults (model, effort, task class)
- **Task classes** — per-class model/effort configuration via `defaults.task_class_models`
- **Caps** — hard limits on phase duration, review rounds, implement turns, and total attempts
- **Gate registry** — pluggable pre/post gates: `tests`, `lint`, `invariant-grep`, `test-critique`, `security-auditor`, `untranslated-strings`; shell and Claude-agent gate types
- **Persona file** — `docs/persona.md` auto-injected into every prompt as persona context
- **Decision log** — `docs/decisions/` files surfaced as context during implement
- **Failure records** — per-attempt structured records stored in `state.json`
- **Resolved config** — `plans/task-{id}/resolved-config.json` shows the exact config used (model, effort, gates, caps, provenance)
- **Dashboard** — web UI at `http://localhost:8420`, started via `claude-workflow-dashboard`

## Migration

If you have an existing `version: 1` tasks file, see [docs/migration-v2.md](docs/migration-v2.md) for a side-by-side upgrade guide. Version 1 files continue to work; v2 features are opt-in.

## Task File Format

```yaml
version: 2

defaults:
  model: opus
  effort: max
  plans_dir: plans
  test_command: "pytest -q"
  caps:                          # hard limits (all optional)
    max_minutes_per_phase: 45
    max_review_rounds: 3
    max_implement_turns: 200
    max_attempts: 3
  on_max_review_rounds: fail     # or "auto_approve" (v1 legacy default)
  task_class_models:             # per-class model/effort map
    schema: { model: opus,   effort: max }
    ui:     { model: sonnet, effort: medium }

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
  - id: 2
    phase: foundation
    title: "Second task"
    description: "Builds on task 1."
    depends_on: [1]
    task_class: ui               # explicit per-task override
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
```

## State & Resume

Progress is tracked in `{plans_dir}/state.json`. If interrupted (Ctrl+C), the script saves state and prints a resume command. On restart, completed tasks are skipped. The state file is versioned (`schema_version: 2`); v1 state files self-heal on first access.

## Requirements

- Python 3.11+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)
- Runtime deps installed via `pip install -e .` (PyYAML, Pydantic ≥ 2.0)
- Dashboard deps: `pip install -e '.[dashboard]'` (FastAPI, uvicorn)
