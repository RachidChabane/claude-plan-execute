# claude-plan-execute

Automated plan-review-implement workflow for Claude Code. Reads a task file, then for each task:

1. **Plan** — Claude produces a detailed implementation plan.
2. **Review** — A second Claude reviews it. If rejected, a third revises it. Loop until approved.
3. **Implement** — Claude implements the approved plan, runs tests, and commits.

## Install

```bash
# 1. Install the package (puts `claude_plan_execute` on your Python path).
pip install -e .

# 2. Symlink the entry script onto your PATH:
ln -s "$(pwd)/claude-plan-execute" ~/.local/bin/claude-plan-execute
ln -s "$(pwd)/workflow-server.py" ~/.local/bin/claude-workflow-dashboard

#    (Alternative: symlink from bin/ instead of the repo root.)
ln -s "$(pwd)/bin/claude-plan-execute" ~/.local/bin/claude-plan-execute
```

Optional dashboard dependencies:
```bash
pip install -e '.[dashboard]'   # FastAPI + uvicorn for workflow-server.py
```

For development:
```bash
pip install -e '.[dev]'         # pytest
pytest -q
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

## Task File Format

```yaml
version: 1

defaults:
  model: opus
  effort: max
  max_review_rounds: 3
  test_command: "pytest"
  plans_dir: "plans"

tasks:
  - id: 1
    title: "Short descriptive title"
    description: |
      Detailed description of what to implement.
    key_files:         # Non-exhaustive starting points
      - src/foo.py
    commit_message: "feat: add foo"
    depends_on: []     # Task IDs that must complete first
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

Progress is tracked in `{plans_dir}/state.json`. If interrupted (Ctrl+C), the script saves state and prints a resume command. On restart, completed tasks are skipped.

## Requirements

- Python 3.11+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)
- Runtime deps installed via `pip install -e .` (PyYAML).
- Dashboard deps installed via `pip install -e '.[dashboard]'` (FastAPI, uvicorn).
