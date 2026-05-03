"""Command-line interface: argparse + dispatch into runner.execute_task."""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import sys
from pathlib import Path

from claude_plan_execute import colors
from claude_plan_execute.colors import bold, dim, green, red, status_color, yellow
from claude_plan_execute.config import resolve
from claude_plan_execute.deps import validate_dependencies
from claude_plan_execute.loader import (
    effective_defaults,
    find_tasks_file,
    load_tasks_data,
    load_tasks_file,
    render_decisions_index,
    scan_decisions,
)
from claude_plan_execute.runner import execute_task, handle_sigint
from claude_plan_execute.state import State


def cmd_decisions_index(args) -> int:
    """Rebuild `<decisions_dir>/index.md` from per-file frontmatter.

    Resolution order for `decisions_dir` (highest priority first):
      1. `--decisions-dir` flag.
      2. `defaults.decisions_dir` from the resolved tasks file
         (auto-detected via `find_tasks_file()` or `--tasks`).
      3. Hardcoded fallback `docs/decisions`.

    Returns 0 on success (file written), 1 on hard error (target
    directory missing). The presence of zero scannable decision files
    is NOT an error: the banner-only `index.md` is still written.
    """
    if args.dir:
        os.chdir(args.dir)

    decisions_dir: str | None = None
    if args.decisions_dir:
        decisions_dir = args.decisions_dir
    else:
        # Try to read decisions_dir from the resolved tasks file.
        tasks_path = Path(args.tasks) if args.tasks else find_tasks_file()
        if tasks_path and tasks_path.exists():
            try:
                raw = load_tasks_data(tasks_path)
                defaults = (raw or {}).get("defaults") or {}
                decisions_dir = defaults.get("decisions_dir")
            except Exception as exc:
                # Don't hard-fail on a malformed tasks file when the
                # user asked for the decisions subcommand — they may
                # be using --decisions-dir on its own. Log and fall
                # through to the hardcoded fallback.
                print(yellow(
                    f"Warning: could not read decisions_dir from "
                    f"{tasks_path}: {exc}"
                ))
        if not decisions_dir:
            from claude_plan_execute.config import HARDCODED_DEFAULTS
            decisions_dir = HARDCODED_DEFAULTS["decisions_dir"]

    d = Path(decisions_dir)
    if not d.exists() or not d.is_dir():
        print(red(f"Error: decisions directory not found: {d}"), file=sys.stderr)
        return 1

    entries = scan_decisions(d)
    rendered = render_decisions_index(entries)
    index_path = d / "index.md"
    index_path.write_text(rendered, encoding="utf-8")

    print(green(f"Wrote {len(entries)} entries to {index_path}"))
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="claude-plan-execute",
        description="Automated plan-review-implement workflow for Claude Code.",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    # 'decisions' subcommand and its 'index' action.
    decisions_parser = subparsers.add_parser(
        "decisions", help="Decision-log management."
    )
    decisions_actions = decisions_parser.add_subparsers(
        dest="action", required=True
    )
    index_parser = decisions_actions.add_parser(
        "index", help="Rebuild decisions/index.md from frontmatter."
    )
    index_parser.add_argument("--tasks", type=str,
        help="Path to tasks file (default: auto-detect)")
    index_parser.add_argument("--dir", type=str,
        help="Working directory (default: cwd)")
    index_parser.add_argument("--decisions-dir", type=str,
        help="Override decisions directory (otherwise resolved from tasks file)")

    # Existing top-level flags. UNCHANGED.
    parser.add_argument("--tasks", type=str, help="Path to tasks file (default: auto-detect)")
    parser.add_argument("--dir", type=str, help="Working directory (default: cwd)")
    parser.add_argument("--task", type=int, help="Run a specific task by ID")
    parser.add_argument("--skip-to", choices=["plan", "review", "implement"],
                        help="Skip to a specific phase")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without executing")
    parser.add_argument("--list", action="store_true", help="List all tasks with status")
    parser.add_argument("--status", action="store_true", help="Show detailed execution state")
    parser.add_argument("--reset", nargs="?", const="all", metavar="TASK_ID",
                        help="Reset state for a task (or all tasks if no ID given)")
    parser.add_argument("--model", type=str, help="Override model for all tasks")
    parser.add_argument("--effort", type=str, choices=["low", "medium", "high", "max"],
                        help="Override effort for all tasks")
    parser.add_argument("--skip-permissions", action="store_true",
                        help="Pass --dangerously-skip-permissions to claude")
    parser.add_argument("--verbose", action="store_true", help="Pass --verbose to claude")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    args = parser.parse_args()

    if getattr(args, "no_color", False):
        colors.set_use_color(False)

    # Subcommand dispatch. The handler returns an int exit code; we
    # propagate it via sys.exit because the entry-point script calls
    # main() without consuming its return value (see §D8). Using
    # sys.exit also matches main's other error branches.
    if args.command == "decisions" and args.action == "index":
        sys.exit(cmd_decisions_index(args))

    # Change working directory
    if args.dir:
        os.chdir(args.dir)

    # Check claude is available
    if not shutil.which("claude"):
        print(red("Error: 'claude' not found on PATH"), file=sys.stderr)
        sys.exit(1)

    # Find and load tasks file
    tasks_path = Path(args.tasks) if args.tasks else find_tasks_file()
    if not tasks_path or not tasks_path.exists():
        print(red(f"Error: No tasks file found. Create tasks.yaml or use --tasks"), file=sys.stderr)
        sys.exit(1)

    tasks_file_obj = load_tasks_file(tasks_path)
    file_defaults = effective_defaults(tasks_file_obj)
    from claude_plan_execute.gates import default_registry
    from claude_plan_execute.gates.invariants import (
        load_invariants,
        register_invariants_on,
    )
    invariants_path = file_defaults.get("invariants_file")
    register_invariants_on(
        default_registry(),
        load_invariants(invariants_path),
    )
    tasks: list[dict] = [t.model_dump(exclude_none=True) for t in tasks_file_obj.tasks]
    phases_by_id: dict[str, dict] = {
        p.id: p.model_dump(exclude_none=True) for p in tasks_file_obj.phases
    }
    task_class_models: dict[str, dict] = {
        name: cfg.model_dump(exclude_none=True)
        for name, cfg in (tasks_file_obj.defaults.task_class_models or {}).items()
    }

    # Resolve plans_dir
    cli_overrides = {
        "model": args.model,
        "effort": args.effort,
        "skip_permissions": args.skip_permissions,
        "verbose": args.verbose,
    }
    plans_dir = Path(resolve("plans_dir", {}, file_defaults, cli_overrides))
    state = State(plans_dir / "state.json")

    # Handle --reset
    if args.reset is not None:
        if args.reset == "all":
            state.reset()
            print(green("All tasks reset to pending."))
        else:
            state.reset(int(args.reset))
            print(green(f"Task {args.reset} reset to pending."))
        return

    # Handle --list
    if args.list:
        print(bold(f"\nTasks from {tasks_path}:\n"))
        for t in tasks:
            s = state.get(t["id"])["status"]
            deps = t.get("depends_on", [])
            dep_str = f" (depends on: {deps})" if deps else ""
            print(f"  {bold(str(t['id']).rjust(3))}. [{status_color(s)}] {t['title']}{dim(dep_str)}")
        print()
        return

    # Handle --status
    if args.status:
        print(bold(f"\nExecution state ({plans_dir / 'state.json'}):\n"))
        for t in tasks:
            entry = state.get(t["id"])
            s = entry["status"]
            extra = ""
            if "review_round" in entry:
                extra += f", review round {entry['review_round']}"
            if "plan_path" in entry:
                extra += f", plan: {entry['plan_path']}"
            print(f"  Task {bold(str(t['id']).rjust(3))}: [{status_color(s)}]{dim(extra)}")
        print()
        return

    # Validate dependencies and get execution order
    ordered_tasks = validate_dependencies(tasks)

    # Install SIGINT handler
    signal.signal(signal.SIGINT, handle_sigint)

    # Select tasks to run
    if args.task:
        task = next((t for t in tasks if t["id"] == args.task), None)
        if not task:
            print(red(f"Error: Task {args.task} not found"), file=sys.stderr)
            sys.exit(1)

        # Check dependencies
        deps = task.get("depends_on", [])
        unmet = [d for d in deps if state.get(d)["status"] != "done"]
        if unmet:
            print(red(f"Error: Task {args.task} has unmet dependencies: {unmet}"), file=sys.stderr)
            print(f"Run those tasks first, or use --reset to clear state.")
            sys.exit(1)

        execute_task(
            task, state, file_defaults, cli_overrides,
            phases_by_id=phases_by_id,
            task_class_models=task_class_models,
            skip_to=args.skip_to,
            dry_run=args.dry_run,
        )
    else:
        for task in ordered_tasks:
            s = state.get(task["id"])["status"]
            if s == "done":
                print(dim(f"\n  Skipping task {task['id']} (done)"))
                continue
            execute_task(
                task, state, file_defaults, cli_overrides,
                phases_by_id=phases_by_id,
                task_class_models=task_class_models,
                dry_run=args.dry_run,
            )

    print(f"\n{bold('=' * 60)}")
    print(bold("ALL TASKS COMPLETE"))
    print(bold("=" * 60))
