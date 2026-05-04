"""Command-line interface: argparse + dispatch into runner.execute_task."""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
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
from claude_plan_execute.services import (
    ServiceManager,
    ServiceUnavailableError,
)
from claude_plan_execute.state import State


def run_preflight(file_defaults: dict) -> int:
    """Run the project's baseline gates once, before the task loop.

    Returns 0 if every configured baseline gate passes; non-zero if any fail.
    The slate-runner aborts on non-zero so a single broken pre-existing test
    can't cascade into N blocked tasks. Bypass with --skip-preflight.

    Currently checks `tests` (always, when test_command is set) and `lint`
    (when defaults.lint_command is set). Untranslated-strings is project-
    specific and not part of the baseline check; nor is design-fidelity
    (it requires a running dev server).
    """
    checks: list[tuple[str, str]] = []
    test_cmd = file_defaults.get("test_command")
    lint_cmd = file_defaults.get("lint_command")
    if test_cmd:
        checks.append(("tests", test_cmd))
    if lint_cmd:
        checks.append(("lint", lint_cmd))
    if not checks:
        return 0

    print(bold("\n--- Preflight: baseline gates ---"))
    failed: list[tuple[str, str, str]] = []
    for name, cmd in checks:
        print(dim(f"  {name}: {cmd}"))
        proc = subprocess.run(
            ["bash", "-c", cmd], capture_output=True, text=True,
        )
        if proc.returncode == 0:
            print(green(f"    PASS"))
        else:
            tail = (proc.stdout or "") + (proc.stderr or "")
            print(red(f"    FAIL (exit {proc.returncode})"))
            failed.append((name, cmd, tail[-3000:]))

    if not failed:
        return 0

    print(red(
        f"\nPreflight detected {len(failed)} broken baseline gate"
        f"{'s' if len(failed) != 1 else ''}. Refusing to start the slate.\n"
        f"A pre-existing failure here would otherwise block every task in\n"
        f"the slate. Fix the baseline first, then re-run.\n"
        f"(To bypass: --skip-preflight.)\n"
    ))
    for name, cmd, tail in failed:
        print(red(f"=== {name} ({cmd}) — last 3000 chars ==="))
        print(tail)
        print()
    return 1


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


def _slate_preflight_audit(
    tasks: list[dict],
    file_defaults: dict,
    cli_overrides: dict,
    phases_by_id: dict[str, dict],
    task_class_models: dict[str, dict],
) -> tuple[bool, list]:
    """T9 §D11: walk every task in the slate and decide:

    - `needs_gh`: True iff at least one task resolves to auto_pr=True
      OR auto_merge=True (forces the gh-PATH check).
    - `auto_pr_without_branch`: list of task IDs that resolve to
      auto_pr/auto_merge True but branch_per_task False (the post-
      Phase-4 PR hook is gated on `branch_per_task and auto_pr`).

    Uses the same `resolve(...)` helper the runner uses, so per-task
    overrides are honored (a task that sets `auto_pr: false` even
    when project defaults say true is skipped).
    """
    needs_gh = False
    auto_pr_without_branch: list = []
    for t in tasks:
        phase_dict_t: dict = {}
        if phases_by_id and t.get("phase") in phases_by_id:
            phase_dict_t = dict(
                phases_by_id[t["phase"]].get("defaults") or {}
            )
        task_class_dict_t: dict = {}
        tc_name = t.get("task_class") or phase_dict_t.get("task_class")
        if task_class_models and tc_name in task_class_models:
            task_class_dict_t = dict(task_class_models[tc_name])
        auto_pr_t = bool(resolve(
            "auto_pr", t, file_defaults, cli_overrides,
            phase=phase_dict_t, task_class=task_class_dict_t,
        ))
        auto_merge_t = bool(resolve(
            "auto_merge", t, file_defaults, cli_overrides,
            phase=phase_dict_t, task_class=task_class_dict_t,
        ))
        branch_per_task_t = bool(resolve(
            "branch_per_task", t, file_defaults, cli_overrides,
            phase=phase_dict_t, task_class=task_class_dict_t,
        ))
        if auto_pr_t or auto_merge_t:
            needs_gh = True
            if not branch_per_task_t:
                auto_pr_without_branch.append(t["id"])
    return needs_gh, auto_pr_without_branch


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
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the baseline tests/lint preflight check before the task loop")
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

    # v0.3 task 8: hoist cli_overrides + plans_dir upward (Plan §D5.1)
    # so the ServiceManager has plans_dir for services.json placement.
    # Both names move together — resolve() reads cli_overrides.
    cli_overrides = {
        "model": args.model,
        "effort": args.effort,
        "skip_permissions": args.skip_permissions,
        "verbose": args.verbose,
    }
    plans_dir = Path(resolve("plans_dir", {}, file_defaults, cli_overrides))
    state = State(plans_dir / "state.json")

    # v0.3 task 8 (Plan §D5.1): construct the ServiceManager. Empty
    # spec list when no `services:` block is declared — manager is
    # still instantiated but has no work to do (zero overhead). The
    # manager's __init__ does NOT write services.json or mkdir
    # services/ (Plan §D2.3 — read-only commands stay read-only).
    service_specs: list[dict] = [
        s.model_dump(exclude_none=False)
        for s in (tasks_file_obj.services or [])
    ]
    service_manager = ServiceManager(service_specs, plans_dir)

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

    # T9 (v0.3 §2.12) §D11: gh CLI preflight + auto_pr/branch_per_task
    # coherence check. Fires only when at least one task in the slate
    # resolves to auto_pr=True or auto_merge=True; projects that don't
    # use the auto-PR machinery aren't forced to install gh. The
    # coherence check fires before the gh-PATH check so a misconfigured
    # slate gets a precise error regardless of whether gh is installed.
    needs_gh, auto_pr_without_branch = _slate_preflight_audit(
        tasks, file_defaults, cli_overrides,
        phases_by_id, task_class_models,
    )
    if auto_pr_without_branch:
        print(red(
            f"Error: tasks {auto_pr_without_branch} resolve to auto_pr "
            f"or auto_merge = True but branch_per_task = False. The "
            f"post-Phase-4 PR hook is gated on `branch_per_task and "
            f"auto_pr`; set `branch_per_task: true` (project, phase, "
            f"task_class, or per-task) to enable PR creation."
        ), file=sys.stderr)
        sys.exit(1)
    if needs_gh and not shutil.which("gh"):
        print(red(
            "Error: 'gh' not found on PATH but auto_pr or auto_merge "
            "is enabled. Install GitHub CLI (https://cli.github.com) "
            "and run `gh auth login`."
        ), file=sys.stderr)
        sys.exit(1)

    # Validate dependencies and get execution order
    ordered_tasks = validate_dependencies(tasks)

    # Preflight: refuse to start when the baseline tests/lint already fail.
    # A single pre-existing failure here would otherwise cascade through
    # every task's gate phase. Bypass with --skip-preflight, or skip
    # implicitly when --task / --reset / --list / --status / --dry-run.
    if not args.skip_preflight and not args.task and not args.dry_run:
        if run_preflight(file_defaults) != 0:
            sys.exit(1)

    # Install SIGINT handler
    signal.signal(signal.SIGINT, handle_sigint)

    # v0.3 task 8 (Plan §D5.2): compute the union of services_required
    # across every task in the slate, even in --task mode (see §11.14
    # for the documented rationale — run-scope services are reused
    # across tasks). The loop re-implements
    # runner._resolve_services_required by hand; if the layering rules
    # ever grow, both sites must move together.
    union_required: set[str] = set()
    for t in ordered_tasks:
        phase_id = t.get("phase")
        pdict = (
            (phases_by_id.get(phase_id) or {}).get("defaults") or {}
            if phase_id else {}
        )
        for name in (
            list(file_defaults.get("services_required") or [])
            + list(pdict.get("services_required_extra") or [])
            + list(t.get("services_required_extra") or [])
        ):
            union_required.add(name)

    run_scope_required = [
        n for n in service_manager.names()
        if n in union_required
        and service_manager.spec_lifetime(n) == "run"
    ]

    # v0.3 task 8 (Plan §D5.3): wrap run-scope ensure + the task loop
    # in try/finally so tear_down_all() runs on SIGINT, exit, or
    # uncaught exception. The dry-run guard around ensure() mirrors
    # the §D4.4 / §D4.5 guards in the runner — no service work fires
    # in dry-run mode at any layer.
    prev_phase: str | None = None
    try:
        if not args.dry_run:
            try:
                service_manager.ensure(run_scope_required)
            except ServiceUnavailableError as exc:
                print(red(
                    f"Error: could not start run-scope services: {exc}"
                ), file=sys.stderr)
                print(red("Aborting before task loop."), file=sys.stderr)
                sys.exit(1)

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
                service_manager=service_manager,
            )
        else:
            for task in ordered_tasks:
                s = state.get(task["id"])["status"]
                if s == "done":
                    print(dim(f"\n  Skipping task {task['id']} (done)"))
                    continue
                # Phase-boundary detection (Plan §D5.3): on phase
                # change, tear down the previous phase's services and
                # ensure the new phase's. Tear-downs are unconditional
                # (no-op on stopped services); ensure() is dry-run
                # guarded.
                cur_phase = task.get("phase")
                if cur_phase != prev_phase:
                    if prev_phase is not None:
                        service_manager.tear_down_for_scope("phase")
                    if cur_phase is not None:
                        pdict = (
                            (phases_by_id.get(cur_phase) or {})
                            .get("defaults") or {}
                        )
                        phase_required = list(
                            pdict.get("services_required_extra") or []
                        )
                        phase_scope = [
                            n for n in phase_required
                            if service_manager.spec_lifetime(n) == "phase"
                        ]
                        if not args.dry_run:
                            try:
                                service_manager.ensure(phase_scope)
                            except ServiceUnavailableError as exc:
                                print(red(
                                    f"Error: phase '{cur_phase}' service "
                                    f"start failed: {exc}"
                                ), file=sys.stderr)
                                prev_phase = cur_phase
                                continue
                    prev_phase = cur_phase
                execute_task(
                    task, state, file_defaults, cli_overrides,
                    phases_by_id=phases_by_id,
                    task_class_models=task_class_models,
                    dry_run=args.dry_run,
                    service_manager=service_manager,
                )
    finally:
        # Unconditional cleanup: idempotent on stopped services, so a
        # dry-run that started nothing pays zero cost here.
        service_manager.tear_down_all()

    print(f"\n{bold('=' * 60)}")
    print(bold("ALL TASKS COMPLETE"))
    print(bold("=" * 60))
