"""Per-task plan / review / implement orchestrator."""
from __future__ import annotations

import sys
from pathlib import Path

from claude_plan_execute.claude import _kill_child, run_claude
from claude_plan_execute.colors import bold, dim, green, red, yellow
from claude_plan_execute.config import resolve
from claude_plan_execute.prompts import (
    build_implement_prompt,
    build_plan_prompt,
    build_review_prompt,
    build_revise_prompt,
)
from claude_plan_execute.state import State


def check_verdict(review_path: Path) -> tuple[bool, str]:
    """Parse review file for APPROVED/NEEDS_REVISION verdict."""
    if not review_path.exists():
        return False, "Review file not found"
    content = review_path.read_text()
    lower = content.lower()
    approved = "verdict: approved" in lower or "verdict:approved" in lower
    return approved, content


# Globals for SIGINT handler — execute_task sets, handle_sigint reads.
_current_state: State | None = None
_current_task_id: int | str = 0
_current_phase: str = "plan"


def execute_task(
    task: dict,
    state: State,
    file_defaults: dict,
    cli: dict,
    *,
    phases_by_id: dict[str, dict] | None = None,
    task_class_models: dict[str, dict] | None = None,
    skip_to: str | None = None,
    dry_run: bool = False,
):
    global _current_state, _current_task_id, _current_phase
    _current_state = state
    _current_task_id = task["id"]

    # Build per-task layer dicts. Empty when None (v1 path) or unmatched.
    phase_dict: dict = {}
    if phases_by_id:
        phase_id = task.get("phase")
        if phase_id and phase_id in phases_by_id:
            phase_dict = dict(phases_by_id[phase_id].get("defaults") or {})

    task_class_dict: dict = {}
    if task_class_models:
        class_name = task.get("task_class") or phase_dict.get("task_class")
        if class_name and class_name in task_class_models:
            task_class_dict = dict(task_class_models[class_name])

    def R(key: str) -> object:
        return resolve(
            key, task, file_defaults, cli,
            phase=phase_dict, task_class=task_class_dict,
        )

    tid = task["id"]
    plans_dir = Path(R("plans_dir"))
    task_dir = plans_dir / f"task-{tid}"
    task_dir.mkdir(parents=True, exist_ok=True)
    plan_path = task_dir / "plan.md"

    model = R("model")
    effort = R("effort")
    max_review_rounds = int(R("max_review_rounds"))
    plan_timeout = int(R("plan_timeout_minutes"))
    review_timeout = int(R("review_timeout_minutes"))
    impl_timeout = int(R("implement_timeout_minutes"))
    impl_turns = int(R("implement_max_turns"))
    test_command = R("test_command")
    skip_perms = cli.get("skip_permissions", False)
    verbose = cli.get("verbose", False)

    commit_msg = task.get("commit_message", f"task-{tid}: {task['title']}")

    print(f"\n{bold('=' * 60)}")
    print(bold(f"TASK {tid}: {task['title']}"))
    print(bold("=" * 60))

    run_kwargs = dict(model=model, effort=effort, skip_permissions=skip_perms, verbose=verbose)

    # --- PHASE 1: PLAN ---
    if skip_to not in ("review", "implement"):
        _current_phase = "plan"
        print(f"\n{bold('--- Phase 1: Planning ---')}")
        state.set_status(tid, "planning", plan_path=str(plan_path))

        prompt = build_plan_prompt(task, str(plan_path))
        if dry_run:
            print(dim(f"  [DRY RUN] {len(prompt)} char prompt"))
            print(prompt[:800])
            return

        code, _ = run_claude(prompt, max_turns=50, timeout_minutes=plan_timeout, **run_kwargs)
        if not plan_path.exists():
            print(red(f"  Plan file not created at {plan_path}"))
            save_error_output(task_dir, code, _)
            return

    # --- PHASE 2: REVIEW LOOP ---
    if skip_to != "implement":
        _current_phase = "review"
        print(f"\n{bold('--- Phase 2: Review Loop ---')}")
        prev_feedback = ""

        for round_num in range(1, max_review_rounds + 1):
            review_path = task_dir / f"review-{round_num}.md"
            print(f"\n  {bold(f'Review round {round_num}/{max_review_rounds}')}")

            state.set_status(tid, "reviewing", review_round=round_num)
            prompt = build_review_prompt(task, str(plan_path), str(review_path), prev_feedback)
            if dry_run:
                print(dim(f"  [DRY RUN] Review prompt: {len(prompt)} chars"))
                return

            run_claude(prompt, max_turns=30, timeout_minutes=review_timeout, **run_kwargs)

            approved, feedback = check_verdict(review_path)
            if approved:
                print(green(f"  APPROVED at round {round_num}"))
                state.set_status(tid, "approved")
                break
            else:
                print(yellow(f"  NEEDS REVISION"))
                prev_feedback = feedback

                # Revise
                state.set_status(tid, "revising", review_round=round_num)
                revise_prompt = build_revise_prompt(task, str(plan_path), str(review_path))
                run_claude(revise_prompt, max_turns=30, timeout_minutes=review_timeout, **run_kwargs)
        else:
            print(yellow(f"  Max review rounds reached. Proceeding with current plan."))
            state.set_status(tid, "approved")

    # --- PHASE 3: IMPLEMENT ---
    _current_phase = "implement"
    print(f"\n{bold('--- Phase 3: Implementation ---')}")
    state.set_status(tid, "implementing")

    prompt = build_implement_prompt(task, str(plan_path), test_command, commit_msg)
    if dry_run:
        print(dim(f"  [DRY RUN] Implement prompt: {len(prompt)} chars"))
        return

    run_claude(prompt, max_turns=impl_turns, timeout_minutes=impl_timeout, **run_kwargs)
    state.set_status(tid, "done")
    print(green(f"\n  Task {tid} complete."))


def save_error_output(task_dir: Path, code: int, output: str):
    err_path = task_dir / "error-output.txt"
    err_path.write_text(f"Exit code: {code}\n\n{output}")
    print(yellow(f"  Raw output saved to {err_path}"))


def handle_sigint(signum, frame):
    _kill_child()  # Kill any running claude process and its children
    if _current_state:
        _current_state.save()
    print(f"\n{yellow('Interrupted.')} Resume with:")
    print(f"  claude-plan-execute --task {_current_task_id} --skip-to {_current_phase}")
    sys.exit(130)
