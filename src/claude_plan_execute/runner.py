"""Per-task plan / review / implement orchestrator."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from claude_plan_execute import claude
from claude_plan_execute.claude import _kill_child, run_claude
from claude_plan_execute.colors import bold, dim, green, red, yellow
from claude_plan_execute.config import resolve_task_config
from claude_plan_execute.gates import Gate, default_registry, run_gates
from claude_plan_execute.gates.builtin import (
    make_lint_gate,
    make_tests_gate,
)
from claude_plan_execute.loader import (
    load_decisions_index,
    load_persona_fragment,
)
from claude_plan_execute.prompts import (
    build_decisions_section,
    build_failure_context_section,
    build_implement_prompt,
    build_persona_section,
    build_plan_prompt,
    build_review_prompt,
    build_revise_prompt,
)
from claude_plan_execute.state import State, make_failure_record


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
    phase_id = task.get("phase")
    if phases_by_id and phase_id and phase_id in phases_by_id:
        phase_dict = dict(phases_by_id[phase_id].get("defaults") or {})

    task_class_dict: dict = {}
    # Effective class name follows phase-inheritance: task wins over phase.
    # Same chain task 4's runner uses; the audit must use this exact name
    # so provenance labels read `task_class:<name>` correctly. (Plan §D5.)
    task_class_name = task.get("task_class") or phase_dict.get("task_class")
    if task_class_models and task_class_name and task_class_name in task_class_models:
        task_class_dict = dict(task_class_models[task_class_name])

    # Resolve every audit-relevant knob in a single call. This dict is the
    # source of truth for both the on-disk audit artifact and the runtime
    # values used below. (Plan §D4.)
    resolved = resolve_task_config(
        task, file_defaults, cli,
        phase=phase_dict, task_class=task_class_dict,
        phase_id=phase_id, task_class_name=task_class_name,
    )

    tid = task["id"]
    plans_dir = Path(resolved["plans_dir"]["value"])
    task_dir = plans_dir / f"task-{tid}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Audit artifact — written before any phase. (Plan §D6.)
    resolved_config_path = task_dir / "resolved-config.json"
    resolved_config_path.write_text(
        json.dumps(resolved, indent=2, sort_keys=False) + "\n"
    )
    # Track the path on the per-task state entry so the dashboard can find it.
    entry = state.get(tid)
    entry["resolved_config_path"] = str(resolved_config_path)
    state.save()

    # Snapshot prior failures for resume-context (task 10). The same
    # section is reused across the Plan and Implement builders below;
    # any failure recorded mid-run is for FUTURE re-runs, not this one.
    prior_attempts = list(entry.get("attempts", []))
    failure_context_section = build_failure_context_section(prior_attempts)

    plan_path = task_dir / "plan.md"

    model = resolved["model"]["value"]
    effort = resolved["effort"]["value"]
    max_review_rounds = int(resolved["max_review_rounds"]["value"])
    plan_timeout = int(resolved["plan_timeout_minutes"]["value"])
    review_timeout = int(resolved["review_timeout_minutes"]["value"])
    impl_timeout = int(resolved["implement_timeout_minutes"]["value"])
    impl_turns = int(resolved["implement_max_turns"]["value"])
    test_command = resolved["test_command"]["value"]
    lint_command = resolved["lint_command"]["value"]
    on_max_review_rounds = resolved["on_max_review_rounds"]["value"]
    skip_perms = cli.get("skip_permissions", False)
    verbose = cli.get("verbose", False)

    # Persona prompt fragment (task 7).
    # Read once per execute_task; reused across plan/review/revise/implement
    # prompts. (Plan §D3.) The path comes from the resolver chain
    # (HARDCODED_DEFAULTS["persona_file"] = "docs/persona.md" if unset);
    # persona_extra is per-phase, read directly from the phase's loose
    # dict (Plan §D5).
    persona_path = resolved["persona_file"]["value"]
    persona_extra_path = phase_dict.get("persona_extra")
    persona_text = load_persona_fragment(persona_path)
    extra_text = load_persona_fragment(persona_extra_path)
    persona_section = build_persona_section(persona_text, extra_text)

    # Architectural decisions index (task 8).
    # Read once per execute_task; reused only for the Plan prompt
    # (§D1). The path comes from the resolver chain
    # (HARDCODED_DEFAULTS["decisions_dir"] = "docs/decisions" if
    # unset). No per-phase overlay (§"Non-goals").
    decisions_dir = resolved["decisions_dir"]["value"]
    decisions_text = load_decisions_index(decisions_dir)
    decisions_section = build_decisions_section(decisions_text)

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

        prompt = build_plan_prompt(
            task, str(plan_path),
            persona=persona_section,
            decisions=decisions_section,
            failure_context=failure_context_section,
        )
        if dry_run:
            print(dim(f"  [DRY RUN] {len(prompt)} char prompt"))
            print(prompt[:800])
            return

        code, _ = run_claude(prompt, max_turns=100, timeout_minutes=plan_timeout, **run_kwargs)
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
            prompt = build_review_prompt(
                task, str(plan_path), str(review_path),
                prev_feedback, persona=persona_section,
            )
            if dry_run:
                print(dim(f"  [DRY RUN] Review prompt: {len(prompt)} chars"))
                return

            run_claude(prompt, max_turns=60, timeout_minutes=review_timeout, **run_kwargs)

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
                revise_prompt = build_revise_prompt(
                    task, str(plan_path), str(review_path),
                    persona=persona_section,
                )
                run_claude(revise_prompt, max_turns=60, timeout_minutes=review_timeout, **run_kwargs)
        else:
            if on_max_review_rounds == "fail":
                print(red(
                    f"  Max review rounds reached ({max_review_rounds}). "
                    f"on_max_review_rounds: fail — stopping; Implement skipped."
                ))
                # When max_review_rounds == 0 the for-loop never bound
                # round_num; guard the path construction.
                if max_review_rounds > 0:
                    last_review_path = task_dir / f"review-{round_num}.md"
                    try:
                        last_review_text = (
                            last_review_path.read_text(encoding="utf-8")
                            if last_review_path.exists()
                            else ""
                        )
                    except OSError:
                        last_review_text = ""
                    artifacts = {
                        "plan": str(plan_path),
                        "last_review": str(last_review_path),
                    }
                else:
                    last_review_text = ""
                    artifacts = {"plan": str(plan_path)}
                state.set_status(
                    tid, "blocked",
                    block_reason="max_review_rounds_exhausted",
                )
                state.record_failure(tid, make_failure_record(
                    failure_class="review_rejected_after_max_rounds",
                    phase="review",
                    reason=(
                        f"Reviewer rejected the plan {max_review_rounds} "
                        f"round{'s' if max_review_rounds != 1 else ''} in a row."
                    ),
                    verbatim=last_review_text,
                    artifacts=artifacts,
                ))
                return
            # auto_approve (legacy v1 default)
            print(yellow(
                f"  Max review rounds reached. Proceeding with current "
                f"plan (on_max_review_rounds: auto_approve)."
            ))
            state.set_status(tid, "approved")

    # --- PHASE 3: IMPLEMENT ---
    _current_phase = "implement"
    print(f"\n{bold('--- Phase 3: Implementation ---')}")
    state.set_status(tid, "implementing")

    prompt = build_implement_prompt(
        task, str(plan_path), commit_msg,
        persona=persona_section,
        failure_context=failure_context_section,
    )
    if dry_run:
        print(dim(f"  [DRY RUN] Implement prompt: {len(prompt)} chars"))
        return

    run_claude(prompt, max_turns=impl_turns, timeout_minutes=impl_timeout, **run_kwargs)

    if claude._last_run_timed_out:
        state.set_status(
            tid, "blocked",
            block_reason="implementer_timeout",
        )
        state.record_failure(tid, make_failure_record(
            failure_class="implementer_timeout",
            phase="implement",
            reason=(
                f"Implementer exceeded the {impl_timeout}-minute "
                f"wall-clock timeout."
            ),
            artifacts={"plan": str(plan_path)},
        ))
        print(red(f"\n  Task {tid} blocked: implementer timeout."))
        return

    if claude._last_run_subtype == "error_max_turns":
        state.set_status(
            tid, "blocked",
            block_reason="implementer_max_turns",
        )
        state.record_failure(tid, make_failure_record(
            failure_class="implementer_max_turns",
            phase="implement",
            reason=(
                f"Implementer hit the {impl_turns}-turn cap "
                f"(CLI subtype 'error_max_turns')."
            ),
            artifacts={"plan": str(plan_path)},
        ))
        print(red(f"\n  Task {tid} blocked: implementer max-turns."))
        return

    # --- PHASE 4: GATES (task 11) ---
    # `defaults.gates` (project-level list of names) + `phases[i].
    # defaults.gates_extra` (phase overlay). Names are resolved
    # against the default registry; unknown names produce a single
    # warning and are skipped (forward-compat with task 12+
    # registering built-ins).
    gate_names_project = list(file_defaults.get("gates") or [])
    gate_names_phase = list(phase_dict.get("gates_extra") or [])
    gate_names = gate_names_project + gate_names_phase
    if gate_names:
        from claude_plan_execute.gates.invariants import loaded_invariant_names
        invariant_names = loaded_invariant_names()
        registry = default_registry()
        resolved_gates: list[tuple[str, Gate]] = []
        unknown_gates: list[str] = []
        unconfigured_lint = False
        for name in gate_names:
            if name == "tests":
                resolved_gates.append((name, make_tests_gate(test_command)))
            elif name == "lint":
                if lint_command is None:
                    unconfigured_lint = True
                else:
                    resolved_gates.append((name, make_lint_gate(lint_command)))
            elif name == "invariant-grep":
                for inv_name in invariant_names:
                    g = registry.get(inv_name)
                    if g is not None:
                        resolved_gates.append((inv_name, g))
            else:
                g = registry.get(name)
                if g is None:
                    unknown_gates.append(name)
                else:
                    resolved_gates.append((name, g))
        if unconfigured_lint:
            print(yellow(
                "  Skipping gate 'lint': no defaults.lint_command configured."
            ))
        if unknown_gates:
            print(yellow(
                f"  Unknown gate(s) skipped: {', '.join(unknown_gates)} "
                f"(not registered; task 17 adds built-ins)"
            ))
        if resolved_gates:
            print(f"\n{bold('--- Phase 4: Gates ---')}")
            gate_results = run_gates(
                resolved_gates,
                task=task,
                task_dir=task_dir,
                cli=cli,
            )
            blocking = [
                r for r in gate_results
                if not r.passed and r.on_failure == "block"
            ]
            if blocking:
                verbatim = "\n\n".join(
                    f"--- {r.name} ---\n{r.output}"
                    for r in blocking
                )
                artifacts: dict[str, str] = {
                    "plan": str(plan_path),
                    "gates_dir": str(task_dir / "gates"),
                }
                for r in blocking:
                    if r.output_path:
                        artifacts[f"gate:{r.name}"] = r.output_path
                state.set_status(
                    tid, "blocked",
                    block_reason="gate_failed",
                )
                state.record_failure(tid, make_failure_record(
                    failure_class="gate_failed",
                    phase="gate",
                    reason=(
                        f"{len(blocking)} gate(s) failed: "
                        f"{', '.join(r.name for r in blocking)}"
                    ),
                    verbatim=verbatim,
                    artifacts=artifacts,
                ))
                print(red(
                    f"\n  Task {tid} blocked: gate(s) failed "
                    f"({', '.join(r.name for r in blocking)})."
                ))
                return

    state.set_status(tid, "done")
    print(green(f"\n  Task {tid} complete."))


def save_error_output(task_dir: Path, code: int, output: str):
    err_path = task_dir / "error-output.txt"
    err_path.write_text(f"Exit code: {code}\n\n{output}")
    print(yellow(f"  Raw output saved to {err_path}"))


def handle_sigint(signum, frame):
    _kill_child()  # Kill any running claude process and its children

    # Best-effort: record a manual_stop FailureRecord when a task is
    # genuinely in flight. Status check filters out completed/idle
    # tasks (the global _current_task_id is not reset on execute_task
    # exit). All I/O wrapped — signal handlers must never raise.
    if _current_state is not None and _current_task_id:
        try:
            entry = _current_state.get(_current_task_id)
            if entry["status"] in (
                "planning", "reviewing", "revising",
                "approved", "implementing",
            ):
                _current_state.record_failure(
                    _current_task_id,
                    make_failure_record(
                        failure_class="manual_stop",
                        phase=_current_phase,
                        reason="User-initiated SIGINT during execution.",
                    ),
                )
        except Exception:
            pass

    # Belt-and-suspenders save: covers the path where `record_failure`
    # was skipped (status mismatch) but earlier state mutations need
    # to flush.
    if _current_state:
        _current_state.save()
    print(f"\n{yellow('Interrupted.')} Resume with:")
    print(f"  claude-plan-execute --task {_current_task_id} --skip-to {_current_phase}")
    sys.exit(130)
