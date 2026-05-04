"""Per-task plan / review / implement orchestrator."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from claude_plan_execute import claude
from claude_plan_execute.claude import _kill_child, run_claude
from claude_plan_execute.colors import bold, dim, green, red, yellow
from claude_plan_execute.config import resolve_task_config
from claude_plan_execute.gates import Gate, default_registry, run_gates
from claude_plan_execute.gates import _when_matches as _gate_when_matches
from claude_plan_execute.gates.shell import _tail_cap
from claude_plan_execute.gates.builtin import (
    make_design_fidelity_gate,
    make_lint_gate,
    make_tests_gate,
    make_untranslated_strings_gate,
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
    build_resume_prompt,
    build_review_prompt,
    build_revise_prompt,
    parse_progress_status,
)
from claude_plan_execute.services import (
    ServiceManager,
    ServiceUnavailableError,
    ServiceUnstableError,
)
from claude_plan_execute.state import (
    State, make_failure_record, summarize_usage,
)


# T7 §D5: 256 KB cap for the precomputed implement-diff artifact.
# Larger than gates/shell.py:_OUTPUT_CAP_BYTES (100 KB) because
# diffs are denser context than test logs (operator-debug value
# justifies the larger budget). The same `_tail_cap` algorithm
# services both — see gates/shell.py:_tail_cap docstring.
_DIFF_CAP_BYTES = 256 * 1024


# T5 (v0.3 §2.20) — definition-origin lookup for explain_task.
# Keep these sets in sync with gates/builtin.py:
#   _BUILTIN_FACTORY_NAMES — names dispatched to make_*_gate factories
#     in execute_task's Phase-4 loop (no pre-registration on the
#     default registry).
#   _BUILTIN_REGISTRY_NAMES — names registered on default_registry()
#     at gates/builtin.py module-import time.
# Future tasks that add new built-ins MUST patch the corresponding
# set here as part of the same diff.
_BUILTIN_FACTORY_NAMES: frozenset[str] = frozenset({
    "tests", "lint", "untranslated-strings", "design-fidelity",
})
_BUILTIN_REGISTRY_NAMES: frozenset[str] = frozenset({
    "test-critique", "security-auditor",
})


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


def _resolve_gate_specs(
    gate_names: list[str],
    *,
    test_command: str | None,
    lint_command: str | None,
    untranslated_strings_command: str | None,
    dev_server_url: str,
    persona_section: str,
    invariant_names: list[str],
    registry: "GateRegistry",
) -> tuple[
    list[tuple[str, "Gate"]],
    list[str],
    bool,
    bool,
]:
    """Walk a gate-name list to (name, Gate) pairs.

    Mirrors the inline dispatch in execute_task's Phase 4. The
    explainer (`explain_task`) and `execute_task` BOTH call this
    helper — the "same resolver + registry path" guarantee is
    structural, not test-enforced. (Plan §D1.)

    Pure: no I/O, no side effects, no logging.
    """
    resolved_gates: list[tuple[str, "Gate"]] = []
    unknown_gates: list[str] = []
    unconfigured_lint = False
    unconfigured_untranslated_strings = False
    for name in gate_names:
        if name == "tests":
            resolved_gates.append((name, make_tests_gate(test_command)))
        elif name == "lint":
            if lint_command is None:
                unconfigured_lint = True
            else:
                resolved_gates.append((name, make_lint_gate(lint_command)))
        elif name == "untranslated-strings":
            if untranslated_strings_command is None:
                unconfigured_untranslated_strings = True
            else:
                resolved_gates.append((
                    name,
                    make_untranslated_strings_gate(untranslated_strings_command),
                ))
        elif name == "design-fidelity":
            resolved_gates.append((
                name,
                make_design_fidelity_gate(
                    dev_server_url=dev_server_url,
                    persona=persona_section,
                ),
            ))
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
    return (
        resolved_gates,
        unknown_gates,
        unconfigured_lint,
        unconfigured_untranslated_strings,
    )


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
    service_manager: "ServiceManager | None" = None,
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
    usage_path = task_dir / "usage.jsonl"

    # v0.3 task 8: compute services_required for this task. Always
    # computed (cheap dict ops) so the audit artifact has a stable
    # field shape even when no manager is wired.
    required_services = _resolve_services_required(
        task=task, file_defaults=file_defaults, phase_dict=phase_dict,
    )

    # Audit artifact — written before any phase. (Plan §D6.) The
    # services_required sibling field is added by T8 (see Plan §D9);
    # KEYS_TO_AUDIT is unchanged because the value is a per-layer
    # union, not a winner-takes-all resolver key.
    resolved_config_path = task_dir / "resolved-config.json"
    audit_payload: dict = dict(resolved)
    audit_payload["services_required"] = {
        "value": list(required_services),
        "from_layers": {
            "defaults": list(file_defaults.get("services_required") or []),
            "phase":    list(phase_dict.get("services_required_extra") or []),
            "task":     list(task.get("services_required_extra") or []),
        },
    }
    resolved_config_path.write_text(
        json.dumps(audit_payload, indent=2, sort_keys=False) + "\n"
    )
    # Track the path on the per-task state entry so the dashboard can find it.
    entry = state.get(tid)
    entry["resolved_config_path"] = str(resolved_config_path)
    state.save()

    # T9 (v0.3 §2.12) §D6: pre-Phase-1 branch hook. Lives BEFORE
    # the try/finally so a branch failure surfaces as a normal
    # early-return — services have not started yet, so there is no
    # teardown to coordinate. The branch decision MUST land on the
    # state entry before Phase 1 so an interrupted-mid-Phase-1
    # retry knows which branch to re-checkout (the next pass re-
    # executes this hook idempotently).
    if bool(resolved["branch_per_task"]["value"]):
        _branch_template = resolved["task_branch_template"]["value"]
        _base_branch = resolved["base_branch"]["value"]
        try:
            _branch = _render_branch_name(
                _branch_template, tid, _slug(task["title"]),
            )
            _parent_branch = _resolve_parent_branch(
                task, state, _base_branch,
            )
            _create_or_checkout_branch(
                _branch, _parent_branch, _base_branch,
            )
        except _GitOperationError as exc:
            state.set_status(
                tid, "blocked",
                block_reason="branch_checkout_failed",
            )
            state.record_failure(tid, make_failure_record(
                failure_class="branch_checkout_failed",
                phase="branch",
                reason=str(exc),
                artifacts={"task_dir": str(task_dir)},
            ))
            print(red(
                f"\n  Task {tid} blocked: branch checkout "
                f"failed: {exc}"
            ))
            return
        state.set_status(
            tid, state.get(tid)["status"], branch=_branch,
        )

    # v0.3 task 8 (Plan §D4.6): wrap the body in try/finally so
    # gate- and task-scope services are torn down on every exit
    # path — success, blocked, dry-run early-return, uncaught
    # exception. The audit write above stays OUTSIDE the try so
    # /api/resolved-config/{task_id} works even if the service
    # block raises.
    try:
        # v0.3 task 8 (Plan §D4.4): task-entry service block.
        # Skipped entirely when no manager is supplied (legacy
        # path) OR in dry-run mode (no real work happens, no need
        # to spin services up). The §D4.6 finally still runs but
        # is a no-op for services that never started.
        if service_manager is not None and not dry_run:
            try:
                events = service_manager.poll_health()
            except ServiceUnstableError as exc:
                state.set_status(
                    tid, "blocked", block_reason="service_unstable",
                )
                state.record_failure(tid, make_failure_record(
                    failure_class="service_unstable",
                    phase="services",
                    reason=str(exc),
                    artifacts={
                        "services_status": str(plans_dir / "services.json"),
                    },
                ))
                print(red(f"\n  Task {tid} blocked: {exc}"))
                return
            for ev in events:
                if ev["event"] == "crashed_restarted":
                    state.record_failure(tid, make_failure_record(
                        failure_class="service_crashed",
                        phase="services",
                        reason=(
                            f"Service '{ev['name']}' crashed (exit "
                            f"{ev['exit_code']}); auto-restarted "
                            f"(crash {ev['crash_count']}/3)."
                        ),
                    ))
            if required_services:
                try:
                    service_manager.ensure(required_services)
                except ServiceUnavailableError as exc:
                    state.set_status(
                        tid, "blocked",
                        block_reason="service_unavailable",
                    )
                    state.record_failure(tid, make_failure_record(
                        failure_class="service_unavailable",
                        phase="services",
                        reason=str(exc),
                        artifacts={
                            "services_status": str(plans_dir / "services.json"),
                        },
                    ))
                    print(red(f"\n  Task {tid} blocked: {exc}"))
                    return

        # Snapshot prior failures for resume-context (task 10). The same
        # section is reused across the Plan and Implement builders below;
        # any failure recorded mid-run is for FUTURE re-runs, not this one.
        prior_attempts = list(entry.get("attempts", []))
        failure_context_section = build_failure_context_section(prior_attempts)
    
        # Resume routing (task 6 v0.3 §D4). progress.md is the sidecar
        # checklist the implementer maintains across chunks. Existence +
        # at-least-one-active-item ([~] or [ ]) → "resume in progress;
        # skip Plan + Review and use the resume builder for Implement."
        # Existence + zero-active-items + at-least-one-done → "all work
        # done pending gates" — skip Plan + Review + Implement and fall to
        # Phase 4. No file (or empty checklist) → fresh run.
        progress_path = task_dir / "progress.md"
        progress_text = _read_progress_text(progress_path)
        n_done, n_in_progress, n_pending = parse_progress_status(progress_text)
        has_active = n_in_progress > 0 or n_pending > 0
        if not progress_text or (n_done == 0 and not has_active):
            resume_mode = "fresh"
        elif has_active:
            resume_mode = "mid_list"
        else:
            resume_mode = "all_done"
        state.set_status(
            tid, state.get(tid)["status"],
            progress_path=str(progress_path),
        )
    
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
        untranslated_strings_command = resolved["untranslated_strings_command"]["value"]
        dev_server_url = resolved["dev_server_url"]["value"]
        on_max_review_rounds = resolved["on_max_review_rounds"]["value"]
        max_rate_limit_retries = int(resolved["max_rate_limit_retries"]["value"])
        rate_limit_base_backoff_seconds = int(
            resolved["rate_limit_base_backoff_seconds"]["value"]
        )
        # T9 (v0.3 §2.12).
        branch_per_task = bool(resolved["branch_per_task"]["value"])
        base_branch = resolved["base_branch"]["value"]
        auto_pr = bool(resolved["auto_pr"]["value"])
        auto_merge = bool(resolved["auto_merge"]["value"])
        pr_body_template = resolved["pr_body_template"]["value"]
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
    
        run_kwargs = dict(
            model=model, effort=effort,
            skip_permissions=skip_perms, verbose=verbose,
            max_rate_limit_retries=max_rate_limit_retries,
            rate_limit_base_backoff_seconds=rate_limit_base_backoff_seconds,
        )
    
        # --- PHASE 1: PLAN ---
        if skip_to not in ("review", "implement") and resume_mode == "fresh":
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
    
            code, _ = run_claude(
                prompt, max_turns=100, timeout_minutes=plan_timeout,
                phase="plan", usage_jsonl_path=usage_path,
                **run_kwargs,
            )
            if claude._last_run_rate_limited:
                _record_rate_limited_and_block(
                    state, tid, phase="plan", plan_path=None,
                    max_rate_limit_retries=max_rate_limit_retries,
                )
                return
            if not plan_path.exists():
                print(red(f"  Plan file not created at {plan_path}"))
                save_error_output(task_dir, code, _)
                return
    
        # --- PHASE 2: REVIEW LOOP ---
        if skip_to != "implement" and resume_mode == "fresh":
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
    
                run_claude(
                    prompt, max_turns=60, timeout_minutes=review_timeout,
                    phase="review", usage_jsonl_path=usage_path,
                    **run_kwargs,
                )
                if claude._last_run_rate_limited:
                    _record_rate_limited_and_block(
                        state, tid, phase="review", plan_path=plan_path,
                        max_rate_limit_retries=max_rate_limit_retries,
                    )
                    return
    
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
                    run_claude(
                        revise_prompt, max_turns=60,
                        timeout_minutes=review_timeout,
                        phase="revise", usage_jsonl_path=usage_path,
                        **run_kwargs,
                    )
                    if claude._last_run_rate_limited:
                        _record_rate_limited_and_block(
                            state, tid, phase="review", plan_path=plan_path,
                            max_rate_limit_retries=max_rate_limit_retries,
                        )
                        return
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
        if resume_mode == "all_done":
            print(f"\n{bold('--- Phase 3: Implementation (skipped) ---')}")
            print(green(
                f"  All progress.md items marked [x]; gates will validate."
            ))
        else:
            print(f"\n{bold('--- Phase 3: Implementation ---')}")
            state.set_status(tid, "implementing")
            if resume_mode == "mid_list":
                porcelain = _capture_git_porcelain()
                prompt = build_resume_prompt(
                    task, str(plan_path), str(progress_path),
                    progress_text, porcelain, commit_msg,
                    persona=persona_section,
                    failure_context=failure_context_section,
                )
            else:  # "fresh"
                prompt = build_implement_prompt(
                    task, str(plan_path), commit_msg,
                    persona=persona_section,
                    failure_context=failure_context_section,
                )
            if dry_run:
                print(dim(f"  [DRY RUN] Implement prompt: {len(prompt)} chars"))
                return
    
            run_claude(
                prompt, max_turns=impl_turns, timeout_minutes=impl_timeout,
                phase="implement", usage_jsonl_path=usage_path,
                **run_kwargs,
            )
    
            if claude._last_run_rate_limited:
                _record_rate_limited_and_block(
                    state, tid, phase="implement", plan_path=plan_path,
                    max_rate_limit_retries=max_rate_limit_retries,
                )
                return
    
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
    
        # --- POST-IMPLEMENT: precompute diff for agent gates (task 7 v0.3) ---
        # Runs unconditionally before Phase 4 (even when no gates are
        # configured) so the artifact has standalone operator-debug
        # value. Empty diff → empty file; templates handle that branch.
        # T9 §D5 patches the call site to pass branch_per_task /
        # base_branch from the resolver chain so the diff form
        # switches to `origin/{base}...HEAD` when the per-task
        # branch is in scope.
        diff_text = _capture_implement_diff(
            branch_per_task=branch_per_task,
            base_branch=base_branch,
        )
        (task_dir / "diff.patch").write_text(diff_text, encoding="utf-8")

        # v0.3 task 8 (Plan §D4.5): ensure gate-scoped services are
        # up before Phase 4. poll_health may have detected a crash
        # that happened during the implement phase.
        if service_manager is not None:
            try:
                events = service_manager.poll_health()
            except ServiceUnstableError as exc:
                state.set_status(
                    tid, "blocked", block_reason="service_unstable",
                )
                state.record_failure(tid, make_failure_record(
                    failure_class="service_unstable",
                    phase="gate",
                    reason=str(exc),
                    artifacts={
                        "services_status": str(plans_dir / "services.json"),
                    },
                ))
                print(red(f"\n  Task {tid} blocked: {exc}"))
                return
            for ev in events:
                if ev["event"] == "crashed_restarted":
                    state.record_failure(tid, make_failure_record(
                        failure_class="service_crashed",
                        phase="gate",
                        reason=(
                            f"Service '{ev['name']}' crashed (exit "
                            f"{ev['exit_code']}); auto-restarted "
                            f"(crash {ev['crash_count']}/3)."
                        ),
                    ))
            if required_services:
                try:
                    service_manager.ensure(required_services)
                except ServiceUnavailableError as exc:
                    state.set_status(
                        tid, "blocked",
                        block_reason="service_unavailable",
                    )
                    state.record_failure(tid, make_failure_record(
                        failure_class="service_unavailable",
                        phase="gate",
                        reason=str(exc),
                        artifacts={
                            "services_status": str(plans_dir / "services.json"),
                        },
                    ))
                    print(red(f"\n  Task {tid} blocked: {exc}"))
                    return

        # --- PHASE 4: GATES (task 11) ---
        # `defaults.gates` (project-level list of names) + `phases[i].
        # defaults.gates_extra` (phase overlay) + `task.gates_extra`
        # (per-task overlay, v0.3 task 4). Layers concatenate in that
        # order; no subtraction. Names are resolved against the default
        # registry; unknown names produce a single warning and are
        # skipped (forward-compat with task 12+ registering built-ins).
        gate_names_project = list(file_defaults.get("gates") or [])
        gate_names_phase = list(phase_dict.get("gates_extra") or [])
        gate_names_task = list(task.get("gates_extra") or [])
        gate_names = gate_names_project + gate_names_phase + gate_names_task
        if gate_names:
            from claude_plan_execute.gates.invariants import loaded_invariant_names
            invariant_names = loaded_invariant_names()
            registry = default_registry()
            (
                resolved_gates,
                unknown_gates,
                unconfigured_lint,
                unconfigured_untranslated_strings,
            ) = _resolve_gate_specs(
                gate_names,
                test_command=test_command,
                lint_command=lint_command,
                untranslated_strings_command=untranslated_strings_command,
                dev_server_url=dev_server_url,
                persona_section=persona_section,
                invariant_names=invariant_names,
                registry=registry,
            )
            if unconfigured_lint:
                print(yellow(
                    "  Skipping gate 'lint': no defaults.lint_command configured."
                ))
            if unconfigured_untranslated_strings:
                print(yellow(
                    "  Skipping gate 'untranslated-strings': no "
                    "defaults.untranslated_strings_command configured."
                ))
            if unknown_gates:
                print(yellow(
                    f"  Unknown gate(s) skipped: {', '.join(unknown_gates)} "
                    f"(not registered; check spelling)"
                ))
            if resolved_gates:
                print(f"\n{bold('--- Phase 4: Gates ---')}")
                # Augment the task dict so phase-inherited task_class
                # is visible to gate `when` filters (task 20 §D9).
                task_for_gates = {**task, "task_class": task_class_name}
                gate_results = run_gates(
                    resolved_gates,
                    task=task_for_gates,
                    task_dir=task_dir,
                    cli=cli,
                    state=state,
                )
                blocking = [
                    r for r in gate_results
                    if not r.passed and r.on_failure == "block"
                ]
                if blocking:
                    # T10 §D19: differentiate human_gate_timeout from
                    # generic gate_failed. BOTH `block_reason` (the
                    # dashboard-visible chip) AND `failure_class`
                    # (telemetry/audits) flip together via a single
                    # `reason_class` local so they cannot drift.
                    has_human_timeout = any(
                        r.error == "human_gate_timeout" for r in blocking
                    )
                    reason_class = (
                        "human_gate_timeout"
                        if has_human_timeout
                        else "gate_failed"
                    )
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
                        block_reason=reason_class,
                    )
                    state.record_failure(tid, make_failure_record(
                        failure_class=reason_class,
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

        # T9 (v0.3 §2.12) §D9: post-Phase-4 PR / merge hook. Skipped
        # entirely when `branch_per_task` or `auto_pr` is False (the
        # backward-compat default — zero new shell-outs). PR side-
        # effects can fail without losing the task's work; failures
        # transition to blocked + a specific failure_class so
        # operators can re-run after fixing (e.g., gh auth).
        pr_url: str | None = None
        if branch_per_task and auto_pr:
            _branch_for_pr = state.get(tid).get("branch")
            push_proc = subprocess.run(
                ["git", "push", "-u", "origin", _branch_for_pr],
                capture_output=True, text=True, timeout=120,
            )
            if push_proc.returncode != 0:
                state.set_status(
                    tid, "blocked", block_reason="push_failed",
                )
                state.record_failure(tid, make_failure_record(
                    failure_class="push_failed",
                    phase="post_implement",
                    reason=(
                        f"git push -u origin {_branch_for_pr} failed "
                        f"(exit {push_proc.returncode}): "
                        f"{(push_proc.stderr or push_proc.stdout).strip()[-500:]}"
                    ),
                    artifacts={"plan": str(plan_path)},
                ))
                print(red(
                    f"\n  Task {tid} blocked: git push failed."
                ))
                return
            try:
                pr_url = _open_or_get_pr_url(
                    branch=_branch_for_pr,
                    base_branch=base_branch,
                    title=task["title"],
                    body=_render_pr_body(
                        task, plan_path, pr_body_template,
                    ),
                )
            except _GitOperationError as exc:
                state.set_status(
                    tid, "blocked", block_reason="pr_create_failed",
                )
                state.record_failure(tid, make_failure_record(
                    failure_class="pr_create_failed",
                    phase="post_implement",
                    reason=str(exc),
                    artifacts={"plan": str(plan_path)},
                ))
                print(red(
                    f"\n  Task {tid} blocked: gh pr create failed."
                ))
                return
            if auto_merge:
                merge_proc = subprocess.run(
                    ["gh", "pr", "merge", "--auto", "--squash",
                     _branch_for_pr],
                    capture_output=True, text=True, timeout=120,
                )
                if merge_proc.returncode != 0:
                    state.set_status(
                        tid, "blocked", block_reason="merge_failed",
                    )
                    state.record_failure(tid, make_failure_record(
                        failure_class="merge_failed",
                        phase="post_implement",
                        reason=(
                            f"gh pr merge --auto --squash failed "
                            f"(exit {merge_proc.returncode}): "
                            f"{(merge_proc.stderr or merge_proc.stdout).strip()[-500:]}"
                        ),
                        artifacts={
                            "plan": str(plan_path),
                            "pr_url": pr_url,
                        },
                    ))
                    print(red(
                        f"\n  Task {tid} blocked: gh pr merge failed."
                    ))
                    return

        # T2 §D8: roll up per-run usage rows into the per-task summary.
        # Failed / blocked tasks short-circuit before this point; they
        # do NOT get a usage_summary. Re-running a done task overwrites
        # the summary (cumulative over the append-only JSONL).
        summary = summarize_usage(usage_path)
        done_extras: dict = {}
        if summary["total_runs"] > 0:
            done_extras["usage_summary"] = summary
        # T9 §D9: pr_url lands on the state entry alongside
        # usage_summary when auto_pr fired and produced a URL.
        if pr_url is not None:
            done_extras["pr_url"] = pr_url

        # v0.3 task 8 (Plan §D4.6): success-path explicit teardown.
        # `reset_crash_count` runs for services that held running
        # through the entire task; the explicit
        # `_service_teardown_after_task` call short-circuits the
        # `finally` (idempotent on already-stopped services).
        if service_manager is not None:
            for name in required_services:
                if service_manager.is_running(name):
                    service_manager.reset_crash_count(name)
        _service_teardown_after_task(service_manager)

        state.set_status(tid, "done", **done_extras)
        print(green(f"\n  Task {tid} complete."))
        return
    finally:
        # v0.3 task 8 (Plan §D4.6): defensive teardown for every
        # non-success exit path (rate-limit returns, implementer
        # timeouts, gate failures, dry-run early-returns,
        # plan-missing return, uncaught exceptions). Idempotent on
        # stopped services, so a re-call after the success-path
        # explicit teardown is a no-op.
        _service_teardown_after_task(service_manager)


def save_error_output(task_dir: Path, code: int, output: str):
    err_path = task_dir / "error-output.txt"
    err_path.write_text(f"Exit code: {code}\n\n{output}")
    print(yellow(f"  Raw output saved to {err_path}"))


def _record_rate_limited_and_block(
    state: State, tid: int | str, phase: str, plan_path: Path | None,
    *, max_rate_limit_retries: int,
) -> None:
    """Record a `rate_limited` FailureRecord and set status to blocked.

    Called from each `run_claude` call site when
    `claude._last_run_rate_limited` is True after the call. `plan_path`
    may be `None` (plan-phase failure where the plan never materialized).
    """
    artifacts: dict[str, str] = {}
    if plan_path is not None:
        artifacts["plan"] = str(plan_path)
    state.set_status(tid, "blocked", block_reason="rate_limited")
    state.record_failure(tid, make_failure_record(
        failure_class="rate_limited",
        phase=phase,
        reason=(
            f"Claude API rate-limit retries exhausted after "
            f"{max_rate_limit_retries + 1} attempts."
        ),
        artifacts=artifacts,
    ))
    print(red(f"\n  Task {tid} blocked: rate-limit retries exhausted."))


def _read_progress_text(progress_path: Path) -> str:
    """Read progress.md if present, else return ''.

    Defensive against transient I/O errors so a stat-race doesn't crash
    routing.
    """
    if not progress_path.is_file():
        return ""
    try:
        return progress_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _capture_git_porcelain() -> str:
    """Run `git status --porcelain` and return stdout (or '' on any failure).

    Only called from the resume route. Failures are silent — a non-git
    working tree, a missing git binary, or a permissions error all yield
    '' and the resume prompt's clean-tree branch fires. Bounded timeout
    because the runner must not hang here.
    """
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout


# --- T9 (v0.3 §2.12) — branch + PR helpers --------------------------------


class _GitOperationError(RuntimeError):
    """Raised by branch / PR helpers on a non-recoverable git/gh failure.

    The runner catches this and writes a FailureRecord. Exists so the
    runner doesn't have to import subprocess.CalledProcessError into
    high-level state-setting code.
    """


_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SLUG_BODY_CAP = 30  # T9 v0.3 §2.12: keep branch names readable

_DEFAULT_PR_BODY_TEMPLATE = (
    "Auto-opened by `claude-plan-execute` for task {id}.\n\n"
    "**Title.** {title}\n\n"
    "**Description.**\n\n{description}\n\n"
    "**Plan.** [`{plan_path}`]({plan_path})\n\n"
    "**Diff.** See `diff.patch` in the task's plans dir.\n"
)


def _slug(title: str) -> str:
    """Lowercase, collapse non-alphanumeric runs to '-', trim, cap at 30 chars.

    Edge cases:
      - Empty / whitespace-only title → "task" sentinel.
      - All-non-alphanumeric (e.g., "!!!") → "task" sentinel.
      - Mid-token cap at 30 chars → trim trailing '-'.
      - Non-ASCII letters dropped (branch names with non-ASCII trip
        several git server implementations).
    """
    lower = title.lower()
    collapsed = _SLUG_NON_ALNUM.sub("-", lower).strip("-")
    if not collapsed:
        return "task"
    body = collapsed[:_SLUG_BODY_CAP].rstrip("-")
    return body or "task"


def _render_branch_name(template: str, tid: int | str, slug: str) -> str:
    """Render a branch name from the template and slug.

    Raises `_GitOperationError` when the template references an unknown
    placeholder (only `{id}` and `{slug}` are supported in v0.3).
    """
    try:
        return template.format(id=tid, slug=slug)
    except KeyError as exc:
        raise _GitOperationError(
            f"task_branch_template {template!r} references unknown "
            f"placeholder {exc.args[0]!r} (only {{id}} and {{slug}} "
            f"are supported)"
        )


def _branch_exists_locally(branch: str) -> bool:
    """Return True iff `git rev-parse --verify <branch>` exits 0.

    Defensive: any subprocess failure (no git binary, timeout, etc.)
    yields False; the caller will see the create-branch step fail with
    a clearer message.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", branch],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def _resolve_parent_branch(
    task: dict, state: "State", base_branch: str,
) -> str:
    """Pick the parent ref to base the new branch from.

    Walks `task["depends_on"]` in declaration order; the LAST done +
    branched dependency wins (operationalizes the v0.3 §2.12 "latest
    done" tie-break as "last in depends_on order"). Falls back to
    `origin/{base_branch}` when no qualifying dependency exists.
    """
    deps = task.get("depends_on") or []
    parent = None
    for dep_id in deps:
        entry = state.get(dep_id)
        if entry.get("status") == "done" and entry.get("branch"):
            parent = entry["branch"]
    if parent is not None:
        return parent
    return f"origin/{base_branch}"


def _create_or_checkout_branch(
    branch: str, parent_branch: str, base_branch: str,
) -> None:
    """Idempotent: re-checkout an existing local branch, else fetch (when
    parent is `origin/<base>`) and `git checkout -b <branch> <parent>`.

    Raises `_GitOperationError` on any non-zero exit.
    """
    if _branch_exists_locally(branch):
        proc = subprocess.run(
            ["git", "checkout", branch],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            raise _GitOperationError(
                f"git checkout {branch} failed (exit "
                f"{proc.returncode}): "
                f"{(proc.stderr or proc.stdout).strip()}"
            )
        return
    if parent_branch.startswith("origin/"):
        fetch_proc = subprocess.run(
            ["git", "fetch", "origin", base_branch],
            capture_output=True, text=True, timeout=60,
        )
        if fetch_proc.returncode != 0:
            raise _GitOperationError(
                f"git fetch origin {base_branch} failed (exit "
                f"{fetch_proc.returncode}): "
                f"{(fetch_proc.stderr or fetch_proc.stdout).strip()}"
            )
    create_proc = subprocess.run(
        ["git", "checkout", "-b", branch, parent_branch],
        capture_output=True, text=True, timeout=10,
    )
    if create_proc.returncode != 0:
        raise _GitOperationError(
            f"git checkout -b {branch} {parent_branch} failed "
            f"(exit {create_proc.returncode}): "
            f"{(create_proc.stderr or create_proc.stdout).strip()}"
        )


def _render_pr_body(
    task: dict, plan_path: Path, template_path: str | None,
) -> str:
    """Render the PR body either from a configured template file or the
    built-in default. A missing template file falls back silently.

    Supported placeholders: `{id}`, `{title}`, `{description}`,
    `{plan_path}`. Raises `_GitOperationError` on KeyError.
    """
    body_template = _DEFAULT_PR_BODY_TEMPLATE
    if template_path:
        p = Path(template_path)
        if p.is_file():
            body_template = p.read_text(encoding="utf-8")
    try:
        return body_template.format(
            id=task["id"],
            title=task["title"],
            description=task["description"],
            plan_path=str(plan_path),
        )
    except KeyError as exc:
        raise _GitOperationError(
            f"pr_body_template references unknown placeholder "
            f"{exc.args[0]!r}"
        )


def _open_or_get_pr_url(
    *, branch: str, base_branch: str, title: str, body: str,
) -> str:
    """Return the URL of the PR for `branch`, opening one if needed.

    Two-step sequence (plan §3.A): `gh pr list --head <branch> --json
    url -q '.[0].url // ""'` first; on empty stdout, `gh pr create
    --base <base> --title <title> --body <body>`. The list step is
    defensive (non-zero exit yields empty URL); the create step is
    mandatory and raises `_GitOperationError` on failure.
    """
    list_proc = subprocess.run(
        ["gh", "pr", "list", "--head", branch,
         "--json", "url", "-q", '.[0].url // ""'],
        capture_output=True, text=True, timeout=30,
    )
    existing = list_proc.stdout.strip() if list_proc.returncode == 0 else ""
    if existing:
        return existing
    create_proc = subprocess.run(
        ["gh", "pr", "create", "--base", base_branch,
         "--title", title, "--body", body],
        capture_output=True, text=True, timeout=120,
    )
    if create_proc.returncode != 0:
        raise _GitOperationError(
            f"gh pr create --base {base_branch} failed (exit "
            f"{create_proc.returncode}): "
            f"{(create_proc.stderr or create_proc.stdout).strip()[-500:]}"
        )
    return create_proc.stdout.strip()


def _capture_implement_diff(
    *, branch_per_task: bool = False, base_branch: str = "main",
) -> str:
    """Run `git diff HEAD~1..HEAD` (or `origin/{base_branch}...HEAD`
    when `branch_per_task=True`) and return stdout tail-capped at
    `_DIFF_CAP_BYTES` (256 KB).

    On any failure (no git binary, no HEAD~1, non-zero exit,
    timeout) returns ''. The caller writes the result to
    `{task_dir}/diff.patch` unconditionally so agent-gate
    templates can substitute `{diff_path}` and read the file —
    even an empty file has a meaningful "diff is empty, read
    source" branch in those templates.

    T7 ships with the HEAD~1..HEAD form (`branch_per_task=False`
    default). T9 will set the kwargs from the resolver chain
    when its `branch_per_task` switch is on; the call site at
    runner.py:<post-Phase-3> changes; this helper does not.
    See plan §D6.

    Defensive against the same failure modes as
    `_capture_git_porcelain`: missing git binary, non-git working
    tree, permissions error, slow/stuck git invocation. The
    timeout is 30 seconds (vs 10 for porcelain) — diffs can be
    heavier on large repos.
    """
    if branch_per_task:
        cmd = ["git", "diff", f"origin/{base_branch}...HEAD"]
    else:
        cmd = ["git", "diff", "HEAD~1..HEAD"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
    if proc.returncode != 0:
        return ""
    return _tail_cap(proc.stdout, cap=_DIFF_CAP_BYTES)


def _resolve_services_required(
    *, task: dict, file_defaults: dict, phase_dict: dict,
) -> list[str]:
    """Compute the union of services_required for one task.

    Layers (additive, like gates):
      1. defaults.services_required
      2. phase.defaults.services_required_extra (loose dict)
      3. task.services_required_extra

    Order: project, then phase, then task (matches gate
    composition). Duplicates within or across layers are dropped;
    first-seen order is preserved for determinism.
    """
    seen: set[str] = set()
    out: list[str] = []
    for src in (
        file_defaults.get("services_required") or [],
        phase_dict.get("services_required_extra") or [],
        task.get("services_required_extra") or [],
    ):
        for name in src:
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out


def _service_teardown_after_task(
    service_manager: "ServiceManager | None",
) -> None:
    """Tear down gate- and task-scope services. No-op when manager
    is None. Idempotent on stopped services."""
    if service_manager is None:
        return
    service_manager.tear_down_for_scope("gate")
    service_manager.tear_down_for_scope("task")


# T5 (v0.3 §2.20) — explainer caps. Iterated in pinned order; see
# plans/task-5/plan.md §"Format invariants" #7. Resolver keys (NOT
# the YAML user-facing cap names — KEYS_TO_AUDIT labels).
_EXPLAIN_CAPS_KEYS: tuple[str, ...] = (
    "implement_max_turns",
    "plan_timeout_minutes",
    "review_timeout_minutes",
    "implement_timeout_minutes",
    "max_review_rounds",
    "max_attempts",
    "max_rate_limit_retries",
    "rate_limit_base_backoff_seconds",
)


def _gate_definition_origin(
    name: str, invariant_names: list[str],
) -> str | None:
    """Lookup-only origin label. None when the name is a plain
    project-registered gate. (Plan §D4.)"""
    if name in _BUILTIN_FACTORY_NAMES:
        return "built-in factory"
    if name in _BUILTIN_REGISTRY_NAMES:
        return "built-in registry"
    if name in invariant_names:
        return "invariants.yaml"
    return None


def _render_when_part(
    when_tuple: tuple[str, ...],
    ok: bool,
    reason: str,
) -> str:
    """Render the trailing `... -> RUN/SKIP (...)` clause for a gate
    row. Format A (`always -> RUN/SKIP`) is reserved for `when=()` or
    `when=("always",)`; otherwise the tokens are listed verbatim and
    the reason is parenthesised. (Plan §"When-evaluation suffix".)"""
    if not when_tuple or when_tuple == ("always",):
        return "always -> RUN" if ok else f"always -> SKIP ({reason})"
    tokens = ",".join(when_tuple)
    verdict = "RUN" if ok else "SKIP"
    return f"when={tokens} -> {verdict} ({reason})"


def explain_task(
    task: dict,
    file_defaults: dict,
    cli_overrides: dict,
    *,
    phases_by_id: dict[str, dict] | None = None,
    task_class_models: dict[str, dict] | None = None,
    registry: "GateRegistry | None" = None,
    invariant_names: list[str] | None = None,
) -> str:
    """Build a multi-line, human-readable explanation of how `task`
    would resolve under the same resolver + registry path
    `execute_task` walks. Returns the entire string; caller decides
    whether to print it.

    See plans/task-5/plan.md §"Pinned output format" for the line-by-
    line shape and §D2 for the signature contract.
    """
    if registry is None:
        registry = default_registry()
    if invariant_names is None:
        from claude_plan_execute.gates.invariants import loaded_invariant_names
        invariant_names = loaded_invariant_names()

    # Mirror execute_task lines 87-108 verbatim for `phase_dict`,
    # `task_class_*` and `resolve_task_config`. Same chain → same
    # provenance. Additionally keep `phase_full` (the OUTER phase
    # dict) — needed for the Phase-line title (per schema.py:142-150,
    # `title` is a sibling of `defaults`, not a child).
    phase_full: dict = {}
    phase_dict: dict = {}
    phase_id = task.get("phase")
    if phases_by_id and phase_id and phase_id in phases_by_id:
        phase_full = phases_by_id[phase_id]
        phase_dict = dict(phase_full.get("defaults") or {})

    task_class_dict: dict = {}
    task_class_name = task.get("task_class") or phase_dict.get("task_class")
    if task_class_models and task_class_name and task_class_name in task_class_models:
        task_class_dict = dict(task_class_models[task_class_name])

    resolved = resolve_task_config(
        task, file_defaults, cli_overrides,
        phase=phase_dict, task_class=task_class_dict,
        phase_id=phase_id, task_class_name=task_class_name,
    )

    # Build gate_names paired with source layers (Plan §D7).
    gate_names_with_layer: list[tuple[str, str]] = []
    for n in (file_defaults.get("gates") or []):
        gate_names_with_layer.append((n, "defaults"))
    for n in (phase_dict.get("gates_extra") or []):
        layer = f"phase:{phase_id}" if phase_id else "phase"
        gate_names_with_layer.append((n, layer))
    for n in (task.get("gates_extra") or []):
        gate_names_with_layer.append((n, "task"))

    # Same resolver + registry path execute_task uses (Plan §D1).
    test_command = resolved["test_command"]["value"]
    lint_command = resolved["lint_command"]["value"]
    untranslated_strings_command = resolved["untranslated_strings_command"]["value"]
    dev_server_url = resolved["dev_server_url"]["value"]
    persona_section = ""  # explain doesn't compose a real prompt body

    # Augment task dict so phase-inherited task_class is visible to
    # gate `when` filters. Mirrors `task_for_gates` at runner.py's
    # Phase-4 dispatch site. (Plan §D9.)
    task_for_when = {**task, "task_class": task_class_name}

    # Per-pair invocation (Plan §D11). Each input pair maps to its
    # own subset of resolved gates so duplicate names across layers
    # retain distinct source attribution and `invariant-grep`
    # expansion is bounded to its own pair.
    gate_lines: list[str] = []
    verdicts: list[str] = []  # "RUN" or "SKIP" — parallel to gate_lines
    unknown_with_layer: list[tuple[str, str]] = []

    for input_name, layer in gate_names_with_layer:
        sub_resolved, sub_unknown, sub_ucl, sub_ucu = _resolve_gate_specs(
            [input_name],
            test_command=test_command,
            lint_command=lint_command,
            untranslated_strings_command=untranslated_strings_command,
            dev_server_url=dev_server_url,
            persona_section=persona_section,
            invariant_names=invariant_names,
            registry=registry,
        )
        for resolved_name, gate in sub_resolved:
            if input_name == "invariant-grep":
                src = f"from {layer}; via invariant-grep:{resolved_name}"
            else:
                src = f"from {layer}"
            origin = _gate_definition_origin(resolved_name, invariant_names)
            if origin == "built-in factory" and gate.kind == "shell":
                origin = None
            origin_part = f"  ({origin})" if origin else ""
            ok, reason = _gate_when_matches(gate.when, task_for_when)
            when_part = _render_when_part(gate.when, ok, reason)
            line = (
                f"    {resolved_name}  [{gate.kind}]  "
                f"{src}{origin_part}  {when_part}"
            )
            gate_lines.append(line)
            verdicts.append("RUN" if ok else "SKIP")
        if sub_ucl:
            line = (
                f"    lint  [shell]  from {layer}  "
                f"always -> SKIP (lint_command not configured)"
            )
            gate_lines.append(line)
            verdicts.append("SKIP")
        if sub_ucu:
            line = (
                f"    untranslated-strings  [shell]  from {layer}  "
                f"always -> SKIP (untranslated_strings_command not configured)"
            )
            gate_lines.append(line)
            verdicts.append("SKIP")
        for u in sub_unknown:
            unknown_with_layer.append((u, layer))

    # ------ assemble the output -------------------------------------
    lines: list[str] = []
    lines.append(f"Task {task['id']}: {task['title']}")

    # Phase line
    if phase_id:
        title = phase_full.get("title") or phase_id
        lines.append(f"  Phase: {title} (id={phase_id})")
    else:
        lines.append("  Phase: (none)")

    # Class line
    model_value = resolved["model"]["value"]
    model_from = resolved["model"]["from"]
    effort_value = resolved["effort"]["value"]
    effort_from = resolved["effort"]["from"]
    class_label = task_class_name if task_class_name else "(none)"
    lines.append(
        f"  Class: {class_label} "
        f"(model={model_value} [from {model_from}], "
        f"effort={effort_value} [from {effort_from}])"
    )
    lines.append("")

    # Gates section
    total = len(verdicts)
    run_count = sum(1 for v in verdicts if v == "RUN")
    skip_count = total - run_count
    lines.append(
        f"  Gates ({total} total; {run_count} will run, "
        f"{skip_count} will skip):"
    )
    lines.extend(gate_lines)
    if unknown_with_layer:
        lines.append("")
        lines.append("  Unknown gate names (will be skipped at runtime):")
        for u, layer in unknown_with_layer:
            lines.append(f"    {u}  (from {layer})")
    lines.append("")

    # Services stub (Plan §D8)
    lines.append(
        "  Services required: (see T8 — services_required + "
        "services_required_extra union goes here once T5's stub is "
        "filled in)"
    )
    lines.append("")

    # Caps section
    lines.append("  Caps:")
    for key in _EXPLAIN_CAPS_KEYS:
        entry = resolved.get(key)
        if entry is None:
            continue
        lines.append(
            f"    {key}: {entry['value']} [from {entry['from']}]"
        )
    lines.append("")

    # Branch stub (Plan §D8)
    lines.append(
        "  Branch: (see T9 — rendered branch name + base_branch goes "
        "here once T5's stub is filled in)"
    )

    return "\n".join(lines) + "\n"


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
