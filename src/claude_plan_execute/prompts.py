"""Prompt builders for plan / review / revise / implement phases.

Pure functions — no I/O, no side effects. Each builder returns the exact
string sent to the `claude` CLI as the `-p` argument.
"""
from __future__ import annotations


# Per-attempt verbatim is already capped at 4 KB by `make_failure_record`
# at write time. Block-level cap applies to the whole rendered failure-
# context section, dropping oldest attempts first when over budget.
_BLOCK_CAP_BYTES = 8192

# Verbatim parenthetical hints by failure_class. Unmapped classes render
# the bare label "Verbatim:". Keys mirror state.FAILURE_CLASSES; absent
# entries are intentional (no useful hint).
_VERBATIM_LABEL_HINTS: dict[str, str] = {
    "review_rejected_after_max_rounds": "last review",
    "gate_failed":                      "gate output",
    "test_failed":                      "test output",
    "implementer_timeout":              "stderr",
    "implementer_max_turns":            "CLI output",
}


def build_persona_section(persona: str, persona_extra: str = "") -> str:
    """Compose the project-conventions block for a prompt.

    Returns "" when both inputs are empty (or whitespace-only). When
    either input is non-empty, returns a block of the form:

        ## Project conventions

        <global persona body>

        <persona_extra body>

    Trailing newlines are normalized so the caller can do
    `persona_section + body` without a separator. The global persona
    precedes the per-phase overlay. (See plan §D2 for composition
    order; future tasks 8 and 10 prepend their own blocks.)
    """
    g = persona.strip()
    e = persona_extra.strip()
    if not g and not e:
        return ""
    body_parts: list[str] = []
    if g:
        body_parts.append(g)
    if e:
        body_parts.append(e)
    body = "\n\n".join(body_parts)
    return f"## Project conventions\n\n{body}\n\n"


def build_decisions_section(decisions: str) -> str:
    """Compose the architectural-decisions block for the Plan prompt.

    Returns "" when input is empty or whitespace-only. Otherwise
    returns:

        ## Architectural decisions on file

        <body, rstripped>

    Trailing newlines are normalized so the caller can do
    `persona + decisions + body` without a manual separator. Only the
    Plan prompt receives this block (§D1); Review/Revise/Implement
    builders do not accept a decisions kwarg.
    """
    body = decisions.strip()
    if not body:
        return ""
    return f"## Architectural decisions on file\n\n{body}\n\n"


def _verbatim_label(failure_class: str) -> str:
    """Compose the verbatim line label for a given failure_class.

    Returns 'Verbatim (<hint>)' if the class has a known hint, else
    'Verbatim'.
    """
    hint = _VERBATIM_LABEL_HINTS.get(failure_class)
    if hint:
        return f"Verbatim ({hint})"
    return "Verbatim"


def _render_attempt(n: int, attempt: dict) -> str:
    """Render a single FailureRecord dict as the body of an Attempt section."""
    failure_class = attempt.get("failure_class", "unknown")
    reason = attempt.get("reason", "") or ""
    verbatim = attempt.get("verbatim", "") or ""
    hypothesis = attempt.get("hypothesis")
    artifacts = attempt.get("artifacts") or {}

    lines: list[str] = [f"Attempt {n}: {failure_class}", f"- Reason: {reason}"]
    if verbatim:
        lines.append(f"- {_verbatim_label(failure_class)}: {verbatim}")
    if hypothesis and str(hypothesis).strip():
        lines.append(f"- Hypothesis: {hypothesis}")
    if artifacts:
        paths = ", ".join(str(v) for v in artifacts.values())
        lines.append(f"- Artifacts: {paths}")
    return "\n".join(lines)


def _render_block(attempts: list[dict], n_dropped: int) -> str:
    """Render the full failure-context block for the given attempts."""
    if not attempts:
        return ""
    last_phase = attempts[-1].get("phase") or "unknown"
    parts: list[str] = [
        f"[FAILURE CONTEXT — previous attempt failed in phase: {last_phase}]"
    ]
    if n_dropped > 0:
        parts.append(f"[{n_dropped} earlier attempts omitted for length]")
    for offset, attempt in enumerate(attempts):
        parts.append(_render_attempt(n_dropped + offset + 1, attempt))
    parts.append("")
    parts.append("This attempt should:")
    parts.append("1. Address the prior failure verbatim before adding new work")
    has_hypothesis = any(
        a.get("hypothesis") and str(a.get("hypothesis")).strip()
        for a in attempts
    )
    if has_hypothesis:
        parts.append("2. Specifically check the hypothesis above")
    parts.append("[END FAILURE CONTEXT]")
    return "\n".join(parts) + "\n\n"


def build_failure_context_section(attempts: list[dict] | None) -> str:
    """Compose the failure-context block prepended to Plan/Implement prompts.

    Returns "" when `attempts` is empty or None — the no-op case for
    fresh runs. Otherwise returns a block of the form:

        [FAILURE CONTEXT — previous attempt failed in phase: <last_phase>]
        [<n> earlier attempts omitted for length]      (only if dropped)
        Attempt <k>: <failure_class>
        - Reason: <reason>
        - Verbatim (<hint>): <verbatim>                (if non-empty)
        - Hypothesis: <hypothesis>                     (if set)
        - Artifacts: <path1>, <path2>, ...             (if non-empty)
        ...

        This attempt should:
        1. Address the prior failure verbatim before adding new work
        2. Specifically check the hypothesis above     (if any)
        [END FAILURE CONTEXT]

    Per-attempt verbatim is already capped at 4 KB by
    `make_failure_record`. The full block is capped at 8 KB UTF-8 by
    dropping the oldest attempts first; if a single attempt still
    exceeds 8 KB, the rendered string is hard-truncated with a
    '[truncated]' marker. Trailing '\\n\\n' so the caller can do
    `failure_context_section + persona + ...` with no manual separator.
    """
    if not attempts:
        return ""
    attempts = [a for a in attempts if isinstance(a, dict)]
    if not attempts:
        return ""
    kept = list(attempts)
    n_dropped = 0
    while kept:
        block = _render_block(kept, n_dropped=n_dropped)
        if len(block.encode("utf-8")) <= _BLOCK_CAP_BYTES:
            return block
        kept = kept[1:]
        n_dropped += 1
    block = _render_block([attempts[-1]], n_dropped=len(attempts) - 1)
    encoded = block.encode("utf-8")
    if len(encoded) > _BLOCK_CAP_BYTES:
        truncated = encoded[:_BLOCK_CAP_BYTES].decode("utf-8", errors="ignore")
        block = truncated + "\n[truncated]\n"
    return block


def build_plan_prompt(
    task: dict, plan_path: str,
    *, persona: str = "", decisions: str = "", failure_context: str = "",
) -> str:
    files_section = ""
    key_files = task.get("key_files", [])
    if key_files:
        files_list = "\n".join(f"  - {f}" for f in key_files)
        files_section = (
            f"\nKey files to START with (non-exhaustive — explore the codebase "
            f"for additional relevant files):\n{files_list}\n"
        )

    return failure_context + persona + decisions + (
        f"You are planning the implementation of a task in this codebase.\n"
        f"Read all relevant files before producing your plan.\n\n"
        f"TASK {task['id']}: {task['title']}\n\n"
        f"DESCRIPTION:\n{task['description']}\n"
        f"{files_section}\n"
        f"INSTRUCTIONS:\n"
        f"1. Explore the codebase thoroughly. Read the listed files and any others you discover.\n"
        f"2. Produce a DETAILED implementation plan covering:\n"
        f"   - Exact files to create or modify\n"
        f"   - For each file: what changes, in what order\n"
        f"   - New models, functions, or classes to create\n"
        f"   - Tests to write or update\n"
        f"   - Dependencies between changes (what must happen first)\n"
        f"   - Potential risks or edge cases\n"
        f"3. The plan must be specific enough that another developer could implement it without questions.\n"
        f"4. Write your plan to {plan_path}\n\n"
        f"Do NOT implement anything yet. Only produce the plan.\n"
    )


def build_review_prompt(task: dict, plan_path: str, review_path: str,
                        prev_feedback: str = "", *, persona: str = "") -> str:
    feedback_section = ""
    if prev_feedback:
        feedback_section = (
            f"\nThe plan was revised based on previous feedback. "
            f"Check whether the issues are resolved.\n"
        )

    return persona + (
        f"You are reviewing an implementation plan.\n"
        f"Your job is to find problems, gaps, and risks BEFORE implementation.\n\n"
        f"TASK {task['id']}: {task['title']}\n\n"
        f"TASK DESCRIPTION:\n{task['description']}\n\n"
        f"Read the plan at: {plan_path}\n"
        f"Read any files the plan references to verify accuracy against the actual codebase.\n"
        f"{feedback_section}\n"
        f"REVIEW CRITERIA:\n"
        f"1. COMPLETENESS — Does the plan cover everything in the task description?\n"
        f"2. ACCURACY — Are file paths, function names, and structures correct? Verify against code.\n"
        f"3. CONSISTENCY — Does the plan conflict with existing code?\n"
        f"4. RISKS — Edge cases, breaking changes, migration issues?\n"
        f"5. TESTS — Adequate test coverage?\n"
        f"6. ORDER — Dependencies correctly ordered?\n\n"
        f"Write your review to {review_path} with:\n\n"
        f"## Verdict: APPROVED or NEEDS_REVISION\n\n"
        f"## Issues (if NEEDS_REVISION)\n"
        f"- What's wrong / Where in the plan / How to fix it\n\n"
        f"## Comments (optional)\n\n"
        f"Block on correctness, completeness, and risk — not style. "
        f"If safe to implement, approve it.\n"
    )


def build_revise_prompt(task: dict, plan_path: str, review_path: str,
                        *, persona: str = "") -> str:
    return persona + (
        f"You are revising an implementation plan based on reviewer feedback.\n\n"
        f"TASK {task['id']}: {task['title']}\n\n"
        f"Read the current plan at: {plan_path}\n"
        f"Read the review feedback at: {review_path}\n\n"
        f"Address ALL issues raised in the review. Update the plan file in place.\n\n"
        f"Do NOT implement anything. Only revise the plan.\n"
    )


def build_implement_prompt(task: dict, plan_path: str,
                           commit_message: str, *, persona: str = "",
                           failure_context: str = "") -> str:
    return failure_context + persona + (
        f"You are implementing a validated plan.\n"
        f"The plan has been reviewed and approved. Follow it precisely.\n\n"
        f"TASK {task['id']}: {task['title']}\n\n"
        f"Read the validated plan at: {plan_path}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Read the plan completely before starting.\n"
        f"2. Implement every item in the plan, in order.\n"
        f"3. Commit with message:\n"
        f"   {commit_message}\n\n"
        f"Do not skip any part of the plan. Do not add unplanned features.\n"
    )
