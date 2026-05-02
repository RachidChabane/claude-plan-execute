"""Prompt builders for plan / review / revise / implement phases.

Pure functions — no I/O, no side effects. Each builder returns the exact
string sent to the `claude` CLI as the `-p` argument.
"""
from __future__ import annotations


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


def build_plan_prompt(task: dict, plan_path: str, *, persona: str = "") -> str:
    files_section = ""
    key_files = task.get("key_files", [])
    if key_files:
        files_list = "\n".join(f"  - {f}" for f in key_files)
        files_section = (
            f"\nKey files to START with (non-exhaustive — explore the codebase "
            f"for additional relevant files):\n{files_list}\n"
        )

    return persona + (
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


def build_implement_prompt(task: dict, plan_path: str, test_command: str,
                           commit_message: str, *, persona: str = "") -> str:
    return persona + (
        f"You are implementing a validated plan.\n"
        f"The plan has been reviewed and approved. Follow it precisely.\n\n"
        f"TASK {task['id']}: {task['title']}\n\n"
        f"Read the validated plan at: {plan_path}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Read the plan completely before starting.\n"
        f"2. Implement every item in the plan, in order.\n"
        f"3. Run tests: {test_command}\n"
        f"4. Fix any test failures before proceeding.\n"
        f"5. Commit with message:\n"
        f"   {commit_message}\n\n"
        f"Do not skip any part of the plan. Do not add unplanned features.\n"
    )
