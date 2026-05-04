"""Prompt builders for plan / review / revise / implement phases.

Pure functions — no I/O, no side effects. Each builder returns the exact
string sent to the `claude` CLI as the `-p` argument.
"""
from __future__ import annotations

import re


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


# T15 (v0.3 §"cross-task-memory"). Each per-phase block is injected
# inside the body string of its respective builder. Keep wording
# tight — every byte ships in every prompt.

_MEMORY_SECTION_PLAN = """## Memory

You have a project memory directory auto-loaded by Claude Code at
session start (`~/.claude/projects/<encoded-cwd>/memory/MEMORY.md`).
Before planning:
1. Read the loaded MEMORY.md, especially entries with type `feedback`
   or `project` — these capture pitfalls and constraints discovered
   in earlier tasks.
2. Apply any learnings that bear on this task's design choices.
3. When a memory drives a step in your plan, ANNOTATE that step with
   `[MEM: <memory_name>]` (using the exact memory file's `name`
   slug). The runner counts these citations to decide which memories
   stay relevant.

"""

_MEMORY_SECTION_REVIEW = """## Memory

The project memory directory (`~/.claude/projects/<encoded-cwd>/
memory/`) is auto-loaded for this session. Before reviewing:
1. Cross-check the plan against MEMORY.md, especially entries with
   type `feedback` or `project`. Flag plans that contradict an
   accepted learning (e.g., "the plan does X, but memory entry
   `feedback_pydantic_v2` says Y").
2. Verify every `[MEM: <name>]` citation in the plan resolves to a
   real, currently-live memory file in the directory. Hallucinated
   names are a NEEDS_REVISION issue.

"""

_MEMORY_SECTION_IMPLEMENT = """## Memory

You have four prompt-overlay layers feeding this session. Pick the
right one when persisting knowledge:

  1. `docs/persona.md` (hand-curated): stable conventions.
  2. `docs/decisions/` (hand-curated): ADR-style architectural
     choices.
  3. `~/.claude/projects/<encoded-cwd>/memory/MEMORY.md` + per-memory
     files (agent-curated, auto-loaded): tactical pitfalls discovered
     in flight. THIS is the layer you write to.
  4. failure_context (runner-injected, transient): per-attempt prior
     failure verbatim.

If during this implementation you discover a non-obvious pitfall,
gotcha, or trick worth saving for future tasks, save it as a memory
using the standard Claude Code pattern.

FRONTMATTER REQUIRED on every new memory:

```
---
name: <slug>
description: <one-line — used to decide relevance in future tasks>
type: user|feedback|project|reference
lifecycle: evergreen | slate:<slate-id> | task:<id>
citation_count: 0
---
```

DEFAULT LIFECYCLE BY TYPE (override only when you know better):
  - type=project   -> lifecycle: slate:$CLAUDE_PLAN_EXECUTE_SLATE_ID
  - type=feedback  -> lifecycle: evergreen
  - type=reference -> lifecycle: evergreen
  - type=user      -> lifecycle: evergreen

The current slate id is exposed via the env var
`CLAUDE_PLAN_EXECUTE_SLATE_ID`. Read it with `echo
$CLAUDE_PLAN_EXECUTE_SLATE_ID` (or via Bash). Override default ->
explicit (e.g., `lifecycle: task:8`) when the lesson applies only
until a specific task lands.

WARNING — `lifecycle: task:<N>` footgun: N must be STRICTLY GREATER
than the current task id. Memories scoped to the CURRENT task
(`task:<self>`) are pruned at this task's success boundary and will
NEVER reach the next planner. If your lesson is meant to die at the
end of this task, just don't write it; if it should survive into the
next planning round, choose `evergreen` or `task:<future_id>`.

TERSENESS BUDGET: title <= 60 chars, body <= 5 lines.

DO NOT SAVE:
  - trivia or one-off task state;
  - restated `docs/persona.md` content;
  - project-state recap of what just changed in code (that belongs
    in commit messages, not memory).

"""


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
        _MEMORY_SECTION_PLAN +
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
        _MEMORY_SECTION_REVIEW +
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
    tid = task["id"]
    progress_path = f"plans/task-{tid}/progress.md"
    return failure_context + persona + (
        _MEMORY_SECTION_IMPLEMENT +
        f"You are implementing a validated plan.\n"
        f"The plan has been reviewed and approved. Follow it precisely.\n\n"
        f"TASK {tid}: {task['title']}\n\n"
        f"Read the validated plan at: {plan_path}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Read the plan completely before starting.\n"
        f"2. Implement every item in the plan, in order.\n"
        f"3. Work in commit-sized chunks. After each logically-complete\n"
        f"   chunk, run:\n"
        f"     git add -A && git commit -m \"task-{tid} WIP: <chunk-description>\"\n"
        f"   Continue with the next chunk.\n"
        f"4. Maintain a checklist at {progress_path} with one item per\n"
        f"   plan step. Mark items as you go:\n"
        f"     [ ] not started\n"
        f"     [~] in progress\n"
        f"     [x] done\n"
        f"   Update the file BEFORE each commit so the on-disk state matches\n"
        f"   what's committed.\n"
        f"5. If you sense the turn budget is running low, commit the current\n"
        f"   chunk, update {progress_path}, and finish the session cleanly\n"
        f"   (do NOT start a new chunk you can't finish). The runner detects\n"
        f"   the partial progress and resumes on the next invocation.\n"
        f"6. After every chunk is committed and the checklist reads all\n"
        f"   [x], create the final commit with message:\n"
        f"   {commit_message}\n\n"
        f"Do not skip any part of the plan. Do not add unplanned features.\n"
    )


def build_resume_prompt(
    task: dict,
    plan_path: str,
    progress_path: str,
    progress_md_text: str,
    dirty_tree_porcelain: str,
    commit_message: str,
    *,
    persona: str = "",
    failure_context: str = "",
) -> str:
    """Build the Implement-phase prompt for a resume run.

    Pure: no I/O. The runner pre-reads `progress.md` and runs
    `git status --porcelain`, passing both as strings. Composition order
    matches `build_implement_prompt`: `failure_context + persona + body`.
    `dirty_tree_porcelain` may be the empty string when the working
    tree is clean — the builder renders a clean-tree section in that
    case.
    """
    tid = task["id"]
    if dirty_tree_porcelain.strip():
        dirty_section = (
            f"## Uncommitted hunks detected\n\n"
            f"`git status --porcelain` reports the following uncommitted\n"
            f"changes from the prior session:\n\n"
            f"```\n{dirty_tree_porcelain.rstrip()}\n```\n\n"
            f"Before continuing the next chunk, choose ONE:\n"
            f"  (a) Commit them as the next chunk:\n"
            f"      git add -A && git commit -m \"task-{tid} WIP: <chunk-description>\"\n"
            f"  (b) Revert them with a brief explanation in your reply\n"
            f"      (`git restore -SW .` for tracked changes; `git clean\n"
            f"      -fd` for untracked).\n"
            f"Do NOT begin new work while uncommitted hunks remain — they\n"
            f"will be lost or conflated with the next chunk's diff.\n\n"
        )
    else:
        dirty_section = (
            f"## Working tree is clean\n\n"
            f"`git status --porcelain` reports no uncommitted changes.\n"
            f"Continue with the next [~] or [ ] item below.\n\n"
        )
    return failure_context + persona + (
        f"You are RESUMING an in-progress implementation.\n"
        f"A prior session was interrupted (turn budget, timeout, or\n"
        f"manual stop). The plan has already been reviewed and approved;\n"
        f"earlier chunks have been committed as `task-{tid} WIP: ...`.\n\n"
        f"TASK {tid}: {task['title']}\n\n"
        f"Read the validated plan at: {plan_path}\n"
        f"Read the existing checklist at: {progress_path}\n\n"
        f"## Current checklist (from {progress_path})\n\n"
        f"```\n{progress_md_text.rstrip()}\n```\n\n"
        f"{dirty_section}"
        f"INSTRUCTIONS:\n"
        f"1. Re-read the plan and the checklist to orient.\n"
        f"2. Continue with the FIRST item marked [~] (in progress) or\n"
        f"   [ ] (not started). Do not redo items already marked [x].\n"
        f"3. Work in commit-sized chunks. After each logically-complete\n"
        f"   chunk, run:\n"
        f"     git add -A && git commit -m \"task-{tid} WIP: <chunk-description>\"\n"
        f"4. Update {progress_path} BEFORE each commit so the on-disk\n"
        f"   state matches what's committed.\n"
        f"5. If you sense the turn budget is running low, commit the\n"
        f"   current chunk, update {progress_path}, and finish cleanly.\n"
        f"6. After every checklist item reads [x], create the final\n"
        f"   commit with message:\n"
        f"   {commit_message}\n\n"
        f"Do not re-plan. Do not add unplanned features. Treat the\n"
        f"existing plan as authoritative; flag any discovered gaps in\n"
        f"a follow-up but do not expand scope here.\n"
    )


_PROGRESS_ITEM_RE = re.compile(r"^\s*[-*+]\s*\[([ x~])\]", re.MULTILINE)


def parse_progress_status(text: str) -> tuple[int, int, int]:
    """Count [x] / [~] / [ ] markers in a progress.md body.

    Returns (n_done, n_in_progress, n_pending). Markers must appear at
    the start of a list item (-, *, or +) optionally preceded by
    whitespace. Other markdown (headings, paragraphs, fenced blocks)
    is ignored. Capital `X` is NOT recognized — the prompt mandates
    lowercase. Pure: no I/O.
    """
    matches = _PROGRESS_ITEM_RE.findall(text)
    n_done = sum(1 for m in matches if m == "x")
    n_in_progress = sum(1 for m in matches if m == "~")
    n_pending = sum(1 for m in matches if m == " ")
    return n_done, n_in_progress, n_pending
