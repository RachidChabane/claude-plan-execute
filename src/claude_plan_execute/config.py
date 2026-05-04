"""Settings resolution: cli > task > task_class_models[class] > phase > project > hardcoded.

`resolve_task_config(...)` produces a per-task dict with provenance per key,
which `runner.execute_task` writes to `plans/task-{id}/resolved-config.json`
as the audit artifact. The runner extracts runtime values from the same dict
to guarantee the on-disk JSON matches what drove the run.
"""
from __future__ import annotations

HARDCODED_DEFAULTS: dict = {
    "model": "opus",
    "effort": "max",
    "max_review_rounds": 3,
    "plan_timeout_minutes": 20,
    "review_timeout_minutes": 15,
    "implement_timeout_minutes": 60,
    "implement_max_turns": 200,
    "test_command": "pytest",
    "plans_dir": "plans",
    "max_attempts": 3,
    "on_max_review_rounds": "fail",
    "persona_file": "docs/persona.md",
    "decisions_dir": "docs/decisions",
    "dev_server_url": "http://localhost:5173",
    "max_rate_limit_retries": 3,
    "rate_limit_base_backoff_seconds": 60,
    # T9 (v0.3 §2.12) — per-task branches + auto-PR. NB:
    # `pr_body_template` deliberately absent so unset → None at the
    # resolver, which `_render_pr_body` handles as "use the built-in
    # default body".
    "branch_per_task": False,
    "base_branch": "main",
    "task_branch_template": "task/{id}-{slug}",
    "auto_pr": False,
    "auto_merge": False,
    # T16 (v0.3) — threshold for memory-curate gate auto-trigger.
    "memory_curation_threshold": 30,
    # T17 (v0.3) — auto-split. Master switch is OFF by default; the
    # feature is opt-in. The hardcoded match list documents the two
    # failure classes the runner currently routes through this path
    # (review-rounds exhaustion, implementer max-turns). Adding a
    # class here without a runner-side trigger does NOT activate
    # split for that class — the runner only invokes the helper at
    # the two pinned sites.
    "auto_split_on_max_rounds": False,
    # NB: shared mutable list. _resolve_with_source returns this
    # reference verbatim; callers MUST treat it as read-only (the
    # splitter helper does `failure_class not in classes`, fine).
    # Mutating this list would leak across all callers. First
    # mutable entry in HARDCODED_DEFAULTS — flagged for future
    # additions.
    "auto_split_failure_classes": [
        "review_rejected_after_max_rounds",
        "implementer_max_turns",
    ],
    "max_split_depth": 1,
}


# Keys serialized to plans/task-{id}/resolved-config.json. Policy
# (task 5 §D3): currently-consumed keys only. Task 6 added
# on_max_review_rounds. Task 7 added persona_file. Task 8 added
# decisions_dir. Task 13 added lint_command. Task 14 added
# invariants_file. Task 17 added untranslated_strings_command.
# Task 20 added dev_server_url. Task 1 of v0.3 added
# max_rate_limit_retries and rate_limit_base_backoff_seconds.
# Task 17 added auto_split_on_max_rounds, auto_split_failure_classes,
# max_split_depth.
# Future tasks extend this tuple as new keys become
# resolver-consumable.
KEYS_TO_AUDIT: tuple[str, ...] = (
    "model",
    "effort",
    "test_command",
    "lint_command",
    "plans_dir",
    "max_review_rounds",
    "plan_timeout_minutes",
    "review_timeout_minutes",
    "implement_timeout_minutes",
    "implement_max_turns",
    "max_attempts",
    "on_max_review_rounds",
    "persona_file",
    "decisions_dir",
    "invariants_file",
    "untranslated_strings_command",
    "dev_server_url",
    "max_rate_limit_retries",
    "rate_limit_base_backoff_seconds",
    # T9 (v0.3 §2.12) — per-task branches + auto-PR.
    "branch_per_task",
    "base_branch",
    "task_branch_template",
    "auto_pr",
    "auto_merge",
    "pr_body_template",
    # T16 (v0.3).
    "memory_curation_threshold",
    # T17 (v0.3) — auto-split.
    "auto_split_on_max_rounds",
    "auto_split_failure_classes",
    "max_split_depth",
)


def _resolve_with_source(
    key: str,
    task: dict,
    file_defaults: dict,
    cli_overrides: dict,
    *,
    phase: dict | None = None,
    task_class: dict | None = None,
    phase_id: str | None = None,
    task_class_name: str | None = None,
) -> tuple[object, str]:
    """Resolve `key` and return `(value, source_label)`.

    Source labels: cli, task, task_class[:name], phase[:id], project_defaults,
    hardcoded, unset. See plan §D2 for the full table. The `:<name>` and
    `:<id>` qualifiers are appended only when `task_class_name=` /
    `phase_id=` are passed; `resolve()` (the legacy single-value API) does
    not pass them and gets unqualified labels back.
    """
    if key in cli_overrides and cli_overrides[key] is not None:
        return cli_overrides[key], "cli"
    if key in task and task[key] is not None:
        return task[key], "task"
    if task_class and key in task_class and task_class[key] is not None:
        label = f"task_class:{task_class_name}" if task_class_name else "task_class"
        return task_class[key], label
    if phase and key in phase and phase[key] is not None:
        label = f"phase:{phase_id}" if phase_id else "phase"
        return phase[key], label
    if key in file_defaults and file_defaults[key] is not None:
        return file_defaults[key], "project_defaults"
    if key in HARDCODED_DEFAULTS:
        return HARDCODED_DEFAULTS[key], "hardcoded"
    return None, "unset"


def resolve(
    key: str,
    task: dict,
    file_defaults: dict,
    cli_overrides: dict,
    *,
    phase: dict | None = None,
    task_class: dict | None = None,
) -> object:
    """Resolve a config key across six layers (highest precedence first).

    Order:
      1. CLI overrides
      2. Task-level fields
      3. Task-class config (`defaults.task_class_models[class]`)
      4. Phase defaults (`phases[i].defaults`)
      5. Project defaults (effective_defaults(tasks_file))
      6. Hardcoded defaults

    `None` values fall through. Missing keys fall through. Unknown keys with
    no hardcoded entry return `None`.

    Implemented as a thin wrapper over `_resolve_with_source` to guarantee
    the resolver and the audit-dict producer never drift. (Plan §D1.)
    """
    value, _ = _resolve_with_source(
        key, task, file_defaults, cli_overrides,
        phase=phase, task_class=task_class,
        phase_id=None, task_class_name=None,
    )
    return value


def resolve_task_config(
    task: dict,
    file_defaults: dict,
    cli_overrides: dict,
    *,
    phase: dict | None = None,
    task_class: dict | None = None,
    phase_id: str | None = None,
    task_class_name: str | None = None,
) -> dict:
    """Resolve every key in `KEYS_TO_AUDIT` and return a JSON-serializable
    dict of `{<key>: {"value": <v>, "from": <source_label>}}`.

    The runner writes this dict verbatim to
    `plans/task-{id}/resolved-config.json` and also extracts every runtime
    value from it (model, effort, timeouts, …). One source of truth.

    `phase_id=` and `task_class_name=` are used only to qualify the
    provenance labels (`"phase:<id>"`, `"task_class:<name>"`); they don't
    influence which value is chosen.
    """
    out: dict = {}
    for key in KEYS_TO_AUDIT:
        value, source = _resolve_with_source(
            key, task, file_defaults, cli_overrides,
            phase=phase, task_class=task_class,
            phase_id=phase_id, task_class_name=task_class_name,
        )
        out[key] = {"value": value, "from": source}
    return out
