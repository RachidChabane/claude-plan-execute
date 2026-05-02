"""Settings resolution: cli > task > task_class_models[class] > phase > project > hardcoded."""
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
}


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
    """
    if key in cli_overrides and cli_overrides[key] is not None:
        return cli_overrides[key]
    if key in task and task[key] is not None:
        return task[key]
    if task_class and key in task_class and task_class[key] is not None:
        return task_class[key]
    if phase and key in phase and phase[key] is not None:
        return phase[key]
    if key in file_defaults and file_defaults[key] is not None:
        return file_defaults[key]
    return HARDCODED_DEFAULTS.get(key)
