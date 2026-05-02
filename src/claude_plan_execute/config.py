"""Settings resolution: CLI > per-task > file defaults > hardcoded."""
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
}


def resolve(key: str, task: dict, file_defaults: dict, cli_overrides: dict) -> object:
    if key in cli_overrides and cli_overrides[key] is not None:
        return cli_overrides[key]
    if key in task and task[key] is not None:
        return task[key]
    if key in file_defaults and file_defaults[key] is not None:
        return file_defaults[key]
    return HARDCODED_DEFAULTS.get(key)
