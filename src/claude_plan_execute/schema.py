"""Pydantic v2 schema for tasks.yaml (versions 1 and 2).

Validation runs at load time. Errors are formatted by `format_validation_error`
which produces actionable messages: file path, dotted key path, and a
'Did you mean ...?' suggestion for unknown-key (typo) errors.
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path
from typing import Any, Literal

try:
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        ValidationError,
        model_validator,
    )
except ImportError:
    print(
        "Error: pydantic>=2.0 is required. Install with: pip install 'pydantic>=2.0,<3.0'",
        file=sys.stderr,
    )
    sys.exit(1)


class Caps(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_minutes_per_phase: int | None = None
    max_review_rounds: int | None = None
    max_implement_turns: int | None = None
    max_attempts: int | None = None


class TaskClassConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    effort: str | None = None


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # v2 known fields
    model: str | None = None
    effort: str | None = None
    test_command: str | None = None
    lint_command: str | None = None
    plans_dir: str | None = None
    persona_file: str | None = None
    decisions_dir: str | None = None
    invariants_file: str | None = None
    gates: list[str] | None = None
    task_class_models: dict[str, TaskClassConfig] | None = None
    caps: Caps | None = None
    on_max_review_rounds: Literal["fail", "auto_approve"] | None = None

    # Legacy v1 fields (still accepted; deprecation warning fires for v2 files)
    plan_timeout_minutes: int | None = None
    review_timeout_minutes: int | None = None
    implement_timeout_minutes: int | None = None
    max_review_rounds: int | None = None
    implement_max_turns: int | None = None


class Phase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str | None = None
    # Loose dict for forward-compat with task-class-name strings, gates_extra,
    # persona_extra, and any future per-phase override key. Trade-off: typos
    # under phase.defaults.<x> won't be caught by Pydantic validation.
    defaults: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | str
    title: str
    description: str
    phase: str | None = None
    task_class: str | None = None
    key_files: list[str] | None = None
    commit_message: str | None = None
    depends_on: list[int | str] = Field(default_factory=list)

    # Per-task resolver overrides (all optional)
    model: str | None = None
    effort: str | None = None
    plans_dir: str | None = None
    test_command: str | None = None
    lint_command: str | None = None
    plan_timeout_minutes: int | None = None
    review_timeout_minutes: int | None = None
    implement_timeout_minutes: int | None = None
    implement_max_turns: int | None = None
    max_review_rounds: int | None = None


class TasksFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int | None = None
    defaults: Defaults = Field(default_factory=Defaults)
    phases: list[Phase] = Field(default_factory=list)
    tasks: list[Task] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_cross_references(self) -> "TasksFile":
        phase_ids = {p.id for p in self.phases}
        class_names: set[str] = set()
        if self.defaults.task_class_models:
            class_names = set(self.defaults.task_class_models.keys())

        # 1. Tasks reference known phases.
        for t in self.tasks:
            if t.phase is not None and t.phase not in phase_ids:
                known = ", ".join(sorted(phase_ids)) if phase_ids else "(none defined)"
                raise ValueError(
                    f"Task {t.id} references unknown phase '{t.phase}'. "
                    f"Known phases: {known}"
                )

        # 2. Tasks reference known task classes (when class map is non-empty).
        if class_names:
            for t in self.tasks:
                if t.task_class is not None and t.task_class not in class_names:
                    raise ValueError(
                        f"Task {t.id} references unknown task_class "
                        f"'{t.task_class}'. Known classes: "
                        f"{', '.join(sorted(class_names))}"
                    )

            # 3. Phase defaults reference known task classes.
            for p in self.phases:
                tc = p.defaults.get("task_class")
                if tc is not None and tc not in class_names:
                    raise ValueError(
                        f"Phase '{p.id}' defaults reference unknown task_class "
                        f"'{tc}'. Known classes: "
                        f"{', '.join(sorted(class_names))}"
                    )
        return self


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------

def _valid_keys_for_extra_error(loc: tuple) -> list[str]:
    """Return the list of valid field names at the parent of an extra_forbidden
    error. Hand-rolled location->model walk for the v2 schema surface — small
    enough to be explicit, large enough to need a comment."""
    parent = tuple(loc[:-1])

    if parent == ():
        return list(TasksFile.model_fields.keys())
    if parent == ("defaults",):
        return list(Defaults.model_fields.keys())
    if parent == ("defaults", "caps"):
        return list(Caps.model_fields.keys())
    # ("defaults", "task_class_models", "<class_name>", ...)
    if len(parent) >= 3 and parent[:2] == ("defaults", "task_class_models"):
        return list(TaskClassConfig.model_fields.keys())
    # ("phases", N, ...) where N is an int
    if len(parent) >= 2 and parent[0] == "phases" and isinstance(parent[1], int):
        if len(parent) == 2:
            return list(Phase.model_fields.keys())
    # ("tasks", N, ...) where N is an int
    if len(parent) >= 2 and parent[0] == "tasks" and isinstance(parent[1], int):
        if len(parent) == 2:
            return list(Task.model_fields.keys())
    return []


def format_validation_error(exc: ValidationError, path: Path) -> str:
    """Format ValidationError into a human-readable, actionable string.

    Includes file path, dotted location, and 'Did you mean X?' for
    extra_forbidden errors when a close-match suggestion exists.
    """
    lines: list[str] = [f"Validation errors in {path}:"]
    for err in exc.errors():
        loc_parts = [str(seg) for seg in err["loc"]]
        loc_str = ".".join(loc_parts) if loc_parts else "<root>"

        if err["type"] == "extra_forbidden":
            offending = str(err["loc"][-1]) if err["loc"] else "<unknown>"
            valid = _valid_keys_for_extra_error(tuple(err["loc"]))
            close = difflib.get_close_matches(offending, valid, n=1, cutoff=0.6)
            suffix = f" Did you mean '{close[0]}'?" if close else ""
            lines.append(f"  - {loc_str}: unknown key '{offending}'.{suffix}")
        elif err["type"] == "missing":
            lines.append(f"  - {loc_str}: missing required field.")
        elif err["type"] == "too_short":
            lines.append(f"  - {loc_str}: list must be non-empty.")
        elif err["type"] == "value_error":
            # `model_validator(mode='after')` raises ValueError; pydantic wraps
            # the message under err['msg'] like "Value error, <text>".
            msg = err["msg"]
            if msg.startswith("Value error, "):
                msg = msg[len("Value error, "):]
            lines.append(f"  - {loc_str}: {msg}")
        else:
            lines.append(f"  - {loc_str}: {err['msg']}")
    return "\n".join(lines)


__all__ = [
    "Caps",
    "TaskClassConfig",
    "Defaults",
    "Phase",
    "Task",
    "TasksFile",
    "ValidationError",
    "format_validation_error",
]
