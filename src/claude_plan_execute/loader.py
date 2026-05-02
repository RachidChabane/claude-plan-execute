"""tasks.yaml / tasks.json discovery, loading, and validation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

from claude_plan_execute.colors import red, yellow
from claude_plan_execute.schema import (
    TasksFile,
    ValidationError,
    format_validation_error,
)


def find_tasks_file(
    filename: str = "tasks.yaml",
    start_dir: Path | None = None,
) -> Path | None:
    base = start_dir if start_dir is not None else Path.cwd()
    for candidate in [filename, f"docs/{filename}"]:
        p = base / candidate
        if p.exists():
            return p
    return None


def load_tasks_data(path: Path) -> dict:
    """Raw read of a tasks file. No validation, no sys.exit. Returns the
    parsed dict (or {} for an empty YAML file)."""
    with open(path) as f:
        if path.suffix == ".json":
            return json.load(f)
        return yaml.safe_load(f) or {}


def load_tasks_file(path: Path) -> TasksFile:
    """Load and validate the tasks file. Returns the validated TasksFile.
    Calls sys.exit(1) on validation errors with a formatted message including
    file path, key path, and (for typos) suggested correction. Emits a
    deprecation warning on v2 files that use legacy `*_timeout_minutes` keys.
    """
    raw = load_tasks_data(path)

    if not isinstance(raw, dict) or "tasks" not in raw:
        print(red(f"Error: {path} must contain a 'tasks' key"), file=sys.stderr)
        sys.exit(1)

    if raw.get("version") is None:
        print(yellow(f"Warning: {path} has no 'version' field. Assuming version 1."))

    try:
        tasks_file = TasksFile.model_validate(raw)
    except ValidationError as exc:
        print(red(format_validation_error(exc, path)), file=sys.stderr)
        sys.exit(1)

    # Deprecation warnings: v2 files that still set legacy timeout/turns/rounds
    # keys at defaults level. v1 files are silent.
    if tasks_file.version == 2:
        legacy_pairs = {
            "plan_timeout_minutes": "caps.max_minutes_per_phase",
            "review_timeout_minutes": "caps.max_minutes_per_phase",
            "implement_timeout_minutes": "caps.max_minutes_per_phase",
            "max_review_rounds": "caps.max_review_rounds",
            "implement_max_turns": "caps.max_implement_turns",
        }
        defaults_set = tasks_file.defaults.model_dump(exclude_none=True)
        for legacy, replacement in legacy_pairs.items():
            if legacy in defaults_set:
                print(yellow(
                    f"Warning: {path}: 'defaults.{legacy}' is deprecated; "
                    f"use defaults.{replacement} instead."
                ))

    # Version-aware default flip warning (task 6 §D3 / §D4).
    # v1 / no-version files default to the legacy "auto_approve"
    # behavior; tell users so they can opt into the safer "fail" or
    # migrate to v2.
    if (
        (tasks_file.version is None or tasks_file.version == 1)
        and tasks_file.defaults.on_max_review_rounds is None
    ):
        print(yellow(
            f"Warning: {path}: version 1 defaults to "
            f"'on_max_review_rounds: auto_approve' (unsafe for autonomous "
            f"runs). The version 2 default is 'fail'. Set "
            f"'defaults.on_max_review_rounds: fail' explicitly or migrate "
            f"to 'version: 2'."
        ))

    return tasks_file


def effective_defaults(tasks_file: TasksFile) -> dict:
    """Flatten the validated defaults into the dict the resolver consumes as
    the project-defaults layer.

    - Drops nested `caps` and `task_class_models` (not resolver-consumable).
    - Expands `caps.max_minutes_per_phase` into three legacy resolver keys
      (`plan_timeout_minutes`, `review_timeout_minutes`,
      `implement_timeout_minutes`). Caps wins over a directly-set legacy
      value (S2). Deprecation warning was already emitted at load.
    - Expands `caps.max_review_rounds`, `caps.max_implement_turns`,
      `caps.max_attempts` into resolver-friendly keys.
    - Injects `on_max_review_rounds: "auto_approve"` for v1 / no-version
      files when the user hasn't set it explicitly (task 6 §D1).
    """
    out = tasks_file.defaults.model_dump(exclude_none=True)
    out.pop("caps", None)
    out.pop("task_class_models", None)

    caps = tasks_file.defaults.caps
    if caps is not None:
        if caps.max_minutes_per_phase is not None:
            out["plan_timeout_minutes"] = caps.max_minutes_per_phase
            out["review_timeout_minutes"] = caps.max_minutes_per_phase
            out["implement_timeout_minutes"] = caps.max_minutes_per_phase
        if caps.max_review_rounds is not None:
            out["max_review_rounds"] = caps.max_review_rounds
        if caps.max_implement_turns is not None:
            out["implement_max_turns"] = caps.max_implement_turns
        if caps.max_attempts is not None:
            out["max_attempts"] = caps.max_attempts

    if "on_max_review_rounds" not in out:
        if tasks_file.version is None or tasks_file.version == 1:
            out["on_max_review_rounds"] = "auto_approve"

    return out


def load_persona_fragment(path: Path | str | None) -> str:
    """Read a persona markdown fragment.

    Returns the file's UTF-8 contents when the path is set, the file
    exists, is a regular file, and has at most 800 lines. Returns
    "" silently when the path is None/empty, the file is missing, or
    the path is a directory. Returns "" with a stdout warning when
    the file is over 800 lines (long personas dilute every prompt;
    the cap is protective). Per-fragment cap, not aggregate — the
    caller may compose multiple fragments via prompts.build_persona_section.

    Path resolution is CWD-relative when `path` is a relative string;
    absolute paths and Path objects pass through unchanged. (See plan
    §D8.)
    """
    if not path:
        return ""
    p = Path(path) if not isinstance(path, Path) else path
    if not p.exists() or not p.is_file():
        return ""
    text = p.read_text(encoding="utf-8")
    line_count = len(text.splitlines())
    if line_count > 800:
        print(yellow(
            f"Warning: persona file {p} has {line_count} lines (>800); "
            f"skipping. Long personas dilute every prompt — split into "
            f"'phases[].defaults.persona_extra' overlays or trim."
        ))
        return ""
    return text
