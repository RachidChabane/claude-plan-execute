"""tasks.yaml / tasks.json discovery and loading."""
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

TASK_FILE_SEARCH = ["tasks.yaml", "docs/tasks.yaml"]


def find_tasks_file() -> Path | None:
    for candidate in TASK_FILE_SEARCH:
        p = Path(candidate)
        if p.exists():
            return p
    return None


def load_tasks_file(path: Path) -> tuple[dict, list[dict]]:
    """Load and validate the tasks file. Returns (defaults, tasks)."""
    with open(path) as f:
        if path.suffix == ".json":
            data = json.load(f)
        else:
            data = yaml.safe_load(f)

    if not isinstance(data, dict) or "tasks" not in data:
        print(red(f"Error: {path} must contain a 'tasks' key"), file=sys.stderr)
        sys.exit(1)

    version = data.get("version")
    if version is None:
        print(yellow(f"Warning: {path} has no 'version' field. Assuming version 1."))

    file_defaults = data.get("defaults", {})
    tasks = data["tasks"]

    if not isinstance(tasks, list) or not tasks:
        print(red("Error: 'tasks' must be a non-empty list"), file=sys.stderr)
        sys.exit(1)

    for t in tasks:
        for required in ("id", "title", "description"):
            if required not in t:
                print(red(f"Error: Task missing required field '{required}': {t}"), file=sys.stderr)
                sys.exit(1)

    return file_defaults, tasks
