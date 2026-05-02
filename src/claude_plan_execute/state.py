"""Persistent task state with atomic writes."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

STATUSES = (
    "pending", "planning", "reviewing", "revising",
    "approved", "implementing", "done", "blocked",
)


class State:
    """Persistent state tracker with atomic writes."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {"tasks": {}}
        if path.exists():
            with open(path) as f:
                self.data = json.load(f)

    def get(self, task_id: int | str) -> dict:
        key = str(task_id)
        if key not in self.data["tasks"]:
            self.data["tasks"][key] = {"status": "pending"}
        return self.data["tasks"][key]

    def set_status(self, task_id: int | str, status: str, **extra):
        entry = self.get(task_id)
        entry["status"] = status
        entry.update(extra)
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.data, f, indent=2)
            os.rename(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def reset(self, task_id: int | str | None = None):
        if task_id is not None:
            self.data["tasks"][str(task_id)] = {"status": "pending"}
        else:
            for k in self.data["tasks"]:
                self.data["tasks"][k] = {"status": "pending"}
        self.save()
