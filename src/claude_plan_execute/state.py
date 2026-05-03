"""Persistent task state with atomic writes."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# State file format version. Bumped to 2 in task 9 to add per-task
# `attempts: list[FailureRecord]` and `current_attempt: int`. Schema-1
# files (no `schema_version` key) self-heal on load: `State.__init__`
# injects defaults; subsequent writes emit v2.
SCHEMA_VERSION = 2

# Verbatim-content byte cap for FailureRecord.verbatim. The truncation
# marker (~30-50 bytes) is appended on top; final string is bounded by
# 4 KB + marker.
_VERBATIM_CAP_BYTES = 4096

STATUSES = (
    "pending", "planning", "reviewing", "revising",
    "approved", "implementing", "done", "blocked",
)

# Documented set of failure_class strings used across the project. Not
# enforced by `make_failure_record` (forward-compat + ergonomics around
# custom values). Tests pin the membership.
FAILURE_CLASSES = (
    "review_rejected_after_max_rounds",   # task 9 — review path + on_max_review_rounds=fail
    "implementer_timeout",                # task 9 — Phase 3 subprocess.TimeoutExpired
    "implementer_max_turns",              # task 9 — Phase 3 subtype=="error_max_turns"
    "manual_stop",                        # task 9 — handle_sigint
    "gate_failed",                        # task 12 — gate registry (shape only in v1)
    "plan_not_written",                   # NOT written in task 9 (Non-goals)
    "test_failed",                        # subtype of gate_failed; task 12+
    "budget_exceeded",                    # NOT written in task 9 (max_attempts deferred)
)


def make_failure_record(
    failure_class: str,
    phase: str,
    *,
    reason: str,
    verbatim: str = "",
    artifacts: dict[str, str] | None = None,
    hypothesis: str | None = None,
) -> dict:
    """Construct a FailureRecord dict.

    Pure: no I/O, no side effects, no validation against
    `FAILURE_CLASSES`. The `verbatim` content is truncated to 4 KB
    UTF-8 bytes; a `[truncated: <N> bytes omitted]` marker is appended
    when truncation happens. Timestamp is UTC ISO-8601.
    """
    encoded = verbatim.encode("utf-8")
    if len(encoded) > _VERBATIM_CAP_BYTES:
        truncated = encoded[:_VERBATIM_CAP_BYTES].decode(
            "utf-8", errors="ignore",
        )
        omitted = len(encoded) - _VERBATIM_CAP_BYTES
        verbatim_capped = (
            truncated + f"\n[truncated: {omitted} bytes omitted]"
        )
    else:
        verbatim_capped = verbatim
    return {
        "failure_class": failure_class,
        "phase": phase,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "verbatim": verbatim_capped,
        "artifacts": dict(artifacts) if artifacts else {},
        "hypothesis": hypothesis,
    }


class State:
    """Persistent state tracker with atomic writes."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {
            "schema_version": SCHEMA_VERSION,
            "tasks": {},
        }
        if path.exists():
            with open(path) as f:
                self.data = json.load(f)
            # Self-heal to schema 2: idempotent on schema-2 files,
            # injects defaults on schema-1 files. The isinstance guard
            # protects against corrupted entries (e.g., bare strings)
            # so the load itself doesn't raise.
            self.data["schema_version"] = SCHEMA_VERSION
            for entry in self.data.setdefault("tasks", {}).values():
                if isinstance(entry, dict):
                    entry.setdefault("attempts", [])
                    entry.setdefault("current_attempt", 1)

    def get(self, task_id: int | str) -> dict:
        key = str(task_id)
        if key not in self.data["tasks"]:
            self.data["tasks"][key] = {
                "status": "pending",
                "attempts": [],
                "current_attempt": 1,
            }
        return self.data["tasks"][key]

    def set_status(self, task_id: int | str, status: str, **extra):
        entry = self.get(task_id)
        entry["status"] = status
        entry.update(extra)
        self.save()

    def record_failure(
        self, task_id: int | str, failure_record: dict,
    ) -> None:
        """Append a FailureRecord to the per-task `attempts` list and save.

        Does NOT modify status, current_attempt, or any other field.
        The caller is responsible for those (typically a sibling
        `set_status(tid, "blocked", block_reason=...)` call).
        """
        entry = self.get(task_id)
        entry.setdefault("attempts", []).append(failure_record)
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
            self.data["tasks"][str(task_id)] = {
                "status": "pending",
                "attempts": [],
                "current_attempt": 1,
            }
        else:
            for k in self.data["tasks"]:
                self.data["tasks"][k] = {
                    "status": "pending",
                    "attempts": [],
                    "current_attempt": 1,
                }
        self.save()
