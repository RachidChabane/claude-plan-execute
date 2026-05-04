"""Persistent task state with atomic writes."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

# State file format version. Bumped to 2 in task 9 to add per-task
# `attempts: list[FailureRecord]` and `current_attempt: int`. Bumped
# to 3 in task 2 of v0.3 (§2.7) to add per-task
# `usage_summary: dict | None`. Schema-N files self-heal on load:
# `State.__init__` overwrites `schema_version` to current
# `SCHEMA_VERSION`; missing optional fields stay missing (callers use
# `.get(...)`).
SCHEMA_VERSION = 3

# Verbatim-content byte cap for FailureRecord.verbatim. The truncation
# marker (~30-50 bytes) is appended on top; final string is bounded by
# 4 KB + marker.
_VERBATIM_CAP_BYTES = 4096

STATUSES = (
    "pending", "planning", "reviewing", "revising",
    "approved", "implementing", "done", "blocked",
    "awaiting_human",
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
    "human_gate_timeout",                 # v0.3 task 10 — human gate timeout (§2.4)
    "budget_exceeded",                    # NOT written in task 9 (max_attempts deferred)
    "rate_limited",                       # v0.3 task 1 — Claude-API rate-limit retries exhausted (§2.15)
    "service_unavailable",                # v0.3 task 8 — ServiceUnavailableError
    "service_crashed",                    # v0.3 task 8 — single crash auto-recovered
    "service_unstable",                   # v0.3 task 8 — 3 consecutive crashes
    # T9 (v0.3 §2.12) — branch / PR side-effect failures.
    "branch_checkout_failed",             # v0.3 task 9 — Pre-Phase-1 git checkout / fetch failure
    "push_failed",                        # v0.3 task 9 — Post-Phase-4 git push -u origin <branch>
    "pr_create_failed",                   # v0.3 task 9 — Post-Phase-4 gh pr create / list failure
    "merge_failed",                       # v0.3 task 9 — Post-Phase-4 gh pr merge --auto --squash failure
    "worker_exception",                   # v0.3 task 13 — uncaught exception in slate worker loop
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


def summarize_usage(jsonl_path: Path) -> dict:
    """Sum a `usage.jsonl` file into a per-task rollup dict.

    See plan task-2 §D9. Pure: no side effects, no I/O beyond the file
    read.

    Returns a dict with six keys, even when the file is missing
    (all-zero rollup, `total_runs == 0` — caller uses this as the
    absence-robust signal to skip writing `usage_summary` to
    state.json).
    """
    summary: dict = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cache_read": 0,
        "total_cache_create": 0,
        "total_runs": 0,
        "duration_seconds": 0.0,
    }
    if not jsonl_path.exists():
        return summary
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                summary["total_input_tokens"] += int(row.get("input", 0))
                summary["total_output_tokens"] += int(row.get("output", 0))
                summary["total_cache_read"] += int(row.get("cache_read", 0))
                summary["total_cache_create"] += int(row.get("cache_create", 0))
                summary["duration_seconds"] += float(row.get("duration_seconds", 0))
                summary["total_runs"] += 1
    except OSError:
        pass
    return summary


class State:
    """Persistent state tracker with atomic writes."""

    def __init__(self, path: Path):
        self._lock = threading.RLock()
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
        with self._lock:
            key = str(task_id)
            if key not in self.data["tasks"]:
                self.data["tasks"][key] = {
                    "status": "pending",
                    "attempts": [],
                    "current_attempt": 1,
                }
            return self.data["tasks"][key]

    def set_status(self, task_id: int | str, status: str, **extra):
        with self._lock:
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
        with self._lock:
            entry = self.get(task_id)
            entry.setdefault("attempts", []).append(failure_record)
            self.save()

    def save(self):
        with self._lock:
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
        with self._lock:
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
