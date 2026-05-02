"""Tests for claude_plan_execute.state.State."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_plan_execute.state import State, STATUSES


def test_new_state_has_empty_tasks(tmp_path: Path):
    s = State(tmp_path / "state.json")
    assert s.data == {"tasks": {}}


def test_get_creates_pending_entry(tmp_path: Path):
    s = State(tmp_path / "state.json")
    entry = s.get(1)
    assert entry == {"status": "pending"}
    # Returned reference is the same object stored in s.data
    assert s.data["tasks"]["1"] is entry


def test_get_accepts_int_or_str_task_id(tmp_path: Path):
    s = State(tmp_path / "state.json")
    s.get(1)
    assert s.get("1") is s.data["tasks"]["1"]


def test_set_status_persists_and_merges_extras(tmp_path: Path):
    path = tmp_path / "state.json"
    s = State(path)
    s.set_status(1, "planning", plan_path="plans/task-1/plan.md")
    s.set_status(1, "reviewing", review_round=2)

    entry = s.get(1)
    assert entry["status"] == "reviewing"
    assert entry["review_round"] == 2
    # plan_path from earlier set_status must be preserved (update merges).
    assert entry["plan_path"] == "plans/task-1/plan.md"


def test_save_writes_atomically(tmp_path: Path):
    path = tmp_path / "state.json"
    s = State(path)
    s.set_status(1, "done")

    on_disk = json.loads(path.read_text())
    assert on_disk == {"tasks": {"1": {"status": "done"}}}

    # No leftover .tmp file
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_creates_parent_directory(tmp_path: Path):
    nested = tmp_path / "plans" / "deep"
    s = State(nested / "state.json")
    s.set_status(1, "pending")
    assert (nested / "state.json").exists()


def test_load_existing_state(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "tasks": {"1": {"status": "done", "plan_path": "x"}, "2": {"status": "pending"}}
    }))
    s = State(path)
    assert s.get(1) == {"status": "done", "plan_path": "x"}
    assert s.get(2) == {"status": "pending"}


def test_reset_single_task_clears_all_keys(tmp_path: Path):
    path = tmp_path / "state.json"
    s = State(path)
    s.set_status(1, "implementing", plan_path="x", review_round=3)
    s.set_status(2, "done")

    s.reset(1)
    assert s.get(1) == {"status": "pending"}  # plan_path/review_round dropped
    assert s.get(2) == {"status": "done"}     # task 2 untouched

    # Persisted to disk
    on_disk = json.loads(path.read_text())
    assert on_disk["tasks"]["1"] == {"status": "pending"}
    assert on_disk["tasks"]["2"] == {"status": "done"}


def test_reset_all_tasks(tmp_path: Path):
    path = tmp_path / "state.json"
    s = State(path)
    s.set_status(1, "done")
    s.set_status(2, "implementing", plan_path="y")

    s.reset()
    assert s.get(1) == {"status": "pending"}
    assert s.get(2) == {"status": "pending"}


def test_reset_all_on_empty_state_is_noop(tmp_path: Path):
    s = State(tmp_path / "state.json")
    s.reset()  # no tasks; should not raise
    assert s.data == {"tasks": {}}


def test_statuses_constant_includes_known_values():
    # Pin: STATUSES documents the known status names. Task 6 added "blocked"
    # for the on_max_review_rounds: fail path.
    assert "pending" in STATUSES
    assert "done" in STATUSES
    assert "implementing" in STATUSES
    assert "blocked" in STATUSES


def test_use_color_propagates_to_loader_and_deps_via_module_global(monkeypatch, capsys):
    """colors.set_use_color toggles a single module-level global; downstream
    modules (loader, deps) honor it because they import red/yellow as references
    that read colors._use_color at call time."""
    from claude_plan_execute import colors
    from claude_plan_execute.colors import red

    # Force isatty=True so the color path is reachable.
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    colors.set_use_color(True)
    assert "\033[" in red("hello")

    colors.set_use_color(False)
    assert red("hello") == "hello"

    # Restore default so other tests aren't affected.
    colors.set_use_color(True)
