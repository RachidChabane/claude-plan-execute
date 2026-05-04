"""Tests for claude_plan_execute.cli — argparse subcommand handlers.

T10 (v0.3 §2.4) — `approve` / `reject` subcommands write a
human-gate sentinel directly without a server round-trip, so they
work even when the dashboard is down. Note is tail-capped at 4 KB
to mirror the server cap.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from claude_plan_execute.cli import (
    _resolve_plans_dir,
    _write_human_gate_sentinel,
    cmd_approve,
    cmd_reject,
)


def _make_args(
    *,
    task_id: str,
    gate_name: str,
    note: str = "",
    tasks: str | None = None,
    dir_: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        task_id=task_id,
        gate_name=gate_name,
        note=note,
        tasks=tasks,
        dir=dir_,
    )


def _write_tasks_yaml(work_dir: Path, plans_dir: str | None = None) -> Path:
    plans = f"\n  plans_dir: {plans_dir}" if plans_dir else ""
    body = f"""\
version: 1
defaults:{plans}
  test_command: "true"
tasks:
  - id: 1
    title: T
    description: D
"""
    p = work_dir / "tasks.yaml"
    p.write_text(body)
    return p


def test_cli_approve_writes_approval_sentinel_with_note(tmp_path, monkeypatch):
    """T10 §D15 — approve subcommand writes
    `<plans_dir>/task-{tid}/gates/{gate_name}.approval` with the note.
    """
    monkeypatch.chdir(tmp_path)
    _write_tasks_yaml(tmp_path)
    args = _make_args(task_id="1", gate_name="review", note="lgtm")
    code = cmd_approve(args)
    assert code == 0
    sentinel = tmp_path / "plans" / "task-1" / "gates" / "review.approval"
    assert sentinel.exists()
    assert sentinel.read_text() == "lgtm"


def test_cli_reject_writes_rejection_sentinel_with_note(tmp_path, monkeypatch):
    """T10 §D15 — reject subcommand writes the rejection sentinel."""
    monkeypatch.chdir(tmp_path)
    _write_tasks_yaml(tmp_path)
    args = _make_args(task_id="1", gate_name="review", note="no go")
    code = cmd_reject(args)
    assert code == 0
    sentinel = tmp_path / "plans" / "task-1" / "gates" / "review.rejection"
    assert sentinel.exists()
    assert sentinel.read_text() == "no go"


def test_cli_approve_uses_default_plans_dir_when_no_tasks_file(
    tmp_path, monkeypatch,
):
    """T10 §D15 — when no tasks file is found, the resolver falls
    back to "plans".
    """
    monkeypatch.chdir(tmp_path)
    # No tasks.yaml in tmp_path.
    args = _make_args(task_id="1", gate_name="review", note="ok")
    code = cmd_approve(args)
    assert code == 0
    sentinel = tmp_path / "plans" / "task-1" / "gates" / "review.approval"
    assert sentinel.exists()


def test_cli_approve_uses_custom_plans_dir_from_tasks_file(
    tmp_path, monkeypatch,
):
    """T10 §D15 — `defaults.plans_dir: artifacts` flows through to
    the sentinel write path.
    """
    monkeypatch.chdir(tmp_path)
    _write_tasks_yaml(tmp_path, plans_dir="artifacts")
    args = _make_args(task_id="1", gate_name="review", note="ok")
    code = cmd_approve(args)
    assert code == 0
    sentinel = (
        tmp_path / "artifacts" / "task-1" / "gates" / "review.approval"
    )
    assert sentinel.exists()


def test_cli_approve_caps_note_at_4kb(tmp_path, monkeypatch):
    """T10 §D15 — the note is tail-capped at 4 KB UTF-8 bytes
    client-side to mirror the server cap.
    """
    monkeypatch.chdir(tmp_path)
    _write_tasks_yaml(tmp_path)
    big_note = "x" * (10 * 1024)  # 10 KB; cap is 4 KB
    args = _make_args(task_id="1", gate_name="review", note=big_note)
    code = cmd_approve(args)
    assert code == 0
    sentinel = tmp_path / "plans" / "task-1" / "gates" / "review.approval"
    assert len(sentinel.read_text().encode("utf-8")) == 4096


def test_cli_approve_creates_gates_dir_if_missing(tmp_path, monkeypatch):
    """T10 §D15 — gates_dir is created on demand."""
    monkeypatch.chdir(tmp_path)
    _write_tasks_yaml(tmp_path)
    # Confirm the gates dir doesn't exist yet.
    assert not (tmp_path / "plans" / "task-1" / "gates").exists()
    args = _make_args(task_id="1", gate_name="review", note="ok")
    code = cmd_approve(args)
    assert code == 0
    assert (tmp_path / "plans" / "task-1" / "gates").is_dir()


def test_cli_approve_writes_to_convention_path(tmp_path, monkeypatch):
    """T10 §D8 — the CLI computes the sentinel path from the
    convention `<plans_dir>/task-<tid>/gates/<name>.approval` and
    ignores any custom `Gate.sentinel_path` set in invariants.yaml.
    Operators who set a non-default sentinel_path must write to the
    configured path manually. This test pins the limitation: even
    if a hypothetical custom path were registered, the CLI uses the
    convention path.
    """
    monkeypatch.chdir(tmp_path)
    _write_tasks_yaml(tmp_path)
    args = _make_args(task_id="1", gate_name="design-review", note="ok")
    code = cmd_approve(args)
    assert code == 0
    convention = (
        tmp_path / "plans" / "task-1" / "gates" / "design-review.approval"
    )
    assert convention.exists()


def test_cli_approve_dir_flag_changes_working_directory(
    tmp_path, monkeypatch,
):
    """T10 §D15 — `--dir <path>` causes the handler to chdir before
    resolving plans_dir.
    """
    project = tmp_path / "subproject"
    project.mkdir()
    _write_tasks_yaml(project)
    monkeypatch.chdir(tmp_path)
    args = _make_args(
        task_id="1", gate_name="review", note="ok", dir_=str(project),
    )
    code = cmd_approve(args)
    assert code == 0
    sentinel = project / "plans" / "task-1" / "gates" / "review.approval"
    assert sentinel.exists()


def test_cli_resolve_plans_dir_falls_back_to_plans_when_no_tasks_file(
    tmp_path, monkeypatch,
):
    """Pin the resolver fallback chain — see plan §D15."""
    monkeypatch.chdir(tmp_path)
    assert _resolve_plans_dir(None) == "plans"


def test_cli_resolve_plans_dir_uses_explicit_tasks_path(
    tmp_path, monkeypatch,
):
    """Pin the `--tasks` override path of the resolver."""
    monkeypatch.chdir(tmp_path)
    p = _write_tasks_yaml(tmp_path, plans_dir="artifacts")
    assert _resolve_plans_dir(str(p)) == "artifacts"


def test_cli_write_sentinel_returns_1_on_filesystem_error(
    tmp_path, monkeypatch,
):
    """Pin: filesystem errors return 1 (not raise)."""
    monkeypatch.chdir(tmp_path)
    _write_tasks_yaml(tmp_path)

    def boom(self, *a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", boom)
    args = _make_args(task_id="1", gate_name="review", note="ok")
    code = _write_human_gate_sentinel(args, "approval")
    assert code == 1
