"""Tests for claude_plan_execute.schema and the version-aware loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from claude_plan_execute import loader
from claude_plan_execute.schema import (
    Caps,
    Defaults,
    Phase,
    Task,
    TaskClassConfig,
    TasksFile,
    ValidationError,
    format_validation_error,
)


FIXTURES = Path(__file__).parent / "fixtures"


# --- A. Round-trip on real fixtures -----------------------------------------

def test_v2_fixture_validates_clean(capsys):
    tf = loader.load_tasks_file(FIXTURES / "tasks-v2.yaml")
    assert isinstance(tf, TasksFile)
    assert tf.version == 2
    assert {p.id for p in tf.phases} == {"foundation", "ui-polish"}
    assert tf.defaults.caps and tf.defaults.caps.max_minutes_per_phase == 45
    assert "schema" in (tf.defaults.task_class_models or {})


def test_v1_fixture_validates_clean(capsys):
    tf = loader.load_tasks_file(FIXTURES / "tasks-v1.yaml")
    assert tf.version == 1
    # No phases, no task_class_models, no caps
    assert tf.phases == []
    assert tf.defaults.task_class_models is None
    assert tf.defaults.caps is None


def test_repo_dogfood_tasks_yaml_validates(capsys):
    """The repo's own tasks.yaml is the strongest end-to-end test of the schema."""
    repo_root = Path(__file__).resolve().parents[1]
    tf = loader.load_tasks_file(repo_root / "tasks.yaml")
    assert tf.version == 2
    assert len(tf.tasks) >= 4
    assert any(t.phase == "config" and t.id == 4 for t in tf.tasks)


# --- B. Deprecation warnings ------------------------------------------------

def test_v2_with_legacy_timeout_emits_deprecation_warning(tmp_path, capsys):
    p = tmp_path / "t.yaml"
    p.write_text(
        "version: 2\n"
        "defaults:\n"
        "  plan_timeout_minutes: 30\n"
        "  caps:\n"
        "    max_minutes_per_phase: 60\n"
        "tasks:\n"
        "  - { id: 1, title: T, description: D }\n"
    )
    loader.load_tasks_file(p)
    out = capsys.readouterr().out
    assert "deprecated" in out
    assert "plan_timeout_minutes" in out
    assert "caps.max_minutes_per_phase" in out


def test_v1_with_legacy_timeout_emits_NO_deprecation_warning(tmp_path, capsys):
    p = tmp_path / "t.yaml"
    p.write_text(
        "version: 1\n"
        "defaults:\n"
        "  plan_timeout_minutes: 30\n"
        "tasks:\n"
        "  - { id: 1, title: T, description: D }\n"
    )
    loader.load_tasks_file(p)
    out = capsys.readouterr().out
    assert "deprecated" not in out


def test_no_version_emits_assuming_v1_warning(tmp_path, capsys):
    p = tmp_path / "t.yaml"
    p.write_text("tasks:\n  - { id: 1, title: T, description: D }\n")
    loader.load_tasks_file(p)
    out = capsys.readouterr().out
    assert "no 'version' field" in out
    assert "Assuming version 1" in out


# --- C. Caps-vs-legacy precedence (S2) --------------------------------------

def test_effective_defaults_caps_wins_over_legacy_when_both_set(tmp_path, capsys):
    p = tmp_path / "t.yaml"
    p.write_text(
        "version: 2\n"
        "defaults:\n"
        "  plan_timeout_minutes: 30\n"
        "  caps:\n"
        "    max_minutes_per_phase: 60\n"
        "tasks:\n"
        "  - { id: 1, title: T, description: D }\n"
    )
    tf = loader.load_tasks_file(p)
    eff = loader.effective_defaults(tf)
    assert eff["plan_timeout_minutes"] == 60
    assert eff["review_timeout_minutes"] == 60
    assert eff["implement_timeout_minutes"] == 60


def test_effective_defaults_legacy_kept_when_no_caps(tmp_path, capsys):
    p = tmp_path / "t.yaml"
    p.write_text(
        "version: 1\n"
        "defaults:\n"
        "  plan_timeout_minutes: 30\n"
        "  review_timeout_minutes: 25\n"
        "tasks:\n"
        "  - { id: 1, title: T, description: D }\n"
    )
    tf = loader.load_tasks_file(p)
    eff = loader.effective_defaults(tf)
    assert eff["plan_timeout_minutes"] == 30
    assert eff["review_timeout_minutes"] == 25


def test_effective_defaults_drops_caps_and_task_class_models(tmp_path, capsys):
    tf = loader.load_tasks_file(FIXTURES / "tasks-v2.yaml")
    eff = loader.effective_defaults(tf)
    assert "caps" not in eff
    assert "task_class_models" not in eff


def test_effective_defaults_expands_caps_max_review_rounds(tmp_path, capsys):
    tf = loader.load_tasks_file(FIXTURES / "tasks-v2.yaml")
    eff = loader.effective_defaults(tf)
    assert eff["max_review_rounds"] == 3
    assert eff["implement_max_turns"] == 200
    assert eff["max_attempts"] == 3


# --- D. Strict validation: extra keys -> typo suggestion -------------------

def test_extra_key_under_defaults_suggests_correction(tmp_path, capsys):
    p = tmp_path / "t.yaml"
    p.write_text(
        "version: 2\n"
        "defaults:\n"
        "  task_clas: refactor\n"   # typo
        "tasks:\n"
        "  - { id: 1, title: T, description: D }\n"
    )
    with pytest.raises(SystemExit) as exc:
        loader.load_tasks_file(p)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "task_clas" in err
    assert "Did you mean 'task_class_models'?" in err or "Did you mean" in err


def test_extra_key_under_caps_is_reported(tmp_path, capsys):
    p = tmp_path / "t.yaml"
    p.write_text(
        "version: 2\n"
        "defaults:\n"
        "  caps:\n"
        "    max_review_round: 3\n"   # typo of max_review_rounds
        "tasks:\n"
        "  - { id: 1, title: T, description: D }\n"
    )
    with pytest.raises(SystemExit):
        loader.load_tasks_file(p)
    err = capsys.readouterr().err
    assert "max_review_round" in err
    assert "max_review_rounds" in err  # appears in the suggestion


def test_extra_key_under_task_class_models_value_is_reported(tmp_path, capsys):
    p = tmp_path / "t.yaml"
    p.write_text(
        "version: 2\n"
        "defaults:\n"
        "  task_class_models:\n"
        "    schema: { modle: opus }\n"   # typo of model
        "tasks:\n"
        "  - { id: 1, title: T, description: D }\n"
    )
    with pytest.raises(SystemExit):
        loader.load_tasks_file(p)
    err = capsys.readouterr().err
    assert "modle" in err
    assert "model" in err


def test_top_level_extra_key_is_rejected(tmp_path, capsys):
    p = tmp_path / "t.yaml"
    p.write_text(
        "version: 2\n"
        "tasksz:\n"   # garbage top-level key
        "  - foo\n"
        "tasks:\n"
        "  - { id: 1, title: T, description: D }\n"
    )
    with pytest.raises(SystemExit):
        loader.load_tasks_file(p)
    err = capsys.readouterr().err
    assert "tasksz" in err


# --- E. Cross-reference validation (S3) -------------------------------------

def test_task_phase_must_exist(tmp_path, capsys):
    p = tmp_path / "t.yaml"
    p.write_text(
        "version: 2\n"
        "phases:\n"
        "  - id: foundation\n"
        "tasks:\n"
        "  - { id: 1, title: T, description: D, phase: foundatio }\n"  # typo
    )
    with pytest.raises(SystemExit):
        loader.load_tasks_file(p)
    err = capsys.readouterr().err
    assert "unknown phase 'foundatio'" in err
    assert "foundation" in err  # listed as known


def test_task_class_must_exist_when_class_map_present(tmp_path, capsys):
    p = tmp_path / "t.yaml"
    p.write_text(
        "version: 2\n"
        "defaults:\n"
        "  task_class_models:\n"
        "    refactor: { model: opus }\n"
        "tasks:\n"
        "  - { id: 1, title: T, description: D, task_class: refacto }\n"  # typo
    )
    with pytest.raises(SystemExit):
        loader.load_tasks_file(p)
    err = capsys.readouterr().err
    assert "unknown task_class 'refacto'" in err
    assert "refactor" in err


def test_task_class_check_skipped_when_class_map_empty(tmp_path, capsys):
    """When no task_class_models is defined, an arbitrary task_class string
    on a task is accepted (no class registry to validate against)."""
    p = tmp_path / "t.yaml"
    p.write_text(
        "version: 2\n"
        "tasks:\n"
        "  - { id: 1, title: T, description: D, task_class: anything }\n"
    )
    tf = loader.load_tasks_file(p)
    assert tf.tasks[0].task_class == "anything"


def test_phase_default_task_class_must_exist(tmp_path, capsys):
    p = tmp_path / "t.yaml"
    p.write_text(
        "version: 2\n"
        "defaults:\n"
        "  task_class_models:\n"
        "    refactor: { model: opus }\n"
        "phases:\n"
        "  - id: split\n"
        "    defaults: { task_class: refacto }\n"  # typo in phase defaults
        "tasks:\n"
        "  - { id: 1, title: T, description: D, phase: split }\n"
    )
    with pytest.raises(SystemExit):
        loader.load_tasks_file(p)
    err = capsys.readouterr().err
    assert "Phase 'split'" in err
    assert "unknown task_class 'refacto'" in err


# --- F. Required field / shape errors --------------------------------------

def test_task_missing_required_field_is_reported(tmp_path, capsys):
    p = tmp_path / "t.yaml"
    p.write_text(
        "version: 2\n"
        "tasks:\n"
        "  - { id: 1, title: T }\n"  # missing description
    )
    with pytest.raises(SystemExit):
        loader.load_tasks_file(p)
    err = capsys.readouterr().err
    assert "description" in err
    assert "missing" in err.lower()


def test_empty_tasks_list_is_rejected(tmp_path, capsys):
    p = tmp_path / "t.yaml"
    p.write_text("version: 2\ntasks: []\n")
    with pytest.raises(SystemExit):
        loader.load_tasks_file(p)
    err = capsys.readouterr().err
    assert "tasks" in err


def test_no_tasks_key_is_rejected_with_helpful_message(tmp_path, capsys):
    p = tmp_path / "t.yaml"
    p.write_text("version: 2\n")
    with pytest.raises(SystemExit):
        loader.load_tasks_file(p)
    err = capsys.readouterr().err
    assert "tasks" in err


# --- G. format_validation_error helper ------------------------------------

def test_format_validation_error_includes_file_path():
    from pydantic import ValidationError as PydValidationError
    try:
        TasksFile.model_validate({"tasks": []})  # too_short
    except PydValidationError as exc:
        out = format_validation_error(exc, Path("/tmp/example.yaml"))
        assert "/tmp/example.yaml" in out
