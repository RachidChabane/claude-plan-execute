"""Tests for claude_plan_execute.config.resolve()."""
from __future__ import annotations

from claude_plan_execute.config import HARDCODED_DEFAULTS, resolve


def test_cli_override_wins_over_all():
    out = resolve(
        "model",
        task={"model": "task"},
        file_defaults={"model": "file"},
        cli_overrides={"model": "cli"},
    )
    assert out == "cli"


def test_task_wins_over_file_and_hardcoded():
    out = resolve(
        "model",
        task={"model": "task"},
        file_defaults={"model": "file"},
        cli_overrides={},
    )
    assert out == "task"


def test_file_defaults_win_over_hardcoded():
    out = resolve(
        "model",
        task={},
        file_defaults={"model": "file"},
        cli_overrides={},
    )
    assert out == "file"


def test_falls_back_to_hardcoded_when_unset():
    assert resolve("model", {}, {}, {}) == HARDCODED_DEFAULTS["model"]
    assert resolve("effort", {}, {}, {}) == HARDCODED_DEFAULTS["effort"]
    assert resolve("plans_dir", {}, {}, {}) == HARDCODED_DEFAULTS["plans_dir"]


def test_unknown_key_returns_none():
    assert resolve("nonexistent_key", {}, {}, {}) is None


def test_none_value_at_cli_falls_through():
    # Critical: argparse defaults produce None for unset flags;
    # those must NOT win over task/file_defaults/hardcoded.
    out = resolve(
        "model",
        task={"model": "task"},
        file_defaults={},
        cli_overrides={"model": None},
    )
    assert out == "task"


def test_none_value_at_task_falls_through():
    out = resolve(
        "model",
        task={"model": None},
        file_defaults={"model": "file"},
        cli_overrides={},
    )
    assert out == "file"


def test_none_value_at_file_defaults_falls_through():
    out = resolve(
        "model",
        task={},
        file_defaults={"model": None},
        cli_overrides={},
    )
    assert out == HARDCODED_DEFAULTS["model"]


def test_all_hardcoded_keys_resolvable():
    # Smoke test: every documented hardcoded default is reachable.
    for key in HARDCODED_DEFAULTS:
        assert resolve(key, {}, {}, {}) == HARDCODED_DEFAULTS[key]


# ---------------------------------------------------------------------------
# Five-tier order coverage (task 4): cli > task > task_class > phase >
# project > hardcoded.
# ---------------------------------------------------------------------------


def test_cli_wins_over_task_and_phase_and_class():
    out = resolve(
        "model",
        task={"model": "task"},
        file_defaults={"model": "file"},
        cli_overrides={"model": "cli"},
        phase={"model": "phase"},
        task_class={"model": "class"},
    )
    assert out == "cli"


def test_task_wins_over_class_and_phase_and_project():
    out = resolve(
        "model",
        task={"model": "task"},
        file_defaults={"model": "file"},
        cli_overrides={},
        phase={"model": "phase"},
        task_class={"model": "class"},
    )
    assert out == "task"


def test_class_wins_over_phase_and_project():
    out = resolve(
        "model",
        task={},
        file_defaults={"model": "file"},
        cli_overrides={},
        phase={"model": "phase"},
        task_class={"model": "class"},
    )
    assert out == "class"


def test_phase_wins_over_project():
    out = resolve(
        "model",
        task={},
        file_defaults={"model": "file"},
        cli_overrides={},
        phase={"model": "phase"},
        task_class={},
    )
    assert out == "phase"


def test_phase_falls_through_when_key_absent():
    out = resolve(
        "model",
        task={},
        file_defaults={"model": "file"},
        cli_overrides={},
        phase={"effort": "max"},   # unrelated key
        task_class={},
    )
    assert out == "file"


def test_task_class_falls_through_when_key_absent():
    out = resolve(
        "model",
        task={},
        file_defaults={"model": "file"},
        cli_overrides={},
        phase=None,
        task_class={"effort": "max"},
    )
    assert out == "file"


def test_phase_none_is_no_op():
    """Passing phase=None is identical to omitting the kwarg."""
    out = resolve("model", {}, {"model": "file"}, {}, phase=None, task_class=None)
    assert out == "file"


def test_phase_empty_dict_is_no_op():
    out = resolve("model", {}, {"model": "file"}, {}, phase={}, task_class={})
    assert out == "file"


def test_phase_with_none_value_falls_through():
    out = resolve(
        "model",
        task={},
        file_defaults={"model": "file"},
        cli_overrides={},
        phase={"model": None},
        task_class={"model": None},
    )
    assert out == "file"


def test_max_attempts_is_a_hardcoded_default():
    assert HARDCODED_DEFAULTS["max_attempts"] == 3
    assert resolve("max_attempts", {}, {}, {}) == 3
