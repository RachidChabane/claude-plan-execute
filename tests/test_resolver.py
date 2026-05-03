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


# ---------------------------------------------------------------------------
# Six-tier order with provenance (task 5):
#   resolve_task_config returns {key: {"value": v, "from": <layer>}} for
#   every KEYS_TO_AUDIT entry. _resolve_with_source labels match plan §D2.
# ---------------------------------------------------------------------------

from claude_plan_execute.config import (
    KEYS_TO_AUDIT,
    _resolve_with_source,
    resolve_task_config,
)


# --- A. _resolve_with_source: per-layer source labels ------------------------

def test_resolve_with_source_cli_label():
    v, src = _resolve_with_source(
        "model", task={"model": "task"}, file_defaults={}, cli_overrides={"model": "cli"},
    )
    assert v == "cli"
    assert src == "cli"


def test_resolve_with_source_task_label():
    v, src = _resolve_with_source(
        "model", task={"model": "task"}, file_defaults={}, cli_overrides={},
    )
    assert v == "task"
    assert src == "task"


def test_resolve_with_source_task_class_unqualified_when_name_absent():
    v, src = _resolve_with_source(
        "model", task={}, file_defaults={}, cli_overrides={},
        task_class={"model": "class-val"},
    )
    assert v == "class-val"
    assert src == "task_class"


def test_resolve_with_source_task_class_qualified_with_name():
    v, src = _resolve_with_source(
        "model", task={}, file_defaults={}, cli_overrides={},
        task_class={"model": "class-val"}, task_class_name="schema",
    )
    assert v == "class-val"
    assert src == "task_class:schema"


def test_resolve_with_source_phase_unqualified_when_id_absent():
    v, src = _resolve_with_source(
        "model", task={}, file_defaults={}, cli_overrides={},
        phase={"model": "phase-val"},
    )
    assert src == "phase"


def test_resolve_with_source_phase_qualified_with_id():
    v, src = _resolve_with_source(
        "model", task={}, file_defaults={}, cli_overrides={},
        phase={"model": "phase-val"}, phase_id="foundation",
    )
    assert src == "phase:foundation"


def test_resolve_with_source_project_defaults_label():
    v, src = _resolve_with_source(
        "model", task={}, file_defaults={"model": "file"}, cli_overrides={},
    )
    assert v == "file"
    assert src == "project_defaults"


def test_resolve_with_source_hardcoded_label():
    v, src = _resolve_with_source(
        "model", task={}, file_defaults={}, cli_overrides={},
    )
    assert v == HARDCODED_DEFAULTS["model"]
    assert src == "hardcoded"


def test_resolve_with_source_unset_label_for_unknown_key():
    v, src = _resolve_with_source(
        "no_such_key", task={}, file_defaults={}, cli_overrides={},
    )
    assert v is None
    assert src == "unset"


def test_resolve_with_source_falls_through_None_at_higher_layers():
    v, src = _resolve_with_source(
        "model",
        task={"model": None}, file_defaults={"model": "file"}, cli_overrides={"model": None},
    )
    # `None` at cli + task falls through to file_defaults
    assert v == "file"
    assert src == "project_defaults"


# --- B. resolve() is identity over _resolve_with_source's value field --------

def test_resolve_is_value_of_resolve_with_source():
    """Pin: resolve() must remain a thin wrapper. If this fails, the two
    code paths have drifted (plan §D1)."""
    cases = [
        # (task, file_defaults, cli_overrides, phase, task_class)
        ({}, {}, {}, None, None),                                                # hardcoded
        ({}, {"model": "file"}, {}, None, None),                                  # file
        ({"model": "task"}, {"model": "file"}, {}, None, None),                   # task wins
        ({}, {}, {"model": "cli"}, None, None),                                   # cli
        ({}, {}, {}, {"model": "phase"}, None),                                   # phase
        ({}, {}, {}, None, {"model": "class"}),                                   # class
        ({"model": None}, {"model": "file"}, {}, None, None),                     # None falls through
        ({"model": None}, {}, {}, {"model": None}, {"model": None}),              # all None
    ]
    for task, file_defaults, cli, phase, task_class in cases:
        wrapped = resolve(
            "model", task, file_defaults, cli, phase=phase, task_class=task_class,
        )
        raw_v, _ = _resolve_with_source(
            "model", task, file_defaults, cli, phase=phase, task_class=task_class,
        )
        assert wrapped == raw_v, (
            f"resolve drifted from _resolve_with_source: "
            f"resolve()={wrapped} vs _resolve_with_source()={raw_v} "
            f"for task={task} file={file_defaults} cli={cli} "
            f"phase={phase} class={task_class}"
        )


# --- C. resolve_task_config shape and provenance ----------------------------

def test_resolve_task_config_returns_dict_keyed_by_audit_keys():
    out = resolve_task_config(task={}, file_defaults={}, cli_overrides={})
    assert set(out.keys()) == set(KEYS_TO_AUDIT)


def test_resolve_task_config_each_entry_has_value_and_from():
    out = resolve_task_config(task={}, file_defaults={}, cli_overrides={})
    for key, entry in out.items():
        assert set(entry.keys()) == {"value", "from"}, key


def test_resolve_task_config_picks_cli_layer():
    out = resolve_task_config(
        task={"model": "task"}, file_defaults={"model": "file"},
        cli_overrides={"model": "cli"},
    )
    assert out["model"] == {"value": "cli", "from": "cli"}


def test_resolve_task_config_picks_task_class_with_name():
    out = resolve_task_config(
        task={}, file_defaults={"model": "file"}, cli_overrides={},
        task_class={"model": "class-val"}, task_class_name="schema",
    )
    assert out["model"] == {"value": "class-val", "from": "task_class:schema"}


def test_resolve_task_config_picks_phase_with_id():
    out = resolve_task_config(
        task={}, file_defaults={"model": "file"}, cli_overrides={},
        phase={"model": "phase-val"}, phase_id="foundation",
    )
    assert out["model"] == {"value": "phase-val", "from": "phase:foundation"}


def test_resolve_task_config_falls_back_to_hardcoded():
    out = resolve_task_config(task={}, file_defaults={}, cli_overrides={})
    assert out["model"] == {"value": "opus", "from": "hardcoded"}
    assert out["plans_dir"] == {"value": "plans", "from": "hardcoded"}


def test_resolve_task_config_serializes_to_json_round_trip(tmp_path):
    """The whole point of the dict shape is direct JSON round-trip."""
    import json
    out = resolve_task_config(
        task={"model": "task"},
        file_defaults={"plans_dir": "myplans"},
        cli_overrides={},
    )
    s = json.dumps(out)
    back = json.loads(s)
    assert back == out
    assert back["model"]["value"] == "task"
    assert back["model"]["from"] == "task"
    assert back["plans_dir"]["value"] == "myplans"
    assert back["plans_dir"]["from"] == "project_defaults"


def test_resolve_task_config_audit_keys_match_doc_constant():
    """Pin: KEYS_TO_AUDIT is the public surface the dashboard renders.
    Adding a key without updating tests should fail loudly."""
    expected = (
        "model",
        "effort",
        "test_command",
        "lint_command",
        "plans_dir",
        "max_review_rounds",
        "plan_timeout_minutes",
        "review_timeout_minutes",
        "implement_timeout_minutes",
        "implement_max_turns",
        "max_attempts",
        "on_max_review_rounds",
        "persona_file",
        "decisions_dir",
        "invariants_file",
        "untranslated_strings_command",
    )
    assert KEYS_TO_AUDIT == expected


# ---------------------------------------------------------------------------
# I. on_max_review_rounds default + provenance (task 6)
# ---------------------------------------------------------------------------


def test_hardcoded_default_for_on_max_review_rounds_is_fail():
    """Task 6 §D1: modern hardcoded default is 'fail'. v1 compat is
    layered in by the loader, not by the resolver."""
    assert HARDCODED_DEFAULTS["on_max_review_rounds"] == "fail"


def test_resolve_returns_fail_when_no_layer_supplies_on_max_review_rounds():
    """No file_defaults injection (simulates v2 + no override): resolver
    falls through to hardcoded → 'fail'."""
    out = resolve("on_max_review_rounds", task={}, file_defaults={}, cli_overrides={})
    assert out == "fail"


def test_resolve_returns_auto_approve_when_file_defaults_supplies_it():
    """Simulates the v1 compat path: loader has injected 'auto_approve'
    into project_defaults; resolver picks it up there."""
    out = resolve(
        "on_max_review_rounds",
        task={},
        file_defaults={"on_max_review_rounds": "auto_approve"},
        cli_overrides={},
    )
    assert out == "auto_approve"


def test_resolve_task_config_includes_on_max_review_rounds_with_provenance_hardcoded():
    """v2 + no override path: from = 'hardcoded'."""
    out = resolve_task_config(task={}, file_defaults={}, cli_overrides={})
    assert out["on_max_review_rounds"] == {"value": "fail", "from": "hardcoded"}


def test_resolve_task_config_includes_on_max_review_rounds_with_provenance_project():
    """v1 compat path: from = 'project_defaults'."""
    out = resolve_task_config(
        task={},
        file_defaults={"on_max_review_rounds": "auto_approve"},
        cli_overrides={},
    )
    assert out["on_max_review_rounds"] == {
        "value": "auto_approve", "from": "project_defaults",
    }


def test_resolve_task_config_on_max_review_rounds_picks_phase_layer():
    out = resolve_task_config(
        task={},
        file_defaults={},
        cli_overrides={},
        phase={"on_max_review_rounds": "auto_approve"},
        phase_id="dashboard",
    )
    assert out["on_max_review_rounds"] == {
        "value": "auto_approve", "from": "phase:dashboard",
    }


def test_resolve_task_config_on_max_review_rounds_cli_wins():
    out = resolve_task_config(
        task={},
        file_defaults={"on_max_review_rounds": "auto_approve"},
        cli_overrides={"on_max_review_rounds": "fail"},
    )
    assert out["on_max_review_rounds"] == {"value": "fail", "from": "cli"}


# ---------------------------------------------------------------------------
# J. lint_command resolver (task 13)
# ---------------------------------------------------------------------------


def test_resolve_lint_command_returns_None_when_no_layer_set():
    """Pin task 13 §D2 / §D6: HARDCODED_DEFAULTS does NOT include
    lint_command, so the resolver returns None when no layer sets
    it. The runner's None-vs-'' discriminator (§D2) treats this
    as 'unconfigured'."""
    out = resolve("lint_command", task={}, file_defaults={}, cli_overrides={})
    assert out is None


def test_resolve_lint_command_picks_project_layer():
    """Project-defaults set → resolver returns it with from='project_defaults'."""
    out = resolve_task_config(
        task={}, file_defaults={"lint_command": "ruff check"}, cli_overrides={},
    )
    assert out["lint_command"] == {
        "value": "ruff check", "from": "project_defaults",
    }


def test_resolve_lint_command_task_wins_over_project():
    """Per-task lint_command override wins over project default."""
    out = resolve_task_config(
        task={"lint_command": "eslint ."},
        file_defaults={"lint_command": "ruff check"},
        cli_overrides={},
    )
    assert out["lint_command"] == {"value": "eslint .", "from": "task"}


def test_resolve_lint_command_unset_renders_unset_provenance():
    """When no layer sets lint_command, the audit dict shows
    {'value': None, 'from': 'unset'} — surfacing the misconfiguration
    to anyone reading resolved-config.json."""
    out = resolve_task_config(task={}, file_defaults={}, cli_overrides={})
    assert out["lint_command"] == {"value": None, "from": "unset"}


def test_resolve_lint_command_explicit_empty_string_preserved():
    """Pin §D2: an explicit empty string is distinct from None.
    The resolver returns ('', 'project_defaults'); the runner's
    discriminator passes it to the factory unchanged."""
    out = resolve_task_config(
        task={}, file_defaults={"lint_command": ""}, cli_overrides={},
    )
    assert out["lint_command"] == {"value": "", "from": "project_defaults"}


# ---------------------------------------------------------------------------
# K. invariants_file resolver (task 14)
# ---------------------------------------------------------------------------


def test_resolve_invariants_file_returns_None_when_no_layer_set():
    """Pin task 14 §D11: HARDCODED_DEFAULTS does NOT include
    invariants_file, so the resolver returns None when no layer
    sets it. cli.main() interprets None as 'no invariants
    configured'."""
    out = resolve("invariants_file", task={}, file_defaults={}, cli_overrides={})
    assert out is None


def test_resolve_invariants_file_picks_project_layer():
    """Project-defaults set → resolver returns it with from='project_defaults'."""
    out = resolve_task_config(
        task={}, file_defaults={"invariants_file": "docs/invariants.yaml"},
        cli_overrides={},
    )
    assert out["invariants_file"] == {
        "value": "docs/invariants.yaml", "from": "project_defaults",
    }


def test_resolve_invariants_file_unset_renders_unset_provenance():
    """When no layer sets invariants_file, the audit dict shows
    {'value': None, 'from': 'unset'} — surfacing the misconfiguration
    to anyone reading resolved-config.json."""
    out = resolve_task_config(task={}, file_defaults={}, cli_overrides={})
    assert out["invariants_file"] == {"value": None, "from": "unset"}


def test_resolve_invariants_file_explicit_empty_string_preserved():
    """Pin: explicit empty string is distinct from None. cli.main()
    treats both as no-op, but the audit log preserves provenance."""
    out = resolve_task_config(
        task={}, file_defaults={"invariants_file": ""}, cli_overrides={},
    )
    assert out["invariants_file"] == {"value": "", "from": "project_defaults"}


def test_resolve_invariants_file_phase_layer_wins_over_project():
    """Phase override picks up correctly through resolve_task_config."""
    out = resolve_task_config(
        task={}, file_defaults={"invariants_file": "docs/x.yaml"},
        cli_overrides={},
        phase={"invariants_file": "docs/phase.yaml"},
        phase_id="ui",
    )
    assert out["invariants_file"] == {
        "value": "docs/phase.yaml", "from": "phase:ui",
    }


# ---------------------------------------------------------------------------
# L. untranslated_strings_command resolver (task 17)
# ---------------------------------------------------------------------------


def test_resolve_untranslated_strings_command_returns_None_when_no_layer_set():
    """Pin task 17 §D2 / §D6: HARDCODED_DEFAULTS does NOT include
    untranslated_strings_command, so the resolver returns None when
    no layer sets it. The runner's None-vs-'' discriminator (§D2)
    treats this as 'unconfigured'."""
    out = resolve(
        "untranslated_strings_command",
        task={}, file_defaults={}, cli_overrides={},
    )
    assert out is None


def test_resolve_untranslated_strings_command_picks_project_layer():
    """Project-defaults set → resolver returns it with
    from='project_defaults'."""
    out = resolve_task_config(
        task={},
        file_defaults={"untranslated_strings_command": "scripts/i18n.sh"},
        cli_overrides={},
    )
    assert out["untranslated_strings_command"] == {
        "value": "scripts/i18n.sh", "from": "project_defaults",
    }


def test_resolve_untranslated_strings_command_task_wins_over_project():
    """Per-task untranslated_strings_command override wins over
    project default."""
    out = resolve_task_config(
        task={"untranslated_strings_command": "python tools/check_i18n.py"},
        file_defaults={"untranslated_strings_command": "scripts/i18n.sh"},
        cli_overrides={},
    )
    assert out["untranslated_strings_command"] == {
        "value": "python tools/check_i18n.py", "from": "task",
    }


def test_resolve_untranslated_strings_command_unset_renders_unset_provenance():
    """When no layer sets untranslated_strings_command, the audit
    dict shows {'value': None, 'from': 'unset'} — surfacing the
    misconfiguration to anyone reading resolved-config.json."""
    out = resolve_task_config(task={}, file_defaults={}, cli_overrides={})
    assert out["untranslated_strings_command"] == {
        "value": None, "from": "unset",
    }


def test_resolve_untranslated_strings_command_explicit_empty_string_preserved():
    """Pin §D2: an explicit empty string is distinct from None.
    The resolver returns ('', 'project_defaults'); the runner's
    discriminator passes it to the factory unchanged."""
    out = resolve_task_config(
        task={},
        file_defaults={"untranslated_strings_command": ""},
        cli_overrides={},
    )
    assert out["untranslated_strings_command"] == {
        "value": "", "from": "project_defaults",
    }
