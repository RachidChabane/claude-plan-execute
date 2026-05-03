"""Tests for claude_plan_execute.gates.invariants — invariants.yaml
loader, compiler, and registration."""
from __future__ import annotations

from pathlib import Path

import pytest

from claude_plan_execute.gates import Gate, GateRegistry
from claude_plan_execute.gates.invariants import (
    InvariantEntry,
    compile_entry,
    load_invariants,
    loaded_invariant_names,
    register_invariants_on,
)


# ---------------------------------------------------------------------------
# A. InvariantEntry Pydantic shape
# ---------------------------------------------------------------------------

def test_invariant_entry_minimal_shell_with_command_validates():
    e = InvariantEntry(name="x", command="true")
    assert e.kind == "shell"
    assert e.command == "true"


def test_invariant_entry_default_kind_is_shell():
    e = InvariantEntry(name="x", command="true")
    assert e.kind == "shell"


def test_invariant_entry_default_on_failure_is_block():
    e = InvariantEntry(name="x", command="true")
    assert e.on_failure == "block"


def test_invariant_entry_default_timeout_is_10_minutes():
    e = InvariantEntry(name="x", command="true")
    assert e.timeout_minutes == 10


def test_invariant_entry_default_parallel_safe_is_true():
    e = InvariantEntry(name="x", command="true")
    assert e.parallel_safe is True


def test_invariant_entry_extra_forbidden():
    """Pin §D4: extra='forbid' for typo detection."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        InvariantEntry(name="x", command="true", typo_field="oops")


def test_invariant_entry_agent_kind_with_prompt_template():
    e = InvariantEntry(name="x", kind="agent", prompt_template="docs/foo.md")
    assert e.kind == "agent"
    assert e.prompt_template == "docs/foo.md"


def test_invariant_entry_shell_with_two_specifiers_raises():
    """Pin §D4: shell entry that sets BOTH command and
    regex_must_not_match is rejected at validate time. Prevents
    silent first-wins precedence."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="exactly one is required"):
        InvariantEntry(
            name="x", command="true", regex_must_not_match="X",
        )


def test_invariant_entry_shell_with_two_regex_fields_raises():
    """Pin §D4: setting both regex_must_match and regex_must_not_match
    is rejected (only one of the three shell specifiers allowed)."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="exactly one is required"):
        InvariantEntry(
            name="x", regex_must_match="X", regex_must_not_match="Y",
        )


def test_invariant_entry_agent_with_command_does_NOT_raise_validate():
    """Pin §D4: the validator only fires for kind='shell'. An agent
    entry that ALSO has 'command' set passes validation; the
    compiler then ignores the unused field."""
    e = InvariantEntry(
        name="x", kind="agent", prompt_template="docs/foo.md",
        command="this-is-ignored",
    )
    assert e.kind == "agent"


# ---------------------------------------------------------------------------
# B. compile_entry — kind=shell with raw command
# ---------------------------------------------------------------------------

def test_compile_shell_command_wraps_in_bash_dash_c():
    e = InvariantEntry(name="x", command="grep -q foo .")
    g = compile_entry(e)
    assert g.kind == "shell"
    assert g.command == ("bash", "-c", "grep -q foo .")


def test_compile_shell_command_preserves_pipeline():
    cmd = "ls && grep -q foo *.py | wc -l"
    g = compile_entry(InvariantEntry(name="x", command=cmd))
    assert g.command == ("bash", "-c", cmd)


def test_compile_shell_default_when_is_always():
    g = compile_entry(InvariantEntry(name="x", command="true"))
    assert g.when == ("always",)


def test_compile_shell_explicit_when_is_passed_through():
    g = compile_entry(
        InvariantEntry(name="x", command="true", when=["task_class:ui"]),
    )
    assert g.when == ("task_class:ui",)


def test_compile_shell_propagates_timeout_and_on_failure():
    g = compile_entry(
        InvariantEntry(name="x", command="true",
                       timeout_minutes=20, on_failure="warn"),
    )
    assert g.timeout_minutes == 20
    assert g.on_failure == "warn"


# ---------------------------------------------------------------------------
# C. compile_entry — kind=shell with regex sugar
# ---------------------------------------------------------------------------

def test_compile_shell_regex_must_not_match_compiles_to_inverted_grep():
    """Pin §D5: regex_must_not_match → `! grep -RIn -E -- <pat> <paths>`.
    Any match → grep exits 0 → bash inverts to 1 → gate FAILS."""
    e = InvariantEntry(
        name="x",
        regex_must_not_match="BadPattern",
        files=["src/"],
    )
    g = compile_entry(e)
    assert g.kind == "shell"
    assert g.command[0] == "bash"
    assert g.command[1] == "-c"
    body = g.command[2]
    assert body.startswith("! grep -RIn -E -- ")
    assert "BadPattern" in body
    assert "src/" in body


def test_compile_shell_regex_must_match_compiles_to_grep():
    """Pin §D5: regex_must_match → `grep -RIn -E -- <pat> <paths>`.
    No match → grep exits 1 → gate FAILS."""
    e = InvariantEntry(
        name="x", regex_must_match="GoodPattern", files=["src/"],
    )
    g = compile_entry(e)
    body = g.command[2]
    assert body.startswith("grep -RIn -E -- ")
    assert "GoodPattern" in body
    assert "src/" in body


def test_compile_shell_regex_default_files_is_repo_root():
    """Pin §D5: default files paths is ['.'] (cwd walk via grep -R).
    Earlier draft defaulted to ['**/*']; that broke under shlex.quote."""
    e = InvariantEntry(name="x", regex_must_not_match="X")
    g = compile_entry(e)
    body = g.command[2]
    assert body.endswith(" .")


def test_compile_shell_regex_special_chars_are_quoted():
    """Pin §D5: shlex.quote protects metacharacters in the regex."""
    e = InvariantEntry(name="x", regex_must_not_match="$VAR`cmd`")
    g = compile_entry(e)
    body = g.command[2]
    assert "'$VAR`cmd`'" in body


def test_compile_shell_regex_path_with_space_is_quoted():
    """Pin §D5: each file path is shlex-quoted individually so paths
    with spaces survive the bash -c round trip."""
    e = InvariantEntry(name="x", regex_must_match="X", files=["src/with space/"])
    g = compile_entry(e)
    body = g.command[2]
    assert "'src/with space/'" in body


def test_compile_shell_regex_with_empty_files_raises():
    """Pin §D5: empty files: [] is a compile-time error."""
    e = InvariantEntry(name="x", regex_must_match="X", files=[])
    with pytest.raises(ValueError):
        compile_entry(e)


def test_compile_shell_with_no_command_or_regex_raises():
    """Pin §D5: shell entry needs ONE of command/regex_must_match/regex_must_not_match."""
    e = InvariantEntry(name="x", kind="shell")
    with pytest.raises(ValueError, match="requires one of"):
        compile_entry(e)


# ---------------------------------------------------------------------------
# C2. compile_entry — regex sugar END-TO-END (executes the bash command)
# ---------------------------------------------------------------------------

def test_compile_shell_regex_must_match_passes_when_pattern_found(tmp_path):
    """End-to-end: regex_must_match + matching fixture file → grep
    finds → exit 0 → gate PASSES."""
    import subprocess
    (tmp_path / "good.py").write_text("GoodPattern in code\n")
    e = InvariantEntry(
        name="x", regex_must_match="GoodPattern", files=[str(tmp_path)],
    )
    g = compile_entry(e)
    r = subprocess.run(g.command, capture_output=True, text=True)
    assert r.returncode == 0


def test_compile_shell_regex_must_match_fails_when_pattern_absent(tmp_path):
    """End-to-end: regex_must_match + no match in fixture tree →
    grep exits 1 → gate FAILS."""
    import subprocess
    (tmp_path / "x.py").write_text("Hello world\n")
    e = InvariantEntry(
        name="x", regex_must_match="MissingPattern", files=[str(tmp_path)],
    )
    g = compile_entry(e)
    r = subprocess.run(g.command, capture_output=True, text=True)
    assert r.returncode != 0


def test_compile_shell_regex_must_not_match_passes_when_pattern_absent(tmp_path):
    """End-to-end: regex_must_not_match + clean fixture tree → grep
    exits 1 → bash inverts to 0 → gate PASSES."""
    import subprocess
    (tmp_path / "clean.py").write_text("nothing forbidden here\n")
    e = InvariantEntry(
        name="x", regex_must_not_match="BadPattern", files=[str(tmp_path)],
    )
    g = compile_entry(e)
    r = subprocess.run(g.command, capture_output=True, text=True)
    assert r.returncode == 0


def test_compile_shell_regex_must_not_match_fails_when_pattern_found(tmp_path):
    """End-to-end: regex_must_not_match + actual match in fixture →
    grep exits 0 → bash inverts to 1 → gate FAILS.

    REGRESSION GUARD for I2 (review-1): an earlier design defaulted
    `files` to `['**/*']` which silently passed without scanning."""
    import subprocess
    (tmp_path / "bad.py").write_text("BadPattern is here\n")
    e = InvariantEntry(
        name="x", regex_must_not_match="BadPattern", files=[str(tmp_path)],
    )
    g = compile_entry(e)
    r = subprocess.run(g.command, capture_output=True, text=True)
    assert r.returncode != 0


# ---------------------------------------------------------------------------
# D. compile_entry — kind=agent
# ---------------------------------------------------------------------------

def test_compile_agent_with_prompt_template_returns_agent_gate():
    e = InvariantEntry(name="x", kind="agent", prompt_template="docs/foo.md")
    g = compile_entry(e)
    assert g.kind == "agent"
    assert g.prompt_template == "docs/foo.md"


def test_compile_agent_propagates_model_and_effort():
    e = InvariantEntry(
        name="x", kind="agent", prompt_template="x",
        model="opus", effort="max",
    )
    g = compile_entry(e)
    assert g.model == "opus"
    assert g.effort == "max"


def test_compile_agent_without_prompt_template_raises():
    e = InvariantEntry(name="x", kind="agent")
    with pytest.raises(ValueError, match="requires prompt_template"):
        compile_entry(e)


# ---------------------------------------------------------------------------
# E. load_invariants — file I/O edges
# ---------------------------------------------------------------------------

def test_load_invariants_returns_empty_for_None():
    assert load_invariants(None) == []


def test_load_invariants_returns_empty_for_empty_string():
    assert load_invariants("") == []


def test_load_invariants_returns_empty_for_missing_file(tmp_path):
    assert load_invariants(tmp_path / "nope.yaml") == []


def test_load_invariants_returns_empty_for_directory(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    assert load_invariants(d) == []


def test_load_invariants_returns_empty_for_empty_yaml(tmp_path):
    """Empty file → yaml.safe_load returns None → []."""
    p = tmp_path / "inv.yaml"
    p.write_text("")
    assert load_invariants(p) == []


def test_load_invariants_warns_for_non_list_root(tmp_path, capsys):
    """Pin §D6: top-level scalar / dict is rejected with a warning."""
    p = tmp_path / "inv.yaml"
    p.write_text("foo: bar\n")
    out = load_invariants(p)
    assert out == []
    assert "must be a YAML list" in capsys.readouterr().out


def test_load_invariants_loads_one_shell_entry(tmp_path):
    p = tmp_path / "inv.yaml"
    p.write_text(
        "- name: invariant-foo\n"
        "  command: 'true'\n"
    )
    out = load_invariants(p)
    assert len(out) == 1
    assert out[0][0] == "invariant-foo"
    assert out[0][1].kind == "shell"
    assert out[0][1].command == ("bash", "-c", "true")


def test_load_invariants_skips_entry_without_name(tmp_path, capsys):
    """Pin §D6: missing 'name' → warn + skip; rest of file loads."""
    p = tmp_path / "inv.yaml"
    p.write_text(
        "- command: 'true'\n"
        "- name: invariant-bar\n"
        "  command: 'false'\n"
    )
    out = load_invariants(p)
    assert len(out) == 1
    assert out[0][0] == "invariant-bar"
    assert "missing 'name'" in capsys.readouterr().out


def test_load_invariants_skips_entry_with_pydantic_error(tmp_path, capsys):
    """Pin §D6: per-entry validation errors warn + skip; rest loads."""
    p = tmp_path / "inv.yaml"
    p.write_text(
        "- name: bad\n"
        "  typo_field: oops\n"
        "- name: good\n"
        "  command: 'true'\n"
    )
    out = load_invariants(p)
    assert len(out) == 1
    assert out[0][0] == "good"
    err = capsys.readouterr().out
    assert "bad" in err


def test_load_invariants_rejects_reserved_name_invariant_grep(tmp_path, capsys):
    """Pin §D7 + §R7: 'invariant-grep' is RESERVED FOREVER as the
    meta-name. This test is the regression guard."""
    p = tmp_path / "inv.yaml"
    p.write_text(
        "- name: invariant-grep\n"
        "  command: 'true'\n"
    )
    out = load_invariants(p)
    assert out == []
    assert "reserved" in capsys.readouterr().out


def test_load_invariants_rejects_reserved_name_tests(tmp_path, capsys):
    """Pin §D7: 'tests' is reserved (built-in special-case)."""
    p = tmp_path / "inv.yaml"
    p.write_text(
        "- name: tests\n"
        "  command: 'true'\n"
    )
    out = load_invariants(p)
    assert out == []


def test_load_invariants_rejects_reserved_name_lint(tmp_path, capsys):
    """Pin §D7: 'lint' is reserved (built-in special-case)."""
    p = tmp_path / "inv.yaml"
    p.write_text(
        "- name: lint\n"
        "  command: 'true'\n"
    )
    out = load_invariants(p)
    assert out == []


def test_load_invariants_skips_duplicate_name(tmp_path, capsys):
    """Pin §D6: in-file duplicate names → warn + skip the second."""
    p = tmp_path / "inv.yaml"
    p.write_text(
        "- name: invariant-foo\n"
        "  command: 'true'\n"
        "- name: invariant-foo\n"
        "  command: 'false'\n"
    )
    out = load_invariants(p)
    assert len(out) == 1
    assert out[0][1].command == ("bash", "-c", "true")
    assert "duplicate" in capsys.readouterr().out


def test_load_invariants_loads_mixed_kinds(tmp_path):
    p = tmp_path / "inv.yaml"
    p.write_text(
        "- name: invariant-shell\n"
        "  command: 'true'\n"
        "- name: invariant-agent\n"
        "  kind: agent\n"
        "  prompt_template: docs/x.md\n"
        "- name: invariant-regex\n"
        "  regex_must_not_match: BadPattern\n"
    )
    out = load_invariants(p)
    assert [n for n, _ in out] == [
        "invariant-shell", "invariant-agent", "invariant-regex",
    ]
    assert out[0][1].kind == "shell"
    assert out[1][1].kind == "agent"
    assert out[2][1].kind == "shell"


# ---------------------------------------------------------------------------
# F. register_invariants_on
# ---------------------------------------------------------------------------

def test_register_invariants_returns_registered_names():
    reg = GateRegistry()
    entries = [
        ("a", Gate(kind="shell", command=("bash", "-c", "true"))),
        ("b", Gate(kind="shell", command=("bash", "-c", "true"))),
    ]
    out = register_invariants_on(reg, entries)
    assert out == ["a", "b"]
    assert sorted(reg.names()) == ["a", "b"]


def test_register_invariants_warn_skips_collision_with_existing(capsys):
    reg = GateRegistry()
    reg.register("a", Gate(kind="shell", command=("bash", "-c", "true")))
    out = register_invariants_on(reg, [
        ("a", Gate(kind="shell", command=("bash", "-c", "false"))),
        ("b", Gate(kind="shell", command=("bash", "-c", "true"))),
    ])
    assert out == ["b"]
    assert "not registered" in capsys.readouterr().out


def test_register_invariants_local_registry_does_NOT_pollute_module_global(monkeypatch):
    """Pin §D8: registering on a non-default registry leaves
    _loaded_invariant_names alone."""
    from claude_plan_execute.gates.invariants import _loaded_invariant_names
    before = list(_loaded_invariant_names)
    local = GateRegistry()
    register_invariants_on(local, [
        ("a", Gate(kind="shell", command=("bash", "-c", "true"))),
    ])
    after = list(_loaded_invariant_names)
    assert before == after


def test_register_invariants_default_registry_appends_to_module_global(monkeypatch):
    """Pin §D8: registering on default_registry() updates the module
    global so the runner's meta-name expansion sees the new names."""
    from claude_plan_execute.gates import default_registry
    monkeypatch.setattr(
        "claude_plan_execute.gates.invariants._loaded_invariant_names", [],
    )
    register_invariants_on(default_registry(), [
        ("loaded-a-task14", Gate(kind="shell", command=("bash", "-c", "true"))),
        ("loaded-b-task14", Gate(kind="shell", command=("bash", "-c", "true"))),
    ])
    assert loaded_invariant_names() == ["loaded-a-task14", "loaded-b-task14"]
    default_registry()._gates.pop("loaded-a-task14", None)
    default_registry()._gates.pop("loaded-b-task14", None)


def test_loaded_invariant_names_returns_a_copy():
    """Mutations on the returned list do not affect the module global."""
    from claude_plan_execute.gates.invariants import _loaded_invariant_names
    snapshot = loaded_invariant_names()
    snapshot.append("INJECTED")
    assert "INJECTED" not in _loaded_invariant_names
