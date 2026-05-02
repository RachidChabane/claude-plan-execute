"""Tests for claude_plan_execute.gates.builtin — built-in gate factories."""
from __future__ import annotations

from claude_plan_execute.gates import Gate
from claude_plan_execute.gates.builtin import make_tests_gate


# ---------------------------------------------------------------------------
# A. make_tests_gate basic shape
# ---------------------------------------------------------------------------

def test_make_tests_gate_returns_a_Gate():
    g = make_tests_gate("pytest")
    assert isinstance(g, Gate)


def test_make_tests_gate_kind_is_shell():
    """Pin §D2: tests gate is a shell gate (not agent)."""
    assert make_tests_gate("pytest").kind == "shell"


# ---------------------------------------------------------------------------
# B. bash -c shell-escape-hatch wrapping
# ---------------------------------------------------------------------------

def test_make_tests_gate_command_uses_bash_minus_c():
    """Pin §D2: command is wrapped in ('bash', '-c', test_command) to
    preserve shell semantics (pipelines, &&, env vars)."""
    g = make_tests_gate("pytest -q")
    assert g.command == ("bash", "-c", "pytest -q")


def test_make_tests_gate_preserves_pipeline_command_verbatim():
    """Pin §D2: shell pipelines like 'make test && lint' must NOT be
    pre-split (shlex.split would be a silent regression)."""
    cmd = "pytest -q && npm test"
    g = make_tests_gate(cmd)
    assert g.command == ("bash", "-c", cmd)


def test_make_tests_gate_preserves_multi_line_command_verbatim():
    """Pin §D8: a YAML multi-line test_command becomes 'cmd1\\ncmd2\\n'
    after PyYAML parsing; the factory must hand bash the verbatim
    string (bash treats newlines as command separators)."""
    cmd = "pytest -q\nnpm test\n"
    g = make_tests_gate(cmd)
    assert g.command == ("bash", "-c", cmd)


def test_make_tests_gate_handles_empty_string_command():
    """Pin §D7: an empty test_command produces a no-op gate
    (`bash -c ''` exits 0). The factory itself does no validation."""
    g = make_tests_gate("")
    assert g.command == ("bash", "-c", "")


# ---------------------------------------------------------------------------
# C. Default knobs (pin §D2)
# ---------------------------------------------------------------------------

def test_make_tests_gate_default_timeout_is_10_minutes():
    assert make_tests_gate("pytest").timeout_minutes == 10


def test_make_tests_gate_default_parallel_safe_is_true():
    assert make_tests_gate("pytest").parallel_safe is True


def test_make_tests_gate_default_on_failure_is_block():
    assert make_tests_gate("pytest").on_failure == "block"


def test_make_tests_gate_default_when_is_always():
    assert make_tests_gate("pytest").when == ("always",)


def test_make_tests_gate_returns_a_frozen_Gate():
    """Pin: the returned Gate inherits the dataclass frozen semantics
    (Gate is frozen by task 11 §D1). A caller that tries to mutate
    the result raises."""
    import pytest
    g = make_tests_gate("pytest")
    with pytest.raises(Exception):
        g.timeout_minutes = 99


def test_make_tests_gate_does_NOT_register_on_default_registry():
    """Pin §D1: the factory does NOT touch default_registry. Calling
    it has no observable side effect on the singleton's name list."""
    from claude_plan_execute.gates import default_registry
    before = set(default_registry().names())
    make_tests_gate("pytest")
    make_tests_gate("rspec")
    after = set(default_registry().names())
    assert before == after
