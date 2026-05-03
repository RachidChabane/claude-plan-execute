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


# ---------------------------------------------------------------------------
# D. make_lint_gate basic shape (task 13)
# ---------------------------------------------------------------------------

from claude_plan_execute.gates.builtin import make_lint_gate


def test_make_lint_gate_returns_a_Gate():
    g = make_lint_gate("ruff check")
    assert isinstance(g, Gate)


def test_make_lint_gate_kind_is_shell():
    """Pin §D4: lint gate is a shell gate (not agent)."""
    assert make_lint_gate("ruff check").kind == "shell"


# ---------------------------------------------------------------------------
# E. bash -c shell-escape-hatch wrapping (mirrors §B for tests)
# ---------------------------------------------------------------------------

def test_make_lint_gate_command_uses_bash_minus_c():
    """Pin §D4: command is wrapped in ('bash', '-c', lint_command) to
    preserve shell semantics (pipelines, &&, env vars)."""
    g = make_lint_gate("ruff check")
    assert g.command == ("bash", "-c", "ruff check")


def test_make_lint_gate_preserves_pipeline_command_verbatim():
    """Pin §D4: shell pipelines like 'npm run lint && npm run typecheck'
    must NOT be pre-split (shlex.split would be a silent regression)."""
    cmd = "npm run lint && npm run typecheck"
    g = make_lint_gate(cmd)
    assert g.command == ("bash", "-c", cmd)


def test_make_lint_gate_preserves_multi_line_command_verbatim():
    """Pin §D4: a YAML multi-line lint_command becomes 'cmd1\\ncmd2\\n'
    after PyYAML parsing; the factory must hand bash the verbatim
    string (bash treats newlines as command separators)."""
    cmd = "ruff check\nmypy src/\n"
    g = make_lint_gate(cmd)
    assert g.command == ("bash", "-c", cmd)


def test_make_lint_gate_handles_empty_string_command():
    """Pin §D5: an empty lint_command produces a no-op gate
    (`bash -c ''` exits 0). The factory itself does no validation;
    the runner's None-vs-'' discriminator (§D2) decides whether to
    call the factory at all."""
    g = make_lint_gate("")
    assert g.command == ("bash", "-c", "")


# ---------------------------------------------------------------------------
# F. Default knobs (pin §D4) — mirror §C for tests
# ---------------------------------------------------------------------------

def test_make_lint_gate_default_timeout_is_10_minutes():
    assert make_lint_gate("ruff check").timeout_minutes == 10


def test_make_lint_gate_default_parallel_safe_is_true():
    assert make_lint_gate("ruff check").parallel_safe is True


def test_make_lint_gate_default_on_failure_is_block():
    assert make_lint_gate("ruff check").on_failure == "block"


def test_make_lint_gate_default_when_is_always():
    assert make_lint_gate("ruff check").when == ("always",)


def test_make_lint_gate_returns_a_frozen_Gate():
    """Pin: the returned Gate inherits the dataclass frozen semantics
    (Gate is frozen by task 11 §D1). A caller that tries to mutate
    the result raises."""
    import pytest
    g = make_lint_gate("ruff check")
    with pytest.raises(Exception):
        g.timeout_minutes = 99


def test_make_lint_gate_does_NOT_register_on_default_registry():
    """Pin §D1: the factory does NOT touch default_registry. Calling
    it has no observable side effect on the singleton's name list."""
    from claude_plan_execute.gates import default_registry
    before = set(default_registry().names())
    make_lint_gate("ruff check")
    make_lint_gate("eslint .")
    after = set(default_registry().names())
    assert before == after


# ---------------------------------------------------------------------------
# G. test-critique built-in agent gate (task 15)
# ---------------------------------------------------------------------------

# These tests exercise gates/builtin.py's import-time registration
# of `test-critique` and the helper `_build_test_critique_gate()`
# used by tests with the `_clean_default_registry` fixture.

from pathlib import Path

from claude_plan_execute.gates import default_registry
from claude_plan_execute.gates.builtin import _build_test_critique_gate


def test_test_critique_is_registered_on_default_registry_after_import():
    """Pin §D1: `test-critique` is registered on the singleton when
    gates.builtin is imported. The import line at the top of this
    file already triggered the side effect; assert it took."""
    assert "test-critique" in default_registry().names()


def test_test_critique_gate_kind_is_agent():
    """Pin §D5: kind=agent (not shell)."""
    g = default_registry().get("test-critique")
    assert g is not None
    assert g.kind == "agent"


def test_test_critique_gate_model_is_sonnet():
    """Pin §D5: model=sonnet baked into the registration."""
    g = default_registry().get("test-critique")
    assert g.model == "sonnet"


def test_test_critique_gate_effort_is_medium():
    """Pin §D5: effort=medium baked into the registration."""
    g = default_registry().get("test-critique")
    assert g.effort == "medium"


def test_test_critique_gate_on_failure_is_block():
    """Pin §D5 / task description: on_failure=block is required so
    a failing gate writes a FailureRecord (which carries the
    suggested cases into the next-attempt prompt)."""
    g = default_registry().get("test-critique")
    assert g.on_failure == "block"


def test_test_critique_gate_parallel_safe_is_true():
    """Pin §D5: parallel_safe=True — runs concurrently with
    `tests`, `lint`, etc."""
    g = default_registry().get("test-critique")
    assert g.parallel_safe is True


def test_test_critique_gate_when_is_always():
    """Pin §D5: when=('always',) — no tests-changed filter
    (deferred per Non-goals)."""
    g = default_registry().get("test-critique")
    assert g.when == ("always",)


def test_test_critique_gate_timeout_is_10_minutes():
    """Pin §D5: timeout_minutes=10 (the Gate default; explicit
    on the registration for clarity)."""
    g = default_registry().get("test-critique")
    assert g.timeout_minutes == 10


def test_test_critique_gate_prompt_template_is_an_existing_file():
    """Pin §D2 / §D3 / §D4: the registered prompt_template path
    resolves to a real file inside the package. Without the
    pyproject.toml package-data inclusion, wheel installs would
    fail this assertion at runtime (the path would be a string,
    not an existing file)."""
    g = default_registry().get("test-critique")
    p = Path(g.prompt_template)
    assert p.is_absolute(), f"prompt_template should be absolute: {p}"
    assert p.is_file(), f"prompt_template should exist: {p}"
    assert p.name == "test-critique.md"


def test_test_critique_template_has_VERDICT_PASS_pin():
    """Pin §D8: the template contains the literal substring
    'VERDICT: PASS' so the agent's pass-detection contract
    (agent.py:69 — case-insensitive substring search) has the
    sentinel the template instructs the agent to emit."""
    g = default_registry().get("test-critique")
    body = Path(g.prompt_template).read_text(encoding="utf-8")
    assert "VERDICT: PASS" in body or "Verdict: PASS" in body


def test_build_test_critique_gate_returns_a_fresh_Gate():
    """Pin §D11: _build_test_critique_gate() returns a Gate
    with the same shape as the registered singleton entry but
    a fresh instance (callers can register on a fresh registry
    without mutating the singleton)."""
    g1 = _build_test_critique_gate()
    g2 = _build_test_critique_gate()
    # Same shape:
    assert g1.kind == g2.kind == "agent"
    assert g1.model == g2.model == "sonnet"
    # Frozen dataclasses are equal-by-value; identity is allowed
    # to differ but value equality holds:
    assert g1 == g2


def test_build_test_critique_gate_does_NOT_register_on_call():
    """Pin §D1 / §D11: calling the helper does NOT mutate the
    default registry. (The registration is a one-shot
    side effect at module import time; the helper is for
    test fixtures.)"""
    before = set(default_registry().names())
    _build_test_critique_gate()
    _build_test_critique_gate()
    after = set(default_registry().names())
    assert before == after


def test_test_critique_gate_template_path_is_under_gates_templates_dir():
    """Pin §D3: the template file lives under
    src/claude_plan_execute/gates/templates/. A future
    refactor that moves the templates elsewhere would break
    the package-data glob and silently break wheel installs;
    this test pins the location."""
    g = default_registry().get("test-critique")
    p = Path(g.prompt_template)
    assert p.parent.name == "templates"
    assert p.parent.parent.name == "gates"


def test_make_test_critique_gate_template_body_starts_with_test_critique_header():
    """Defensive: the template body has a recognizable header so
    accidental file corruption / wrong-file-shipped errors
    surface in tests."""
    g = default_registry().get("test-critique")
    body = Path(g.prompt_template).read_text(encoding="utf-8")
    # Expect a header that names the gate (or a recognizable
    # opening sentence). The exact wording is part of the
    # template body and should not be tested for verbatim
    # match — that would create a brittle pin every prompt
    # tweak triggers. Test only for the gate name.
    assert "test-critique" in body.lower() or "test critic" in body.lower()


# ---------------------------------------------------------------------------
# H. security-auditor built-in agent gate (task 16)
# ---------------------------------------------------------------------------

# These tests exercise gates/builtin.py's import-time registration
# of `security-auditor` and the helper `_build_security_auditor_gate()`
# used by tests with the `_clean_default_registry` fixture.

from claude_plan_execute.gates.builtin import _build_security_auditor_gate


def test_security_auditor_is_registered_on_default_registry_after_import():
    """Pin §D1: `security-auditor` is registered on the singleton
    when gates.builtin is imported. The import line at the top of
    this file already triggered the side effect; assert it took."""
    assert "security-auditor" in default_registry().names()


def test_security_auditor_gate_kind_is_agent():
    """Pin §D5: kind=agent (not shell)."""
    g = default_registry().get("security-auditor")
    assert g is not None
    assert g.kind == "agent"


def test_security_auditor_gate_model_is_opus():
    """Pin §D5: model=opus baked into the registration (the task
    description names this explicitly: 'Opus/max budget')."""
    g = default_registry().get("security-auditor")
    assert g.model == "opus"


def test_security_auditor_gate_effort_is_max():
    """Pin §D5: effort=max baked into the registration (the task
    description names this explicitly: 'Opus/max budget')."""
    g = default_registry().get("security-auditor")
    assert g.effort == "max"


def test_security_auditor_gate_on_failure_is_block():
    """Pin §D5 / task description ('Block on any found'): a finding
    must write a FailureRecord and stop the task."""
    g = default_registry().get("security-auditor")
    assert g.on_failure == "block"


def test_security_auditor_gate_parallel_safe_is_true():
    """Pin §D5: parallel_safe=True — runs concurrently with
    `tests`, `lint`, `test-critique`, etc."""
    g = default_registry().get("security-auditor")
    assert g.parallel_safe is True


def test_security_auditor_gate_when_is_always():
    """Pin §D5: when=('always',) — no touches:<glob> filter
    (deferred per Non-goals; in-template heuristic handles
    docs-only diffs)."""
    g = default_registry().get("security-auditor")
    assert g.when == ("always",)


def test_security_auditor_gate_timeout_is_10_minutes():
    """Pin §D5: timeout_minutes=10 (the Gate default; explicit on
    the registration for clarity)."""
    g = default_registry().get("security-auditor")
    assert g.timeout_minutes == 10


def test_security_auditor_gate_prompt_template_is_an_existing_file():
    """Pin §D2 / §D3 / §D4: the registered prompt_template path
    resolves to a real file inside the package. The pyproject.toml
    package-data glob `gates/templates/*.md` (set in task 15)
    covers this file too."""
    g = default_registry().get("security-auditor")
    p = Path(g.prompt_template)
    assert p.is_absolute(), f"prompt_template should be absolute: {p}"
    assert p.is_file(), f"prompt_template should exist: {p}"
    assert p.name == "security-auditor.md"


def test_security_auditor_template_has_VERDICT_PASS_pin():
    """Pin §D8: the template contains the literal substring
    'VERDICT: PASS' (case-insensitive forms accepted) so the
    agent's pass-detection contract (agent.py:69) has the
    sentinel the template instructs the agent to emit."""
    g = default_registry().get("security-auditor")
    body = Path(g.prompt_template).read_text(encoding="utf-8")
    assert "VERDICT: PASS" in body or "Verdict: PASS" in body


def test_build_security_auditor_gate_returns_a_fresh_Gate():
    """Pin §D12: _build_security_auditor_gate() returns a Gate
    with the same shape as the registered singleton entry. Frozen
    dataclasses are equal-by-value; the helper is for test
    fixtures."""
    g1 = _build_security_auditor_gate()
    g2 = _build_security_auditor_gate()
    assert g1.kind == g2.kind == "agent"
    assert g1.model == g2.model == "opus"
    assert g1.effort == g2.effort == "max"
    assert g1 == g2


def test_build_security_auditor_gate_does_NOT_register_on_call():
    """Pin §D1 / §D12: calling the helper does NOT mutate the
    default registry. (Registration is a one-shot side effect at
    module import time; the helper is for test fixtures.)"""
    before = set(default_registry().names())
    _build_security_auditor_gate()
    _build_security_auditor_gate()
    after = set(default_registry().names())
    assert before == after


def test_security_auditor_gate_template_path_is_under_gates_templates_dir():
    """Pin §D3: the template file lives under
    src/claude_plan_execute/gates/templates/. A future refactor
    that moves templates elsewhere would break the package-data
    glob and silently break wheel installs; this test pins the
    location."""
    g = default_registry().get("security-auditor")
    p = Path(g.prompt_template)
    assert p.parent.name == "templates"
    assert p.parent.parent.name == "gates"


def test_security_auditor_template_body_starts_with_recognizable_header():
    """Defensive: the template body has a recognizable header so
    accidental file corruption / wrong-file-shipped errors surface
    in tests. The gate's name or 'security auditor' must appear in
    the body somewhere — exact wording is not pinned."""
    g = default_registry().get("security-auditor")
    body = Path(g.prompt_template).read_text(encoding="utf-8")
    assert (
        "security-auditor" in body.lower()
        or "security auditor" in body.lower()
    )


def test_test_critique_and_security_auditor_are_both_registered():
    """Pin §D1: both task-15 (test-critique) and task-16
    (security-auditor) registrations coexist on the default
    singleton after gates.builtin is imported. Regression guard
    against a future refactor that accidentally deletes one."""
    names = default_registry().names()
    assert "test-critique" in names
    assert "security-auditor" in names
