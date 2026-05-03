"""Built-in gate factories and registrations.

Exports two kinds of built-ins:

  - Factory functions for built-ins WITH per-task config:
    `make_tests_gate(test_command)`, `make_lint_gate(lint_command)`.
    The runner calls these factories at gate-resolution time
    inside `execute_task` (task 12 / 13 §D3); the resulting
    Gates are NOT pre-registered on `default_registry()`.

  - Module-import-time registrations for built-ins WITHOUT
    per-task config: `test-critique` (task 15) and
    `security-auditor` (task 16). The Gate is constructed once
    and registered on the default singleton when this module is
    imported (transitively via `runner.py` or directly in tests).

The factory pattern and the import-time pattern coexist; the
test-critique entry was the first of the import-time kind, and
security-auditor (task 16) is the second. If a third built-in
agent gate joins (e.g., design-fidelity — task 20), this
module should refactor to a `_register_all_builtin_agents(registry)`
helper; with two entries the explicit calls remain the simplest
form.
"""
from __future__ import annotations

from pathlib import Path

from claude_plan_execute.gates import Gate, default_registry


# ---------------------------------------------------------------------------
# Factories for built-ins with per-task config
# ---------------------------------------------------------------------------

def make_tests_gate(test_command: str) -> Gate:
    """Construct the `tests` Gate from a resolved test_command string.

    Wraps the command in `("bash", "-c", test_command)` to preserve
    shell semantics (pipelines, `&&`, env-var substitution) that the
    pre-task-12 implement-prompt path implicitly supported. The
    runner calls this factory inside `execute_task` (task 12 §D3) when
    it sees the name "tests" in the configured gate-name list; the
    Gate is NOT pre-registered on `default_registry()` because the
    command is per-task.
    """
    return Gate(
        kind="shell",
        command=("bash", "-c", test_command),
        timeout_minutes=10,
        parallel_safe=True,
        on_failure="block",
        when=("always",),
    )


def make_lint_gate(lint_command: str) -> Gate:
    """Construct the `lint` Gate from a resolved lint_command string.

    Wraps the command in `("bash", "-c", lint_command)` to preserve
    shell semantics (pipelines, `&&`, env-var substitution) so a
    project's `lint_command: "npm run lint && npm run typecheck"`
    works without the runner pre-splitting. The runner calls this
    factory inside `execute_task` (task 13 §D3) when it sees the
    name "lint" in the configured gate-name list AND the resolved
    lint_command is non-None; the Gate is NOT pre-registered on
    `default_registry()` because the command is per-task.

    An explicit empty string produces a no-op gate (`bash -c ''`
    exits 0). The runner reserves None for the "unconfigured"
    state and skips with a warning instead of calling this
    factory; see §D2.
    """
    return Gate(
        kind="shell",
        command=("bash", "-c", lint_command),
        timeout_minutes=10,
        parallel_safe=True,
        on_failure="block",
        when=("always",),
    )


# ---------------------------------------------------------------------------
# Built-in agent gates registered on default_registry() at import time
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _build_test_critique_gate() -> Gate:
    """Construct the `test-critique` agent Gate.

    Sonnet/medium agent gate that reads the diff and new tests,
    asking 'would these tests catch a regression in the actual
    contract?' Returns PASS/FAIL with suggested cases. Flag-only.
    See plans/task-15/plan.md §D5 for the knob choices.

    The Gate's `prompt_template` is the absolute path to the
    shipped template at `gates/templates/test-critique.md`,
    computed via `Path(__file__).parent / "templates"` so the
    gate is location-independent. (§D2.)

    Tests using the `_clean_default_registry` fixture (which
    replaces the singleton with a fresh empty registry) call
    this helper to register `test-critique` on their fresh
    registry. (§D11.)
    """
    return Gate(
        kind="agent",
        prompt_template=str(_TEMPLATES_DIR / "test-critique.md"),
        model="sonnet",
        effort="medium",
        timeout_minutes=10,
        parallel_safe=True,
        on_failure="block",
        when=("always",),
    )


def _build_security_auditor_gate() -> Gate:
    """Construct the `security-auditor` agent Gate.

    Opus/max agent gate that reads the diff and scans for OWASP-
    relevant security regressions: ACL/IDOR bypasses, missing 403
    paths, secrets, injection, unsafe deserialization, weak
    crypto, permissive CORS, sensitive-data leaks. Returns
    PASS/FAIL with itemized findings (file:line, category,
    evidence, suggested fix). Block on any finding.
    See plans/task-16/plan.md §D5 for the knob choices.

    The Gate's `prompt_template` is the absolute path to the
    shipped template at `gates/templates/security-auditor.md`,
    computed via the module-level `_TEMPLATES_DIR` constant
    (task 15 §D2) so the gate is location-independent.

    Tests using the `_clean_default_registry` fixture (which
    replaces the singleton with a fresh empty registry) call
    this helper to register `security-auditor` on their fresh
    registry. (§D12.)
    """
    return Gate(
        kind="agent",
        prompt_template=str(_TEMPLATES_DIR / "security-auditor.md"),
        model="opus",
        effort="max",
        timeout_minutes=10,
        parallel_safe=True,
        on_failure="block",
        when=("always",),
    )


# Side-effecting registration. Runs once when this module is first
# imported (via `runner.py:13` in production, via test files
# directly in tests). See §D1 for the design choice.
default_registry().register(
    "test-critique", _build_test_critique_gate(),
)
default_registry().register(
    "security-auditor", _build_security_auditor_gate(),
)


__all__ = [
    "make_lint_gate",
    "make_tests_gate",
]
