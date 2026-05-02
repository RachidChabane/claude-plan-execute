"""Built-in gate factories for gates that need per-task config.

Exports `make_tests_gate(test_command)` and
`make_lint_gate(lint_command)`. The factory pattern is the
v1-canonical way to build a Gate whose config is per-task; built-ins
WITHOUT per-task config (e.g., a future `security-auditor` agent
gate) register on `default_registry()` at module-import time.

Generalizing into a name → factory map is deferred until a third
factory exists; with two, an explicit `if name == "tests": ... elif
name == "lint": ...` chain in the runner is simpler and clearer than
a dict-driven dispatch.
"""
from __future__ import annotations

from claude_plan_execute.gates import Gate


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


__all__ = ["make_lint_gate", "make_tests_gate"]
