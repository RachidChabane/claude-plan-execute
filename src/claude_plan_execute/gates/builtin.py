"""Built-in gate factories for gates that need per-task config.

Currently exports `make_tests_gate(test_command)` for the `tests`
built-in (relocated from the implement prompt — see task 12 plan).
The factory pattern is the v1-canonical way to build a Gate whose
config is per-task; built-ins WITHOUT per-task config (e.g., a future
`security-auditor` agent gate) register on `default_registry()` at
module-import time as task 11's plan intended.

Task 13 lands `make_lint_gate(lint_command)` here. When a second
factory exists, generalizing into a name → factory map is the natural
next refactor; v1 keeps things simple with explicit imports.
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


__all__ = ["make_tests_gate"]
