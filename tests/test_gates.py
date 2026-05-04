"""Tests for claude_plan_execute.gates — Gate Registry foundation."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from claude_plan_execute.gates import (
    AgentGate,
    Gate,
    GateRegistry,
    GateResult,
    ShellGate,
    _when_matches,
    default_registry,
    run_gates,
)
from claude_plan_execute.gates.agent import (
    _FINDINGS_RE,
    _RAW_CAP_BYTES,
    _extract_findings,
)
from claude_plan_execute.gates.shell import _OUTPUT_CAP_BYTES, _tail_cap


# ---------------------------------------------------------------------------
# Smoke test (pin §R6: late-import ordering must not break package load)
# ---------------------------------------------------------------------------

def test_gates_package_imports_cleanly():
    """Pin §R6: gates/__init__.py defines dataclasses BEFORE late
    sibling imports. If the ordering breaks, this import fails."""
    import claude_plan_execute.gates as g
    assert hasattr(g, "Gate")
    assert hasattr(g, "GateResult")
    assert hasattr(g, "GateRegistry")
    assert hasattr(g, "default_registry")
    assert hasattr(g, "run_gates")
    assert hasattr(g, "ShellGate")
    assert hasattr(g, "AgentGate")


# ---------------------------------------------------------------------------
# A. Gate / GateResult dataclasses
# ---------------------------------------------------------------------------

def test_gate_field_order_matches_task_description():
    """Task 11 description names fields in this exact order:
    kind, command, prompt_template, model, effort, timeout_minutes,
    parallel_safe, on_failure, when. Pin field order as a regression
    guard against accidental reordering."""
    fields = [f.name for f in Gate.__dataclass_fields__.values()]
    assert fields == [
        "kind", "command", "prompt_template", "model", "effort",
        "timeout_minutes", "parallel_safe", "on_failure", "when",
    ]


def test_gate_is_frozen():
    """Pin §D1: frozen dataclass."""
    g = Gate(kind="shell", command=("echo", "hi"))
    with pytest.raises(Exception):
        g.kind = "agent"


def test_gate_default_timeout_is_10_minutes():
    g = Gate(kind="shell", command=("true",))
    assert g.timeout_minutes == 10


def test_gate_default_parallel_safe_is_true():
    g = Gate(kind="shell", command=("true",))
    assert g.parallel_safe is True


def test_gate_default_on_failure_is_block():
    g = Gate(kind="shell", command=("true",))
    assert g.on_failure == "block"


def test_gate_default_when_is_always():
    g = Gate(kind="shell", command=("true",))
    assert g.when == ("always",)


def test_gate_result_required_fields_present():
    r = GateResult(
        name="x", passed=True, output="", output_path=None,
        duration_seconds=0.0, on_failure="block",
    )
    assert r.name == "x"
    assert r.passed is True
    assert r.error is None  # default


# ---------------------------------------------------------------------------
# B. GateRegistry
# ---------------------------------------------------------------------------

def test_registry_register_and_get():
    reg = GateRegistry()
    g = Gate(kind="shell", command=("true",))
    reg.register("tests", g)
    assert reg.get("tests") is g


def test_registry_get_unknown_returns_None():
    reg = GateRegistry()
    assert reg.get("missing") is None


def test_registry_register_duplicate_raises():
    reg = GateRegistry()
    reg.register("a", Gate(kind="shell", command=("true",)))
    with pytest.raises(ValueError, match="already registered"):
        reg.register("a", Gate(kind="shell", command=("false",)))


def test_registry_names_returns_registered_names():
    reg = GateRegistry()
    reg.register("a", Gate(kind="shell", command=("true",)))
    reg.register("b", Gate(kind="shell", command=("true",)))
    assert sorted(reg.names()) == ["a", "b"]


def test_registry_resolve_returns_pairs_in_input_order():
    reg = GateRegistry()
    reg.register("b", Gate(kind="shell", command=("true",)))
    reg.register("a", Gate(kind="shell", command=("true",)))
    resolved, unknown = reg.resolve(["a", "b"])
    names = [n for (n, _) in resolved]
    assert names == ["a", "b"]
    assert unknown == []


def test_registry_resolve_separates_unknown_names():
    reg = GateRegistry()
    reg.register("known", Gate(kind="shell", command=("true",)))
    resolved, unknown = reg.resolve(["known", "missing", "other"])
    assert [n for (n, _) in resolved] == ["known"]
    assert unknown == ["missing", "other"]


def test_registry_resolve_preserves_duplicates():
    """Pin §D12: no deduplication; the user's literal list is honored."""
    reg = GateRegistry()
    reg.register("x", Gate(kind="shell", command=("true",)))
    resolved, _ = reg.resolve(["x", "x", "x"])
    assert len(resolved) == 3


def test_default_registry_is_a_singleton():
    """Pin §D3: default_registry() returns the same instance."""
    assert default_registry() is default_registry()


def test_default_registry_has_test_critique_after_builtin_import():
    """Task 15: importing gates.builtin registers the test-critique
    agent gate on the default singleton at module-import time.
    The pre-task-15 'starts empty' contract is preserved IF
    builtin has not been imported, but every production path
    (cli → runner → builtin) imports it transitively. The new
    pin asserts containment after explicit import, not strict
    equality (other test modules may add unrelated entries to
    the singleton across a pytest session)."""
    import claude_plan_execute.gates.builtin  # noqa: F401 — import-time side effect
    assert "test-critique" in default_registry().names()


# ---------------------------------------------------------------------------
# C. _when_matches filter
# ---------------------------------------------------------------------------

def test_when_matches_always_matches():
    ok, _ = _when_matches(("always",), {"id": 1})
    assert ok is True


def test_when_matches_empty_tuple_matches():
    ok, _ = _when_matches((), {"id": 1})
    assert ok is True


def test_when_matches_task_class_matches_when_task_class_equals_name():
    """Task 20: task_class:<name> is implemented; matches when the
    task's effective class equals the named class."""
    ok, reason = _when_matches(("task_class:ui",), {"id": 1, "task_class": "ui"})
    assert ok is True
    assert "ui" in reason


def test_when_matches_task_class_does_not_match_when_class_differs():
    """Task 20: a task whose class doesn't equal the named class
    is skipped with a clear reason."""
    ok, reason = _when_matches(
        ("task_class:ui",), {"id": 1, "task_class": "schema"},
    )
    assert ok is False
    assert "schema" in reason
    assert "ui" in reason


def test_when_matches_task_class_does_not_match_when_class_unset():
    """Task 20: a task with no task_class is skipped against a
    task_class:<name> filter."""
    ok, reason = _when_matches(
        ("task_class:ui",), {"id": 1},
    )
    assert ok is False
    assert "ui" in reason


def test_when_matches_task_class_or_semantics_with_multiple_tokens():
    """Task 20: multiple task_class:<name> tokens are OR-ed."""
    ok, _ = _when_matches(
        ("task_class:ui", "task_class:landing"),
        {"id": 1, "task_class": "landing"},
    )
    assert ok is True


def test_when_matches_always_wins_over_task_class():
    """Task 20: 'always' anywhere in `when` short-circuits class
    checking; gate runs even if the class doesn't match."""
    ok, _ = _when_matches(
        ("task_class:ui", "always"),
        {"id": 1, "task_class": "schema"},
    )
    assert ok is True


def test_when_matches_touches_with_no_key_files_attribute():
    """Task 3 case (c): a task with no `key_files` key on the dict
    yields an empty key-files list; touches patterns can't match
    anything; result is False with a touches=... not in key_files=[]
    reason."""
    ok, reason = _when_matches(("touches:src/auth/**",), {"id": 1})
    assert ok is False
    assert "touches=" in reason
    assert "key_files=[]" in reason


# --- task 3: touches:<glob> matcher (gitignore-style globs) ---

def test_when_matches_touches_recursive_double_star_matches_at_any_depth():
    """Task 3 case (a) — the design-critical bug-lock test.

    `touches:**/auth/**` MUST match `frontend/src/auth/Login.tsx`
    via gitignore semantics. stdlib fnmatch handles this case
    accidentally correctly (see plan §"Documented deviations"),
    but fails on `auth/foo.py` and over-matches `src/api/*.py`
    against `src/api/v1/users.py`. Pathspec's GitWildMatchPattern
    is the contract; this test pins that the contract is what
    we ship."""
    ok, reason = _when_matches(
        ("touches:**/auth/**",),
        {"id": 1, "key_files": ["frontend/src/auth/Login.tsx"]},
    )
    assert ok is True
    assert "frontend/src/auth/Login.tsx" in reason


def test_when_matches_touches_recursive_double_star_matches_at_root_depth():
    """Task 3 case (a) extended: gitignore's leading `**/` matches
    zero or more path components. `touches:**/auth/**` must match
    `auth/foo.py` (no prefix). This is the under-match case stdlib
    fnmatch gets wrong."""
    ok, _ = _when_matches(
        ("touches:**/auth/**",),
        {"id": 1, "key_files": ["auth/foo.py"]},
    )
    assert ok is True


def test_when_matches_touches_single_component_glob_does_not_cross_directories():
    """Task 3 case (b): `src/api/*.py` matches `src/api/users.py`
    but NOT `src/api/v1/users.py`. This is the over-match case
    stdlib fnmatch gets wrong; pathspec's path-component awareness
    is the fix."""
    ok, reason = _when_matches(
        ("touches:src/api/*.py",),
        {"id": 1, "key_files": ["src/api/users.py"]},
    )
    assert ok is True
    assert "src/api/users.py" in reason

    # Pin the negative half — a deeper path does NOT match.
    ok2, _ = _when_matches(
        ("touches:src/api/*.py",),
        {"id": 1, "key_files": ["src/api/v1/users.py"]},
    )
    assert ok2 is False


def test_when_matches_touches_with_empty_key_files_list():
    """Task 3 case (d): `key_files: []` (explicit empty list)
    yields no matches. Distinct from case (c) which has no
    key_files key at all; both produce False but exercise
    different code paths in `task.get('key_files') or []`."""
    ok, reason = _when_matches(
        ("touches:src/auth/**",),
        {"id": 1, "key_files": []},
    )
    assert ok is False
    assert "key_files=[]" in reason


def test_when_matches_mixed_task_class_and_touches_passes_when_task_class_matches():
    """Task 3 case (e): `when=[task_class:ui, touches:**/auth/**]`
    on a task with `task_class=ui` (and key_files that DON'T match
    the glob) PASSES via the task_class branch. Verifies task_class
    short-circuits success without reaching touches."""
    ok, reason = _when_matches(
        ("task_class:ui", "touches:**/auth/**"),
        {"id": 1, "task_class": "ui", "key_files": ["src/random.go"]},
    )
    assert ok is True
    assert "task_class='ui'" in reason


def test_when_matches_touches_negative_match_with_unrelated_key_files():
    """Task 3 case (f): a touches glob with no matching key_file
    returns False. Reason names the patterns and the (non-matching)
    key_files for diagnosis."""
    ok, reason = _when_matches(
        ("touches:**/auth/**",),
        {"id": 1, "key_files": ["src/random.go", "tests/test_x.py"]},
    )
    assert ok is False
    assert "touches=['**/auth/**']" in reason
    assert "src/random.go" in reason


def test_when_matches_multiple_touches_or_semantics():
    """Pin OR-of-touches: `[touches:A, touches:B]` matches if
    EITHER pattern matches any key_file. The matched-files list
    in the reason includes only files matched by the spec
    (regardless of which pattern matched them)."""
    ok, reason = _when_matches(
        ("touches:**/auth/**", "touches:**/api/**"),
        {"id": 1, "key_files": ["src/api/users.py"]},
    )
    assert ok is True
    assert "src/api/users.py" in reason


def test_when_matches_falls_through_from_task_class_miss_to_touches_match():
    """Pin §D3: the load-bearing control-flow change. A `when` with
    BOTH `task_class:ui` and `touches:**/auth/**` on a task whose
    task_class is NOT ui but whose key_files ARE under auth/ MUST
    pass via touches. Pre-task-3, the eager `False` return in the
    task_class branch blocked this. Locks the new fall-through."""
    ok, reason = _when_matches(
        ("task_class:ui", "touches:**/auth/**"),
        {"id": 1, "task_class": "schema", "key_files": ["src/auth/api.py"]},
    )
    assert ok is True
    assert "src/auth/api.py" in reason


def test_when_matches_always_wins_over_touches():
    """Pin §D4 rule 1 extended to touches: `('touches:**/x/**', 'always')`
    on a task with no matching key_files still PASSES because
    `always` short-circuits all predicate checks."""
    ok, reason = _when_matches(
        ("touches:**/x/**", "always"),
        {"id": 1, "key_files": ["src/random.go"]},
    )
    assert ok is True
    assert reason == "always"


# ---------------------------------------------------------------------------
# D. ShellGate
# ---------------------------------------------------------------------------

class _CompletedProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _stub_subprocess_run(monkeypatch, *, returncode=0, stdout="", stderr="",
                         raises=None):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        if raises is not None:
            raise raises
        return _CompletedProc(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    return captured


def test_shell_gate_runs_command_and_returns_passed_on_zero_exit(monkeypatch):
    captured = _stub_subprocess_run(monkeypatch, returncode=0, stdout="ok\n")
    g = Gate(kind="shell", command=("echo", "ok"))
    result = ShellGate(name="echo", gate=g).run()
    assert result.passed is True
    assert "ok" in result.output
    assert captured["cmd"] == ["echo", "ok"]


def test_shell_gate_returns_failed_on_nonzero_exit(monkeypatch):
    _stub_subprocess_run(monkeypatch, returncode=1, stderr="boom")
    g = Gate(kind="shell", command=("false",))
    result = ShellGate(name="false", gate=g).run()
    assert result.passed is False
    assert "boom" in result.output


def test_shell_gate_concatenates_stdout_and_stderr(monkeypatch):
    _stub_subprocess_run(monkeypatch, returncode=0, stdout="OUT\n", stderr="ERR\n")
    g = Gate(kind="shell", command=("noop",))
    result = ShellGate(name="x", gate=g).run()
    assert "OUT" in result.output and "ERR" in result.output


def test_shell_gate_passes_timeout_in_seconds(monkeypatch):
    captured = _stub_subprocess_run(monkeypatch)
    g = Gate(kind="shell", command=("true",), timeout_minutes=3)
    ShellGate(name="x", gate=g).run()
    assert captured["kwargs"]["timeout"] == 3 * 60


def test_shell_gate_handles_subprocess_timeout(monkeypatch):
    _stub_subprocess_run(
        monkeypatch,
        raises=subprocess.TimeoutExpired(cmd=["x"], timeout=60),
    )
    g = Gate(kind="shell", command=("sleep", "999"), timeout_minutes=1)
    result = ShellGate(name="slow", gate=g).run()
    assert result.passed is False
    assert result.error == "timeout"
    assert "[timeout after 1 minutes]" in result.output


def test_shell_gate_does_not_use_shell_true(monkeypatch):
    """Pin §D9: subprocess.run is invoked WITHOUT shell=True."""
    captured = _stub_subprocess_run(monkeypatch)
    g = Gate(kind="shell", command=("ls",))
    ShellGate(name="x", gate=g).run()
    assert captured["kwargs"].get("shell", False) is False


def test_shell_gate_raises_when_command_missing():
    g = Gate(kind="shell")  # no command
    with pytest.raises(ValueError, match="requires a command"):
        ShellGate(name="x", gate=g).run()


def test_shell_gate_raises_on_wrong_kind():
    g = Gate(kind="agent", prompt_template="x")
    with pytest.raises(ValueError, match="kind='shell'"):
        ShellGate(name="x", gate=g).run()


def test_shell_gate_carries_on_failure_through_to_result(monkeypatch):
    _stub_subprocess_run(monkeypatch, returncode=1)
    g = Gate(kind="shell", command=("false",), on_failure="warn")
    result = ShellGate(name="x", gate=g).run()
    assert result.on_failure == "warn"


# ---------------------------------------------------------------------------
# E. AgentGate
# ---------------------------------------------------------------------------

def _stub_run_claude(monkeypatch, *, returncode=0, text="VERDICT: PASS\n"):
    captured = {}

    def fake_run_claude(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return returncode, text

    monkeypatch.setattr(
        "claude_plan_execute.gates.agent.run_claude", fake_run_claude,
    )
    return captured


def test_agent_gate_calls_run_claude_with_prompt_template(monkeypatch):
    captured = _stub_run_claude(monkeypatch)
    g = Gate(kind="agent", prompt_template="INLINE GATE BODY")
    task = {"id": 5, "title": "T", "description": "D"}
    AgentGate(name="x", gate=g, task=task, cli={}).run()
    assert "INLINE GATE BODY" in captured["prompt"]


def test_agent_gate_includes_task_header(monkeypatch):
    captured = _stub_run_claude(monkeypatch)
    g = Gate(kind="agent", prompt_template="body")
    task = {"id": 9, "title": "MyTitle", "description": "MyDesc"}
    AgentGate(name="x", gate=g, task=task, cli={}).run()
    assert "TASK 9: MyTitle" in captured["prompt"]
    assert "MyDesc" in captured["prompt"]


def test_agent_gate_loads_template_from_file(monkeypatch, tmp_path):
    captured = _stub_run_claude(monkeypatch)
    template_path = tmp_path / "tmpl.md"
    template_path.write_text("FILE_TEMPLATE_BODY")
    g = Gate(kind="agent", prompt_template=str(template_path))
    task = {"id": 1, "title": "T", "description": "D"}
    AgentGate(name="x", gate=g, task=task, cli={}).run()
    assert "FILE_TEMPLATE_BODY" in captured["prompt"]


def test_agent_gate_uses_inline_body_when_path_does_not_exist(monkeypatch):
    captured = _stub_run_claude(monkeypatch)
    g = Gate(kind="agent", prompt_template="this is not a path")
    task = {"id": 1, "title": "T", "description": "D"}
    AgentGate(name="x", gate=g, task=task, cli={}).run()
    assert "this is not a path" in captured["prompt"]


def test_agent_gate_default_model_is_sonnet(monkeypatch):
    captured = _stub_run_claude(monkeypatch)
    g = Gate(kind="agent", prompt_template="x")
    AgentGate(name="x", gate=g, task={"id": 1, "title": "T", "description": "D"}, cli={}).run()
    assert captured["kwargs"]["model"] == "sonnet"


def test_agent_gate_default_effort_is_medium(monkeypatch):
    captured = _stub_run_claude(monkeypatch)
    g = Gate(kind="agent", prompt_template="x")
    AgentGate(name="x", gate=g, task={"id": 1, "title": "T", "description": "D"}, cli={}).run()
    assert captured["kwargs"]["effort"] == "medium"


def test_agent_gate_overrides_model_from_gate_field(monkeypatch):
    captured = _stub_run_claude(monkeypatch)
    g = Gate(kind="agent", prompt_template="x", model="opus", effort="max")
    AgentGate(name="x", gate=g, task={"id": 1, "title": "T", "description": "D"}, cli={}).run()
    assert captured["kwargs"]["model"] == "opus"
    assert captured["kwargs"]["effort"] == "max"


def test_agent_gate_passes_timeout_minutes_through(monkeypatch):
    captured = _stub_run_claude(monkeypatch)
    g = Gate(kind="agent", prompt_template="x", timeout_minutes=7)
    AgentGate(name="x", gate=g, task={"id": 1, "title": "T", "description": "D"}, cli={}).run()
    assert captured["kwargs"]["timeout_minutes"] == 7


def test_agent_gate_max_turns_is_60(monkeypatch):
    captured = _stub_run_claude(monkeypatch)
    g = Gate(kind="agent", prompt_template="x")
    AgentGate(name="x", gate=g, task={"id": 1, "title": "T", "description": "D"}, cli={}).run()
    assert captured["kwargs"]["max_turns"] == 60


def test_agent_gate_pass_detection_on_VERDICT_PASS(monkeypatch):
    _stub_run_claude(monkeypatch, text="some text\nVERDICT: PASS\nmore text")
    g = Gate(kind="agent", prompt_template="x")
    r = AgentGate(name="x", gate=g, task={"id": 1, "title": "T", "description": "D"}, cli={}).run()
    assert r.passed is True


def test_agent_gate_pass_detection_is_case_insensitive(monkeypatch):
    _stub_run_claude(monkeypatch, text="verdict: pass")
    g = Gate(kind="agent", prompt_template="x")
    r = AgentGate(name="x", gate=g, task={"id": 1, "title": "T", "description": "D"}, cli={}).run()
    assert r.passed is True


def test_agent_gate_fail_detection_on_VERDICT_FAIL(monkeypatch):
    _stub_run_claude(monkeypatch, text="VERDICT: FAIL — found 3 issues")
    g = Gate(kind="agent", prompt_template="x")
    r = AgentGate(name="x", gate=g, task={"id": 1, "title": "T", "description": "D"}, cli={}).run()
    assert r.passed is False


def test_agent_gate_fails_on_empty_response(monkeypatch):
    """Defensive: an empty / malformed agent reply is a fail."""
    _stub_run_claude(monkeypatch, text="")
    g = Gate(kind="agent", prompt_template="x")
    r = AgentGate(name="x", gate=g, task={"id": 1, "title": "T", "description": "D"}, cli={}).run()
    assert r.passed is False


def test_agent_gate_raises_when_prompt_template_missing():
    g = Gate(kind="agent")  # no prompt_template
    with pytest.raises(ValueError, match="requires a prompt_template"):
        AgentGate(name="x", gate=g, task={"id": 1, "title": "T", "description": "D"}, cli={}).run()


def test_agent_gate_raises_on_wrong_kind():
    g = Gate(kind="shell", command=("true",))
    with pytest.raises(ValueError, match="kind='agent'"):
        AgentGate(name="x", gate=g, task={"id": 1, "title": "T", "description": "D"}, cli={}).run()


def test_agent_gate_forwards_skip_permissions_from_cli(monkeypatch):
    """Pin task 15 §D9: AgentGate forwards `skip_permissions` from
    its `cli` dict to run_claude. Without this, agent-gate bash
    tool use auto-denies in autonomous mode (--skip-permissions)
    and gates that need bash (test-critique) cannot fulfill their
    contract."""
    captured = _stub_run_claude(monkeypatch)
    g = Gate(kind="agent", prompt_template="x")
    AgentGate(
        name="x", gate=g,
        task={"id": 1, "title": "T", "description": "D"},
        cli={"skip_permissions": True},
    ).run()
    assert captured["kwargs"]["skip_permissions"] is True


def test_agent_gate_forwards_verbose_from_cli(monkeypatch):
    """Pin task 15 §D9: AgentGate forwards `verbose` from its
    `cli` dict to run_claude. Mirrors the implementer's behavior
    at runner.py:121."""
    captured = _stub_run_claude(monkeypatch)
    g = Gate(kind="agent", prompt_template="x")
    AgentGate(
        name="x", gate=g,
        task={"id": 1, "title": "T", "description": "D"},
        cli={"verbose": True},
    ).run()
    assert captured["kwargs"]["verbose"] is True


def test_agent_gate_skip_permissions_defaults_to_False(monkeypatch):
    """Pin task 15 §D9: an empty cli dict produces
    skip_permissions=False (preserves pre-task-15 default for
    callers who don't care)."""
    captured = _stub_run_claude(monkeypatch)
    g = Gate(kind="agent", prompt_template="x")
    AgentGate(
        name="x", gate=g,
        task={"id": 1, "title": "T", "description": "D"},
        cli={},
    ).run()
    assert captured["kwargs"]["skip_permissions"] is False
    assert captured["kwargs"]["verbose"] is False


def test_agent_gate_substitutes_diff_path_in_inline_template(monkeypatch):
    """Pin T7 §D3 / §D7: AgentGate.diff_path is substituted into
    `{diff_path}` placeholders via .replace() on the loaded body.
    Inline-body case — the substitution happens after _load_template
    falls through (path-too-long → OSError → verbatim return)."""
    captured = _stub_run_claude(monkeypatch)
    g = Gate(kind="agent",
             prompt_template="The diff is at: {diff_path}\nDone.")
    AgentGate(
        name="x", gate=g,
        task={"id": 1, "title": "T", "description": "D"},
        cli={}, diff_path="/tmp/foo/diff.patch",
    ).run()
    assert "The diff is at: /tmp/foo/diff.patch" in captured["prompt"]
    assert "{diff_path}" not in captured["prompt"]


def test_agent_gate_substitutes_diff_path_in_file_template(
    monkeypatch, tmp_path,
):
    """Pin T7 §D3: file-loaded body case — _load_template reads the
    file, AgentGate.run() substitutes {diff_path} on the returned
    body. Identical user-facing behavior to inline-body case."""
    captured = _stub_run_claude(monkeypatch)
    template_path = tmp_path / "tmpl.md"
    template_path.write_text("Diff at: {diff_path}")
    g = Gate(kind="agent", prompt_template=str(template_path))
    AgentGate(
        name="x", gate=g,
        task={"id": 1, "title": "T", "description": "D"},
        cli={}, diff_path="/path/to/diff.patch",
    ).run()
    assert "Diff at: /path/to/diff.patch" in captured["prompt"]
    assert "{diff_path}" not in captured["prompt"]


def test_agent_gate_diff_path_None_substitutes_with_empty_string(monkeypatch):
    """Pin T7 §D3: `diff_path=None` (the default) replaces
    `{diff_path}` with the empty string, NOT the literal token.
    Avoids leaking template internals to the agent when callers
    don't supply a diff_path (existing tests, future direct
    callers without runner context)."""
    captured = _stub_run_claude(monkeypatch)
    g = Gate(kind="agent", prompt_template="path={diff_path}|end")
    AgentGate(
        name="x", gate=g,
        task={"id": 1, "title": "T", "description": "D"},
        cli={},  # diff_path defaults to None
    ).run()
    assert "path=|end" in captured["prompt"]
    assert "{diff_path}" not in captured["prompt"]


def test_agent_gate_diff_path_default_is_None():
    """Pin T7 §D7: the new `diff_path` field defaults to None so
    every existing test call site (16 of them in this module)
    keeps working without modification."""
    g = Gate(kind="agent", prompt_template="x")
    ag = AgentGate(
        name="x", gate=g,
        task={"id": 1, "title": "T", "description": "D"},
        cli={},
    )
    assert ag.diff_path is None


# ---------------------------------------------------------------------------
# F. _tail_cap helper
# ---------------------------------------------------------------------------

def test_tail_cap_passes_through_short_text():
    text = "short"
    assert _tail_cap(text) == text


def test_tail_cap_passes_through_at_exact_cap_size():
    text = "X" * _OUTPUT_CAP_BYTES  # ASCII → 1 byte/char
    assert _tail_cap(text) == text
    assert "[truncated" not in _tail_cap(text)


def test_tail_cap_truncates_over_cap():
    text = "Y" * (_OUTPUT_CAP_BYTES + 5000)
    out = _tail_cap(text)
    encoded = out.encode("utf-8")
    # Cap + marker overhead < ~50 bytes
    assert len(encoded) <= _OUTPUT_CAP_BYTES + 50
    assert "[truncated:" in out


def test_tail_cap_keeps_tail_not_head():
    """Pin §D8: tail-truncate; the LAST characters survive."""
    head = "HEAD" * 30000  # plenty long
    body = "TAILMARKER_AT_END"
    text = head + body
    out = _tail_cap(text)
    assert "TAILMARKER_AT_END" in out
    # The head portion should be truncated.
    assert out.count("HEAD") < text.count("HEAD")


def test_tail_cap_respects_explicit_cap_kwarg():
    """Pin T7 §D5: `_tail_cap` accepts a `cap=` kwarg that
    overrides the default `_OUTPUT_CAP_BYTES`. The runner uses
    this with `cap=_DIFF_CAP_BYTES` (256 KB) for the
    precomputed-diff artifact; gate output keeps the 100 KB
    default. Regression guard for the refactor: the kwarg path
    must produce a body bounded by the new cap, not the old
    constant."""
    cap = 1024  # 1 KB — small for a fast assertion
    text = "Z" * 5000  # 5 KB — well over the 1 KB cap
    out = _tail_cap(text, cap=cap)
    encoded = out.encode("utf-8")
    # Cap + marker overhead (~50 bytes for "[truncated: <N>
    # bytes omitted at head]\n") is the budget.
    assert len(encoded) <= cap + 50
    assert "[truncated:" in out


# ---------------------------------------------------------------------------
# G. run_gates orchestration
# ---------------------------------------------------------------------------

def _stub_subprocess_run_global(monkeypatch, *, returncode=0,
                                stdout="ok\n", stderr=""):
    """Stub subprocess.run on the gates.shell module path. Returns
    a list of captured commands so tests can assert each gate's
    invocation."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return _CompletedProc(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    return calls


def test_run_gates_with_empty_list_returns_empty(tmp_path):
    """Edge: zero gates → empty list, no errors."""
    results = run_gates(
        [],
        task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path / "task-1",
        cli={},
    )
    assert results == []


def test_run_gates_creates_gates_subdirectory(monkeypatch, tmp_path):
    _stub_subprocess_run_global(monkeypatch)
    g = Gate(kind="shell", command=("true",))
    run_gates(
        [("a", g)], task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    assert (tmp_path / "gates").is_dir()


def test_run_gates_writes_output_file_for_each_gate(monkeypatch, tmp_path):
    _stub_subprocess_run_global(monkeypatch, stdout="alpha-out", stderr="")
    g = Gate(kind="shell", command=("true",))
    run_gates(
        [("alpha", g), ("beta", g)],
        task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    assert (tmp_path / "gates" / "alpha.txt").is_file()
    assert (tmp_path / "gates" / "beta.txt").is_file()


def test_run_gates_output_file_contains_gate_output(monkeypatch, tmp_path):
    _stub_subprocess_run_global(monkeypatch, stdout="MARKER123")
    g = Gate(kind="shell", command=("true",))
    run_gates(
        [("g", g)], task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    body = (tmp_path / "gates" / "g.txt").read_text()
    assert "MARKER123" in body


def test_run_gates_sets_output_path_on_each_result(monkeypatch, tmp_path):
    _stub_subprocess_run_global(monkeypatch)
    g = Gate(kind="shell", command=("true",))
    results = run_gates(
        [("g", g)], task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    assert results[0].output_path is not None
    assert results[0].output_path.endswith("gates/g.txt")


def test_run_gates_returns_one_result_per_input_gate(monkeypatch, tmp_path):
    _stub_subprocess_run_global(monkeypatch)
    g = Gate(kind="shell", command=("true",))
    results = run_gates(
        [("a", g), ("b", g), ("c", g)],
        task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    names = sorted(r.name for r in results)
    assert names == ["a", "b", "c"]


def test_run_gates_serial_then_parallel(monkeypatch, tmp_path):
    """Pin §D5: serial gates run in input order BEFORE parallel."""
    invocation_log: list[str] = []
    real_completed_proc = _CompletedProc()

    def fake_run(cmd, **kwargs):
        # The "command" tuple is the gate name (we set it that way below).
        invocation_log.append(cmd[0])
        return real_completed_proc

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    serial_g = Gate(kind="shell", command=("S",), parallel_safe=False)
    parallel_g = Gate(kind="shell", command=("P",), parallel_safe=True)
    run_gates(
        [("p1", parallel_g), ("s1", serial_g), ("p2", parallel_g)],
        task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    # Serial gate fires before any parallel ones (order-of-completion in
    # parallel_log isn't deterministic, but the first invocation is "S").
    assert invocation_log[0] == "S"


def test_run_gates_parallel_gates_can_run_concurrently(monkeypatch, tmp_path):
    """Soft pin: with three parallel gates, all three subprocess.run
    calls happen. The exact ordering is non-deterministic; we only
    assert all three were invoked."""
    invocations: list[str] = []

    def fake_run(cmd, **kwargs):
        invocations.append(cmd[0])
        return _CompletedProc()

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    g1 = Gate(kind="shell", command=("a",), parallel_safe=True)
    g2 = Gate(kind="shell", command=("b",), parallel_safe=True)
    g3 = Gate(kind="shell", command=("c",), parallel_safe=True)
    run_gates(
        [("g1", g1), ("g2", g2), ("g3", g3)],
        task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    assert sorted(invocations) == ["a", "b", "c"]


def test_run_gates_when_filter_skips_with_passive_result(monkeypatch, tmp_path):
    """Pin task 20: a gate with when=('task_class:ui',) is skipped when
    the task has no matching task_class. Task 20 implemented the matcher;
    the test body's task fixture (no task_class set) still produces a
    skip, exercising the negative path."""
    g = Gate(kind="shell", command=("true",), when=("task_class:ui",))
    results = run_gates(
        [("ui-only", g)],
        task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    assert len(results) == 1
    assert results[0].passed is True
    assert "[skipped:" in results[0].output


def test_run_gates_skip_writes_output_file(monkeypatch, tmp_path):
    """Pin §D16 (task 11): skipped gates DO write output files for
    audit. Test fixture uses task_class:ui filter against a task
    with no task_class — skip path exercised."""
    g = Gate(kind="shell", command=("true",), when=("task_class:ui",))
    run_gates(
        [("ui-only", g)],
        task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    body = (tmp_path / "gates" / "ui-only.txt").read_text()
    assert "[skipped:" in body


def test_run_gates_output_tail_capped(monkeypatch, tmp_path):
    """Pin §D8: gate output written to file is tail-capped at 100 KB."""
    big_stdout = "X" * (_OUTPUT_CAP_BYTES + 10000)
    _stub_subprocess_run_global(monkeypatch, stdout=big_stdout)
    g = Gate(kind="shell", command=("true",))
    run_gates(
        [("big", g)],
        task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    body = (tmp_path / "gates" / "big.txt").read_text()
    assert len(body.encode("utf-8")) <= _OUTPUT_CAP_BYTES + 50


def test_run_gates_carries_on_failure_through(monkeypatch, tmp_path):
    _stub_subprocess_run_global(monkeypatch, returncode=1)
    g_block = Gate(kind="shell", command=("false",), on_failure="block")
    g_warn = Gate(kind="shell", command=("false",), on_failure="warn")
    results = run_gates(
        [("b", g_block), ("w", g_warn)],
        task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    by_name = {r.name: r for r in results}
    assert by_name["b"].on_failure == "block"
    assert by_name["w"].on_failure == "warn"


def test_run_gates_exception_in_gate_yields_failed_result(monkeypatch, tmp_path):
    """A gate that raises is reported as passed=False with error set."""
    def boom(cmd, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", boom,
    )
    g = Gate(kind="shell", command=("true",))
    results = run_gates(
        [("crashy", g)],
        task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    assert results[0].passed is False
    assert "disk full" in (results[0].error or "")


def test_run_gates_passes_diff_path_into_agent_gate(monkeypatch, tmp_path):
    """Pin T7 §D7: run_gates computes
    `str(task_dir / 'diff.patch')` and threads it to AgentGate
    via _run_one. The agent's prompt body has the substituted
    path."""
    captured: list[tuple[str, dict]] = []

    def fake_run_claude(prompt, **kwargs):
        captured.append((prompt, kwargs))
        return 0, "VERDICT: PASS\n"

    monkeypatch.setattr(
        "claude_plan_execute.gates.agent.run_claude", fake_run_claude,
    )
    g = Gate(kind="agent",
             prompt_template="diff_path is {diff_path}")
    run_gates(
        [("agent-gate", g)],
        task={"id": 7, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    assert len(captured) == 1
    expected = str(tmp_path / "diff.patch")
    assert f"diff_path is {expected}" in captured[0][0]


def test_run_gates_diff_path_threading_does_NOT_affect_shell_gates(
    monkeypatch, tmp_path,
):
    """Pin T7 §D8: ShellGate has no template substitution surface.
    Threading diff_path through _run_one is a no-op for shell-kind
    gates. The shell gate runs unchanged; no error from the
    diff_path kwarg presence."""
    _stub_subprocess_run_global(monkeypatch, stdout="ok\n")
    g = Gate(kind="shell", command=("true",))
    results = run_gates(
        [("shell-only", g)],
        task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    assert len(results) == 1
    assert results[0].passed is True


# ---------------------------------------------------------------------------
# H. Misc
# ---------------------------------------------------------------------------

def test_output_cap_constant_is_100_kb():
    """Pin §D8: 100 KB cap matches enhancement plan §2.1."""
    assert _OUTPUT_CAP_BYTES == 100 * 1024


# ---------------------------------------------------------------------------
# I. Findings extraction & AgentGate findings-population (T11, v0.3)
# ---------------------------------------------------------------------------
#
# Pure-helper tests for `_extract_findings`: the seven cases (a)-(g)
# enumerated in the task description, plus corner cases pinning the
# regex semantics, the case-5 `_raw` cap, and the warn-suppression
# kwarg. AgentGate findings-population tests follow the helper unit
# tests (same Section I).
#
# Markers are HTML comments; see plan §D2 / gates/agent.py:_FINDINGS_RE.
# ---------------------------------------------------------------------------

def _wrap_findings(body: str) -> str:
    """Helper: wrap a JSON body in the appendix marker pair."""
    return (
        "VERDICT: PASS\nSome prose.\n"
        "<!-- BEGIN_FINDINGS_JSON -->\n"
        f"{body}\n"
        "<!-- END_FINDINGS_JSON -->\n"
    )


# --- (a) well-formed array populates findings -------------------------------

def test_extract_findings_well_formed_array_populates_findings(capsys):
    """Task description test (a): markers + a well-formed JSON list of
    dicts → findings is the parsed list; sidecar wraps it under
    `findings:` key. No warning printed."""
    text = _wrap_findings('[{"k": "v"}, {"k": "w"}]')
    findings, sidecar = _extract_findings(text, "g")
    assert findings == [{"k": "v"}, {"k": "w"}]
    assert sidecar == {"findings": [{"k": "v"}, {"k": "w"}]}
    out = capsys.readouterr().out
    assert "malformed" not in out
    assert "not a list[dict]" not in out


# --- (b) bare array normalizes to {findings: [...]} -------------------------

def test_extract_findings_bare_array_normalizes_to_findings_object(capsys):
    """Task description test (b): bare JSON list inside markers
    normalizes to `{"findings": [...]}` in the sidecar — NOT the raw
    list. The dataclass `findings` is the bare list; the sidecar
    wraps."""
    text = _wrap_findings('[{"k": "v"}, {"k": "w"}]')
    findings, sidecar = _extract_findings(text, "g")
    assert isinstance(findings, list)
    assert isinstance(sidecar, dict)
    assert sidecar.get("findings") == findings
    assert "findings" in sidecar


# --- (c) malformed JSON warns, returns None, no sidecar ---------------------

def test_extract_findings_malformed_json_warns_returns_None_no_sidecar(capsys):
    """Task description test (c): markers present but body fails
    json.loads → findings is None, sidecar is None, yellow warning
    printed. Gate verdict is unaffected (caller's `passed` decision
    derives only from `VERDICT: PASS` substring search)."""
    text = _wrap_findings("{this is not valid json}")
    findings, sidecar = _extract_findings(text, "secaudit")
    assert findings is None
    assert sidecar is None
    out = capsys.readouterr().out
    assert "malformed findings JSON" in out
    assert "secaudit" in out
    assert "verdict unchanged" in out


# --- (d) no markers → None, None, no warning --------------------------------

def test_extract_findings_no_markers_returns_None_no_warning(capsys):
    """Task description test (d): plain text without marker pair →
    findings is None, sidecar is None, no warning."""
    findings, sidecar = _extract_findings("VERDICT: PASS\nNothing here.", "g")
    assert findings is None
    assert sidecar is None
    out = capsys.readouterr().out
    assert "malformed" not in out
    assert "not a list[dict]" not in out


# --- (g) list with non-dict elements → empty + raw + warning ----------------

def test_extract_findings_list_with_non_dict_elements_returns_empty_with_raw(capsys):
    """Task description test (g): a list whose elements are not all
    dicts → strict reject. findings == [] (NOT a partial list);
    sidecar has `_raw` for debugging. Warning printed."""
    text = _wrap_findings('[{"a": 1}, "bad", {"b": 2}]')
    findings, sidecar = _extract_findings(text, "g")
    assert findings == []
    assert sidecar is not None
    assert sidecar["findings"] == []
    assert "_raw" in sidecar
    assert "bad" in sidecar["_raw"]
    out = capsys.readouterr().out
    assert "not a list[dict]" in out
    assert "treating as empty" in out


# --- additional helper-level tests -----------------------------------------

def test_extract_findings_object_with_findings_list_returns_inner_list(capsys):
    """Object form: `{"findings": [...]}` → findings is the inner list,
    sidecar is the full object (with the `findings` key preserved)."""
    text = _wrap_findings('{"findings": [{"a": 1}, {"b": 2}]}')
    findings, sidecar = _extract_findings(text, "g")
    assert findings == [{"a": 1}, {"b": 2}]
    assert sidecar == {"findings": [{"a": 1}, {"b": 2}]}
    out = capsys.readouterr().out
    assert "malformed" not in out
    assert "not a list[dict]" not in out


def test_extract_findings_object_with_findings_list_preserves_extra_keys_in_sidecar(capsys):
    """Pin §D1 case 4: an object with `findings: [...]` AND extra keys
    (e.g., `summary`) preserves the extra keys verbatim in the sidecar.
    Templates own their inner shape; the build tool is gate-agnostic."""
    text = _wrap_findings('{"findings": [{"a": 1}], "summary": "1 critical"}')
    findings, sidecar = _extract_findings(text, "g")
    assert findings == [{"a": 1}]
    assert sidecar == {"findings": [{"a": 1}], "summary": "1 critical"}


def test_extract_findings_scalar_returns_empty_with_raw(capsys):
    """Pin §D1 case 5 (scalar): an integer between markers → findings is
    [], sidecar has `_raw` with the raw body. Warning printed."""
    text = _wrap_findings("42")
    findings, sidecar = _extract_findings(text, "g")
    assert findings == []
    assert sidecar == {"findings": [], "_raw": "42"}
    out = capsys.readouterr().out
    assert "not a list[dict]" in out


def test_extract_findings_null_returns_empty_with_raw(capsys):
    """Pin §D1 case 5 (null): JSON `null` → findings is [], sidecar has
    `_raw == "null"`. Warning printed."""
    text = _wrap_findings("null")
    findings, sidecar = _extract_findings(text, "g")
    assert findings == []
    assert sidecar == {"findings": [], "_raw": "null"}


def test_extract_findings_object_without_findings_key_returns_empty_with_raw(capsys):
    """Pin §D1 case 5 (object lacking `findings` key): `{"summary":
    "ok"}` → findings is [], sidecar has `_raw` with the raw body
    JSON. Warning printed. Templates that omit the `findings` key
    are surfaced as malformed via the warning."""
    text = _wrap_findings('{"summary": "ok"}')
    findings, sidecar = _extract_findings(text, "g")
    assert findings == []
    assert sidecar is not None
    assert sidecar["findings"] == []
    assert sidecar["_raw"] == '{"summary": "ok"}'


def test_extract_findings_object_with_non_list_findings_returns_empty_with_raw(capsys):
    """Pin §D1 case 5 (object with non-list `findings`): `{"findings":
    "not a list"}` → findings is [], sidecar has `_raw`. Warning."""
    text = _wrap_findings('{"findings": "not a list"}')
    findings, sidecar = _extract_findings(text, "g")
    assert findings == []
    assert sidecar is not None
    assert sidecar["findings"] == []
    assert "not a list" in sidecar["_raw"]


def test_extract_findings_object_with_findings_list_with_non_dict_returns_empty_with_raw(capsys):
    """Pin §D1 case 5 (object.findings list with a non-dict element):
    strict reject at the inner-list layer too. Mirrors test (g) but
    nested under the object form."""
    text = _wrap_findings('{"findings": [{"a": 1}, "bad"]}')
    findings, sidecar = _extract_findings(text, "g")
    assert findings == []
    assert sidecar is not None
    assert sidecar["findings"] == []
    assert "bad" in sidecar["_raw"]


def test_extract_findings_first_appendix_wins(capsys):
    """Pin §D2: `re.search` returns the FIRST match. With two
    well-formed appendices in the text, the first body wins."""
    text = (
        "header\n"
        "<!-- BEGIN_FINDINGS_JSON -->\n"
        '[{"order": "first"}]\n'
        "<!-- END_FINDINGS_JSON -->\n"
        "filler\n"
        "<!-- BEGIN_FINDINGS_JSON -->\n"
        '[{"order": "second"}]\n'
        "<!-- END_FINDINGS_JSON -->\n"
    )
    findings, _ = _extract_findings(text, "g")
    assert findings == [{"order": "first"}]


def test_extract_findings_non_greedy_does_not_swallow_intermediate_markers(capsys):
    """Pin §A: non-greedy `.*?` ensures BEGIN binds to the NEAREST END,
    not the LAST END. With two marker pairs, the first BEGIN must
    bind to the first END (not swallow the second BEGIN)."""
    text = (
        "<!-- BEGIN_FINDINGS_JSON -->\n"
        '[{"first": 1}]\n'
        "<!-- END_FINDINGS_JSON -->\n"
        "interleaved text\n"
        "<!-- BEGIN_FINDINGS_JSON -->\n"
        '[{"second": 2}]\n'
        "<!-- END_FINDINGS_JSON -->\n"
    )
    match = _FINDINGS_RE.search(text)
    assert match is not None
    body = match.group(1)
    # Body is the FIRST appendix's JSON, not the swallowed-everything
    # case where body would include the literal END marker.
    assert "first" in body
    assert "BEGIN_FINDINGS_JSON" not in body
    assert "END_FINDINGS_JSON" not in body


def test_extract_findings_regex_dotall_matches_across_newlines(capsys):
    """Pin §A: `re.DOTALL` makes `.` match newlines so a multi-line
    JSON body is captured intact."""
    text = (
        "<!-- BEGIN_FINDINGS_JSON -->\n"
        "[\n"
        '  {"k": 1},\n'
        '  {"k": 2}\n'
        "]\n"
        "<!-- END_FINDINGS_JSON -->\n"
    )
    findings, _ = _extract_findings(text, "g")
    assert findings == [{"k": 1}, {"k": 2}]


def test_extract_findings_strips_whitespace_around_body(capsys):
    """Pin §D2: the regex's `\\s*` flanks strip surrounding whitespace
    (newlines, spaces) so `json.loads` sees the raw JSON. An empty
    list with extra whitespace inside the markers parses cleanly."""
    text = (
        "<!-- BEGIN_FINDINGS_JSON -->  \n  []  \n  <!-- END_FINDINGS_JSON -->"
    )
    findings, sidecar = _extract_findings(text, "g")
    assert findings == []
    assert sidecar == {"findings": []}


def test_extract_findings_empty_array_is_valid(capsys):
    """Pin: an explicit empty list is case 3 (clean parse), NOT case 5
    (malformed). Distinguishes intentional zero findings from absent
    appendix or malformed body. NO warning."""
    text = _wrap_findings("[]")
    findings, sidecar = _extract_findings(text, "g")
    assert findings == []
    assert sidecar == {"findings": []}
    out = capsys.readouterr().out
    assert "malformed" not in out
    assert "not a list[dict]" not in out


def test_extract_findings_warn_false_silences_warnings(capsys):
    """Pin §D5 / §6.2: `warn=False` suppresses the case-2 / case-5
    warnings. `run_gates` uses `warn=False` on its second extraction
    to avoid double-printing what AgentGate already printed."""
    text = _wrap_findings("{not valid json}")
    findings, sidecar = _extract_findings(text, "g", warn=False)
    assert findings is None
    assert sidecar is None
    out = capsys.readouterr().out
    assert "malformed findings JSON" not in out


def test_extract_findings_raw_text_is_capped_at_raw_cap_bytes(capsys):
    """Pin §D8: `_RAW_CAP_BYTES = 16 KB` defensive cap on the case-5
    `_raw` text. A 50KB JSON string scalar produces an oversized body
    that gets tail-capped before stuffing into `_raw`. Read the
    constant rather than hard-coding 16 KB so the test stays in sync
    if the cap is ever tuned."""
    big_string = "X" * 50000
    body = json.dumps(big_string)  # JSON-encoded string scalar
    text = (
        "<!-- BEGIN_FINDINGS_JSON -->\n"
        f"{body}\n"
        "<!-- END_FINDINGS_JSON -->\n"
    )
    findings, sidecar = _extract_findings(text, "g")
    assert findings == []
    assert sidecar is not None
    raw_bytes = sidecar["_raw"].encode("utf-8")
    # Cap + ~50-byte truncation marker overhead.
    assert len(raw_bytes) <= _RAW_CAP_BYTES + 50
    assert "[truncated:" in sidecar["_raw"]


def test_raw_cap_constant_is_16_kb():
    """Pin §D8: the cap is 16 KB."""
    assert _RAW_CAP_BYTES == 16 * 1024


# --- AgentGate findings-population (Section I sub-block) -------------------

def test_agent_gate_populates_findings_when_appendix_present(monkeypatch):
    """Pin §D5 site 1: AgentGate.run() threads the extracted findings
    onto the returned GateResult."""
    text = _wrap_findings('[{"category": "ACL", "severity": "high"}]')
    _stub_run_claude(monkeypatch, text=text)
    g = Gate(kind="agent", prompt_template="x")
    r = AgentGate(name="secaudit", gate=g,
                  task={"id": 1, "title": "T", "description": "D"},
                  cli={}).run()
    assert r.findings == [{"category": "ACL", "severity": "high"}]


def test_agent_gate_findings_is_None_when_no_appendix(monkeypatch):
    """Pin §D1 case 1 + AgentGate wiring: text without markers →
    findings is None on the result."""
    _stub_run_claude(monkeypatch, text="VERDICT: PASS\nNo appendix here.\n")
    g = Gate(kind="agent", prompt_template="x")
    r = AgentGate(name="g", gate=g,
                  task={"id": 1, "title": "T", "description": "D"},
                  cli={}).run()
    assert r.findings is None


def test_agent_gate_findings_is_None_on_malformed_appendix(monkeypatch, capsys):
    """Pin §D1 case 2 + AgentGate wiring: malformed JSON → findings is
    None; warning printed."""
    _stub_run_claude(
        monkeypatch,
        text=_wrap_findings("{this is not json}"),
    )
    g = Gate(kind="agent", prompt_template="x")
    r = AgentGate(name="g", gate=g,
                  task={"id": 1, "title": "T", "description": "D"},
                  cli={}).run()
    assert r.findings is None
    out = capsys.readouterr().out
    assert "malformed findings JSON" in out


def test_agent_gate_findings_default_value_in_dataclass_is_None(monkeypatch):
    """Pin §D4: empty agent text yields `findings is None` (the
    helper's case-1 return value flows onto the dataclass)."""
    _stub_run_claude(monkeypatch, text="")
    g = Gate(kind="agent", prompt_template="x")
    r = AgentGate(name="g", gate=g,
                  task={"id": 1, "title": "T", "description": "D"},
                  cli={}).run()
    assert r.findings is None


def test_agent_gate_passed_is_independent_of_findings_extraction_failure(
    monkeypatch, capsys,
):
    """Pin §D7: a malformed appendix on a PASS verdict does NOT block
    the gate. `passed` derives only from `verdict: pass` substring;
    `findings` extraction failure surfaces via warning + None on the
    dataclass field."""
    _stub_run_claude(
        monkeypatch,
        text=(
            "VERDICT: PASS\n"
            "<!-- BEGIN_FINDINGS_JSON -->\ngarbage\n<!-- END_FINDINGS_JSON -->"
        ),
    )
    g = Gate(kind="agent", prompt_template="x")
    r = AgentGate(name="g", gate=g,
                  task={"id": 1, "title": "T", "description": "D"},
                  cli={}).run()
    assert r.passed is True
    assert r.findings is None
    out = capsys.readouterr().out
    assert "malformed findings JSON" in out


def test_agent_gate_uses_capped_output_for_extraction(monkeypatch):
    """Pin §6.1: AgentGate passes `output_capped` (post-tail-cap)
    to `_extract_findings`, NOT the raw text. If the appendix lands
    in the omitted-head portion of an oversized text, both AgentGate's
    extraction AND run_gates' re-extraction return (None, None) —
    they agree because they share the same input."""
    # Layout: appendix appears at the head; ~200 KB of filler tail
    # pushes the head past the 100 KB tail-cap window.
    appendix = (
        "<!-- BEGIN_FINDINGS_JSON -->\n[{\"k\": 1}]\n"
        "<!-- END_FINDINGS_JSON -->\n"
    )
    filler = "Z" * (_OUTPUT_CAP_BYTES + 50_000)
    text = appendix + filler
    _stub_run_claude(monkeypatch, text=text)
    g = Gate(kind="agent", prompt_template="x")
    r = AgentGate(name="g", gate=g,
                  task={"id": 1, "title": "T", "description": "D"},
                  cli={}).run()
    # Markers were truncated out of `r.output` (the post-cap text);
    # extraction sees no markers → findings is None.
    assert "<!-- BEGIN_FINDINGS_JSON -->" not in r.output
    assert r.findings is None


# ---------------------------------------------------------------------------
# J. run_gates sidecar `.json` write (T11, v0.3)
# ---------------------------------------------------------------------------
#
# Integration tests at the run_gates layer. The .json file is written
# next to <name>.txt under task_dir/gates/ when the agent populated
# `findings`. Shell gates and skip-path results never trigger a write.
# ---------------------------------------------------------------------------

def test_run_gates_writes_json_sidecar_when_findings_present(
    monkeypatch, tmp_path,
):
    """Pin task description test (a) at integration: a passing agent
    gate with a well-formed appendix produces both <name>.txt and
    <name>.json. The .json content parses cleanly and contains the
    expected findings list."""
    text = _wrap_findings('[{"category": "ACL", "severity": "high"}]')
    _stub_run_claude(monkeypatch, text=text)
    g = Gate(kind="agent", prompt_template="x")
    run_gates(
        [("secaudit", g)],
        task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    json_path = tmp_path / "gates" / "secaudit.json"
    assert json_path.is_file()
    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    assert sidecar["findings"] == [{"category": "ACL", "severity": "high"}]


def test_run_gates_json_sidecar_for_object_with_findings_keeps_extra_keys(
    monkeypatch, tmp_path,
):
    """Pin §D1 case 4: object form with extra keys (e.g., `summary`)
    keeps every key in the written sidecar. Belt-and-suspenders for
    aggregator forward-compat."""
    text = _wrap_findings('{"findings": [{"a": 1}], "summary": "S"}')
    _stub_run_claude(monkeypatch, text=text)
    g = Gate(kind="agent", prompt_template="x")
    run_gates(
        [("g", g)], task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    sidecar = json.loads((tmp_path / "gates" / "g.json").read_text())
    assert sidecar["findings"] == [{"a": 1}]
    assert sidecar["summary"] == "S"


def test_run_gates_json_sidecar_for_malformed_with_raw(
    monkeypatch, tmp_path, capsys,
):
    """Pin §D1 case 5 at integration: a list with a non-dict element
    produces a sidecar with `findings: []` and `_raw`."""
    text = _wrap_findings('[{"a": 1}, "bad"]')
    _stub_run_claude(monkeypatch, text=text)
    g = Gate(kind="agent", prompt_template="x")
    run_gates(
        [("g", g)], task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    sidecar = json.loads((tmp_path / "gates" / "g.json").read_text())
    assert sidecar["findings"] == []
    assert "bad" in sidecar["_raw"]
    out = capsys.readouterr().out
    assert "not a list[dict]" in out


def test_run_gates_does_NOT_write_json_when_no_markers(monkeypatch, tmp_path):
    """Pin task description test (d): no markers → no .json file. The
    .txt file IS written (gate output is always captured)."""
    _stub_run_claude(monkeypatch, text="VERDICT: PASS\nNo findings here.")
    g = Gate(kind="agent", prompt_template="x")
    run_gates(
        [("g", g)], task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    assert (tmp_path / "gates" / "g.txt").is_file()
    assert not (tmp_path / "gates" / "g.json").exists()


def test_run_gates_does_NOT_write_json_on_malformed_parse(
    monkeypatch, tmp_path, capsys,
):
    """Pin task description test (c) at integration: malformed JSON
    inside markers → no .json file (case 2 is asymmetric vs. case 5).
    Gate verdict (PASS) is unchanged. Warning printed."""
    _stub_run_claude(
        monkeypatch,
        text=_wrap_findings("garbage"),
    )
    g = Gate(kind="agent", prompt_template="x")
    results = run_gates(
        [("g", g)], task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    assert results[0].passed is True
    assert not (tmp_path / "gates" / "g.json").exists()
    out = capsys.readouterr().out
    assert "malformed findings JSON" in out


def test_run_gates_does_NOT_write_json_for_shell_gate(monkeypatch, tmp_path):
    """Pin task description test (e): shell gates NEVER produce a .json
    sidecar even if the captured stdout literally contains valid
    findings markers. The contract is "shell gates don't extract"
    (predicate: `r.findings is not None`), NOT "shell stdout was
    empty". A future regression that switched the predicate to scan
    `r.output` for markers would pass with empty stdout but FAIL
    here."""
    stdout_with_markers = (
        "shell ran\n"
        "<!-- BEGIN_FINDINGS_JSON -->\n[{\"k\": \"v\"}]\n"
        "<!-- END_FINDINGS_JSON -->\n"
    )
    _stub_subprocess_run_global(monkeypatch, stdout=stdout_with_markers)
    g = Gate(kind="shell", command=("true",))
    run_gates(
        [("shellgate", g)],
        task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    txt_body = (tmp_path / "gates" / "shellgate.txt").read_text()
    assert "BEGIN_FINDINGS_JSON" in txt_body  # markers in stdout
    assert not (tmp_path / "gates" / "shellgate.json").exists()


def test_run_gates_findings_survive_reconstruction(monkeypatch, tmp_path):
    """Pin task description test (f): the GateResult re-construction
    in run_gates' write loop MUST thread `findings=r.findings`
    explicitly. Without that line, the returned result silently drops
    findings (default-None applies). This is the load-bearing
    regression guard for the `findings=r.findings` line in §6.2 step
    4. Production would be quietly broken; only this test catches
    it."""
    text = _wrap_findings('[{"a": 1}]')
    _stub_run_claude(monkeypatch, text=text)
    g = Gate(kind="agent", prompt_template="x")
    results = run_gates(
        [("g", g)], task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    assert results[0].findings == [{"a": 1}]


def test_run_gates_json_pretty_printed_with_indent_2(monkeypatch, tmp_path):
    """Pin §D6: sidecar JSON uses `json.dumps(..., indent=2)` so the
    file is human-readable. Verify by checking for indented braces /
    keys in the raw text."""
    text = _wrap_findings('[{"k": "v"}]')
    _stub_run_claude(monkeypatch, text=text)
    g = Gate(kind="agent", prompt_template="x")
    run_gates(
        [("g", g)], task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    raw = (tmp_path / "gates" / "g.json").read_text(encoding="utf-8")
    # Indent=2 produces lines beginning with two spaces inside
    # objects; key/value pairs land on their own lines.
    assert "  \"findings\"" in raw
    assert "\n" in raw  # multi-line


def test_run_gates_json_sidecar_has_trailing_newline(monkeypatch, tmp_path):
    """Pin §6.2 step 4: the .json file ends with a single trailing
    newline (matches the .txt convention). `json.dumps` does NOT emit
    a trailing newline; the implementation appends `+ "\\n"`."""
    text = _wrap_findings('[{"k": "v"}]')
    _stub_run_claude(monkeypatch, text=text)
    g = Gate(kind="agent", prompt_template="x")
    run_gates(
        [("g", g)], task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    raw_bytes = (tmp_path / "gates" / "g.json").read_bytes()
    assert raw_bytes.endswith(b"\n")


def test_run_gates_does_not_double_warn_on_malformed(
    monkeypatch, tmp_path, capsys,
):
    """Pin §D5 / §6.2: the second extraction in run_gates uses
    `warn=False` so the malformed-JSON warning is printed exactly
    ONCE (from AgentGate.run's first call, not duplicated by
    run_gates' second call). Counts canonical fragment occurrences."""
    _stub_run_claude(monkeypatch, text=_wrap_findings("{bad json}"))
    g = Gate(kind="agent", prompt_template="x")
    run_gates(
        [("g", g)], task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    out = capsys.readouterr().out
    # Exactly ONE occurrence — AgentGate's call site.
    assert out.count("malformed findings JSON") == 1


def test_run_gates_findings_None_when_skip_path_taken(monkeypatch, tmp_path):
    """Pin §D4: a gate whose `when` filter rejects the task is skipped
    via the passive-result path; the produced GateResult uses the
    dataclass default → `findings is None`. No .json written."""
    g = Gate(
        kind="agent", prompt_template="x", when=("task_class:ui",),
    )
    results = run_gates(
        [("ui-only", g)],
        task={"id": 1, "title": "T", "description": "D",
              "task_class": "schema"},
        task_dir=tmp_path, cli={},
    )
    assert len(results) == 1
    assert results[0].findings is None
    assert not (tmp_path / "gates" / "ui-only.json").exists()


def test_run_gates_findings_survive_exception_path(monkeypatch, tmp_path):
    """Pin §D4: an exception inside the gate runner produces a fallback
    GateResult; `findings` defaults to None. The reconstruction in
    run_gates' write loop preserves the None (no .json written)."""
    def boom(prompt, **kwargs):
        raise RuntimeError("agent crashed")

    monkeypatch.setattr(
        "claude_plan_execute.gates.agent.run_claude", boom,
    )
    g = Gate(kind="agent", prompt_template="x")
    results = run_gates(
        [("crashy", g)], task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    assert results[0].findings is None
    assert results[0].error is not None
    assert "agent crashed" in (results[0].error or "")
    assert not (tmp_path / "gates" / "crashy.json").exists()


def test_run_gates_passes_when_findings_is_empty_list(monkeypatch, tmp_path):
    """Pin §D3: an INTENTIONAL empty list (case 3 with `[]`) writes a
    sidecar with `{"findings": []}` and is distinct from the
    no-appendix path (case 1, no .json written). The predicate
    `r.findings is not None` accepts `[]`."""
    text = (
        "VERDICT: PASS\n"
        "<!-- BEGIN_FINDINGS_JSON -->\n[]\n<!-- END_FINDINGS_JSON -->\n"
    )
    _stub_run_claude(monkeypatch, text=text)
    g = Gate(kind="agent", prompt_template="x")
    results = run_gates(
        [("g", g)], task={"id": 1, "title": "T", "description": "D"},
        task_dir=tmp_path, cli={},
    )
    assert results[0].passed is True
    assert results[0].findings == []
    sidecar = json.loads((tmp_path / "gates" / "g.json").read_text())
    assert sidecar == {"findings": []}


# ---------------------------------------------------------------------------
# K. GateResult `findings` field (T11, v0.3)
# ---------------------------------------------------------------------------

def test_gate_result_findings_default_is_None():
    """Pin §D4: `findings` defaults to None so existing call sites
    that omit the kwarg keep working. Backward-compat invariant."""
    r = GateResult(
        name="x", passed=True, output="", output_path=None,
        duration_seconds=0.0, on_failure="block",
    )
    assert r.findings is None


def test_gate_result_findings_field_present():
    """Pin §D4: the field accepts a list[dict] and surfaces it via
    attribute access."""
    r = GateResult(
        name="x", passed=True, output="", output_path=None,
        duration_seconds=0.0, on_failure="block",
        findings=[{"a": 1}],
    )
    assert r.findings == [{"a": 1}]


def test_gate_result_field_order_includes_findings_last():
    """Pin §D4: field order ends with `..., error, findings`. Frozen
    dataclass field order is part of the public contract for
    positional construction; tests pin it as a regression guard
    against accidental reordering."""
    fields = [f.name for f in GateResult.__dataclass_fields__.values()]
    assert fields == [
        "name", "passed", "output", "output_path",
        "duration_seconds", "on_failure", "error", "findings",
    ]


def test_gate_result_is_still_frozen_after_findings_added():
    """Pin §D4: frozen-dataclass invariant preserved post-T11.
    Mutation raises."""
    r = GateResult(
        name="x", passed=True, output="", output_path=None,
        duration_seconds=0.0, on_failure="block",
    )
    with pytest.raises(Exception):
        r.findings = [{"k": "v"}]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# L. Built-in template content for security-auditor + test-critique (T11)
# ---------------------------------------------------------------------------
#
# Pin: each template ships a "### Structured findings appendix
# (recommended)" subsection with a concrete example, the per-finding
# schema keys (backticked), the omit disclaimer, and the literal
# HTML-comment markers. The example body is parseable JSON so a
# template typo cannot ship invalid examples.
#
# (Design-fidelity template tests live in test_gates_design_fidelity.py
# per the existing repo convention.)
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = (
    Path(__file__).parent.parent
    / "src" / "claude_plan_execute" / "gates" / "templates"
)


def _read_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


def _extract_template_example_block(body: str) -> str:
    """Return the JSON body between the FIRST marker pair in `body`."""
    match = _FINDINGS_RE.search(body)
    assert match is not None, "template lacks findings marker pair"
    return match.group(1)


# --- security-auditor.md --------------------------------------------------

def test_security_auditor_template_has_findings_appendix_section():
    """Pin §D / §6.3: literal `### Structured findings appendix
    (recommended)` heading present."""
    body = _read_template("security-auditor.md")
    assert "### Structured findings appendix (recommended)" in body


def test_security_auditor_template_documents_per_finding_schema_keys():
    """Pin §D10: the per-finding schema documents `category`,
    `severity`, `file`, `line`, `evidence`, `suggested_fix` as
    backticked field names."""
    body = _read_template("security-auditor.md")
    for key in (
        "category", "severity", "file", "line", "evidence", "suggested_fix",
    ):
        assert f"`{key}`" in body, f"missing backticked key: {key}"


def test_security_auditor_template_lists_common_severity_values_without_enforcing():
    """Pin §D10 / §D11: severity is documented with common values but
    explicitly NOT enforced. Pins the no-enum-constraints rule."""
    body = _read_template("security-auditor.md")
    for sev in ("critical", "high", "medium", "low"):
        assert f"`{sev}`" in body
    assert "NOT enforced" in body


def test_security_auditor_template_has_html_comment_markers():
    """Pin §D2: literal HTML-comment markers appear in the template
    so the agent has the exact text to copy."""
    body = _read_template("security-auditor.md")
    assert "<!-- BEGIN_FINDINGS_JSON -->" in body
    assert "<!-- END_FINDINGS_JSON -->" in body


def test_security_auditor_template_has_omit_disclaimer():
    """Pin §D10: each template instructs the agent that the appendix
    is optional ("emit an empty `[]` or omit the appendix
    entirely")."""
    body = _read_template("security-auditor.md")
    assert "emit an empty `[]` or omit the appendix entirely" in body


# --- test-critique.md -----------------------------------------------------

def test_test_critique_template_has_findings_appendix_section():
    body = _read_template("test-critique.md")
    assert "### Structured findings appendix (recommended)" in body


def test_test_critique_template_documents_per_finding_schema_keys():
    """Pin §D10: schema keys for test-critique."""
    body = _read_template("test-critique.md")
    for key in ("missing_test_for", "contract", "suggested_test_name"):
        assert f"`{key}`" in body, f"missing backticked key: {key}"


def test_test_critique_template_has_html_comment_markers():
    body = _read_template("test-critique.md")
    assert "<!-- BEGIN_FINDINGS_JSON -->" in body
    assert "<!-- END_FINDINGS_JSON -->" in body


def test_test_critique_template_has_omit_disclaimer():
    body = _read_template("test-critique.md")
    assert "emit an empty `[]` or omit the appendix entirely" in body


# --- belt-and-suspenders: examples are valid extractable appendices -------

def test_security_auditor_template_example_is_valid_findings_appendix():
    """Pin: the JSON example in the template parses cleanly via the
    same _extract_findings helper the build tool uses. Catches typos
    at CI time before agents ever see them."""
    body = _read_template("security-auditor.md")
    findings, sidecar = _extract_findings(body, "security-auditor")
    assert findings is not None, "template example failed extraction"
    assert isinstance(findings, list)
    assert all(isinstance(el, dict) for el in findings)
    # Sidecar must be writable JSON (no _raw).
    assert sidecar is not None
    assert "_raw" not in sidecar


def test_test_critique_template_example_is_valid_findings_appendix():
    """Pin: ditto for test-critique."""
    body = _read_template("test-critique.md")
    findings, sidecar = _extract_findings(body, "test-critique")
    assert findings is not None
    assert all(isinstance(el, dict) for el in findings)
    assert "_raw" not in (sidecar or {})
