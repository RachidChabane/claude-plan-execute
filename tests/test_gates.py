"""Tests for claude_plan_execute.gates — Gate Registry foundation."""
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# H. Misc
# ---------------------------------------------------------------------------

def test_output_cap_constant_is_100_kb():
    """Pin §D8: 100 KB cap matches enhancement plan §2.1."""
    assert _OUTPUT_CAP_BYTES == 100 * 1024
