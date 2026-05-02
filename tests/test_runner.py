"""Tests for prompts, claude subprocess wrapper, and runner orchestrator."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from claude_plan_execute import claude, runner
from claude_plan_execute.claude import _kill_child, run_claude
from claude_plan_execute.prompts import (
    build_implement_prompt,
    build_plan_prompt,
    build_review_prompt,
    build_revise_prompt,
)
from claude_plan_execute.runner import (
    check_verdict,
    execute_task,
    handle_sigint,
    save_error_output,
)
from claude_plan_execute.state import State


@pytest.fixture(autouse=True)
def _reset_module_globals():
    """Reset claude._active_child and runner._current_* between tests."""
    claude._active_child = None
    runner._current_state = None
    runner._current_task_id = 0
    runner._current_phase = "plan"
    yield
    claude._active_child = None
    runner._current_state = None
    runner._current_task_id = 0
    runner._current_phase = "plan"


class FakePopen:
    """Minimal subprocess.Popen stub for run_claude tests."""

    def __init__(self, cmd, *, stdout_data='{"result": "ok"}', stderr_data="",
                 returncode=0, raises_timeout=False):
        self.cmd = cmd
        self.pid = 12345
        self.returncode = returncode
        self._stdout_data = stdout_data
        self._stderr_data = stderr_data
        self._raises_timeout = raises_timeout
        self._poll_returns: int | None = None

    def communicate(self, timeout=None):
        if self._raises_timeout:
            raise subprocess.TimeoutExpired(cmd=self.cmd, timeout=timeout)
        self._poll_returns = self.returncode
        return self._stdout_data, self._stderr_data

    def poll(self):
        return self._poll_returns

    def terminate(self):
        self._poll_returns = -15

    def kill(self):
        self._poll_returns = -9

    def wait(self, timeout=None):
        return self.returncode


def _stub_run(*args, **kwargs):
    """No-op replacement for subprocess.run (pkill calls)."""
    return subprocess.CompletedProcess(args=args[0] if args else [], returncode=0)


# ---------------------------------------------------------------------------
# A. Prompt builders — pure functions
# ---------------------------------------------------------------------------


def test_plan_prompt_includes_task_id_and_title():
    out = build_plan_prompt({"id": 7, "title": "Refactor", "description": "D"}, "p.md")
    assert "TASK 7: Refactor" in out


def test_plan_prompt_includes_description():
    out = build_plan_prompt({"id": 1, "title": "T", "description": "describe me"}, "p.md")
    assert "describe me" in out


def test_plan_prompt_writes_to_supplied_path():
    out = build_plan_prompt({"id": 1, "title": "T", "description": "D"}, "plans/task-1/plan.md")
    assert "plans/task-1/plan.md" in out


def test_plan_prompt_omits_files_section_when_key_files_missing():
    out = build_plan_prompt({"id": 1, "title": "T", "description": "D"}, "p.md")
    assert "Key files to START with" not in out


def test_plan_prompt_omits_files_section_when_key_files_empty_list():
    out = build_plan_prompt(
        {"id": 1, "title": "T", "description": "D", "key_files": []}, "p.md"
    )
    assert "Key files to START with" not in out


def test_plan_prompt_renders_each_key_file_with_dash_indent():
    out = build_plan_prompt(
        {"id": 1, "title": "T", "description": "D", "key_files": ["a.py", "b.py"]},
        "p.md",
    )
    assert "  - a.py" in out
    assert "  - b.py" in out


def test_plan_prompt_byte_length_matches_baseline():
    # Pin: byte-identical output for fixed input. If this fails, the format
    # changed — verify the change is intentional before updating the integer.
    out = build_plan_prompt({"id": 1, "title": "T", "description": "D"}, "p.md")
    assert len(out) == 734


def test_review_prompt_includes_review_path():
    out = build_review_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "review.md"
    )
    assert "review.md" in out


def test_review_prompt_omits_revision_sentence_when_no_prev_feedback():
    out = build_review_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "r.md"
    )
    assert "plan was revised" not in out


def test_review_prompt_includes_revision_sentence_when_prev_feedback_truthy():
    out = build_review_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "r.md", prev_feedback="x"
    )
    assert "plan was revised based on previous feedback" in out


def test_review_prompt_does_NOT_inline_prev_feedback_body():
    feedback = "SECRET_FEEDBACK_TOKEN_42"
    out = build_review_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "r.md", feedback
    )
    assert feedback not in out


def test_revise_prompt_includes_plan_and_review_paths():
    out = build_revise_prompt(
        {"id": 1, "title": "T", "description": "D"}, "plans/p.md", "plans/r.md"
    )
    assert "plans/p.md" in out
    assert "plans/r.md" in out


def test_revise_prompt_does_not_mention_implementation():
    out = build_revise_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "r.md"
    )
    assert "Do NOT implement anything" in out


def test_implement_prompt_includes_test_command_verbatim():
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md",
        "pytest tests/ -k foo", "msg",
    )
    assert "pytest tests/ -k foo" in out


def test_implement_prompt_includes_commit_message_verbatim():
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md",
        "pytest", "feat: my custom commit message",
    )
    assert "feat: my custom commit message" in out


def test_implement_prompt_byte_length_matches_baseline():
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "pytest", "msg",
    )
    assert len(out) == 415


# ---------------------------------------------------------------------------
# B. check_verdict (in runner.py)
# ---------------------------------------------------------------------------


def test_check_verdict_returns_false_when_file_missing(tmp_path):
    approved, content = check_verdict(tmp_path / "missing.md")
    assert approved is False


def test_check_verdict_returns_sentinel_string_when_missing(tmp_path):
    _, content = check_verdict(tmp_path / "missing.md")
    assert content == "Review file not found"


def test_check_verdict_approved_with_space(tmp_path):
    p = tmp_path / "review.md"
    p.write_text("## Verdict: APPROVED\n\nLooks good.")
    approved, _ = check_verdict(p)
    assert approved is True


def test_check_verdict_approved_no_space(tmp_path):
    p = tmp_path / "review.md"
    p.write_text("## Verdict:APPROVED\n\nLooks good.")
    approved, _ = check_verdict(p)
    assert approved is True


def test_check_verdict_case_insensitive(tmp_path):
    p = tmp_path / "review.md"
    p.write_text("## verdict: approved\n")
    approved, _ = check_verdict(p)
    assert approved is True


def test_check_verdict_needs_revision_returns_false(tmp_path):
    p = tmp_path / "review.md"
    p.write_text("## Verdict: NEEDS_REVISION\n\nFix things.")
    approved, _ = check_verdict(p)
    assert approved is False


def test_check_verdict_returns_full_content_as_second_element(tmp_path):
    p = tmp_path / "review.md"
    body = "## Verdict: NEEDS_REVISION\n\nIssue 1\nIssue 2\n"
    p.write_text(body)
    _, content = check_verdict(p)
    assert content == body


# ---------------------------------------------------------------------------
# C. _kill_child (in claude.py)
# ---------------------------------------------------------------------------


def test_kill_child_noop_when_no_active_child(monkeypatch):
    monkeypatch.setattr("claude_plan_execute.claude.subprocess.run", _stub_run)
    claude._active_child = None
    claude._kill_child()
    assert claude._active_child is None


def test_kill_child_calls_terminate_on_running_child(monkeypatch):
    monkeypatch.setattr("claude_plan_execute.claude.subprocess.run", _stub_run)
    fake = FakePopen(["claude"])
    claude._active_child = fake
    claude._kill_child()
    # terminate sets _poll_returns to -15
    assert fake.poll() == -15


def test_kill_child_force_kills_after_terminate_timeout(monkeypatch):
    class FakePopenSlowToDie(FakePopen):
        def wait(self, timeout=None):
            if timeout is not None and timeout <= 5:
                raise subprocess.TimeoutExpired(cmd=self.cmd, timeout=timeout)
            return self.returncode

    monkeypatch.setattr("claude_plan_execute.claude.subprocess.run", _stub_run)
    fake = FakePopenSlowToDie(["claude"])
    claude._active_child = fake
    claude._kill_child()
    # kill sets _poll_returns to -9 after the wait timeout
    assert fake.poll() == -9


def test_kill_child_invokes_pkill_with_minus_P_for_descendants(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr("claude_plan_execute.claude.subprocess.run", fake_run)
    fake = FakePopen(["claude"])
    claude._active_child = fake
    claude._kill_child()
    assert captured["cmd"] == ["pkill", "-TERM", "-P", "12345"]


def test_kill_child_swallows_pkill_FileNotFoundError(monkeypatch):
    def _stub_run_raises_fnf(*args, **kwargs):
        raise FileNotFoundError("pkill not installed")

    monkeypatch.setattr(
        "claude_plan_execute.claude.subprocess.run", _stub_run_raises_fnf
    )
    fake = FakePopen(["claude"])
    claude._active_child = fake
    # Must not raise
    claude._kill_child()
    assert claude._active_child is None


def test_kill_child_clears_active_child_to_none(monkeypatch):
    monkeypatch.setattr("claude_plan_execute.claude.subprocess.run", _stub_run)
    claude._active_child = FakePopen(["claude"])
    claude._kill_child()
    assert claude._active_child is None


def test_kill_child_does_NOT_terminate_already_exited_child(monkeypatch):
    monkeypatch.setattr("claude_plan_execute.claude.subprocess.run", _stub_run)
    fake = FakePopen(["claude"])
    fake._poll_returns = 0  # already exited
    claude._active_child = fake
    claude._kill_child()
    # If terminate() had run, _poll_returns would be -15. It must remain 0.
    assert fake.poll() == 0


# ---------------------------------------------------------------------------
# D. run_claude (in claude.py)
# ---------------------------------------------------------------------------


def _patch_run_claude_subprocess(monkeypatch, *, stdout='{"result": "ok"}',
                                 stderr="", returncode=0, raises_timeout=False):
    """Install Popen + run stubs and return a captured-state dict."""
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["popen_kwargs"] = kwargs
        instance = FakePopen(
            cmd, stdout_data=stdout, stderr_data=stderr,
            returncode=returncode, raises_timeout=raises_timeout,
        )
        captured["instance"] = instance

        original_communicate = instance.communicate

        def communicate_recording(timeout=None):
            captured["communicate_timeout"] = timeout
            return original_communicate(timeout=timeout)

        instance.communicate = communicate_recording
        return instance

    monkeypatch.setattr("claude_plan_execute.claude.subprocess.Popen", fake_popen)
    monkeypatch.setattr("claude_plan_execute.claude.subprocess.run", _stub_run)
    return captured


def test_run_claude_command_includes_required_flags(monkeypatch):
    captured = _patch_run_claude_subprocess(monkeypatch)
    run_claude(
        "the prompt",
        model="opus", effort="max", max_turns=10, timeout_minutes=5,
        skip_permissions=False, verbose=False,
    )
    cmd = captured["cmd"]
    assert cmd[0] == "claude"
    assert "-p" in cmd and cmd[cmd.index("-p") + 1] == "the prompt"
    assert "--output-format" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert cmd[cmd.index("--effort") + 1] == "max"
    assert cmd[cmd.index("--max-turns") + 1] == "10"


def test_run_claude_command_omits_skip_perms_when_false(monkeypatch):
    captured = _patch_run_claude_subprocess(monkeypatch)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert "--dangerously-skip-permissions" not in captured["cmd"]


def test_run_claude_command_includes_skip_perms_when_true(monkeypatch):
    captured = _patch_run_claude_subprocess(monkeypatch)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=True, verbose=False,
    )
    assert "--dangerously-skip-permissions" in captured["cmd"]


def test_run_claude_command_omits_verbose_when_false(monkeypatch):
    captured = _patch_run_claude_subprocess(monkeypatch)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert "--verbose" not in captured["cmd"]


def test_run_claude_command_includes_verbose_when_true(monkeypatch):
    captured = _patch_run_claude_subprocess(monkeypatch)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=True,
    )
    assert "--verbose" in captured["cmd"]


def test_run_claude_extracts_result_from_json_dict(monkeypatch):
    _patch_run_claude_subprocess(monkeypatch, stdout='{"result": "extracted"}')
    code, text = run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert text == "extracted"


def test_run_claude_returns_raw_stdout_when_json_invalid(monkeypatch):
    _patch_run_claude_subprocess(monkeypatch, stdout="not json at all")
    _, text = run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert text == "not json at all"


def test_run_claude_returns_raw_stdout_when_json_dict_lacks_result_key(monkeypatch):
    _patch_run_claude_subprocess(monkeypatch, stdout='{"other": "value"}')
    _, text = run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert text == '{"other": "value"}'


def test_run_claude_returns_raw_stdout_when_json_is_a_list(monkeypatch):
    _patch_run_claude_subprocess(monkeypatch, stdout='[1, 2, 3]')
    _, text = run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert text == "[1, 2, 3]"


def test_run_claude_returns_returncode_from_popen(monkeypatch):
    _patch_run_claude_subprocess(monkeypatch, returncode=42)
    code, _ = run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert code == 42


def test_run_claude_passes_timeout_in_seconds_to_communicate(monkeypatch):
    captured = _patch_run_claude_subprocess(monkeypatch)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=7,
        skip_permissions=False, verbose=False,
    )
    assert captured["communicate_timeout"] == 7 * 60


def test_run_claude_clears_active_child_after_call(monkeypatch):
    _patch_run_claude_subprocess(monkeypatch)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert claude._active_child is None


# Note: the timeout path lets the real _kill_child run; subprocess.run (the
# pkill call inside _kill_child) is stubbed via _stub_run. _kill_child's
# internals are exercised separately in section C.


def test_run_claude_timeout_returns_1_and_empty_string(monkeypatch):
    _patch_run_claude_subprocess(monkeypatch, raises_timeout=True)
    code, text = run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert code == 1
    assert text == ""


def test_run_claude_timeout_invokes_kill_child(monkeypatch):
    called = []

    def stub_kill():
        called.append(True)

    _patch_run_claude_subprocess(monkeypatch, raises_timeout=True)
    monkeypatch.setattr("claude_plan_execute.claude._kill_child", stub_kill)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert called == [True]


def test_run_claude_timeout_clears_active_child(monkeypatch):
    _patch_run_claude_subprocess(monkeypatch, raises_timeout=True)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert claude._active_child is None


def test_run_claude_does_NOT_set_preexec_fn(monkeypatch):
    # Load-bearing pin: workflow-server.py runs the CLI script with
    # preexec_fn=os.setsid so its killpg covers the entire tree. If
    # run_claude added its own setsid, the child would escape that group.
    captured = _patch_run_claude_subprocess(monkeypatch)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert "preexec_fn" not in captured["popen_kwargs"]


def test_run_claude_prints_invocation_banner_with_model_effort_and_prompt_length(
    monkeypatch, capsys
):
    _patch_run_claude_subprocess(monkeypatch)
    run_claude(
        "hello world", model="opus", effort="max", max_turns=10,
        timeout_minutes=5, skip_permissions=False, verbose=False,
    )
    out = capsys.readouterr().out
    assert "claude --model opus --effort max --max-turns 10" in out
    assert "(timeout 5m, 11 char prompt)" in out


def test_run_claude_prints_completion_line_with_elapsed_and_exit_code(
    monkeypatch, capsys
):
    _patch_run_claude_subprocess(monkeypatch, returncode=0)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    out = capsys.readouterr().out
    assert "Completed in" in out
    assert "(exit 0)" in out


def test_run_claude_prints_stderr_truncated_at_300_chars_on_nonzero_exit(
    monkeypatch, capsys
):
    big_stderr = "X" * 500
    _patch_run_claude_subprocess(monkeypatch, returncode=1, stderr=big_stderr)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    out = capsys.readouterr().out
    assert "stderr: " in out
    assert "X" * 300 in out
    assert "X" * 301 not in out


def test_run_claude_does_NOT_print_stderr_when_returncode_is_zero(
    monkeypatch, capsys
):
    _patch_run_claude_subprocess(monkeypatch, returncode=0, stderr="some stderr")
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    out = capsys.readouterr().out
    assert "stderr: " not in out


def test_run_claude_does_NOT_print_stderr_when_stderr_is_empty(
    monkeypatch, capsys
):
    _patch_run_claude_subprocess(monkeypatch, returncode=1, stderr="")
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    out = capsys.readouterr().out
    assert "stderr: " not in out


# ---------------------------------------------------------------------------
# E. execute_task — dry-run paths only
# ---------------------------------------------------------------------------


def _no_run_claude_called(monkeypatch):
    """Patch runner.run_claude with a stub that records calls."""
    called = []

    def stub(*args, **kwargs):
        called.append((args, kwargs))
        return 0, ""

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub)
    return called


def _make_task(**overrides):
    base = {"id": 1, "title": "T", "description": "D"}
    base.update(overrides)
    return base


def test_execute_task_dry_run_creates_task_directory(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    assert (tmp_path / "task-1").is_dir()


def test_execute_task_dry_run_plan_phase_writes_planning_state(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=5), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    entry = state.get(5)
    assert entry["status"] == "planning"
    assert entry["plan_path"].endswith("task-5/plan.md")


def test_execute_task_dry_run_plan_phase_returns_before_run_claude(monkeypatch, tmp_path):
    called = _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    assert called == []


def test_execute_task_dry_run_skip_to_review_returns_in_phase_2(monkeypatch, tmp_path):
    called = _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="review", dry_run=True,
    )
    assert called == []
    assert state.get(1)["status"] == "reviewing"


def test_execute_task_dry_run_skip_to_implement_returns_in_phase_3(monkeypatch, tmp_path):
    called = _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement", dry_run=True,
    )
    assert called == []
    assert state.get(1)["status"] == "implementing"


def test_execute_task_sets_sigint_globals_on_entry(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=42), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    assert runner._current_state is state
    assert runner._current_task_id == 42
    assert runner._current_phase == "plan"


def test_execute_task_uses_resolved_plans_dir_from_file_defaults(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    plans_dir = tmp_path / "custom-plans"
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(), state,
        file_defaults={"plans_dir": str(plans_dir)},
        cli={}, dry_run=True,
    )
    assert (plans_dir / "task-1").is_dir()


def test_execute_task_default_commit_msg_when_task_has_none(monkeypatch, tmp_path):
    # The default `commit_msg` is constructed but not exposed; verify
    # indirectly: dry-run completes without KeyError on commit_message.
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    task = _make_task()
    assert "commit_message" not in task
    execute_task(
        task, state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement", dry_run=True,
    )
    assert state.get(1)["status"] == "implementing"


# ---------------------------------------------------------------------------
# F. handle_sigint
# ---------------------------------------------------------------------------


def test_handle_sigint_calls_kill_child(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(
        "claude_plan_execute.runner._kill_child", lambda: called.append(True)
    )
    runner._current_state = State(tmp_path / "state.json")
    with pytest.raises(SystemExit):
        handle_sigint(2, None)
    assert called == [True]


def test_handle_sigint_saves_state_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    state = State(tmp_path / "state.json")
    state.set_status(1, "implementing")  # ensure data on disk
    saved = []
    original_save = state.save
    state.save = lambda: saved.append(True) or original_save()
    runner._current_state = state
    with pytest.raises(SystemExit):
        handle_sigint(2, None)
    assert saved == [True]


def test_handle_sigint_skips_save_when_state_is_none(monkeypatch):
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    runner._current_state = None
    # Must not raise AttributeError when state is None
    with pytest.raises(SystemExit):
        handle_sigint(2, None)


def test_handle_sigint_exits_with_code_130(monkeypatch, tmp_path):
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    runner._current_state = State(tmp_path / "state.json")
    runner._current_task_id = 7
    runner._current_phase = "implement"
    with pytest.raises(SystemExit) as exc_info:
        handle_sigint(2, None)
    assert exc_info.value.code == 130


def test_handle_sigint_prints_resume_command(monkeypatch, capsys):
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    runner._current_state = None
    runner._current_task_id = 7
    runner._current_phase = "implement"
    with pytest.raises(SystemExit):
        handle_sigint(2, None)
    out = capsys.readouterr().out
    assert "--task 7 --skip-to implement" in out


# ---------------------------------------------------------------------------
# G. save_error_output
# ---------------------------------------------------------------------------


def test_save_error_output_writes_error_file_with_exit_code_and_output(tmp_path):
    save_error_output(tmp_path, 1, "raw output text")
    body = (tmp_path / "error-output.txt").read_text()
    assert "Exit code: 1" in body
    assert "raw output text" in body


def test_save_error_output_overwrites_existing_file(tmp_path):
    err_path = tmp_path / "error-output.txt"
    err_path.write_text("OLD")
    save_error_output(tmp_path, 2, "NEW")
    body = err_path.read_text()
    assert "OLD" not in body
    assert "Exit code: 2" in body
    assert "NEW" in body
