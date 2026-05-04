"""Tests for prompts, claude subprocess wrapper, and runner orchestrator."""
from __future__ import annotations

import json
import subprocess
import threading
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
def _reset_module_globals(monkeypatch, tmp_path_factory):
    """Reset claude._active_child, claude._last_run_*, and runner._current_*
    between tests. Without the claude flag reset, a test that sets
    _last_run_timed_out via a stub leaks the flag into subsequent tests.

    T15: also default-stub runner.memory_dir_for_cwd to an absent
    path, so the cross-task memory passes never touch the operator's
    real ~/.claude/projects/... directory. Tests that need a writable
    memdir opt in via _stub_memory_dir(monkeypatch, tmp_path).
    """
    claude._active_child.clear()
    claude._reset_last_run_state()
    runner._current_state = None
    runner._current_tasks.clear()
    runner._worker_ids.clear()

    # T15: default isolation. Each test gets a fresh non-existent path
    # so neither citation pass nor lifecycle prune ever touches the
    # operator's real ~/.claude/projects/<encoded-cwd>/memory/.
    isolated_absent = tmp_path_factory.mktemp("memdir_iso") / "absent"
    monkeypatch.setattr(
        "claude_plan_execute.runner.memory_dir_for_cwd",
        lambda _cwd: isolated_absent,
    )
    # T16 §J.1: also stub the gates-side resolution. `run_gates`
    # calls `memory_dir_for_cwd(Path.cwd())` to compute the
    # `{memory_dir}` substitution; without this patch any
    # integration test running an agent gate would resolve the
    # operator's real ~/.claude/projects/... directory.
    monkeypatch.setattr(
        "claude_plan_execute.gates.memory_dir_for_cwd",
        lambda _cwd=None: isolated_absent,
    )

    yield
    claude._active_child.clear()
    claude._reset_last_run_state()
    runner._current_state = None
    runner._current_tasks.clear()
    runner._worker_ids.clear()


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
    # T15 (v0.3 §"cross-task-memory") added _MEMORY_SECTION_PLAN to the body.
    out = build_plan_prompt({"id": 1, "title": "T", "description": "D"}, "p.md")
    assert len(out) == 1341


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


def test_implement_prompt_includes_commit_message_verbatim():
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md",
        "feat: my custom commit message",
    )
    assert "feat: my custom commit message" in out


def test_implement_prompt_byte_length_matches_baseline():
    # Task 6 v0.3 §D1: WIP-commit + progress.md instructions added.
    # T15 (v0.3 §"cross-task-memory") added _MEMORY_SECTION_IMPLEMENT.
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
    )
    assert len(out) == 3255


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
    # Per T13 §F3.1: autouse fixture's `_active_child.clear()` already
    # gives an empty dict; no per-thread entry → kill_child is a no-op.
    claude._kill_child()
    assert threading.get_ident() not in claude._active_child


def test_kill_child_calls_terminate_on_running_child(monkeypatch):
    monkeypatch.setattr("claude_plan_execute.claude.subprocess.run", _stub_run)
    fake = FakePopen(["claude"])
    claude._set_active_child(fake)
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
    claude._set_active_child(fake)
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
    claude._set_active_child(fake)
    claude._kill_child()
    assert captured["cmd"] == ["pkill", "-TERM", "-P", "12345"]


def test_kill_child_swallows_pkill_FileNotFoundError(monkeypatch):
    def _stub_run_raises_fnf(*args, **kwargs):
        raise FileNotFoundError("pkill not installed")

    monkeypatch.setattr(
        "claude_plan_execute.claude.subprocess.run", _stub_run_raises_fnf
    )
    fake = FakePopen(["claude"])
    claude._set_active_child(fake)
    # Must not raise
    claude._kill_child()
    assert threading.get_ident() not in claude._active_child


def test_kill_child_clears_active_child_to_none(monkeypatch):
    monkeypatch.setattr("claude_plan_execute.claude.subprocess.run", _stub_run)
    claude._set_active_child(FakePopen(["claude"]))
    claude._kill_child()
    assert threading.get_ident() not in claude._active_child


def test_kill_child_does_NOT_terminate_already_exited_child(monkeypatch):
    monkeypatch.setattr("claude_plan_execute.claude.subprocess.run", _stub_run)
    fake = FakePopen(["claude"])
    fake._poll_returns = 0  # already exited
    claude._set_active_child(fake)
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
    assert threading.get_ident() not in claude._active_child


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
    assert threading.get_ident() not in claude._active_child


def test_run_claude_does_NOT_set_preexec_fn(monkeypatch):
    # Load-bearing pin: server.py runs the CLI script with
    # preexec_fn=os.setsid so its killpg covers the entire tree. If
    # run_claude added its own setsid, the child would escape that group.
    # The parallel pin in test_server.py
    # (test_post_run_uses_setsid_for_process_group) asserts the server
    # side; this one asserts the run_claude side.
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


def test_run_claude_sets_subtype_from_list_result_event(monkeypatch):
    """Pin §D6: CLI v2.x list-of-events output. The terminal
    `type: result` event's subtype is captured into the side-channel
    flag for the runner to inspect post-call."""
    stdout = json.dumps([
        {"type": "system", "subtype": "init"},
        {"type": "assistant"},
        {"type": "result", "subtype": "error_max_turns", "is_error": True},
    ])
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert claude._last_run_state.subtype == "error_max_turns"


def test_run_claude_sets_subtype_from_dict_legacy_shape(monkeypatch):
    """Pin §D6: legacy CLI single-dict output. Subtype on the
    top-level dict still flows through."""
    stdout = json.dumps({
        "type": "result", "subtype": "success", "result": "hi",
    })
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert claude._last_run_state.subtype == "success"


def test_run_claude_extracts_result_text_from_list_result_event(monkeypatch):
    """Pin §R17: the new parser extracts `result` from the list-shape
    output (pre-task-9 the parser silently fell through and returned
    raw stdout). No runner caller depends on the text, but the
    extraction is now real."""
    stdout = json.dumps([
        {"type": "result", "subtype": "success", "result": "extracted text"},
    ])
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout)
    _, text = run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert text == "extracted text"


def test_run_claude_resets_subtype_at_call_entry(monkeypatch):
    """Pin §D6: stale telemetry from a prior call is cleared. Pre-set
    the flag, then make a call whose stdout has no parseable result
    event — the flag must be reset to None."""
    claude._last_run_state.subtype = "leftover_from_prior_call"
    _patch_run_claude_subprocess(monkeypatch, stdout="not json")
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert claude._last_run_state.subtype is None


def test_run_claude_timeout_sets_last_run_timed_out_flag(monkeypatch):
    """Pin §D6: subprocess.TimeoutExpired path sets the side-channel
    flag. The runner reads this immediately after the call to
    classify the failure as `implementer_timeout`."""
    _patch_run_claude_subprocess(monkeypatch, raises_timeout=True)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert claude._last_run_state.timed_out is True


def test_run_claude_resets_timed_out_flag_at_call_entry(monkeypatch):
    """Stale-flag pin: a clean call after a prior timeout must clear the flag."""
    claude._last_run_state.timed_out = True
    _patch_run_claude_subprocess(monkeypatch)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert claude._last_run_state.timed_out is False


# ---------------------------------------------------------------------------
# D2. run_claude rate-limit retry (v0.3 task 1)
# ---------------------------------------------------------------------------


def _patch_run_claude_once(monkeypatch, responses):
    """Drive run_claude with a deterministic sequence of (code, text, stderr)
    tuples. Returns (helper_calls, sleep_calls).

    T2 §D2: the underlying `_run_claude_once` returns a 5-tuple
    `(code, text, stderr, usage, elapsed)`. The fake widens the
    3-tuple `responses` entries to that shape with `usage=None,
    elapsed=0.0` so existing callers stay backward-compatible —
    rate-limited retries / timeout behavior is unchanged."""
    helper_calls = []
    sleep_calls = []

    def fake_once(prompt, **kwargs):
        i = len(helper_calls)
        helper_calls.append(kwargs)
        if i >= len(responses):
            raise AssertionError(
                f"_run_claude_once called more times than expected "
                f"(got {i + 1}, expected {len(responses)})"
            )
        code, text, stderr = responses[i]
        # Simulate side-channel flags exactly like the real helper would:
        # reset at entry, set timed_out only on the explicit timeout sentinel.
        claude._last_run_state.timed_out = (code == 1 and text == "" and stderr == "TIMEOUT")
        claude._last_run_state.subtype = None
        return (
            code,
            text,
            "" if stderr == "TIMEOUT" else stderr,
            None,
            0.0,
        )

    monkeypatch.setattr("claude_plan_execute.claude._run_claude_once", fake_once)
    monkeypatch.setattr(
        "claude_plan_execute.claude.time.sleep",
        lambda s: sleep_calls.append(s),
    )
    return helper_calls, sleep_calls


def test_run_claude_success_first_attempt_no_retry(monkeypatch):
    helper_calls, sleeps = _patch_run_claude_once(monkeypatch, [
        (0, "ok", ""),
    ])
    code, text = run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert (code, text) == (0, "ok")
    assert len(helper_calls) == 1
    assert sleeps == []
    assert claude._last_run_state.rate_limited is False


def test_run_claude_one_rate_limit_then_success(monkeypatch):
    helper_calls, sleeps = _patch_run_claude_once(monkeypatch, [
        (1, "", "rate_limit_error: too many requests"),
        (0, "ok", ""),
    ])
    code, text = run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert (code, text) == (0, "ok")
    assert len(helper_calls) == 2
    assert sleeps == [60]
    assert claude._last_run_state.rate_limited is False


def test_run_claude_all_retries_exhausted_sets_flag(monkeypatch):
    helper_calls, sleeps = _patch_run_claude_once(monkeypatch, [
        (1, "", "Anthropic API rate limit reached"),
        (1, "", "rate_limit_error"),
        (1, "", "429 Too Many Requests"),
        (1, "", "Retry-After: 60"),
    ])
    code, text = run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert (code, text) == (1, "")
    assert len(helper_calls) == 4
    assert sleeps == [60, 120, 240]
    assert claude._last_run_state.rate_limited is True


def test_run_claude_non_rate_limit_failure_not_retried(monkeypatch):
    helper_calls, sleeps = _patch_run_claude_once(monkeypatch, [
        (1, "", "compile error: undefined symbol"),
    ])
    code, text = run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert (code, text) == (1, "")
    assert len(helper_calls) == 1
    assert sleeps == []
    assert claude._last_run_state.rate_limited is False


def test_run_claude_timeout_does_not_consume_rate_limit_budget(monkeypatch):
    """A timeout must short-circuit the retry loop: 1 attempt, 0 sleeps,
    `_last_run_timed_out=True`, `_last_run_rate_limited=False`. Otherwise
    a 60-minute timeout would block on three more 60s sleeps."""
    helper_calls, sleeps = _patch_run_claude_once(monkeypatch, [
        (1, "", "TIMEOUT"),
    ])
    code, text = run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert (code, text) == (1, "")
    assert len(helper_calls) == 1
    assert sleeps == []
    assert claude._last_run_state.timed_out is True
    assert claude._last_run_state.rate_limited is False


def test_run_claude_max_retries_zero_disables_rate_limit_awareness(monkeypatch):
    """Pin §D5.3 opt-out branch: 0 retries → rate-limit detection
    skipped entirely; pre-task behavior preserved."""
    helper_calls, sleeps = _patch_run_claude_once(monkeypatch, [
        (1, "", "rate_limit_error"),
    ])
    code, text = run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
        max_rate_limit_retries=0,
    )
    assert (code, text) == (1, "")
    assert len(helper_calls) == 1
    assert sleeps == []
    assert claude._last_run_state.rate_limited is False


@pytest.mark.parametrize("stderr_text", [
    "RATE_LIMIT_ERROR: blah",
    "ratE_limiT_error",
    "HTTP 429 returned",
    "Anthropic API rate limit hit",
    "anthropic api rate limit",
    "Retry-After: 30",
    "retry-after: 30",
])
def test_run_claude_rate_limit_pattern_match_is_case_insensitive(
    monkeypatch, stderr_text,
):
    helper_calls, sleeps = _patch_run_claude_once(monkeypatch, [
        (1, "", stderr_text),
        (0, "ok", ""),
    ])
    code, _ = run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert code == 0
    assert sleeps == [60]


def test_run_claude_exponential_backoff_doubles_per_attempt(monkeypatch):
    helper_calls, sleeps = _patch_run_claude_once(monkeypatch, [
        (1, "", "rate_limit_error")
    ] * 5 + [(0, "ok", "")])
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
        max_rate_limit_retries=5, rate_limit_base_backoff_seconds=10,
    )
    # 5 retries → 6 attempts; sequence succeeds on attempt 6.
    assert sleeps == [10, 20, 40, 80, 160]


def test_run_claude_retry_banner_format(monkeypatch, capsys):
    _patch_run_claude_once(monkeypatch, [
        (1, "", "rate_limit_error"),
        (0, "ok", ""),
    ])
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    out = capsys.readouterr().out
    assert "Rate-limited; waiting 60s before retry 2/4" in out


def test_run_claude_exhaustion_banner_format(monkeypatch, capsys):
    _patch_run_claude_once(monkeypatch, [
        (1, "", "rate_limit_error"),
    ] * 4)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    out = capsys.readouterr().out
    assert "Rate-limit retries exhausted after 4 attempts." in out


def test_run_claude_resets_rate_limited_flag_at_entry(monkeypatch):
    """A clean call after a prior exhaustion must clear the flag."""
    claude._last_run_state.rate_limited = True  # leftover
    _patch_run_claude_once(monkeypatch, [
        (0, "ok", ""),
    ])
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert claude._last_run_state.rate_limited is False


def test_is_rate_limited_matches_each_pattern():
    from claude_plan_execute.claude import _is_rate_limited
    for txt in ["rate_limit_error", "429", "anthropic api rate limit", "retry-after"]:
        assert _is_rate_limited(txt) is True


def test_is_rate_limited_returns_false_on_empty_stderr():
    from claude_plan_execute.claude import _is_rate_limited
    assert _is_rate_limited("") is False


def test_is_rate_limited_returns_false_on_unrelated_text():
    from claude_plan_execute.claude import _is_rate_limited
    assert _is_rate_limited("syntax error: undefined name 'x'") is False


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
    """Pin F4.4: when execute_task starts, `_set_current_task` is
    called so the per-thread `_current_tasks` entry is populated for
    `handle_sigint`'s iteration. Capture during the call (via stub
    hook) because F4.8's outer try/finally CLEARS the entry on exit
    (deliberate per parallel-mode correctness — see plan §F4.8). A
    naive post-call read would KeyError on the empty dict and miss
    the invariant being pinned.
    """
    captured: dict = {}
    original = runner._set_current_task

    def hook(tid, phase):
        original(tid, phase)
        captured.setdefault("tasks", dict(runner._current_tasks))
        captured.setdefault("state", runner._current_state)

    monkeypatch.setattr(runner, "_set_current_task", hook)
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=42), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    assert captured["state"] is state
    assert captured["tasks"][threading.get_ident()] == (42, "plan")


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
    runner._set_current_task(7, "implement")
    with pytest.raises(SystemExit) as exc_info:
        handle_sigint(2, None)
    assert exc_info.value.code == 130


def test_handle_sigint_prints_resume_command(monkeypatch, capsys):
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    runner._current_state = None
    runner._set_current_task(7, "implement")
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


# ---------------------------------------------------------------------------
# H. resolved-config.json audit artifact (task 5)
# ---------------------------------------------------------------------------


def _read_resolved_config(plans_dir: Path, tid: int | str) -> dict:
    p = plans_dir / f"task-{tid}" / "resolved-config.json"
    return json.loads(p.read_text())


def test_execute_task_writes_resolved_config_json_to_task_dir(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=42), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    assert (tmp_path / "task-42" / "resolved-config.json").exists()


def test_execute_task_resolved_config_contains_all_audit_keys(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    cfg = _read_resolved_config(tmp_path, 1)
    from claude_plan_execute.config import KEYS_TO_AUDIT
    assert set(KEYS_TO_AUDIT) <= set(cfg.keys())
    # v0.3 task 8: a services_required sibling is allowed
    # (deliberately NOT in KEYS_TO_AUDIT — see Plan §1 Non-goals
    # and §D9 for the rationale).


def test_execute_task_resolved_config_provenance_for_task_override(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, model="haiku"), state,
        file_defaults={"plans_dir": str(tmp_path), "model": "opus"},
        cli={}, dry_run=True,
    )
    cfg = _read_resolved_config(tmp_path, 1)
    assert cfg["model"] == {"value": "haiku", "from": "task"}


def test_execute_task_resolved_config_provenance_for_cli_override(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, model="haiku"), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={"model": "sonnet"}, dry_run=True,
    )
    cfg = _read_resolved_config(tmp_path, 1)
    assert cfg["model"] == {"value": "sonnet", "from": "cli"}


def test_execute_task_resolved_config_provenance_for_phase_inheritance(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, phase="foundation"), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={},
        phases_by_id={
            "foundation": {"id": "foundation", "defaults": {"model": "phase-val"}},
        },
        dry_run=True,
    )
    cfg = _read_resolved_config(tmp_path, 1)
    assert cfg["model"] == {"value": "phase-val", "from": "phase:foundation"}


def test_execute_task_resolved_config_provenance_for_task_class_inherited_from_phase(
    monkeypatch, tmp_path,
):
    """Plan §D5: when class is inherited from phase, the audit label still
    carries the class name (not bare 'task_class')."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, phase="foundation"), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={},
        phases_by_id={
            "foundation": {"id": "foundation", "defaults": {"task_class": "schema"}},
        },
        task_class_models={"schema": {"model": "schema-model"}},
        dry_run=True,
    )
    cfg = _read_resolved_config(tmp_path, 1)
    assert cfg["model"] == {"value": "schema-model", "from": "task_class:schema"}


def test_execute_task_resolved_config_provenance_for_explicit_task_class(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, task_class="polish"), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={},
        task_class_models={"polish": {"effort": "medium"}},
        dry_run=True,
    )
    cfg = _read_resolved_config(tmp_path, 1)
    assert cfg["effort"] == {"value": "medium", "from": "task_class:polish"}


def test_execute_task_records_resolved_config_path_on_state(monkeypatch, tmp_path):
    """Plan §D6: dashboard reads ts.resolved_config_path to know whether
    the artifact exists."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=7), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    entry = state.get(7)
    assert entry["resolved_config_path"] == str(tmp_path / "task-7" / "resolved-config.json")


def test_execute_task_resolved_config_path_persists_to_state_json(monkeypatch, tmp_path):
    """The path field must survive a state round-trip via state.save()."""
    _no_run_claude_called(monkeypatch)
    state_path = tmp_path / "state.json"
    state = State(state_path)
    execute_task(
        _make_task(id=7), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    on_disk = json.loads(state_path.read_text())
    assert on_disk["tasks"]["7"]["resolved_config_path"].endswith("task-7/resolved-config.json")


def test_execute_task_resolved_config_overwritten_on_rerun_with_different_cli(
    monkeypatch, tmp_path,
):
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={"model": "opus"}, dry_run=True,
    )
    first = _read_resolved_config(tmp_path, 1)
    assert first["model"]["value"] == "opus"

    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={"model": "sonnet"}, dry_run=True,
    )
    second = _read_resolved_config(tmp_path, 1)
    assert second["model"]["value"] == "sonnet"
    assert second["model"]["from"] == "cli"


def test_execute_task_resolved_config_written_in_dry_run(monkeypatch, tmp_path):
    """Plan §D6: dry-run writes the artifact so users can audit without running."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    assert (tmp_path / "task-1" / "resolved-config.json").exists()


def test_execute_task_resolved_config_written_when_skip_to_implement(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement", dry_run=True,
    )
    assert (tmp_path / "task-1" / "resolved-config.json").exists()


def test_execute_task_resolved_config_json_is_valid_json(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    raw = (tmp_path / "task-1" / "resolved-config.json").read_text()
    # Round-trip: parse then compare to the in-memory shape we expect.
    parsed = json.loads(raw)
    assert "model" in parsed and "value" in parsed["model"] and "from" in parsed["model"]


# ---------------------------------------------------------------------------
# I. on_max_review_rounds review-loop branches (task 6)
# ---------------------------------------------------------------------------


# These tests do NOT use dry_run=True. The dry_run path returns inside
# the review loop on the first iteration, before for-else can fire.
# Instead they stub run_claude (no-op), use skip_to="review" to bypass
# Phase 1, set max_review_rounds=1, and never write a review file —
# check_verdict returns False, the loop body completes once, and
# for-else triggers. (Plan §8 in plans/task-6/plan.md.)


def test_execute_task_max_rounds_with_fail_sets_blocked_status(monkeypatch, tmp_path):
    """Task 6 §D6: on_max_review_rounds=fail → status 'blocked'."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "fail",
        },
        cli={}, skip_to="review",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"


def test_execute_task_max_rounds_with_fail_records_block_reason(monkeypatch, tmp_path):
    """Task 6 §D8: block_reason persisted as a sibling extra."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "fail",
        },
        cli={}, skip_to="review",
    )
    entry = state.get(1)
    assert entry.get("block_reason") == "max_review_rounds_exhausted"


def test_execute_task_max_rounds_with_fail_skips_implement(monkeypatch, tmp_path):
    """Task 6 §D6: blocked path returns before Phase 3. Status must NOT
    transition to 'implementing' or 'done'."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "fail",
        },
        cli={}, skip_to="review",
    )
    entry = state.get(1)
    assert entry["status"] != "implementing"
    assert entry["status"] != "done"


def test_execute_task_max_rounds_with_fail_does_NOT_mark_approved(monkeypatch, tmp_path):
    """Pin §D6: the existing state.set_status(tid, 'approved') line must
    move INSIDE the auto_approve branch. The fail path must skip it
    entirely — flickering 'approved' → 'blocked' would churn the
    dashboard's WebSocket stream pointlessly."""
    _no_run_claude_called(monkeypatch)
    state_path = tmp_path / "state.json"
    state = State(state_path)
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "fail",
        },
        cli={}, skip_to="review",
    )
    # The persisted state must read 'blocked' — never an intermediate
    # 'approved' that got overwritten. (We can't directly observe
    # intermediate states, but we can assert the final state is right
    # AND that no spurious approved-then-blocked sequence appears in the
    # JSON file as a recovered partial write. The test is more about the
    # branching logic being correct than catching live churn.)
    on_disk = json.loads(state_path.read_text())
    assert on_disk["tasks"]["1"]["status"] == "blocked"


def test_execute_task_max_rounds_with_auto_approve_proceeds_to_implement(
    monkeypatch, tmp_path,
):
    """Task 6 §D6: on_max_review_rounds=auto_approve → existing behavior
    (mark 'approved', fall through to Phase 3 'implementing')."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "auto_approve",
        },
        cli={}, skip_to="review",
    )
    entry = state.get(1)
    # After auto_approve fall-through + Phase 3 stub run_claude, status
    # ends at 'done' (the last set_status in execute_task). The exact
    # final state is whatever the stubbed run_claude leaves; the
    # invariant is that it's NOT 'blocked'.
    assert entry["status"] != "blocked"
    assert entry.get("block_reason") is None


def test_execute_task_max_rounds_with_auto_approve_does_NOT_set_block_reason(
    monkeypatch, tmp_path,
):
    """Pin: block_reason is NOT set on the auto_approve path."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "auto_approve",
        },
        cli={}, skip_to="review",
    )
    entry = state.get(1)
    assert "block_reason" not in entry


def test_execute_task_resolved_config_includes_on_max_review_rounds_for_v1(
    monkeypatch, tmp_path,
):
    """Task 6 §D9 / §D10: on_max_review_rounds is in the audit JSON.
    Provenance for v1 compat reads 'project_defaults'."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "on_max_review_rounds": "auto_approve",
        },
        cli={}, dry_run=True,
    )
    cfg = _read_resolved_config(tmp_path, 1)
    assert cfg["on_max_review_rounds"] == {
        "value": "auto_approve", "from": "project_defaults",
    }


def test_execute_task_resolved_config_includes_on_max_review_rounds_for_v2_default(
    monkeypatch, tmp_path,
):
    """Task 6 §D9: v2 + no override → from = 'hardcoded'."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},  # no on_max_review_rounds
        cli={}, dry_run=True,
    )
    cfg = _read_resolved_config(tmp_path, 1)
    assert cfg["on_max_review_rounds"] == {"value": "fail", "from": "hardcoded"}


# ---------------------------------------------------------------------------
# J. Persona injection (task 7)
# ---------------------------------------------------------------------------


def test_execute_task_loads_persona_from_resolved_persona_file_into_plan_prompt(
    monkeypatch, tmp_path,
):
    """Task 7: runner reads persona_file from the resolved dict, loads it,
    and passes the persona section to build_plan_prompt. Pin the wiring;
    the loader internals are tested in test_prompts.py."""
    # Capture the prompt that build_plan_prompt produces so we can assert
    # the persona was prepended.
    captured: dict = {}

    real_build_plan = runner.build_plan_prompt

    def spy_build_plan(*args, **kwargs):
        captured["persona"] = kwargs.get("persona", "")
        return real_build_plan(*args, **kwargs)

    monkeypatch.setattr("claude_plan_execute.runner.build_plan_prompt", spy_build_plan)
    _no_run_claude_called(monkeypatch)

    persona_file = tmp_path / "p.md"
    persona_file.write_text("# Persona\nMy project rules.")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "persona_file": str(persona_file),
        },
        cli={}, dry_run=True,
    )

    assert "## Project conventions" in captured["persona"]
    assert "My project rules." in captured["persona"]


def test_execute_task_passes_empty_persona_when_persona_file_missing(
    monkeypatch, tmp_path,
):
    """Pin: when persona_file resolves to a missing path, the runner still
    runs cleanly (silent fallback) and the prompt builder gets persona=''.

    Note: we pin `persona_file` explicitly to a tmp_path-owned filename
    that we never create, rather than relying on the hardcoded fallback
    `docs/persona.md` being absent from the project root. The hardcoded
    fallback resolves CWD-relative (plan §D8); pytest typically runs from
    the project root, so the moment a human authors `docs/persona.md` (an
    expected outcome, see §R1), a test that depended on its absence would
    flip from passing to failing. This pin makes the test independent of
    project filesystem state.
    """
    captured: dict = {}
    real_build_plan = runner.build_plan_prompt

    def spy_build_plan(*args, **kwargs):
        captured["persona"] = kwargs.get("persona", "")
        return real_build_plan(*args, **kwargs)

    monkeypatch.setattr("claude_plan_execute.runner.build_plan_prompt", spy_build_plan)
    _no_run_claude_called(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            # Pin to a tmp_path-owned filename we never create, so the
            # missing-file path is independent of project filesystem state.
            "persona_file": str(tmp_path / "definitely_missing.md"),
        },
        cli={}, dry_run=True,
    )
    assert captured["persona"] == ""


def test_execute_task_appends_persona_extra_from_phase(monkeypatch, tmp_path):
    """Pin: when the task's phase has persona_extra in defaults, the runner
    loads that overlay and appends it after the global persona."""
    captured: dict = {}
    real_build_plan = runner.build_plan_prompt

    def spy_build_plan(*args, **kwargs):
        captured["persona"] = kwargs.get("persona", "")
        return real_build_plan(*args, **kwargs)

    monkeypatch.setattr("claude_plan_execute.runner.build_plan_prompt", spy_build_plan)
    _no_run_claude_called(monkeypatch)

    persona_file = tmp_path / "global.md"
    persona_file.write_text("GLOBAL_MARKER")
    extra_file = tmp_path / "phase-ui.md"
    extra_file.write_text("EXTRA_MARKER")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, phase="ui"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "persona_file": str(persona_file),
        },
        cli={},
        phases_by_id={
            "ui": {"id": "ui", "defaults": {"persona_extra": str(extra_file)}},
        },
        dry_run=True,
    )

    assert "GLOBAL_MARKER" in captured["persona"]
    assert "EXTRA_MARKER" in captured["persona"]
    g_idx = captured["persona"].index("GLOBAL_MARKER")
    e_idx = captured["persona"].index("EXTRA_MARKER")
    assert g_idx < e_idx


# ---------------------------------------------------------------------------
# K. Decisions injection (task 8)
# ---------------------------------------------------------------------------


def test_execute_task_loads_decisions_index_into_plan_prompt(monkeypatch, tmp_path):
    """Task 8: runner reads decisions_dir from the resolved dict, loads
    index.md, and passes the section to build_plan_prompt. Pin the wiring;
    the loader internals are tested in test_decisions.py."""
    captured: dict = {}
    real_build_plan = runner.build_plan_prompt

    def spy_build_plan(*args, **kwargs):
        captured["decisions"] = kwargs.get("decisions", "")
        return real_build_plan(*args, **kwargs)

    monkeypatch.setattr("claude_plan_execute.runner.build_plan_prompt", spy_build_plan)
    _no_run_claude_called(monkeypatch)

    decisions_dir = tmp_path / "decisions"
    decisions_dir.mkdir()
    (decisions_dir / "index.md").write_text("- [001](one.md) — First decision")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "decisions_dir": str(decisions_dir),
        },
        cli={}, dry_run=True,
    )

    assert "## Architectural decisions on file" in captured["decisions"]
    assert "First decision" in captured["decisions"]


def test_execute_task_passes_empty_decisions_when_index_missing(monkeypatch, tmp_path):
    """Pin: when decisions_dir/index.md doesn't exist, the runner runs
    cleanly (silent fallback) and the prompt builder gets decisions=''.

    Note: pin decisions_dir explicitly to a tmp_path-owned filename we
    never create, so the missing-index path is independent of the
    project's filesystem state. Same approach as task 7 §R1's persona
    test (`test_execute_task_passes_empty_persona_when_persona_file_missing`)."""
    captured: dict = {}
    real_build_plan = runner.build_plan_prompt

    def spy_build_plan(*args, **kwargs):
        captured["decisions"] = kwargs.get("decisions", "")
        return real_build_plan(*args, **kwargs)

    monkeypatch.setattr("claude_plan_execute.runner.build_plan_prompt", spy_build_plan)
    _no_run_claude_called(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "decisions_dir": str(tmp_path / "definitely_missing_decisions"),
        },
        cli={}, dry_run=True,
    )
    assert captured["decisions"] == ""


def test_execute_task_does_NOT_pass_decisions_to_review_prompt(monkeypatch, tmp_path):
    """Pin §D1: decisions are Plan-only. The Review builder must not
    receive a decisions kwarg.

    We can't directly assert "kwarg never passed" without monkeypatching
    the function call site. Easier: monkeypatch build_review_prompt with
    a strict signature that rejects unknown kwargs, and verify the
    review-phase code path completes without TypeError."""
    decisions_dir = tmp_path / "decisions"
    decisions_dir.mkdir()
    (decisions_dir / "index.md").write_text("- [001](one.md) — DECISIONS_MARKER")

    captured: dict = {}
    real_build_review = runner.build_review_prompt

    def strict_build_review(task, plan_path, review_path, prev_feedback="",
                            *, persona=""):
        # If runner passed decisions=..., this raises TypeError before
        # the call returns. The dry-run path returns inside the review
        # loop on round 1 before this matters, but we also assert that
        # the captured kwargs do NOT include decisions.
        captured["called"] = True
        return real_build_review(task, plan_path, review_path,
                                 prev_feedback, persona=persona)

    monkeypatch.setattr("claude_plan_execute.runner.build_review_prompt",
                        strict_build_review)
    _no_run_claude_called(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "decisions_dir": str(decisions_dir),
        },
        cli={}, skip_to="review", dry_run=True,
    )
    # No TypeError → review builder was called with the existing
    # signature (no decisions kwarg leaked).
    assert captured.get("called") is True


# ---------------------------------------------------------------------------
# L. Failure capture (task 9)
# ---------------------------------------------------------------------------

from claude_plan_execute.state import make_failure_record


# --- L.1: review-rounds exhaustion + fail records FailureRecord -----------


def test_max_rounds_with_fail_records_failure(monkeypatch, tmp_path):
    """Task 9 §D5: review-rounds exhaustion + fail writes a
    FailureRecord with failure_class=review_rejected_after_max_rounds."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "fail",
        },
        cli={}, skip_to="review",
    )
    attempts = state.get(1)["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["failure_class"] == "review_rejected_after_max_rounds"


def test_max_rounds_with_fail_failure_record_phase_is_review(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "fail",
        },
        cli={}, skip_to="review",
    )
    assert state.get(1)["attempts"][0]["phase"] == "review"


def test_max_rounds_with_fail_failure_record_includes_artifacts(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "fail",
        },
        cli={}, skip_to="review",
    )
    artifacts = state.get(1)["attempts"][0]["artifacts"]
    assert "plan" in artifacts
    assert artifacts["plan"].endswith("task-1/plan.md")
    assert "last_review" in artifacts
    assert artifacts["last_review"].endswith("task-1/review-1.md")


def test_max_rounds_with_fail_failure_record_captures_last_review_verbatim(
    monkeypatch, tmp_path,
):
    """When the last review file exists on disk, its body is captured
    into the FailureRecord's verbatim field."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    # Pre-write a review file at the path execute_task will use.
    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    review_path = task_dir / "review-1.md"
    review_path.write_text("## Verdict: NEEDS_REVISION\n\nMy specific complaint")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "fail",
        },
        cli={}, skip_to="review",
    )
    verbatim = state.get(1)["attempts"][0]["verbatim"]
    assert "My specific complaint" in verbatim


def test_max_rounds_with_fail_failure_record_verbatim_empty_when_review_missing(
    monkeypatch, tmp_path,
):
    """When the last review file doesn't exist, verbatim is empty (no raise)."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "fail",
        },
        cli={}, skip_to="review",
    )
    assert state.get(1)["attempts"][0]["verbatim"] == ""


def test_max_rounds_with_auto_approve_does_NOT_record_failure(
    monkeypatch, tmp_path,
):
    """Pin: auto_approve path falls through to Phase 3; no FailureRecord."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "auto_approve",
        },
        cli={}, skip_to="review",
    )
    assert state.get(1)["attempts"] == []


def test_max_rounds_with_fail_records_failure_BEFORE_returning(
    monkeypatch, tmp_path,
):
    """Pin: the FailureRecord is persisted before the function returns,
    so a subsequent State() reload sees it."""
    _no_run_claude_called(monkeypatch)
    state_path = tmp_path / "state.json"
    state = State(state_path)
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "fail",
        },
        cli={}, skip_to="review",
    )
    # Reload from disk to confirm persistence.
    fresh = State(state_path)
    assert len(fresh.get(1)["attempts"]) == 1
    assert fresh.get(1)["attempts"][0]["failure_class"] == \
        "review_rejected_after_max_rounds"


def test_max_rounds_with_fail_records_failure_when_max_review_rounds_is_zero(
    monkeypatch, tmp_path,
):
    """Pin §R26: max_review_rounds=0 + on_max_review_rounds=fail must
    record a FailureRecord with empty verbatim and no `last_review`
    artifact, without raising NameError on `round_num`."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 0,
            "on_max_review_rounds": "fail",
        },
        cli={}, skip_to="review",
    )
    attempts = state.get(1)["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["failure_class"] == "review_rejected_after_max_rounds"
    assert attempts[0]["verbatim"] == ""
    assert "last_review" not in attempts[0]["artifacts"]
    assert "plan" in attempts[0]["artifacts"]
    # Status transitions to blocked, just like the rounds>=1 path.
    assert state.get(1)["status"] == "blocked"
    assert state.get(1).get("block_reason") == "max_review_rounds_exhausted"


# --- L.2: implement timeout records FailureRecord -------------------------


def _patch_run_claude_simulating_timeout(monkeypatch):
    """Stub run_claude that simulates a timeout: sets the side-channel
    flag and returns the timeout sentinel (1, '')."""
    called = []

    def stub(*args, **kwargs):
        called.append((args, kwargs))
        claude._last_run_state.timed_out = True
        claude._last_run_state.subtype = None
        return 1, ""

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub)
    return called


def _patch_run_claude_simulating_max_turns(monkeypatch):
    """Stub run_claude that simulates a max-turns hit: sets the
    subtype side-channel flag."""
    called = []

    def stub(*args, **kwargs):
        called.append((args, kwargs))
        claude._last_run_state.timed_out = False
        claude._last_run_state.subtype = "error_max_turns"
        return 0, ""  # CLI exits cleanly even on max-turns

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub)
    return called


def _patch_run_claude_simulating_success(monkeypatch):
    """Stub run_claude that simulates a clean run."""
    called = []

    def stub(*args, **kwargs):
        called.append((args, kwargs))
        claude._last_run_state.timed_out = False
        claude._last_run_state.subtype = "success"
        return 0, "ok"

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub)
    return called


def test_implement_timeout_records_failure(monkeypatch, tmp_path):
    """Pin §D7: implementer_timeout FailureRecord written when
    claude._last_run_state.timed_out is True after Phase 3 run_claude."""
    _patch_run_claude_simulating_timeout(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    attempts = state.get(1)["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["failure_class"] == "implementer_timeout"
    assert attempts[0]["phase"] == "implement"


def test_implement_timeout_sets_blocked_status(monkeypatch, tmp_path):
    """Pin §D7: the task transitions to 'blocked', not 'done'."""
    _patch_run_claude_simulating_timeout(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry.get("block_reason") == "implementer_timeout"


def test_implement_timeout_includes_plan_artifact(monkeypatch, tmp_path):
    _patch_run_claude_simulating_timeout(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    artifacts = state.get(1)["attempts"][0]["artifacts"]
    assert "plan" in artifacts


# --- L.3: implement max-turns records FailureRecord -----------------------


def test_implement_max_turns_records_failure(monkeypatch, tmp_path):
    """Pin §D7: implementer_max_turns FailureRecord written when
    claude._last_run_state.subtype == 'error_max_turns'."""
    _patch_run_claude_simulating_max_turns(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    attempts = state.get(1)["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["failure_class"] == "implementer_max_turns"
    assert attempts[0]["phase"] == "implement"


def test_implement_max_turns_sets_blocked_status(monkeypatch, tmp_path):
    _patch_run_claude_simulating_max_turns(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry.get("block_reason") == "implementer_max_turns"


# --- L.4: implement success does NOT record FailureRecord -----------------


def test_implement_success_does_NOT_record_failure(monkeypatch, tmp_path):
    """Pin §D7: clean run reaches state.set_status(tid, 'done') and writes
    no FailureRecord."""
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert state.get(1)["attempts"] == []
    assert state.get(1)["status"] == "done"


def test_implement_does_NOT_record_failure_when_pre_set_flag_is_overridden(
    monkeypatch, tmp_path,
):
    """Stronger pin than 'flag is False after success': the stub sets
    `_last_run_timed_out = False` at call entry, overriding any prior
    value. Pre-set the flag to True before execute_task — the stub
    must reset it via its assignment, otherwise a stale flag from a
    previous call would falsely trigger the timeout branch."""
    claude._last_run_state.timed_out = True  # leftover from a prior call
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    # The stub explicitly sets _last_run_timed_out=False; the runner's
    # post-call check should NOT see True.
    assert state.get(1)["attempts"] == []
    assert state.get(1)["status"] == "done"


# --- L.5: handle_sigint records manual_stop --------------------------------


def test_handle_sigint_records_manual_stop_when_task_in_flight(
    monkeypatch, tmp_path,
):
    """Pin §D10: SIGINT during an active task records a manual_stop record."""
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    state = State(tmp_path / "state.json")
    state.set_status(1, "implementing")
    runner._current_state = state
    runner._set_current_task(1, "implement")
    with pytest.raises(SystemExit):
        handle_sigint(2, None)
    attempts = state.get(1)["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["failure_class"] == "manual_stop"
    assert attempts[0]["phase"] == "implement"


def test_handle_sigint_records_manual_stop_during_planning(monkeypatch, tmp_path):
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    state = State(tmp_path / "state.json")
    state.set_status(1, "planning")
    runner._current_state = state
    runner._set_current_task(1, "plan")
    with pytest.raises(SystemExit):
        handle_sigint(2, None)
    assert state.get(1)["attempts"][0]["phase"] == "plan"


def test_handle_sigint_does_NOT_record_when_status_is_done(monkeypatch, tmp_path):
    """Pin §D10: tasks already in terminal status are not over-recorded."""
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    state = State(tmp_path / "state.json")
    state.set_status(1, "done")
    runner._current_state = state
    runner._set_current_task(1, "plan")
    with pytest.raises(SystemExit):
        handle_sigint(2, None)
    assert state.get(1)["attempts"] == []


def test_handle_sigint_does_NOT_record_when_status_is_blocked(monkeypatch, tmp_path):
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    state = State(tmp_path / "state.json")
    state.set_status(1, "blocked", block_reason="implementer_timeout")
    runner._current_state = state
    runner._set_current_task(1, "plan")
    with pytest.raises(SystemExit):
        handle_sigint(2, None)
    assert state.get(1)["attempts"] == []


def test_handle_sigint_does_NOT_record_when_state_is_None(monkeypatch):
    """Pin §D10: SIGINT before any execute_task is a noop on the failure surface."""
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    runner._current_state = None
    # Per T13 §T4.5 sentinel-zero migration: drop the `_current_task_id = 0`
    # write entirely (autouse fixture's `_current_tasks.clear()` already
    # gives the empty-dict default).
    with pytest.raises(SystemExit):
        handle_sigint(2, None)


def test_handle_sigint_does_NOT_record_when_task_id_is_zero(monkeypatch, tmp_path):
    """Pin §D10: empty `_current_tasks` dict means no task in flight.
    Sentinel-zero semantics post-T13: no entry == "no task in flight".
    """
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    state = State(tmp_path / "state.json")
    runner._current_state = state
    # No _set_current_task call — autouse fixture's clear() leaves the dict empty.
    with pytest.raises(SystemExit):
        handle_sigint(2, None)
    assert state.data["tasks"] == {}


def test_handle_sigint_swallows_record_failure_exception(monkeypatch, tmp_path):
    """Pin §D10: signal handlers must never raise. record_failure I/O
    failure is silently absorbed; sys.exit(130) still fires."""
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    state = State(tmp_path / "state.json")
    state.set_status(1, "implementing")

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    state.record_failure = boom  # type: ignore[method-assign]
    runner._current_state = state
    runner._set_current_task(1, "implement")
    with pytest.raises(SystemExit) as exc_info:
        handle_sigint(2, None)
    assert exc_info.value.code == 130


def test_handle_sigint_still_exits_130(monkeypatch, tmp_path):
    """Sanity: the recording path does NOT change the SIGINT exit code."""
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    state = State(tmp_path / "state.json")
    state.set_status(1, "implementing")
    runner._current_state = state
    runner._set_current_task(1, "implement")
    with pytest.raises(SystemExit) as exc_info:
        handle_sigint(2, None)
    assert exc_info.value.code == 130


def test_handle_sigint_still_prints_resume_command(monkeypatch, capsys, tmp_path):
    """Sanity: the recording path does NOT change the user-facing print."""
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    state = State(tmp_path / "state.json")
    state.set_status(7, "implementing")
    runner._current_state = state
    runner._set_current_task(7, "implement")
    with pytest.raises(SystemExit):
        handle_sigint(2, None)
    out = capsys.readouterr().out
    assert "--task 7 --skip-to implement" in out


# --- L.6: rate-limit exhaustion records FailureRecord (v0.3 task 1) -------


def _patch_run_claude_simulating_rate_limit_exhausted(monkeypatch):
    """Stub that simulates the wrapper exhausting all retries: sets
    `_last_run_rate_limited=True`, returns (1, '')."""
    called = []

    def stub(*args, **kwargs):
        called.append((args, kwargs))
        claude._last_run_state.timed_out = False
        claude._last_run_state.subtype = None
        claude._last_run_state.rate_limited = True
        return 1, ""

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub)
    return called


def test_implement_rate_limit_records_failure(monkeypatch, tmp_path):
    """Pin §D6: rate_limited FailureRecord written when
    claude._last_run_state.rate_limited is True after Phase 3 run_claude."""
    _patch_run_claude_simulating_rate_limit_exhausted(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    attempts = state.get(1)["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["failure_class"] == "rate_limited"
    assert attempts[0]["phase"] == "implement"


def test_implement_rate_limit_sets_blocked_status(monkeypatch, tmp_path):
    _patch_run_claude_simulating_rate_limit_exhausted(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry.get("block_reason") == "rate_limited"


def test_implement_rate_limit_does_NOT_run_gates(monkeypatch, tmp_path):
    """Mirror the implementer_timeout test: rate-limit return short-circuits
    the gates phase."""
    _patch_run_claude_simulating_rate_limit_exhausted(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["tests"]},
        cli={}, skip_to="implement",
    )
    # No "gate failed" attempt was appended; only the rate-limit one.
    attempts = state.get(1)["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["failure_class"] == "rate_limited"


def test_implement_rate_limit_takes_priority_over_timeout(monkeypatch, tmp_path):
    """If both `_last_run_rate_limited=True` AND `_last_run_timed_out=True`
    were set (theoretically impossible by §D5 exit guards, but defensive),
    the rate-limit branch runs first."""
    def stub(*args, **kwargs):
        claude._last_run_state.timed_out = True       # spurious
        claude._last_run_state.rate_limited = True    # authoritative
        return 1, ""
    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    attempts = state.get(1)["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["failure_class"] == "rate_limited"


def test_plan_phase_rate_limit_records_failure(monkeypatch, tmp_path):
    """Plan-phase rate-limit exhaustion records a rate_limited
    FailureRecord with phase='plan'. The plan file was never written, so
    artifacts['plan'] is absent."""
    _patch_run_claude_simulating_rate_limit_exhausted(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={},
    )
    attempts = state.get(1)["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["failure_class"] == "rate_limited"
    assert attempts[0]["phase"] == "plan"
    assert "plan" not in attempts[0]["artifacts"]
    assert state.get(1)["status"] == "blocked"
    assert state.get(1).get("block_reason") == "rate_limited"


def test_review_phase_rate_limit_records_failure(monkeypatch, tmp_path):
    """Review-phase rate-limit exhaustion records phase='review'."""
    _patch_run_claude_simulating_rate_limit_exhausted(monkeypatch)
    plan_dir = tmp_path / "task-1"
    plan_dir.mkdir()
    (plan_dir / "plan.md").write_text("placeholder plan")
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="review",
    )
    attempts = state.get(1)["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["failure_class"] == "rate_limited"
    assert attempts[0]["phase"] == "review"
    assert "plan" in attempts[0]["artifacts"]


def test_rate_limit_failure_reason_includes_attempt_count(monkeypatch, tmp_path):
    """Pin: the reason string mentions the total attempts (retries + 1).
    With default max_rate_limit_retries=3 → '4 attempts'."""
    _patch_run_claude_simulating_rate_limit_exhausted(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    reason = state.get(1)["attempts"][0]["reason"]
    assert "4 attempts" in reason


def test_execute_task_resolved_config_includes_rate_limit_keys(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    cfg = _read_resolved_config(tmp_path, 1)
    assert cfg["max_rate_limit_retries"] == {"value": 3, "from": "hardcoded"}
    assert cfg["rate_limit_base_backoff_seconds"] == {"value": 60, "from": "hardcoded"}


def test_execute_task_resolved_config_rate_limit_from_caps(monkeypatch, tmp_path):
    """After loader.effective_defaults expands caps, the resolver labels
    them as project_defaults."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_rate_limit_retries": 5,           # post-effective_defaults
            "rate_limit_base_backoff_seconds": 30,
        },
        cli={}, dry_run=True,
    )
    cfg = _read_resolved_config(tmp_path, 1)
    assert cfg["max_rate_limit_retries"] == {"value": 5, "from": "project_defaults"}
    assert cfg["rate_limit_base_backoff_seconds"] == {"value": 30, "from": "project_defaults"}


def test_execute_task_passes_rate_limit_kwargs_to_run_claude(monkeypatch, tmp_path):
    captured = {}

    def stub(*args, **kwargs):
        captured.update(kwargs)
        claude._last_run_state.timed_out = False
        claude._last_run_state.subtype = "success"
        claude._last_run_state.rate_limited = False
        return 0, ""

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_rate_limit_retries": 7,
            "rate_limit_base_backoff_seconds": 45,
        },
        cli={}, skip_to="implement",
    )
    assert captured.get("max_rate_limit_retries") == 7
    assert captured.get("rate_limit_base_backoff_seconds") == 45


# ---------------------------------------------------------------------------
# M. Resume failure-context wiring (task 10)
# ---------------------------------------------------------------------------


def test_execute_task_passes_empty_failure_context_when_no_attempts(
    monkeypatch, tmp_path,
):
    """Pin: fresh task → builder receives failure_context=''."""
    captured: dict = {}
    real_build_plan = runner.build_plan_prompt

    def spy_build_plan(*args, **kwargs):
        captured["fc"] = kwargs.get("failure_context", "")
        return real_build_plan(*args, **kwargs)

    monkeypatch.setattr("claude_plan_execute.runner.build_plan_prompt", spy_build_plan)
    _no_run_claude_called(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    assert captured["fc"] == ""


def test_execute_task_passes_failure_context_when_attempts_present(
    monkeypatch, tmp_path,
):
    """Pin: state with prior attempts → builder receives a non-empty section
    that names the last failure's phase."""
    captured: dict = {}
    real_build_plan = runner.build_plan_prompt

    def spy_build_plan(*args, **kwargs):
        captured["fc"] = kwargs.get("failure_context", "")
        return real_build_plan(*args, **kwargs)

    monkeypatch.setattr("claude_plan_execute.runner.build_plan_prompt", spy_build_plan)
    _no_run_claude_called(monkeypatch)

    state = State(tmp_path / "state.json")
    state.record_failure(1, make_failure_record(
        "implementer_timeout", "implement",
        reason="prior implementer timed out",
        verbatim="stderr from prior run",
        artifacts={"plan": "plans/task-1/plan.md"},
    ))

    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    fc = captured["fc"]
    assert "[FAILURE CONTEXT" in fc
    assert "phase: implement" in fc
    assert "implementer_timeout" in fc
    assert "prior implementer timed out" in fc


def test_execute_task_passes_failure_context_to_implement_builder(
    monkeypatch, tmp_path,
):
    """Pin: the same section reaches the Implement builder, not just Plan."""
    captured: dict = {}
    real_build_impl = runner.build_implement_prompt

    def spy_build_impl(*args, **kwargs):
        captured["fc"] = kwargs.get("failure_context", "")
        return real_build_impl(*args, **kwargs)

    monkeypatch.setattr("claude_plan_execute.runner.build_implement_prompt", spy_build_impl)
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    state.record_failure(1, make_failure_record(
        "manual_stop", "implement",
        reason="user-aborted prior run",
    ))

    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert "[FAILURE CONTEXT" in captured["fc"]
    assert "manual_stop" in captured["fc"]


def test_execute_task_does_NOT_pass_failure_context_to_review_builder(
    monkeypatch, tmp_path,
):
    """Pin §D11 / §D2: failure context is Plan/Implement only. The Review
    builder must not receive a failure_context kwarg.

    Mirrors the task-8 'decisions don't leak to review' regression — a
    strict-signature spy raises TypeError if a failure_context kwarg
    leaks in.
    """
    state = State(tmp_path / "state.json")
    state.record_failure(1, make_failure_record(
        "review_rejected_after_max_rounds", "review", reason="x",
    ))

    captured: dict = {}
    real_build_review = runner.build_review_prompt

    def strict_build_review(task, plan_path, review_path, prev_feedback="",
                            *, persona=""):
        captured["called"] = True
        return real_build_review(task, plan_path, review_path,
                                 prev_feedback, persona=persona)

    monkeypatch.setattr("claude_plan_execute.runner.build_review_prompt",
                        strict_build_review)
    _no_run_claude_called(monkeypatch)

    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "fail",
        },
        cli={}, skip_to="review",
    )
    assert captured.get("called") is True


def test_execute_task_does_NOT_pass_failure_context_to_revise_builder(
    monkeypatch, tmp_path,
):
    """Pin §D11 / §D2: same regression for Revise — narrow scope wins."""
    state = State(tmp_path / "state.json")
    state.record_failure(1, make_failure_record(
        "review_rejected_after_max_rounds", "review", reason="x",
    ))

    captured: dict = {}
    real_build_revise = runner.build_revise_prompt

    def strict_build_revise(task, plan_path, review_path, *, persona=""):
        captured["called"] = True
        return real_build_revise(task, plan_path, review_path, persona=persona)

    monkeypatch.setattr("claude_plan_execute.runner.build_revise_prompt",
                        strict_build_revise)

    _no_run_claude_called(monkeypatch)

    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 2,
            "on_max_review_rounds": "fail",
        },
        cli={}, skip_to="review",
    )
    assert captured.get("called") is True


def test_execute_task_failure_context_renders_two_attempts(
    monkeypatch, tmp_path,
):
    """Integration pin: two prior FailureRecords on disk → both are
    rendered in the section, in the FailureRecord-list order."""
    captured: dict = {}
    real_build_plan = runner.build_plan_prompt

    def spy_build_plan(*args, **kwargs):
        captured["fc"] = kwargs.get("failure_context", "")
        return real_build_plan(*args, **kwargs)

    monkeypatch.setattr("claude_plan_execute.runner.build_plan_prompt", spy_build_plan)
    _no_run_claude_called(monkeypatch)

    state = State(tmp_path / "state.json")
    state.record_failure(1, make_failure_record(
        "manual_stop", "plan", reason="OLDER_REASON",
    ))
    state.record_failure(1, make_failure_record(
        "implementer_timeout", "implement",
        reason="NEWER_REASON",
        artifacts={"plan": "plans/task-1/plan.md"},
    ))

    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    fc = captured["fc"]
    assert "Attempt 1: manual_stop" in fc
    assert "Attempt 2: implementer_timeout" in fc
    assert "OLDER_REASON" in fc
    assert "NEWER_REASON" in fc
    assert "phase: implement" in fc


def test_execute_task_failure_context_persists_across_plan_to_implement(
    monkeypatch, tmp_path,
):
    """Pin §D9: the same snapshot section reaches both builders in one
    execute_task call (no re-read between phases)."""
    fc_at_plan: dict = {}
    fc_at_impl: dict = {}
    real_build_plan = runner.build_plan_prompt
    real_build_impl = runner.build_implement_prompt

    def spy_plan(*args, **kwargs):
        fc_at_plan["v"] = kwargs.get("failure_context", "")
        return real_build_plan(*args, **kwargs)

    def spy_impl(*args, **kwargs):
        fc_at_impl["v"] = kwargs.get("failure_context", "")
        return real_build_impl(*args, **kwargs)

    monkeypatch.setattr("claude_plan_execute.runner.build_plan_prompt", spy_plan)
    monkeypatch.setattr("claude_plan_execute.runner.build_implement_prompt", spy_impl)
    _patch_run_claude_simulating_success(monkeypatch)

    (tmp_path / "task-1").mkdir(parents=True, exist_ok=True)
    (tmp_path / "task-1" / "plan.md").write_text("placeholder plan")

    state = State(tmp_path / "state.json")
    state.record_failure(1, make_failure_record(
        "implementer_max_turns", "implement",
        reason="prior max-turns",
    ))

    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 0,
            "on_max_review_rounds": "auto_approve",
        },
        cli={},
    )
    assert fc_at_plan["v"] != ""
    assert fc_at_plan["v"] == fc_at_impl["v"]
    assert "implementer_max_turns" in fc_at_plan["v"]


# ---------------------------------------------------------------------------
# N. Gate integration in execute_task (task 11)
# ---------------------------------------------------------------------------

from claude_plan_execute.gates import (
    Gate,
    GateRegistry,
    GateResult,
)


@pytest.fixture
def _clean_default_registry(monkeypatch):
    """Replace the module-level default registry with a fresh local one
    for the duration of a test. Tests that need built-ins register them
    explicitly within the fixture's scope. Also clears
    `_loaded_invariant_names` so each Section Q test sees only the
    invariants IT registers via the loader."""
    fresh = GateRegistry()
    monkeypatch.setattr(
        "claude_plan_execute.runner.default_registry", lambda: fresh,
    )
    monkeypatch.setattr(
        "claude_plan_execute.gates.default_registry", lambda: fresh,
    )
    monkeypatch.setattr(
        "claude_plan_execute.gates.invariants._loaded_invariant_names",
        [],
    )
    yield fresh


def test_execute_task_with_no_gates_configured_reaches_done(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D12 / §D13: empty gates list → Phase 4 is skipped → done."""
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},  # NO 'gates' key
        cli={}, skip_to="implement",
    )
    assert state.get(1)["status"] == "done"
    assert state.get(1)["attempts"] == []


def test_execute_task_with_unknown_gate_name_warns_and_proceeds_to_done(
    monkeypatch, tmp_path, capsys, _clean_default_registry,
):
    """Pin §D13: unknown gate name → warning printed, runner reaches done."""
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["definitely-unknown"]},
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    assert "Unknown gate(s)" in out
    assert "definitely-unknown" in out
    assert state.get(1)["status"] == "done"


def test_execute_task_with_passing_gate_reaches_done(
    monkeypatch, tmp_path, _clean_default_registry,
):
    _patch_run_claude_simulating_success(monkeypatch)
    _clean_default_registry.register(
        "ok-gate", Gate(kind="shell", command=("true",)),
    )

    # Stub subprocess.run inside gates.shell so the gate "passes."
    def fake_run(cmd, **kwargs):
        class P:
            pass
        p = P()
        p.returncode = 0
        p.stdout = "all good\n"
        p.stderr = ""
        return p
    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["ok-gate"]},
        cli={}, skip_to="implement",
    )
    assert state.get(1)["status"] == "done"
    assert state.get(1)["attempts"] == []
    assert (tmp_path / "task-1" / "gates" / "ok-gate.txt").is_file()


def test_execute_task_with_blocking_gate_failure_blocks_and_records(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D12: a `block`-mode gate failing transitions to blocked + records."""
    _patch_run_claude_simulating_success(monkeypatch)
    _clean_default_registry.register(
        "bad-gate",
        Gate(kind="shell", command=("false",), on_failure="block"),
    )

    def fake_run(cmd, **kwargs):
        class P:
            pass
        p = P()
        p.returncode = 1
        p.stdout = ""
        p.stderr = "test failed: 3 errors\n"
        return p
    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["bad-gate"]},
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry["block_reason"] == "gate_failed"
    assert len(entry["attempts"]) == 1
    rec = entry["attempts"][0]
    assert rec["failure_class"] == "gate_failed"
    assert rec["phase"] == "gate"
    assert "test failed: 3 errors" in rec["verbatim"]
    assert "plan" in rec["artifacts"]
    assert "gates_dir" in rec["artifacts"]
    assert "gate:bad-gate" in rec["artifacts"]


def test_execute_task_with_warn_gate_failure_does_NOT_block(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin: `warn`-mode gate failing → still reaches done. No FailureRecord."""
    _patch_run_claude_simulating_success(monkeypatch)
    _clean_default_registry.register(
        "warn-only",
        Gate(kind="shell", command=("false",), on_failure="warn"),
    )

    def fake_run(cmd, **kwargs):
        class P:
            pass
        p = P()
        p.returncode = 1
        p.stdout = "warning text"
        p.stderr = ""
        return p
    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["warn-only"]},
        cli={}, skip_to="implement",
    )
    assert state.get(1)["status"] == "done"
    assert state.get(1)["attempts"] == []


def test_execute_task_phase_gates_extra_appended_to_project_gates(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D12: phases[].defaults.gates_extra appended to defaults.gates."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[str] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(cmd[0])

        class P:
            pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p
    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    _clean_default_registry.register("g-proj",  Gate(kind="shell", command=("PROJ",)))
    _clean_default_registry.register("g-phase", Gate(kind="shell", command=("PHASE",)))

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, phase="ui"), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["g-proj"]},
        cli={},
        phases_by_id={"ui": {"id": "ui", "defaults": {"gates_extra": ["g-phase"]}}},
        skip_to="implement",
    )
    assert sorted(invoked) == ["PHASE", "PROJ"]


def test_execute_task_task_gates_extra_runs_with_no_project_or_phase(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """v0.3 task 4 §2.21: a task-level gates_extra runs even when
    defaults.gates is empty and no phase overlay is configured."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[str] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:
            invoked.append(cmd[0])

        class P:
            pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p
    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    _clean_default_registry.register(
        "g-task", Gate(kind="shell", command=("TASK",)),
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, gates_extra=["g-task"]), state,
        file_defaults={"plans_dir": str(tmp_path)},  # no 'gates' key
        cli={}, skip_to="implement",
    )
    assert invoked == ["TASK"]
    assert state.get(1)["status"] == "done"


def test_execute_task_task_gates_extra_appended_to_phase_gates_extra(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """v0.3 task 4 §2.21: task gates_extra is appended after phase
    gates_extra. Mirrors test_execute_task_phase_gates_extra_appended_
    to_project_gates at line 2796."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[str] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:
            invoked.append(cmd[0])

        class P:
            pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p
    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    _clean_default_registry.register(
        "g-phase", Gate(kind="shell", command=("PHASE",)),
    )
    _clean_default_registry.register(
        "g-task",  Gate(kind="shell", command=("TASK",)),
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, phase="ui", gates_extra=["g-task"]), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={},
        phases_by_id={"ui": {"id": "ui",
                             "defaults": {"gates_extra": ["g-phase"]}}},
        skip_to="implement",
    )
    assert sorted(invoked) == ["PHASE", "TASK"]
    assert state.get(1)["status"] == "done"


def test_execute_task_three_layer_gates_concat_order_is_project_phase_task(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """v0.3 task 4 §2.21: the resolved gate list passed into run_gates is
    project + phase + task in that order. Pin via a run_gates spy because
    parallel_safe shell gates run in a ThreadPoolExecutor and subprocess
    invocation order can't pin concat order."""
    _patch_run_claude_simulating_success(monkeypatch)
    _clean_default_registry.register("g-proj",
        Gate(kind="shell", command=("PROJ",)))
    _clean_default_registry.register("g-phase",
        Gate(kind="shell", command=("PHASE",)))
    _clean_default_registry.register("g-task",
        Gate(kind="shell", command=("TASK",)))

    captured: dict = {}
    real_run_gates = runner.run_gates

    def spy(resolved_gates, **kwargs):
        captured["names"] = [n for (n, _g) in resolved_gates]
        return real_run_gates(resolved_gates, **kwargs)
    monkeypatch.setattr("claude_plan_execute.runner.run_gates", spy)

    # Stub the actual gate execution so the spy result is the only
    # thing we assert on.
    def fake_run(cmd, **kwargs):
        class P:
            pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p
    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, phase="ui", gates_extra=["g-task"]), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["g-proj"]},
        cli={},
        phases_by_id={"ui": {"id": "ui",
                             "defaults": {"gates_extra": ["g-phase"]}}},
        skip_to="implement",
    )
    assert captured["names"] == ["g-proj", "g-phase", "g-task"]
    assert state.get(1)["status"] == "done"


def test_execute_task_task_gates_extra_unknown_name_warns_and_proceeds(
    monkeypatch, tmp_path, capsys, _clean_default_registry,
):
    """v0.3 task 4 §2.21: a typo in a task-level gate NAME (the field
    itself is correctly spelled) produces the existing 'Unknown gate(s)
    skipped' warning, not a hard error. Mirrors
    test_execute_task_with_unknown_gate_name_warns_and_proceeds_to_done
    at line 2674; only difference is the unknown name lives in
    task.gates_extra rather than defaults.gates."""
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, gates_extra=["definitely-unknown"]), state,
        file_defaults={"plans_dir": str(tmp_path)},  # no project gates
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    assert "Unknown gate(s)" in out
    assert "definitely-unknown" in out
    assert state.get(1)["status"] == "done"


def test_execute_task_blocking_gate_does_NOT_set_done(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Regression: status='done' must NOT be set after a block-failure."""
    _patch_run_claude_simulating_success(monkeypatch)
    _clean_default_registry.register(
        "blocker", Gate(kind="shell", command=("false",), on_failure="block"),
    )

    def fake_run(cmd, **kwargs):
        class P:
            pass
        p = P()
        p.returncode = 1
        p.stdout = ""
        p.stderr = ""
        return p
    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["blocker"]},
        cli={}, skip_to="implement",
    )
    assert state.get(1)["status"] == "blocked"
    assert state.get(1)["status"] != "done"


def test_execute_task_gate_run_skipped_when_implementer_timed_out(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D12: implementer_timeout path returns BEFORE the gate run."""
    invoked = []

    def fake_run(cmd, **kwargs):
        invoked.append(cmd)

        class P:
            pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p
    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    _clean_default_registry.register(
        "x", Gate(kind="shell", command=("X",)),
    )
    _patch_run_claude_simulating_timeout(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["x"]},
        cli={}, skip_to="implement",
    )
    # Gate must NOT have been invoked.
    assert invoked == []
    assert state.get(1)["status"] == "blocked"
    assert state.get(1)["block_reason"] == "implementer_timeout"


def test_execute_task_gate_run_skipped_when_implementer_max_turns(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin: max_turns path also returns before the gate run."""
    invoked = []

    def fake_run(cmd, **kwargs):
        invoked.append(cmd)

        class P:
            pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p
    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    _clean_default_registry.register(
        "x", Gate(kind="shell", command=("X",)),
    )
    _patch_run_claude_simulating_max_turns(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["x"]},
        cli={}, skip_to="implement",
    )
    assert invoked == []
    assert state.get(1)["status"] == "blocked"
    assert state.get(1)["block_reason"] == "implementer_max_turns"


def test_execute_task_dry_run_does_NOT_run_gates(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin: dry_run returns from execute_task before reaching Phase 4."""
    invoked = []

    def fake_run(cmd, **kwargs):
        invoked.append(cmd)

        class P:
            pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p
    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    _clean_default_registry.register(
        "x", Gate(kind="shell", command=("X",)),
    )

    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["x"]},
        cli={}, skip_to="implement", dry_run=True,
    )
    assert invoked == []


# ---------------------------------------------------------------------------
# O. `tests` built-in gate integration (task 12)
# ---------------------------------------------------------------------------

# These tests do NOT use the _clean_default_registry fixture. The runner's
# special-case for `tests` (task 12 §D3) constructs the Gate via
# make_tests_gate(test_command) WITHOUT consulting default_registry —
# the fixture's emptiness is irrelevant for the `tests` path. (See §D9.)


def test_execute_task_tests_gate_runs_resolved_test_command(monkeypatch, tmp_path):
    """Pin §D2 / §D3: when 'tests' is in the gate list, the runner
    constructs the Gate with command=('bash', '-c', test_command) using
    the resolved test_command from the resolver chain."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["tests"],
            "test_command": "pytest -q",
        },
        cli={}, skip_to="implement",
    )
    assert invoked == [["bash", "-c", "pytest -q"]]
    assert state.get(1)["status"] == "done"


def test_execute_task_tests_gate_uses_per_task_test_command(monkeypatch, tmp_path):
    """Pin §D3: per-task test_command override is what the gate runs."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, test_command="rspec --fail-fast"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["tests"],
            "test_command": "pytest",
        },
        cli={}, skip_to="implement",
    )
    assert invoked == [["bash", "-c", "rspec --fail-fast"]]


def test_execute_task_tests_gate_failure_blocks_and_records(monkeypatch, tmp_path):
    """Pin §D3 / task 11 §D12: a failing tests gate blocks the task
    with block_reason='gate_failed' and records a FailureRecord."""
    _patch_run_claude_simulating_success(monkeypatch)

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 1
        p.stdout = "FAILED tests/test_x.py::test_y\n"
        p.stderr = "1 failed, 0 passed\n"
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["tests"],
            "test_command": "pytest",
        },
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry["block_reason"] == "gate_failed"
    assert len(entry["attempts"]) == 1
    rec = entry["attempts"][0]
    assert rec["failure_class"] == "gate_failed"
    assert rec["phase"] == "gate"
    assert "FAILED tests/test_x.py::test_y" in rec["verbatim"]
    assert "gate:tests" in rec["artifacts"]


def test_execute_task_tests_gate_writes_output_file(monkeypatch, tmp_path):
    """Pin task 11 §D8: the gate's output is captured to
    plans/task-{id}/gates/tests.txt."""
    _patch_run_claude_simulating_success(monkeypatch)

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = "OUTPUT_MARKER\n"
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["tests"],
            "test_command": "pytest",
        },
        cli={}, skip_to="implement",
    )
    out_file = tmp_path / "task-1" / "gates" / "tests.txt"
    assert out_file.is_file()
    assert "OUTPUT_MARKER" in out_file.read_text()


def test_execute_task_tests_name_does_NOT_warn_as_unknown(
    monkeypatch, tmp_path, capsys,
):
    """Pin §D3: the runner's special-case prevents 'tests' from being
    reported as an unknown-gate name. (Task 11's empty-registry path
    would have warned for 'tests'; task 12 makes the warning silence.)"""
    _patch_run_claude_simulating_success(monkeypatch)

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["tests"],
            "test_command": "pytest",
        },
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    assert "Unknown gate(s) skipped" not in out
    assert "tests" not in out or "Phase 4: Gates" in out  # may print 'tests: PASS'


def test_execute_task_tests_gate_skipped_when_implementer_timed_out(
    monkeypatch, tmp_path,
):
    """Pin task 11 §D12 carry-forward: implementer_timeout returns
    BEFORE the gate run, so the tests gate does not fire."""
    invoked = []

    def fake_run(cmd, **kwargs):
        invoked.append(cmd)
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_timeout(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["tests"],
            "test_command": "pytest",
        },
        cli={}, skip_to="implement",
    )
    assert invoked == []
    assert state.get(1)["status"] == "blocked"
    assert state.get(1)["block_reason"] == "implementer_timeout"


def test_execute_task_tests_gate_runs_alongside_unconfigured_lint(
    monkeypatch, tmp_path, capsys,
):
    """Integration with task 13: 'tests' resolves via the runner's
    special-case; 'lint' warns and skips because lint_command is
    unconfigured (None). Both coexist without the runner crashing.
    Renamed from `..._unknown_lint` after task 13 — the lint name
    is now known, but its command is None."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["tests", "lint"],     # mirrors the repo's tasks.yaml
            "test_command": "pytest",
            # No lint_command — lint should warn and skip.
        },
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    # tests gate ran:
    assert invoked == [["bash", "-c", "pytest"]]
    # lint warned (lint_command is unconfigured):
    assert "Skipping gate 'lint': no defaults.lint_command configured." in out
    # task reaches done because the unconfigured-lint warning is non-blocking:
    assert state.get(1)["status"] == "done"


def test_execute_task_tests_gate_phase_extra_appended(monkeypatch, tmp_path):
    """Pin task 11 §D12 + §D3: the runner walks gate_names in order;
    a phase's gates_extra='tests' is appended to project gates and runs
    after them. (Phase ordering doesn't change the tests gate's
    behavior — both invocations produce the same Gate via the factory.)"""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, phase="ui"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": [],          # nothing at project level
            "test_command": "pytest",
        },
        cli={},
        phases_by_id={"ui": {"id": "ui", "defaults": {"gates_extra": ["tests"]}}},
        skip_to="implement",
    )
    assert invoked == [["bash", "-c", "pytest"]]
    assert state.get(1)["status"] == "done"


def test_execute_task_tests_gate_does_NOT_pass_test_command_to_implement_prompt(
    monkeypatch, tmp_path,
):
    """Pin §D4 / §D10: the runner stops passing test_command to
    build_implement_prompt. We assert the spy receives the new
    signature (commit_msg in the third positional, no test_command)."""
    captured: dict = {}
    real_build_impl = runner.build_implement_prompt

    def spy(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return real_build_impl(*args, **kwargs)

    monkeypatch.setattr(
        "claude_plan_execute.runner.build_implement_prompt", spy,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, commit_message="test-commit-msg"), state,
        file_defaults={"plans_dir": str(tmp_path), "test_command": "pytest"},
        cli={}, skip_to="implement",
    )
    # Positional args: (task, plan_path, commit_msg). Verify length and
    # types — the third arg is the commit message, NOT a test_command.
    args = captured["args"]
    assert len(args) == 3
    assert args[0]["id"] == 1                      # task dict
    assert args[1].endswith("task-1/plan.md")      # plan_path
    assert args[2] == "test-commit-msg"            # commit_msg
    # And test_command is not anywhere in the kwargs:
    assert "test_command" not in captured["kwargs"]


# ---------------------------------------------------------------------------
# P. `lint` built-in gate integration (task 13)
# ---------------------------------------------------------------------------

# These tests do NOT use the _clean_default_registry fixture. The runner's
# special-case for `lint` (task 13 §D3) constructs the Gate via
# make_lint_gate(lint_command) WITHOUT consulting default_registry —
# the fixture's emptiness is irrelevant for the `lint` path. (See §D13.)


def test_execute_task_lint_gate_runs_resolved_lint_command(monkeypatch, tmp_path):
    """Pin §D2 / §D3: when 'lint' is in the gate list and lint_command
    is set, the runner constructs the Gate with command=('bash', '-c',
    lint_command) using the resolved value from the resolver chain."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["lint"],
            "lint_command": "ruff check",
        },
        cli={}, skip_to="implement",
    )
    assert invoked == [["bash", "-c", "ruff check"]]
    assert state.get(1)["status"] == "done"


def test_execute_task_lint_gate_uses_per_task_lint_command(monkeypatch, tmp_path):
    """Pin §D6: per-task lint_command override is what the gate runs."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, lint_command="eslint --max-warnings 0 ."), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["lint"],
            "lint_command": "ruff check",
        },
        cli={}, skip_to="implement",
    )
    assert invoked == [["bash", "-c", "eslint --max-warnings 0 ."]]


def test_execute_task_lint_gate_failure_blocks_and_records(monkeypatch, tmp_path):
    """Pin §D10 / task 11 §D12: a failing lint gate blocks the task
    with block_reason='gate_failed' and records a FailureRecord."""
    _patch_run_claude_simulating_success(monkeypatch)

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 1
        p.stdout = "src/foo.py:42: E501 line too long\n"
        p.stderr = "1 error\n"
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["lint"],
            "lint_command": "ruff check",
        },
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry["block_reason"] == "gate_failed"
    assert len(entry["attempts"]) == 1
    rec = entry["attempts"][0]
    assert rec["failure_class"] == "gate_failed"
    assert rec["phase"] == "gate"
    assert "src/foo.py:42: E501" in rec["verbatim"]
    assert "gate:lint" in rec["artifacts"]


def test_execute_task_lint_gate_writes_output_file(monkeypatch, tmp_path):
    """Pin task 11 §D8: the gate's output is captured to
    plans/task-{id}/gates/lint.txt."""
    _patch_run_claude_simulating_success(monkeypatch)

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = "LINT_OUTPUT_MARKER\n"
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["lint"],
            "lint_command": "ruff check",
        },
        cli={}, skip_to="implement",
    )
    out_file = tmp_path / "task-1" / "gates" / "lint.txt"
    assert out_file.is_file()
    assert "LINT_OUTPUT_MARKER" in out_file.read_text()


def test_execute_task_lint_unconfigured_warns_and_skips(
    monkeypatch, tmp_path, capsys,
):
    """Pin §D2: lint_command is None (no layer set it) → runner prints
    the unconfigured-lint warning and does NOT add the gate to the
    resolved list. The task still reaches done."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["lint"],
            # NO lint_command — None across the chain.
        },
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    assert "Skipping gate 'lint': no defaults.lint_command configured." in out
    assert invoked == []                          # gate never built
    assert state.get(1)["status"] == "done"
    # No FailureRecord written for the warn-skip path:
    assert state.get(1)["attempts"] == []


def test_execute_task_lint_explicit_empty_command_runs_noop(
    monkeypatch, tmp_path,
):
    """Pin §D2 / §D5: an explicit defaults.lint_command: '' (empty
    string) is a valid 'no-op' choice — the factory builds the Gate
    and `bash -c ''` exits 0. Distinguishes from the None case
    (which warns and skips)."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["lint"],
            "lint_command": "",                   # explicit empty string
        },
        cli={}, skip_to="implement",
    )
    assert invoked == [["bash", "-c", ""]]
    assert state.get(1)["status"] == "done"


def test_execute_task_lint_unconfigured_warning_does_not_double_print(
    monkeypatch, tmp_path, capsys,
):
    """Pin §D2: the warning fires once even if `lint` is listed
    multiple times (e.g., gates: [lint, lint])."""
    _patch_run_claude_simulating_success(monkeypatch)

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["lint", "lint"],
        },
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    assert out.count("Skipping gate 'lint':") == 1


def test_execute_task_lint_name_does_NOT_warn_as_unknown(
    monkeypatch, tmp_path, capsys,
):
    """Pin §D3: the runner's special-case prevents 'lint' from being
    reported as an unknown-gate name. Pre-task-13 the empty-registry
    path warned for 'lint'; task 13 makes the unknown-gate warning
    silent for the literal name 'lint'."""
    _patch_run_claude_simulating_success(monkeypatch)

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["lint"],
            "lint_command": "ruff check",
        },
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    assert "Unknown gate(s) skipped" not in out


def test_execute_task_lint_gate_skipped_when_implementer_timed_out(
    monkeypatch, tmp_path,
):
    """Pin task 11 §D12 carry-forward: implementer_timeout returns
    BEFORE the gate run, so the lint gate does not fire."""
    invoked = []

    def fake_run(cmd, **kwargs):
        invoked.append(cmd)
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_timeout(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["lint"],
            "lint_command": "ruff check",
        },
        cli={}, skip_to="implement",
    )
    assert invoked == []
    assert state.get(1)["status"] == "blocked"
    assert state.get(1)["block_reason"] == "implementer_timeout"


def test_execute_task_lint_runs_alongside_tests_in_parallel(
    monkeypatch, tmp_path,
):
    """Pin §D10: both `tests` and `lint` are parallel_safe → they end up
    in the parallel pool. We can't test concurrency deterministically
    in unit tests, but we can verify both subprocess.run invocations
    happen and both gate output files are written."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = "ok\n"
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["tests", "lint"],
            "test_command": "pytest -q",
            "lint_command": "ruff check",
        },
        cli={}, skip_to="implement",
    )
    cmds = sorted(c[2] for c in invoked)
    assert cmds == ["pytest -q", "ruff check"]
    assert (tmp_path / "task-1" / "gates" / "tests.txt").is_file()
    assert (tmp_path / "task-1" / "gates" / "lint.txt").is_file()
    assert state.get(1)["status"] == "done"


def test_execute_task_lint_phase_extra_appended(monkeypatch, tmp_path):
    """Pin task 11 §D12 + §D3: a phase's gates_extra='lint' is
    appended to project gates and resolves through the same
    special-case dispatch."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, phase="ui"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": [],
            "lint_command": "ruff check",
        },
        cli={},
        phases_by_id={"ui": {"id": "ui", "defaults": {"gates_extra": ["lint"]}}},
        skip_to="implement",
    )
    assert invoked == [["bash", "-c", "ruff check"]]
    assert state.get(1)["status"] == "done"


# ---------------------------------------------------------------------------
# Q. `invariant-grep` meta-name + invariants.yaml integration (task 14)
# ---------------------------------------------------------------------------


def _write_invariants_file(tmp_path, body: str) -> Path:
    p = tmp_path / "invariants.yaml"
    p.write_text(body)
    return p


def test_execute_task_invariant_grep_meta_name_expands_to_loaded_invariants(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D3: gates: ['invariant-grep'] expands to every loaded
    invariant. Each compiled gate runs once."""
    from claude_plan_execute.gates import default_registry
    from claude_plan_execute.gates.invariants import (
        load_invariants,
        register_invariants_on,
    )
    p = _write_invariants_file(tmp_path,
        "- name: invariant-foo\n"
        "  command: 'true'\n"
        "- name: invariant-bar\n"
        "  command: 'true'\n"
    )
    register_invariants_on(default_registry(), load_invariants(p))

    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        x = P(); x.returncode = 0; x.stdout = ""; x.stderr = ""
        return x

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["invariant-grep"],
        },
        cli={}, skip_to="implement",
    )
    cmds = sorted(c[2] for c in invoked)
    assert cmds == ["true", "true"]
    assert (tmp_path / "task-1" / "gates" / "invariant-foo.txt").is_file()
    assert (tmp_path / "task-1" / "gates" / "invariant-bar.txt").is_file()
    assert state.get(1)["status"] == "done"


def test_execute_task_invariant_grep_with_no_loaded_invariants_silently_noops(
    monkeypatch, tmp_path, _clean_default_registry, capsys,
):
    """Pin §D3: invariant-grep meta-name with empty loaded set is a
    silent no-op (no warning, no skipped gate). The task still
    reaches done."""
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["invariant-grep"],
        },
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    assert "Unknown gate(s)" not in out
    assert state.get(1)["status"] == "done"


def test_execute_task_invariant_grep_after_tests_runs_in_dispatch_order(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D3: gates: [tests, invariant-grep, lint] order is
    preserved in dispatch — tests gate first, then expanded
    invariants in file order, then lint."""
    from claude_plan_execute.gates import default_registry
    from claude_plan_execute.gates.invariants import (
        load_invariants,
        register_invariants_on,
    )
    p = _write_invariants_file(tmp_path,
        "- name: invariant-a\n"
        "  command: 'echo INVA'\n"
        "- name: invariant-b\n"
        "  command: 'echo INVB'\n"
    )
    register_invariants_on(default_registry(), load_invariants(p))

    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[str] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(cmd[2])
        class P: pass
        x = P(); x.returncode = 0; x.stdout = ""; x.stderr = ""
        return x

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["tests", "invariant-grep", "lint"],
            "test_command": "pytest",
            "lint_command": "ruff",
        },
        cli={}, skip_to="implement",
    )
    assert sorted(invoked) == sorted(["pytest", "echo INVA", "echo INVB", "ruff"])


def test_execute_task_specific_invariant_name_runs_directly(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D3: a user can list a specific invariant by name in
    gates:; the registry-lookup branch handles it without going
    through the meta-name."""
    from claude_plan_execute.gates import default_registry
    from claude_plan_execute.gates.invariants import (
        load_invariants,
        register_invariants_on,
    )
    p = _write_invariants_file(tmp_path,
        "- name: invariant-foo\n"
        "  command: 'echo FOO'\n"
        "- name: invariant-bar\n"
        "  command: 'echo BAR'\n"
    )
    register_invariants_on(default_registry(), load_invariants(p))

    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[str] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(cmd[2])
        class P: pass
        x = P(); x.returncode = 0; x.stdout = ""; x.stderr = ""
        return x

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["invariant-foo"],
        },
        cli={}, skip_to="implement",
    )
    assert invoked == ["echo FOO"]


def test_execute_task_invariant_failure_blocks_with_gate_failed(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin: a failing invariant blocks the task with
    block_reason='gate_failed' and records a FailureRecord through
    the existing Phase-4 path."""
    from claude_plan_execute.gates import default_registry
    from claude_plan_execute.gates.invariants import (
        load_invariants,
        register_invariants_on,
    )
    p = _write_invariants_file(tmp_path,
        "- name: invariant-no-bad\n"
        "  command: 'echo BAD; exit 1'\n"
    )
    register_invariants_on(default_registry(), load_invariants(p))

    _patch_run_claude_simulating_success(monkeypatch)

    def fake_run(cmd, **kwargs):
        class P: pass
        x = P()
        x.returncode = 1
        x.stdout = "BAD\n"
        x.stderr = ""
        return x

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["invariant-grep"],
        },
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry["block_reason"] == "gate_failed"
    rec = entry["attempts"][0]
    assert rec["failure_class"] == "gate_failed"
    assert "BAD" in rec["verbatim"]
    assert "gate:invariant-no-bad" in rec["artifacts"]


def test_execute_task_invariant_grep_meta_name_does_NOT_warn_as_unknown(
    monkeypatch, tmp_path, _clean_default_registry, capsys,
):
    """Pin §D3: invariant-grep is special-cased before the unknown-
    gate branch. Even with no invariants loaded, no 'Unknown
    gate(s)' warning fires."""
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["invariant-grep"],
        },
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    assert "Unknown gate(s)" not in out


def test_execute_task_invariant_loaded_but_not_in_gates_does_NOT_run(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D3: an invariant loaded into the registry but NOT named
    in gates: (and gates: doesn't include 'invariant-grep') is not
    run. The meta-name is the user's opt-in; loading alone is not."""
    from claude_plan_execute.gates import default_registry
    from claude_plan_execute.gates.invariants import (
        load_invariants,
        register_invariants_on,
    )
    p = _write_invariants_file(tmp_path,
        "- name: invariant-foo\n"
        "  command: 'echo FOO'\n"
    )
    register_invariants_on(default_registry(), load_invariants(p))

    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[str] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(cmd[2])
        class P: pass
        x = P(); x.returncode = 0; x.stdout = ""; x.stderr = ""
        return x

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": [],
        },
        cli={}, skip_to="implement",
    )
    assert invoked == []
    assert state.get(1)["status"] == "done"


# ---------------------------------------------------------------------------
# R. `test-critique` built-in agent gate integration (task 15)
# ---------------------------------------------------------------------------

# These tests exercise the runner's integration with the
# import-time-registered `test-critique` agent gate. They use
# `_clean_default_registry` and explicitly register the gate on
# the fresh local registry via `_build_test_critique_gate()`.
# `run_claude` is stubbed because AgentGate calls it directly;
# no real `claude` subprocess is spawned.


def _patch_run_claude_for_agent_gate(monkeypatch, *, text="VERDICT: PASS\n"):
    """Stub run_claude in BOTH locations it's called from:
    - claude_plan_execute.runner.run_claude (Phase 3 implementer)
    - claude_plan_execute.gates.agent.run_claude (AgentGate)
    Returns a list of (prompt, kwargs) tuples for assertions."""
    captured: list[tuple[str, dict]] = []

    def stub(prompt, **kwargs):
        captured.append((prompt, kwargs))
        claude._last_run_state.timed_out = False
        claude._last_run_state.subtype = "success"
        return 0, text

    monkeypatch.setattr(
        "claude_plan_execute.runner.run_claude", stub,
    )
    monkeypatch.setattr(
        "claude_plan_execute.gates.agent.run_claude", stub,
    )
    return captured


def test_execute_task_test_critique_runs_via_registry_lookup(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D10: `test-critique` flows through the existing
    `else: g = registry.get(name)` branch. The runner's Phase-4
    dispatch needs zero new special-case for it."""
    from claude_plan_execute.gates.builtin import _build_test_critique_gate
    _clean_default_registry.register(
        "test-critique", _build_test_critique_gate(),
    )

    captured = _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["test-critique"],
        },
        cli={}, skip_to="implement",
    )
    # Two run_claude calls: implementer (Phase 3) + agent gate.
    assert len(captured) == 2
    # The gate's call uses sonnet/medium per the registered Gate:
    gate_call = captured[1]
    assert gate_call[1]["model"] == "sonnet"
    assert gate_call[1]["effort"] == "medium"
    # The task reaches done because the stub returns VERDICT: PASS:
    assert state.get(1)["status"] == "done"


def test_execute_task_test_critique_writes_output_file(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin task 11 §D8: the gate's response is captured to
    plans/task-{id}/gates/test-critique.txt."""
    from claude_plan_execute.gates.builtin import _build_test_critique_gate
    _clean_default_registry.register(
        "test-critique", _build_test_critique_gate(),
    )
    _patch_run_claude_for_agent_gate(
        monkeypatch, text="VERDICT: PASS\n\nReason: looks good"
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["test-critique"],
        },
        cli={}, skip_to="implement",
    )
    out_file = tmp_path / "task-1" / "gates" / "test-critique.txt"
    assert out_file.is_file()
    body = out_file.read_text()
    assert "VERDICT: PASS" in body
    assert "looks good" in body


def test_execute_task_test_critique_failure_blocks_with_gate_failed(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin task description: 'On FAIL, the failure record's
    next-attempt prompt sees the suggested cases.' A FAIL verdict
    blocks the task with block_reason='gate_failed' and writes a
    FailureRecord whose verbatim contains the suggested cases."""
    from claude_plan_execute.gates.builtin import _build_test_critique_gate
    _clean_default_registry.register(
        "test-critique", _build_test_critique_gate(),
    )
    fail_text = (
        "## Verdict: FAIL\n\n"
        "## Reason\nTests don't cover the empty-input case.\n\n"
        "## Suggested additional cases\n"
        "- empty list input\n"
        "- single-element input\n"
    )
    _patch_run_claude_for_agent_gate(monkeypatch, text=fail_text)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["test-critique"],
        },
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry["block_reason"] == "gate_failed"
    rec = entry["attempts"][0]
    assert rec["failure_class"] == "gate_failed"
    assert rec["phase"] == "gate"
    assert "empty list input" in rec["verbatim"]
    assert "gate:test-critique" in rec["artifacts"]


def test_execute_task_test_critique_forwards_skip_permissions_from_cli(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D9: when the CLI is invoked with --skip-permissions,
    the AgentGate's run_claude call sees skip_permissions=True.
    Without this, the gate's bash tool calls (git diff) auto-deny
    and the gate cannot fulfill its contract."""
    from claude_plan_execute.gates.builtin import _build_test_critique_gate
    _clean_default_registry.register(
        "test-critique", _build_test_critique_gate(),
    )
    captured = _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["test-critique"],
        },
        cli={"skip_permissions": True}, skip_to="implement",
    )
    # Two calls: implementer + gate. The gate's call (index 1) sees
    # skip_permissions=True forwarded from the cli dict:
    assert captured[1][1]["skip_permissions"] is True


def test_execute_task_test_critique_runs_after_tests_in_parallel_dispatch(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D5 / §R5: both `tests` and `test-critique` are
    parallel_safe → both end up in the parallel pool. We can't
    test concurrency deterministically in unit tests, but we
    can verify both are invoked and both write output files."""
    from claude_plan_execute.gates.builtin import _build_test_critique_gate
    _clean_default_registry.register(
        "test-critique", _build_test_critique_gate(),
    )
    _patch_run_claude_for_agent_gate(monkeypatch)

    shell_invoked: list[list[str]] = []
    def fake_shell_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            shell_invoked.append(list(cmd))
        class P: pass
        x = P(); x.returncode = 0; x.stdout = "ok"; x.stderr = ""
        return x
    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_shell_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["tests", "test-critique"],
            "test_command": "pytest -q",
        },
        cli={}, skip_to="implement",
    )
    # The shell gate was invoked once with the test command:
    assert shell_invoked == [["bash", "-c", "pytest -q"]]
    # Both output files are written:
    assert (tmp_path / "task-1" / "gates" / "tests.txt").is_file()
    assert (tmp_path / "task-1" / "gates" / "test-critique.txt").is_file()
    assert state.get(1)["status"] == "done"


def test_execute_task_test_critique_phase_extra_appended(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin: a phase's gates_extra='test-critique' is appended to
    project gates and resolves through the same registry-lookup
    branch."""
    from claude_plan_execute.gates.builtin import _build_test_critique_gate
    _clean_default_registry.register(
        "test-critique", _build_test_critique_gate(),
    )
    captured = _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, phase="risky"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": [],
        },
        cli={},
        phases_by_id={"risky": {"id": "risky",
                                "defaults": {"gates_extra": ["test-critique"]}}},
        skip_to="implement",
    )
    # Implementer + gate:
    assert len(captured) == 2
    # The gate ran (writes its output file):
    assert (tmp_path / "task-1" / "gates" / "test-critique.txt").is_file()


def test_execute_task_test_critique_does_NOT_warn_as_unknown(
    monkeypatch, tmp_path, _clean_default_registry, capsys,
):
    """Pin §D10: `test-critique` is found via registry.get() so
    no Unknown gate(s) warning fires. (The registration is on the
    fresh fixture registry; without registering it, this test
    would warn.)"""
    from claude_plan_execute.gates.builtin import _build_test_critique_gate
    _clean_default_registry.register(
        "test-critique", _build_test_critique_gate(),
    )
    _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["test-critique"],
        },
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    assert "Unknown gate(s)" not in out


def test_execute_task_test_critique_skipped_when_implementer_timed_out(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin task 11 §D12 carry-forward: implementer_timeout returns
    BEFORE the gate run, so test-critique does not fire. (No
    run_claude calls for the gate; only the implementer's timed-out
    call.)"""
    from claude_plan_execute.gates.builtin import _build_test_critique_gate
    _clean_default_registry.register(
        "test-critique", _build_test_critique_gate(),
    )
    # Implementer times out (uses the existing helper):
    _patch_run_claude_simulating_timeout(monkeypatch)
    # Independent stub for the gate's run_claude (would NOT be called):
    gate_called: list[bool] = []
    def gate_stub(prompt, **kwargs):
        gate_called.append(True)
        return 0, "VERDICT: PASS"
    monkeypatch.setattr(
        "claude_plan_execute.gates.agent.run_claude", gate_stub,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["test-critique"],
        },
        cli={}, skip_to="implement",
    )
    assert gate_called == []
    assert state.get(1)["status"] == "blocked"
    assert state.get(1)["block_reason"] == "implementer_timeout"


# ---------------------------------------------------------------------------
# S. `security-auditor` built-in agent gate integration (task 16)
# ---------------------------------------------------------------------------

# These tests exercise the runner's integration with the
# import-time-registered `security-auditor` agent gate. They use
# `_clean_default_registry` and explicitly register the gate on
# the fresh local registry via `_build_security_auditor_gate()`.
# `run_claude` is stubbed (via `_patch_run_claude_for_agent_gate`
# defined in Section R) — no real `claude` subprocess is spawned.


def test_execute_task_security_auditor_runs_via_registry_lookup(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D10: `security-auditor` flows through the existing
    `else: g = registry.get(name)` branch. The runner's Phase-4
    dispatch needs zero new special-case for it."""
    from claude_plan_execute.gates.builtin import (
        _build_security_auditor_gate,
    )
    _clean_default_registry.register(
        "security-auditor", _build_security_auditor_gate(),
    )

    captured = _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["security-auditor"],
        },
        cli={}, skip_to="implement",
    )
    # Two run_claude calls: implementer (Phase 3) + agent gate.
    assert len(captured) == 2
    # The gate's call uses opus/max per the registered Gate:
    gate_call = captured[1]
    assert gate_call[1]["model"] == "opus"
    assert gate_call[1]["effort"] == "max"
    # The task reaches done because the stub returns VERDICT: PASS:
    assert state.get(1)["status"] == "done"


def test_execute_task_security_auditor_writes_output_file(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin task 11 §D8: the gate's response is captured to
    plans/task-{id}/gates/security-auditor.txt."""
    from claude_plan_execute.gates.builtin import (
        _build_security_auditor_gate,
    )
    _clean_default_registry.register(
        "security-auditor", _build_security_auditor_gate(),
    )
    _patch_run_claude_for_agent_gate(
        monkeypatch, text="VERDICT: PASS\n\nReason: no diff-introduced findings",
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["security-auditor"],
        },
        cli={}, skip_to="implement",
    )
    out_file = tmp_path / "task-1" / "gates" / "security-auditor.txt"
    assert out_file.is_file()
    body = out_file.read_text()
    assert "VERDICT: PASS" in body
    assert "no diff-introduced findings" in body


def test_execute_task_security_auditor_failure_blocks_with_gate_failed(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin task description ('Block on any found'): a FAIL verdict
    blocks the task with block_reason='gate_failed' and writes a
    FailureRecord whose verbatim contains the agent's findings."""
    from claude_plan_execute.gates.builtin import (
        _build_security_auditor_gate,
    )
    _clean_default_registry.register(
        "security-auditor", _build_security_auditor_gate(),
    )
    fail_text = (
        "## Verdict: FAIL\n\n"
        "## Findings\n\n"
        "### 🔴 Critical\n\n"
        "**[A] - src/api/docs.py:42**\n"
        "Risk: missing ownership check\n"
        "Evidence:\n```\nreturn db.get_doc(id)\n```\n"
        "Fix: add `if doc.owner_id != user.id: raise Forbidden()`\n"
    )
    _patch_run_claude_for_agent_gate(monkeypatch, text=fail_text)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["security-auditor"],
        },
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry["block_reason"] == "gate_failed"
    rec = entry["attempts"][0]
    assert rec["failure_class"] == "gate_failed"
    assert rec["phase"] == "gate"
    assert "missing ownership check" in rec["verbatim"]
    assert "gate:security-auditor" in rec["artifacts"]


def test_execute_task_security_auditor_forwards_skip_permissions_from_cli(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D9 (inherited from task 15 §D9): when the CLI is
    invoked with --skip-permissions, the AgentGate's run_claude
    call sees skip_permissions=True. Without this, the gate's
    bash tool calls (git diff) auto-deny and the gate cannot
    fulfill its contract."""
    from claude_plan_execute.gates.builtin import (
        _build_security_auditor_gate,
    )
    _clean_default_registry.register(
        "security-auditor", _build_security_auditor_gate(),
    )
    captured = _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["security-auditor"],
        },
        cli={"skip_permissions": True}, skip_to="implement",
    )
    assert captured[1][1]["skip_permissions"] is True


def test_execute_task_security_auditor_runs_alongside_test_critique_in_parallel(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D5 / inherited task-15 §R5: both `test-critique` and
    `security-auditor` are parallel_safe → both end up in the
    parallel pool. We can't test concurrency deterministically in
    unit tests, but we can verify both are invoked and both write
    output files."""
    from claude_plan_execute.gates.builtin import (
        _build_security_auditor_gate,
        _build_test_critique_gate,
    )
    _clean_default_registry.register(
        "test-critique", _build_test_critique_gate(),
    )
    _clean_default_registry.register(
        "security-auditor", _build_security_auditor_gate(),
    )
    captured = _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["test-critique", "security-auditor"],
        },
        cli={}, skip_to="implement",
    )
    # Implementer (1) + two gates (2) = 3 run_claude calls:
    assert len(captured) == 3
    # Both output files written:
    assert (tmp_path / "task-1" / "gates" / "test-critique.txt").is_file()
    assert (tmp_path / "task-1" / "gates" / "security-auditor.txt").is_file()
    assert state.get(1)["status"] == "done"


def test_execute_task_security_auditor_phase_extra_appended(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin: a phase's gates_extra=['security-auditor'] is appended
    to project gates and resolves through the same registry-lookup
    branch. Mirrors the canonical opt-in shape from enhancement
    plan §2.1 / §2.4 (phases like `schema` opt in via gates_extra)."""
    from claude_plan_execute.gates.builtin import (
        _build_security_auditor_gate,
    )
    _clean_default_registry.register(
        "security-auditor", _build_security_auditor_gate(),
    )
    captured = _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, phase="schema"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": [],
        },
        cli={},
        phases_by_id={"schema": {"id": "schema",
                                 "defaults": {"gates_extra": ["security-auditor"]}}},
        skip_to="implement",
    )
    # Implementer + gate:
    assert len(captured) == 2
    assert (tmp_path / "task-1" / "gates" / "security-auditor.txt").is_file()


def test_execute_task_security_auditor_does_NOT_warn_as_unknown(
    monkeypatch, tmp_path, _clean_default_registry, capsys,
):
    """Pin §D10: `security-auditor` is found via registry.get() so
    no Unknown gate(s) warning fires. (The registration is on the
    fresh fixture registry; without registering it, this test
    would warn.)"""
    from claude_plan_execute.gates.builtin import (
        _build_security_auditor_gate,
    )
    _clean_default_registry.register(
        "security-auditor", _build_security_auditor_gate(),
    )
    _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["security-auditor"],
        },
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    assert "Unknown gate(s)" not in out


def test_execute_task_security_auditor_skipped_when_implementer_timed_out(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin task 11 §D12 carry-forward: implementer_timeout returns
    BEFORE the gate run, so security-auditor does not fire. (No
    run_claude calls for the gate; only the implementer's timed-out
    call.)"""
    from claude_plan_execute.gates.builtin import (
        _build_security_auditor_gate,
    )
    _clean_default_registry.register(
        "security-auditor", _build_security_auditor_gate(),
    )
    _patch_run_claude_simulating_timeout(monkeypatch)
    gate_called: list[bool] = []
    def gate_stub(prompt, **kwargs):
        gate_called.append(True)
        return 0, "VERDICT: PASS"
    monkeypatch.setattr(
        "claude_plan_execute.gates.agent.run_claude", gate_stub,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["security-auditor"],
        },
        cli={}, skip_to="implement",
    )
    assert gate_called == []
    assert state.get(1)["status"] == "blocked"
    assert state.get(1)["block_reason"] == "implementer_timeout"


# ---------------------------------------------------------------------------
# T. `untranslated-strings` built-in gate integration (task 17)
# ---------------------------------------------------------------------------

# These tests do NOT use the _clean_default_registry fixture. The runner's
# special-case for `untranslated-strings` (task 17 §D3) constructs the
# Gate via make_untranslated_strings_gate(...) WITHOUT consulting
# default_registry — the fixture's emptiness is irrelevant for the
# `untranslated-strings` path. (See §D13.)


def test_execute_task_untranslated_strings_gate_runs_resolved_command(
    monkeypatch, tmp_path,
):
    """Pin §D2 / §D3: when 'untranslated-strings' is in the gate
    list and untranslated_strings_command is set, the runner
    constructs the Gate with command=('bash', '-c', <cmd>) using
    the resolved value from the resolver chain."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["untranslated-strings"],
            "untranslated_strings_command": "scripts/i18n.sh",
        },
        cli={}, skip_to="implement",
    )
    assert invoked == [["bash", "-c", "scripts/i18n.sh"]]
    assert state.get(1)["status"] == "done"


def test_execute_task_untranslated_strings_gate_uses_per_task_command(
    monkeypatch, tmp_path,
):
    """Pin §D6: per-task untranslated_strings_command override is
    what the gate runs."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(
            id=1,
            untranslated_strings_command="python tools/check_i18n.py --strict",
        ),
        state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["untranslated-strings"],
            "untranslated_strings_command": "scripts/i18n.sh",
        },
        cli={}, skip_to="implement",
    )
    assert invoked == [
        ["bash", "-c", "python tools/check_i18n.py --strict"]
    ]


def test_execute_task_untranslated_strings_gate_failure_blocks_and_records(
    monkeypatch, tmp_path,
):
    """Pin §D10 / task 11 §D12: a failing untranslated-strings
    gate blocks the task with block_reason='gate_failed' and
    records a FailureRecord."""
    _patch_run_claude_simulating_success(monkeypatch)

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 1
        p.stdout = "Untranslated key found: src/locales/en.json:42 'submit_button'\n"
        p.stderr = "1 untranslated key\n"
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["untranslated-strings"],
            "untranslated_strings_command": "scripts/i18n.sh",
        },
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry["block_reason"] == "gate_failed"
    assert len(entry["attempts"]) == 1
    rec = entry["attempts"][0]
    assert rec["failure_class"] == "gate_failed"
    assert rec["phase"] == "gate"
    assert "submit_button" in rec["verbatim"]
    assert "gate:untranslated-strings" in rec["artifacts"]


def test_execute_task_untranslated_strings_gate_writes_output_file(
    monkeypatch, tmp_path,
):
    """Pin task 11 §D8: the gate's output is captured to
    plans/task-{id}/gates/untranslated-strings.txt."""
    _patch_run_claude_simulating_success(monkeypatch)

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = "I18N_OUTPUT_MARKER\n"
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["untranslated-strings"],
            "untranslated_strings_command": "scripts/i18n.sh",
        },
        cli={}, skip_to="implement",
    )
    out_file = tmp_path / "task-1" / "gates" / "untranslated-strings.txt"
    assert out_file.is_file()
    assert "I18N_OUTPUT_MARKER" in out_file.read_text()


def test_execute_task_untranslated_strings_unconfigured_warns_and_skips(
    monkeypatch, tmp_path, capsys,
):
    """Pin §D2: untranslated_strings_command is None (no layer set
    it) → runner prints the unconfigured warning and does NOT add
    the gate to the resolved list. The task still reaches done."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["untranslated-strings"],
            # NO untranslated_strings_command — None across the chain.
        },
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    assert (
        "Skipping gate 'untranslated-strings': no "
        "defaults.untranslated_strings_command configured."
    ) in out
    assert invoked == []                          # gate never built
    assert state.get(1)["status"] == "done"
    # No FailureRecord written for the warn-skip path:
    assert state.get(1)["attempts"] == []


def test_execute_task_untranslated_strings_explicit_empty_command_runs_noop(
    monkeypatch, tmp_path,
):
    """Pin §D2 / §D5: an explicit
    defaults.untranslated_strings_command: '' (empty string) is a
    valid 'no-op' choice — the factory builds the Gate and `bash
    -c ''` exits 0. Distinguishes from the None case (which warns
    and skips)."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["untranslated-strings"],
            "untranslated_strings_command": "",   # explicit empty string
        },
        cli={}, skip_to="implement",
    )
    assert invoked == [["bash", "-c", ""]]
    assert state.get(1)["status"] == "done"


def test_execute_task_untranslated_strings_unconfigured_warning_does_not_double_print(
    monkeypatch, tmp_path, capsys,
):
    """Pin §D2: the warning fires once even if
    `untranslated-strings` is listed multiple times (e.g.,
    gates: [untranslated-strings, untranslated-strings])."""
    _patch_run_claude_simulating_success(monkeypatch)

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["untranslated-strings", "untranslated-strings"],
        },
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    assert out.count("Skipping gate 'untranslated-strings':") == 1


def test_execute_task_untranslated_strings_name_does_NOT_warn_as_unknown(
    monkeypatch, tmp_path, capsys,
):
    """Pin §D3: the runner's special-case prevents
    'untranslated-strings' from being reported as an unknown-gate
    name. Pre-task-17 the empty-registry path warned for
    'untranslated-strings'; task 17 makes the unknown-gate warning
    silent for the literal name 'untranslated-strings'."""
    _patch_run_claude_simulating_success(monkeypatch)

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["untranslated-strings"],
            "untranslated_strings_command": "scripts/i18n.sh",
        },
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    assert "Unknown gate(s) skipped" not in out


def test_execute_task_untranslated_strings_gate_skipped_when_implementer_timed_out(
    monkeypatch, tmp_path,
):
    """Pin task 11 §D12 carry-forward: implementer_timeout returns
    BEFORE the gate run, so the untranslated-strings gate does not
    fire."""
    invoked = []

    def fake_run(cmd, **kwargs):
        invoked.append(cmd)
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_timeout(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["untranslated-strings"],
            "untranslated_strings_command": "scripts/i18n.sh",
        },
        cli={}, skip_to="implement",
    )
    assert invoked == []
    assert state.get(1)["status"] == "blocked"
    assert state.get(1)["block_reason"] == "implementer_timeout"


def test_execute_task_untranslated_strings_runs_alongside_tests_and_lint_in_parallel(
    monkeypatch, tmp_path,
):
    """Pin §D10: tests, lint, and untranslated-strings are all
    parallel_safe → they end up in the parallel pool. We can't
    test concurrency deterministically in unit tests, but we can
    verify all three subprocess.run invocations happen and all
    three gate output files are written."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = "ok\n"
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["tests", "lint", "untranslated-strings"],
            "test_command": "pytest -q",
            "lint_command": "ruff check",
            "untranslated_strings_command": "scripts/i18n.sh",
        },
        cli={}, skip_to="implement",
    )
    cmds = sorted(c[2] for c in invoked)
    assert cmds == ["pytest -q", "ruff check", "scripts/i18n.sh"]
    assert (tmp_path / "task-1" / "gates" / "tests.txt").is_file()
    assert (tmp_path / "task-1" / "gates" / "lint.txt").is_file()
    assert (tmp_path / "task-1" / "gates" / "untranslated-strings.txt").is_file()
    assert state.get(1)["status"] == "done"


def test_execute_task_untranslated_strings_phase_extra_appended(
    monkeypatch, tmp_path,
):
    """Pin task 11 §D12 + §D3: a phase's
    gates_extra=['untranslated-strings'] is appended to project
    gates and resolves through the same special-case dispatch."""
    _patch_run_claude_simulating_success(monkeypatch)
    invoked: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] != ["git", "diff"]:  # T7: ignore precomputed-diff capture
            invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, phase="ui"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": [],
            "untranslated_strings_command": "scripts/i18n.sh",
        },
        cli={},
        phases_by_id={
            "ui": {
                "id": "ui",
                "defaults": {"gates_extra": ["untranslated-strings"]},
            },
        },
        skip_to="implement",
    )
    assert invoked == [["bash", "-c", "scripts/i18n.sh"]]
    assert state.get(1)["status"] == "done"


# ---------------------------------------------------------------------------
# U. `design-fidelity` built-in agent gate integration (task 20)
# ---------------------------------------------------------------------------

# These tests exercise the runner's Phase-4 dispatch for the
# `design-fidelity` factory and the integration with phase-
# inherited task_class via task_for_gates augmentation. `run_claude`
# is stubbed via `_patch_run_claude_for_agent_gate` (defined in
# Section R). NO live browser, NO live dev server, NO real Playwright
# MCP traffic.


def test_execute_task_design_fidelity_runs_for_ui_task(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D7 / §D8: a task in a `ui` task_class triggers the gate;
    the factory dispatch runs the AgentGate."""
    captured = _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, task_class="ui"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["design-fidelity"],
        },
        cli={}, skip_to="implement",
    )
    # Two run_claude calls: implementer + design-fidelity gate.
    assert len(captured) == 2
    gate_call = captured[1]
    assert gate_call[1]["model"] == "sonnet"
    assert gate_call[1]["effort"] == "medium"
    assert state.get(1)["status"] == "done"


def test_execute_task_design_fidelity_skipped_for_non_ui_task(
    monkeypatch, tmp_path, _clean_default_registry, capsys,
):
    """Pin §D8: a non-UI task skips the gate via the task_class:ui
    filter. The runner's stdout shows the 'Skipping gate' message."""
    captured = _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, task_class="schema"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["design-fidelity"],
        },
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    assert "Skipping gate 'design-fidelity'" in out
    # Implementer ran but the gate did not.
    assert len(captured) == 1
    # The skipped gate writes a passive PASS output file.
    assert (tmp_path / "task-1" / "gates" / "design-fidelity.txt").is_file()
    assert state.get(1)["status"] == "done"


def test_execute_task_design_fidelity_skipped_for_no_class_task(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D9: a task with no task_class (and no phase-inherited
    class) skips the gate. The augmentation logic still injects the
    None resolved class; _when_matches handles None correctly."""
    captured = _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,                    # NO task_class
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["design-fidelity"],
        },
        cli={}, skip_to="implement",
    )
    # Implementer only.
    assert len(captured) == 1
    assert state.get(1)["status"] == "done"


def test_execute_task_design_fidelity_uses_phase_inherited_task_class(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D9: a task without a per-task task_class but a phase
    `defaults: { task_class: ui }` triggers the gate via phase
    inheritance. Without the runner's task_for_gates augmentation,
    this would skip incorrectly."""
    captured = _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, phase="ui"), state,        # NO per-task task_class
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["design-fidelity"],
        },
        cli={},
        phases_by_id={
            "ui": {"id": "ui", "defaults": {"task_class": "ui"}},
        },
        task_class_models={"ui": {"model": "sonnet", "effort": "medium"}},
        skip_to="implement",
    )
    # Implementer + gate.
    assert len(captured) == 2
    assert state.get(1)["status"] == "done"


def test_execute_task_design_fidelity_uses_resolved_dev_server_url(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D6: the resolved dev_server_url is substituted into
    the gate's prompt body. With project_defaults setting
    dev_server_url, the gate's run_claude call sees a prompt
    containing the configured URL."""
    captured = _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, task_class="ui"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["design-fidelity"],
            "dev_server_url": "http://example.test:8080",
        },
        cli={}, skip_to="implement",
    )
    gate_prompt = captured[1][0]
    assert "http://example.test:8080" in gate_prompt


def test_execute_task_design_fidelity_uses_hardcoded_default_url_when_unset(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D6: when no project override is set, the hardcoded
    default URL (http://localhost:5173) flows into the gate's prompt."""
    captured = _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, task_class="ui"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["design-fidelity"],
        },
        cli={}, skip_to="implement",
    )
    gate_prompt = captured[1][0]
    assert "http://localhost:5173" in gate_prompt


def test_execute_task_design_fidelity_failure_blocks_with_gate_failed(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin task description ('On FAIL, the gate writes its findings
    to plans/task-{id}/gates/design-fidelity.txt; the next implement
    attempt sees the verbatim findings'): a FAIL verdict produces a
    FailureRecord whose verbatim contains the agent's findings."""
    fail_text = (
        "## Verdict: FAIL\n\n"
        "## Findings\n\n"
        "**[A] - / on 320×800**\n"
        "Risk: horizontal scroll on mobile\n"
        "Fix: tighten layout\n"
    )
    _patch_run_claude_for_agent_gate(monkeypatch, text=fail_text)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, task_class="ui"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["design-fidelity"],
        },
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry["block_reason"] == "gate_failed"
    rec = entry["attempts"][0]
    assert rec["failure_class"] == "gate_failed"
    assert rec["phase"] == "gate"
    assert "horizontal scroll on mobile" in rec["verbatim"]
    assert "gate:design-fidelity" in rec["artifacts"]


def test_execute_task_design_fidelity_does_NOT_warn_as_unknown(
    monkeypatch, tmp_path, _clean_default_registry, capsys,
):
    """Pin §D7: design-fidelity is found via the elif factory branch
    (not through registry.get) so no Unknown gate(s) warning fires."""
    _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, task_class="ui"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["design-fidelity"],
        },
        cli={}, skip_to="implement",
    )
    out = capsys.readouterr().out
    assert "Unknown gate(s)" not in out


def test_execute_task_design_fidelity_persona_section_flows_into_prompt(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D2: the persona section loaded by the runner flows
    through the factory into the gate's prompt body."""
    persona_file = tmp_path / "persona.md"
    persona_file.write_text("# Project conventions\n\n8-pt grid.\n")

    captured = _patch_run_claude_for_agent_gate(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, task_class="ui"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["design-fidelity"],
            "persona_file": str(persona_file),
        },
        cli={}, skip_to="implement",
    )
    gate_prompt = captured[1][0]
    assert "8-pt grid" in gate_prompt


def test_execute_task_design_fidelity_skipped_when_implementer_timed_out(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin task 11 §D12 carry-forward: implementer_timeout returns
    BEFORE the gate run, so design-fidelity does not fire."""
    _patch_run_claude_simulating_timeout(monkeypatch)
    gate_called: list[bool] = []
    def gate_stub(prompt, **kwargs):
        gate_called.append(True)
        return 0, "VERDICT: PASS"
    monkeypatch.setattr(
        "claude_plan_execute.gates.agent.run_claude", gate_stub,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, task_class="ui"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["design-fidelity"],
        },
        cli={}, skip_to="implement",
    )
    assert gate_called == []
    assert state.get(1)["status"] == "blocked"
    assert state.get(1)["block_reason"] == "implementer_timeout"


# ---------------------------------------------------------------------------
# V. progress.md resume routing (task 6 v0.3)
# ---------------------------------------------------------------------------


def _seed_progress(tmp_path, tid, body):
    task_dir = tmp_path / f"task-{tid}"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "progress.md").write_text(body, encoding="utf-8")
    # plan.md must also exist so the resume route can reference it
    # without the runner short-circuiting on missing plan.
    (task_dir / "plan.md").write_text("placeholder plan")


# V1. Fresh: no progress.md → normal flow


def test_execute_task_no_progress_md_uses_implement_builder(monkeypatch, tmp_path):
    """Pin §D4: with no progress.md present, the runner uses
    build_implement_prompt (not build_resume_prompt) at the
    Phase-3 call site."""
    captured: dict = {}
    real_build_impl = runner.build_implement_prompt

    def spy_impl(*args, **kwargs):
        captured["impl"] = True
        return real_build_impl(*args, **kwargs)

    def spy_resume(*args, **kwargs):
        captured["resume"] = True
        return ""  # never called; assert this

    monkeypatch.setattr("claude_plan_execute.runner.build_implement_prompt", spy_impl)
    monkeypatch.setattr("claude_plan_execute.runner.build_resume_prompt", spy_resume)
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert captured.get("impl") is True
    assert captured.get("resume") is None


def test_execute_task_no_progress_md_records_progress_path_anyway(
    monkeypatch, tmp_path,
):
    """Pin §D6: progress_path is tracked on the entry even when the
    file doesn't exist yet (so the dashboard knows where to look)."""
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["progress_path"].endswith("task-1/progress.md")


# V2. Mid-list resume: [~] / [ ] present


def test_execute_task_mid_list_progress_uses_resume_builder(monkeypatch, tmp_path):
    """Pin §D4.3: progress.md with [~] or [ ] items routes to
    build_resume_prompt and skips build_implement_prompt."""
    captured: dict = {}
    real_build_resume = runner.build_resume_prompt

    def spy_resume(*args, **kwargs):
        captured["resume"] = True
        captured["args"] = args
        captured["kwargs"] = kwargs
        return real_build_resume(*args, **kwargs)

    def spy_impl(*args, **kwargs):
        captured["impl"] = True
        return ""

    monkeypatch.setattr("claude_plan_execute.runner.build_resume_prompt", spy_resume)
    monkeypatch.setattr("claude_plan_execute.runner.build_implement_prompt", spy_impl)
    _patch_run_claude_simulating_success(monkeypatch)

    _seed_progress(tmp_path, 1, "- [x] step a\n- [~] step b\n- [ ] step c")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={},  # No skip_to — progress.md should force resume route.
    )
    assert captured.get("resume") is True
    assert captured.get("impl") is None


def test_execute_task_mid_list_skips_plan_and_review(monkeypatch, tmp_path):
    """Pin §D4.1 / §D4.2: progress.md mid-list forces skip of Phase 1
    and Phase 2 even when --skip-to is unset (i.e., the default flow
    that would otherwise plan + review)."""
    called = []

    def stub(prompt, **kwargs):
        called.append(kwargs)
        claude._last_run_state.subtype = "success"
        claude._last_run_state.timed_out = False
        return 0, "ok"

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub)
    _seed_progress(tmp_path, 1, "- [~] step b\n- [ ] step c")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={},  # No skip_to: would normally plan + review + implement.
    )
    # Expect EXACTLY one run_claude call (the resume Implement). The
    # plan and review calls would push this above 1.
    assert len(called) == 1


def test_execute_task_mid_list_passes_progress_text_to_builder(
    monkeypatch, tmp_path,
):
    """Pin §D2: the runner passes the on-disk progress.md content
    verbatim into the resume builder."""
    captured: dict = {}

    def spy(*args, **kwargs):
        captured["args"] = args
        return "PROMPT"

    monkeypatch.setattr("claude_plan_execute.runner.build_resume_prompt", spy)
    _patch_run_claude_simulating_success(monkeypatch)

    body = "- [x] done step\n- [~] mid step\n- [ ] todo"
    _seed_progress(tmp_path, 1, body)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={},
    )
    # build_resume_prompt(task, plan_path, progress_path,
    #                     progress_md_text, dirty_porcelain, commit_msg, ...)
    args = captured["args"]
    assert args[3] == body  # progress_md_text


# V3. Dirty-tree porcelain capture


def test_execute_task_mid_list_runs_git_status_porcelain(monkeypatch, tmp_path):
    """Pin §D5: the runner shells out `git status --porcelain` on
    the resume route. Stub subprocess.run; assert the cmd."""
    invocations = []

    def fake_run(cmd, **kwargs):
        invocations.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = " M src/foo.py\n"
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)
    _seed_progress(tmp_path, 1, "- [~] step")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={},
    )
    assert ["git", "status", "--porcelain"] in invocations


def test_execute_task_mid_list_surfaces_porcelain_into_resume_prompt(
    monkeypatch, tmp_path,
):
    """Pin §D2: the porcelain output reaches the resume prompt
    body via the dirty_tree_porcelain kwarg."""
    captured: dict = {}

    def spy(*args, **kwargs):
        captured["args"] = args
        return "PROMPT"

    monkeypatch.setattr("claude_plan_execute.runner.build_resume_prompt", spy)

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = "?? new_file.py\n"
        p.stderr = ""
        return p
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)
    _seed_progress(tmp_path, 1, "- [~] step")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={},
    )
    # build_resume_prompt(task, plan_path, progress_path,
    #                     progress_md_text, dirty_porcelain, ...)
    assert "?? new_file.py" in captured["args"][4]


def test_execute_task_mid_list_git_failure_yields_empty_porcelain(
    monkeypatch, tmp_path,
):
    """Pin §D5: defensive — git binary missing / non-repo / timeout
    → empty porcelain string, builder's clean-tree branch fires."""
    captured: dict = {}

    def spy(*args, **kwargs):
        captured["args"] = args
        return "PROMPT"

    monkeypatch.setattr("claude_plan_execute.runner.build_resume_prompt", spy)

    def boom(*args, **kwargs):
        raise FileNotFoundError("git not installed")
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", boom,
    )
    _patch_run_claude_simulating_success(monkeypatch)
    _seed_progress(tmp_path, 1, "- [~] step")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={},
    )
    assert captured["args"][4] == ""


# V4. All-[x] route: skip Phase 3, run Phase 4 only


def test_execute_task_all_done_progress_skips_implement(monkeypatch, tmp_path):
    """Pin §D4.3: all-[x] progress.md → Phase 3 is skipped entirely;
    no run_claude invocation; runner falls through to Phase 4."""
    called = []

    def stub(prompt, **kwargs):
        called.append(kwargs)
        claude._last_run_state.subtype = "success"
        return 0, "ok"

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub)
    _seed_progress(tmp_path, 1, "- [x] one\n- [x] two\n- [x] three")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={},
    )
    # Zero run_claude calls — Plan + Review + Implement all skipped.
    assert called == []


def test_execute_task_all_done_progress_still_runs_gates(
    monkeypatch, tmp_path,
):
    """Pin §D7: Phase 4 fires even when Implement was skipped."""
    invoked = []

    def fake_run(cmd, **kwargs):
        invoked.append(list(cmd))
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p
    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    # The capture-porcelain helper also routes through subprocess.run
    # — but only on the mid-list route, not all-[x]. Stub it here too
    # to be safe.
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )

    _seed_progress(tmp_path, 1, "- [x] all done")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["tests"],
            "test_command": "pytest -q",
        },
        cli={},
    )
    # The tests gate ran:
    assert ["bash", "-c", "pytest -q"] in invoked
    assert state.get(1)["status"] == "done"


def test_execute_task_all_done_progress_does_NOT_call_resume_builder(
    monkeypatch, tmp_path,
):
    """Pin §D4.3: all-[x] route does NOT call build_resume_prompt."""
    captured: dict = {}

    def spy(*args, **kwargs):
        captured["resume"] = True
        return ""

    monkeypatch.setattr("claude_plan_execute.runner.build_resume_prompt", spy)
    _seed_progress(tmp_path, 1, "- [x] one\n- [x] two")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={},
    )
    assert captured.get("resume") is None


# V5. Edge cases


def test_execute_task_empty_progress_md_treated_as_fresh(
    monkeypatch, tmp_path,
):
    """Pin §D4: empty file (no checklist items) → fresh route."""
    captured: dict = {}
    real_build_impl = runner.build_implement_prompt

    def spy_impl(*args, **kwargs):
        captured["impl"] = True
        return real_build_impl(*args, **kwargs)

    monkeypatch.setattr("claude_plan_execute.runner.build_implement_prompt", spy_impl)
    _patch_run_claude_simulating_success(monkeypatch)

    _seed_progress(tmp_path, 1, "")  # empty file

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert captured.get("impl") is True


def test_execute_task_progress_md_with_no_items_treated_as_fresh(
    monkeypatch, tmp_path,
):
    """Pin: a progress.md that has paragraphs but no checklist items
    → fresh route (no_done == 0, no_active == 0)."""
    captured: dict = {}
    real_build_impl = runner.build_implement_prompt

    def spy_impl(*args, **kwargs):
        captured["impl"] = True
        return real_build_impl(*args, **kwargs)

    monkeypatch.setattr("claude_plan_execute.runner.build_implement_prompt", spy_impl)
    _patch_run_claude_simulating_success(monkeypatch)

    _seed_progress(
        tmp_path, 1,
        "# Progress\n\nSome notes without a checklist.\n",
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert captured.get("impl") is True


def test_execute_task_status_stays_implementing_during_resume(
    monkeypatch, tmp_path,
):
    """Pin §D8: resume run sets status to 'implementing' (no
    'resuming' status added). On success, transitions to 'done'."""
    _patch_run_claude_simulating_success(monkeypatch)
    _seed_progress(tmp_path, 1, "- [~] step")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={},
    )
    # Final state after a successful resume run: done.
    # No 'resuming' enum value should ever appear in STATUSES.
    from claude_plan_execute.state import STATUSES
    assert "resuming" not in STATUSES
    assert state.get(1)["status"] == "done"


# ---------------------------------------------------------------------------
# W. Precomputed diff write before Phase 4 (task 7 v0.3)
# ---------------------------------------------------------------------------

def test_execute_task_writes_diff_patch_post_implement(monkeypatch, tmp_path):
    """Pin T7 §D1: after a successful implement, the runner writes
    `task_dir / 'diff.patch'` with the captured `git diff HEAD~1..HEAD`
    output. The file exists and contains the stubbed git stdout."""
    diff_body = "diff --git a/foo b/foo\n+added line\n"

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 0
        if cmd[:2] == ["git", "diff"]:
            p.stdout = diff_body
        else:
            p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    diff_path = tmp_path / "task-1" / "diff.patch"
    assert diff_path.is_file()
    assert diff_body in diff_path.read_text(encoding="utf-8")


def test_execute_task_diff_patch_empty_when_git_returns_empty(
    monkeypatch, tmp_path,
):
    """Pin T7 §D10: empty git stdout → empty diff.patch (zero
    bytes). The file always exists; templates handle "diff is
    empty, read source"."""
    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    diff_path = tmp_path / "task-1" / "diff.patch"
    assert diff_path.is_file()
    assert diff_path.read_text(encoding="utf-8") == ""


def test_execute_task_diff_patch_empty_when_subprocess_fails(
    monkeypatch, tmp_path,
):
    """Pin T7 §D6: defensive — git binary missing / non-repo /
    timeout → helper returns ''; the runner writes an empty
    diff.patch. Mirrors `_capture_git_porcelain` behavior."""
    def boom(cmd, **kwargs):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", boom,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    diff_path = tmp_path / "task-1" / "diff.patch"
    assert diff_path.is_file()
    assert diff_path.read_text(encoding="utf-8") == ""


def test_execute_task_diff_patch_tail_capped_at_256kb(
    monkeypatch, tmp_path,
):
    """Pin T7 §D5: a synthesized git stdout > 256 KB is tail-
    capped to fit `_DIFF_CAP_BYTES + marker overhead`. Verifies
    the runner→helper→`_tail_cap(cap=)` chain."""
    from claude_plan_execute.runner import _DIFF_CAP_BYTES
    big = "Y" * (_DIFF_CAP_BYTES + 50000)

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = big if cmd[:2] == ["git", "diff"] else ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    diff_path = tmp_path / "task-1" / "diff.patch"
    body = diff_path.read_text(encoding="utf-8")
    assert len(body.encode("utf-8")) <= _DIFF_CAP_BYTES + 100
    assert "[truncated:" in body


def test_execute_task_all_done_resume_writes_diff_patch(
    monkeypatch, tmp_path,
):
    """Pin T7 §D1: `resume_mode == "all_done"` skips Phase 3 but
    still reaches the diff-write site. The diff.patch is created
    (capturing the latest commit, however incomplete that may be
    for multi-WIP-commit implementations — see §R3)."""
    diff_body = "diff --git a/x b/x\n+last commit\n"

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 0
        if cmd[:2] == ["git", "diff"]:
            p.stdout = diff_body
        else:
            p.stdout = ""  # porcelain (not called on all_done route, but defensive)
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _seed_progress(tmp_path, 1, "- [x] one\n- [x] two")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={},
    )
    diff_path = tmp_path / "task-1" / "diff.patch"
    assert diff_path.is_file()
    assert diff_body in diff_path.read_text(encoding="utf-8")


def test_execute_task_implementer_timeout_does_NOT_write_diff_patch(
    monkeypatch, tmp_path,
):
    """Pin T7 §D12: the implementer_timeout early-return short-
    circuits before the diff-write site. No diff.patch is created.
    The next attempt's runner pass writes one if it reaches Phase 4."""
    def fake_run(cmd, **kwargs):
        # Only invoked if execute_task reaches a subprocess site;
        # the timeout early-return prevents that.
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )

    def stub_timeout(*args, **kwargs):
        claude._last_run_state.timed_out = True
        return 1, ""

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub_timeout)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    diff_path = tmp_path / "task-1" / "diff.patch"
    assert not diff_path.exists()
    assert state.get(1)["status"] == "blocked"


def test_execute_task_implementer_max_turns_does_NOT_write_diff_patch(
    monkeypatch, tmp_path,
):
    """Pin T7 §D12: the implementer_max_turns early-return short-
    circuits before the diff-write site. No diff.patch is created."""
    def fake_run(cmd, **kwargs):
        class P: pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )

    def stub_max_turns(*args, **kwargs):
        claude._last_run_state.subtype = "error_max_turns"
        return 0, "ok"

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub_max_turns)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    diff_path = tmp_path / "task-1" / "diff.patch"
    assert not diff_path.exists()
    assert state.get(1)["status"] == "blocked"


# ---------------------------------------------------------------------------
# X. Token usage — claude.run_claude side-channel and JSONL append
#    (task 2 v0.3)
# ---------------------------------------------------------------------------


def test_run_claude_sets_input_tokens_side_channel_flag(monkeypatch):
    """Pin §D1: the four token flags are set from the parsed CLI
    `usage` event."""
    stdout = json.dumps({
        "type": "result", "subtype": "success", "result": "ok",
        "usage": {
            "input_tokens": 12345,
            "output_tokens": 678,
            "cache_read_input_tokens": 8000,
            "cache_creation_input_tokens": 200,
        },
    })
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout)
    run_claude("p", model="opus", effort="max", max_turns=1,
               timeout_minutes=1, skip_permissions=False, verbose=False)
    assert claude._last_run_state.input_tokens == 12345
    assert claude._last_run_state.output_tokens == 678
    assert claude._last_run_state.cache_read_tokens == 8000
    assert claude._last_run_state.cache_creation_tokens == 200


def test_run_claude_sets_duration_seconds_side_channel_flag(monkeypatch):
    """Pin §D1: duration is set unconditionally from the existing
    `elapsed` measurement."""
    _patch_run_claude_subprocess(monkeypatch)
    run_claude("p", model="opus", effort="max", max_turns=1,
               timeout_minutes=1, skip_permissions=False, verbose=False)
    assert claude._last_run_state.duration_seconds is not None
    assert claude._last_run_state.duration_seconds >= 0


def test_run_claude_resets_usage_flags_at_call_entry(monkeypatch):
    """Pin §D1: stale flags from a prior run are cleared. The new
    call's stdout has no `usage` event → flags reset to None."""
    claude._last_run_state.input_tokens = 999_999  # leftover
    stdout = json.dumps({"type": "result", "subtype": "success", "result": "ok"})
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout)
    run_claude("p", model="opus", effort="max", max_turns=1,
               timeout_minutes=1, skip_permissions=False, verbose=False)
    assert claude._last_run_state.input_tokens is None


def test_run_claude_usage_flags_None_when_usage_event_absent(monkeypatch):
    """Pin §D1: older CLI versions (no `usage` field in result event)
    leave the four token flags as None — `summarize_usage` /
    `_append_usage_row` skip the row."""
    stdout = json.dumps({"type": "result", "subtype": "success", "result": "ok"})
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout)
    run_claude("p", model="opus", effort="max", max_turns=1,
               timeout_minutes=1, skip_permissions=False, verbose=False)
    assert claude._last_run_state.input_tokens is None
    assert claude._last_run_state.output_tokens is None
    assert claude._last_run_state.cache_read_tokens is None
    assert claude._last_run_state.cache_creation_tokens is None


def test_run_claude_extracts_usage_from_list_result_event(monkeypatch):
    """Pin §D1: CLI v2.x list-of-events output. Usage on the
    terminal `type: result` event flows into the side-channel."""
    stdout = json.dumps([
        {"type": "system", "subtype": "init"},
        {"type": "result", "subtype": "success", "result": "ok",
         "usage": {"input_tokens": 100, "output_tokens": 50}},
    ])
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout)
    run_claude("p", model="opus", effort="max", max_turns=1,
               timeout_minutes=1, skip_permissions=False, verbose=False)
    assert claude._last_run_state.input_tokens == 100
    assert claude._last_run_state.output_tokens == 50


def test_run_claude_extracts_usage_from_dict_legacy_shape(monkeypatch):
    """Pin §D1: legacy single-dict CLI output. Usage at the top-
    level dict still flows through."""
    stdout = json.dumps({
        "type": "result", "subtype": "success", "result": "ok",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout)
    run_claude("p", model="opus", effort="max", max_turns=1,
               timeout_minutes=1, skip_permissions=False, verbose=False)
    assert claude._last_run_state.input_tokens == 1


def test_run_claude_appends_jsonl_row_when_path_and_phase_provided(
    monkeypatch, tmp_path,
):
    """Pin §D3: the row predicate fires when usage_jsonl_path,
    phase, and parsed usage are all present."""
    stdout = json.dumps({
        "type": "result", "subtype": "success", "result": "ok",
        "usage": {"input_tokens": 100, "output_tokens": 50,
                  "cache_read_input_tokens": 30,
                  "cache_creation_input_tokens": 5},
    })
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout)
    p = tmp_path / "usage.jsonl"
    run_claude("p", model="opus", effort="max", max_turns=1,
               timeout_minutes=1, skip_permissions=False, verbose=False,
               phase="implement", usage_jsonl_path=p)
    assert p.is_file()
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["input"] == 100
    assert rows[0]["output"] == 50
    assert rows[0]["cache_read"] == 30
    assert rows[0]["cache_create"] == 5


def test_run_claude_jsonl_row_schema_matches_spec(monkeypatch, tmp_path):
    """Pin §D10: row keys are exactly the eight named in tasks.yaml,
    in this order."""
    stdout = json.dumps({
        "type": "result", "subtype": "success", "result": "ok",
        "usage": {"input_tokens": 1, "output_tokens": 1,
                  "cache_read_input_tokens": 1,
                  "cache_creation_input_tokens": 1},
    })
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout)
    p = tmp_path / "u.jsonl"
    run_claude("p", model="sonnet", effort="medium", max_turns=1,
               timeout_minutes=1, skip_permissions=False, verbose=False,
               phase="plan", usage_jsonl_path=p)
    row = json.loads(p.read_text().strip())
    assert list(row.keys()) == [
        "timestamp", "phase", "model",
        "input", "output", "cache_read", "cache_create",
        "duration_seconds",
    ]
    assert row["phase"] == "plan"
    assert row["model"] == "sonnet"


def test_run_claude_jsonl_row_timestamp_parseable(monkeypatch, tmp_path):
    """Pin §D10: timestamps are UTC ISO-8601."""
    from datetime import datetime
    stdout = json.dumps({
        "type": "result", "subtype": "success", "result": "ok",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    })
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout)
    p = tmp_path / "u.jsonl"
    run_claude("p", model="opus", effort="max", max_turns=1,
               timeout_minutes=1, skip_permissions=False, verbose=False,
               phase="plan", usage_jsonl_path=p)
    row = json.loads(p.read_text().strip())
    parsed = datetime.fromisoformat(row["timestamp"])
    assert parsed.utcoffset().total_seconds() == 0


def test_run_claude_appends_multiple_rows_to_same_file(monkeypatch, tmp_path):
    """Pin: the file is open-append, NOT open-write. Multiple
    run_claude calls accumulate rows."""
    stdout = json.dumps({
        "type": "result", "subtype": "success", "result": "ok",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout)
    p = tmp_path / "u.jsonl"
    for _ in range(3):
        run_claude("p", model="opus", effort="max", max_turns=1,
                   timeout_minutes=1, skip_permissions=False, verbose=False,
                   phase="plan", usage_jsonl_path=p)
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert len(rows) == 3


def test_run_claude_does_NOT_append_when_path_is_None(monkeypatch, tmp_path):
    """Pin §D3: usage_jsonl_path=None disables the append."""
    stdout = json.dumps({
        "type": "result", "subtype": "success", "result": "ok",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout)
    p = tmp_path / "u.jsonl"
    run_claude("p", model="opus", effort="max", max_turns=1,
               timeout_minutes=1, skip_permissions=False, verbose=False,
               phase="plan", usage_jsonl_path=None)
    assert not p.exists()


def test_run_claude_does_NOT_append_when_phase_is_None(monkeypatch, tmp_path):
    """Pin §D3: phase=None disables the append even when path is
    provided."""
    stdout = json.dumps({
        "type": "result", "subtype": "success", "result": "ok",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout)
    p = tmp_path / "u.jsonl"
    run_claude("p", model="opus", effort="max", max_turns=1,
               timeout_minutes=1, skip_permissions=False, verbose=False,
               phase=None, usage_jsonl_path=p)
    assert not p.exists()


def test_run_claude_does_NOT_append_when_usage_event_absent(
    monkeypatch, tmp_path,
):
    """Pin §D3 (absence-robust headline): no `usage` field in stdout
    → no row appended. This is THE test the task description names
    in 'robust to absence — older CLI versions, dry-run, etc.'"""
    stdout = json.dumps({"type": "result", "subtype": "success", "result": "ok"})
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout)
    p = tmp_path / "u.jsonl"
    run_claude("p", model="opus", effort="max", max_turns=1,
               timeout_minutes=1, skip_permissions=False, verbose=False,
               phase="plan", usage_jsonl_path=p)
    assert not p.exists()


def test_run_claude_appends_row_on_nonzero_exit_with_usage(monkeypatch, tmp_path):
    """Pin §D3: an attempt with non-zero exit but a usage event in
    its result still records the row. The operator-debug question
    'how much did the failed attempt cost?' is answered."""
    stdout = json.dumps({
        "type": "result", "subtype": "error_max_turns",
        "result": "", "is_error": True,
        "usage": {"input_tokens": 50, "output_tokens": 0},
    })
    _patch_run_claude_subprocess(monkeypatch, stdout=stdout, returncode=1)
    p = tmp_path / "u.jsonl"
    # Disable rate-limit retry to keep the test single-attempt.
    run_claude("p", model="opus", effort="max", max_turns=1,
               timeout_minutes=1, skip_permissions=False, verbose=False,
               phase="implement", usage_jsonl_path=p,
               max_rate_limit_retries=0)
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["input"] == 50


def test_run_claude_writes_no_rows_when_no_attempt_has_usage(
    monkeypatch, tmp_path,
):
    """Pin §D3 (absence-robust under the per-call contract): when
    every retry attempt's `_run_claude_once` returns usage=None,
    the row write at the return site has nothing to write — ZERO
    rows in the JSONL file."""
    helper_calls, _sleeps = _patch_run_claude_once(monkeypatch, [
        # Attempts 1-3: rate-limited, no usage.
        (1, "", "rate_limit_error"),
        (1, "", "rate_limit_error"),
        (1, "", "rate_limit_error"),
        # Attempt 4: success with no usage (helper stub returns
        # usage=None for ALL attempts via its 5-tuple defaults).
        (0, "ok", ""),
    ])
    p = tmp_path / "u.jsonl"
    run_claude("p", model="opus", effort="max", max_turns=1,
               timeout_minutes=1, skip_permissions=False, verbose=False,
               phase="implement", usage_jsonl_path=p)
    # The fake _run_claude_once returns usage=None for every
    # attempt → no row appended. This is the correct outcome
    # under the per-call contract.
    assert not p.exists() or p.read_text() == ""


def test_run_claude_one_row_when_final_retry_succeeds_with_usage(
    monkeypatch, tmp_path,
):
    """Pin §D3 (per-call contract; final-attempt-wins): when a
    rate-limited retry succeeds on the final attempt with usage
    data, exactly one row is written using the final attempt's
    usage."""
    seq_iter = iter([
        # Three rate-limited attempts (no usage)
        (1, "", "rate_limit_error", None, 0.0),
        (1, "", "rate_limit_error", None, 0.0),
        (1, "", "rate_limit_error", None, 0.0),
        # Final success with usage
        (0, "ok", "", {"input_tokens": 99, "output_tokens": 11}, 5.0),
    ])

    def fake_once(prompt, **kwargs):
        return next(seq_iter)

    monkeypatch.setattr(
        "claude_plan_execute.claude._run_claude_once", fake_once,
    )
    monkeypatch.setattr(
        "claude_plan_execute.claude.time.sleep", lambda s: None,
    )
    p = tmp_path / "u.jsonl"
    code, text = run_claude(
        "p", model="opus", effort="max", max_turns=1,
        timeout_minutes=1, skip_permissions=False, verbose=False,
        phase="implement", usage_jsonl_path=p,
    )
    assert (code, text) == (0, "ok")
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["input"] == 99
    assert rows[0]["output"] == 11


def test_run_claude_jsonl_thread_safety(tmp_path):
    """Pin §D4: the module-level Lock serializes concurrent
    appends. 16 threads × 10 rows each → 160 valid rows; no
    interleaving / partial writes."""
    import threading as _threading
    p = tmp_path / "u.jsonl"

    def writer():
        for _i in range(10):
            claude._append_usage_row(
                p, phase="plan", model="opus",
                input=1, output=1, cache_read=0, cache_create=0,
                duration_seconds=0.5,
            )

    threads = [_threading.Thread(target=writer) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert len(rows) == 160


def test_append_usage_row_swallows_OSError(monkeypatch, tmp_path, capsys):
    """Pin §D4: I/O errors do not raise; a yellow warning prints."""
    p = tmp_path / "missing-dir" / "u.jsonl"
    real_open = open

    def boom(path, *a, **k):
        if str(path).endswith("u.jsonl"):
            raise PermissionError("simulated")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", boom)
    # Must not raise:
    claude._append_usage_row(
        p, phase="plan", model="opus",
        input=1, output=1, cache_read=0, cache_create=0,
        duration_seconds=0.5,
    )
    out = capsys.readouterr().out
    assert "[usage]" in out


def test_run_claude_5tuple_return_shape_from_run_claude_once(monkeypatch):
    """Pin §D2: _run_claude_once returns a 5-tuple. Direct
    invocation with a passing stub asserts the shape."""
    _patch_run_claude_subprocess(monkeypatch, stdout=json.dumps({
        "type": "result", "subtype": "success", "result": "ok",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }))
    from claude_plan_execute.claude import _run_claude_once
    out = _run_claude_once(
        "p", model="opus", effort="max", max_turns=1,
        timeout_minutes=1, skip_permissions=False, verbose=False,
    )
    assert len(out) == 5
    code, text, stderr, usage, elapsed = out
    assert code == 0
    assert usage == {"input_tokens": 1, "output_tokens": 1}
    assert elapsed >= 0


def test_run_claude_once_5tuple_shape_on_timeout_path(monkeypatch):
    """Pin §D2 / Rev-1 C2: the timeout path also returns a 5-tuple
    `(1, "", "", None, 0.0)`."""
    _patch_run_claude_subprocess(monkeypatch, raises_timeout=True)
    from claude_plan_execute.claude import _run_claude_once
    out = _run_claude_once(
        "p", model="opus", effort="max", max_turns=1,
        timeout_minutes=1, skip_permissions=False, verbose=False,
    )
    assert len(out) == 5
    assert out == (1, "", "", None, 0.0)
    # C3: side-channel duration flag is set to 0.0 on the timeout
    # path for symmetry with the returned tuple's elapsed.
    assert claude._last_run_state.duration_seconds == 0.0
    assert claude._last_run_state.timed_out is True


def test_run_claude_once_returns_None_usage_when_stdout_is_garbage(
    monkeypatch,
):
    """Pin Rev-1 I1 (regression): non-JSON stdout drives the
    `except (json.JSONDecodeError, KeyError, TypeError):` branch
    in `_run_claude_once`. With the `usage = None` pre-bind, the
    function returns a well-formed 5-tuple."""
    _patch_run_claude_subprocess(monkeypatch, stdout="not json output")
    from claude_plan_execute.claude import _run_claude_once
    out = _run_claude_once(
        "p", model="opus", effort="max", max_turns=1,
        timeout_minutes=1, skip_permissions=False, verbose=False,
    )
    assert len(out) == 5
    code, text, stderr, usage, elapsed = out
    assert usage is None
    assert elapsed >= 0


def test_run_claude_once_returns_None_usage_when_parsed_is_empty_list(
    monkeypatch,
):
    """Pin Rev-1 I1 (regression): when `parsed = []` (CLI emits no
    events at all), the for-loop never assigns `result_event`, so
    the `if result_event is not None:` branch never executes. The
    pre-bind keeps `usage = None` and the 5-tuple return is
    well-formed."""
    _patch_run_claude_subprocess(monkeypatch, stdout="[]")
    from claude_plan_execute.claude import _run_claude_once
    out = _run_claude_once(
        "p", model="opus", effort="max", max_turns=1,
        timeout_minutes=1, skip_permissions=False, verbose=False,
    )
    assert len(out) == 5
    assert out[3] is None  # usage


def test_run_claude_once_returns_None_usage_when_parsed_is_not_list_or_dict(
    monkeypatch,
):
    """Pin Rev-1 I1 (regression): when `parsed` is a string / number
    (legacy CLI shapes, adversarial output), neither branch fires →
    `result_event` stays None → pre-bind keeps `usage = None`."""
    _patch_run_claude_subprocess(monkeypatch, stdout='"a string"')
    from claude_plan_execute.claude import _run_claude_once
    out = _run_claude_once(
        "p", model="opus", effort="max", max_turns=1,
        timeout_minutes=1, skip_permissions=False, verbose=False,
    )
    assert len(out) == 5
    assert out[3] is None


def test_run_claude_writes_zero_rows_when_intermediate_attempt_has_usage_but_final_does_not(
    monkeypatch, tmp_path,
):
    """Pin §D3 / Rev-1 I2 (per-call contract — discriminating
    test): attempt 1 emits a usage event but is rate-limited;
    attempt 2 (the final attempt) is also rate-limited and emits
    no usage event. Under the per-call contract, the row write at
    the post-loop `return` site sees `last_usage = None` (from
    attempt 2) and writes ZERO rows."""
    seq_iter = iter([
        # Attempt 1: emits usage but is "rate-limited" to force
        # the loop to continue.
        (1, "", "rate_limit_error",
         {"input_tokens": 50, "output_tokens": 10}, 1.0),
        # Attempt 2 (final): rate-limited, no usage.
        (1, "", "rate_limit_error", None, 1.0),
    ])

    def fake_once(prompt, **kwargs):
        return next(seq_iter)

    monkeypatch.setattr(
        "claude_plan_execute.claude._run_claude_once", fake_once,
    )
    monkeypatch.setattr(
        "claude_plan_execute.claude.time.sleep", lambda s: None,
    )
    p = tmp_path / "u.jsonl"
    run_claude(
        "p", model="opus", effort="max", max_turns=1,
        timeout_minutes=1, skip_permissions=False, verbose=False,
        phase="implement", usage_jsonl_path=p,
        max_rate_limit_retries=1,  # 2 attempts total
    )
    # Per-call contract: final attempt's usage (None) wins.
    assert not p.exists() or p.read_text() == ""


# ---------------------------------------------------------------------------
# Y. Token usage — runner threads usage_jsonl_path through phases
#    (task 2 v0.3)
# ---------------------------------------------------------------------------


def _capture_run_claude_kwargs(monkeypatch, *, write_plan=True, tmp_path=None):
    """Capture every run_claude call's kwargs across phases. By
    default the stub writes the plan file on the first call so
    Phase 1 transitions cleanly into Phase 2."""
    calls = []

    def stub(prompt, **kwargs):
        calls.append(kwargs)
        claude._last_run_state.timed_out = False
        claude._last_run_state.subtype = "success"
        # Write the plan file the first time so Phase 1 progresses.
        if write_plan and tmp_path is not None and len(calls) == 1:
            plan_dir = tmp_path / "task-1"
            plan_dir.mkdir(parents=True, exist_ok=True)
            plan_path = plan_dir / "plan.md"
            if not plan_path.exists():
                plan_path.write_text("plan")
        return 0, "ok"

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub)
    return calls


def test_execute_task_passes_phase_plan_to_run_claude(monkeypatch, tmp_path):
    """Pin §D7: Phase 1 plan call carries phase='plan'."""
    calls = _capture_run_claude_kwargs(monkeypatch, tmp_path=tmp_path)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 0,
            "on_max_review_rounds": "auto_approve",
        },
        cli={},
    )
    # First call is Phase 1 plan.
    assert calls[0]["phase"] == "plan"


def test_execute_task_passes_phase_review_to_run_claude(monkeypatch, tmp_path):
    """Pin §D7: skip_to='review' bypasses Plan; the first call is review."""
    calls = _capture_run_claude_kwargs(monkeypatch, tmp_path=tmp_path)
    plan_dir = tmp_path / "task-1"
    plan_dir.mkdir()
    (plan_dir / "plan.md").write_text("plan")
    # Pre-write a passing review file so the loop terminates.
    (plan_dir / "review-1.md").write_text("Verdict: APPROVED\n")
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "auto_approve",
        },
        cli={}, skip_to="review",
    )
    assert calls[0]["phase"] == "review"


def test_execute_task_passes_phase_implement_to_run_claude(monkeypatch, tmp_path):
    """Pin §D7: skip_to='implement' bypasses Plan + Review. The first
    (and only) Phase-3 call carries phase='implement'."""
    calls = _capture_run_claude_kwargs(monkeypatch, tmp_path=tmp_path)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert calls[0]["phase"] == "implement"


def test_execute_task_passes_usage_jsonl_path_to_every_call(
    monkeypatch, tmp_path,
):
    """Pin §D7: every run_claude call gets the per-task usage_path."""
    calls = _capture_run_claude_kwargs(monkeypatch, tmp_path=tmp_path)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 0,
            "on_max_review_rounds": "auto_approve",
        },
        cli={},
    )
    expected = tmp_path / "task-1" / "usage.jsonl"
    assert calls, "expected at least one run_claude call"
    for c in calls:
        assert c["usage_jsonl_path"] == expected


def test_execute_task_dry_run_does_NOT_create_usage_jsonl(monkeypatch, tmp_path):
    """Pin §D7: dry-run returns before any run_claude → no rows
    appended → no file created."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    assert not (tmp_path / "task-1" / "usage.jsonl").exists()


def test_execute_task_revise_phase_uses_phase_revise(monkeypatch, tmp_path):
    """Pin §D7: the revise call (after a NEEDS_REVISION verdict)
    uses phase='revise', distinct from the prior phase='review'."""
    calls = []

    def stub(prompt, **kwargs):
        calls.append(kwargs)
        # On the first review call, write the review file with
        # NEEDS_REVISION → triggers the revise call.
        if len(calls) == 1:
            review_path = tmp_path / "task-1" / "review-1.md"
            review_path.write_text("## Verdict: NEEDS_REVISION\n")
        claude._last_run_state.timed_out = False
        claude._last_run_state.subtype = "success"
        return 0, "ok"

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub)
    plan_dir = tmp_path / "task-1"
    plan_dir.mkdir()
    (plan_dir / "plan.md").write_text("plan")
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "auto_approve",
        },
        cli={}, skip_to="review",
    )
    # Calls: [review-1, revise]. Phase strings: ["review", "revise"].
    assert calls[0]["phase"] == "review"
    assert calls[1]["phase"] == "revise"


# ---------------------------------------------------------------------------
# Z. Token usage — gate runner appends rows (task 2 v0.3)
# ---------------------------------------------------------------------------


def test_run_gates_appends_shell_gate_row_with_duration(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D6: a passing shell gate produces a JSONL row with
    `phase: 'gate:<name>'` and `model: null`."""
    _patch_run_claude_simulating_success(monkeypatch)
    _clean_default_registry.register(
        "ok", Gate(kind="shell", command=("true",)),
    )

    def fake_run(cmd, **kwargs):
        class P:
            pass
        p = P()
        p.returncode = 0
        p.stdout = "ok"
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["ok"]},
        cli={}, skip_to="implement",
    )
    p = tmp_path / "task-1" / "usage.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    gate_rows = [r for r in rows if r["phase"].startswith("gate:")]
    assert len(gate_rows) == 1
    assert gate_rows[0]["phase"] == "gate:ok"
    assert gate_rows[0]["model"] is None
    assert gate_rows[0]["input"] == 0
    assert gate_rows[0]["output"] == 0
    assert gate_rows[0]["duration_seconds"] >= 0


def test_run_gates_passes_usage_jsonl_path_to_agent_gate(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D5/§D6: the AgentGate's run_claude call carries the
    per-task usage.jsonl path."""
    _patch_run_claude_simulating_success(monkeypatch)
    captured = {}

    def stub_agent_run(prompt, **kwargs):
        captured.update(kwargs)
        return 0, "VERDICT: PASS\n"

    monkeypatch.setattr(
        "claude_plan_execute.gates.agent.run_claude", stub_agent_run,
    )
    _clean_default_registry.register(
        "agent-x",
        Gate(kind="agent", prompt_template="body", on_failure="warn"),
    )
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["agent-x"]},
        cli={}, skip_to="implement",
    )
    expected = str(tmp_path / "task-1" / "usage.jsonl")
    assert captured["usage_jsonl_path"] == expected


def test_run_gates_agent_gate_phase_has_gate_prefix(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D5: the agent gate's run_claude call uses
    phase=f'gate:{name}'."""
    _patch_run_claude_simulating_success(monkeypatch)
    captured = {}

    def stub_agent_run(prompt, **kwargs):
        captured.update(kwargs)
        return 0, "VERDICT: PASS\n"

    monkeypatch.setattr(
        "claude_plan_execute.gates.agent.run_claude", stub_agent_run,
    )
    _clean_default_registry.register(
        "test-critique",
        Gate(kind="agent", prompt_template="body", on_failure="warn"),
    )
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["test-critique"],
        },
        cli={}, skip_to="implement",
    )
    assert captured["phase"] == "gate:test-critique"


def test_run_gates_skipped_gate_does_NOT_append_row(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D6: gates filtered by `_when_matches` (e.g.,
    task_class:foo on a task with class=bar) get a passive
    GateResult; no usage row is appended for the skipped gate."""
    _patch_run_claude_simulating_success(monkeypatch)
    _clean_default_registry.register(
        "task-class-only",
        Gate(kind="shell", command=("true",), when=("task_class:other",)),
    )

    def fake_run(cmd, **kwargs):
        class P:
            pass
        p = P()
        p.returncode = 0
        p.stdout = ""
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, task_class="ui"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["task-class-only"],
        },
        cli={}, skip_to="implement",
    )
    p = tmp_path / "task-1" / "usage.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    else:
        rows = []
    gate_rows = [r for r in rows if r["phase"].startswith("gate:")]
    assert len(gate_rows) == 0


def test_run_gates_does_NOT_double_write_agent_row(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §D6: an agent gate's row is written by run_claude inside
    AgentGate.run() (one row). run_gates does NOT also write a row
    in its post-loop."""
    _patch_run_claude_simulating_success(monkeypatch)

    def stub_agent_run(prompt, **kwargs):
        # Have the stub append a row to demonstrate that's the ONE
        # source of agent rows.
        if kwargs.get("usage_jsonl_path") and kwargs.get("phase"):
            from claude_plan_execute.claude import _append_usage_row
            _append_usage_row(
                kwargs["usage_jsonl_path"],
                phase=kwargs["phase"],
                model=kwargs.get("model", "opus"),
                input=10, output=10,
                duration_seconds=0.5,
            )
        return 0, "VERDICT: PASS\n"

    monkeypatch.setattr(
        "claude_plan_execute.gates.agent.run_claude", stub_agent_run,
    )
    _clean_default_registry.register(
        "agent-x",
        Gate(kind="agent", prompt_template="body", on_failure="warn"),
    )
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["agent-x"]},
        cli={}, skip_to="implement",
    )
    p = tmp_path / "task-1" / "usage.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    agent_rows = [r for r in rows if r["phase"] == "gate:agent-x"]
    assert len(agent_rows) == 1   # NOT 2


def test_run_gates_shell_row_when_no_agent_gates_configured(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Cover the kind_by_name path purely for shell gates."""
    _patch_run_claude_simulating_success(monkeypatch)
    _clean_default_registry.register(
        "shell-a", Gate(kind="shell", command=("true",)),
    )
    _clean_default_registry.register(
        "shell-b", Gate(kind="shell", command=("true",)),
    )

    def fake_run(cmd, **kwargs):
        class P:
            pass
        p = P()
        p.returncode = 0
        p.stdout = "ok"
        p.stderr = ""
        return p

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["shell-a", "shell-b"],
        },
        cli={}, skip_to="implement",
    )
    p = tmp_path / "task-1" / "usage.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    gate_phases = sorted(
        r["phase"] for r in rows if r["phase"].startswith("gate:")
    )
    assert gate_phases == ["gate:shell-a", "gate:shell-b"]


def test_run_gates_agent_gate_row_has_model(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Agent rows preserve the actual model name (NOT null)."""
    _patch_run_claude_simulating_success(monkeypatch)

    def stub_agent_run(prompt, **kwargs):
        if kwargs.get("usage_jsonl_path") and kwargs.get("phase"):
            from claude_plan_execute.claude import _append_usage_row
            _append_usage_row(
                kwargs["usage_jsonl_path"],
                phase=kwargs["phase"],
                model=kwargs.get("model", "opus"),
                input=1, output=1, duration_seconds=0.0,
            )
        return 0, "VERDICT: PASS\n"

    monkeypatch.setattr(
        "claude_plan_execute.gates.agent.run_claude", stub_agent_run,
    )
    _clean_default_registry.register(
        "agent-x",
        Gate(kind="agent", prompt_template="body",
             model="haiku", on_failure="warn"),
    )
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["agent-x"]},
        cli={}, skip_to="implement",
    )
    p = tmp_path / "task-1" / "usage.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    agent_rows = [r for r in rows if r["phase"] == "gate:agent-x"]
    assert agent_rows[0]["model"] == "haiku"


# ---------------------------------------------------------------------------
# AA. Token usage — usage_summary roll-up on done (task 2 v0.3)
# ---------------------------------------------------------------------------


def _seed_usage_jsonl(tmp_path, tid, rows):
    """Pre-write a usage.jsonl file at the task's expected path."""
    task_dir = tmp_path / f"task-{tid}"
    task_dir.mkdir(parents=True, exist_ok=True)
    p = task_dir / "usage.jsonl"
    p.write_text(
        "".join(json.dumps(r) + "\n" for r in rows),
        encoding="utf-8",
    )
    return p


def test_execute_task_writes_usage_summary_on_done(monkeypatch, tmp_path):
    """Pin §D8: a task that reaches done with rows in usage.jsonl
    gets a usage_summary on its state entry."""
    _seed_usage_jsonl(tmp_path, 1, [
        {"phase": "plan", "model": "opus",
         "input": 100, "output": 50,
         "cache_read": 30, "cache_create": 5,
         "duration_seconds": 12.0},
    ])
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert "usage_summary" in entry
    # The implement run_claude stub doesn't write to JSONL (it's
    # stubbed at run_claude level, not _run_claude_once); only the
    # seeded row is summed.
    assert entry["usage_summary"]["total_runs"] == 1
    assert entry["usage_summary"]["total_input_tokens"] == 100


def test_execute_task_does_NOT_write_summary_when_jsonl_empty(
    monkeypatch, tmp_path,
):
    """Pin §D8 / absence-robust: zero rows → no usage_summary key."""
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert "usage_summary" not in entry


def test_execute_task_does_NOT_write_summary_on_blocked(
    monkeypatch, tmp_path,
):
    """Pin §D8: blocked-by-rate-limit task does NOT reach the
    summary write site."""
    _seed_usage_jsonl(tmp_path, 1, [
        {"phase": "implement", "model": "opus",
         "input": 100, "output": 50,
         "cache_read": 0, "cache_create": 0,
         "duration_seconds": 10.0},
    ])
    _patch_run_claude_simulating_rate_limit_exhausted(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert "usage_summary" not in entry


def test_execute_task_does_NOT_write_summary_on_dry_run(
    monkeypatch, tmp_path,
):
    """Dry-run path returns BEFORE any run_claude / summary write."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    entry = state.get(1)
    assert "usage_summary" not in entry


def test_execute_task_summary_includes_all_six_keys(monkeypatch, tmp_path):
    """Pin §D9: summary has six keys exactly."""
    _seed_usage_jsonl(tmp_path, 1, [
        {"phase": "plan", "model": "opus",
         "input": 1, "output": 2,
         "cache_read": 3, "cache_create": 4,
         "duration_seconds": 5.0},
    ])
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    s = state.get(1)["usage_summary"]
    assert set(s.keys()) == {
        "total_input_tokens", "total_output_tokens",
        "total_cache_read", "total_cache_create",
        "total_runs", "duration_seconds",
    }


def test_usage_summary_sums_correctly_across_multiple_runs(
    monkeypatch, tmp_path,
):
    """Pin (headline test from task description): summary correctly
    sums multiple rows."""
    _seed_usage_jsonl(tmp_path, 1, [
        {"phase": "plan", "model": "opus",
         "input": 100, "output": 50,
         "cache_read": 30, "cache_create": 5,
         "duration_seconds": 10.0},
        {"phase": "implement", "model": "opus",
         "input": 200, "output": 80,
         "cache_read": 100, "cache_create": 10,
         "duration_seconds": 50.0},
        {"phase": "gate:tests", "model": None,
         "input": 0, "output": 0,
         "cache_read": 0, "cache_create": 0,
         "duration_seconds": 5.5},
    ])
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    s = state.get(1)["usage_summary"]
    assert s["total_input_tokens"] == 300
    assert s["total_output_tokens"] == 130
    assert s["total_cache_read"] == 130
    assert s["total_cache_create"] == 15
    assert s["total_runs"] == 3
    assert s["duration_seconds"] == 65.5


def test_re_run_done_task_summary_reflects_cumulative_jsonl(
    monkeypatch, tmp_path,
):
    """Pin §D8: re-running a task replaces `usage_summary` with a
    fresh roll-up of the WHOLE `usage.jsonl` file (which is
    append-only across runs). The new summary is cumulative, NOT
    just-this-run."""
    _patch_run_claude_simulating_success(monkeypatch)
    # First run with one seeded row.
    _seed_usage_jsonl(tmp_path, 1, [
        {"phase": "plan", "model": "opus",
         "input": 100, "output": 50,
         "cache_read": 0, "cache_create": 0,
         "duration_seconds": 1.0},
    ])
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert state.get(1)["usage_summary"]["total_runs"] == 1
    # Append a second row to JSONL (simulating the second run's writes).
    p = tmp_path / "task-1" / "usage.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps({
            "phase": "implement", "model": "opus",
            "input": 200, "output": 80,
            "cache_read": 0, "cache_create": 0,
            "duration_seconds": 2.0,
        }) + "\n")
    # Re-run.
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    s = state.get(1)["usage_summary"]
    assert s["total_runs"] == 2  # cumulative
    assert s["total_input_tokens"] == 300


def test_usage_summary_robust_to_corrupt_jsonl_line(monkeypatch, tmp_path):
    """Pin §D9: corrupt lines are skipped; valid rows still sum."""
    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    p = task_dir / "usage.jsonl"
    p.write_text(
        json.dumps({"phase": "plan", "input": 100, "output": 50,
                    "cache_read": 0, "cache_create": 0,
                    "duration_seconds": 1.0}) + "\n"
        + "this is not json\n"
        + json.dumps({"phase": "implement", "input": 50, "output": 30,
                      "cache_read": 0, "cache_create": 0,
                      "duration_seconds": 2.0}) + "\n"
    )
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    s = state.get(1)["usage_summary"]
    # Two valid rows summed; one corrupt skipped.
    assert s["total_runs"] == 2
    assert s["total_input_tokens"] == 150


def test_usage_summary_persists_to_state_json(monkeypatch, tmp_path):
    """Pin: usage_summary survives a state.save() round-trip."""
    _seed_usage_jsonl(tmp_path, 1, [
        {"phase": "plan", "model": "opus",
         "input": 100, "output": 50,
         "cache_read": 0, "cache_create": 0,
         "duration_seconds": 5.0},
    ])
    _patch_run_claude_simulating_success(monkeypatch)
    state_path = tmp_path / "state.json"
    state = State(state_path)
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    on_disk = json.loads(state_path.read_text())
    assert "usage_summary" in on_disk["tasks"]["1"]
    assert on_disk["tasks"]["1"]["usage_summary"]["total_input_tokens"] == 100


def test_execute_task_done_status_set_when_no_summary(
    monkeypatch, tmp_path,
):
    """Pin: even when no JSONL exists, status still transitions to
    done (the empty-extras branch)."""
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert state.get(1)["status"] == "done"


# ---------------------------------------------------------------------------
# v0.3 task 8: services integration (Plan §D4 / §D10.4)
# ---------------------------------------------------------------------------

from claude_plan_execute.runner import (  # noqa: E402
    _resolve_services_required,
    _service_teardown_after_task,
)
from claude_plan_execute.services import (  # noqa: E402
    ServiceUnavailableError,
    ServiceUnstableError,
)


# --- Audit-artifact extension (§D4.3) --------------------------------------


def test_execute_task_resolved_config_includes_services_required(
    monkeypatch, tmp_path,
):
    """Plan §D4.3: the audit artifact gains a top-level
    services_required field with {value, from_layers}. Layer-merge is
    verified in test_resolve_services_required_*; this test only pins
    the shape on the audit."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, services_required_extra=["redis"]), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "services_required": ["postgres"],
        },
        cli={}, dry_run=True,
    )
    cfg = _read_resolved_config(tmp_path, 1)
    assert "services_required" in cfg
    assert "value" in cfg["services_required"]
    assert "from_layers" in cfg["services_required"]
    assert set(cfg["services_required"]["from_layers"].keys()) == {
        "defaults", "phase", "task",
    }
    assert cfg["services_required"]["value"] == ["postgres", "redis"]


# --- _resolve_services_required pure helper (§D4.1) ------------------------


def test_resolve_services_required_empty_returns_empty():
    """No layers contribute → []."""
    assert _resolve_services_required(
        task={}, file_defaults={}, phase_dict={},
    ) == []


def test_resolve_services_required_defaults_only():
    """defaults.services_required only."""
    out = _resolve_services_required(
        task={},
        file_defaults={"services_required": ["a", "b"]},
        phase_dict={},
    )
    assert out == ["a", "b"]


def test_resolve_services_required_phase_extra_appended():
    """phase.services_required_extra is appended after defaults."""
    out = _resolve_services_required(
        task={},
        file_defaults={"services_required": ["a"]},
        phase_dict={"services_required_extra": ["b"]},
    )
    assert out == ["a", "b"]


def test_resolve_services_required_task_extra_appended():
    """task.services_required_extra is appended last."""
    out = _resolve_services_required(
        task={"services_required_extra": ["c"]},
        file_defaults={"services_required": ["a"]},
        phase_dict={"services_required_extra": ["b"]},
    )
    assert out == ["a", "b", "c"]


def test_resolve_services_required_dedupes_first_seen_order():
    """Duplicates are dropped; first-seen order is preserved."""
    out = _resolve_services_required(
        task={"services_required_extra": ["a", "x"]},
        file_defaults={"services_required": ["a", "b"]},
        phase_dict={"services_required_extra": ["b", "c"]},
    )
    assert out == ["a", "b", "c", "x"]


def test_resolve_services_required_overlay_three_layers():
    """All three layers contribute distinct names in order."""
    out = _resolve_services_required(
        task={"services_required_extra": ["t"]},
        file_defaults={"services_required": ["d1", "d2"]},
        phase_dict={"services_required_extra": ["p"]},
    )
    assert out == ["d1", "d2", "p", "t"]


# --- _StubServiceManager + integration tests (§D10.4) ----------------------


class _StubServiceManager:
    def __init__(self, *, ensure_raises=None, poll_returns=None,
                 poll_raises=None):
        self.ensure_calls: list[list[str]] = []
        self.tear_down_calls: list[str] = []
        self.reset_count_calls: list[str] = []
        self._ensure_raises = ensure_raises
        self._poll_returns = poll_returns or []
        self._poll_raises = poll_raises

    def names(self):
        return []

    def spec_lifetime(self, n):
        return "run"

    def is_running(self, n):
        return True

    def reset_crash_count(self, n):
        self.reset_count_calls.append(n)

    def ensure(self, required):
        self.ensure_calls.append(list(required))
        if self._ensure_raises:
            raise self._ensure_raises

    def poll_health(self):
        if self._poll_raises:
            raise self._poll_raises
        return list(self._poll_returns)

    def tear_down_for_scope(self, scope):
        self.tear_down_calls.append(scope)

    def tear_down_all(self):
        self.tear_down_calls.append("all")


def test_execute_task_calls_ensure_at_task_entry_with_required_services(
    monkeypatch, tmp_path,
):
    """Plan §D4.4: the task-entry block calls ensure() with the
    resolved required-services list."""
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    stub = _StubServiceManager()
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "services_required": ["dev"],
        },
        cli={}, skip_to="implement",
        service_manager=stub,
    )
    # The task-entry ensure runs first; the pre-Phase-4 ensure runs
    # second. Both with the same required set.
    assert stub.ensure_calls == [["dev"], ["dev"]]


def test_execute_task_calls_ensure_again_before_phase_4(
    monkeypatch, tmp_path,
):
    """Plan §D4.5: a second ensure() fires right before Phase 4."""
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    stub = _StubServiceManager()
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "services_required": ["dev"],
        },
        cli={}, skip_to="implement",
        service_manager=stub,
    )
    assert len(stub.ensure_calls) == 2


def test_execute_task_blocks_on_service_unavailable_at_entry(
    monkeypatch, tmp_path,
):
    """Plan §D4.4: ServiceUnavailableError at task-entry blocks the
    task with block_reason=service_unavailable."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    stub = _StubServiceManager(
        ensure_raises=ServiceUnavailableError("dev", "boom"),
    )
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "services_required": ["dev"],
        },
        cli={}, skip_to="implement",
        service_manager=stub,
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry["block_reason"] == "service_unavailable"


def test_execute_task_records_service_unavailable_failure_record(
    monkeypatch, tmp_path,
):
    """Plan §D4.4: the failure_class on the recorded record is
    'service_unavailable'."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    stub = _StubServiceManager(
        ensure_raises=ServiceUnavailableError("dev", "boom"),
    )
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "services_required": ["dev"],
        },
        cli={}, skip_to="implement",
        service_manager=stub,
    )
    attempts = state.get(1)["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["failure_class"] == "service_unavailable"
    assert attempts[0]["phase"] == "services"


def test_execute_task_blocks_on_service_unstable_from_poll_health(
    monkeypatch, tmp_path,
):
    """Plan §D4.4: ServiceUnstableError from poll_health() at task
    entry blocks with block_reason=service_unstable."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    stub = _StubServiceManager(
        poll_raises=ServiceUnstableError("dev", "3 crashes"),
    )
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "services_required": ["dev"],
        },
        cli={}, skip_to="implement",
        service_manager=stub,
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry["block_reason"] == "service_unstable"
    attempts = state.get(1)["attempts"]
    assert attempts[0]["failure_class"] == "service_unstable"


def test_execute_task_records_service_crashed_for_auto_recovered_event(
    monkeypatch, tmp_path,
):
    """Plan §D4.4: a crashed_restarted poll event creates a
    service_crashed FailureRecord (informational, non-blocking)."""
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    stub = _StubServiceManager(
        poll_returns=[{
            "name": "dev", "event": "crashed_restarted",
            "exit_code": 1, "crash_count": 1,
        }],
    )
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "services_required": ["dev"],
        },
        cli={}, skip_to="implement",
        service_manager=stub,
    )
    entry = state.get(1)
    # A crashed_restarted record was added but the task still completes
    assert entry["status"] == "done"
    crashed = [a for a in entry["attempts"]
               if a["failure_class"] == "service_crashed"]
    assert len(crashed) >= 1


def test_execute_task_tears_down_gate_then_task_on_success(
    monkeypatch, tmp_path,
):
    """Plan §D4.6: success-path explicit teardown calls
    tear_down_for_scope('gate') then tear_down_for_scope('task'),
    then the finally re-calls them as no-ops."""
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    stub = _StubServiceManager()
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
        service_manager=stub,
    )
    # Both explicit (success) and finally tear-downs fire.
    assert stub.tear_down_calls.count("gate") >= 1
    assert stub.tear_down_calls.count("task") >= 1
    assert state.get(1)["status"] == "done"


def test_execute_task_tears_down_on_blocked_path_too(
    monkeypatch, tmp_path,
):
    """Plan §D4.6: even when the task is blocked, the finally clause
    runs the gate/task tear-downs."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    stub = _StubServiceManager(
        ensure_raises=ServiceUnavailableError("dev", "boom"),
    )
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "services_required": ["dev"],
        },
        cli={}, skip_to="implement",
        service_manager=stub,
    )
    assert "gate" in stub.tear_down_calls
    assert "task" in stub.tear_down_calls


def test_execute_task_skips_service_logic_when_manager_is_None(
    monkeypatch, tmp_path,
):
    """Plan §D4.4: with manager=None, no AttributeError; the runner
    behaves as in the legacy path."""
    _patch_run_claude_simulating_success(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
        service_manager=None,
    )
    assert state.get(1)["status"] == "done"


def test_execute_task_finally_tears_down_on_uncaught_exception(
    monkeypatch, tmp_path,
):
    """Plan §D4.6 (review-2 C4): an uncaught exception mid-Phase-3
    propagates out, but the finally clause still tears down gate/task
    scopes."""
    state = State(tmp_path / "state.json")
    stub = _StubServiceManager()

    def boom(*_a, **_kw):
        raise RuntimeError("boom from claude")

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", boom)
    with pytest.raises(RuntimeError):
        execute_task(
            _make_task(id=1), state,
            file_defaults={"plans_dir": str(tmp_path)},
            cli={}, skip_to="implement",
            service_manager=stub,
        )
    assert "gate" in stub.tear_down_calls
    assert "task" in stub.tear_down_calls


def test_execute_task_skips_service_block_in_dry_run(
    monkeypatch, tmp_path,
):
    """Plan §D4.4 (review-2 Issue 1): with dry_run=True, the
    task-entry service block is skipped — ensure() / poll_health()
    are NOT called even when manager is non-None and required is
    non-empty. Tear-downs still fire (no-ops)."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    stub = _StubServiceManager()
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "services_required": ["dev"],
        },
        cli={}, dry_run=True,
        service_manager=stub,
    )
    assert stub.ensure_calls == []
    # tear_down_for_scope IS called via the finally
    assert "gate" in stub.tear_down_calls
    assert "task" in stub.tear_down_calls


# --- cli.main integration (Plan §D5 / §D10.4) ------------------------------


def _write_minimal_tasks_yaml_with_service(tmp_path: Path) -> Path:
    """Write a tasks.yaml fixture declaring one lifetime=run service
    and a single task that requires it via defaults.services_required."""
    p = tmp_path / "tasks.yaml"
    p.write_text(
        "version: 2\n"
        "defaults:\n"
        "  services_required: [dev]\n"
        "services:\n"
        "  - name: dev\n"
        "    command: \"true\"\n"
        "    lifetime: run\n"
        "    ready_check: { kind: shell, command: 'true' }\n"
        "tasks:\n"
        "  - { id: 1, title: T, description: D }\n",
    )
    return p


def _write_two_phase_tasks_yaml_with_phase_services(tmp_path: Path) -> Path:
    """Write a tasks.yaml fixture with two phases each declaring a
    lifetime=phase service in their defaults.services_required_extra."""
    p = tmp_path / "tasks.yaml"
    p.write_text(
        "version: 2\n"
        "phases:\n"
        "  - id: a\n"
        "    defaults:\n"
        "      services_required_extra: [svc-a]\n"
        "  - id: b\n"
        "    defaults:\n"
        "      services_required_extra: [svc-b]\n"
        "services:\n"
        "  - name: svc-a\n"
        "    command: \"true\"\n"
        "    lifetime: phase\n"
        "    ready_check: { kind: shell, command: 'true' }\n"
        "  - name: svc-b\n"
        "    command: \"true\"\n"
        "    lifetime: phase\n"
        "    ready_check: { kind: shell, command: 'true' }\n"
        "tasks:\n"
        "  - { id: 1, title: T1, description: D, phase: a }\n"
        "  - { id: 2, title: T2, description: D, phase: b, depends_on: [1] }\n",
    )
    return p


def _patch_service_manager_recording(monkeypatch):
    """Replace cli.ServiceManager with a stub that records calls.
    Returns the stub instance after construction (set on the first call)."""
    captured: dict = {"stub": None}

    class _Recorder:
        def __init__(self, services, plans_dir):
            self.services = services
            self.plans_dir = plans_dir
            self.ensure_calls: list[list[str]] = []
            self.tear_down_calls: list[str] = []
            self._lifetime_by_name = {
                s["name"]: s.get("lifetime", "run") for s in services
            }
            captured["stub"] = self

        def names(self):
            return list(self._lifetime_by_name.keys())

        def spec_lifetime(self, name):
            return self._lifetime_by_name.get(name, "run")

        def is_running(self, name):
            return False

        def ensure(self, required):
            self.ensure_calls.append(list(required))

        def tear_down_for_scope(self, scope):
            self.tear_down_calls.append(scope)

        def tear_down_all(self):
            self.tear_down_calls.append("all")

        def poll_health(self):
            return []

        def reset_crash_count(self, name):
            pass

    monkeypatch.setattr(
        "claude_plan_execute.cli.ServiceManager", _Recorder,
    )
    return captured


def _patch_cli_external_dependencies(monkeypatch):
    """Stub external CLI deps so cli.main can run without claude
    binary, real preflight commands, or signals."""
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/true")
    monkeypatch.setattr(
        "claude_plan_execute.cli.signal.signal",
        lambda *_a, **_kw: None,
    )
    # Force preflight short-circuit via --skip-preflight (we'll add
    # to argv in each test).


def test_cli_main_does_not_start_services_in_dry_run(
    monkeypatch, tmp_path,
):
    """Plan §D5.2 (review-2 Issue 1): --dry-run must NOT call
    ensure() at the run-scope layer."""
    from claude_plan_execute import cli

    _write_minimal_tasks_yaml_with_service(tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_cli_external_dependencies(monkeypatch)
    cap = _patch_service_manager_recording(monkeypatch)

    monkeypatch.setattr(
        "sys.argv",
        ["claude-plan-execute", "--dry-run", "--skip-preflight"],
    )
    cli.main()
    stub = cap["stub"]
    assert stub is not None
    assert stub.ensure_calls == []
    # tear_down_all must still run via the finally
    assert "all" in stub.tear_down_calls


def test_cli_main_starts_run_scope_services_in_real_run(
    monkeypatch, tmp_path,
):
    """Plan §D5.2 positive complement: a non-dry-run invocation
    triggers ensure(['dev']) before the task loop."""
    from claude_plan_execute import cli

    _write_minimal_tasks_yaml_with_service(tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_cli_external_dependencies(monkeypatch)
    cap = _patch_service_manager_recording(monkeypatch)

    # Mark the only task as done so execute_task is skipped (we're
    # only verifying the run-scope ensure here).
    state_path = tmp_path / "plans" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({
            "schema_version": 3,
            "tasks": {"1": {"status": "done", "attempts": [],
                            "current_attempt": 1}},
        }),
    )

    monkeypatch.setattr(
        "sys.argv",
        ["claude-plan-execute", "--skip-preflight"],
    )
    cli.main()
    stub = cap["stub"]
    assert stub is not None
    assert stub.ensure_calls == [["dev"]]
    assert "all" in stub.tear_down_calls


def test_cli_main_skips_phase_scope_ensure_in_dry_run(
    monkeypatch, tmp_path,
):
    """Plan §D5.3 (review-2 Issue 1): in --dry-run, phase-boundary
    ensure() is NOT called even when phase-scope services are
    declared. tear_down_for_scope('phase') IS called between phases
    (idempotent no-op)."""
    from claude_plan_execute import cli

    _write_two_phase_tasks_yaml_with_phase_services(tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_cli_external_dependencies(monkeypatch)
    cap = _patch_service_manager_recording(monkeypatch)

    monkeypatch.setattr(
        "sys.argv",
        ["claude-plan-execute", "--dry-run", "--skip-preflight"],
    )
    cli.main()
    stub = cap["stub"]
    assert stub is not None
    # No ensure() at any level in dry-run
    assert stub.ensure_calls == []
    # tear_down_for_scope('phase') fires on phase change (between t1 and t2)
    assert "phase" in stub.tear_down_calls
    assert "all" in stub.tear_down_calls


def test_single_task_phase_scope_service_torn_down_only_at_run_end(
    monkeypatch, tmp_path,
):
    """Plan §11.13: a single --task invocation does NOT iterate over
    `ordered_tasks` and therefore never observes a phase transition.
    A lifetime=phase service started by execute_task's task-entry
    ensure is reaped only by the final tear_down_all (the §D5.3
    finally), NOT by a mid-task tear_down_for_scope('phase').
    Documented as intentional — bounded leak in single-task mode."""
    from claude_plan_execute import cli

    _write_two_phase_tasks_yaml_with_phase_services(tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_cli_external_dependencies(monkeypatch)
    cap = _patch_service_manager_recording(monkeypatch)

    monkeypatch.setattr(
        "sys.argv",
        ["claude-plan-execute", "--task", "1",
         "--dry-run", "--skip-preflight"],
    )
    cli.main()
    stub = cap["stub"]
    assert stub is not None
    # Single-task branch has no phase tracking — phase-scope tear-down
    # is NOT called mid-task.
    assert "phase" not in stub.tear_down_calls
    # tear_down_all IS called via the §D5.3 outer finally — this is
    # what cleans up phase-scope services in single-task mode.
    assert "all" in stub.tear_down_calls


# ===========================================================================
# BB. T9 — branch_per_task + auto_pr + auto_merge (v0.3 §2.12)
# ===========================================================================
#
# Pattern: every integration test in this section uses the
# `runner.subprocess.run` discriminator (matching cmd[0]/cmd[1]/cmd[2]) so
# the helpers run end-to-end. Direct helper-mocking is reserved for pure-
# function tests (slug / render / parent resolution).


# --- BB.1: slug + render + parent-branch helpers (pure) -------------------


def test_slug_normal_title():
    """Pin §D4: lowercase + dash collapse for a typical title."""
    from claude_plan_execute.runner import _slug
    assert _slug("Add Feature X") == "add-feature-x"


def test_slug_collapses_underscores():
    """Pin §D4: underscores are non-alphanumeric → collapsed to '-'."""
    from claude_plan_execute.runner import _slug
    assert _slug("feat: add_user_auth") == "feat-add-user-auth"


def test_slug_strips_non_ascii_letters():
    """Pin §D4: non-ASCII characters drop to '-' then trailing-strip."""
    from claude_plan_execute.runner import _slug
    assert _slug("café") == "caf"


def test_slug_empty_title_returns_sentinel():
    """Pin §D4: empty / non-alphanumeric titles → 'task' sentinel."""
    from claude_plan_execute.runner import _slug
    assert _slug("") == "task"
    assert _slug("   ") == "task"
    assert _slug("!!!") == "task"


def test_slug_caps_body_at_30_chars():
    """Pin §D4: 30-char body cap, never trailing '-'."""
    from claude_plan_execute.runner import _slug
    long_title = "a" * 60
    s = _slug(long_title)
    assert len(s) <= 30
    assert not s.endswith("-")
    # Mid-separator-run case
    s2 = _slug("aaaaa bbbbb ccccc ddddd eeeee fffff ggggg hhhhh")
    assert len(s2) <= 30
    assert not s2.endswith("-")


def test_render_branch_name():
    """Pin §D7: standard template substitution."""
    from claude_plan_execute.runner import _render_branch_name
    assert _render_branch_name("task/{id}-{slug}", 17, "foo") == "task/17-foo"


def test_render_branch_name_unknown_placeholder_raises():
    """Pin §D7: unknown placeholder → _GitOperationError with hint."""
    from claude_plan_execute.runner import (
        _render_branch_name, _GitOperationError,
    )
    with pytest.raises(_GitOperationError) as exc_info:
        _render_branch_name("task/{wat}", 1, "x")
    assert "wat" in str(exc_info.value)


def test_resolve_parent_branch_no_deps_returns_origin_base():
    """Pin §D7: empty depends_on → 'origin/{base}'."""
    from claude_plan_execute.runner import _resolve_parent_branch
    state = State(_BB_tmp_state())
    task = {"id": 1, "title": "T", "description": "D"}
    assert _resolve_parent_branch(task, state, "main") == "origin/main"


def test_resolve_parent_branch_done_dep_with_branch_wins():
    """Pin §D7: a single done+branched dep returns that branch."""
    from claude_plan_execute.runner import _resolve_parent_branch
    state = State(_BB_tmp_state())
    state.set_status(3, "done", branch="task/3-foo")
    task = {"id": 9, "title": "T", "description": "D", "depends_on": [3]}
    assert _resolve_parent_branch(task, state, "main") == "task/3-foo"


def test_resolve_parent_branch_undone_dep_falls_back_to_origin():
    """Pin §D7: a pending dep without a branch falls through to origin."""
    from claude_plan_execute.runner import _resolve_parent_branch
    state = State(_BB_tmp_state())
    # 3 is pending by default; no branch field
    task = {"id": 9, "title": "T", "description": "D", "depends_on": [3]}
    assert _resolve_parent_branch(task, state, "develop") == "origin/develop"


def test_resolve_parent_branch_last_in_depends_on_wins():
    """Pin §D7: depends_on order is the tie-break — last done+branched dep
    wins (operationalizes 'latest done' deterministically)."""
    from claude_plan_execute.runner import _resolve_parent_branch
    state = State(_BB_tmp_state())
    state.set_status(3, "done", branch="task/3-foo")
    state.set_status(7, "done", branch="task/7-bar")
    task = {"id": 9, "title": "T", "description": "D", "depends_on": [3, 7]}
    assert _resolve_parent_branch(task, state, "main") == "task/7-bar"


# Helper: per-test temp state.json path (uses tempfile to avoid cross-test
# fixture leakage; State auto-creates the file on first save).
def _BB_tmp_state():
    import tempfile
    fd, path = tempfile.mkstemp(prefix="bb_state_", suffix=".json")
    import os
    os.close(fd)
    os.unlink(path)
    return Path(path)


def _BB_make_fake_run(*, branch_exists=False, push_rc=0,
                     pr_list_stdout="", pr_list_rc=0,
                     pr_create_rc=0, pr_create_stdout="https://github.com/x/y/pull/1",
                     merge_rc=0, fetch_rc=0, checkout_rc=0,
                     create_branch_rc=0, diff_stdout=""):
    """Build a `subprocess.run` stub that discriminates on cmd[]
    shape and records every invocation. Returns `(fake_run, calls)`.
    All defaults are happy-path; pass per-test overrides to inject
    failures.
    """
    calls: list[list] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))

        class P:
            pass

        p = P()
        p.stderr = ""
        if cmd[:2] == ["git", "rev-parse"]:
            p.returncode = 0 if branch_exists else 1
            p.stdout = ""
        elif cmd[:2] == ["git", "fetch"]:
            p.returncode = fetch_rc
            p.stdout = ""
            p.stderr = "fetch failed" if fetch_rc != 0 else ""
        elif cmd[:3] == ["git", "checkout", "-b"]:
            p.returncode = create_branch_rc
            p.stdout = ""
            p.stderr = "checkout -b failed" if create_branch_rc != 0 else ""
        elif cmd[:2] == ["git", "checkout"]:
            p.returncode = checkout_rc
            p.stdout = ""
        elif cmd[:2] == ["git", "diff"]:
            p.returncode = 0
            p.stdout = diff_stdout
        elif cmd[:2] == ["git", "push"]:
            p.returncode = push_rc
            p.stdout = ""
            p.stderr = "push rejected" if push_rc != 0 else ""
        elif cmd[:3] == ["gh", "pr", "list"]:
            p.returncode = pr_list_rc
            p.stdout = pr_list_stdout
        elif cmd[:3] == ["gh", "pr", "create"]:
            p.returncode = pr_create_rc
            p.stdout = pr_create_stdout
            p.stderr = "create failed" if pr_create_rc != 0 else ""
        elif cmd[:3] == ["gh", "pr", "merge"]:
            p.returncode = merge_rc
            p.stdout = ""
            p.stderr = "merge failed" if merge_rc != 0 else ""
        elif cmd[:2] == ["git", "status"]:
            p.returncode = 0
            p.stdout = ""
        else:
            p.returncode = 0
            p.stdout = ""
        return p

    return fake_run, calls


# --- BB.2: branch hook integration in execute_task -----------------------


def test_branch_per_task_false_no_git_subprocess_calls(monkeypatch, tmp_path):
    """Pin §D13: default-off → zero new git/gh shell-outs. Only the
    T7 diff-capture (`git diff HEAD~1..HEAD`) and T6 porcelain
    (`git status --porcelain`) calls are allowed; no rev-parse, no
    fetch, no push, no gh."""
    fake_run, calls = _BB_make_fake_run()
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    # Only allowed shapes: diff (T7) and status (T6 / porcelain)
    for cmd in calls:
        assert cmd[:2] in (["git", "diff"], ["git", "status"]), (
            f"Unexpected git/gh invocation when branch_per_task=False: {cmd}"
        )
    # No gh at all
    assert not any(cmd[0] == "gh" for cmd in calls)
    # No rev-parse / fetch / push / checkout
    assert not any(
        cmd[:2] in (["git", "rev-parse"], ["git", "fetch"],
                    ["git", "push"], ["git", "checkout"])
        for cmd in calls
    )


def test_branch_per_task_true_creates_branch_from_origin_main(
    monkeypatch, tmp_path,
):
    """Pin §D6: branch_per_task=True with no existing local branch
    → fetch origin main, then `git checkout -b task/1-t origin/main`."""
    fake_run, calls = _BB_make_fake_run(branch_exists=False)
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "branch_per_task": True,
        },
        cli={}, skip_to="implement",
    )
    # Fetch origin main
    assert ["git", "fetch", "origin", "main"] in calls
    # Create branch from origin/main
    assert ["git", "checkout", "-b", "task/1-t", "origin/main"] in calls


def test_branch_per_task_true_idempotent_re_checkout_when_exists(
    monkeypatch, tmp_path,
):
    """Pin §D7: existing local branch → plain `git checkout <branch>`,
    NOT `-b`."""
    fake_run, calls = _BB_make_fake_run(branch_exists=True)
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "branch_per_task": True,
        },
        cli={}, skip_to="implement",
    )
    # Idempotent re-checkout (no -b)
    assert ["git", "checkout", "task/1-t"] in calls
    assert not any(cmd[:3] == ["git", "checkout", "-b"] for cmd in calls)
    # No fetch when branch exists locally
    assert not any(cmd[:2] == ["git", "fetch"] for cmd in calls)


def test_branch_recorded_on_state_entry(monkeypatch, tmp_path):
    """Pin §D6: after the branch hook, state entry has `branch=<name>`."""
    fake_run, _ = _BB_make_fake_run(branch_exists=False)
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "branch_per_task": True,
        },
        cli={}, skip_to="implement",
    )
    assert state.get(1).get("branch") == "task/1-t"


def test_branch_failure_blocks_with_branch_checkout_failed_class(
    monkeypatch, tmp_path,
):
    """Pin §D6: a failing `git fetch` surfaces as blocked +
    failure_class=branch_checkout_failed."""
    fake_run, _ = _BB_make_fake_run(branch_exists=False, fetch_rc=1)
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "branch_per_task": True,
        },
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry.get("block_reason") == "branch_checkout_failed"
    attempts = entry["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["failure_class"] == "branch_checkout_failed"
    assert attempts[0]["phase"] == "branch"


def test_diff_target_switches_when_branch_per_task_true(
    monkeypatch, tmp_path,
):
    """Pin §D5: with branch_per_task=True + base_branch=develop, the
    `git diff` cmd is `["git", "diff", "origin/develop...HEAD"]` —
    NOT the default `HEAD~1..HEAD`."""
    fake_run, calls = _BB_make_fake_run(branch_exists=True)
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "branch_per_task": True,
            "base_branch": "develop",
        },
        cli={}, skip_to="implement",
    )
    diff_cmds = [cmd for cmd in calls if cmd[:2] == ["git", "diff"]]
    assert diff_cmds, "expected at least one git diff invocation"
    assert ["git", "diff", "origin/develop...HEAD"] in diff_cmds
    # Old HEAD~1..HEAD form must NOT fire when branch_per_task=True
    assert ["git", "diff", "HEAD~1..HEAD"] not in diff_cmds


# --- BB.3: auto_pr / auto_merge integration ------------------------------


def test_auto_pr_creates_when_no_existing_pr(monkeypatch, tmp_path):
    """Pin §3.A / §D9: empty pr list stdout → gh pr create runs;
    state ends `done` with `pr_url` set to the create stdout."""
    fake_run, calls = _BB_make_fake_run(
        branch_exists=True,
        pr_list_stdout="",
        pr_create_stdout="https://github.com/foo/bar/pull/42\n",
    )
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "branch_per_task": True,
            "auto_pr": True,
        },
        cli={}, skip_to="implement",
    )
    assert state.get(1)["status"] == "done"
    assert state.get(1).get("pr_url") == "https://github.com/foo/bar/pull/42"
    # gh pr create fired
    assert any(cmd[:3] == ["gh", "pr", "create"] for cmd in calls)


def test_auto_pr_uses_existing_url_when_pr_already_open(
    monkeypatch, tmp_path,
):
    """Pin §3.A: non-empty pr list stdout → use that URL; NO gh pr
    create invoked."""
    fake_run, calls = _BB_make_fake_run(
        branch_exists=True,
        pr_list_stdout="https://github.com/foo/bar/pull/7\n",
    )
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "branch_per_task": True,
            "auto_pr": True,
        },
        cli={}, skip_to="implement",
    )
    assert state.get(1)["status"] == "done"
    assert state.get(1).get("pr_url") == "https://github.com/foo/bar/pull/7"
    assert not any(cmd[:3] == ["gh", "pr", "create"] for cmd in calls)


def test_auto_pr_push_failure_blocks_with_push_failed_class(
    monkeypatch, tmp_path,
):
    """Pin §D9: failing git push → blocked + failure_class=push_failed."""
    fake_run, _ = _BB_make_fake_run(branch_exists=True, push_rc=1)
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "branch_per_task": True,
            "auto_pr": True,
        },
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry.get("block_reason") == "push_failed"
    attempts = entry["attempts"]
    assert any(
        a["failure_class"] == "push_failed" and a["phase"] == "post_implement"
        for a in attempts
    )


def test_auto_pr_create_failure_blocks_with_pr_create_failed_class(
    monkeypatch, tmp_path,
):
    """Pin §D9: empty pr list stdout + non-zero pr create exit →
    blocked + failure_class=pr_create_failed."""
    fake_run, _ = _BB_make_fake_run(
        branch_exists=True, pr_list_stdout="", pr_create_rc=1,
    )
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "branch_per_task": True,
            "auto_pr": True,
        },
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry.get("block_reason") == "pr_create_failed"
    attempts = entry["attempts"]
    assert any(
        a["failure_class"] == "pr_create_failed"
        and a["phase"] == "post_implement"
        for a in attempts
    )


def test_auto_merge_invokes_gh_pr_merge_with_correct_args(
    monkeypatch, tmp_path,
):
    """Pin §D9 Comment-A: with auto_merge=True, after a successful PR
    open, `gh pr merge --auto --squash <branch>` runs (positional
    branch arg) and state ends `done`."""
    fake_run, calls = _BB_make_fake_run(
        branch_exists=True,
        pr_list_stdout="https://github.com/foo/bar/pull/9\n",
    )
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "branch_per_task": True,
            "auto_pr": True,
            "auto_merge": True,
        },
        cli={}, skip_to="implement",
    )
    assert state.get(1)["status"] == "done"
    merge_cmds = [c for c in calls if c[:3] == ["gh", "pr", "merge"]]
    assert merge_cmds, "expected gh pr merge invocation"
    cmd = merge_cmds[0]
    assert cmd == ["gh", "pr", "merge", "--auto", "--squash", "task/1-t"]


def test_auto_merge_failure_blocks_with_merge_failed_class(
    monkeypatch, tmp_path,
):
    """Pin §D9: failing gh pr merge → blocked + failure_class=merge_failed."""
    fake_run, _ = _BB_make_fake_run(
        branch_exists=True,
        pr_list_stdout="https://github.com/foo/bar/pull/9\n",
        merge_rc=1,
    )
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "branch_per_task": True,
            "auto_pr": True,
            "auto_merge": True,
        },
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry.get("block_reason") == "merge_failed"
    attempts = entry["attempts"]
    assert any(
        a["failure_class"] == "merge_failed" and a["phase"] == "post_implement"
        for a in attempts
    )


def test_create_branch_with_missing_parent_blocks_with_clear_error(
    monkeypatch, tmp_path,
):
    """Pin §R2 edge case: dependency's local branch deleted between
    runs. Setup: dep task is `done` with `branch="task/3-foo"` on
    its state entry, but `git rev-parse --verify task/3-foo`
    returncode=1 AND `git checkout -b new task/3-foo` returncode=1
    (parent ref unknown). Assert state goes blocked + failure_class
    branch_checkout_failed, AND `reason` contains the missing parent
    name so the operator sees which ref is missing."""
    fake_run, _ = _BB_make_fake_run(
        branch_exists=False, create_branch_rc=1,
    )
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    state.set_status(3, "done", branch="task/3-foo")
    task = _make_task(id=9, depends_on=[3])
    execute_task(
        task, state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "branch_per_task": True,
        },
        cli={}, skip_to="implement",
    )
    entry = state.get(9)
    assert entry["status"] == "blocked"
    assert entry.get("block_reason") == "branch_checkout_failed"
    attempts = entry["attempts"]
    assert len(attempts) >= 1
    rec = attempts[-1]
    assert rec["failure_class"] == "branch_checkout_failed"
    # Reason must reference the parent ref that's missing
    assert "task/3-foo" in rec["reason"]


# --- BB.4: bonus tests ----------------------------------------------------


def test_branch_per_task_without_auto_pr_records_branch_only(
    monkeypatch, tmp_path,
):
    """Pin §D9 `if branch_per_task and auto_pr:` — with auto_pr=False,
    state has `branch` but NO `pr_url`."""
    fake_run, calls = _BB_make_fake_run(branch_exists=True)
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "branch_per_task": True,
            "auto_pr": False,
        },
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "done"
    assert entry.get("branch") == "task/1-t"
    assert "pr_url" not in entry
    assert not any(cmd[0] == "gh" for cmd in calls)
    assert not any(cmd[:2] == ["git", "push"] for cmd in calls)


def test_pr_body_falls_back_to_default_when_template_missing(
    monkeypatch, tmp_path,
):
    """Pin §D8: missing template file → built-in default body.
    Asserted via the `--body` arg passed to `gh pr create`."""
    fake_run, calls = _BB_make_fake_run(
        branch_exists=True, pr_list_stdout="",
    )
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, title="My T", description="Some desc"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "branch_per_task": True,
            "auto_pr": True,
            "pr_body_template": str(tmp_path / "missing.md"),  # nonexistent
        },
        cli={}, skip_to="implement",
    )
    create_cmds = [c for c in calls if c[:3] == ["gh", "pr", "create"]]
    assert create_cmds, "expected gh pr create"
    cmd = create_cmds[0]
    body_idx = cmd.index("--body") + 1
    body = cmd[body_idx]
    assert body.startswith("Auto-opened by `claude-plan-execute`")


def test_pr_body_uses_template_file_when_present(
    monkeypatch, tmp_path,
):
    """Pin §D8: when the template file exists, its contents are
    rendered with the supported placeholders."""
    fake_run, calls = _BB_make_fake_run(
        branch_exists=True, pr_list_stdout="",
    )
    monkeypatch.setattr(
        "claude_plan_execute.runner.subprocess.run", fake_run,
    )
    _patch_run_claude_simulating_success(monkeypatch)

    tpl = tmp_path / "tpl.md"
    tpl.write_text("My body for {title}", encoding="utf-8")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, title="MyT", description="D"), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "branch_per_task": True,
            "auto_pr": True,
            "pr_body_template": str(tpl),
        },
        cli={}, skip_to="implement",
    )
    create_cmds = [c for c in calls if c[:3] == ["gh", "pr", "create"]]
    assert create_cmds
    cmd = create_cmds[0]
    body_idx = cmd.index("--body") + 1
    assert cmd[body_idx] == "My body for MyT"


# --- BB.5: cli._slate_preflight_audit unit tests --------------------------


def test_slate_preflight_audit_default_off_no_gh_no_violation():
    """Pin §D11: empty file_defaults → no auto_pr, no auto_merge,
    no gh required, no coherence violations."""
    from claude_plan_execute.cli import _slate_preflight_audit
    tasks = [{"id": 1, "title": "T", "description": "D"}]
    needs_gh, viol = _slate_preflight_audit(
        tasks, file_defaults={}, cli_overrides={},
        phases_by_id={}, task_class_models={},
    )
    assert needs_gh is False
    assert viol == []


def test_slate_preflight_audit_auto_pr_with_branch_per_task_passes():
    """Pin §D11: auto_pr=True AND branch_per_task=True → needs_gh=True,
    no coherence violation."""
    from claude_plan_execute.cli import _slate_preflight_audit
    tasks = [{"id": 1, "title": "T", "description": "D"}]
    needs_gh, viol = _slate_preflight_audit(
        tasks,
        file_defaults={"auto_pr": True, "branch_per_task": True},
        cli_overrides={},
        phases_by_id={}, task_class_models={},
    )
    assert needs_gh is True
    assert viol == []


def test_slate_preflight_audit_auto_pr_without_branch_per_task_flags():
    """Pin §D11: auto_pr=True without branch_per_task=True → needs_gh=True
    AND task is flagged in the coherence list."""
    from claude_plan_execute.cli import _slate_preflight_audit
    tasks = [{"id": 1, "title": "T", "description": "D"}]
    needs_gh, viol = _slate_preflight_audit(
        tasks,
        file_defaults={"auto_pr": True},  # branch_per_task defaults False
        cli_overrides={},
        phases_by_id={}, task_class_models={},
    )
    assert needs_gh is True
    assert viol == [1]


def test_slate_preflight_audit_per_task_override_respects_off():
    """Pin §D11: a task-level override of `auto_pr: False` defeats a
    project default of True → no gh needed for that task."""
    from claude_plan_execute.cli import _slate_preflight_audit
    tasks = [{
        "id": 1, "title": "T", "description": "D", "auto_pr": False,
    }]
    needs_gh, viol = _slate_preflight_audit(
        tasks,
        file_defaults={"auto_pr": True, "branch_per_task": True},
        cli_overrides={},
        phases_by_id={}, task_class_models={},
    )
    assert needs_gh is False
    assert viol == []


def test_slate_preflight_audit_auto_merge_alone_triggers_needs_gh():
    """Pin §D11: auto_merge=True alone (without auto_pr) still flags
    the slate as needing gh AND a coherence violation when
    branch_per_task is False."""
    from claude_plan_execute.cli import _slate_preflight_audit
    tasks = [{"id": 1, "title": "T", "description": "D"}]
    needs_gh, viol = _slate_preflight_audit(
        tasks,
        file_defaults={"auto_merge": True},
        cli_overrides={},
        phases_by_id={}, task_class_models={},
    )
    assert needs_gh is True
    assert viol == [1]


# ---------------------------------------------------------------------------
# R. --explain (task 5 v0.3 §2.20)
# ---------------------------------------------------------------------------

def test_resolve_gate_specs_extraction_preserves_execute_task_behavior(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §"Documented design choice": the helper extraction is
    behavior-preserving. A 2-layer (defaults + task) gate config
    mixing a built-in factory ('tests') and a project-registered
    shell gate produces the exact same gates/<name>.txt files as
    the pre-T5 inline dispatch."""
    _patch_run_claude_simulating_success(monkeypatch)
    _clean_default_registry.register(
        "g-task", Gate(kind="shell", command=("true",)),
    )

    def fake_run(cmd, **kwargs):
        class P: pass
        p = P(); p.returncode = 0; p.stdout = ""; p.stderr = ""
        return p
    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, gates_extra=["g-task"]), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["tests"],
            "test_command": "echo ok",
        },
        cli={}, skip_to="implement",
    )
    gates_dir = tmp_path / "task-1" / "gates"
    files = sorted(p.name for p in gates_dir.glob("*.txt"))
    assert files == ["g-task.txt", "tests.txt"]
    assert state.get(1)["status"] == "done"


def test_resolve_gate_specs_invariant_grep_expands_to_loaded_names(
    _clean_default_registry,
):
    """Pin §D7: invariant-grep expansion. One input name → many
    resolved gates."""
    from claude_plan_execute.runner import _resolve_gate_specs
    _clean_default_registry.register(
        "inv-foo", Gate(kind="shell", command=("true",)),
    )
    _clean_default_registry.register(
        "inv-bar", Gate(kind="shell", command=("true",)),
    )
    resolved, unknown, ucl, ucu = _resolve_gate_specs(
        ["invariant-grep"],
        test_command="pytest",
        lint_command=None,
        untranslated_strings_command=None,
        dev_server_url="http://x",
        persona_section="",
        invariant_names=["inv-foo", "inv-bar"],
        registry=_clean_default_registry,
    )
    assert [n for (n, _) in resolved] == ["inv-foo", "inv-bar"]
    assert unknown == []
    assert ucl is False
    assert ucu is False


def test_resolve_gate_specs_unconfigured_lint_returns_flag(
    _clean_default_registry,
):
    """Pin §D5: lint with lint_command=None returns the flag, not
    a resolved gate."""
    from claude_plan_execute.runner import _resolve_gate_specs
    resolved, unknown, ucl, ucu = _resolve_gate_specs(
        ["lint"],
        test_command="pytest",
        lint_command=None,
        untranslated_strings_command=None,
        dev_server_url="http://x",
        persona_section="",
        invariant_names=[],
        registry=_clean_default_registry,
    )
    assert resolved == []
    assert unknown == []
    assert ucl is True
    assert ucu is False


def test_explain_task_header_lines_pin_format(_clean_default_registry):
    """Pin §"Format invariants" #1-#3: Task line, Phase line, Class line.

    The phases_by_id fixture sets `title="UI polish"` (≠ phase id) to
    catch a regression where the renderer reads `title` from the
    `defaults` sub-dict instead of the outer phase dict (per
    schema.py:142-150 the title is a SIBLING of `defaults`, not a
    child)."""
    from claude_plan_execute.runner import explain_task
    text = explain_task(
        _make_task(id=17, title="feat: KB CRUD UI", phase="ui-polish",
                   task_class="ui"),
        file_defaults={"plans_dir": "plans", "model": "opus"},
        cli_overrides={},
        phases_by_id={"ui-polish": {"id": "ui-polish",
                                     "title": "UI polish",
                                     "defaults": {"task_class": "ui"}}},
        task_class_models={"ui": {"model": "sonnet", "effort": "medium"}},
        registry=_clean_default_registry,
        invariant_names=[],
    )
    lines = text.splitlines()
    assert lines[0] == "Task 17: feat: KB CRUD UI"
    assert "Phase: UI polish" in lines[1]      # title from OUTER dict
    assert "(id=ui-polish)" in lines[1]
    assert "Class: ui" in lines[2]
    assert "model=sonnet" in lines[2]
    assert "from task_class:ui" in lines[2]


def test_explain_task_gates_section_renders_run_skip_split(
    _clean_default_registry,
):
    """Pin §"Format invariants" #4: Gates header counts.
    A task_class:ui gate on a non-ui task is SKIP; an always gate
    is RUN. Counts surface that 1+1, of which 1 RUN and 1 SKIP."""
    from claude_plan_execute.runner import explain_task
    _clean_default_registry.register(
        "always-gate",
        Gate(kind="shell", command=("true",), when=("always",)),
    )
    _clean_default_registry.register(
        "ui-only",
        Gate(kind="agent", prompt_template="x", when=("task_class:ui",)),
    )
    text = explain_task(
        _make_task(id=1, task_class="schema"),
        file_defaults={"plans_dir": "plans",
                       "gates": ["always-gate", "ui-only"]},
        cli_overrides={},
        registry=_clean_default_registry,
        invariant_names=[],
    )
    assert "Gates (2 total; 1 will run, 1 will skip):" in text
    assert "always-gate" in text
    assert "always -> RUN" in text
    assert "ui-only" in text
    assert "-> SKIP" in text
    assert "task_class='schema' not in" in text


def test_explain_task_renders_services_and_branch_stubs(
    _clean_default_registry,
):
    """Pin §D8: services/branch sections are stubs in T5."""
    from claude_plan_execute.runner import explain_task
    text = explain_task(
        _make_task(id=1),
        file_defaults={"plans_dir": "plans"},
        cli_overrides={},
        registry=_clean_default_registry,
        invariant_names=[],
    )
    assert "Services required: (see T8" in text
    assert "Branch: (see T9" in text


def test_explain_task_caps_section_renders_with_provenance(
    _clean_default_registry,
):
    """Pin §"Format invariants" #7: caps each have a [from <source>]
    provenance label."""
    from claude_plan_execute.runner import explain_task
    text = explain_task(
        _make_task(id=1),
        file_defaults={"plans_dir": "plans", "implement_max_turns": 250},
        cli_overrides={},
        registry=_clean_default_registry,
        invariant_names=[],
    )
    assert "Caps:" in text
    assert "implement_max_turns:" in text  # resolver key
    assert "250" in text
    assert "from project_defaults" in text
    # max_attempts isn't set on file_defaults → comes from hardcoded
    assert "max_attempts:" in text
    assert "from hardcoded" in text


def test_explain_task_uses_tasks_v2_fixture_with_phase_inherited_class(
    _clean_default_registry,
):
    """Task description: 'Use tests/fixtures/tasks-v2.yaml; assert
    specific gates appear / do not appear given the touches/task_class
    filters.' Task 1 in the fixture has phase='foundation' which
    inherits task_class='schema'. A gate with when=task_class:schema
    must MATCH; with when=task_class:ui must SKIP."""
    from claude_plan_execute import loader as L
    from claude_plan_execute.runner import explain_task
    FIXTURES = Path(__file__).parent / "fixtures"
    tf = L.load_tasks_file(FIXTURES / "tasks-v2.yaml")
    file_defaults = L.effective_defaults(tf)
    file_defaults["gates"] = ["schema-only", "ui-only"]
    tasks = [t.model_dump(exclude_none=True) for t in tf.tasks]
    phases_by_id = {p.id: p.model_dump(exclude_none=True) for p in tf.phases}
    task_class_models = {
        name: cfg.model_dump(exclude_none=True)
        for name, cfg in (tf.defaults.task_class_models or {}).items()
    }

    _clean_default_registry.register(
        "schema-only",
        Gate(kind="shell", command=("true",), when=("task_class:schema",)),
    )
    _clean_default_registry.register(
        "ui-only",
        Gate(kind="agent", prompt_template="x", when=("task_class:ui",)),
    )

    # Task 1 — phase='foundation' → task_class inherited as 'schema'.
    text = explain_task(
        tasks[0], file_defaults, {},
        phases_by_id=phases_by_id,
        task_class_models=task_class_models,
        registry=_clean_default_registry,
        invariant_names=[],
    )
    schema_line = next(l for l in text.splitlines() if "schema-only" in l)
    ui_line = next(l for l in text.splitlines() if "ui-only" in l)
    assert "RUN" in schema_line and "SKIP" not in schema_line
    assert "SKIP" in ui_line and "RUN" not in ui_line


def test_explain_task_touches_filter_evaluated_against_key_files(
    _clean_default_registry,
):
    """Task description: '... given the touches/task_class filters.'
    A gate with when=touches:**/auth/** must MATCH a task whose
    key_files include 'src/auth/api.py' and SKIP a task whose
    key_files don't."""
    from claude_plan_execute.runner import explain_task
    _clean_default_registry.register(
        "auth-touched",
        Gate(kind="shell", command=("true",), when=("touches:**/auth/**",)),
    )

    matched_text = explain_task(
        _make_task(id=1, key_files=["src/auth/api.py"]),
        file_defaults={"plans_dir": "plans", "gates": ["auth-touched"]},
        cli_overrides={},
        registry=_clean_default_registry,
        invariant_names=[],
    )
    assert "auth-touched" in matched_text
    matched_line = next(l for l in matched_text.splitlines()
                        if "auth-touched" in l)
    assert "RUN" in matched_line
    assert "src/auth/api.py" in matched_text

    skipped_text = explain_task(
        _make_task(id=2, key_files=["docs/readme.md"]),
        file_defaults={"plans_dir": "plans", "gates": ["auth-touched"]},
        cli_overrides={},
        registry=_clean_default_registry,
        invariant_names=[],
    )
    skipped_line = next(l for l in skipped_text.splitlines()
                        if "auth-touched" in l)
    assert "SKIP" in skipped_line
    assert "key_files=['docs/readme.md']" in skipped_text


def test_explain_task_when_eval_uses_phase_inherited_task_class(
    _clean_default_registry,
):
    """Pin §D9: a phase-inherited task_class must drive `when`
    evaluation. Without `task_for_when = {**task, 'task_class':
    task_class_name}`, a task whose class is set on the PHASE (not
    the task) gets evaluated with task_class=None and the
    task_class:ui filter falsely skips."""
    from claude_plan_execute.runner import explain_task
    _clean_default_registry.register(
        "ui-gate",
        Gate(kind="agent", prompt_template="x", when=("task_class:ui",)),
    )
    text = explain_task(
        _make_task(id=1, phase="ui-shell"),
        file_defaults={"plans_dir": "plans", "gates": ["ui-gate"]},
        cli_overrides={},
        phases_by_id={"ui-shell": {"id": "ui-shell",
                                    "defaults": {"task_class": "ui"}}},
        task_class_models={"ui": {"model": "sonnet"}},
        registry=_clean_default_registry,
        invariant_names=[],
    )
    ui_line = next(l for l in text.splitlines() if "ui-gate" in l)
    assert "RUN" in ui_line
    assert "task_class='ui'" in ui_line


def test_explain_task_duplicate_gate_name_across_layers_attributed_per_layer(
    monkeypatch, _clean_default_registry,
):
    """Pin §D11: when the same gate name appears in two layers
    (defaults + task), the runner runs it twice and the explainer
    must show TWO rows with distinct source labels."""
    from claude_plan_execute.runner import explain_task
    text = explain_task(
        _make_task(id=1, gates_extra=["lint"]),
        file_defaults={
            "plans_dir": "plans",
            "gates": ["lint"],
            "lint_command": "ruff check",
        },
        cli_overrides={},
        registry=_clean_default_registry,
        invariant_names=[],
    )
    lint_lines = [l for l in text.splitlines()
                  if "lint" in l and "[shell]" in l]
    assert len(lint_lines) == 2
    assert any("from defaults" in l for l in lint_lines)
    assert any("from task" in l for l in lint_lines)


# ---------------------------------------------------------------------------
# T10 — runner failure-class + block-reason differentiation (§D19)
# ---------------------------------------------------------------------------
#
# Three pins for the post-Phase-4 blocking-gate loop:
#   - human_gate_timeout flips both block_reason AND failure_class.
#   - the dashboard-visible block_reason field flips (separate pin
#     because §D19 emphasizes the smoke-test-#3 dependency).
#   - non-human blocking gates still get gate_failed (regression).

def test_runner_classifies_human_gate_timeout_as_distinct_failure_class(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """§D19 pin (failure_class): a HumanGate that times out →
    record_failure called with failure_class='human_gate_timeout'.
    """
    _patch_run_claude_simulating_success(monkeypatch)

    # Fast-mode: monkey-patch HumanGate so its run() returns a
    # synthetic timeout result without polling.
    from claude_plan_execute.gates import GateResult
    import claude_plan_execute.gates.human as human_mod

    def fake_run(self):
        return GateResult(
            name=self.name, passed=False, output="",
            output_path=None, duration_seconds=0.0,
            on_failure=self.gate.on_failure,
            error="human_gate_timeout",
        )
    monkeypatch.setattr(human_mod.HumanGate, "run", fake_run)

    _clean_default_registry.register(
        "design-review",
        Gate(
            kind="human", prompt_template="please review",
            parallel_safe=False, on_failure="block",
        ),
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["design-review"],
        },
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert len(entry["attempts"]) == 1
    rec = entry["attempts"][0]
    assert rec["failure_class"] == "human_gate_timeout"


def test_runner_sets_block_reason_to_human_gate_timeout_on_human_gate_timeout(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """§D19 pin (block_reason): the dashboard-visible block_reason
    field flips to 'human_gate_timeout' alongside failure_class.
    """
    _patch_run_claude_simulating_success(monkeypatch)

    from claude_plan_execute.gates import GateResult
    import claude_plan_execute.gates.human as human_mod

    def fake_run(self):
        return GateResult(
            name=self.name, passed=False, output="",
            output_path=None, duration_seconds=0.0,
            on_failure=self.gate.on_failure,
            error="human_gate_timeout",
        )
    monkeypatch.setattr(human_mod.HumanGate, "run", fake_run)

    _clean_default_registry.register(
        "design-review",
        Gate(
            kind="human", prompt_template="please review",
            parallel_safe=False, on_failure="block",
        ),
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "gates": ["design-review"],
        },
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry["block_reason"] == "human_gate_timeout"


def test_runner_keeps_block_reason_gate_failed_for_non_timeout_blocking_gates(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """§D19 regression: shell-gate failures still get
    block_reason='gate_failed' and failure_class='gate_failed'.
    """
    _patch_run_claude_simulating_success(monkeypatch)
    _clean_default_registry.register(
        "bad-gate",
        Gate(kind="shell", command=("false",), on_failure="block"),
    )

    def fake_run(cmd, **kwargs):
        class P:
            pass
        p = P()
        p.returncode = 1
        p.stdout = ""
        p.stderr = "regular gate failed\n"
        return p
    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", fake_run,
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path), "gates": ["bad-gate"]},
        cli={}, skip_to="implement",
    )
    entry = state.get(1)
    assert entry["status"] == "blocked"
    assert entry["block_reason"] == "gate_failed"
    rec = entry["attempts"][0]
    assert rec["failure_class"] == "gate_failed"


# ===========================================================================
# CC. Memory passes (task 15 v0.3 §"cross-task-memory")
# ===========================================================================

import os
import yaml
from claude_plan_execute import memory as memory_mod
from claude_plan_execute.cli import _resolve_slate_id


def _write_memory_file(
    memory_dir: Path, name: str, *,
    type: str = "feedback",
    lifecycle: str = "evergreen",
    citation_count: int = 0,
    description: str = "test memory",
) -> Path:
    memory_dir.mkdir(parents=True, exist_ok=True)
    fm = {
        "name": name, "description": description, "type": type,
        "lifecycle": lifecycle, "citation_count": citation_count,
    }
    body = "Body line 1.\nBody line 2.\n"
    text = "---\n" + yaml.safe_dump(fm, sort_keys=False) + "---\n\n" + body
    p = memory_dir / f"{name}.md"
    p.write_text(text, encoding="utf-8")
    return p


def _read_citation_count_from_file(p: Path) -> int:
    fm, _ = memory_mod.parse_memory_frontmatter(p.read_text())
    return int(fm.get("citation_count", 0))


def _stub_memory_dir(monkeypatch, tmp_path) -> Path:
    """Patch memory_dir_for_cwd in runner so execute_task uses tmp_path."""
    fake_dir = tmp_path / "memdir"
    monkeypatch.setattr(
        "claude_plan_execute.runner.memory_dir_for_cwd",
        lambda cwd: fake_dir,
    )
    return fake_dir


# CC.1 — citation pass


def test_citation_pass_increments_for_single_reference(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    mem_path = _write_memory_file(mem_dir, "foo_lesson", citation_count=2)

    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    (task_dir / "plan.md").write_text(
        "Step 1: do thing [MEM: foo_lesson].\n"
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )

    assert _read_citation_count_from_file(mem_path) == 3


def test_citation_pass_counts_multiple_references_in_one_plan(
    monkeypatch, tmp_path,
):
    _no_run_claude_called(monkeypatch)
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    mem_path = _write_memory_file(mem_dir, "foo_lesson", citation_count=0)

    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    (task_dir / "plan.md").write_text(
        "Step 1: [MEM: foo_lesson]\n"
        "Step 2: see also [MEM: foo_lesson]\n"
        "Step 3: revisit [MEM: foo_lesson]\n"
    )

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )

    assert _read_citation_count_from_file(mem_path) == 3


def test_citation_pass_silently_ignores_unknown_memory_names(
    monkeypatch, tmp_path,
):
    """Pin: hallucinated [MEM: ghost] must NOT raise. The reviewer
    catches it earlier; the runner doesn't double-fault."""
    _no_run_claude_called(monkeypatch)
    _stub_memory_dir(monkeypatch, tmp_path)

    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    (task_dir / "plan.md").write_text("[MEM: ghost]")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert state.get(1)["status"] == "done"


def test_citation_pass_is_idempotent_on_already_done_task(
    monkeypatch, tmp_path,
):
    """Re-running a task whose status is already 'done' must NOT
    increment citation counts a second time."""
    _no_run_claude_called(monkeypatch)
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    mem_path = _write_memory_file(
        mem_dir, "foo_lesson", citation_count=0,
    )

    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    (task_dir / "plan.md").write_text("[MEM: foo_lesson]")

    state = State(tmp_path / "state.json")
    state.set_status(1, "done")  # pre-mark
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    # No increment — task was already done.
    assert _read_citation_count_from_file(mem_path) == 0


def test_citation_pass_skipped_when_plan_md_missing(monkeypatch, tmp_path):
    """No plan.md (e.g., dry-run never wrote one) → no-op, no error."""
    _no_run_claude_called(monkeypatch)
    _stub_memory_dir(monkeypatch, tmp_path)
    state = State(tmp_path / "state.json")
    # Must not raise.
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )


# CC.2 — lifecycle prune


def test_lifecycle_prune_deletes_task_scoped_when_task_done(
    monkeypatch, tmp_path,
):
    """A memory with `lifecycle: task:5` is deleted when task 5
    transitions to done."""
    _no_run_claude_called(monkeypatch)
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    target = _write_memory_file(
        mem_dir, "lesson_for_t5", lifecycle="task:5",
    )

    task_dir = tmp_path / "task-5"
    task_dir.mkdir()
    (task_dir / "plan.md").write_text("plan")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=5), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert not target.exists()


def test_lifecycle_prune_keeps_task_scoped_when_task_not_done(
    monkeypatch, tmp_path,
):
    _no_run_claude_called(monkeypatch)
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    target = _write_memory_file(
        mem_dir, "lesson_for_t99", lifecycle="task:99",
    )

    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    (task_dir / "plan.md").write_text("plan")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert target.exists()


def test_lifecycle_prune_evergreen_never_deleted(monkeypatch, tmp_path):
    _no_run_claude_called(monkeypatch)
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    target = _write_memory_file(mem_dir, "everlasting", lifecycle="evergreen")

    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    (task_dir / "plan.md").write_text("plan")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
        all_task_ids=[1],   # slate-complete after this task
    )
    assert target.exists()


def test_lifecycle_prune_slate_scoped_when_slate_complete(
    monkeypatch, tmp_path,
):
    """A `lifecycle: slate:foo` memory is deleted when the env-var
    slate id matches AND every task in all_task_ids is done."""
    _no_run_claude_called(monkeypatch)
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_PLAN_EXECUTE_SLATE_ID", "foo")
    target = _write_memory_file(
        mem_dir, "slate_lesson", lifecycle="slate:foo",
    )

    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    (task_dir / "plan.md").write_text("plan")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
        all_task_ids=[1],   # slate-complete after this task
    )
    assert not target.exists()


def test_lifecycle_prune_slate_scoped_kept_when_other_tasks_pending(
    monkeypatch, tmp_path,
):
    """all_task_ids contains tasks 1 + 2; only task 1 runs in this
    test → slate is incomplete → slate-scoped memory survives."""
    _no_run_claude_called(monkeypatch)
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_PLAN_EXECUTE_SLATE_ID", "foo")
    target = _write_memory_file(
        mem_dir, "slate_lesson", lifecycle="slate:foo",
    )

    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    (task_dir / "plan.md").write_text("plan")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
        all_task_ids=[1, 2],   # task 2 still pending
    )
    assert target.exists()


def test_lifecycle_prune_slate_scoped_other_slate_id_never_pruned(
    monkeypatch, tmp_path,
):
    """A `lifecycle: slate:other` memory is NEVER pruned by a
    different slate's run, even when slate_complete=True for `foo`."""
    _no_run_claude_called(monkeypatch)
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_PLAN_EXECUTE_SLATE_ID", "foo")
    other = _write_memory_file(
        mem_dir, "x_lesson", lifecycle="slate:other",
    )

    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    (task_dir / "plan.md").write_text("plan")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
        all_task_ids=[1],
    )
    assert other.exists()


def test_lifecycle_prune_removes_memory_md_index_line(
    monkeypatch, tmp_path,
):
    """When a memory file is deleted, its line in MEMORY.md must
    also be removed."""
    _no_run_claude_called(monkeypatch)
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _write_memory_file(mem_dir, "lesson_for_t5", lifecycle="task:5")
    _write_memory_file(mem_dir, "keeper", lifecycle="evergreen")
    (mem_dir / "MEMORY.md").write_text(
        "- [Drop me](lesson_for_t5.md) — should be removed\n"
        "- [Keep me](keeper.md) — must survive\n"
    )

    task_dir = tmp_path / "task-5"
    task_dir.mkdir()
    (task_dir / "plan.md").write_text("plan")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=5), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )

    text = (mem_dir / "MEMORY.md").read_text()
    assert "lesson_for_t5.md" not in text
    assert "keeper.md" in text


def test_runner_passes_through_when_memory_dir_missing(
    monkeypatch, tmp_path,
):
    """No memory dir on disk (typical first run on a fresh project) →
    citation pass and prune pass are silent no-ops; task completes."""
    _no_run_claude_called(monkeypatch)
    fake_dir = tmp_path / "definitely_missing_memdir"
    monkeypatch.setattr(
        "claude_plan_execute.runner.memory_dir_for_cwd",
        lambda cwd: fake_dir,
    )

    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    (task_dir / "plan.md").write_text("[MEM: ghost]")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert state.get(1)["status"] == "done"


def test_runner_does_not_run_slate_prune_when_all_task_ids_is_None(
    monkeypatch, tmp_path,
):
    """The default kwarg path (all_task_ids=None, used by every
    existing test) must NEVER prune slate-scoped memories."""
    _no_run_claude_called(monkeypatch)
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_PLAN_EXECUTE_SLATE_ID", "foo")
    target = _write_memory_file(
        mem_dir, "slate_lesson", lifecycle="slate:foo",
    )

    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    (task_dir / "plan.md").write_text("plan")

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
        # NOT passing all_task_ids — slate detection should
        # short-circuit.
    )
    assert target.exists()


# CC.3 — slate-id resolution


def test_resolve_slate_id_caches_in_state(tmp_path, monkeypatch):
    state = State(tmp_path / "state.json")
    tasks_path = tmp_path / "tasks.yaml"
    tasks_path.write_text("tasks: []")

    # Force the no-git fallback by patching cli.subprocess.run.
    def fake_run(cmd, **kw):
        return type("P", (), {
            "returncode": 1, "stdout": "", "stderr": "",
        })()
    monkeypatch.setattr("claude_plan_execute.cli.subprocess.run", fake_run)

    first = _resolve_slate_id(state, tasks_path)
    assert first.startswith("mtime-")

    # Sticky on second call — the patched subprocess returns 1, so
    # without caching we'd get a fresh mtime; equality proves cache
    # was used.
    second = _resolve_slate_id(state, tasks_path)
    assert second == first
    assert state.data["slate_id"] == first


def test_resolve_slate_id_uses_branch_name_when_available(
    tmp_path, monkeypatch,
):
    state = State(tmp_path / "state.json")
    tasks_path = tmp_path / "tasks.yaml"
    tasks_path.write_text("tasks: []")

    def fake_run(cmd, **kw):
        if cmd[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return type("P", (), {
                "returncode": 0, "stdout": "feat/foo\n", "stderr": "",
            })()
        return type("P", (), {
            "returncode": 0, "stdout": "", "stderr": "",
        })()

    monkeypatch.setattr("claude_plan_execute.cli.subprocess.run", fake_run)
    sid = _resolve_slate_id(state, tasks_path)
    assert sid == "feat/foo"


def test_resolve_slate_id_skips_task_branches(tmp_path, monkeypatch):
    """If the current branch is `task/5-foo` (per-task branch from
    T9), we fall through to mtime — otherwise slate id would change
    per task."""
    state = State(tmp_path / "state.json")
    tasks_path = tmp_path / "tasks.yaml"
    tasks_path.write_text("tasks: []")

    def fake_run(cmd, **kw):
        return type("P", (), {
            "returncode": 0, "stdout": "task/5-foo\n", "stderr": "",
        })()

    monkeypatch.setattr("claude_plan_execute.cli.subprocess.run", fake_run)
    sid = _resolve_slate_id(state, tasks_path)
    assert sid.startswith("mtime-")


def test_resolve_slate_id_skips_detached_HEAD(tmp_path, monkeypatch):
    """git rev-parse --abbrev-ref HEAD on a detached HEAD prints
    'HEAD' verbatim; treat as no-branch and fall through."""
    state = State(tmp_path / "state.json")
    tasks_path = tmp_path / "tasks.yaml"
    tasks_path.write_text("tasks: []")

    def fake_run(cmd, **kw):
        return type("P", (), {
            "returncode": 0, "stdout": "HEAD\n", "stderr": "",
        })()

    monkeypatch.setattr("claude_plan_execute.cli.subprocess.run", fake_run)
    sid = _resolve_slate_id(state, tasks_path)
    assert sid.startswith("mtime-")


# ---------------------------------------------------------------------------
# DD. memory-curate threshold + post-process (T16)
# ---------------------------------------------------------------------------


def _seed_memory_files(memory_dir: Path, n: int) -> None:
    """Write `n` per-memory files with valid frontmatter under
    memory_dir, plus a MEMORY.md indexing them."""
    memory_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        name = f"m{i}"
        (memory_dir / f"{name}.md").write_text(
            "---\n"
            f"name: {name}\n"
            "description: x\n"
            "type: project\n"
            "lifecycle: evergreen\n"
            "citation_count: 0\n"
            "---\n\nbody\n",
            encoding="utf-8",
        )
        lines.append(f"- [{name}]({name}.md) — desc\n")
    (memory_dir / "MEMORY.md").write_text("".join(lines), encoding="utf-8")


def _stub_curate_agent(monkeypatch, agent_text: str) -> None:
    """Stub run_claude inside gates.agent so memory-curate returns
    `agent_text` verbatim. Combined with the runner-level
    _patch_run_claude_simulating_success stub for non-gate phases."""
    def stub(prompt, **kwargs):
        return 0, agent_text
    monkeypatch.setattr(
        "claude_plan_execute.gates.agent.run_claude", stub,
    )


def test_threshold_under_does_not_set_pending_flag(
    monkeypatch, tmp_path, _clean_default_registry,
):
    _patch_run_claude_simulating_success(monkeypatch)
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _seed_memory_files(mem_dir, 29)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert not state.data.get("pending_memory_curate")


def test_threshold_over_sets_pending_flag(
    monkeypatch, tmp_path, _clean_default_registry,
):
    _patch_run_claude_simulating_success(monkeypatch)
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _seed_memory_files(mem_dir, 31)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert state.data.get("pending_memory_curate") is True


def test_threshold_resolver_aware_per_task(
    monkeypatch, tmp_path, _clean_default_registry,
):
    _patch_run_claude_simulating_success(monkeypatch)
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _seed_memory_files(mem_dir, 28)

    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1, memory_curation_threshold=25), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert state.data.get("pending_memory_curate") is True


def test_threshold_resolver_aware_per_phase(
    monkeypatch, tmp_path, _clean_default_registry,
):
    _patch_run_claude_simulating_success(monkeypatch)
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _seed_memory_files(mem_dir, 35)

    state = State(tmp_path / "state.json")
    phases_by_id = {
        "p1": {"id": "p1", "defaults": {"memory_curation_threshold": 50}},
    }
    execute_task(
        _make_task(id=1, phase="p1"), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
        phases_by_id=phases_by_id,
    )
    assert not state.data.get("pending_memory_curate")


def test_pending_curate_flag_appends_memory_curate_to_gates(
    monkeypatch, tmp_path, _clean_default_registry,
):
    _patch_run_claude_simulating_success(monkeypatch)
    _stub_curate_agent(monkeypatch, "VERDICT: PASS\n")
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _seed_memory_files(mem_dir, 1)

    _clean_default_registry.register(
        "memory-curate",
        Gate(kind="agent", prompt_template="body",
             parallel_safe=False, on_failure="warn"),
    )

    state = State(tmp_path / "state.json")
    state.data["pending_memory_curate"] = True
    state.save()

    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert (tmp_path / "task-1" / "gates" / "memory-curate.txt").is_file()


def test_pending_curate_flag_cleared_after_gate_runs(
    monkeypatch, tmp_path, _clean_default_registry,
):
    _patch_run_claude_simulating_success(monkeypatch)
    _stub_curate_agent(monkeypatch, "VERDICT: PASS\n")
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _seed_memory_files(mem_dir, 1)
    _clean_default_registry.register(
        "memory-curate",
        Gate(kind="agent", prompt_template="body",
             parallel_safe=False, on_failure="warn"),
    )

    state = State(tmp_path / "state.json")
    state.data["pending_memory_curate"] = True
    state.save()
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert not state.data.get("pending_memory_curate")


def _curate_agent_text(findings):
    return (
        "VERDICT: PASS\n\n"
        "<!-- BEGIN_FINDINGS_JSON -->\n"
        f"{json.dumps(findings)}\n"
        "<!-- END_FINDINGS_JSON -->\n"
    )


def test_apply_findings_drops_memory_marked_drop(
    monkeypatch, tmp_path, _clean_default_registry,
):
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _seed_memory_files(mem_dir, 1)
    # Add foo memory + index it.
    (mem_dir / "foo.md").write_text(
        "---\nname: foo\nlifecycle: evergreen\ncitation_count: 0\n"
        "---\n\nbody\n", encoding="utf-8",
    )
    idx = mem_dir / "MEMORY.md"
    idx.write_text(idx.read_text() + "- [foo](foo.md) — desc\n")

    _patch_run_claude_simulating_success(monkeypatch)
    _stub_curate_agent(monkeypatch, _curate_agent_text(
        [{"memory": "foo", "verdict": "drop", "reason": "stale"}],
    ))
    _clean_default_registry.register(
        "memory-curate",
        Gate(kind="agent", prompt_template="body",
             parallel_safe=False, on_failure="warn"),
    )

    state = State(tmp_path / "state.json")
    state.data["pending_memory_curate"] = True
    state.save()
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert not (mem_dir / "foo.md").exists()
    assert "(foo.md)" not in (mem_dir / "MEMORY.md").read_text()


def test_apply_findings_keeps_memory_marked_keep(
    monkeypatch, tmp_path, _clean_default_registry,
):
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _seed_memory_files(mem_dir, 1)
    (mem_dir / "keepme.md").write_text(
        "---\nname: keepme\nlifecycle: evergreen\ncitation_count: 0\n"
        "---\n\nbody\n", encoding="utf-8",
    )
    _patch_run_claude_simulating_success(monkeypatch)
    _stub_curate_agent(monkeypatch, _curate_agent_text(
        [{"memory": "keepme", "verdict": "keep", "reason": "ok"}],
    ))
    _clean_default_registry.register(
        "memory-curate",
        Gate(kind="agent", prompt_template="body",
             parallel_safe=False, on_failure="warn"),
    )
    state = State(tmp_path / "state.json")
    state.data["pending_memory_curate"] = True
    state.save()
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert (mem_dir / "keepme.md").exists()


def test_apply_findings_records_merge_into_to_suggestions_file(
    monkeypatch, tmp_path, _clean_default_registry,
):
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _seed_memory_files(mem_dir, 0)
    for n in ("foo", "bar"):
        (mem_dir / f"{n}.md").write_text(
            f"---\nname: {n}\nlifecycle: evergreen\ncitation_count: 0\n"
            "---\n\nbody\n", encoding="utf-8",
        )
    _patch_run_claude_simulating_success(monkeypatch)
    _stub_curate_agent(monkeypatch, _curate_agent_text(
        [{"memory": "foo", "verdict": "merge_into:bar", "reason": "dupe"}],
    ))
    _clean_default_registry.register(
        "memory-curate",
        Gate(kind="agent", prompt_template="body",
             parallel_safe=False, on_failure="warn"),
    )
    state = State(tmp_path / "state.json")
    state.data["pending_memory_curate"] = True
    state.save()
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    suggestions_path = tmp_path / "memory-merge-suggestions.json"
    assert suggestions_path.is_file()
    data = json.loads(suggestions_path.read_text())
    assert len(data) == 1
    assert data[0]["from"] == "foo"
    assert data[0]["into"] == "bar"
    # Both memories survive on disk.
    assert (mem_dir / "foo.md").exists()
    assert (mem_dir / "bar.md").exists()


def test_apply_findings_skips_merge_into_with_missing_target(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §I.3(c): merge_into target absent → no suggestion added,
    warning logged, task completes."""
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _seed_memory_files(mem_dir, 0)
    (mem_dir / "src.md").write_text(
        "---\nname: src\nlifecycle: evergreen\ncitation_count: 0\n"
        "---\n\nbody\n", encoding="utf-8",
    )
    # Note: no target.md — verdict references a missing target.
    _patch_run_claude_simulating_success(monkeypatch)
    _stub_curate_agent(monkeypatch, _curate_agent_text(
        [{"memory": "src", "verdict": "merge_into:nope", "reason": "x"}],
    ))
    _clean_default_registry.register(
        "memory-curate",
        Gate(kind="agent", prompt_template="body",
             parallel_safe=False, on_failure="warn"),
    )
    state = State(tmp_path / "state.json")
    state.data["pending_memory_curate"] = True
    state.save()
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    suggestions_path = tmp_path / "memory-merge-suggestions.json"
    assert not suggestions_path.exists()
    # Source file untouched.
    assert (mem_dir / "src.md").exists()


def test_apply_findings_drop_silently_no_op_on_missing_file(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §I.3(c): drop verdict for an already-missing file is a
    silent no-op (the lifecycle prune may have eaten it earlier the
    same task). No exception, task completes."""
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _seed_memory_files(mem_dir, 1)
    # Note: no ghost.md — drop targets a name that never existed.
    _patch_run_claude_simulating_success(monkeypatch)
    _stub_curate_agent(monkeypatch, _curate_agent_text(
        [{"memory": "ghost", "verdict": "drop", "reason": "stale"}],
    ))
    _clean_default_registry.register(
        "memory-curate",
        Gate(kind="agent", prompt_template="body",
             parallel_safe=False, on_failure="warn"),
    )
    state = State(tmp_path / "state.json")
    state.data["pending_memory_curate"] = True
    state.save()
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert state.get(1)["status"] == "done"


def test_curate_gate_failure_does_not_block_task(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin: on_failure=warn means an absent VERDICT does NOT block."""
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _seed_memory_files(mem_dir, 1)
    _patch_run_claude_simulating_success(monkeypatch)
    _stub_curate_agent(monkeypatch, "no verdict here\n")
    _clean_default_registry.register(
        "memory-curate",
        Gate(kind="agent", prompt_template="body",
             parallel_safe=False, on_failure="warn"),
    )
    state = State(tmp_path / "state.json")
    state.data["pending_memory_curate"] = True
    state.save()
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert state.get(1)["status"] == "done"
    # Flag still cleared (gate ran).
    assert not state.data.get("pending_memory_curate")


def test_malformed_findings_json_skips_application(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin: malformed JSON between markers → findings=None → no
    memories dropped, task still done."""
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _seed_memory_files(mem_dir, 1)
    (mem_dir / "x.md").write_text(
        "---\nname: x\nlifecycle: evergreen\ncitation_count: 0\n"
        "---\n\nbody\n", encoding="utf-8",
    )
    _patch_run_claude_simulating_success(monkeypatch)
    _stub_curate_agent(monkeypatch,
        "VERDICT: PASS\n\n"
        "<!-- BEGIN_FINDINGS_JSON -->\n"
        "this is not json {{{\n"
        "<!-- END_FINDINGS_JSON -->\n"
    )
    _clean_default_registry.register(
        "memory-curate",
        Gate(kind="agent", prompt_template="body",
             parallel_safe=False, on_failure="warn"),
    )
    state = State(tmp_path / "state.json")
    state.data["pending_memory_curate"] = True
    state.save()
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert state.get(1)["status"] == "done"
    # x.md survives.
    assert (mem_dir / "x.md").exists()


def test_findings_none_branch_skips_application(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin: NO appendix markers → _extract_findings returns None
    → r.findings or [] → []. Task still done; flag cleared."""
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _seed_memory_files(mem_dir, 1)
    _patch_run_claude_simulating_success(monkeypatch)
    _stub_curate_agent(monkeypatch, "VERDICT: PASS\n\nNo appendix here.\n")
    _clean_default_registry.register(
        "memory-curate",
        Gate(kind="agent", prompt_template="body",
             parallel_safe=False, on_failure="warn"),
    )
    state = State(tmp_path / "state.json")
    state.data["pending_memory_curate"] = True
    state.save()
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    assert state.get(1)["status"] == "done"
    assert not state.data.get("pending_memory_curate")


def test_findings_non_dict_list_reduced_to_empty_by_extractor(
    monkeypatch, tmp_path, _clean_default_registry,
):
    """Pin §I.3(d): a non-dict list at the agent's output is
    reduced to findings=[] by _extract_findings's case-5
    fall-through. The runner's post-process step iterates zero
    elements; the per-element `not isinstance(finding, dict)`
    branch is NOT reachable through the runner."""
    mem_dir = _stub_memory_dir(monkeypatch, tmp_path)
    _seed_memory_files(mem_dir, 1)
    (mem_dir / "foo.md").write_text(
        "---\nname: foo\nlifecycle: evergreen\ncitation_count: 0\n"
        "---\n\nbody\n", encoding="utf-8",
    )
    _patch_run_claude_simulating_success(monkeypatch)
    _stub_curate_agent(monkeypatch,
        "VERDICT: PASS\n\n"
        "<!-- BEGIN_FINDINGS_JSON -->\n"
        '["drop foo", "keep bar"]\n'
        "<!-- END_FINDINGS_JSON -->\n"
    )
    _clean_default_registry.register(
        "memory-curate",
        Gate(kind="agent", prompt_template="body",
             parallel_safe=False, on_failure="warn"),
    )
    state = State(tmp_path / "state.json")
    state.data["pending_memory_curate"] = True
    state.save()
    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, skip_to="implement",
    )
    # foo survives (no verdicts applied).
    assert (mem_dir / "foo.md").exists()
    assert not state.data.get("pending_memory_curate")


# ===========================================================================
# CC. T13 — Parallel scheduler (v0.3 §2.5)
# ===========================================================================
#
# Pattern: stub execute_task with a barrier-respecting fake to drive
# concurrency assertions deterministically. threading.Event ensures
# no time.sleep dependence.


def test_per_thread_last_run_flags_no_cross_contamination(
    monkeypatch, tmp_path,
):
    """Pin §F3.5: two threads' _last_run_state.rate_limited flags
    are independent (threading.local provides per-thread namespaces).
    Without per-thread isolation, thread B's True bleeds into thread
    A's check.
    """
    from claude_plan_execute import claude as claude_module

    saw: dict[int, bool] = {}
    barrier = threading.Barrier(2)

    def worker(rate_limited: bool, key: int):
        claude_module._last_run_state.rate_limited = rate_limited
        barrier.wait(timeout=2.0)
        saw[key] = claude_module._last_run("rate_limited", False)

    t1 = threading.Thread(target=worker, args=(False, 1))
    t2 = threading.Thread(target=worker, args=(True, 2))
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert saw[1] is False
    assert saw[2] is True


from claude_plan_execute.runner import run_slate, _SlateScheduler


def _make_scheduler_inputs(tasks: list[dict], tmp_path: Path):
    """Build the kwargs run_slate needs for an end-to-end scheduler test.

    NB: `all_task_ids` MUST be present — `run_slate`'s signature has it
    as a REQUIRED keyword-only kwarg (no default) per §F4.9.
    """
    state = State(tmp_path / "state.json")
    return {
        "ordered_tasks": tasks,
        "all_task_ids": [t["id"] for t in tasks],
        "state": state,
        "file_defaults": {"plans_dir": str(tmp_path)},
        "cli_overrides": {},
        "phases_by_id": {},
        "task_class_models": {},
        "service_manager": _StubServiceManager(),
        "dry_run": False,
    }


def test_run_slate_parallelism_1_serializes_independent_tasks_in_input_order(
    monkeypatch, tmp_path,
):
    """Pin (a): parallelism=1 behaves identically to v0.2.0 — three
    independent tasks (no `depends_on` chain) run sequentially in the
    `ordered_tasks` declaration order.
    """
    order: list = []

    def stub_execute_task(task, state, *a, **kw):
        order.append(task["id"])
        state.set_status(task["id"], "done")

    monkeypatch.setattr(
        "claude_plan_execute.runner.execute_task", stub_execute_task,
    )
    tasks = [
        {"id": 1, "title": "A", "description": "D"},
        {"id": 2, "title": "B", "description": "D"},
        {"id": 3, "title": "C", "description": "D"},
    ]
    kwargs = _make_scheduler_inputs(tasks, tmp_path)
    run_slate(parallelism=1, **kwargs)
    assert order == [1, 2, 3]


def test_run_slate_forwards_skip_to_to_execute_task(
    monkeypatch, tmp_path,
):
    """Pin §F4.9 / §F5.1 (review-5 §N11): `run_slate` forwards
    `skip_to=...` to every inner `execute_task` call.
    """
    captured: list = []
    captured_lock = threading.Lock()

    def stub_execute_task(task, state, *a, **kw):
        with captured_lock:
            captured.append(kw.get("skip_to"))
        state.set_status(task["id"], "done")

    monkeypatch.setattr(
        "claude_plan_execute.runner.execute_task", stub_execute_task,
    )
    tasks = [
        {"id": 1, "title": "A", "description": "D",
         "key_files": ["src/a.py"]},
        {"id": 2, "title": "B", "description": "D",
         "key_files": ["src/b.py"]},
    ]

    kwargs = _make_scheduler_inputs(tasks, tmp_path)
    run_slate(parallelism=1, skip_to="implement", **kwargs)
    assert captured == ["implement", "implement"]

    captured.clear()
    kwargs = _make_scheduler_inputs(tasks, tmp_path / "p2")
    run_slate(parallelism=2, skip_to="implement", **kwargs)
    assert len(captured) == 2
    assert all(v == "implement" for v in captured)

    captured.clear()
    kwargs = _make_scheduler_inputs(tasks, tmp_path / "p3")
    run_slate(parallelism=1, **kwargs)
    assert captured == [None, None]


def test_serial_blocked_parent_blocks_child(monkeypatch, tmp_path):
    """Pin §B1 + §B4: T13's eligibility predicate diverges from
    v0.2.0 on the failed-dep path — a blocked parent now prevents
    children from running, AND the blocked parent itself is NOT
    re-attempted in the same run.
    """
    state = State(tmp_path / "state.json")
    state.set_status(1, "blocked", block_reason="any")
    ran: list = []
    ran_lock = threading.Lock()

    def stub_execute_task(task, *a, **kw):
        with ran_lock:
            ran.append(task["id"])

    monkeypatch.setattr(
        "claude_plan_execute.runner.execute_task", stub_execute_task,
    )
    tasks = [
        {"id": 1, "title": "A", "description": "D"},
        {"id": 2, "title": "B", "description": "D", "depends_on": [1]},
    ]
    kwargs = {
        "ordered_tasks": tasks,
        "all_task_ids": [t["id"] for t in tasks],
        "state": state,
        "file_defaults": {"plans_dir": str(tmp_path)},
        "cli_overrides": {},
        "phases_by_id": {},
        "task_class_models": {},
        "service_manager": _StubServiceManager(),
        "dry_run": False,
    }
    runner_thread = threading.Thread(
        target=run_slate, kwargs={"parallelism": 1, **kwargs},
    )
    runner_thread.start()
    runner_thread.join(timeout=2.0)
    assert not runner_thread.is_alive(), (
        "run_slate did not exit within 2.0s — likely a re-claim loop "
        "on the blocked task. See plan §B1 / Option A."
    )
    assert ran == []
    assert state.get(2)["status"] == "pending"
    assert state.get(1)["status"] == "blocked"


def test_run_slate_parallelism_2_runs_two_tasks_concurrently(
    monkeypatch, tmp_path,
):
    """Pin (b): parallelism=2 runs two independent tasks simultaneously."""
    started = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]

    def stub_execute_task(task, state, *a, **kw):
        idx = task["id"] - 1
        started[idx].set()
        release[idx].wait(timeout=5.0)
        state.set_status(task["id"], "done")

    monkeypatch.setattr(
        "claude_plan_execute.runner.execute_task", stub_execute_task,
    )
    tasks = [
        {"id": 1, "title": "A", "description": "D",
         "key_files": ["src/a.py"]},
        {"id": 2, "title": "B", "description": "D",
         "key_files": ["src/b.py"]},
    ]
    kwargs = _make_scheduler_inputs(tasks, tmp_path)

    runner_thread = threading.Thread(
        target=run_slate,
        kwargs={"parallelism": 2, **kwargs},
    )
    runner_thread.start()

    assert started[0].wait(timeout=2.0)
    assert started[1].wait(timeout=2.0)
    release[0].set()
    release[1].set()
    runner_thread.join(timeout=5.0)
    assert not runner_thread.is_alive()


def test_eligibility_blocks_third_task_when_two_share_key_files(
    monkeypatch, tmp_path,
):
    """Pin (c): worker-2 must NOT pick up task-3 while task-1 (shared
    key_files with task-3) is in flight.
    """
    started = {tid: threading.Event() for tid in (1, 2, 3)}
    release = {tid: threading.Event() for tid in (1, 2, 3)}

    def stub_execute_task(task, state, *a, **kw):
        started[task["id"]].set()
        release[task["id"]].wait(timeout=5.0)
        state.set_status(task["id"], "done")

    monkeypatch.setattr(
        "claude_plan_execute.runner.execute_task", stub_execute_task,
    )
    tasks = [
        {"id": 1, "title": "A", "description": "D",
         "key_files": ["src/auth.py"]},
        {"id": 2, "title": "B", "description": "D",
         "key_files": ["src/ui.py"]},
        {"id": 3, "title": "C", "description": "D",
         "key_files": ["src/auth.py"]},
    ]
    kwargs = _make_scheduler_inputs(tasks, tmp_path)
    runner_thread = threading.Thread(
        target=run_slate,
        kwargs={"parallelism": 2, **kwargs},
    )
    runner_thread.start()

    assert started[1].wait(timeout=2.0)
    assert started[2].wait(timeout=2.0)
    assert not started[3].is_set()
    release[1].set()
    assert started[3].wait(timeout=2.0)
    release[2].set(); release[3].set()
    runner_thread.join(timeout=5.0)


def test_eligibility_respects_depends_on(monkeypatch, tmp_path):
    """A task's depends_on chain blocks scheduling until parents are done."""
    order: list[int] = []
    order_lock = threading.Lock()

    def stub_execute_task(task, state, *a, **kw):
        with order_lock:
            order.append(task["id"])
        state.set_status(task["id"], "done")

    monkeypatch.setattr(
        "claude_plan_execute.runner.execute_task", stub_execute_task,
    )
    tasks = [
        {"id": 1, "title": "A", "description": "D"},
        {"id": 2, "title": "B", "description": "D", "depends_on": [1]},
        {"id": 3, "title": "C", "description": "D", "depends_on": [2]},
    ]
    kwargs = _make_scheduler_inputs(tasks, tmp_path)
    run_slate(parallelism=4, **kwargs)
    assert order == [1, 2, 3]


def test_failure_in_one_worker_does_not_kill_others(
    monkeypatch, tmp_path,
):
    """Pin (e) + §B1 regression guard: a raised exception in
    execute_task is caught by the worker; the slate continues.
    """
    started_2 = threading.Event()
    release_2 = threading.Event()

    def stub_execute_task(task, state, *a, **kw):
        if task["id"] == 1:
            raise RuntimeError("boom")
        started_2.set()
        release_2.wait(timeout=5.0)
        state.set_status(task["id"], "done")

    monkeypatch.setattr(
        "claude_plan_execute.runner.execute_task", stub_execute_task,
    )
    tasks = [
        {"id": 1, "title": "A", "description": "D",
         "key_files": ["src/a.py"]},
        {"id": 2, "title": "B", "description": "D",
         "key_files": ["src/b.py"]},
    ]
    kwargs = _make_scheduler_inputs(tasks, tmp_path)

    runner_thread = threading.Thread(
        target=run_slate,
        kwargs={"parallelism": 2, **kwargs},
    )
    runner_thread.start()
    assert started_2.wait(timeout=2.0)
    release_2.set()
    runner_thread.join(timeout=5.0)
    assert not runner_thread.is_alive()
    state = kwargs["state"]
    assert state.get(1)["status"] == "blocked"
    assert state.get(1)["attempts"][0]["failure_class"] == "worker_exception"
    assert len(state.get(1)["attempts"]) == 1


def test_state_atomic_under_parallel_workers(monkeypatch, tmp_path):
    """Pin (f): state.json updates are atomic under contention."""
    def stub_execute_task(task, state, *a, **kw):
        for phase in ("planning", "implementing", "done"):
            state.set_status(task["id"], phase)

    monkeypatch.setattr(
        "claude_plan_execute.runner.execute_task", stub_execute_task,
    )
    tasks = [
        {"id": i, "title": f"T{i}", "description": "D",
         "key_files": [f"src/f{i}.py"]}
        for i in range(1, 21)
    ]
    kwargs = _make_scheduler_inputs(tasks, tmp_path)
    run_slate(parallelism=4, **kwargs)
    state = kwargs["state"]
    for i in range(1, 21):
        assert state.get(i)["status"] == "done"
    on_disk = json.loads((tmp_path / "state.json").read_text())
    assert len(on_disk["tasks"]) == 20


def test_run_slate_parallelism_1_preserves_phase_teardown(
    monkeypatch, tmp_path,
):
    """Pin §D7.1: parallelism=1 preserves the v0.2.0 phase-boundary
    tear-down (calls service_manager.tear_down_for_scope('phase'))."""
    tasks = [
        {"id": 1, "title": "A", "description": "D", "phase": "p1"},
        {"id": 2, "title": "B", "description": "D", "phase": "p2",
         "depends_on": [1]},
    ]
    kwargs = _make_scheduler_inputs(tasks, tmp_path)

    def stub_with_completion(task, state, *a, **kw):
        state.set_status(task["id"], "done")

    monkeypatch.setattr(
        "claude_plan_execute.runner.execute_task", stub_with_completion,
    )
    run_slate(parallelism=1, **kwargs)
    stub = kwargs["service_manager"]
    assert "phase" in stub.tear_down_calls


def test_run_slate_parallel_path_does_not_track_phase(
    monkeypatch, tmp_path,
):
    """Pin §D7.1 / §D7: parallelism>1 does NOT call
    tear_down_for_scope('phase').
    """
    def stub(task, state, *a, **kw):
        state.set_status(task["id"], "done")
        sm = kw.get("service_manager")
        if sm is not None:
            sm.tear_down_for_scope("task")

    monkeypatch.setattr(
        "claude_plan_execute.runner.execute_task", stub,
    )
    tasks = [
        {"id": 1, "title": "A", "description": "D", "phase": "p1",
         "key_files": ["src/a.py"]},
        {"id": 2, "title": "B", "description": "D", "phase": "p2",
         "key_files": ["src/b.py"]},
    ]
    kwargs = _make_scheduler_inputs(tasks, tmp_path)
    run_slate(parallelism=2, **kwargs)
    stub_mgr = kwargs["service_manager"]
    assert "task" in stub_mgr.tear_down_calls
    assert "phase" not in stub_mgr.tear_down_calls


def test_log_prefix_appears_when_worker_id_set(
    monkeypatch, tmp_path, capsys,
):
    """Pin §D9: with worker_id=N, runner stdout lines gain '[task-N]'
    prefix. Without worker_id, no prefix (v0.2.0 behavior)."""
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")

    execute_task(
        _make_task(id=1), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
        worker_id=42,
    )
    out_with = capsys.readouterr().out
    assert "[task-42]" in out_with

    execute_task(
        _make_task(id=2), state,
        file_defaults={"plans_dir": str(tmp_path)},
        cli={}, dry_run=True,
    )
    out_without = capsys.readouterr().out
    assert "[task-" not in out_without


def test_eligibility_check_atomic_no_double_claim(monkeypatch, tmp_path):
    """Pin §D2: the eligibility-check + claim happens under one lock,
    so two workers can't both pick the same task.
    """
    call_count = 0
    call_count_lock = threading.Lock()
    proceed = threading.Event()

    def stub(task, state, *a, **kw):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
        proceed.wait(timeout=5.0)
        state.set_status(task["id"], "done")

    monkeypatch.setattr(
        "claude_plan_execute.runner.execute_task", stub,
    )
    tasks = [{"id": 1, "title": "A", "description": "D"}]
    kwargs = _make_scheduler_inputs(tasks, tmp_path)
    runner_thread = threading.Thread(
        target=run_slate,
        kwargs={"parallelism": 4, **kwargs},
    )
    runner_thread.start()
    threading.Event().wait(0.3)
    proceed.set()
    runner_thread.join(timeout=5.0)
    assert call_count == 1


def test_handle_sigint_records_manual_stop_when_awaiting_human(
    monkeypatch, tmp_path,
):
    """Pin §F4.6: `awaiting_human` is in the SIGINT-active-set."""
    state = State(tmp_path / "state.json")
    state.set_status(7, "awaiting_human")
    runner._current_state = state
    runner._set_current_task(7, "implement")
    monkeypatch.setattr(
        "claude_plan_execute.runner._kill_child", lambda: None,
    )
    with pytest.raises(SystemExit) as ei:
        handle_sigint(2, None)
    assert ei.value.code == 130
    attempts = state.get(7)["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["failure_class"] == "manual_stop"
    assert attempts[0]["phase"] == "implement"


# ---------------------------------------------------------------------------
# EE. T17 auto-split trigger integration
# ---------------------------------------------------------------------------

# NB: tests below construct `file_defaults=` inline. A shared
# `_auto_split_defaults` fixture was considered (and proposed in an
# earlier draft) but dropped: the inline form makes each test's
# precondition self-contained without an injected fixture's
# action-at-a-distance, and matches the surrounding test style in
# this file.


def _stub_run_splitter(monkeypatch):
    called = []

    def stub(**kwargs):
        called.append(kwargs)

    monkeypatch.setattr(
        "claude_plan_execute.runner.run_splitter", stub,
    )
    return called


def test_review_rejected_max_rounds_invokes_splitter_when_opted_in(
    monkeypatch, tmp_path,
):
    called = _stub_run_splitter(monkeypatch)
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "fail",
            "auto_split_on_max_rounds": True,
        },
        cli={}, skip_to="review",
        tasks_yaml_path=tmp_path / "tasks.yaml",
    )
    assert len(called) == 1


def test_implementer_max_turns_invokes_splitter_when_opted_in(
    monkeypatch, tmp_path,
):
    called = _stub_run_splitter(monkeypatch)
    _patch_run_claude_simulating_max_turns(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "auto_split_on_max_rounds": True,
        },
        cli={}, skip_to="implement",
        tasks_yaml_path=tmp_path / "tasks.yaml",
    )
    assert len(called) == 1


def test_review_rejected_max_rounds_does_NOT_invoke_splitter_by_default(
    monkeypatch, tmp_path,
):
    called = _stub_run_splitter(monkeypatch)
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "fail",
        },
        cli={}, skip_to="review",
    )
    assert called == []


def test_failure_class_outside_match_list_skips_splitter(
    monkeypatch, tmp_path,
):
    """A failure_class not in auto_split_failure_classes
    short-circuits the helper before invoking run_splitter."""
    called = _stub_run_splitter(monkeypatch)
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    execute_task(
        _make_task(id=1), state,
        file_defaults={
            "plans_dir": str(tmp_path),
            "max_review_rounds": 1,
            "on_max_review_rounds": "fail",
            "auto_split_on_max_rounds": True,
            "auto_split_failure_classes": ["implementer_max_turns"],
        },
        cli={}, skip_to="review",
        tasks_yaml_path=tmp_path / "tasks.yaml",
    )
    assert called == []


def test_maybe_run_splitter_does_not_wrap_in_try_except(
    monkeypatch, tmp_path,
):
    """Pin: `_maybe_run_splitter` is a thin forwarder — it does NOT
    catch exceptions. Fail-soft is `splitter.run_splitter`'s
    responsibility (the inner module wraps in try/except per its
    docstring). Belt-and-suspenders fail-soft at the runner site
    would mask logic bugs in the splitter; we want exceptions there
    to surface, then be caught one frame higher inside the splitter
    itself.

    A future refactor that adds a try/except inside
    `_maybe_run_splitter` would silently swallow the boom below; this
    test is the canary for that.
    """
    def boom(**kwargs):
        raise RuntimeError("splitter exploded")
    monkeypatch.setattr(
        "claude_plan_execute.runner.run_splitter", boom,
    )
    _no_run_claude_called(monkeypatch)
    state = State(tmp_path / "state.json")
    with pytest.raises(RuntimeError):
        execute_task(
            _make_task(id=1), state,
            file_defaults={
                "plans_dir": str(tmp_path),
                "max_review_rounds": 1,
                "on_max_review_rounds": "fail",
                "auto_split_on_max_rounds": True,
            },
            cli={}, skip_to="review",
            tasks_yaml_path=tmp_path / "tasks.yaml",
        )


def test_split_status_short_circuits_scheduler_eligibility(tmp_path):
    """T17 §7.3 — _SlateScheduler._is_eligible returns False when the
    state status is 'split'."""
    from claude_plan_execute.runner import _SlateScheduler
    state = State(tmp_path / "state.json")
    state.set_status(1, "split", split_children=["1a", "1b"])
    sched = _SlateScheduler(
        ordered_tasks=[{"id": 1}],
        all_task_ids=[1],
        state=state,
        file_defaults={}, cli_overrides={},
        phases_by_id={}, task_class_models={},
        service_manager=None, dry_run=False,
        parallelism=1,
    )
    assert sched._is_eligible({"id": 1}) is False
