"""Tests for prompts, claude subprocess wrapper, and runner orchestrator."""
from __future__ import annotations

import json
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
    """Reset claude._active_child, claude._last_run_*, and runner._current_*
    between tests. Without the claude flag reset, a test that sets
    _last_run_timed_out via a stub leaks the flag into subsequent tests."""
    claude._active_child = None
    claude._last_run_timed_out = False
    claude._last_run_subtype = None
    claude._last_run_rate_limited = False
    runner._current_state = None
    runner._current_task_id = 0
    runner._current_phase = "plan"
    yield
    claude._active_child = None
    claude._last_run_timed_out = False
    claude._last_run_subtype = None
    claude._last_run_rate_limited = False
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


def test_implement_prompt_includes_commit_message_verbatim():
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md",
        "feat: my custom commit message",
    )
    assert "feat: my custom commit message" in out


def test_implement_prompt_byte_length_matches_baseline():
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
    )
    assert len(out) == 350


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
    assert claude._last_run_subtype == "error_max_turns"


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
    assert claude._last_run_subtype == "success"


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
    claude._last_run_subtype = "leftover_from_prior_call"
    _patch_run_claude_subprocess(monkeypatch, stdout="not json")
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert claude._last_run_subtype is None


def test_run_claude_timeout_sets_last_run_timed_out_flag(monkeypatch):
    """Pin §D6: subprocess.TimeoutExpired path sets the side-channel
    flag. The runner reads this immediately after the call to
    classify the failure as `implementer_timeout`."""
    _patch_run_claude_subprocess(monkeypatch, raises_timeout=True)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert claude._last_run_timed_out is True


def test_run_claude_resets_timed_out_flag_at_call_entry(monkeypatch):
    """Stale-flag pin: a clean call after a prior timeout must clear the flag."""
    claude._last_run_timed_out = True
    _patch_run_claude_subprocess(monkeypatch)
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert claude._last_run_timed_out is False


# ---------------------------------------------------------------------------
# D2. run_claude rate-limit retry (v0.3 task 1)
# ---------------------------------------------------------------------------


def _patch_run_claude_once(monkeypatch, responses):
    """Drive run_claude with a deterministic sequence of (code, text, stderr)
    tuples. Returns (helper_calls, sleep_calls)."""
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
        claude._last_run_timed_out = (code == 1 and text == "" and stderr == "TIMEOUT")
        claude._last_run_subtype = None
        return code, text, "" if stderr == "TIMEOUT" else stderr

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
    assert claude._last_run_rate_limited is False


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
    assert claude._last_run_rate_limited is False


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
    assert claude._last_run_rate_limited is True


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
    assert claude._last_run_rate_limited is False


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
    assert claude._last_run_timed_out is True
    assert claude._last_run_rate_limited is False


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
    assert claude._last_run_rate_limited is False


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
    claude._last_run_rate_limited = True  # leftover
    _patch_run_claude_once(monkeypatch, [
        (0, "ok", ""),
    ])
    run_claude(
        "p", model="opus", effort="max", max_turns=1, timeout_minutes=1,
        skip_permissions=False, verbose=False,
    )
    assert claude._last_run_rate_limited is False


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
    assert set(cfg.keys()) == set(KEYS_TO_AUDIT)


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
        claude._last_run_timed_out = True
        claude._last_run_subtype = None
        return 1, ""

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub)
    return called


def _patch_run_claude_simulating_max_turns(monkeypatch):
    """Stub run_claude that simulates a max-turns hit: sets the
    subtype side-channel flag."""
    called = []

    def stub(*args, **kwargs):
        called.append((args, kwargs))
        claude._last_run_timed_out = False
        claude._last_run_subtype = "error_max_turns"
        return 0, ""  # CLI exits cleanly even on max-turns

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub)
    return called


def _patch_run_claude_simulating_success(monkeypatch):
    """Stub run_claude that simulates a clean run."""
    called = []

    def stub(*args, **kwargs):
        called.append((args, kwargs))
        claude._last_run_timed_out = False
        claude._last_run_subtype = "success"
        return 0, "ok"

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub)
    return called


def test_implement_timeout_records_failure(monkeypatch, tmp_path):
    """Pin §D7: implementer_timeout FailureRecord written when
    claude._last_run_timed_out is True after Phase 3 run_claude."""
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
    claude._last_run_subtype == 'error_max_turns'."""
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
    claude._last_run_timed_out = True  # leftover from a prior call
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
    runner._current_task_id = 1
    runner._current_phase = "implement"
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
    runner._current_task_id = 1
    runner._current_phase = "plan"
    with pytest.raises(SystemExit):
        handle_sigint(2, None)
    assert state.get(1)["attempts"][0]["phase"] == "plan"


def test_handle_sigint_does_NOT_record_when_status_is_done(monkeypatch, tmp_path):
    """Pin §D10: tasks already in terminal status are not over-recorded."""
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    state = State(tmp_path / "state.json")
    state.set_status(1, "done")
    runner._current_state = state
    runner._current_task_id = 1
    with pytest.raises(SystemExit):
        handle_sigint(2, None)
    assert state.get(1)["attempts"] == []


def test_handle_sigint_does_NOT_record_when_status_is_blocked(monkeypatch, tmp_path):
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    state = State(tmp_path / "state.json")
    state.set_status(1, "blocked", block_reason="implementer_timeout")
    runner._current_state = state
    runner._current_task_id = 1
    with pytest.raises(SystemExit):
        handle_sigint(2, None)
    assert state.get(1)["attempts"] == []


def test_handle_sigint_does_NOT_record_when_state_is_None(monkeypatch):
    """Pin §D10: SIGINT before any execute_task is a noop on the failure surface."""
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    runner._current_state = None
    runner._current_task_id = 0
    # Must not raise AttributeError or write anything.
    with pytest.raises(SystemExit):
        handle_sigint(2, None)


def test_handle_sigint_does_NOT_record_when_task_id_is_zero(monkeypatch, tmp_path):
    """Pin §D10: _current_task_id == 0 (initial sentinel) means no task in flight."""
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    state = State(tmp_path / "state.json")
    runner._current_state = state
    runner._current_task_id = 0
    runner._current_phase = "plan"
    with pytest.raises(SystemExit):
        handle_sigint(2, None)
    # No task entry should have been touched.
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
    runner._current_task_id = 1
    runner._current_phase = "implement"
    with pytest.raises(SystemExit) as exc_info:
        handle_sigint(2, None)
    assert exc_info.value.code == 130


def test_handle_sigint_still_exits_130(monkeypatch, tmp_path):
    """Sanity: the recording path does NOT change the SIGINT exit code."""
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    state = State(tmp_path / "state.json")
    state.set_status(1, "implementing")
    runner._current_state = state
    runner._current_task_id = 1
    runner._current_phase = "implement"
    with pytest.raises(SystemExit) as exc_info:
        handle_sigint(2, None)
    assert exc_info.value.code == 130


def test_handle_sigint_still_prints_resume_command(monkeypatch, capsys, tmp_path):
    """Sanity: the recording path does NOT change the user-facing print."""
    monkeypatch.setattr("claude_plan_execute.runner._kill_child", lambda: None)
    state = State(tmp_path / "state.json")
    state.set_status(7, "implementing")
    runner._current_state = state
    runner._current_task_id = 7
    runner._current_phase = "implement"
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
        claude._last_run_timed_out = False
        claude._last_run_subtype = None
        claude._last_run_rate_limited = True
        return 1, ""

    monkeypatch.setattr("claude_plan_execute.runner.run_claude", stub)
    return called


def test_implement_rate_limit_records_failure(monkeypatch, tmp_path):
    """Pin §D6: rate_limited FailureRecord written when
    claude._last_run_rate_limited is True after Phase 3 run_claude."""
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
        claude._last_run_timed_out = True       # spurious
        claude._last_run_rate_limited = True    # authoritative
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
        claude._last_run_timed_out = False
        claude._last_run_subtype = "success"
        claude._last_run_rate_limited = False
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
        claude._last_run_timed_out = False
        claude._last_run_subtype = "success"
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
