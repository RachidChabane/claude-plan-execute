"""Subprocess wrapper around the `claude` CLI.

Owns the module-level `_active_child` reference so that the SIGINT handler
(in runner.py) can kill the in-flight child via `_kill_child()`.
"""
from __future__ import annotations

import json
import subprocess
import time

from claude_plan_execute.colors import dim, red, yellow

_active_child: subprocess.Popen | None = None

# Side-channel telemetry set by `run_claude`. The runner reads these
# IMMEDIATELY after a `run_claude` call to differentiate normal
# completion from timeout (subprocess.TimeoutExpired) and from
# max-turns (CLI subtype "error_max_turns"). Reset to defaults at
# every `run_claude` entry; never read except through the runner's
# post-call inspection.
_last_run_timed_out: bool = False
_last_run_subtype: str | None = None


def _kill_child():
    """Kill the active claude child process and all its descendants."""
    global _active_child
    if _active_child and _active_child.poll() is None:
        pid = _active_child.pid
        # First, try SIGTERM on the child directly
        try:
            _active_child.terminate()
        except OSError:
            pass
        try:
            _active_child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill
            try:
                _active_child.kill()
            except OSError:
                pass
        # Also kill any orphaned descendants via pkill (belt and suspenders)
        try:
            subprocess.run(
                ["pkill", "-TERM", "-P", str(pid)],
                capture_output=True, timeout=3,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    _active_child = None


def run_claude(
    prompt: str,
    *,
    model: str,
    effort: str,
    max_turns: int,
    timeout_minutes: int,
    skip_permissions: bool,
    verbose: bool,
) -> tuple[int, str]:
    """Invoke claude CLI. Returns (return_code, stdout_text).

    Side-channel telemetry: sets module-level `_last_run_timed_out`
    and `_last_run_subtype` for the runner to inspect post-call.
    Both are reset at entry.
    """
    global _active_child, _last_run_timed_out, _last_run_subtype
    _last_run_timed_out = False
    _last_run_subtype = None

    cmd = [
        "claude",
        "-p", prompt,
        "--output-format", "json",
        "--model", model,
        "--effort", effort,
        "--max-turns", str(max_turns),
    ]
    if skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    if verbose:
        cmd.append("--verbose")

    print(dim(f"  claude --model {model} --effort {effort} --max-turns {max_turns} "
              f"(timeout {timeout_minutes}m, {len(prompt)} char prompt)"))

    start = time.time()
    try:
        # Do NOT use setsid here — we want claude children in our process group
        # so that killing our group (from the server) kills everything.
        # For direct CLI usage, the SIGINT handler calls _kill_child explicitly.
        _active_child = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = _active_child.communicate(timeout=timeout_minutes * 60)
        returncode = _active_child.returncode
    except subprocess.TimeoutExpired:
        print(red(f"  TIMEOUT after {timeout_minutes} minutes"))
        _last_run_timed_out = True
        _kill_child()
        return 1, ""
    finally:
        _active_child = None

    elapsed = time.time() - start
    print(dim(f"  Completed in {elapsed:.0f}s (exit {returncode})"))

    if returncode != 0 and stderr:
        print(yellow(f"  stderr: {stderr[:300]}"))

    # Parse stdout for both telemetry (subtype) and result-text
    # extraction. CLI v2.x emits a JSON ARRAY of events; the terminal
    # "result" event carries `subtype` and `result`. Older CLI
    # versions emitted a single dict; both shapes handled for compat.
    #
    # Dict-path note: legacy single-dict shape is treated as the
    # result event WITHOUT a `type == "result"` filter to preserve
    # backward-compat with `{"result": "..."}` outputs (no `type`
    # key) used by older CLI versions and existing tests. Future
    # subtype-conditioned branches must guard the dict path with
    # `evt.get("type") == "result"` if they consume other subtypes.
    result_text = stdout
    try:
        parsed = json.loads(stdout)
        result_event: dict | None = None
        if isinstance(parsed, list):
            for evt in reversed(parsed):
                if isinstance(evt, dict) and evt.get("type") == "result":
                    result_event = evt
                    break
        elif isinstance(parsed, dict):
            result_event = parsed
        if result_event is not None:
            _last_run_subtype = result_event.get("subtype")
            if "result" in result_event:
                result_text = result_event["result"]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return returncode, result_text
