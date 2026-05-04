"""Subprocess wrapper around the `claude` CLI.

Owns the module-level `_active_child` reference so that the SIGINT handler
(in runner.py) can kill the in-flight child via `_kill_child()`.

Side-channel telemetry flags read by the runner immediately after each
`run_claude` call: `_last_run_timed_out`, `_last_run_subtype`, and
`_last_run_rate_limited` (set when retries exhaust on a rate-limit
signal).
"""
from __future__ import annotations

import json
import subprocess
import time

from claude_plan_execute.colors import dim, red, yellow

_active_child: subprocess.Popen | None = None

# Side-channel telemetry set by `run_claude`. The runner reads these
# IMMEDIATELY after a `run_claude` call to differentiate normal
# completion from timeout (subprocess.TimeoutExpired), max-turns
# (CLI subtype "error_max_turns"), and rate-limit retry exhaustion.
# `_last_run_timed_out` and `_last_run_subtype` are reset per-attempt
# at `_run_claude_once` entry; `_last_run_rate_limited` is reset
# per-call at `run_claude` entry.
_last_run_timed_out: bool = False
_last_run_subtype: str | None = None
_last_run_rate_limited: bool = False


# Substring patterns matched (lower-case) against subprocess stderr to
# detect Claude-API rate-limit signals. Any one match triggers a retry.
# The "Retry-After" pattern is matched as a SIGNAL only — the value is
# not parsed (deferred refinement).
_RATE_LIMIT_PATTERNS = (
    "rate_limit_error",
    "429",
    "anthropic api rate limit",
    "retry-after",
)


def _is_rate_limited(stderr: str) -> bool:
    """Case-insensitive substring match against `_RATE_LIMIT_PATTERNS`.

    Empty stderr → False. The match is intentionally permissive —
    Anthropic's CLI has emitted multiple stderr shapes over time;
    surfacing any of the four patterns is enough to retry."""
    if not stderr:
        return False
    s = stderr.lower()
    return any(p in s for p in _RATE_LIMIT_PATTERNS)


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


def _run_claude_once(
    prompt: str,
    *,
    model: str,
    effort: str,
    max_turns: int,
    timeout_minutes: int,
    skip_permissions: bool,
    verbose: bool,
) -> tuple[int, str, str]:
    """Single subprocess attempt. Returns (returncode, result_text, stderr_text).

    Resets `_last_run_timed_out` and `_last_run_subtype` at entry —
    each call's side-channel flags reflect this attempt only.
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
        return 1, "", ""
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
    return returncode, result_text, stderr


def run_claude(
    prompt: str,
    *,
    model: str,
    effort: str,
    max_turns: int,
    timeout_minutes: int,
    skip_permissions: bool,
    verbose: bool,
    max_rate_limit_retries: int = 3,
    rate_limit_base_backoff_seconds: int = 60,
) -> tuple[int, str]:
    """Invoke claude CLI with rate-limit retry. Returns (return_code, stdout_text).

    Retries on Claude-API rate-limit signals in stderr (see
    `_RATE_LIMIT_PATTERNS`). Backoff is exponential:
    `base * 2^(attempt-1)` seconds — 60s/120s/240s for the default
    3 retries (~7 min total). Non-rate-limit failures and timeouts
    return immediately and DO NOT consume the retry budget.

    `max_rate_limit_retries` counts retry attempts AFTER the initial
    one — N retries → N+1 total attempts → N waits with values
    `base, base*2, ..., base*2^(N-1)`. `max_rate_limit_retries=0`
    disables rate-limit awareness (single attempt, detection skipped).

    Sets module-level `_last_run_rate_limited = True` iff retries
    exhaust on a rate-limit signal. The runner reads this flag
    immediately post-call to record a `rate_limited` FailureRecord.

    Per-attempt side-channel flags (`_last_run_timed_out`,
    `_last_run_subtype`) reflect the most recent attempt.
    """
    global _last_run_rate_limited
    _last_run_rate_limited = False

    # Opt-out: 0 retries → skip rate-limit awareness entirely.
    if max_rate_limit_retries == 0:
        code, result_text, _stderr = _run_claude_once(
            prompt,
            model=model, effort=effort, max_turns=max_turns,
            timeout_minutes=timeout_minutes,
            skip_permissions=skip_permissions, verbose=verbose,
        )
        return code, result_text

    last_code, last_text = 1, ""
    total_attempts = max_rate_limit_retries + 1
    for attempt in range(1, total_attempts + 1):
        code, result_text, stderr = _run_claude_once(
            prompt,
            model=model, effort=effort, max_turns=max_turns,
            timeout_minutes=timeout_minutes,
            skip_permissions=skip_permissions, verbose=verbose,
        )
        last_code, last_text = code, result_text

        if code == 0:
            return code, result_text

        # Timeout: do NOT retry; do NOT consume rate-limit budget.
        if _last_run_timed_out:
            return code, result_text

        # Non-rate-limit non-zero exit: pass through unchanged.
        if not _is_rate_limited(stderr):
            return code, result_text

        # Rate-limited; sleep before next attempt unless this was the last.
        if attempt < total_attempts:
            wait = rate_limit_base_backoff_seconds * (2 ** (attempt - 1))
            print(yellow(
                f"  Rate-limited; waiting {wait}s before retry "
                f"{attempt + 1}/{total_attempts}"
            ))
            time.sleep(wait)

    _last_run_rate_limited = True
    print(red(
        f"  Rate-limit retries exhausted after {total_attempts} attempts."
    ))
    return last_code, last_text
