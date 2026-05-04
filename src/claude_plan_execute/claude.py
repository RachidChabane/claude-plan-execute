"""Subprocess wrapper around the `claude` CLI.

Owns the module-level `_active_child` reference so that the SIGINT handler
(in runner.py) can kill the in-flight child via `_kill_child()`.

Side-channel telemetry flags read by the runner immediately after each
`run_claude` call: `_last_run_timed_out`, `_last_run_subtype`,
`_last_run_rate_limited` (set when retries exhaust on a rate-limit
signal), and the five usage flags
`_last_run_input_tokens` / `_last_run_output_tokens` /
`_last_run_cache_read_tokens` / `_last_run_cache_creation_tokens` /
`_last_run_duration_seconds` (T2, v0.3 §2.7).

Token-usage capture also writes one JSONL row per `run_claude` call
to a path supplied by the runner (`usage_jsonl_path` kwarg). Writes
are guarded by `_usage_lock` (a module-level `threading.Lock`) for
correctness under `run_gates`'s ThreadPoolExecutor (up to 4 parallel
agent gates). The side-channel flags are best-effort under parallel
use; the JSONL append predicate uses the `_run_claude_once` 5-tuple
return value (race-free per-thread local).
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from claude_plan_execute.colors import dim, red, yellow

# T13 (v0.3 §2.5) — per-thread active claude subprocess, keyed on
# `threading.get_ident()`. The SIGINT-handler iterates the entire map
# (snapshot pattern) so all in-flight workers receive SIGTERM/SIGKILL.
_active_child: dict[int, subprocess.Popen] = {}
_active_child_lock = threading.Lock()

# T13 §F3.4 — local SIGTERM grace window literal. Avoids a
# cross-module import from services.py; matches v0.2.0's prior
# wait(timeout=5) literal at this site.
_SIGTERM_GRACE_SECONDS = 5

# T13 §F3.5 — per-thread namespace for the eight side-channel flags
# formerly named `_last_run_<flag>`. Each thread sees its own attribute
# bag, so sub-millisecond races between workers' run_claude calls can
# no longer clobber another thread's read. Attribute names:
#   subtype, timed_out, rate_limited, input_tokens, output_tokens,
#   cache_read_tokens, cache_creation_tokens, duration_seconds.
_last_run_state = threading.local()


def _set_active_child(proc: subprocess.Popen) -> None:
    """Record the current thread's running claude subprocess.

    Per-thread keying so parallel `run_claude` calls (T13) don't
    overwrite one another. SIGINT iterates the entire map.
    """
    with _active_child_lock:
        _active_child[threading.get_ident()] = proc


def _clear_active_child() -> None:
    """Drop the current thread's entry. Called from `run_claude`'s
    finally block. Safe to call when the entry is absent."""
    with _active_child_lock:
        _active_child.pop(threading.get_ident(), None)


def _last_run(name: str, default=None):
    """Read a per-thread flag with a default if not yet set on
    this thread. Used by runner.py for the cross-thread reads."""
    return getattr(_last_run_state, name, default)


def _reset_last_run_state() -> None:
    """Clear every per-thread flag for the current thread."""
    for attr in list(vars(_last_run_state).keys()):
        delattr(_last_run_state, attr)


# T2 §D4: serializes JSONL appends across run_gates' parallel
# agent-gate ThreadPoolExecutor (up to 4 concurrent writers).
_usage_lock: threading.Lock = threading.Lock()


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


def _append_usage_row(
    path: str | Path,
    *,
    phase: str,
    model: str | None,
    input: int = 0,
    output: int = 0,
    cache_read: int = 0,
    cache_create: int = 0,
    duration_seconds: float,
) -> None:
    """Append one JSONL row to `path` describing a single LLM/shell
    invocation. Schema: see plan task-2 §D10.

    Thread-safe via the module-level `_usage_lock` — required because
    `run_gates`' ThreadPoolExecutor runs up to 4 agent gates in
    parallel, each calling `run_claude` and (transitively) this
    helper.

    I/O errors are swallowed with a yellow warning; observability
    failure must not cascade into runner failure.
    """
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "model": model,
        "input": int(input),
        "output": int(output),
        "cache_read": int(cache_read),
        "cache_create": int(cache_create),
        "duration_seconds": float(duration_seconds),
    }
    line = json.dumps(row, sort_keys=False) + "\n"
    try:
        with _usage_lock:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except OSError as exc:
        print(yellow(f"  [usage] Failed to append row: {exc}"))


def _maybe_append_usage(
    usage_jsonl_path: str | Path | None,
    phase: str | None,
    model: str,
    usage: dict | None,
    elapsed: float,
) -> None:
    """Append a usage row when path + phase + usage are all non-None.

    Centralizes the predicate so all `run_claude` return-site call
    sites share it. (T2 §D3.)
    """
    if usage is None or usage_jsonl_path is None or phase is None:
        return
    _append_usage_row(
        usage_jsonl_path,
        phase=phase,
        model=model,
        input=usage.get("input_tokens", 0),
        output=usage.get("output_tokens", 0),
        cache_read=usage.get("cache_read_input_tokens", 0),
        cache_create=usage.get("cache_creation_input_tokens", 0),
        duration_seconds=elapsed,
    )


def _kill_child() -> None:
    """SIGTERM-then-SIGKILL every active claude subprocess.

    Iterates ALL per-thread entries (T13 parallel mode). Snapshot
    pattern avoids holding the lock across syscalls. Per-pid
    `pkill -TERM -P <pid>` preserves the v0.2.0 orphan-cleanup
    semantics. Per-proc `wait(timeout=_SIGTERM_GRACE_SECONDS)`
    mirrors v0.2.0 so synchronous test mocks
    (TimeoutExpired-on-wait + kill flips poll to -9) keep working.
    After the per-proc loop finishes, the dict is `clear()`-ed so
    post-SIGINT introspection reflects "no active children".
    """
    with _active_child_lock:
        snapshot = list(_active_child.values())
    for proc in snapshot:
        if proc.poll() is None:
            try:
                proc.terminate()
            except (ProcessLookupError, OSError):
                pass
            try:
                subprocess.run(
                    ["pkill", "-TERM", "-P", str(proc.pid)],
                    capture_output=True, timeout=3,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            try:
                proc.wait(timeout=_SIGTERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except (ProcessLookupError, OSError):
                    pass
    with _active_child_lock:
        _active_child.clear()


def _run_claude_once(
    prompt: str,
    *,
    model: str,
    effort: str,
    max_turns: int,
    timeout_minutes: int,
    skip_permissions: bool,
    verbose: bool,
) -> tuple[int, str, str, dict | None, float]:
    """Single subprocess attempt. Returns
    (returncode, result_text, stderr_text, usage_or_None, elapsed_seconds).

    Resets `_last_run_state.timed_out`, `_last_run_state.subtype`, and
    the five usage side-channel flags at entry — each call's flags
    reflect this attempt only.
    """
    _last_run_state.timed_out = False
    _last_run_state.subtype = None
    _last_run_state.input_tokens = None
    _last_run_state.output_tokens = None
    _last_run_state.cache_read_tokens = None
    _last_run_state.cache_creation_tokens = None
    _last_run_state.duration_seconds = None

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
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _set_active_child(proc)
        stdout, stderr = proc.communicate(timeout=timeout_minutes * 60)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        print(red(f"  TIMEOUT after {timeout_minutes} minutes"))
        _last_run_state.timed_out = True
        _kill_child()
        _last_run_state.duration_seconds = 0.0
        return 1, "", "", None, 0.0
    finally:
        _clear_active_child()

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
    usage: dict | None = None
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
            _last_run_state.subtype = result_event.get("subtype")
            if "result" in result_event:
                result_text = result_event["result"]
            usage_field = result_event.get("usage")
            if isinstance(usage_field, dict):
                usage = usage_field
                _last_run_state.input_tokens = int(usage.get("input_tokens", 0))
                _last_run_state.output_tokens = int(usage.get("output_tokens", 0))
                _last_run_state.cache_read_tokens = int(
                    usage.get("cache_read_input_tokens", 0)
                )
                _last_run_state.cache_creation_tokens = int(
                    usage.get("cache_creation_input_tokens", 0)
                )
            else:
                usage = None
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    _last_run_state.duration_seconds = elapsed
    return returncode, result_text, stderr, usage, elapsed


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
    usage_jsonl_path: str | Path | None = None,
    phase: str | None = None,
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

    Per-attempt side-channel flags (`_last_run_state.timed_out`,
    `_last_run_state.subtype`) reflect the most recent attempt.

    `usage_jsonl_path` and `phase` (T2, v0.3 §2.7): when both are
    non-None and the FINAL attempt's CLI output included a `usage`
    event, append one JSONL row to `usage_jsonl_path` recording the
    final attempt's token counts and elapsed seconds. ONE row per
    `run_claude` call (per-call contract). Either kwarg unset
    disables the append.
    """
    _last_run_state.rate_limited = False

    # Opt-out: 0 retries → skip rate-limit awareness entirely.
    if max_rate_limit_retries == 0:
        code, result_text, _stderr, usage, elapsed = _run_claude_once(
            prompt,
            model=model, effort=effort, max_turns=max_turns,
            timeout_minutes=timeout_minutes,
            skip_permissions=skip_permissions, verbose=verbose,
        )
        _maybe_append_usage(
            usage_jsonl_path, phase, model, usage, elapsed,
        )
        return code, result_text

    last_code, last_text = 1, ""
    last_usage: dict | None = None
    last_elapsed: float = 0.0
    total_attempts = max_rate_limit_retries + 1
    for attempt in range(1, total_attempts + 1):
        code, result_text, stderr, usage, elapsed = _run_claude_once(
            prompt,
            model=model, effort=effort, max_turns=max_turns,
            timeout_minutes=timeout_minutes,
            skip_permissions=skip_permissions, verbose=verbose,
        )
        last_code, last_text = code, result_text
        last_usage, last_elapsed = usage, elapsed

        if code == 0:
            _maybe_append_usage(
                usage_jsonl_path, phase, model, last_usage, last_elapsed,
            )
            return code, result_text

        # Timeout: do NOT retry; do NOT consume rate-limit budget.
        if _last_run("timed_out", False):
            _maybe_append_usage(
                usage_jsonl_path, phase, model, last_usage, last_elapsed,
            )
            return code, result_text

        # Non-rate-limit non-zero exit: pass through unchanged.
        if not _is_rate_limited(stderr):
            _maybe_append_usage(
                usage_jsonl_path, phase, model, last_usage, last_elapsed,
            )
            return code, result_text

        # Rate-limited; sleep before next attempt unless this was the last.
        if attempt < total_attempts:
            wait = rate_limit_base_backoff_seconds * (2 ** (attempt - 1))
            print(yellow(
                f"  Rate-limited; waiting {wait}s before retry "
                f"{attempt + 1}/{total_attempts}"
            ))
            time.sleep(wait)

    _maybe_append_usage(
        usage_jsonl_path, phase, model, last_usage, last_elapsed,
    )
    _last_run_state.rate_limited = True
    print(red(
        f"  Rate-limit retries exhausted after {total_attempts} attempts."
    ))
    return last_code, last_text
