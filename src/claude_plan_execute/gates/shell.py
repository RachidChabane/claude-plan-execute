"""ShellGate runner — executes a registered shell-kind Gate.

The Gate's `command` tuple is passed to `subprocess.run` directly
(no `shell=True`). stdout and stderr are concatenated; the gate's
exit code is the sole PASS / FAIL signal. Output is tail-truncated
to 100 KB UTF-8 bytes. Timeouts produce a `passed=False` GateResult
with `error="timeout"` (the runner does not raise).
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

from claude_plan_execute.gates import Gate, GateResult

_OUTPUT_CAP_BYTES = 100 * 1024  # 100 KB; matches enhancement plan §2.1


def _tail_cap(text: str) -> str:
    """Tail-truncate `text` to fit within _OUTPUT_CAP_BYTES UTF-8 bytes.

    Prepends a `[truncated: <N> bytes omitted at head]` marker when
    truncation happens. Tail-cap (not head-cap) preserves the most
    actionable signal — the end of test output, the final exit
    summary. (Plan §D8.)
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= _OUTPUT_CAP_BYTES:
        return text
    omitted = len(encoded) - _OUTPUT_CAP_BYTES
    tailed = encoded[-_OUTPUT_CAP_BYTES:].decode("utf-8", errors="ignore")
    return f"[truncated: {omitted} bytes omitted at head]\n{tailed}"


@dataclass
class ShellGate:
    """Runner for `kind: shell` Gates."""

    name: str
    gate: Gate

    def run(self) -> GateResult:
        if self.gate.kind != "shell":
            raise ValueError(
                f"ShellGate requires kind='shell'; got {self.gate.kind!r}"
            )
        if not self.gate.command:
            raise ValueError(
                f"Shell gate {self.name!r} requires a command tuple"
            )
        start = time.time()
        try:
            proc = subprocess.run(
                list(self.gate.command),
                capture_output=True,
                text=True,
                timeout=self.gate.timeout_minutes * 60,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return GateResult(
                name=self.name,
                passed=(proc.returncode == 0),
                output=_tail_cap(output),
                output_path=None,
                duration_seconds=time.time() - start,
                on_failure=self.gate.on_failure,
            )
        except subprocess.TimeoutExpired as exc:
            output = (
                f"[timeout after {self.gate.timeout_minutes} minutes]\n"
                f"stdout:\n{exc.stdout or ''}\n"
                f"stderr:\n{exc.stderr or ''}"
            )
            return GateResult(
                name=self.name,
                passed=False,
                output=_tail_cap(output),
                output_path=None,
                duration_seconds=time.time() - start,
                on_failure=self.gate.on_failure,
                error="timeout",
            )
