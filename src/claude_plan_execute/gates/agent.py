"""AgentGate runner — invokes a Claude subprocess for a registered agent Gate.

Reads the Gate's `prompt_template` (a path or inline body), prepends
a small task header, and calls `run_claude` with sonnet/medium
defaults. PASS / FAIL is detected by a case-insensitive search for
the `VERDICT: PASS` sentinel in the agent's response.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from claude_plan_execute.claude import run_claude
from claude_plan_execute.gates import Gate, GateResult
from claude_plan_execute.gates.shell import _tail_cap


def _load_template(path_or_inline: str) -> str:
    """Read `path_or_inline` as a file path; if not a file, return verbatim.

    Tasks 15/16/20 ship template markdown under
    `src/claude_plan_execute/gates/templates/<name>.md`. The dual-
    purpose form lets a registration pass either a path or an
    inline body without a discriminator field on Gate. (Plan §D7.)
    """
    p = Path(path_or_inline)
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return path_or_inline


@dataclass
class AgentGate:
    """Runner for `kind: agent` Gates."""

    name: str
    gate: Gate
    task: dict
    cli: dict

    def run(self) -> GateResult:
        if self.gate.kind != "agent":
            raise ValueError(
                f"AgentGate requires kind='agent'; got {self.gate.kind!r}"
            )
        if not self.gate.prompt_template:
            raise ValueError(
                f"Agent gate {self.name!r} requires a prompt_template"
            )

        body = _load_template(self.gate.prompt_template)
        prompt = (
            f"You are running a post-implement gate for the "
            f"following task.\n\n"
            f"TASK {self.task['id']}: {self.task['title']}\n\n"
            f"DESCRIPTION:\n{self.task.get('description', '')}\n\n"
            f"{body}"
        )
        start = time.time()
        code, text = run_claude(
            prompt,
            model=self.gate.model or "sonnet",
            effort=self.gate.effort or "medium",
            max_turns=60,
            timeout_minutes=self.gate.timeout_minutes,
            skip_permissions=self.cli.get("skip_permissions", False),
            verbose=self.cli.get("verbose", False),
        )
        passed = "verdict: pass" in (text or "").lower()
        return GateResult(
            name=self.name,
            passed=passed,
            output=_tail_cap(text or ""),
            output_path=None,
            duration_seconds=time.time() - start,
            on_failure=self.gate.on_failure,
        )
