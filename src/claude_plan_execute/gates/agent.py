"""AgentGate runner — invokes a Claude subprocess for a registered agent Gate.

Reads the Gate's `prompt_template` (a path or inline body), prepends
a small task header, and calls `run_claude` with sonnet/medium
defaults. PASS / FAIL is detected by a case-insensitive search for
the `VERDICT: PASS` sentinel in the agent's response.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from claude_plan_execute.claude import run_claude
from claude_plan_execute.colors import yellow
from claude_plan_execute.gates import Gate, GateResult
from claude_plan_execute.gates.shell import _tail_cap

_RAW_CAP_BYTES = 16 * 1024  # 16 KB; defensive cap on _raw text in malformed-but-parsed appendix.

# Findings appendix: HTML-comment-delimited JSON. Markers chosen to be
# invisible in rendered Markdown (the dashboard renders gate output as
# Markdown). Regex semantics:
#   - re.DOTALL: `.` matches newlines.
#   - non-greedy `.*?`: nearest END binds (first appendix wins).
#   - re.search (used in _extract_findings): first match in text.
# The `\s*` flanks strip surrounding whitespace from the captured body
# so json.loads sees the raw JSON.
_FINDINGS_RE = re.compile(
    r"<!-- BEGIN_FINDINGS_JSON -->\s*(.*?)\s*<!-- END_FINDINGS_JSON -->",
    re.DOTALL,
)


def _extract_findings(
    text: str, gate_name: str, *, warn: bool = True,
) -> tuple[list[dict] | None, dict | None]:
    """Extract a structured findings appendix from agent text output.

    Returns (findings, sidecar) where:

      - findings: what to put on GateResult.findings.
        - None when markers are absent or json.loads raised.
        - list[dict] (possibly empty) otherwise.
      - sidecar: what run_gates writes as the .json body.
        - None means "do not write".
        - {"findings": [...]} for a clean bare list.
        - <full parsed object> for an object with findings: list[dict],
          including any extra keys.
        - {"findings": [], "_raw": <body, capped>} for malformed-but-
          parsed values.

    `warn=True` (default) prints yellow warnings on cases 2 and 5
    (see plan §D1). `run_gates` calls this helper a second time with
    `warn=False` to avoid double-warning.
    """
    match = _FINDINGS_RE.search(text)
    if not match:
        return (None, None)  # case 1: no markers

    body = match.group(1)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        if warn:
            print(yellow(
                f"  malformed findings JSON in {gate_name!r}; "
                f"gate verdict unchanged"
            ))
        return (None, None)  # case 2: parse failed

    # Case 3: bare list of dicts.
    if isinstance(parsed, list) and all(isinstance(el, dict) for el in parsed):
        return (parsed, {"findings": parsed})

    # Case 4: object with `findings: list[dict]` — preserve full object
    # (including extra keys like `summary`) in the sidecar.
    if isinstance(parsed, dict):
        inner = parsed.get("findings")
        if isinstance(inner, list) and all(isinstance(el, dict) for el in inner):
            return (inner, parsed)

    # Case 5: anything else.
    if warn:
        print(yellow(
            f"  findings appendix in {gate_name!r} not a list[dict] or "
            f"object with findings list; treating as empty. "
            f"gate verdict unchanged"
        ))
    raw_capped = _tail_cap(body, cap=_RAW_CAP_BYTES)
    return ([], {"findings": [], "_raw": raw_capped})


def _load_template(path_or_inline: str) -> str:
    """Read `path_or_inline` as a file path; if not a file, return verbatim.

    Tasks 15/16/20 ship template markdown under
    `src/claude_plan_execute/gates/templates/<name>.md`. The dual-
    purpose form lets a registration pass either a path or an
    inline body without a discriminator field on Gate. (Plan §D7.)

    On Python 3.8+, `Path.is_file()` raises OSError (ENAMETOOLONG)
    for strings longer than the OS filename limit (~255 bytes on
    macOS/Linux). Task 20's design-fidelity factory passes a multi-
    KB rendered template body; OSError is treated as "not a file"
    so the inline body falls through to verbatim-return.
    """
    p = Path(path_or_inline)
    try:
        if p.is_file():
            return p.read_text(encoding="utf-8")
    except OSError:
        pass
    return path_or_inline


@dataclass
class AgentGate:
    """Runner for `kind: agent` Gates.

    `diff_path` (T7, v0.3) is the path to the precomputed
    implementation diff written by the runner immediately before
    Phase 4. The substitution `body.replace("{diff_path}",
    self.diff_path or "")` runs after the template body is loaded
    so agent-gate templates can reference the diff via
    `{diff_path}` placeholder. Persona substitution still happens
    at factory time (per gate); diff_path substitution happens
    here at gate-run time because the diff is per-task and not
    known until implement finishes. (Plan §D2 / §D3.)
    """

    name: str
    gate: Gate
    task: dict
    cli: dict
    diff_path: str | None = None

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
        body = body.replace("{diff_path}", self.diff_path or "")
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
        output_capped = _tail_cap(text or "")
        # T11: extract structured findings appendix (if present) from
        # the SAME post-tail-cap text that run_gates writes to disk and
        # re-extracts from. Both call sites must see the same input;
        # see plan §6.1 "AgentGate passes output_capped to
        # _extract_findings" pin.
        findings, _ = _extract_findings(output_capped, self.name)
        return GateResult(
            name=self.name,
            passed=passed,
            output=output_capped,
            output_path=None,
            duration_seconds=time.time() - start,
            on_failure=self.gate.on_failure,
            findings=findings,
        )
