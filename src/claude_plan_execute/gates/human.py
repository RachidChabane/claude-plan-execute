"""HumanGate runner — pauses the runner pending an operator decision.

Phase-4 dispatch for `kind: human` Gates. The runner writes a
human-readable request file (`<name>.request.md`) into the task's
gates dir, sets state to `awaiting_human`, then polls for an
operator-supplied sentinel (`<name>.approval` or `<name>.rejection`).
The sentinel content (the operator's note) becomes
`GateResult.output`. See plans/task-10/plan.md §6 D3-D11.

Sentinel paths default to the task-dir convention
(`task_dir/gates/<name>.{approval|rejection}`); the Gate's
`sentinel_path` field overrides the approval path verbatim, and the
rejection path is derived by swapping the `.approval` suffix for
`.rejection`.

The HumanGate cooperates with the runner via a `state: State` reference
threaded through `run_gates`. When `state` is None (unit-test path),
status updates are skipped — the rest of the contract still applies.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from claude_plan_execute.gates import Gate, GateResult
from claude_plan_execute.gates.agent import _load_template
from claude_plan_execute.gates.shell import _tail_cap


@dataclass
class HumanGate:
    """Runner for `kind: human` Gates.

    `state` is a forward reference to `claude_plan_execute.state.State`;
    typed as `object | None` so importing this module doesn't trigger a
    cycle (state -> gates is the canonical direction; we don't want
    gates.human -> state). When None, status updates are skipped — the
    request file still gets written and the poll loop still runs.

    `poll_interval_seconds` is the testability seam (§D9). Production
    callers (the runner) leave it at 10.0; tests pass 0.01 and seed
    sentinels from a worker thread to keep wall time bounded.
    """

    name: str
    gate: Gate
    task: dict
    task_dir: Path
    state: object | None = None
    poll_interval_seconds: float = 10.0

    def run(self) -> GateResult:
        if self.gate.kind != "human":
            raise ValueError(
                f"HumanGate requires kind='human'; got {self.gate.kind!r}"
            )
        if not self.gate.prompt_template:
            raise ValueError(
                f"Human gate {self.name!r} requires a prompt_template"
            )

        approval_path, rejection_path = self._compute_sentinel_paths()
        gates_dir = self.task_dir / "gates"
        gates_dir.mkdir(parents=True, exist_ok=True)

        # §D3 — pre-clear stale sentinels so a re-run after a prior
        # rejection doesn't return FAIL instantly.
        approval_path.unlink(missing_ok=True)
        rejection_path.unlink(missing_ok=True)

        request_body = self._render_request_body(
            approval_path=approval_path,
            rejection_path=rejection_path,
        )
        request_file = gates_dir / f"{self.name}.request.md"
        request_file.write_text(request_body, encoding="utf-8")

        tid = self.task.get("id")
        if self.state is not None and tid is not None:
            self.state.set_status(  # type: ignore[attr-defined]
                tid, "awaiting_human", awaiting_gate_name=self.name,
            )

        start = time.time()
        timeout_seconds = (self.gate.timeout_minutes or 1440) * 60
        deadline = start + timeout_seconds

        try:
            while True:
                # §D7 — approval wins on simultaneous appearance.
                if approval_path.exists():
                    note = approval_path.read_text(encoding="utf-8")
                    return GateResult(
                        name=self.name,
                        passed=True,
                        output=_tail_cap(note),
                        output_path=None,
                        duration_seconds=time.time() - start,
                        on_failure=self.gate.on_failure,
                    )
                if rejection_path.exists():
                    note = rejection_path.read_text(encoding="utf-8")
                    return GateResult(
                        name=self.name,
                        passed=False,
                        output=_tail_cap(note),
                        output_path=None,
                        duration_seconds=time.time() - start,
                        on_failure=self.gate.on_failure,
                    )
                if time.time() >= deadline:
                    return GateResult(
                        name=self.name,
                        passed=False,
                        output="",
                        output_path=None,
                        duration_seconds=time.time() - start,
                        on_failure=self.gate.on_failure,
                        error="human_gate_timeout",
                    )
                time.sleep(self.poll_interval_seconds)
        finally:
            # §D10 — drop awaiting_gate_name and restore status. The
            # post-Phase-4 dispatch will overwrite "implementing" with
            # "done" / "blocked" as appropriate; if the gate is mid-
            # serial-pool, "implementing" is the right interim status.
            if self.state is not None and tid is not None:
                entry = self.state.get(tid)  # type: ignore[attr-defined]
                entry.pop("awaiting_gate_name", None)
                self.state.set_status(  # type: ignore[attr-defined]
                    tid, "implementing",
                )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _compute_sentinel_paths(self) -> tuple[Path, Path]:
        """Return (approval_path, rejection_path) per §D4.

        When `Gate.sentinel_path` is set, it overrides the approval
        path verbatim; the rejection path swaps the `.approval`
        suffix for `.rejection`. Otherwise, both paths are derived
        from `task_dir/gates/<name>.{approval|rejection}`.
        """
        if self.gate.sentinel_path:
            approval = Path(self.gate.sentinel_path)
            if approval.suffix == ".approval":
                rejection = approval.with_suffix(".rejection")
            else:
                rejection = Path(
                    str(approval).removesuffix(".approval") + ".rejection"
                )
            return approval, rejection
        gates_dir = self.task_dir / "gates"
        return (
            gates_dir / f"{self.name}.approval",
            gates_dir / f"{self.name}.rejection",
        )

    def _render_request_body(
        self, *, approval_path: Path, rejection_path: Path,
    ) -> str:
        """Produce the human-readable request markdown.

        Mirrors AgentGate's header pattern (`gates/agent.py:159-165`)
        plus a footer with response instructions. Five placeholders
        are substituted via `str.replace` (no template engine, no
        f-string evaluation — see §D5).
        """
        body = _load_template(self.gate.prompt_template or "")
        gates_dir = self.task_dir / "gates"
        diff_path = self.task_dir / "diff.patch"
        plan_path = self.task_dir / "plan.md"
        body = body.replace("{plan_path}", str(plan_path))
        body = body.replace("{gates_dir}", str(gates_dir))
        body = body.replace("{diff_path}", str(diff_path))
        body = body.replace("{approval_path}", str(approval_path))
        body = body.replace("{rejection_path}", str(rejection_path))

        tid = self.task.get("id", "?")
        title = self.task.get("title", "")
        description = self.task.get("description", "")
        return (
            f"# Human-gate review request\n\n"
            f"TASK {tid}: {title}\n\n"
            f"DESCRIPTION:\n{description}\n\n"
            f"---\n\n"
            f"{body}\n\n"
            f"---\n\n"
            f"## How to respond\n\n"
            f"Approve: write your note to:\n"
            f"  {approval_path}\n\n"
            f"Reject: write your note to:\n"
            f"  {rejection_path}\n\n"
            f"Or via CLI:\n"
            f"  claude-plan-execute approve {tid} {self.name} --note '...'\n"
            f"  claude-plan-execute reject {tid} {self.name} --note '...'\n"
        )


__all__ = ["HumanGate"]
