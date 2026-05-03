"""Gate Registry foundation.

Public API:
  - Gate, GateResult            (frozen dataclasses)
  - GateRegistry                (in-memory map of name → Gate)
  - default_registry()          (module-level singleton)
  - run_gates(...)              (parallel runner; see runner.py call site)
  - ShellGate, AgentGate        (re-exported from sibling modules)

Task 11 ships zero registrations. Tasks 12–17 and 20 add built-ins by
importing the default registry and calling `register()` at module load.
The runner reads `defaults.gates` (project) + `phases[i].defaults.
gates_extra` (phase overlay) and resolves names against the default
registry; unknown names produce a single warning per execute_task and
are skipped (forward-compat with the staged rollout of built-ins).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from claude_plan_execute.colors import dim, red, yellow

# ---------------------------------------------------------------------------
# Dataclasses (defined FIRST per §D11 to avoid sibling-import cycles)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    """Post-implement validation gate registration.

    Field order matches the task-11 description verbatim:
    kind, command, prompt_template, model, effort, timeout_minutes,
    parallel_safe, on_failure, when. The registry key (`name`) is
    the dict key, NOT a Gate field — see §D1.
    """
    kind: Literal["shell", "agent"]
    command: tuple[str, ...] | None = None
    prompt_template: str | None = None
    model: str | None = None
    effort: str | None = None
    timeout_minutes: int = 10
    parallel_safe: bool = True
    on_failure: Literal["block", "warn"] = "block"
    when: tuple[str, ...] = ("always",)


@dataclass(frozen=True)
class GateResult:
    """Result of running a single gate.

    `output_path` is set by `run_gates` after writing the captured
    output; the per-gate runners (ShellGate / AgentGate) leave it None.
    """
    name: str
    passed: bool
    output: str
    output_path: str | None
    duration_seconds: float
    on_failure: Literal["block", "warn"]
    error: str | None = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class GateRegistry:
    """In-memory map of gate name → Gate.

    Used by tasks 12+ to register built-in gates at module load and
    by the runner to resolve `defaults.gates` name lists against
    registered Gates. Tests can construct local registries without
    touching the module-level singleton.
    """

    def __init__(self) -> None:
        self._gates: dict[str, Gate] = {}

    def register(self, name: str, gate: Gate) -> None:
        if name in self._gates:
            raise ValueError(f"Gate {name!r} is already registered")
        self._gates[name] = gate

    def get(self, name: str) -> Gate | None:
        return self._gates.get(name)

    def names(self) -> list[str]:
        return list(self._gates.keys())

    def resolve(
        self, names: list[str],
    ) -> tuple[list[tuple[str, Gate]], list[str]]:
        """Resolve a list of names. Returns (resolved_pairs, unknown_names).

        Order of resolved pairs follows the input order. Duplicates in
        the input are preserved (no deduplication — see runner §D12).
        """
        resolved: list[tuple[str, Gate]] = []
        unknown: list[str] = []
        for n in names:
            g = self._gates.get(n)
            if g is None:
                unknown.append(n)
            else:
                resolved.append((n, g))
        return resolved, unknown


_DEFAULT_REGISTRY: "GateRegistry" = GateRegistry()


def default_registry() -> "GateRegistry":
    """Return the module-level GateRegistry singleton.

    Tasks 12+ register built-ins on this singleton at import time;
    the runner calls `default_registry().resolve(...)` to look gates
    up. Tests should construct local `GateRegistry()` instances and
    pass them via `run_gates(..., registry=...)` instead of mutating
    the singleton.
    """
    return _DEFAULT_REGISTRY


# ---------------------------------------------------------------------------
# Sibling-module imports (deferred per §D11 to avoid the dataclass cycle)
# ---------------------------------------------------------------------------

from claude_plan_execute.gates.shell import ShellGate, _OUTPUT_CAP_BYTES, _tail_cap  # noqa: E402
from claude_plan_execute.gates.agent import AgentGate  # noqa: E402


# ---------------------------------------------------------------------------
# `when` filter (task 11: "always" only — see §D6)
# ---------------------------------------------------------------------------

def _when_matches(when: tuple[str, ...], task: dict) -> tuple[bool, str]:
    """Determine whether a Gate's `when` filter permits running for this task.

    Task 11 supports:
      - empty tuple, or any tuple containing "always" → match.
      - any other value                                → no match (deferred).

    Future tasks add cases here:
      - "task_class:<name>"  (task 20: design-fidelity)
      - "touches:<glob>"     (enhancement plan §2.1)
    """
    if not when or "always" in when:
        return True, "always"
    return False, (
        f"when={list(when)} not implemented in v1 "
        f"(future task adds matchers)"
    )


# ---------------------------------------------------------------------------
# Per-gate worker (catches exceptions; produces a GateResult either way)
# ---------------------------------------------------------------------------

def _run_one(name: str, gate: Gate, task: dict, cli: dict) -> GateResult:
    """Execute one gate, returning a GateResult on success OR failure.

    Exceptions are caught and reported; the gate-run loop never raises
    out of `run_gates`.
    """
    start = time.time()
    try:
        if gate.kind == "shell":
            return ShellGate(name=name, gate=gate).run()
        if gate.kind == "agent":
            return AgentGate(name=name, gate=gate, task=task, cli=cli).run()
        return GateResult(
            name=name, passed=False, output="",
            output_path=None,
            duration_seconds=time.time() - start,
            on_failure=gate.on_failure,
            error=f"unknown kind: {gate.kind!r}",
        )
    except Exception as exc:
        import traceback
        return GateResult(
            name=name, passed=False,
            output=_tail_cap(traceback.format_exc()),
            output_path=None,
            duration_seconds=time.time() - start,
            on_failure=gate.on_failure,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_gates(
    gates: list[tuple[str, Gate]],
    *,
    task: dict,
    task_dir: Path,
    cli: dict,
    registry: "GateRegistry | None" = None,
) -> list[GateResult]:
    """Run the given gates in serial-then-parallel order, capturing output.

    Returns a list of GateResult — one per input gate (including
    skipped gates, with on_failure="warn" and a "[skipped: …]"
    output). Each result's `output_path` is populated after the
    runner writes the captured output to
    `task_dir / "gates" / "<name>.txt"`.

    The `registry` parameter is currently unused inside this function
    (the resolution happens at the call site in runner.py); it's
    accepted for forward-compat with future call sites that defer
    resolution to here.
    """
    gates_dir = task_dir / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)

    # Filter by `when` first; skipped gates produce passive results.
    runnable: list[tuple[str, Gate]] = []
    skipped: list[GateResult] = []
    for name, gate in gates:
        ok, reason = _when_matches(gate.when, task)
        if ok:
            runnable.append((name, gate))
        else:
            print(yellow(f"  Skipping gate '{name}': {reason}"))
            skipped.append(GateResult(
                name=name,
                passed=True,
                output=f"[skipped: {reason}]\n",
                output_path=None,
                duration_seconds=0.0,
                on_failure=gate.on_failure,
            ))

    serial = [(n, g) for (n, g) in runnable if not g.parallel_safe]
    parallel = [(n, g) for (n, g) in runnable if g.parallel_safe]

    results: list[GateResult] = []
    results.extend(skipped)

    # Serial first (single-runner dependence assumed; see §D5).
    for name, gate in serial:
        print(dim(f"  Running gate '{name}' (serial)..."))
        results.append(_run_one(name, gate, task, cli))

    # Then parallel.
    if parallel:
        max_workers = min(len(parallel), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_run_one, n, g, task, cli): n
                for (n, g) in parallel
            }
            for fut in as_completed(futures):
                results.append(fut.result())

    # Write output files and patch output_path on each result.
    final_results: list[GateResult] = []
    for r in results:
        out_path = gates_dir / f"{r.name}.txt"
        out_path.write_text(r.output, encoding="utf-8")
        # GateResult is frozen → reconstruct with output_path set.
        final_results.append(
            GateResult(
                name=r.name,
                passed=r.passed,
                output=r.output,
                output_path=str(out_path),
                duration_seconds=r.duration_seconds,
                on_failure=r.on_failure,
                error=r.error,
            )
        )
        verdict = "PASS" if r.passed else "FAIL"
        color = (lambda s: s) if r.passed else red
        print(color(f"    {r.name}: {verdict} ({r.duration_seconds:.1f}s)"))

    return final_results


__all__ = [
    "Gate", "GateResult", "GateRegistry",
    "default_registry", "run_gates",
    "ShellGate", "AgentGate",
]
