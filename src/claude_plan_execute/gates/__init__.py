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

import pathspec
from pathspec.patterns import GitWildMatchPattern

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

    Supported matchers:
      - empty tuple, or any tuple containing "always" → match.
      - "task_class:<name>" → matches when task["task_class"] == "<name>".
        Multiple `task_class:<name>` tokens in the same `when` are OR-ed.
      - "touches:<glob>" → matches when any gitignore-style glob in the
        `when` tuple matches any path in `task["key_files"]`. Globs use
        `pathspec.GitWildMatchPattern` semantics — path-component aware,
        gitignore-equivalent. Multiple `touches:<glob>` tokens are OR-ed
        (so `["touches:**/auth/**", "touches:**/api/**"]` matches a task
        whose key_files touches EITHER subtree). To get AND semantics,
        declare multiple gates each with one pattern.
      - `task_class` and `touches` ALSO OR across categories: a `when`
        of `["task_class:ui", "touches:**/auth/**"]` matches if either
        the class OR any glob matches.
      - any unrecognised token, in absence of "always" or a matching
        predicate — no match (skipped with a passive-result reason).

    The `touches:` matcher uses pathspec rather than stdlib fnmatch
    because fnmatch lacks path-component awareness: `*` matches `/`,
    so `src/api/*.py` over-matches `src/api/v1/users.py`, and
    `**/auth/**` under-matches `auth/foo.py` at the leading edge.
    See plans/task-3/plan.md §"Documented deviations" for the
    empirical comparison.
    """
    if not when or "always" in when:
        return True, "always"

    actual_class = task.get("task_class")
    class_tokens = [t for t in when if t.startswith("task_class:")]
    touch_tokens = [t for t in when if t.startswith("touches:")]

    # task_class branch — cheap, runs first.
    if class_tokens:
        wanted = {t[len("task_class:"):] for t in class_tokens}
        if actual_class in wanted:
            return True, f"task_class={actual_class!r}"
        # task_class miss falls through to touches; do NOT return False yet.

    # touches branch — gitignore-style glob match against key_files.
    if touch_tokens:
        patterns = [t[len("touches:"):] for t in touch_tokens]
        spec = pathspec.PathSpec.from_lines(GitWildMatchPattern, patterns)
        key_files = task.get("key_files") or []
        matched = [kf for kf in key_files if spec.match_file(kf)]
        if matched:
            return True, f"touches=match:{matched[:3]}"

    # Recognised predicates considered, none matched.
    if class_tokens or touch_tokens:
        parts: list[str] = []
        if class_tokens:
            wanted = sorted({t[len("task_class:"):] for t in class_tokens})
            parts.append(f"task_class={actual_class!r} not in {wanted}")
        if touch_tokens:
            patterns = [t[len("touches:"):] for t in touch_tokens]
            key_files = task.get("key_files") or []
            parts.append(f"touches={patterns} not in key_files={key_files}")
        return False, "; ".join(parts)

    # No recognised matcher tokens at all.
    return False, (
        f"when={list(when)} not implemented (no recognised matcher)"
    )


# ---------------------------------------------------------------------------
# Per-gate worker (catches exceptions; produces a GateResult either way)
# ---------------------------------------------------------------------------

def _run_one(
    name: str, gate: Gate, task: dict, cli: dict,
    diff_path: str | None = None,
) -> GateResult:
    """Execute one gate, returning a GateResult on success OR failure.

    `diff_path` (T7, v0.3) is the path to the precomputed
    implementation diff. Threaded only into `AgentGate`; `ShellGate`
    has no template substitution surface and ignores it.

    Exceptions are caught and reported; the gate-run loop never raises
    out of `run_gates`.
    """
    start = time.time()
    try:
        if gate.kind == "shell":
            return ShellGate(name=name, gate=gate).run()
        if gate.kind == "agent":
            return AgentGate(
                name=name, gate=gate, task=task, cli=cli,
                diff_path=diff_path,
            ).run()
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

    # T7 §D7: derive the precomputed-diff path from task_dir and
    # thread it into every _run_one call (agent gates substitute it;
    # shell gates ignore it). Path is computed unconditionally even
    # when no agent gates are configured — cheap; the runner has
    # already written the file.
    diff_path_str = str(task_dir / "diff.patch")

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
        results.append(_run_one(name, gate, task, cli,
                                diff_path=diff_path_str))

    # Then parallel.
    if parallel:
        max_workers = min(len(parallel), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_run_one, n, g, task, cli, diff_path_str): n
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
